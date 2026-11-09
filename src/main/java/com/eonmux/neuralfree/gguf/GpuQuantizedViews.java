package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.QuantizedView;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.ops.encdec.GpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.io.IOException;
import java.util.Collections;

/**
 * The quantized target views of two-level verification over a {@link GGUFModelGPU}: the nested 8-bit Q view of a
 * W12 target (this model's own rows through an open {@code W12QView} scope) and, for the other targets, the copy
 * {@link QCopySource} restates beside this model's passes on the same device. The view shapes the model refuses
 * are the ones whose passes would diverge from the target's.
 */
final class GpuQuantizedViews {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModelGPU.class);

    private GpuQuantizedViews() {
    }

    /** Runs the view described on {@link GGUFModelGPU#openQuantizedView}. */
    static QuantizedView open(GGUFModelGPU model, int bits) throws IOException {
        String refusal = quantizedViewRefusal(model);
        if (refusal != null) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " copies the target's projection"
                    + " matrices onto the target's device, which this load cannot shadow: " + refusal);
        }
        if (model.uploader.w12TensorCount() > 0) {
            return openW12QuantizedView(model, bits);
        }
        QCopySource qSource = QCopySource.open(model.source, bits);
        QuantizedView.Build build = new QuantizedView.Build();
        try {
            TensorCatalog qCatalog = TensorCatalog.of(qSource);
            InputTransformLoader.Result qTransforms = InputTransformLoader.load(qSource, GgufKv.of(model.source),
                    qCatalog, model.config.activationQuant());
            WeightStore qStore = qSource.createMappedStore(StoreUse.RESIDENT_UPLOAD);
            build.owns(qStore);
            GpuWeightUploader qUploader = new GpuWeightUploader(model.backend, model.queue, qSource.getTensors(), qStore,
                    model.spec.arch().ggufName(),
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(model.kernels))
                            .transforms(qTransforms)
                            .w12(false)
                            .w12Head(false)
                            .build());
            build.owns(qUploader);
            qUploader.uploadResident(t -> false);
            GpuForwardContext qCtx = GpuForwardContext.builder(model.kernels, model.ops, qUploader, model.forwardSpec)
                    .memory(model.memory)
                    .heat(model.ranking.heat())
                    .build();
            SwitchedLoraExtra.Router router = GpuFactoryCatalog.usesSwitchedLora(model.forwardSpec)
                    ? SwitchedLoraExtra.Router.of(model.spec.hparams()) : null;
            GpuBlockRegistry qRegistry = GpuFactoryCatalog.defaults(router, null);
            ForwardBackend qBackend = model.spec.supportClass() == SupportClass.ENC_DEC
                    ? new GpuEncDecBackend(qCtx, qRegistry, GpuFactoryCatalog.headPreprocessors(),
                            model.tokenizer.specialTokens().bos())
                    : new GpuForwardBackend(qCtx, qRegistry, GpuFactoryCatalog.headPreprocessors());
            build.owns(qBackend);
            LOGGER.info("{}: quantized target view for two-level verification: {} restated matrices, {} bytes"
                            + " beside the target's weights on {}", model.spec.arch().ggufName(), qSource.getTensorCount(),
                    qSource.plannedBytes(), model.deviceName());
            return new QuantizedView(qBackend, build.owned(),
                    "quantized target view (" + bits + "-bit rows) of " + model.source.getFilePath(),
                    qSource.plannedBytes(), bits);
        } catch (RuntimeException e) {
            throw build.close(e);
        }
    }

    /**
     * The nested 8-bit Q view of a W12 target: the Q round runs this model's own backend inside a
     * {@link GpuOps#openW12QView} scope, which routes every projection of a W12 weight through the nested 8-bit
     * view of its code bytes — no copy is built, no second weight upload or backend exists, and the view adds
     * no memory. The scope opens per Q round and closes before the exact round reads the same weights. The
     * nested view is fixed 8-bit, so the copy's 6-bit width does not apply.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv.bits} for any width but 8
     */
    private static QuantizedView openW12QuantizedView(GGUFModelGPU model, int bits) {
        if (bits != 8) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV_BITS + "=" + bits + ": this target"
                    + " streams its projection matrices as W12 rows, whose nested Q view is fixed 8-bit; the"
                    + " 6-bit copy does not apply to a W12 target");
        }
        GpuOps viewOps = model.ops;
        LOGGER.info("{}: two-level verification reads the target's own W12 rows through the nested 8-bit Q view:"
                + " no copy and no second backend on {}", model.spec.arch().ggufName(), model.deviceName());
        return new QuantizedView(model.forwardBackend(), java.util.Collections.emptyList(),
                "nested 8-bit Q view of the W12 rows of " + model.source.getFilePath(), 0L, 8,
                () -> {
                    GpuOps.W12QView scope = viewOps.openW12QView();
                    return scope::close;
                });
    }

    /**
     * Why this load cannot run a quantized copy of itself beside its own passes, or null: every shape below makes
     * the copy's passes diverge from the target's (a split or streamed or host-mapped weight source it cannot
     * mirror, expert machinery whose state the copy would not share) or apply to one side only (adapters and
     * control vectors, expert-parallel balancing).
     */
    private static String quantizedViewRefusal(GGUFModelGPU model) {
        if (model.split != null) {
            return "the model runs as a pipeline split across devices (" + model.split + ")";
        }
        if (model.parallelGroup != null) {
            return "the model runs as a tensor split across devices";
        }
        if (model.hybridSplit != null) {
            return "a layer offload keeps host layers beside the device ones (neuralfree.offload.layers)";
        }
        if (model.streaming != null || !model.streamedSets.isEmpty()) {
            return "trunk layers stream from host storage instead of running resident on the device";
        }
        if (!model.hostMappedPlan.names().isEmpty()) {
            return "trunk weights sit in host-mapped memory (" + EngineConfig.PROP_CPU_OFFLOAD_MB + ")";
        }
        if (model.moeCache != null) {
            return "a MoE expert cache keeps experts in RAM and swaps them per step (neuralfree.moe.cache)";
        }
        if (!model.hostedLayers.isEmpty()) {
            return "MoE layers keep their experts on the host";
        }
        if (model.eplb != null) {
            return "the expert load balancer rearranges experts across devices (neuralfree.eplb)";
        }
        if (!model.config.loras().isEmpty()) {
            return "the target's passes run LoRA adapters, which the copy's would not";
        }
        if (model.adapters != null && model.adapters.controlVectors() != null) {
            return "the target's passes apply control vectors, which the copy's would not";
        }
        return null;
    }
}
