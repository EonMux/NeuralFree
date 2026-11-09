package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import com.eonmux.neuralfree.gguf.gpu.parallel.EplbRebalancer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.parallel.EplbSession;
import com.eonmux.neuralfree.gguf.gpu.parallel.TensorParallelGroup;
import com.eonmux.neuralfree.gguf.gpu.parallel.FfnRowSplitPlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.ShardBytes;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpAttentionPlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpMoePlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpVocabPlan;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.TensorPlacementOverrides;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * The split planner of a GPU model load: what the row, tensor and context split modes refuse once they engage over
 * two or more devices, the per-mode plans ({@code FfnRowSplitPlan}, {@code TpAttentionPlan}, {@code TpVocabPlan},
 * {@code TpMoePlan}) and the shard weights they upload, and the load log of the selected split. The refusals and
 * plans depend only on the configuration, the model spec and the uploader.
 */
final class GpuSplitPlanner {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModelGPU.class);

    /**
     * Opens the tensor-parallel group and plans the splits of the selected mode, refusing the shapes the mode cannot
     * serve and preparing the uploader's shard weights. Returns null (with its log line) when the configuration
     * selected fewer than two GPU devices, so the model loads on one device.
     *
     * @param eplb the expert-load balancer session, or null when {@code neuralfree.eplb} is off
     */
    static Selected select(EngineConfig cfg, ModelSpec initial, ComputeBackend backend, GpuWeightUploader uploader,
                           String modelName, EplbSession eplb, List<LoraAttachment.Binding> loraBindings) {
        TensorParallelGroup parallelGroup = TensorParallelGroup.select(cfg, backend);
        if (parallelGroup == null) {
            if (cfg.prefillContextParallel()) {
                throw new IllegalArgumentException(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "="
                        + cfg.prefillContextParallelSize() + " splits every prefill's query rows over a context"
                        + " split's ranks, but " + EngineConfig.PROP_SPLIT_MODE + "="
                        + cfg.splitMode().llamaName() + " selected fewer than two GPU devices; list the devices"
                        + " with " + EngineConfig.PROP_GPU_DEVICES + " or keep "
                        + EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "=1");
            }
            LOGGER.info("{}: {}= {} selected fewer than two GPU devices; the model loads on one device",
                    initial.arch().ggufName(), EngineConfig.PROP_SPLIT_MODE, cfg.splitMode().llamaName());
            return null;
        }
        int ranks = parallelGroup.peerCount() + 1;
        refuseRowUnsupported(cfg, initial);
        FfnRowSplitPlan rowSplitPlan = null;
        TpAttentionPlan attentionPlan = null;
        TpVocabPlan vocabPlan = null;
        TpMoePlan moePlan = null;
        EplbRebalancer eplbRebalancer = null;
        if (cfg.contextMode()) {
            refuseContextUnsupported(cfg, initial);
            if (cfg.prefillContextParallel() && cfg.prefillContextParallelSize() != ranks) {
                throw new IllegalArgumentException(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "="
                        + cfg.prefillContextParallelSize() + " splits every prefill's query rows over the"
                        + " context split's own ranks, but this split holds " + ranks + "; set "
                        + EngineConfig.PROP_SPLIT_MODE + "=context with that many GPUs or keep "
                        + EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "=1");
            }
            parallelGroup.prefillContextParallelSize(cfg.prefillContextParallelSize());
            if (cfg.prefillContextParallel()) {
                LOGGER.info("{}: {}= {} splits every prefill's query rows over the {} rank(s) at their"
                                + " context blocks", modelName,
                        EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL, cfg.prefillContextParallelSize(),
                        ranks);
            }
            attentionPlan = TpAttentionPlan.context(initial, ranks, uploader);
        } else if (cfg.tensorMode()) {
            refuseTensorUnsupported(cfg, initial);
            rowSplitPlan = denseFfnPlan(initial, ranks, uploader);
            attentionPlan = TpAttentionPlan.of(initial, ranks, uploader);
            vocabPlan = TpVocabPlan.of(initial, ranks, uploader);
            moePlan = TpMoePlan.of(initial, ranks, uploader, cfg.expertParallel(),
                    eplb == null ? null : eplb.placement(ranks), cfg.eplb().rebalance());
            eplbRebalancer = EplbRebalancer.open(cfg.eplb(), initial, ranks, moePlan,
                    parallelGroup::rearrangeExperts);
        } else {
            rowSplitPlan = FfnRowSplitPlan.of(initial, ranks, uploader, cfg.tensorSplit());
        }
        refuseRowOverrides(cfg, rowSplitPlan, attentionPlan, vocabPlan, moePlan);
        for (LoraAttachment.Binding binding : loraBindings) {
            refuseRowLora(rowSplitPlan, attentionPlan, vocabPlan, moePlan, binding);
        }
        uploader.splitShards(ShardBytes.splitShards(rowSplitPlan, attentionPlan, vocabPlan, moePlan, 0,
                uploader));
        refuseOffloadWithMultipleDevices(initial, cfg, ranks);
        LOGGER.info("{}: {}= {} splits {} attention and {} dense feed-forward of {} trunk layer(s) over {}"
                        + " device(s){}", modelName, EngineConfig.PROP_SPLIT_MODE,
                cfg.splitMode().llamaName(),
                attentionPlan == null ? 0 : attentionPlan.layers().size(),
                rowSplitPlan == null ? 0 : rowSplitPlan.layers().size(), initial.layers().size(), ranks,
                moePlan == null || moePlan.layers().isEmpty() ? ""
                        : !cfg.expertParallel() ? " with each expert's width split over the ranks"
                        : balancedLayers(moePlan) > 0
                        ? " with " + balancedLayers(moePlan) + " load-balanced and "
                        + (moePlan.layers().size() - balancedLayers(moePlan))
                        + " contiguous whole-expert layer(s) per rank"
                        : " with whole experts per rank");
        if (rowSplitPlan != null && !rowSplitPlan.ineligible().isEmpty()) {
            LOGGER.info("{}: feed-forward kept whole: {}", modelName,
                    String.join(", ", rowSplitPlan.ineligible()));
        }
        if (attentionPlan != null && !attentionPlan.skipped().isEmpty()) {
            LOGGER.info("{}: attention kept whole on the main device: {}", modelName,
                    String.join(", ", attentionPlan.skipped()));
        }
        if (eplb != null && eplb.balances() && (moePlan == null || balancedLayers(moePlan) == 0)) {
            LOGGER.warn("{}: the expert-load profile places no split layer; the contiguous expert blocks"
                    + " are kept", modelName);
        }
        return new Selected(parallelGroup, rowSplitPlan, attentionPlan, vocabPlan, moePlan, eplbRebalancer);
    }

    /** The plans a selected split: the group and the per-mode plans, each null when the mode has none. */
    static final class Selected {
        final TensorParallelGroup group;
        final FfnRowSplitPlan rowSplit;
        final TpAttentionPlan attention;
        final TpVocabPlan vocab;
        final TpMoePlan moe;
        final EplbRebalancer rebalancer;

        Selected(TensorParallelGroup group, FfnRowSplitPlan rowSplit, TpAttentionPlan attention, TpVocabPlan vocab,
                 TpMoePlan moe, EplbRebalancer rebalancer) {
            this.group = group;
            this.rowSplit = rowSplit;
            this.attention = attention;
            this.vocab = vocab;
            this.moe = moe;
            this.rebalancer = rebalancer;
        }
    }

    /**
     * The shapes a row or tensor split refuses once it engages over two or more devices: a switched-LoRA graph (one
     * device's routing), an encoder-decoder model (its cross-attention) and {@code --n-cpu-ffn} (host FFN layers
     * diverge from the split ranks). A load that selects only the model's device runs like {@code split-mode none}
     * and takes none of these.
     */
    static void refuseRowUnsupported(EngineConfig cfg, ModelSpec initial) {
        String arch = initial.arch().ggufName();
        String row = EngineConfig.PROP_SPLIT_MODE + "=" + cfg.splitMode().llamaName();
        if (GpuFactoryCatalog.usesSwitchedLora(initial)) {
            throw new UnsupportedOperationException(arch + " routes switched-LoRA adapters through one device's"
                    + " graph; " + row + " does not split it");
        }
        if (initial.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(arch + " is an encoder-decoder model; " + row
                    + " cannot place its cross-attention");
        }
        if (cfg.nCpuFfn() > EngineConfig.N_CPU_LAYERS_UNSET) {
            throw new UnsupportedOperationException(EngineConfig.PROP_N_CPU_FFN + " keeps the feed-forward of the"
                    + " first layers on the host, which " + row + " cannot slice; clear it or use "
                    + EngineConfig.PROP_SPLIT_MODE + "=layer");
        }
    }

    /**
     * Refuses a {@code -ot} rule that keeps a split tensor in host RAM: the split uploads a shard of it to every
     * rank instead, so the rule would silently not apply.
     */
    static void refuseRowOverrides(EngineConfig cfg, FfnRowSplitPlan ffn, TpAttentionPlan attention,
                                          TpVocabPlan vocab, TpMoePlan moe) {
        TensorPlacementOverrides overrides = TensorPlacementOverrides.none();
        for (TensorOverride o : cfg.tensorOverrides()) {
            overrides = overrides.with(TensorPlacementOverrides.builder()
                    .add(o.pattern(), o.host() ? TensorPlacementOverrides.Target.HOST
                            : TensorPlacementOverrides.Target.DEVICE)
                    .build());
        }
        for (TensorPlacementOverrides.Rule rule : overrides.rules()) {
            if (rule.target() != TensorPlacementOverrides.Target.HOST) {
                continue;
            }
            for (String tensor : splitTensorNames(ffn, attention, vocab, moe)) {
                if (rule.matches(tensor)) {
                    throw new UnsupportedOperationException(EngineConfig.PROP_TENSOR_OVERRIDES + " '" + rule
                            .pattern() + "' keeps " + tensor + " in host RAM, but " + EngineConfig.PROP_SPLIT_MODE
                            + " shards it over the ranks; drop the rule for the split tensors or use "
                            + EngineConfig.PROP_SPLIT_MODE + "=layer");
                }
            }
        }
    }

    /** Refuses a LoRA adapter that targets a projection the split shards: its whole weight never exists. */
    static void refuseRowLora(FfnRowSplitPlan ffn, TpAttentionPlan attention, TpVocabPlan vocab,
                                      TpMoePlan moe, LoraAttachment.Binding binding) {
        for (String target : binding.adapter().targets()) {
            if (splitTensorNames(ffn, attention, vocab, moe).contains(target)) {
                throw new UnsupportedOperationException("lora " + binding.adapter().path() + " targets "
                        + target + ", a projection " + EngineConfig.PROP_SPLIT_MODE
                        + " shards over the ranks; drop the adapter or load with "
                        + EngineConfig.PROP_SPLIT_MODE + "=layer");
            }
        }
    }

    /** Every tensor name the plans split, in model order. */
    static List<String> splitTensorNames(FfnRowSplitPlan ffn, TpAttentionPlan attention, TpVocabPlan vocab,
                                                TpMoePlan moe) {
        List<String> out = new ArrayList<>();
        if (ffn != null) {
            for (FfnRowSplitPlan.Layer layer : ffn.layers()) {
                for (String suffix : new String[]{layer.up(), layer.gate(), layer.down(), layer.upBias(),
                        layer.gateBias(), layer.downBias(), layer.downScale(), layer.actScales()}) {
                    if (suffix != null) {
                        out.add(layer.tensor(suffix));
                    }
                }
            }
        }
        if (attention != null) {
            for (TpAttentionPlan.Layer layer : attention.layers()) {
                for (String suffix : layer.tensors().keySet()) {
                    out.add(layer.tensor(suffix));
                }
            }
        }
        if (vocab != null) {
            out.add(vocab.tokenTensor());
            if (!vocab.tied()) {
                out.add(vocab.headTensor());
            }
            if (vocab.headBias() != null) {
                out.add(vocab.headBias());
            }
        }
        if (moe != null) {
            for (TpMoePlan.Layer layer : moe.layers()) {
                for (String suffix : layer.tensors().keySet()) {
                    out.add(layer.tensor(suffix));
                }
            }
        }
        return out;
    }

    /**
     * Plans the dense feed-forward half of a tensor split, or returns null when no dense trunk FFN exists (a routed
     * MoE trunk keeps its experts whole on the main device while the attention splits).
     */
    static FfnRowSplitPlan denseFfnPlan(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        try {
            return FfnRowSplitPlan.of(spec, ranks, uploader);
        } catch (UnsupportedOperationException noDenseFfn) {
            return null;
        }
    }

    /**
     * Refuses the tiers that need one device beside a multi-device split: the host-mapped offload, whose tensors
     * belong to exactly one device, and the selective streaming set, which a split plans per stage. Naming the
     * property keeps the configuration failure explicit instead of silently dropping the setting.
     *
     * @param ranks the ranks of a row or tensor split, or null for a layer split
     */
    static void refuseOffloadWithMultipleDevices(ModelSpec spec, EngineConfig cfg, Integer ranks) {
        String where = EngineConfig.PROP_SPLIT_MODE + "=" + cfg.splitMode().llamaName()
                + (ranks == null ? "" : " across " + ranks + " devices");
        if (cfg.cpuOffloadMb() > 0) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + ": " + EngineConfig.PROP_CPU_OFFLOAD_MB
                    + "=" + cfg.cpuOffloadMb() + " keeps trunk weights in host-mapped memory, which " + where
                    + " cannot place; use " + EngineConfig.PROP_SPLIT_MODE + "=none, or clear "
                    + EngineConfig.PROP_CPU_OFFLOAD_MB);
        }
        if (!cfg.streamParams().isEmpty()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + ": " + EngineConfig.PROP_STREAM_PARAMS
                    + "=" + cfg.streamParams() + " selects the streamed tensors of one device's trunk, which " + where
                    + " does not stream; use " + EngineConfig.PROP_SPLIT_MODE + "=none, or clear "
                    + EngineConfig.PROP_STREAM_PARAMS);
        }
    }

    /**
     * What the tensor mode refuses once it engages over two or more devices, beyond the row-split refusals: a model
     * whose loaded NextN / MTP blocks would need their own per-rank caches, and llama.cpp's {@code --tensor-split}
     * (vLLM divides every head and neuron evenly).
     */
    static void refuseTensorUnsupported(EngineConfig cfg, ModelSpec initial) {
        String tensor = EngineConfig.PROP_SPLIT_MODE + "=" + EngineConfig.SplitMode.TENSOR.llamaName();
        if (initial.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " is an encoder-decoder model; "
                    + tensor + " cannot place its cross-attention");
        }
        if (!initial.nextnLayers().isEmpty()) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " loads " + initial.nextnLayers().size()
                    + " NextN / MTP block(s) that need their own per-rank caches; " + tensor + " does not split them;"
                    + " disable MTP (" + EngineConfig.PROP_MTP + "=false) or use "
                    + EngineConfig.PROP_SPLIT_MODE + "=row");
        }
        if (cfg.memory().paged()) {
            // the per-rank caches mirror the main memory's page tables and copies; the group verifies at bind that
            // every rank's page geometry matches the memory that prepares the steps
            LOGGER.info("{}: {} serves paged batches by mirroring each step's page tables and page copies onto every"
                    + " rank cache", initial.arch().ggufName(), tensor);
        }
    }

    /**
     * Refuses the features a context split cannot run: the NextN / MTP draft blocks need caches of their own that
     * the split does not cover. The other multi-device refusals ({@link #refuseRowUnsupported},
     * {@link #refuseOffloadWithMultipleDevices}) apply unchanged. Over a paged memory the split partitions the page
     * pools at page granularity ({@code CpPageShard}); what it cannot carry refuses by name when it is asked for:
     * media / embedding segments whose rows share a base position, KV-chunk reuse's shifted-row page copies, the
     * per-sequence shifts and the tensor-parallel sessions.
     */
    static void refuseContextUnsupported(EngineConfig cfg, ModelSpec initial) {
        String context = EngineConfig.PROP_SPLIT_MODE + "=" + EngineConfig.SplitMode.CONTEXT.llamaName();
        if (initial.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " is an encoder-decoder model; "
                    + context + " cannot place its cross-attention");
        }
        if (!initial.nextnLayers().isEmpty()) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " loads " + initial.nextnLayers().size()
                    + " NextN / MTP block(s) that need their own per-rank caches; " + context + " does not split"
                    + " them; disable MTP (" + EngineConfig.PROP_MTP + "=false) or use "
                    + EngineConfig.PROP_SPLIT_MODE + "=row");
        }
        if (cfg.memory().paged()) {
            LOGGER.info("{}: {} over a paged memory assigns every page to the rank owning its first stored"
                            + " position (page-granular interleave); the ranks carry the paged batches through"
                            + " their own page tables",
                    initial.arch().ggufName(), context);
        }
    }

    static String splitDeviceCount(EngineConfig cfg) {
        return cfg.gpuDevices().isEmpty() ? "auto-selected" : String.valueOf(cfg.gpuDevices().size());
    }

    /** The MoE layers of {@code plan} whose whole experts the load balancer packed. */
    static int balancedLayers(TpMoePlan plan) {
        int n = 0;
        for (TpMoePlan.Layer layer : plan.layers()) {
            if (layer.balanced()) {
                n++;
            }
        }
        return n;
    }
}
