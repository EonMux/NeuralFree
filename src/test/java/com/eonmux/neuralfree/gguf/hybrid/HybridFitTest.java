package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The device-side byte accounting a hybrid fit charges: the plan bytes of the offloaded layers plus the plan's
 * position and addressing buffers,
 * so a split fits a context that the whole-model accounting would refuse.
 */
public class HybridFitTest {

    private static final int N_LAYER = 4;
    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int VOCAB = 32;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 32;
    private static final int N_CTX = 512;

    @Test
    public void contextBytesSumTheDeviceLayersAndTheStepInputs() {
        ModelSpec spec = spec();
        MemoryConfig cfg = memoryConfig();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        long bytes = HybridFit.contextBytes(plan, il -> il >= 2);
        long layers = 0;
        for (LayerMemoryPlan lp : plan.layers()) {
            if (lp.layer() >= 2) {
                layers = Math.addExact(layers, lp.totalBytes());
            }
        }
        assertEquals(layers + plan.positionBytes() + plan.addressingBytes(), bytes);
        assertTrue(bytes + " must be below the whole plan " + plan.totalBytes(), bytes < plan.totalBytes());
    }

    @Test
    public void theDeviceSideSizerFitsAContextTheWholeModelRefuses() {
        ModelSpec spec = spec();
        MemoryConfig cfg = memoryConfig();
        long budget = 96L * 1024 * 1024;
        int whole = ContextFitter.fit(spec, cfg, budget);
        int deviceHalf = ContextFitter.fit(spec, cfg, budget, ContextFitter.Companion.NONE,
                HybridFit.contextSizer(offload()));
        assertEquals(N_CTX, whole);
        assertEquals(N_CTX, deviceHalf);

        // a budget the whole-model accounting cannot serve at the training context
        long wholeAtTrain = ContextFitter.requiredBytes(spec, cfg, N_CTX);
        long deviceAtTrain = ContextFitter.requiredBytes(spec, cfg, N_CTX, ContextFitter.Companion.NONE,
                HybridFit.contextSizer(offload()));
        assertTrue(deviceAtTrain + " must be below " + wholeAtTrain, deviceAtTrain < wholeAtTrain);
    }

    @Test
    public void everyDeviceLayerChargesTheWholePlan() {
        MemoryPlan plan = MemoryPlanner.planForContext(spec(), memoryConfig(), N_CTX);
        assertEquals(plan.totalBytes(), HybridFit.contextBytes(plan, il -> true));
    }

    /** The offload of the last two trunk layers. */
    private static LayerOffload offload() {
        ModelSpec spec = spec();
        EngineConfig cfg = EngineConfig.builder().gpuLayers(3).build();
        return LayerOffload.auto(spec, cfg, tensors(), 2);
    }

    private static MemoryConfig memoryConfig() {
        return MemoryConfig.defaults().toBuilder().nCtx(N_CTX).ubatch(16).build();
    }

    private static ModelSpec spec() {
        int nLayer = N_LAYER;
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA))
                .layerCounts(nLayer, 0).nCtxTrain(4096).nEmbd(N_EMBD).nVocab(VOCAB)
                .normRmsEps(1e-6f).normEps(1e-5f).headKFull(HEAD_DIM).headVFull(HEAD_DIM);
        for (int il = 0; il < nLayer; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        TensorCatalog tc = TensorCatalog.of(Arch.LLAMA.ggufName(), tensors());
        NormResolver norms = new NormResolver(hb.build(), tc);
        HeadSpec.Builder head = HeadSpec.builder()
                .outputNorm(tc.get(TensorNames.weight(TensorNames.OUTPUT_NORM)), null)
                .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false);
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), tc)
                .schedule(LayerSchedule.linear(nLayer))
                .memoryKind(MemoryKind.KV)
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), N_EMBD).build())
                .head(head.build());
        for (int il = 0; il < nLayer; il++) {
            b.layer(LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                    .norm(norms.resolve(il))
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .attention(AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM)
                            .headDimV(HEAD_DIM).noRope(true).nRot(0)
                            .kqScale(1.0f / (float) Math.sqrt(HEAD_DIM)).kvOwnerLayer(il).build())
                    .ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).build())
                    .build());
        }
        return b.build();
    }

    private static List<GGUFTensorInfo> tensors() {
        List<GGUFTensorInfo> t = new ArrayList<>();
        t.add(info(TensorNames.weight(TensorNames.TOKEN_EMBD), N_EMBD, VOCAB));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT_NORM), N_EMBD));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT), N_EMBD, VOCAB));
        for (int il = 0; il < N_LAYER; il++) {
            t.add(info(blk(il, TensorNames.ATTN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_GATE), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_UP), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_DOWN), N_FF, N_EMBD));
        }
        return t;
    }

    private static GGUFTensorInfo info(String name, long... dims) {
        return new GGUFTensorInfo(name, dims, GGMLType.F32, 0);
    }

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }
}
