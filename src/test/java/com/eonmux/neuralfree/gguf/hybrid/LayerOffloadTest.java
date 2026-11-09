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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Resolution of {@link LayerOffload}: the llama.cpp {@code -ngl} count over a model's trunk layers, the host tensor
 * overrides and the MoE expert settings, with the refusals of the models and settings a split cannot serve.
 */
public class LayerOffloadTest {

    private static final int N_LAYER = 4;
    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int VOCAB = 32;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 32;

    /** The stored tensor names of the fixture, as a loader would see them. */
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

    /** Dense llama with a causal KV memory of {@value N_LAYER} layers. */
    private static ModelSpec spec() {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA))
                .layerCounts(N_LAYER, 0).nCtxTrain(4096).nEmbd(N_EMBD).nVocab(VOCAB)
                .normRmsEps(1e-6f).normEps(1e-5f).headKFull(HEAD_DIM).headVFull(HEAD_DIM);
        for (int il = 0; il < N_LAYER; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        List<GGUFTensorInfo> list = tensors();
        TensorCatalog tc = TensorCatalog.of(Arch.LLAMA.ggufName(), list);
        NormResolver norms = new NormResolver(hb.build(), tc);
        HeadSpec.Builder head = HeadSpec.builder()
                .outputNorm(tc.get(TensorNames.weight(TensorNames.OUTPUT_NORM)), null)
                .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false);
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), tc)
                .schedule(LayerSchedule.linear(N_LAYER))
                .memoryKind(MemoryKind.KV)
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), N_EMBD).build())
                .head(head.build());
        for (int il = 0; il < N_LAYER; il++) {
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

    private static EngineConfig config(int gpuLayers, String overrides) {
        EngineConfig.Builder b = EngineConfig.builder().gpuLayers(gpuLayers);
        if (overrides != null) {
            for (String entry : overrides.split(",")) {
                int eq = entry.indexOf('=');
                b.addTensorOverride(entry.substring(0, eq), entry.substring(eq + 1));
            }
        }
        return b.build();
    }

    @Test
    public void explicitCountOffloadsTheLastLayersPlusTheOutputHead() {
        LayerOffload offload = LayerOffload.requested(spec(), config(3, null), tensors());
        assertTrue(offload.active());
        assertEquals(2, offload.nGpuLayers());
        assertEquals(2, offload.firstGpuLayer());
        assertTrue(offload.hostLayer(0));
        assertTrue(offload.hostLayer(1));
        assertFalse(offload.hostLayer(2));
        assertFalse(offload.hostLayer(3));
        assertTrue(offload.hostTensor(blk(1, TensorNames.FFN_GATE)));
        assertTrue(offload.hostTensor("model.layers.0.ffn_up.weight"));
        assertFalse(offload.hostTensor(blk(2, TensorNames.FFN_GATE)));
        assertFalse(offload.hostTensor(TensorNames.weight(TensorNames.TOKEN_EMBD)));
        assertFalse(offload.hostTensor(TensorNames.weight(TensorNames.OUTPUT)));
    }

    @Test
    public void fullOffloadAndAutomaticWithoutHostTensorsResolveToNoSplit() {
        assertNull(LayerOffload.requested(spec(), config(N_LAYER + 1, null), tensors()));
        assertNull(LayerOffload.requested(spec(), config(EngineConfig.GPU_LAYERS_AUTO, null), tensors()));
        assertNull(LayerOffload.requested(spec(), config(EngineConfig.GPU_LAYERS_ALL, null), tensors()));
    }

    @Test
    public void refusalsNameTheSetting() {
        IllegalArgumentException zero = assertThrows(IllegalArgumentException.class,
                () -> LayerOffload.requested(spec(), config(0, null), tensors()));
        assertTrue(zero.getMessage().contains(EngineConfig.PROP_N_GPU_LAYERS));
    }

    /**
     * llama.cpp clamps {@code -ngl} instead of refusing it (src/llama-model.cpp:1692-1693:
     * {@code act_gpu_layers = min(n_gpu_layers, n_layer + 1)}), so the {@code -ngl 99} every guide passes keeps the
     * whole model on the device, exactly as {@code n_layer + 1} and {@code -ngl all} do, without the
     * {@code -ngl auto} VRAM search.
     */
    @Test
    public void aCountPastTheLayersOffloadsTheWholeModelAsAnExactFullOffload() {
        assertNull(LayerOffload.requested(spec(), config(99, null), tensors()));
        assertNull(LayerOffload.requested(spec(), config(N_LAYER + 2, null), tensors()));

        LayerOffload over = LayerOffload.requested(spec(),
                config(99, null).toBuilder().kvOffload(false).build(), tensors());
        LayerOffload exact = LayerOffload.requested(spec(),
                config(N_LAYER + 1, null).toBuilder().kvOffload(false).build(), tensors());
        assertEquals("the clamped count keeps every trunk layer on the device", 0, over.firstGpuLayer());
        assertEquals(N_LAYER, over.nGpuLayers());
        assertTrue(over.hostKv());
        assertEquals("the clamped count resolves exactly as nLayer + 1", exact.firstGpuLayer(), over.firstGpuLayer());
        assertEquals(exact.nGpuLayers(), over.nGpuLayers());

        LayerOffload overSplit = LayerOffload.requested(spec(), config(99, "blk\\.2\\.ffn.*=cpu"), tensors());
        LayerOffload exactSplit = LayerOffload.requested(spec(),
                config(EngineConfig.GPU_LAYERS_ALL, "blk\\.2\\.ffn.*=cpu"), tensors());
        assertEquals(3, overSplit.firstGpuLayer());
        assertEquals("the clamped count behaves as -ngl all beside a host override",
                exactSplit.firstGpuLayer(), overSplit.firstGpuLayer());
        assertEquals(exactSplit.nGpuLayers(), overSplit.nGpuLayers());
    }

    @Test
    public void hostOverrideGrowsTheHostSideToCoverItsLowestLayer() {
        LayerOffload offload = LayerOffload.requested(spec(),
                config(EngineConfig.GPU_LAYERS_ALL, "blk\\.2\\.ffn.*=cpu"), tensors());
        assertTrue(offload.active());
        assertEquals(1, offload.nGpuLayers());
        assertEquals(3, offload.firstGpuLayer());
        assertTrue(offload.hostLayer(2));
        assertTrue(offload.hostLayer(0));
        assertFalse(offload.hostLayer(3));
        assertTrue(offload.hostTensor(blk(2, TensorNames.FFN_GATE)));
        assertTrue(offload.hostTensor(blk(2, TensorNames.ATTN_NORM)));
    }

    @Test
    public void hostOverridesOnSeveralLayersPullTheHighestToTheHost() {
        LayerOffload offload = LayerOffload.requested(spec(),
                config(EngineConfig.GPU_LAYERS_ALL, "blk\\.0\\.ffn_up=cpu,blk\\.2\\.attn_norm=cpu"), tensors());
        assertEquals("a host-named layer never computes on the device", 3, offload.firstGpuLayer());
        assertTrue(offload.hostLayer(2));
        assertFalse(offload.hostLayer(3));
    }

    @Test
    public void nCpuFfnRunsTheLayersOfItsDenseFfnWeightsOnTheHost() {
        EngineConfig cfg = config(EngineConfig.GPU_LAYERS_AUTO, null).toBuilder().nCpuFfn(2).build();
        LayerOffload offload = LayerOffload.requested(spec(), cfg, tensors());
        assertEquals(2, offload.firstGpuLayer());
        assertTrue(offload.hostTensor(blk(1, TensorNames.FFN_DOWN)));
        assertTrue(offload.hostTensor(blk(1, TensorNames.ATTN_NORM)));
        assertFalse(offload.hostTensor(blk(2, TensorNames.FFN_UP)));

        EngineConfig beyond = config(EngineConfig.GPU_LAYERS_AUTO, null).toBuilder().nCpuFfn(N_LAYER + 5).build();
        assertEquals("a count beyond the trunk keeps no layer on the device", N_LAYER,
                LayerOffload.requested(spec(), beyond, tensors()).firstGpuLayer());
    }

    @Test
    public void hostOverrideInsideTheHostSideKeepsTheSplit() {
        LayerOffload offload = LayerOffload.requested(spec(), config(3, "blk\\.[01]\\.attn_norm=cpu"), tensors());
        assertEquals(2, offload.nGpuLayers());
        assertEquals(2, offload.firstGpuLayer());
        assertTrue(offload.hostTensor(blk(0, TensorNames.ATTN_NORM)));
        assertFalse(offload.hostTensor(blk(2, TensorNames.ATTN_NORM)));
    }

    @Test
    public void deviceOverrideKeepsTensorsOnTheDevice() {
        LayerOffload offload = LayerOffload.requested(spec(),
                config(EngineConfig.GPU_LAYERS_AUTO, "blk\\.0\\.attn_norm=cpu,ffn_down=opencl"), tensors());
        assertTrue(offload.active());
        assertEquals(1, offload.firstGpuLayer());
        assertTrue(offload.hostLayer(0));
        assertFalse(offload.hostLayer(1));
        assertFalse(offload.hostTensor(blk(3, TensorNames.FFN_DOWN)));
    }

    @Test
    public void hostOverrideOnAModelLevelTensorIsRefusedByName() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), config(EngineConfig.GPU_LAYERS_AUTO, "output.*=cpu"), tensors()));
        assertTrue(e.getMessage().contains("output"));
        e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), config(EngineConfig.GPU_LAYERS_AUTO, "token_embd=cpu"), tensors()));
        assertTrue(e.getMessage().contains("token_embd"));
    }

    @Test
    public void cpuMoeKeepsExpertTensorsOnTheHostWhileTheLayerStaysOnTheDevice() {
        LayerOffload offload = LayerOffload.auto(spec(), config(EngineConfig.GPU_LAYERS_ALL, null).toBuilder()
                .cpuMoe(true).build(), tensors(), N_LAYER);
        assertTrue(offload.hostExpertOverrides().size() == N_LAYER);
        assertFalse(offload.active());
        assertTrue(offload.hostTensor(blk(2, TensorNames.FFN_GATE).replace("ffn_gate", "ffn_up_exps")));
        assertFalse(offload.hostTensor(blk(2, TensorNames.FFN_GATE)));
        // the fixture's layers are dense: a dense layer has no experts to keep on the host
        assertFalse(offload.hostExperts(2));
    }

    @Test
    public void nCpuMoeCoversOnlyTheFirstLayers() {
        EngineConfig cfg = config(EngineConfig.GPU_LAYERS_AUTO, null).toBuilder().nCpuMoe(2).build();
        LayerOffload offload = LayerOffload.auto(spec(), cfg, tensors(), N_LAYER);
        assertEquals(2, offload.hostExpertOverrides().size());
        assertTrue(offload.hostTensor(blk(0, TensorNames.FFN_GATE).replace("ffn_gate", "ffn_gate_up_exps")));
        assertFalse(offload.hostTensor(blk(2, TensorNames.FFN_GATE).replace("ffn_gate", "ffn_down_exps")));
        assertFalse(offload.hostExperts(0));
        assertFalse(offload.hostExperts(3));
    }

    @Test
    public void hostOverrideOnExpertTensorsKeepsTheLayerOnTheDevice() {
        List<GGUFTensorInfo> withExperts = new ArrayList<>(tensors());
        for (int il = 0; il < N_LAYER; il++) {
            withExperts.add(info(blk(il, TensorNames.FFN_UP_EXPS), N_FF, 4, 8));
            withExperts.add(info(blk(il, TensorNames.FFN_DOWN_EXPS), N_EMBD, N_FF, 8));
        }
        assertNull("a host override on routed expert tensors hosts them per tensor; no layer leaves the device",
                LayerOffload.requested(spec(), config(EngineConfig.GPU_LAYERS_ALL, "ffn_(up|down)_exps=cpu"),
                        withExperts));
        assertNull(LayerOffload.auto(spec(), config(EngineConfig.GPU_LAYERS_AUTO, "ffn_up_exps=cpu"),
                withExperts, N_LAYER));
    }

    @Test
    public void autoCandidateCountsAreRangeChecked() {
        assertThrows(IllegalArgumentException.class,
                () -> LayerOffload.auto(spec(), config(EngineConfig.GPU_LAYERS_AUTO, null), tensors(), -1));
        assertThrows(IllegalArgumentException.class,
                () -> LayerOffload.auto(spec(), config(EngineConfig.GPU_LAYERS_AUTO, null), tensors(), N_LAYER + 1));
    }

    @Test
    public void loraAdaptersRunOverASplit() {
        EngineConfig cfg = config(3, null).toBuilder().addLora("adapter.gguf", 1.0f).build();
        assertNotNull("the host layers take the adapters' host deltas",
                LayerOffload.requested(spec(), cfg, tensors()));
    }

    @Test
    public void noKvOffloadKeepsTheKvOfEveryDeviceLayerInHostRam() {
        LayerOffload offload = LayerOffload.requested(spec(),
                EngineConfig.builder().kvOffload(false).build(), tensors());
        assertNotNull(offload);
        assertTrue(offload.hostKv());
        assertEquals("every trunk layer stays on the device", 0, offload.firstGpuLayer());
        assertEquals(N_LAYER, offload.nGpuLayers());
        assertFalse(offload.active());
        for (int il = 0; il < N_LAYER; il++) {
            assertFalse(offload.hostLayer(il));
            assertTrue("layer " + il + " keeps its KV in host RAM", offload.hostKvLayer(il));
            assertTrue(offload.hostMemoryLayer(il));
            assertFalse("the device allocates no cache of layer " + il, offload.deviceLayer(il));
        }
        assertFalse("the weights stay on the device", offload.hostTensor(blk(0, TensorNames.ATTN_NORM)));
    }

    @Test
    public void noKvOffloadCombinesWithTheHostLayerPrefix() {
        LayerOffload offload = LayerOffload.requested(spec(),
                config(3, null).toBuilder().kvOffload(false).build(), tensors());
        assertEquals(2, offload.firstGpuLayer());
        assertTrue(offload.hostKv());
        assertFalse("a host layer owns its whole layer, cache included", offload.hostKvLayer(0));
        assertFalse(offload.hostKvLayer(1));
        assertTrue(offload.hostKvLayer(2));
        assertTrue(offload.hostKvLayer(3));
        assertTrue(offload.hostMemoryLayer(0));
        assertTrue(offload.hostMemoryLayer(2));
        assertFalse(offload.deviceLayer(0));
        assertFalse(offload.deviceLayer(2));
        assertTrue(offload.hostLayer(0));
        assertFalse(offload.hostLayer(2));
    }

    @Test
    public void noKvOffloadRefusesAPagedMemory() {
        EngineConfig cfg = EngineConfig.builder().kvOffload(false)
                .memory(com.eonmux.neuralfree.gguf.memory.MemoryConfig.defaults().toBuilder().kvPaged(true).build())
                .build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), cfg, tensors()));
        assertTrue(e.getMessage().contains("paged"));
    }

    @Test
    public void noKvOffloadRefusesTheMultiGpuLayerSplit() {
        EngineConfig cfg = EngineConfig.builder().kvOffload(false)
                .splitMode(EngineConfig.SplitMode.LAYER).build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), cfg, tensors()));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
    }

    @Test
    public void noKvOffloadRefusesNextnMtpDrafting() {
        EngineConfig cfg = EngineConfig.builder().kvOffload(false).mtp(true).build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), cfg, tensors()));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_MTP));
    }

    @Test
    public void pagedMemoryIsRefusedOverASplit() {
        EngineConfig cfg = config(3, null).toBuilder()
                .memory(com.eonmux.neuralfree.gguf.memory.MemoryConfig.defaults().toBuilder()
                        .kvPaged(true).build())
                .build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerOffload.requested(spec(), cfg, tensors()));
        assertTrue(e.getMessage().contains("paged"));
    }
}
