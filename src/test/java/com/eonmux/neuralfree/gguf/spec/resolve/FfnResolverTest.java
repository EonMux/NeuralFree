package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Dense FFN resolution: separate-gate SwiGLU, the fused {@code [gate | up]} shape rule and FFN-less layers. */
public class FfnResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_FF = 128;

    private static ModelHparams llama(int blocks) {
        return hparams(Arch.LLAMA, blocks);
    }

    private static ModelHparams hparams(Arch arch, int blocks) {
        String a = arch.ggufName();
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", a);
        m.put(a + ".context_length", 4096);
        m.put(a + ".embedding_length", N_EMBD);
        m.put(a + ".block_count", blocks);
        m.put(a + ".feed_forward_length", N_FF);
        m.put(a + ".attention.head_count", 4);
        m.put(a + ".attention.layer_norm_rms_epsilon", 1e-6f);
        return HparamsLoader.load(GgufKv.ofMap(m), arch);
    }

    private static List<GGUFTensorInfo> biasedSwiglu() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        ts.add(t("blk.0.ffn_gate.bias", N_FF));
        ts.add(t("blk.0.ffn_up.bias", N_FF));
        ts.add(t("blk.0.ffn_down.bias", N_EMBD));
        return ts;
    }

    @Test
    public void ffnBiasesApplyUnlessTheGraphIgnoresThem() {
        FfnSpec llama = new FfnResolver(llama(1), catalog(biasedSwiglu())).resolve(0).ffn();
        assertTrue(llama.biases());

        FfnSpec refact = new FfnResolver(hparams(Arch.REFACT, 1), TensorCatalog.of("refact", biasedSwiglu()))
                .resolve(0).ffn();
        assertFalse("refact.cpp:128-133 passes NULL for every FFN bias", refact.biases());
        assertTrue(UnusedBiases.ffnUnused(Arch.LLADA));
        assertFalse(UnusedBiases.ffnUnused(Arch.MINICPM));
        assertTrue(UnusedBiases.attnOutputUnused(Arch.ERNIE4_5_MOE));
        assertFalse(UnusedBiases.attnOutputUnused(Arch.PADDLEOCR));
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static TensorCatalog catalog(List<GGUFTensorInfo> tensors) {
        return TensorCatalog.of("llama", tensors);
    }

    @Test
    public void separateGateIsParallelSwiglu() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        FfnResolver.Result r = new FfnResolver(llama(1), catalog(ts)).resolve(0);
        assertEquals(FfnKind.DENSE, r.kind());
        assertNull(r.moe());
        FfnSpec ffn = r.ffn();
        assertEquals(FfnSpec.Role.DENSE, ffn.role());
        assertEquals(N_FF, ffn.nFf());
        assertEquals(FfnGating.PAR, ffn.gating());
        assertEquals(GluOp.SWIGLU, ffn.glu());
    }

    @Test
    public void doubleWidthUpWithoutGateIsFusedSwiglu() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, 2L * N_FF));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        FfnSpec ffn = new FfnResolver(llama(1), catalog(ts)).resolve(0).ffn();
        assertEquals(FfnGating.FUSED_UP, ffn.gating());
        assertEquals(GluOp.SWIGLU, ffn.glu());
        assertEquals(N_FF, ffn.nFf());
    }

    @Test
    public void layerWithoutFfnTensorsHasNoFfn() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        FfnResolver.Result r = new FfnResolver(llama(2), catalog(ts)).resolve(1);
        assertEquals(FfnKind.NONE, r.kind());
        assertNull(r.ffn());
        assertNull(r.moe());
    }

    @Test
    public void partialFfnNamesTheMissingTensor() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF));
        FfnResolver resolver = new FfnResolver(llama(1), catalog(ts));
        MissingTensorException e = assertThrows(MissingTensorException.class, () -> resolver.resolve(0));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.ffn_down.weight"));
    }

    @Test
    public void expertsWithoutRouterNameTheMissingRouter() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate_exps.weight", N_EMBD, 32, 4));
        ts.add(t("blk.0.ffn_up_exps.weight", N_EMBD, 32, 4));
        ts.add(t("blk.0.ffn_down_exps.weight", 32, N_EMBD, 4));
        FfnResolver resolver = new FfnResolver(llama(1), catalog(ts));
        MissingTensorException e = assertThrows(MissingTensorException.class, () -> resolver.resolve(0));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.ffn_gate_inp.weight"));
    }

    @Test
    public void mismatchedUpRowsAreRejected() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF + 2));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        FfnResolver resolver = new FfnResolver(llama(1), catalog(ts));
        assertThrows(IllegalStateException.class, () -> resolver.resolve(0));
    }
}
