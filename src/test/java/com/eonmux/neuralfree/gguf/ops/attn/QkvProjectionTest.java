package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.UnusedBiases;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Projection plans: the output bias the graph passes or suppresses (models/falcon-h1.cpp:151-153,
 * models/ernie4-5.cpp:118-120), the MoVA value projection of k2-horizon (models/k2-horizon.cpp:137-205, 248) and the
 * Q-only projection of a layer that reads an earlier layer's cache (models/gemma3n.cpp:173-182), and the Q/K/V
 * biases that graphs projecting without {@code build_qkv} never add (models/mimo2.cpp:121-148).
 */
public class QkvProjectionTest {

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 16;
    private static final int WIDTH = 64;

    private static AttentionSpec.Builder attention() {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .nRot(HEAD_DIM).kqScale(0.25f).kvOwnerLayer(0);
    }

    private static LayerSpec layer(AttentionSpec a) {
        return layer(0, a);
    }

    private static LayerSpec layer(int il, AttentionSpec a) {
        return LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(a)
                .build();
    }

    private static Set<String> names(String... suffixes) {
        return new HashSet<>(Arrays.asList(suffixes));
    }

    private static Set<String> separateWithBiases() {
        return names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
                TensorNames.weight(TensorNames.ATTN_V), TensorNames.weight(TensorNames.ATTN_OUT),
                TensorNames.bias(TensorNames.ATTN_OUT));
    }

    private static MoeSpec valueExperts(boolean probBias) {
        return MoeSpec.builder().experts(8, 2, N_HEAD_KV * HEAD_DIM).gating(GatingFunc.SIGMOID).probBias(probBias)
                .normWeights(true).weightsScale(1f).act(ActOp.SILU).build();
    }

    @Test
    public void outputBiasIsAppliedWhenTheGraphPassesIt() {
        QkvProjection p = QkvProjection.of(layer(attention().build()), separateWithBiases()::contains, Arch.LLAMA);
        assertEquals(TensorNames.bias(TensorNames.ATTN_OUT), p.outBias());
    }

    @Test
    public void outputBiasIsSuppressedWhenTheGraphPassesNull() {
        QkvProjection p = QkvProjection.of(layer(attention().outBias(false).build()), separateWithBiases()::contains,
                Arch.FALCON_H1);
        assertNull(p.outBias());
        assertEquals(TensorNames.weight(TensorNames.ATTN_OUT), p.outWeight());
    }

    @Test
    public void movaLayerProjectsValuesThroughRoutedExperts() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
                TensorNames.weight(TensorNames.ATTN_V_GATE), TensorNames.bias(TensorNames.ATTN_V_GATE),
                TensorNames.weight(TensorNames.ATTN_V_EXPS), TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(attention().valueExperts(valueExperts(true)).build()), has::contains,
                Arch.K2_HORIZON);
        assertEquals(QkvProjection.Source.SEPARATE, p.source());
        assertTrue(p.valueExperts());
        assertFalse(p.vFromK());
        assertNull(p.vWeight());
        assertNull(p.vBias());
        assertEquals(TensorNames.weight(TensorNames.ATTN_V_EXPS), p.vExpsWeight());
        assertEquals(TensorNames.weight(TensorNames.ATTN_V_GATE), p.vGateWeight());
        assertEquals(TensorNames.bias(TensorNames.ATTN_V_GATE), p.vGateBias());
        assertNotNull(p.kWeight());
    }

    @Test
    public void movaRouterWithoutBiasHasNoSelectionBias() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
                TensorNames.weight(TensorNames.ATTN_V_GATE), TensorNames.weight(TensorNames.ATTN_V_EXPS),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(attention().valueExperts(valueExperts(false)).build()), has::contains,
                Arch.K2_HORIZON);
        assertTrue(p.valueExperts());
        assertNull(p.vGateBias());
    }

    @Test
    public void plainLayerHasNoValueExperts() {
        QkvProjection p = QkvProjection.of(layer(attention().build()), separateWithBiases()::contains, Arch.LLAMA);
        assertFalse(p.valueExperts());
        assertNull(p.vExpsWeight());
        assertEquals(TensorNames.weight(TensorNames.ATTN_V), p.vWeight());
    }

    @Test
    public void sharedKvReaderTakesQFromTheFusedRows() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_QKV), TensorNames.bias(TensorNames.ATTN_QKV),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(1, attention().build()), has::contains, Arch.GEMMA3N);
        assertEquals(QkvProjection.Source.FUSED, p.source());
        assertFalse(p.projectsKv());
        assertEquals(TensorNames.weight(TensorNames.ATTN_QKV), p.qkvWeight());
        assertEquals(0, p.qOffset());
        assertEquals(N_HEAD * HEAD_DIM, p.qProjWidth());
        assertEquals((N_HEAD + 2 * N_HEAD_KV) * HEAD_DIM, p.fusedWidth());
        assertNull(p.qkvBias());
        assertNull(p.qBias());
        assertNull(p.kBias());
        assertNull(p.vBias());
    }

    @Test
    public void sharedKvReaderIgnoresFusedSplitBiases() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_QKV), TensorNames.bias(TensorNames.ATTN_Q),
                TensorNames.bias(TensorNames.ATTN_K), TensorNames.bias(TensorNames.ATTN_V),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(1, attention().build()), has::contains, Arch.GEMMA3N);
        assertEquals(QkvProjection.Source.FUSED, p.source());
        assertNull(p.qBias());
        assertNull(p.kBias());
        assertNull(p.vBias());
    }

    @Test
    public void sharedKvReaderProjectsQAloneWithoutItsBias() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.bias(TensorNames.ATTN_Q),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(1, attention().build()), has::contains, Arch.GEMMA3N);
        assertEquals(QkvProjection.Source.SEPARATE, p.source());
        assertFalse(p.projectsKv());
        assertEquals(TensorNames.weight(TensorNames.ATTN_Q), p.qWeight());
        assertNull(p.qBias());
        assertNull(p.kWeight());
        assertNull(p.vWeight());
    }

    @Test
    public void cacheOwnerKeepsTheFusedBias() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_QKV), TensorNames.bias(TensorNames.ATTN_QKV),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(attention().build()), has::contains, Arch.GEMMA3N);
        assertTrue(p.projectsKv());
        assertEquals(TensorNames.bias(TensorNames.ATTN_QKV), p.qkvBias());
    }

    @Test
    public void mimo2IgnoresItsProjectionBiases() {
        assertTrue(UnusedBiases.attnInputUnused(Arch.MIMO2));
        Set<String> separate = names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
                TensorNames.weight(TensorNames.ATTN_V), TensorNames.bias(TensorNames.ATTN_Q),
                TensorNames.bias(TensorNames.ATTN_K), TensorNames.bias(TensorNames.ATTN_V),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(attention().inBias(false).build()), separate::contains, Arch.MIMO2);
        assertTrue(p.projectsKv());
        assertNull(p.qBias());
        assertNull(p.kBias());
        assertNull(p.vBias());
        Set<String> fused = names(TensorNames.weight(TensorNames.ATTN_QKV), TensorNames.bias(TensorNames.ATTN_QKV),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection f = QkvProjection.of(layer(attention().inBias(false).build()), fused::contains, Arch.MIMO2);
        assertEquals(QkvProjection.Source.FUSED, f.source());
        assertNull(f.qkvBias());
    }

    @Test
    public void buildQkvGraphsAddTheirProjectionBiases() {
        assertFalse(UnusedBiases.attnInputUnused(Arch.LLAMA));
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
                TensorNames.weight(TensorNames.ATTN_V), TensorNames.bias(TensorNames.ATTN_Q),
                TensorNames.bias(TensorNames.ATTN_K), TensorNames.bias(TensorNames.ATTN_V),
                TensorNames.weight(TensorNames.ATTN_OUT));
        QkvProjection p = QkvProjection.of(layer(attention().build()), has::contains, Arch.LLAMA);
        assertEquals(TensorNames.bias(TensorNames.ATTN_Q), p.qBias());
        assertEquals(TensorNames.bias(TensorNames.ATTN_K), p.kBias());
        assertEquals(TensorNames.bias(TensorNames.ATTN_V), p.vBias());
    }

    @Test
    public void valueExpertsOnAFusedProjectionAreRefused() {
        Set<String> has = names(TensorNames.weight(TensorNames.ATTN_QKV), TensorNames.weight(TensorNames.ATTN_OUT));
        LayerSpec l = layer(attention().valueExperts(valueExperts(false)).build());
        assertThrows(UnsupportedOperationException.class, () -> QkvProjection.of(l, has::contains, Arch.K2_HORIZON));
    }
}
