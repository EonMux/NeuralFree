package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The tensors a pooled hosted MoE block reads: the router, the expert scales and biases, the post-expert norm and the
 * shared expert travel with the block to the GPU that caches its experts, while the norms the layer executor applies
 * around the block stay with the layer's own stage.
 */
public class MoePoolTensorsTest {

    @Test
    public void theRouterTheSharedExpertAndTheScalesAreBlockTensors() {
        MoePoolTensors tensors = MoePoolTensors.of(moeLayer(3, true));
        assertTrue(tensors.contains("blk.3.ffn_gate_inp.weight"));
        assertTrue(tensors.contains("blk.3.exp_probs_b.bias"));
        assertTrue(tensors.contains("blk.3.ffn_gate_inp_shexp.weight"));
        assertTrue(tensors.contains("blk.3.ffn_gate_shexp.weight"));
        assertTrue(tensors.contains("blk.3.ffn_up_shexp.weight"));
        assertTrue(tensors.contains("blk.3.ffn_down_shexp.weight"));
        assertTrue(tensors.contains("blk.3.ffn_up_shexp.scale"));
        assertTrue(tensors.contains("blk.3.ffn_gate_exps.weight"));
        assertTrue(tensors.contains("blk.3.ffn_down_exps.bias"));
        assertTrue(tensors.contains("blk.3.ffn_post_exp_norm.weight"));
    }

    @Test
    public void theExecutorsNormsAndTheMixerStayWithTheLayer() {
        MoePoolTensors tensors = MoePoolTensors.of(moeLayer(3, true));
        assertFalse("the FFN norm is applied by the executor before the block", tensors.contains("blk.3.ffn_norm.weight"));
        assertFalse(tensors.contains("blk.3.post_ffw_norm.weight"));
        assertFalse(tensors.contains("blk.3.attn_norm.weight"));
        assertFalse(tensors.contains("blk.3.attn_q.weight"));
        assertFalse("another layer's tensor is not part of the set", tensors.contains("blk.4.ffn_gate_inp.weight"));
    }

    @Test
    public void aParallelDenseFfnTravelsWithTheBlock() {
        FfnSpec parallel = FfnSpec.builder(FfnSpec.Role.PARALLEL_DENSE).nFf(32)
                .glu(FfnGating.PAR, GluOp.SWIGLU).build();
        MoeSpec moe = MoeSpec.builder().experts(4, 2, 16).gating(GatingFunc.SOFTMAX)
                .glu(GluOp.SWIGLU).build();
        LayerSpec layer = base(5, FfnKind.MOE_PARALLEL_DENSE).ffn(parallel).moe(moe).build();
        MoePoolTensors tensors = MoePoolTensors.of(layer);
        assertTrue(tensors.contains("blk.5.ffn_gate.weight"));
        assertTrue(tensors.contains("blk.5.ffn_up.weight"));
        assertTrue(tensors.contains("blk.5.ffn_down.weight"));
    }

    @Test
    public void aDenseLayerHasNoBlockTensorsToMove() {
        assertThrows(IllegalArgumentException.class, () -> MoePoolTensors.of(base(1, FfnKind.NONE).build()));
    }

    // ------------------------------------------------------------------ fixtures

    private static LayerSpec moeLayer(int il, boolean shared) {
        MoeSpec.Builder moe = MoeSpec.builder().experts(4, 2, 16).gating(GatingFunc.SOFTMAX)
                .glu(GluOp.SWIGLU).postExpertNorm(true);
        if (shared) {
            FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(32)
                    .glu(FfnGating.PAR, GluOp.SWIGLU).build();
            moe.sharedExpert(shexp, true);
        }
        return base(il, FfnKind.MOE).moe(moe.build()).build();
    }

    private static LayerSpec.Builder base(int il, FfnKind kind) {
        return LayerSpec.builder(il, MixerKind.NONE, kind)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, 64).build());
    }
}
