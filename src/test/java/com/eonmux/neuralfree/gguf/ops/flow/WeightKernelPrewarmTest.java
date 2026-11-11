package com.eonmux.neuralfree.gguf.ops.flow;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Which projected tensors {@link WeightKernelPrewarm} prepares the stacked Q/K/V matvecs for. */
public class WeightKernelPrewarmTest {

    @Test
    public void onlyTheSeparateAttentionQWeightsFeedTheStackedQkvPrewarm() {
        assertTrue(WeightKernelPrewarm.feedsStackedQkv("blk.0.attn_q.weight"));
        assertTrue(WeightKernelPrewarm.feedsStackedQkv("blk.27.attn_q.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv("blk.0.attn_k.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv("blk.0.attn_qkv.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv("blk.0.attn_q_a.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv("blk.0.ffn_down.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv("output.weight"));
        assertFalse(WeightKernelPrewarm.feedsStackedQkv(null));
    }
}
