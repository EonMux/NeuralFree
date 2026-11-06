package com.eonmux.neuralfree.gguf.quant;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** {@link FusedProjections}: the GGUF members of vLLM's fused projections, grouped per layer in fusion order. */
public class FusedProjectionsTest {

    @Test
    public void membersGroupPerLayerInFusionOrder() {
        assertEquals("blk.3.qkv", FusedProjections.groupOf("blk.3.attn_q.weight"));
        assertEquals("blk.3.qkv", FusedProjections.groupOf("blk.3.attn_v.weight"));
        assertEquals(0, FusedProjections.ordinalOf("blk.3.attn_q.weight"));
        assertEquals(1, FusedProjections.ordinalOf("blk.3.attn_k.weight"));
        assertEquals(2, FusedProjections.ordinalOf("blk.3.attn_v.weight"));
        assertEquals("blk.0.gate_up", FusedProjections.groupOf("blk.0.ffn_up.weight"));
        assertEquals(1, FusedProjections.ordinalOf("blk.0.ffn_up.weight"));
        assertEquals("blk.0.gate_up_shexp", FusedProjections.groupOf("blk.0.ffn_gate_shexp.weight"));
        assertEquals("blk.0.w13", FusedProjections.groupOf("blk.0.ffn_gate_exps.weight"));
        assertEquals(1, FusedProjections.ordinalOf("blk.0.ffn_up_exps.weight"));
        assertEquals("blk.12.qkv", FusedProjections.groupOf("blk.12.attn_k.weight"));
    }

    @Test
    public void mlaAndGatedDeltaNetProjectionsGroupInVllmsFusionOrder() {
        assertMember("blk.1.fused_qkv_a", 0, "blk.1.attn_q_a.weight");
        assertMember("blk.1.fused_qkv_a", 1, "blk.1.attn_kv_a_mqa.weight");
        assertMember("blk.2.in_proj_qkvz", 0, "blk.2.attn_qkv.weight");
        assertMember("blk.2.in_proj_qkvz", 1, "blk.2.attn_gate.weight");
        assertMember("blk.2.in_proj_ba", 0, "blk.2.ssm_beta.weight");
        assertMember("blk.2.in_proj_ba", 1, "blk.2.ssm_alpha.weight");
        assertMember("blk.4.kv_b", 0, "blk.4.attn_k_b.weight");
        assertMember("blk.4.kv_b", 1, "blk.4.attn_v_b.weight");
    }

    @Test
    public void groupsReportTheirLayoutAndSize() {
        assertEquals(FusedProjections.Layout.ROWS, FusedProjections.layoutOf("blk.0.attn_k.weight"));
        assertEquals(FusedProjections.Layout.ROWS, FusedProjections.layoutOf("blk.0.attn_kv_a_mqa.weight"));
        assertEquals(FusedProjections.Layout.ROWS, FusedProjections.layoutOf("blk.0.ffn_down.weight"));
        assertEquals(FusedProjections.Layout.EXPERT_SHARDS, FusedProjections.layoutOf("blk.0.ffn_up_exps.weight"));
        assertEquals(FusedProjections.Layout.MLA_KV_B, FusedProjections.layoutOf("blk.0.attn_k_b.weight"));
        assertEquals(FusedProjections.Layout.MLA_KV_B, FusedProjections.layoutOf("blk.0.attn_v_b.weight"));
        assertEquals(3, FusedProjections.sizeOf("blk.0.attn_q.weight"));
        assertEquals(2, FusedProjections.sizeOf("blk.0.attn_v_b.weight"));
        assertEquals(2, FusedProjections.sizeOf("blk.0.ssm_alpha.weight"));
        assertEquals(0, FusedProjections.sizeOf("blk.0.ffn_down.weight"));
    }

    @Test
    public void onlyTheMlaKeyPartIsTransposed() {
        assertTrue(FusedProjections.transposed("blk.7.attn_k_b.weight"));
        assertFalse(FusedProjections.transposed("blk.7.attn_v_b.weight"));
        assertFalse(FusedProjections.transposed("blk.7.attn_k.weight"));
        assertFalse(FusedProjections.transposed("blk.7.attn_k_b.bias"));
    }

    @Test
    public void unfusedTensorsHaveNoGroup() {
        for (String name : new String[]{"blk.0.attn_output.weight", "blk.0.ffn_down.weight",
            "blk.0.ffn_down_exps.weight", "blk.0.ffn_down_shexp.weight", "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_gate_up_exps.weight", "blk.0.ssm_out.weight",
            "blk.0.attn_q.bias", "token_embd.weight", "output.weight", "blk.0.attn_q_norm.weight",
            "blk.0.attn_q_b.weight", "blk.0.attn_kv_b.weight"}) {
            assertNull(name, FusedProjections.groupOf(name));
            assertEquals(name, -1, FusedProjections.ordinalOf(name));
            assertEquals(name, 0, FusedProjections.sizeOf(name));
        }
    }

    private static void assertMember(String group, int ordinal, String name) {
        assertEquals(name, group, FusedProjections.groupOf(name));
        assertEquals(name, ordinal, FusedProjections.ordinalOf(name));
        assertEquals(name, 2, FusedProjections.sizeOf(name));
    }
}
