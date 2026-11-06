package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfLinearHooksB#qwen4ExpTensors} against LC/conversion/qwen4exp.py: the shared-input merges of
 * {@code _merge_shared_input} (:177-208) with their {@code prepare_tensors} check (:250-262), the QSA indexer split
 * and the MTP {@code fc_embedding}/{@code fc_hidden} concat (:148-161).
 */
public class HfQwen4ExpTensorsTest {

    /** Two linear heads of one entry each; layer 0 linear attention, layer 1 full attention. */
    private static final String CONFIG = "'linear_num_key_heads': 2, 'linear_num_value_heads': 2,"
            + " 'linear_key_head_dim': 1, 'linear_value_head_dim': 1, 'linear_conv_kernel_dim': 4,"
            + " 'num_experts': 2, 'num_experts_per_tok': 1, 'moe_intermediate_size': 2,"
            + " 'shared_expert_intermediate_size': 6, 'layer_types': ['linear_attention', 'full_attention'],"
            + " 'full_attention_interval': 2, 'head_dim': 8, 'indexer_n_heads': 2, 'indexer_head_dim': 2";

    private static final String QKV = "model.layers.0.linear_attn.in_proj_qkv.weight";
    private static final String Z = "model.layers.0.linear_attn.in_proj_z.weight";
    private static final String B = "model.layers.0.linear_attn.in_proj_b.weight";
    private static final String A = "model.layers.0.linear_attn.in_proj_a.weight";

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private HfLoadContext context(String extra) throws IOException {
        return HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.QWEN4EXP, "Qwen4ExpForCausalLM",
                CONFIG + extra, Collections.emptyMap(), false);
    }

    private static void feed(HfLoadContext ctx, RecordingSink sink, String name, RowSource src) throws IOException {
        HfLinearHooksB.qwen4ExpTensors(ctx, name, src, sink);
    }

    @Test
    public void linearAttentionPairsMergeRowsOfTheFirstHalfFirst() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, Z, offset(seq(2, 3), 100));
        assertTrue("the first half waits for its partner", sink.tensors.isEmpty());
        feed(ctx, sink, QKV, seq(6, 3));
        feed(ctx, sink, B, offset(seq(2, 3), 200));
        feed(ctx, sink, A, offset(seq(2, 3), 300));
        assertEquals(new HashSet<>(Arrays.asList("blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight")),
                sink.tensors.keySet());
        RowSource qkvGate = sink.tensors.get("blk.0.attn_qkv_gate.weight");
        assertArrayEquals(new long[]{8, 3}, qkvGate.shape());
        float[] v = values(qkvGate);
        assertEquals(0f, v[0], 0f);
        assertEquals("attn_gate rows follow the six attn_qkv rows", 100f, v[6 * 3], 0f);
        float[] ba = values(sink.tensors.get("blk.0.ssm_beta_alpha.weight"));
        assertEquals("beta rows first", 200f, ba[0], 0f);
        assertEquals("then alpha", 300f, ba[2 * 3], 0f);
        HfLinearHooksB.qwen4ExpExtra(ctx, sink);
    }

    @Test
    public void hyperConnectionPairsMergeInEveryBlock() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, "model.layers.1.attn_hyper_connection.block_inject_weight.weight", offset(seq(4, 16), 50));
        feed(ctx, sink, "model.layers.1.attn_hyper_connection.input_mix_weight_down.weight", seq(3, 16));
        RowSource merged = sink.tensors.get("blk.1.hc_attn_down_inject.weight");
        assertEquals(sink.tensors.keySet().toString(), 1, sink.tensors.size());
        assertArrayEquals(new long[]{7, 16}, merged.shape());
        assertEquals("inject rows follow the down rows", 50f, values(merged)[3 * 16], 0f);
    }

    @Test
    public void aPreQuantizedCheckpointKeepsThePairsSeparate() throws IOException {
        HfLoadContext ctx = context(", 'quantization_config': {'quant_method': 'fp8'}");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, QKV, seq(6, 3));
        feed(ctx, sink, Z, seq(2, 3));
        assertEquals(new HashSet<>(Arrays.asList("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight")),
                sink.tensors.keySet());
    }

    @Test
    public void aHalfWithoutItsPartnerIsRefusedAtTheEnd() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, B, seq(2, 3));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfLinearHooksB.qwen4ExpExtra(ctx, sink));
        assertEquals("merged weights missing a half: [0:SSM_BETA_ALPHA_CAT]", e.getMessage());
    }

    @Test
    public void theIndexerProjectionSplitsIntoQueryAndKey() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, "model.layers.1.self_attn.indexer.index_qk_proj.weight", seq(6, 3));
        assertEquals(new HashSet<>(Arrays.asList("blk.1.indexer.q_proj.weight", "blk.1.indexer.k_proj.weight")),
                sink.tensors.keySet());
        assertArrayEquals(new long[]{4, 3}, sink.tensors.get("blk.1.indexer.q_proj.weight").shape());
        assertArrayEquals(new long[]{2, 3}, sink.tensors.get("blk.1.indexer.k_proj.weight").shape());
    }

    @Test
    public void theMtpProjectionHalvesConcatenateIntoEhProj() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        feed(ctx, sink, "model.layers.2.nextn_fc_embedding.weight", seq(4, 2));
        feed(ctx, sink, "model.layers.2.nextn_fc_hidden.weight", seq(4, 3));
        assertEquals(Collections.singleton("blk.2.nextn.eh_proj.weight"), sink.tensors.keySet());
        assertArrayEquals(new long[]{4, 5}, sink.tensors.get("blk.2.nextn.eh_proj.weight").shape());
    }

    /** {@code src} plus {@code delta}, so the halves of a merge are told apart by value. */
    private static RowSource offset(RowSource src, float delta) throws IOException {
        float[] v = values(src);
        for (int i = 0; i < v.length; i++) {
            v[i] += delta;
        }
        return RopeTensors.fromFloats(src.shape(), v);
    }
}
