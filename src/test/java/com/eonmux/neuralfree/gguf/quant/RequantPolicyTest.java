package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * {@link RequantPolicy} (QZ-1): the ported eligibility rules of llama-quant.cpp:295-396 (one keep and one drop per
 * rule), the type, rank and block rules, and the requantization producer of the planning step, dequantizing the
 * stored float rows and quantizing them exactly as {@link KvBlockCodec} does.
 */
public class RequantPolicyTest {

    private static final String KEPT = "blk.0.attn_q.weight";

    private static final String[] TIME_MIX = {"time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
        "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight",
        "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight", "time_mix_g1.weight",
        "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight"};

    private static boolean eligible(String name) {
        return RequantPolicy.eligible(name, GGMLType.F32, new long[]{128, 64}, GGMLType.Q8_0);
    }

    private static void keeps(String name) {
        assertTrue(name, eligible(name));
    }

    private static void drops(String name) {
        assertFalse(name, eligible(name));
    }

    @Test
    public void theNameRulesKeepOneAndDropOneEach() {
        keeps("blk.0.attn_q.weight");
        drops("blk.0.attn_q.bias");                       // ends with 'weight'
        keeps("token_embd.weight");
        drops("blk.0.attn_norm.weight");                  // _norm.weight
        drops("blk.0.ffn_gate_inp.weight");               // the expert gating tensor
        drops("blk.0.ffn_gate_tid2eid.weight");           // the routing table
        drops("blk.0.altup_p1.weight");                   // 4x4 helper projections
        drops("blk.0.laurel_p1.weight");
        drops("blk.0.per_layer_model_proj.weight");
        drops("pos_embd.weight");                         // BERT positional embeddings
        drops("token_types.weight");
        drops("blk.0.ssm_conv1d.weight");                 // Mamba/Kimi convolutions
        drops("blk.0.shortconv.conv.weight");
        drops("blk.0.indexer.k_proj.weight");             // MiniMax's indexer projections
        drops("blk.0.indexer.q_proj.weight");
        drops("output_hc3_proj.weight");                  // GLM5-next's hc and indexer names
        drops("blk.0.indexer.attn_q_b.weight");
        drops("blk.0.indexer.attn_k.weight");
        drops("blk.0.indexer.proj.weight");
        drops("blk.0.indexer_compressor_gate.weight");
        drops("blk.0.indexer_compressor_ape.weight");
        drops("blk.0.ssm_f_a.weight");
        drops("blk.0.ssm_f_b.weight");
        drops("blk.0.ssm_g_a.weight");
        drops("blk.0.ssm_g_b.weight");
        drops("blk.0.ssm_beta.weight");
        drops("blk.0.attn_kv_a_mqa.weight");              // MLA-a/k_b/v_b
        drops("blk.0.attn_k_b.weight");
        drops("blk.0.attn_v_b.weight");
        drops("decision.head.weight");                    // the decision head
        for (String tm : TIME_MIX) {
            drops("blk.0." + tm);                         // RWKV's time mixes
        }
        keeps("pos_embd_spatial.weight");                 // only the exact and
        keeps("token_types_bias.weight");                 // spelled drops are excluded
        keeps("blk.0.decision_gate.weight");
        keeps("blk.0.attn_k.weight");                     // the ordinary attention weights stay
        keeps("blk.0.attn_v.weight");
    }

    @Test
    public void theTypeRankAndBlockRulesDecideEligibility() {
        long[] ne = {128, 64};
        assertTrue(RequantPolicy.eligible(KEPT, GGMLType.F32, ne, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, GGMLType.F16, ne, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, GGMLType.BF16, ne, GGMLType.Q8_0));
        assertFalse("an already-quantized tensor is not requantized",
                RequantPolicy.eligible(KEPT, GGMLType.Q8_0, ne, GGMLType.Q4_0));
        assertFalse("F64 is not a listed float type",
                RequantPolicy.eligible(KEPT, GGMLType.F64, ne, GGMLType.Q8_0));
        assertFalse("rank 1 tensors are not requantized",
                RequantPolicy.eligible(KEPT, GGMLType.F32, new long[]{128}, GGMLType.Q8_0));
        assertFalse("a row that is not whole blocks of the target is not requantized",
                RequantPolicy.eligible(KEPT, GGMLType.F32, new long[]{130, 64}, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, GGMLType.F32, ne, GGMLType.IQ4_NL));
    }

    @Test
    public void requantizedRowsEqualTheKvBlockCodecOutput() throws IOException {
        int inner = 64;
        RowSource src = new FakeFloatRows(2, inner);
        RowSource weight = RequantPolicy.replace("blk.0.attn_q.weight", src, GGMLType.F32, OnlineQuant.Q8_0);
        assertTrue(weight.describe(), weight != src && weight.describe().contains("quantized to Q8_0"));
        assertEquals(GGMLType.Q8_0, weight.format().ggmlType());
        assertEquals("the shape keeps, only the row bytes change", 2, weight.rows());
        assertEquals(inner / 32 * GGMLType.Q8_0.getTypeSize(), weight.rowBytes());
        assertTrue(weight.rowLocal());
        ByteBuffer produced = ByteBuffer.allocate(2 * (int) weight.rowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        weight.readRows(0, 2, produced);
        float[] x = new float[inner];
        for (int r = 0; r < 2; r++) {
            for (int b = 0; b < inner / 32; b++) {
                for (int i = 0; i < 32; i++) {
                    x[b * 32 + i] = FakeFloatRows.value(r, i);
                }
            }
            byte[] want = new byte[34];
            KvBlockCodec.quantize(KvDType.Q8_0, x, 0, want, 0);
            byte[] mine = new byte[34];
            System.arraycopy(produced.array(), r * 34, mine, 0, 34);
            assertArrayEquals("row " + r, want, mine);
        }
    }

    @Test
    public void ineligibilityOptionNoneAndNonFloatSourcesStayUnchanged() throws IOException {
        RowSource src = new FakeFloatRows(2, 64);
        assertSame("NONE requantizes nothing",
                src, RequantPolicy.replace("blk.0.attn_q.weight", src, GGMLType.F32, OnlineQuant.NONE));
        assertSame("a renamed tensor is ineligible",
                src, RequantPolicy.replace("blk.0.attn_norm.weight", src, GGMLType.F32, OnlineQuant.Q8_0));
        assertSame("a non-float source is never requantized",
                src, RequantPolicy.replace("blk.0.attn_q.weight", new FakeFloatRows(2, 64, GGMLType.F8_E4M3),
                        GGMLType.F8_E4M3, OnlineQuant.Q8_0));
        assertSame("rows that are not whole blocks are ineligible",
                src, RequantPolicy.replace("blk.0.attn_q.weight", new FakeFloatRows(2, 33), GGMLType.F32,
                        OnlineQuant.Q8_0));
    }

    /** A rank-2 {@code <base>.weight} source of {@code rows} x {@code inner}, reading little-endian elements. */
    private static final class FakeFloatRows implements RowSource {

        private final int rows;
        private final int inner;
        private final ElementFormat format;

        FakeFloatRows(int rows, int inner) {
            this(rows, inner, GGMLType.F32);
        }

        FakeFloatRows(int rows, int inner, GGMLType type) {
            this.rows = rows;
            this.inner = inner;
            this.format = ElementFormat.of(type);
        }

        static float value(int row, int i) {
            return (row * 7 + i % 16) - 8 + 0.125f * (i % 4);
        }

        @Override
        public String describe() {
            return "fake source " + rows + "x" + inner;
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return new long[]{rows, inner};
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            return format.rowBytes(Math.max(1, inner));
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            for (int r = 0; r < count; r++) {
                for (int i = 0; i < inner; i++) {
                    if (format.stDType().equals(StDType.F32)) {
                        dst.putFloat(value((int) first + r, i));
                    } else {
                        dst.put((byte) ((first + r + i) & 0x3E));
                    }
                }
            }
        }
    }
}
