package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.LayerKind;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link RequantPolicy}: the ported eligibility rules of llama-quant.cpp:302-421 (one keep and one drop per
 * rule), the type, rank and block rules, and the requantization producer of the planning step, dequantizing the
 * stored float rows and quantizing them exactly as {@link KvBlockCodec} does, and the shared scales of a fused
 * projection's members.
 */
public class RequantPolicyTest {

    private static final String KEPT = "blk.0.attn_q.weight";

    private static final String[] TIME_MIX = {"time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
        "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight",
        "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight", "time_mix_g1.weight",
        "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight"};

    private static boolean eligible(String name) {
        return RequantPolicy.eligible(name, Arch.LLAMA, true, GGMLType.F32, new long[]{128, 64}, GGMLType.Q8_0);
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
        keeps("output_hc_fn.weight");                     // GLM5-next's names are gated on its architecture
        keeps("blk.0.hc_attn_fn.weight");
        keeps("blk.0.indexer.attn_q_b.weight");
        keeps("blk.0.indexer.attn_k.weight");
        keeps("blk.0.indexer.proj.weight");
        keeps("blk.0.indexer_compressor_gate.weight");
        keeps("blk.0.indexer_compressor_ape.weight");
        keeps("blk.0.ssm_f_a.weight");
        keeps("blk.0.ssm_f_b.weight");
        keeps("blk.0.ssm_g_a.weight");
        keeps("blk.0.ssm_g_b.weight");
        keeps("blk.0.ssm_beta.weight");
        keeps("blk.0.attn_kv_a_mqa.weight");              // MLA-a/k_b/v_b
        keeps("blk.0.attn_k_b.weight");
        keeps("blk.0.attn_v_b.weight");
        keeps("decision.head.weight");                    // the decision head
        drops("blk.0.engram_gate.weight");                // the Engram elementwise gate
        drops("blk.0.conv2d_spatial.weight");             // LFM3 depthwise and audio convolutions
        drops("blk.0.conv_dw.weight");
        drops("a.conv1d.0.weight");
        drops("a.gen.wav.quant.proj.weight");
        drops("blk.0.attn_rel_b.weight");                 // T5's relative position bias
        drops("v.position_embd.weight");                  // multimodal position, patch and codebook tensors
        drops("v.sam.pos_embd.weight");
        drops("v.sam.neck.0.weight");
        drops("v.sam.net_2.weight");
        drops("v.blk.0.attn.rel_pos_h.weight");
        drops("v.patch_embd.weight");
        drops("mm.patch_merger.weight");
        drops("a.rvq.codebook.0.weight");
        drops("mm.a.code_embd.weight");
        for (String tm : TIME_MIX) {
            drops("blk.0." + tm);                         // RWKV's time mixes
        }
        keeps("pos_embd_spatial.weight");                 // only the exact and
        keeps("token_types_bias.weight");                 // spelled drops are excluded
        keeps("blk.0.decision_gate.weight");
        keeps("enc.a.conv1d.0.weight");                   // the LFM3 audio rules are prefixes
        keeps("blk.0.output_hc3_proj.weight");            // 'hc_' is matched as spelled
        keeps("blk.0.attn_k.weight");                     // the ordinary attention weights stay
        keeps("blk.0.attn_v.weight");
    }

    @Test
    public void theTypeRankAndBlockRulesDecideEligibility() {
        long[] ne = {128, 64};
        assertTrue(RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F32, ne, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F16, ne, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.BF16, ne, GGMLType.Q8_0));
        assertFalse("an already-quantized tensor is not requantized",
                RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.Q8_0, ne, GGMLType.Q4_0));
        assertFalse("F64 is not a listed float type",
                RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F64, ne, GGMLType.Q8_0));
        assertFalse("rank 1 tensors are not requantized",
                RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F32, new long[]{128}, GGMLType.Q8_0));
        assertFalse("a row that is not whole blocks of the target is not requantized",
                RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F32, new long[]{130, 64}, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, Arch.LLAMA, true, GGMLType.F32, ne, GGMLType.IQ4_NL));
    }

    @Test
    public void architectureGatedNamesQuantizeOnlyOnTheirModel() {
        long[] ne = {128, 64};
        for (String name : new String[]{"blk.0.attn_gate.weight", "blk.0.attn_ve_gate.weight",
                "blk.0.mudd_mixer.weight"}) {
            assertFalse(name + " is a LIMITE gate", RequantPolicy.eligible(name, Arch.LIMITE, true,
                    GGMLType.F32, ne, GGMLType.Q8_0));
            assertTrue(name + " is an ordinary matmul elsewhere", RequantPolicy.eligible(name, Arch.LLAMA, true,
                    GGMLType.F32, ne, GGMLType.Q8_0));
        }
        assertTrue("an ordinary projection still quantizes on LIMITE",
                RequantPolicy.eligible("blk.0.attn_q.weight", Arch.LIMITE, true, GGMLType.F32, ne, GGMLType.Q8_0));
        String[] glm5 = {"output_hc_fn.weight", "blk.0.hc_attn_fn.weight", "blk.0.indexer.attn_q_b.weight",
            "blk.0.indexer.attn_k.weight", "blk.0.indexer.proj.weight", "blk.0.indexer_compressor_gate.weight",
            "blk.0.indexer_compressor_ape.weight", "blk.0.ssm_f_a.weight", "blk.0.ssm_f_b.weight",
            "blk.0.ssm_g_a.weight", "blk.0.ssm_g_b.weight", "blk.0.ssm_beta.weight",
            "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"};
        for (String name : glm5) {
            assertFalse(name + " is a GLM5-next tensor", RequantPolicy.eligible(name, Arch.GLM5_NEXT, true,
                    GGMLType.F32, ne, GGMLType.Q8_0));
            assertTrue(name + " is an ordinary matmul elsewhere", RequantPolicy.eligible(name, Arch.LLAMA, true,
                    GGMLType.F32, ne, GGMLType.Q8_0));
        }
        for (String name : new String[]{"blk.0.attn_kv_a_mqa.weight", "blk.0.attn_k_b.weight",
                "blk.0.attn_v_b.weight"}) {
            assertTrue(name + " is an ordinary matmul on DeepSeek", RequantPolicy.eligible(name, Arch.DEEPSEEK2,
                    true, GGMLType.F32, ne, GGMLType.Q8_0));
        }
        assertFalse("the decision head is gated on vllm-decision2",
                RequantPolicy.eligible("decision.head.weight", Arch.VLLM_DECISION2, true, GGMLType.F32, ne,
                        GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible("decision.head.weight", Arch.LLAMA, true, GGMLType.F32, ne,
                GGMLType.Q8_0));
        assertTrue("vLLM's online methods still quantize kv_b on GLM5-next",
                RequantPolicy.nameEligible("blk.0.attn_k_b.weight", Arch.GLM5_NEXT, true, true));
        assertFalse("but never the decision head",
                RequantPolicy.nameEligible("decision.head.weight", Arch.VLLM_DECISION2, true, true));
    }

    @Test
    public void theOutputTensorIsQuantizedUnlessLeftAsStored() {
        long[] ne = {128, 64};
        assertTrue(RequantPolicy.eligible("output.weight", Arch.LLAMA, true, GGMLType.F32, ne, GGMLType.Q8_0));
        assertFalse("--leave-output-tensor keeps the LM head",
                RequantPolicy.eligible("output.weight", Arch.LLAMA, false, GGMLType.F32, ne, GGMLType.Q8_0));
        assertTrue(RequantPolicy.eligible(KEPT, Arch.LLAMA, false, GGMLType.F32, ne, GGMLType.Q8_0));
    }

    @Test
    public void requantizedRowsEqualTheKvBlockCodecOutput() throws IOException {
        int inner = 64;
        RowSource src = new FakeFloatRows(2, inner);
        RowSource weight = RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.Q8_0);
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
            int rowBytes = (int) weight.rowBytes();
            byte[] want = new byte[rowBytes];
            for (int b = 0; b < inner / 32; b++) {
                KvBlockCodec.quantize(KvDType.Q8_0, x, b * 32, want, b * 34);
            }
            byte[] mine = new byte[rowBytes];
            System.arraycopy(produced.array(), r * rowBytes, mine, 0, rowBytes);
            assertArrayEquals("row " + r, want, mine);
        }
    }

    @Test
    public void ineligibilityOptionNoneAndNonFloatSourcesStayUnchanged() throws IOException {
        RowSource src = new FakeFloatRows(2, 64);
        assertSame("NONE requantizes nothing",
                src, RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.NONE));
        assertSame("a renamed tensor is ineligible",
                src, RequantPolicy.replace("blk.0.attn_norm.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.Q8_0));
        RowSource fp8 = new FakeFloatRows(2, 64, GGMLType.F8_E4M3);
        assertSame("a non-float source is never requantized",
                fp8, RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, fp8, GGMLType.F8_E4M3, OnlineQuant.Q8_0));
        RowSource partial = new FakeFloatRows(2, 33);
        assertSame("rows that are not whole blocks are ineligible",
                partial, RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, partial, GGMLType.F32, OnlineQuant.Q8_0));
    }

    @Test
    public void vllmMethodsQuantizeTheLinearAndRoutedExpertWeightsTheyTarget() {
        for (OnlineQuant q : new OnlineQuant[]{OnlineQuant.FP8_PER_TENSOR, OnlineQuant.FP8_PER_BLOCK,
                OnlineQuant.FP8_PER_CHANNEL, OnlineQuant.MXFP8, OnlineQuant.MXFP4}) {
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.attn_q.weight", q));
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_down_shexp.weight", q));
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_gate_exps.weight", q));
            assertFalse(q.name() + " leaves the embedding", RequantPolicy.vllmQuantizes("token_embd.weight", q));
            assertFalse(q.name() + " leaves the LM head", RequantPolicy.vllmQuantizes("output.weight", q));
            assertFalse(q.name(), RequantPolicy.vllmQuantizes("per_layer_token_embd.weight", q));
            assertFalse(q.name(), RequantPolicy.vllmQuantizes("blk.3.nextn.shared_head_head.weight", q));
            assertFalse(q.name(), RequantPolicy.vllmQuantizes("blk.3.nextn.embed_tokens.weight", q));
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.3.nextn.eh_proj.weight", q));
            assertFalse(q.name() + " leaves the shared-expert gate, built without a quantization config",
                    RequantPolicy.vllmQuantizes("blk.0.ffn_gate_inp_shexp.weight", q));
        }
        for (OnlineQuant q : new OnlineQuant[]{OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY, OnlineQuant.EXPERTS_INT8,
                OnlineQuant.NVFP4_PER_TOKEN}) {
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_gate_exps.weight", q));
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_gate_up_exps.weight", q));
            assertTrue(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_down_exps.weight", q));
            assertFalse(q.name() + " leaves the linears", RequantPolicy.vllmQuantizes("blk.0.attn_q.weight", q));
            assertFalse(q.name() + " leaves the shared experts",
                    RequantPolicy.vllmQuantizes("blk.0.ffn_up_shexp.weight", q));
            assertFalse(q.name(), RequantPolicy.vllmQuantizes("blk.0.ffn_gate.weight", q));
            assertFalse(q.name(), RequantPolicy.vllmQuantizes("output.weight", q));
        }
        assertTrue("the llama-quantize targets keep llama-quantize's rules",
                RequantPolicy.vllmQuantizes("token_embd.weight", OnlineQuant.Q8_0));
        RowSource src = new FakeFloatRows(2, 128);
        assertSame(src, RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.EXPERTS_INT8));
        assertSame(src, RequantPolicy.replace("output.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.FP8_PER_CHANNEL));
        assertSame(GGMLType.F8_E4M3_B128, RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32,
                OnlineQuant.FP8_PER_CHANNEL).format().ggmlType());
    }

    @Test
    public void aScanningTargetQuantizesAStackedTensorPerExpertAndReadsAnyRowRange() throws IOException {
        int experts = 3;
        int perExpert = 2;
        int inner = 128;
        FakeFloatRows src = new FakeFloatRows(new long[]{experts, perExpert, inner}, GGMLType.F32);
        RowSource w = RequantPolicy.replace("blk.0.ffn_up_exps.weight", Arch.LLAMA, true, src, GGMLType.F32,
                OnlineQuant.NVFP4_PER_TOKEN);
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        assertArrayEquals(new long[]{experts, perExpert, inner}, w.shape());

        ScanningQuantizer s = OnlineQuantEncoders.nvfp4PerToken().open(experts * perExpert, inner, experts);
        float[][] rows = new float[experts * perExpert][inner];
        for (int r = 0; r < rows.length; r++) {
            for (int i = 0; i < inner; i++) {
                rows[r][i] = FakeFloatRows.value(r, i);
            }
            s.scanRow(r, rows[r], 0, inner);
        }
        s.freeze();
        assertArrayEquals("one weight_scale_2 per expert", s.outputScale(), w.outputScale(), 0f);
        assertEquals(experts, w.outputScale().length);
        int rowBytes = (int) w.rowBytes();
        byte[] want = new byte[rows.length * rowBytes];
        for (int r = 0; r < rows.length; r++) {
            s.quantizeRow(r, rows[r], 0, inner, want, r * rowBytes);
        }
        ByteBuffer all = ByteBuffer.allocate(want.length);
        w.readRows(0, rows.length, all);
        assertArrayEquals(want, all.array());
        ByteBuffer middle = ByteBuffer.allocate(2 * rowBytes);
        w.readRows(3, 2, middle);
        byte[] slice = new byte[2 * rowBytes];
        System.arraycopy(want, 3 * rowBytes, slice, 0, slice.length);
        assertArrayEquals("a row range reads the rows the whole-tensor scan quantized", slice, middle.array());
    }

    @Test
    public void layerKindsAreVllmsLayerClasses() {
        assertSame(LayerKind.LINEAR, RequantPolicy.layerKind("blk.0.attn_q.weight"));
        assertSame(LayerKind.LINEAR, RequantPolicy.layerKind("blk.0.ffn_down_shexp.weight"));
        assertSame(LayerKind.MOE, RequantPolicy.layerKind("blk.0.ffn_gate_up_exps.weight"));
        assertSame(LayerKind.EMBEDDING, RequantPolicy.layerKind("token_embd.weight"));
        assertSame(LayerKind.EMBEDDING, RequantPolicy.layerKind("output.weight"));
        assertSame(LayerKind.EMBEDDING, RequantPolicy.layerKind("blk.3.nextn.shared_head_head.weight"));
        assertNull(RequantPolicy.layerKind("blk.0.ffn_gate_inp_shexp.weight"));
        assertFalse("the bare online method selects per layer", RequantPolicy.vllmQuantizes("blk.0.attn_q.weight",
                OnlineQuant.ONLINE));
        RowSource src = new FakeFloatRows(2, 64);
        assertThrows(IllegalArgumentException.class,
                () -> RequantPolicy.replace("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.ONLINE));
        assertThrows(IllegalArgumentException.class,
                () -> RequantPolicy.quantize("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32, OnlineQuant.NONE));
        assertSame(GGMLType.MXFP4, RequantPolicy.quantize("blk.0.attn_q.weight", Arch.LLAMA, true, src, GGMLType.F32,
                OnlineQuant.MXFP4).format().ggmlType());
        RowSource partial = new FakeFloatRows(2, 48);
        assertSame("an MXFP4 row is whole 32-element blocks", partial,
                RequantPolicy.quantize("blk.0.attn_q.weight", Arch.LLAMA, true, partial, GGMLType.F32, OnlineQuant.MXFP4));
    }

    @Test
    public void fusedMembersShareOnePerTensorScaleOverTheirConcatenatedRows() throws IOException {
        int inner = 128;
        ArrayRows q = new ArrayRows(new long[]{4, inner}, 1, 0.5f);
        ArrayRows k = new ArrayRows(new long[]{2, inner}, 2, 3.0f);
        ArrayRows v = new ArrayRows(new long[]{2, inner}, 3, 1.0f);
        List<RowSource> fused = RequantPolicy.quantizeFused(Arrays.asList("blk.0.attn_q.weight",
                "blk.0.attn_k.weight", "blk.0.attn_v.weight"), Arrays.<RowSource>asList(q, k, v),
                OnlineQuant.FP8_PER_TENSOR);
        byte[][] want = reference(OnlineQuantEncoders.fp8PerTensor(), 1, new ArrayRows[]{q, k, v},
                new long[][]{{0, 4}, {4, 2}, {6, 2}}, 8);
        for (int m = 0; m < 3; m++) {
            assertEquals(GGMLType.F8_E4M3_B128, fused.get(m).format().ggmlType());
            assertArrayEquals("member " + m, want[m], read(fused.get(m)));
        }
        float scale = ByteBuffer.wrap(want[0]).order(ByteOrder.LITTLE_ENDIAN).getFloat(0);
        assertEquals("the scale is the group's amax (k's) over 448", k.amax() / 448.0f, scale, 0.0f);
        byte[] alone = read(RequantPolicy.quantize("blk.0.attn_q.weight", Arch.LLAMA, true, q, GGMLType.F32,
                OnlineQuant.FP8_PER_TENSOR));
        assertFalse("a member quantized alone has its own scale", Arrays.equals(want[0], alone));
    }

    @Test
    public void routedExpertShardsStartOnWhole128RowTilesOfEachExpert() throws IOException {
        int inner = 256;
        int experts = 2;
        ArrayRows gate = new ArrayRows(new long[]{experts, 3, inner}, 4, 1.0f);
        ArrayRows up = new ArrayRows(new long[]{experts, 3, inner}, 5, 2.0f);
        List<String> names = Arrays.asList("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight");
        List<RowSource> shards = RequantPolicy.quantizeFused(names, Arrays.<RowSource>asList(gate, up),
                OnlineQuant.FP8_PER_BLOCK);
        // Per expert, gate fills rows 0..2 of a 128-row tile and up starts on the next tile: 256 virtual rows.
        byte[][] padded = reference(OnlineQuantEncoders.fp8PerBlock(), experts, new ArrayRows[]{gate, up},
                new long[][]{{0, 3}, {128, 3}}, 256);
        assertArrayEquals(padded[0], read(shards.get(0)));
        assertArrayEquals(padded[1], read(shards.get(1)));
        // Without the shard padding, gate and up rows of an expert would share one row tile.
        byte[][] tiled = reference(OnlineQuantEncoders.fp8PerBlock(), experts, new ArrayRows[]{gate, up},
                new long[][]{{0, 3}, {3, 3}}, 6);
        assertFalse(Arrays.equals(padded[0], tiled[0]));
    }

    @Test
    public void aFusedGroupNeedsATensorScopedTargetAndMatchingMembers() {
        ArrayRows a = new ArrayRows(new long[]{2, 128}, 6, 1.0f);
        ArrayRows wide = new ArrayRows(new long[]{2, 256}, 7, 1.0f);
        List<String> names = Arrays.asList("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight");
        assertThrows(IllegalArgumentException.class, () -> RequantPolicy.quantizeFused(names,
                Arrays.<RowSource>asList(a, a), OnlineQuant.MXFP4));
        String e = assertThrows(IllegalArgumentException.class, () -> RequantPolicy.quantizeFused(names,
                Arrays.<RowSource>asList(a, wide), OnlineQuant.FP8_PER_TENSOR)).getMessage();
        assertTrue(e, e.contains("mixes row lengths"));
        assertThrows(IllegalArgumentException.class, () -> RequantPolicy.quantizeFused(names,
                Arrays.<RowSource>asList(a), OnlineQuant.FP8_PER_TENSOR));
    }

    /**
     * Each member quantized by one scan of {@code prototype} over a virtual tensor of {@code virtualPerExpert}
     * rows per expert, member {@code m} at rows {@code spans[m][0]..+spans[m][1]} of each expert.
     */
    private static byte[][] reference(ScanningQuantizer prototype, int experts, ArrayRows[] members, long[][] spans,
                                      int virtualPerExpert) {
        int inner = members[0].inner;
        ScanningQuantizer s = prototype.open((long) experts * virtualPerExpert, inner, experts);
        for (int m = 0; m < members.length; m++) {
            for (int e = 0; e < experts; e++) {
                for (int r = 0; r < spans[m][1]; r++) {
                    s.scanRow(e * virtualPerExpert + spans[m][0] + r,
                            members[m].row(e * (int) spans[m][1] + r), 0, inner);
                }
            }
        }
        s.freeze();
        int rowBytes = (int) prototype.type().getBytesForElements(inner);
        byte[][] out = new byte[members.length][];
        for (int m = 0; m < members.length; m++) {
            out[m] = new byte[(int) members[m].rows() * rowBytes];
            for (int e = 0; e < experts; e++) {
                for (int r = 0; r < spans[m][1]; r++) {
                    int row = e * (int) spans[m][1] + r;
                    s.quantizeRow(e * virtualPerExpert + spans[m][0] + r, members[m].row(row), 0, inner, out[m],
                            row * rowBytes);
                }
            }
        }
        return out;
    }

    private static byte[] read(RowSource w) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(w.rows() * w.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        w.readRows(0, (int) w.rows(), b);
        return b.array();
    }

    /** An F32 source of pseudo-random values of magnitude up to {@code scale}. */
    private static final class ArrayRows implements RowSource {
        private final long[] shape;
        private final int inner;
        private final float[][] rows;

        ArrayRows(long[] shape, long seed, float scale) {
            this.shape = shape.clone();
            this.inner = (int) shape[shape.length - 1];
            int n = 1;
            for (int d = 0; d < shape.length - 1; d++) {
                n *= (int) shape[d];
            }
            Random rnd = new Random(seed);
            this.rows = new float[n][inner];
            for (float[] row : rows) {
                for (int i = 0; i < inner; i++) {
                    row[i] = (rnd.nextFloat() * 2 - 1) * scale;
                }
            }
        }

        float[] row(int r) {
            return rows[r];
        }

        float amax() {
            float a = 0;
            for (float[] row : rows) {
                for (float x : row) {
                    a = Math.max(a, Math.abs(x));
                }
            }
            return a;
        }

        @Override
        public String describe() {
            return "array source " + Arrays.toString(shape);
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            return rows.length;
        }

        @Override
        public long rowBytes() {
            return 4L * inner;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            for (int r = 0; r < count; r++) {
                for (int i = 0; i < inner; i++) {
                    dst.putFloat(rows[(int) first + r][i]);
                }
            }
        }
    }

    /** A {@code <base>.weight} source of the given shape, reading little-endian elements. */
    private static final class FakeFloatRows implements RowSource {

        private final long[] shape;
        private final int rows;
        private final int inner;
        private final ElementFormat format;

        FakeFloatRows(int rows, int inner) {
            this(rows, inner, GGMLType.F32);
        }

        FakeFloatRows(int rows, int inner, GGMLType type) {
            this(new long[]{rows, inner}, type);
        }

        FakeFloatRows(long[] shape, GGMLType type) {
            this.shape = shape.clone();
            long r = 1;
            for (int d = 0; d < shape.length - 1; d++) {
                r *= shape[d];
            }
            this.rows = (int) r;
            this.inner = (int) shape[shape.length - 1];
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
            return shape.clone();
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
