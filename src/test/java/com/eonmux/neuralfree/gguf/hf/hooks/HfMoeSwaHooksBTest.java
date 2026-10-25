package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The part B hooks: the MiMo-V2 FP8 qkv index hook that regroups the TP-sharded {@code [Q|K|V]} rows
 * (mimo.py:74-172), the MTP rename and expert stacking (mimo.py:319-403), the Beam layer-type validation and
 * per-head gate bias (beam.py:25-141).
 */
public class HfMoeSwaHooksBTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ MimoV2Model

    private static final String MIMO_CONFIG = "'quantization_config': {'quant_method': 'fp8'},"
            + " 'hybrid_layer_pattern': [0, 1], 'swa_num_key_value_heads': 1,"
            + " 'v_head_dim': 4, 'head_dim': 4, 'sliding_window': 4, 'n_routed_experts': 2,"
            + " 'moe_intermediate_size': 4, 'partial_rotary_factor': 0.5, 'num_nextn_predict_layers': 1";

    @Test
    public void mimoIndexDequantizesAndRegroupsTheTpShardedQkv() throws IOException {
        // q 8 + k 4 + v 4 = 16 rows detected at TP 4 (scale rows [4,1]); scales are rank-major:
        // a stored byte 0x38 is the e4m3 code of 1.0, so a row dequantizes to its rank's block scale
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.qkv_proj.weight", u8(16, 8));
        index.put("model.layers.0.self_attn.qkv_proj.weight_scale_inv",
                com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.fromFloats(new long[]{4, 1},
                        new float[]{1, 2, 3, 4}));
        HfLoadContext ctx = context(dir(), Arch.MIMO2, "MiMoV2FlashForCausalLM", MIMO_CONFIG, index, false);
        com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex remapped =
                HfMoeSwaHooksB.mimoIndex(ctx, ctx.index());
        RowSource qkv = remapped.get("model.layers.0.self_attn.qkv_proj.weight");
        assertArrayEquals(new long[]{16, 8}, qkv.shape());
        assertTrue(!remapped.has("model.layers.0.self_attn.qkv_proj.weight_scale_inv"));
        // the fused-order rows dequantize to 1.0 * the block scale of their (rank, block) scale row: the
        // first eight output rows read sharded rows 0..7, rows 8..11 read sharded rows 4..7 again ranked
        // after the q block and rows 12..15 the v block of ranks 2..3; rows 0..3 rank 0, 4..7 rank 1 and
        // 8..15 rank 2 (with the same 16-row weight matrix two reads share a shard)
        float[] out = values(qkv);
        for (int row = 0; row < 16; row++) {
            long oldRow = row < 8 ? row : 4 + (row % 8);
            float expected = oldRow / 4 + 1;
            for (int j = 0; j < 8; j++) {
                org.junit.Assert.assertEquals("row " + row + " col " + j, expected, out[row * 8 + j], 1e-6f);
            }
        }
    }

    @Test
    public void mimoWithoutAFp8QuantConfigKeepsTheIndex() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.qkv_proj.weight", u8(16, 8));
        HfLoadContext ctx = context(dir(), Arch.MIMO2, "MiMoV2FlashForCausalLM",
                MIMO_CONFIG.replace("'quantization_config': {'quant_method': 'fp8'},", ""),
                index, false);
        com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex remapped =
                HfMoeSwaHooksB.mimoIndex(ctx, ctx.index());
        assertTrue(remapped == ctx.index());
    }

    @Test
    public void mimoUnquantizedScalesAreRejectedAndMtpExpertsStack() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.qkv_proj.weight", u8(16, 8));
        // scale rows 3 do not match the TP 4/8 detection from the 16x8 matrix
        index.put("model.layers.0.self_attn.qkv_proj.weight_scale_inv", seq(3, 1));
        HfLoadContext ctx = context(dir(), Arch.MIMO2, "MiMoV2FlashForCausalLM", MIMO_CONFIG, index, false);
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksB.mimoIndex(ctx, ctx.index()));
        HfLoadContext mtp = context(dir(), Arch.MIMO2, "MiMoV2FlashForCausalLM", MIMO_CONFIG,
                java.util.Collections.emptyMap(), true);
        RecordingSink sink = new RecordingSink();
        HfMoeSwaHooksB.mimoTensors(mtp, "model.mtp.layers.0.mlp.experts.0.gate_proj.weight", seq(2, 8), sink);
        HfMoeSwaHooksB.mimoTensors(mtp, "model.mtp.layers.0.mlp.experts.1.gate_proj.weight", seq(2, 8), sink);
        HfMoeSwaHooksB.mimoExtra(mtp, sink);
        assertTrue(!sink.tensors.isEmpty());
    }

    // ------------------------------------------------------------------ BeamModel

    @Test
    public void beamRequiresALeadingDenseOnlyMlpPatternAndRejectsBias() throws IOException {
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksB.beamFixup(config(
                "'layer_types': ['full_attention', 'full_attention'],"
                        + " 'mlp_layer_types': ['sparse', 'sparse']")));
        HfMoeSwaHooksB.beamFixup(config("'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'mlp_layer_types': ['dense', 'sparse'], 'attention_bias': false"));
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksB.beamFixup(config(
                "'layer_types': ['sliding_attention', 'full_attention'],"
                        + " 'mlp_layer_types': ['dense', 'sparse'], 'attention_bias': true")));
    }

    @Test
    public void beamExpertsStackPerProjectionAndTheGateBiasIsRenamed() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.1.mlp.experts." + x + ".gate_proj.weight", seq(2, 8));
            index.put("model.layers.1.mlp.experts." + x + ".up_proj.weight", seq(2, 8));
            index.put("model.layers.1.mlp.experts." + x + ".down_proj.weight", seq(2, 2));
        }
        index.put("model.layers.1.self_attn.gate_bias", seq(2));
        HfLoadContext ctx = context(dir(), Arch.BEAM, "BeamForCausalLM", "'num_local_experts': 2,", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksB.beamTensors(ctx, name, ctx.index().get(name), sink);
        }
        HfMoeSwaHooksB.beamExtra(ctx, sink);
        assertTrue(sink.tensors.size() >= 3);
        HfLoadContext dense = context(dir(), Arch.BEAM, "BeamForCausalLM", "'num_local_experts': 2,",
                java.util.Collections.singletonMap("model.layers.0.mlp.experts.0.gate_proj.weight", seq(2, 8)),
                false);
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksB.beamTensors(dense,
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                dense.index().get("model.layers.0.mlp.experts.0.gate_proj.weight"), new RecordingSink()));
    }

    // ------------------------------------------------------------------ helpers

    private com.eonmux.neuralfree.gguf.hf.config.HfConfig config(String extra) throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BEAM, "BeamForCausalLM", extra);
        return ctx.config();
    }

    /** A U8 source of {@code shape[rows, k]} holding byte 0x38 (the e4m3 code of 1.0) per element. */
    static RowSource u8(int rows, int k) {
        byte[] bytes = new byte[rows * k];
        Arrays.fill(bytes, (byte) 0x38);
        return new U8Source(rows, k, bytes);
    }

    static final class U8Source implements RowSource {
        private final int rows;
        private final int k;
        private final byte[] bytes;

        U8Source(int rows, int k, byte[] bytes) {
            this.rows = rows;
            this.k = k;
            this.bytes = bytes;
        }

        @Override
        public String describe() {
            return "u8 " + rows + "x" + k;
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.I8);
        }

        @Override
        public long[] shape() {
            return new long[]{rows, k};
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            return k;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            int base = (int) (first * k);
            for (int i = 0; i < count * k; i++) {
                dst.put(bytes[base + i]);
            }
        }
    }
}
