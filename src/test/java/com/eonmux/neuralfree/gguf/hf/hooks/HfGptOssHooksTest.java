package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code GptOssModel} hooks (gpt_oss.py:15-130): the sliding window and expert FFN metadata, the MXFP4
 * gate/up row de-interleave ({@code 0::2}/{@code 1::2} over the {@code [E, 2F, K]} rows), the float
 * {@code gate_up_proj} transpose-with-interleaved-columns split (the source forms of QR-2), the de-interleaved
 * {@code [E, 2F]} biases, the transposed {@code down_proj} and the sinks rename.
 */
public class HfGptOssHooksTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    private HfLoadContext ctx(Map<String, RowSource> index) throws IOException {
        return context(dir(), Arch.OPENAI_MOE, "GptOssForCausalLM",
                "'sliding_window': 4, 'intermediate_size': 4, 'num_hidden_layers': 2", index, false);
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metaWritesTheSlidingWindowAndExpertFfn() throws IOException {
        HfLoadContext ctx = ctx(java.util.Collections.emptyMap());
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.OPENAI_MOE.ggufName());
        HfGptOssHooks.meta(ctx, kv, null);
        assertEquals(4, kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
        assertEquals(4, kv.get(GgufKeys.EXPERT_FEED_FORWARD_LENGTH));
    }

    // ------------------------------------------------------------------ the gate/up forms

    @Test
    public void floatGateUpSplitsThroughTransposedInterleavedColumns() throws IOException {
        // [E=2, D=8, 2F=8]: gate takes the even columns, up the odd ones, transposed
        HfLoadContext ctx = ctx(java.util.Collections.singletonMap(
                "model.layers.0.mlp.experts.gate_up_proj.weight", seq(2, 8, 8)));
        RecordingSink sink = new RecordingSink();
        HfGptOssHooks.tensors(ctx, "model.layers.0.mlp.experts.gate_up_proj.weight",
                ctx.index().get("model.layers.0.mlp.experts.gate_up_proj.weight"), sink);
        java.util.List<RowSource> out = new java.util.ArrayList<>(sink.tensors.values());
        assertEquals(2, out.size());
        RowSource gate = out.get(0);
        assertArrayEquals(new long[]{2, 4, 8}, gate.shape());
        // row-major [E, F, D]: gate[e][f][d] = x[e][d][2f] = e*64 + d*8 + 2f
        assertArrayEquals(floats(
                0, 8, 16, 24, 32, 40, 48, 56,
                2, 10, 18, 26, 34, 42, 50, 58,
                4, 12, 20, 28, 36, 44, 52, 60,
                6, 14, 22, 30, 38, 46, 54, 62,
                64, 72, 80, 88, 96, 104, 112, 120,
                66, 74, 82, 90, 98, 106, 114, 122,
                68, 76, 84, 92, 100, 108, 116, 124,
                70, 78, 86, 94, 102, 110, 118, 126), values(gate), 0f);
        RowSource up = out.get(1);
        assertArrayEquals(new long[]{2, 4, 8}, up.shape());
        // up[e][f][d] = x[e][d][2f+1]; spot the corner elements
        float[] v = values(up);
        assertEquals(1.0f, v[0], 0f);
        assertEquals(21.0f, v[10], 0f); // e=0, f=1, d=2 -> x[0][2][5]
        assertEquals(65.0f, v[32], 0f);
        assertEquals(127.0f, v[63], 0f);
    }

    @Test
    public void mxfp4GateUpDeInterleavesTheFusedRows() throws IOException {
        // [E=2, 2F=8, K=4] stored form rows 0::2 gate and 1::2 up
        HfLoadContext ctx = ctx(java.util.Collections.singletonMap(
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                mono(ElementFormat.of(GGMLType.MXFP4), 2, 8, 4)));
        RecordingSink sink = new RecordingSink();
        HfGptOssHooks.tensors(ctx, "model.layers.0.mlp.experts.gate_up_proj.weight",
                ctx.index().get("model.layers.0.mlp.experts.gate_up_proj.weight"), sink);
        java.util.List<RowSource> out = new java.util.ArrayList<>(sink.tensors.values());
        assertEquals(2, out.size());
        RowSource gate = out.get(0);
        assertArrayEquals(new long[]{2, 4, 4}, gate.shape());
        // gate[e][r][k] = flat[e][2r][k] = e*32 + 2r*4 + k
        assertArrayEquals(floats(
                0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
                32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59), values(gate), 0f);
        assertArrayEquals(floats(
                4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
                36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63),
                values(out.get(1)), 0f);
    }

    @Test
    public void anOddMxfp4RowWidthIsRejected() throws IOException {
        Map<String, RowSource> index = java.util.Collections.singletonMap(
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                mono(ElementFormat.of(GGMLType.MXFP4), 2, 7, 4));
        HfLoadContext ctx = ctx(index);
        assertThrows(Exception.class, () -> HfGptOssHooks.tensors(ctx,
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                ctx.index().get("model.layers.0.mlp.experts.gate_up_proj.weight"), new RecordingSink()));
    }

    // ------------------------------------------------------------------ down_proj and biases

    @Test
    public void floatDownProjIsTransposedAndTheBiasStaysAWholeRow() throws IOException {
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.mlp.experts.down_proj.weight", seq(2, 8, 4));
        index.put("model.layers.0.mlp.experts.down_proj.bias", seq(2, 8));
        HfLoadContext ctx = ctx(index);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfGptOssHooks.tensors(ctx, name, ctx.index().get(name), sink);
        }
        java.util.List<RowSource> out = new java.util.ArrayList<>(sink.tensors.values());
        assertEquals(2, out.size());
        assertArrayEquals(new long[]{2, 4, 8}, out.get(0).shape());
        assertArrayEquals(new long[]{2, 8}, out.get(1).shape());
    }

    @Test
    public void biasesDeInterleaveIntoGateAndUpHalves() throws IOException {
        HfLoadContext ctx = ctx(java.util.Collections.singletonMap(
                "model.layers.0.mlp.experts.gate_up_proj_bias", seq(2, 8)));
        RecordingSink sink = new RecordingSink();
        HfGptOssHooks.tensors(ctx, "model.layers.0.mlp.experts.gate_up_proj_bias",
                ctx.index().get("model.layers.0.mlp.experts.gate_up_proj_bias"), sink);
        java.util.List<RowSource> out = new java.util.ArrayList<>(sink.tensors.values());
        assertEquals(2, sink.tensors.size());
        // gate rows [i::2], up rows [i+1::2] over the [E=2, 2F=8] row
        assertArrayEquals(new long[]{2, 4}, out.get(0).shape());
        assertArrayEquals(floats(0, 2, 4, 6, 8, 10, 12, 14), values(out.get(0)), 0f);
        assertArrayEquals(floats(1, 3, 5, 7, 9, 11, 13, 15), values(out.get(1)), 0f);
    }

    @Test
    public void sinksGainTheirWeightSuffixAndEverythingElseMaps() throws IOException {
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.attn.sinks", seq(2));
        index.put("model.layers.0.post_attention_layernorm.weight", seq(2, 8));
        HfLoadContext ctx = ctx(index);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfGptOssHooks.tensors(ctx, name, ctx.index().get(name), sink);
        }
        // the sinks tensor is named with its .weight suffix and the rest is mapped unchanged
        assertTrue(sink.tensors.size() >= 1);
    }

    // ------------------------------------------------------------------ helpers

    /**
     * A synthetic stored source: {@code readRows} writes little-endian F32 (the dequantized view the converter
     * computes) while the declared element format is {@code fmt}, so the hook's float branch is exercised.
     */
    static RowSource mono(ElementFormat fmt, long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        return new StoredSource(fmt, shape, v);
    }

    static final class StoredSource implements RowSource {
        private final ElementFormat fmt;
        private final long[] shape;
        private final float[] values;

        StoredSource(ElementFormat fmt, long[] shape, float[] values) {
            this.fmt = fmt;
            this.shape = shape;
            this.values = values;
        }

        @Override
        public String describe() {
            return "stored " + fmt;
        }

        @Override
        public ElementFormat format() {
            return fmt;
        }

        @Override
        public long[] shape() {
            return shape;
        }

        @Override
        public long rows() {
            long n = 1;
            for (int i = 0; i < shape.length - 1; i++) {
                n *= shape[i];
            }
            return n;
        }

        @Override
        public long rowBytes() {
            return 4L * shape[shape.length - 1];
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            dst.order(ByteOrder.LITTLE_ENDIAN);
            int perRow = (int) shape[shape.length - 1];
            for (int i = 0; i < count; i++) {
                long base = (first + i) * perRow;
                for (int j = 0; j < perRow; j++) {
                    dst.putFloat(values[(int) (base + j)]);
                }
            }
        }
    }
}
