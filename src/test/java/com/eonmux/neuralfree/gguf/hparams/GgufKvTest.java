package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Typed metadata access through {@link GgufKv}: map-backed values, llama.cpp {@code get_key_or_arr}
 * semantics, and a parser-backed file covering unsigned UINT32 values, typed arrays, the alignment key and
 * ranged tensor reads.
 */
public class GgufKvTest {

    private static Map<String, Object> llama() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        return m;
    }

    @Test
    public void formatsArchScopedKeys() {
        GgufKv kv = GgufKv.ofMap(llama());
        assertEquals("llama", kv.archName());
        assertEquals("llama.attention.head_count", kv.key(GgufKeys.ATTENTION_HEAD_COUNT));
        assertEquals("general.name", kv.key(GgufKeys.GENERAL_NAME));
    }

    @Test
    public void archScopedKeyWithoutArchitectureThrows() {
        GgufKv kv = GgufKv.ofMap(new HashMap<String, Object>());
        assertThrows(GgufFormatException.class, () -> kv.key(GgufKeys.BLOCK_COUNT));
    }

    @Test
    public void typedScalarGetters() {
        Map<String, Object> m = llama();
        m.put("llama.block_count", 32);
        m.put("llama.rope.freq_base", 500000.0f);
        m.put("llama.attention.layer_norm_rms_epsilon", 1e-5);
        m.put("llama.attention.causal", false);
        m.put("general.name", "tiny");
        m.put("llama.expert_count", 16);
        GgufKv kv = GgufKv.ofMap(m);

        assertEquals(32, kv.u32(GgufKeys.BLOCK_COUNT, 0));
        assertEquals(32, kv.u32Required(GgufKeys.BLOCK_COUNT));
        assertEquals(500000.0f, kv.f32(GgufKeys.ROPE_FREQ_BASE, 0f), 0f);
        assertEquals(1e-5f, kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, 0f), 0f);
        assertEquals(16.0f, kv.f32(GgufKeys.EXPERT_COUNT, 0f), 0f);
        assertFalse(kv.bool(GgufKeys.ATTENTION_CAUSAL, true));
        assertEquals("tiny", kv.str(GgufKeys.GENERAL_NAME, null));
        assertEquals(7, kv.u32(GgufKeys.CONTEXT_LENGTH, 7));
        assertNull(kv.str(GgufKeys.GENERAL_AUTHOR, null));
    }

    @Test
    public void missingRequiredKeyNamesTheKey() {
        GgufKv kv = GgufKv.ofMap(llama());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> kv.u32Required(GgufKeys.CONTEXT_LENGTH));
        assertTrue(e.getMessage().contains("llama.context_length"));
    }

    @Test
    public void wrongTypeThrows() {
        Map<String, Object> m = llama();
        m.put("llama.block_count", "thirty-two");
        m.put("llama.context_length", -1);
        m.put("llama.feed_forward_length", Arrays.asList(1, 2));
        GgufKv kv = GgufKv.ofMap(m);
        assertThrows(GgufFormatException.class, () -> kv.u32(GgufKeys.BLOCK_COUNT, 0));
        assertThrows(GgufFormatException.class, () -> kv.u32(GgufKeys.CONTEXT_LENGTH, 0));
        assertThrows(GgufFormatException.class, () -> kv.u32(GgufKeys.FEED_FORWARD_LENGTH, 0));
    }

    @Test
    public void unsignedValuesAboveIntRange() {
        Map<String, Object> m = llama();
        m.put("llama.context_length", 3_000_000_000L);
        GgufKv kv = GgufKv.ofMap(m);
        assertEquals(3_000_000_000L, kv.u32Long(GgufKeys.CONTEXT_LENGTH, 0));
        assertThrows(GgufFormatException.class, () -> kv.u32(GgufKeys.CONTEXT_LENGTH, 0));

        m.put("llama.context_length", 5_000_000_000L);
        GgufKv tooLarge = GgufKv.ofMap(m);
        assertThrows(GgufFormatException.class, () -> tooLarge.u32Long(GgufKeys.CONTEXT_LENGTH, 0));
    }

    @Test
    public void keyOrArrBroadcastsScalar() {
        Map<String, Object> m = llama();
        m.put("llama.attention.head_count_kv", 8);
        GgufKv kv = GgufKv.ofMap(m);
        assertArrayEquals(new int[] {8, 8, 8}, kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 3, 0));
    }

    @Test
    public void keyOrArrReadsPerLayerArray() {
        Map<String, Object> m = llama();
        m.put("llama.attention.head_count_kv", new int[] {8, 4, 0, 2});
        GgufKv kv = GgufKv.ofMap(m);
        assertTrue(kv.isArray(GgufKeys.ATTENTION_HEAD_COUNT_KV));
        assertEquals(4, kv.arrayLength(GgufKeys.ATTENTION_HEAD_COUNT_KV));
        assertArrayEquals(new int[] {8, 4, 0, 2}, kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 4, 0));
    }

    @Test
    public void keyOrArrRejectsWrongLength() {
        Map<String, Object> m = llama();
        m.put("llama.attention.head_count_kv", new int[] {8, 4, 2});
        GgufKv kv = GgufKv.ofMap(m);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 4, 0));
        assertTrue(e.getMessage().contains("expected 4, got 3"));
    }

    @Test
    public void keyOrArrAbsentUsesFallback() {
        GgufKv kv = GgufKv.ofMap(llama());
        assertArrayEquals(new int[] {5, 5}, kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 2, 5));
        assertArrayEquals(new int[] {3, 1}, kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 2, new int[] {3, 1}));
        assertArrayEquals(new float[] {0.5f, 0.5f}, kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_EXP, 2, 0.5f), 0f);
    }

    @Test
    public void typedArrays() {
        Map<String, Object> m = llama();
        m.put("llama.attention.sliding_window_pattern", new boolean[] {true, false, true});
        m.put("llama.classifier.output_labels", new String[] {"neg", "pos"});
        m.put("llama.swiglu_clamp_exp", new float[] {7f, 0f});
        m.put("llama.ple.layer_multipliers", new long[] {20_000_000_000_000L, 3L});
        GgufKv kv = GgufKv.ofMap(m);
        assertArrayEquals(new boolean[] {true, false, true}, kv.boolArray(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
        assertArrayEquals(new String[] {"neg", "pos"}, kv.strArray(GgufKeys.CLASSIFIER_OUTPUT_LABELS));
        assertArrayEquals(new float[] {7f, 0f}, kv.f32Array(GgufKeys.SWIGLU_CLAMP_EXP), 0f);
        assertArrayEquals(new long[] {20_000_000_000_000L, 3L}, kv.u64Array(GgufKeys.PLE_LAYER_MULTIPLIERS));
        assertNull(kv.u32Array(GgufKeys.ATTENTION_HEAD_COUNT));
        assertThrows(GgufFormatException.class, () -> kv.strArray(GgufKeys.SWIGLU_CLAMP_EXP));
    }

    @Test
    public void mixedListIsRejected() {
        Map<String, Object> m = llama();
        m.put("llama.attention.head_count", Arrays.asList(1, "two"));
        assertThrows(IllegalArgumentException.class, () -> GgufKv.ofMap(m));
    }

    // ---- parser-backed source ----

    @Test
    public void parserReadsUnsignedUint32AndTypedArrays() throws IOException {
        File f = writeGguf();
        try {
            GGUFFileParser p = new GGUFFileParser(f.getPath());
            try {
                assertEquals(GGUFValueType.UINT32, p.getValueType("llama.context_length"));
                assertEquals(Long.valueOf(3_000_000_000L), p.getUInt32("llama.context_length"));
                assertEquals(64L, p.getAlignment());
                assertArrayEquals(new long[] {8L, 4L, 3_000_000_000L}, p.getUInt32Array("llama.attention.head_count_kv"));
                assertArrayEquals(new int[] {-1, 2}, p.getInt32Array("llama.deepstack_mapping"));
                assertArrayEquals(new float[] {1.5f, -2f}, p.getFloatArray("llama.swiglu_clamp_exp"), 0f);
                assertArrayEquals(new boolean[] {true, false}, p.getBoolArray("llama.attention.sliding_window_pattern"));
                assertArrayEquals(new String[] {"a", "b"}, p.getStringArray("llama.classifier.output_labels"));
                assertNull(p.getUInt32Array("llama.absent"));
                assertThrows(IllegalArgumentException.class, () -> p.getStringArray("llama.swiglu_clamp_exp"));

                GgufKv kv = GgufKv.of(p);
                assertEquals(3_000_000_000L, kv.u32Long(GgufKeys.CONTEXT_LENGTH, 0));
                assertThrows(GgufFormatException.class, () -> kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, 3, 0));
                assertArrayEquals(new boolean[] {true, false}, kv.boolArray(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
            } finally {
                p.close();
            }
        } finally {
            Files.deleteIfExists(f.toPath());
        }
    }

    @Test
    public void parserReadsTensorRanges() throws IOException {
        File f = writeGguf();
        try {
            GGUFFileParser p = new GGUFFileParser(f.getPath());
            try {
                assertEquals(0, p.getDataOffset() % 64);
                GGUFTensorInfo t = p.getTensor("t");
                byte[] all = p.readTensorData(t);
                assertEquals(16, all.length);
                ByteBuffer whole = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
                assertEquals(3.0f, whole.getFloat(8), 0f);

                byte[] mid = new byte[8];
                p.readTensorData(t, 4L, mid, 0, 8);
                ByteBuffer m = ByteBuffer.wrap(mid).order(ByteOrder.LITTLE_ENDIAN);
                assertEquals(2.0f, m.getFloat(0), 0f);
                assertEquals(3.0f, m.getFloat(4), 0f);

                ByteBuffer direct = ByteBuffer.allocateDirect(4).order(ByteOrder.LITTLE_ENDIAN);
                p.readTensorData(t, 12L, direct);
                assertEquals(4.0f, direct.getFloat(0), 0f);

                assertThrows(IllegalArgumentException.class, () -> p.readTensorData(t, 12L, new byte[8], 0, 8));
            } finally {
                p.close();
            }
        } finally {
            Files.deleteIfExists(f.toPath());
        }
    }

    /** A GGUF v3 file with typed metadata, alignment 64 and one F32 tensor "t" = [1, 2, 3, 4]. */
    private static File writeGguf() throws IOException {
        Out o = new Out();
        o.bytes("GGUF".getBytes(StandardCharsets.US_ASCII));
        o.u32(3);
        o.u64(1);
        o.u64(8);
        o.str("general.architecture"); o.u32(8); o.str("llama");
        o.str("general.alignment"); o.u32(4); o.u32(64);
        o.str("llama.context_length"); o.u32(4); o.u32((int) 3_000_000_000L);
        o.str("llama.attention.head_count_kv"); o.u32(9); o.u32(4); o.u64(3); o.u32(8); o.u32(4); o.u32((int) 3_000_000_000L);
        o.str("llama.deepstack_mapping"); o.u32(9); o.u32(5); o.u64(2); o.u32(-1); o.u32(2);
        o.str("llama.swiglu_clamp_exp"); o.u32(9); o.u32(6); o.u64(2); o.f32(1.5f); o.f32(-2f);
        o.str("llama.attention.sliding_window_pattern"); o.u32(9); o.u32(7); o.u64(2); o.u8(1); o.u8(0);
        o.str("llama.classifier.output_labels"); o.u32(9); o.u32(8); o.u64(2); o.str("a"); o.str("b");
        o.str("t"); o.u32(1); o.u64(4); o.u32(0); o.u64(0);
        while (o.size() % 64 != 0) {
            o.u8(0);
        }
        o.f32(1f); o.f32(2f); o.f32(3f); o.f32(4f);
        File f = File.createTempFile("nf-ggufkv-", ".gguf");
        Files.write(f.toPath(), o.toByteArray());
        return f;
    }

    private static final class Out extends ByteArrayOutputStream {
        void bytes(byte[] b) {
            write(b, 0, b.length);
        }

        void u8(int v) {
            write(v);
        }

        void u32(int v) {
            bytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
        }

        void u64(long v) {
            bytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array());
        }

        void f32(float v) {
            bytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(v).array());
        }

        void str(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            u64(b.length);
            bytes(b);
        }
    }
}
