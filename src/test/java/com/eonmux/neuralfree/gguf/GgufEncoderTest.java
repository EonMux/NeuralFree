package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GgufEncoder} checks at the byte level. The encoder and {@link GgufTestWriter} write the same model
 * by construction, so the full file bytes of both must be identical; the value encodings are additionally
 * read back through {@link GGUFFileParser}, whose reader is independent of both. Rejected inputs carry
 * named errors, so the tests assert on messages, not on types alone.
 */
public class GgufEncoderTest {

    private static final String ARCH = "synthetic";

    /** Immutable name/type/value triple of one key/value call, shared by both build paths below. */
    private static final class KvName {
        final String key;
        final GGUFValueType type;
        /** Element type of an ARRAY value; null for every other type. */
        final GGUFValueType elementType;
        final Object value;

        KvName(String key, GGUFValueType type, Object value) {
            this(key, type, null, value);
        }

        KvName(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            this.key = key;
            this.type = type;
            this.elementType = elementType;
            this.value = value;
        }
    }

    /** One key/value pair of every gguf value type and a UTF-8 string in both. */
    private static List<KvName> kvs() {
        List<KvName> kv = new ArrayList<>();
        kv.add(new KvName(GgufTestWriter.ARCHITECTURE_KEY, GGUFValueType.STRING, ARCH));
        kv.add(new KvName("k.u8", GGUFValueType.UINT8, 255));
        kv.add(new KvName("k.i8", GGUFValueType.INT8, -128));
        kv.add(new KvName("k.u16", GGUFValueType.UINT16, 65535));
        kv.add(new KvName("k.i16", GGUFValueType.INT16, -32768));
        kv.add(new KvName("k.u32", GGUFValueType.UINT32, -1)); // UINT32 of 2^32 - 1, its bit pattern
        kv.add(new KvName("k.i32", GGUFValueType.INT32, -1));
        kv.add(new KvName("k.f32", GGUFValueType.FLOAT32, 1.5f));
        kv.add(new KvName("k.bool", GGUFValueType.BOOL, true));
        kv.add(new KvName("k.str", GGUFValueType.STRING, "hello \u00e4"));
        kv.add(new KvName("k.u64", GGUFValueType.UINT64, -1L));
        kv.add(new KvName("k.i64", GGUFValueType.INT64, -1L));
        kv.add(new KvName("k.f64", GGUFValueType.FLOAT64, 2.5));
        kv.add(new KvName("k.au8", GGUFValueType.ARRAY, GGUFValueType.UINT8, Arrays.asList(0, 1, 255)));
        kv.add(new KvName("k.ai8", GGUFValueType.ARRAY, GGUFValueType.INT8, Arrays.asList(-1, 0, 1)));
        kv.add(new KvName("k.au16", GGUFValueType.ARRAY, GGUFValueType.UINT16, Arrays.asList(0, 1, 65535)));
        kv.add(new KvName("k.ai16", GGUFValueType.ARRAY, GGUFValueType.INT16, Arrays.asList(-1, 0, 32767)));
        kv.add(new KvName("k.au32", GGUFValueType.ARRAY, GGUFValueType.UINT32, Arrays.asList(0, 1, -1))); // -1 = 2^32 - 1
        kv.add(new KvName("k.ai32", GGUFValueType.ARRAY, GGUFValueType.INT32, Arrays.asList(-1, 0, 2147483647)));
        kv.add(new KvName("k.af32", GGUFValueType.ARRAY, GGUFValueType.FLOAT32, Arrays.asList(-1.5f, 0.0f, 2.0f)));
        kv.add(new KvName("k.abool", GGUFValueType.ARRAY, GGUFValueType.BOOL, Arrays.asList(true, false, true)));
        kv.add(new KvName("k.astr", GGUFValueType.ARRAY, GGUFValueType.STRING, Arrays.asList("a", "b\u00e9", "c")));
        kv.add(new KvName("k.au64", GGUFValueType.ARRAY, GGUFValueType.UINT64, Arrays.asList(-1L, 0L, 1L)));
        kv.add(new KvName("k.ai64", GGUFValueType.ARRAY, GGUFValueType.INT64,
                Arrays.asList(Long.MIN_VALUE, 0L, Long.MAX_VALUE)));
        kv.add(new KvName("k.af64", GGUFValueType.ARRAY, GGUFValueType.FLOAT64, Arrays.asList(-1.25, 0.0, 2.5)));
        return kv;
    }

    /** The same pairs through the typed builders of {@link GgufTestWriter}. */
    private static GgufTestWriter fixtureWriter() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture(ARCH);
        w.u8("k.u8", 255);
        w.i8("k.i8", -128);
        w.u16("k.u16", 65535);
        w.i16("k.i16", -32768);
        w.u32("k.u32", 4294967295L);
        w.i32("k.i32", -1);
        w.f32("k.f32", 1.5f);
        w.bool("k.bool", true);
        w.str("k.str", "hello \u00e4");
        w.u64("k.u64", -1L);
        w.i64("k.i64", -1L);
        w.f64("k.f64", 2.5);
        w.u8Array("k.au8", 0, 1, 255);
        w.i8Array("k.ai8", -1, 0, 1);
        w.u16Array("k.au16", 0, 1, 65535);
        w.i16Array("k.ai16", -1, 0, 32767);
        w.u32Array("k.au32", 0, 1, 4294967295L);
        w.i32Array("k.ai32", -1, 0, 2147483647);
        w.f32Array("k.af32", -1.5f, 0.0f, 2.0f);
        w.boolArray("k.abool", true, false, true);
        w.strArray("k.astr", "a", "b\u00e9", "c");
        w.u64Array("k.au64", -1L, 0L, 1L);
        w.i64Array("k.ai64", Long.MIN_VALUE, 0L, Long.MAX_VALUE);
        w.f64Array("k.af64", -1.25, 0.0, 2.5);
        w.tensorF32("w", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        return w;
    }

    /** The file bytes of the same model, produced through {@link GgufEncoder} only. */
    private static byte[] encoderFile() throws IOException {
        List<KvName> kv = kvs();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        GgufEncoder enc = new GgufEncoder(Channels.newChannel(out), "test file");
        enc.writeHeader(Channels.newChannel(out), 1, kv.size());
        for (KvName pair : kv) {
            enc.writeKv(pair.key, pair.type, pair.elementType, pair.value);
        }
        int alignment = GgufTestWriter.DEFAULT_ALIGNMENT;
        enc.writeTensorInfo(new GGUFTensorInfo("w", new long[]{4}, GGMLType.F32, 0), 0);
        enc.align(alignment);
        byte[] data = new byte[16];
        ByteBuffer le = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < values.length; i++) {
            le.putInt(i * 4, Float.floatToRawIntBits(values[i]));
        }
        data = le.array();
        enc.writeData(data);
        enc.align(alignment);
        enc.flush();
        return out.toByteArray();
    }

    @Test
    public void everyValueTypeAndTensorMatchTheFixtureWriterByteForByte() throws IOException {
        byte[] fixture = fixtureWriter().toByteArray();
        byte[] encoded = encoderFile();
        assertEquals(fixture.length, encoded.length);
        assertArrayEquals(encoded, fixture);
    }

    @Test
    public void parserReadsEveryValueTypeBack() throws IOException {
        String path = Files.createTempFile("nf-enc-", ".gguf").toString();
        try {
            Files.write(Paths.get(path), encoderFile());
            try (GGUFFileParser parser = new GGUFFileParser(path)) {
                assertEquals(GGUFValueType.UINT8, parser.getValueType("k.u8"));
                assertEquals(255L, parser.getUInt32("k.u8").longValue());
                assertEquals(-128, parser.getInt("k.i8").intValue());
                assertEquals(65535L, parser.getUInt32("k.u16").longValue());
                assertEquals(-32768, parser.getInt("k.i16").intValue());
                assertEquals(4294967295L, parser.getUInt32("k.u32").longValue());
                assertEquals(-1, parser.getInt("k.i32").intValue());
                assertEquals(1.5f, parser.getFloat("k.f32"), 0f);
                assertEquals(Boolean.TRUE, parser.getBool("k.bool"));
                assertEquals("hello \u00e4", parser.getString("k.str"));
                assertEquals(-1L, parser.getLong("k.u64").longValue());
                assertEquals(-1L, parser.getLong("k.i64").longValue());
                assertEquals(2.5f, parser.getFloat("k.f64"), 0f);
                assertArrayEquals(new long[]{0, 1, 4294967295L}, parser.getUInt32Array("k.au32"));
                assertArrayEquals(new int[]{-1, 0, 2147483647}, parser.getInt32Array("k.ai32"));
                assertArrayEquals(new float[]{-1.5f, 0.0f, 2.0f}, parser.getFloatArray("k.af32"), 0f);
                assertArrayEquals(new boolean[]{true, false, true}, parser.getBoolArray("k.abool"));
                assertArrayEquals(new String[]{"a", "b\u00e9", "c"}, parser.getStringArray("k.astr"));
                assertArrayEquals(new long[]{-1L, 0L, 1L}, parser.getUInt64Array("k.au64"));
                assertEquals(GGMLType.F32, parser.getTensor("w").type());
                assertArrayEquals(new byte[]{
                        0, 0, (byte) 0x80, 0x3F, 0, 0, 0, 0x40, 0, 0, 0x40, 0x40, 0, 0, (byte) 0x80, 0x40
                }, parser.readTensorData(parser.getTensor("w")));
            }
        } finally {
            Files.deleteIfExists(Paths.get(path));
        }
    }

    @Test
    public void rejectsEmptyKeyNestedArraysAndWrongValueTypes() {
        assertThrows(IOException.class, () -> throwaway().writeKv("", GGUFValueType.UINT8, null, 1));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", null, null, 1));
        IOException missing = assertThrows(IOException.class, () -> throwaway()
                .writeKv("k", GGUFValueType.STRING, GGUFValueType.UINT8, "s"));
        assertTrue(missing.getMessage(), missing.getMessage().contains("'k'"));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY,
                GGUFValueType.ARRAY, new ArrayList<Object>()));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY, null, List.of()));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY,
                GGUFValueType.UINT8, "not a list"));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.UINT8, null, -1));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.UINT8, null, 256));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.UINT16, null, 65536));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.INT16, null,
                Short.MAX_VALUE + 1));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.BOOL, null, 1));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.STRING, null, 5));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.UINT64, null, 5));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.FLOAT32, null, 1.5));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.FLOAT64, null, 2.5f));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY,
                GGUFValueType.UINT8, Arrays.asList(0, 256)));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY,
                GGUFValueType.UINT32, Arrays.asList(0L, 4294967296L)));
        assertThrows(IOException.class, () -> throwaway().writeKv("k", GGUFValueType.ARRAY,
                GGUFValueType.STRING, Arrays.asList("a", null)));
        IOException range = assertThrows(IOException.class, () -> throwaway()
                .writeKv("k[1]", GGUFValueType.UINT8, null, -1));
        assertTrue(range.getMessage(), range.getMessage().contains("k[1]"));
    }

    @Test
    public void rejectsOversizedNamesAndBadTensorInfos() throws IOException {
        char[] nameChars = new char[128];
        Arrays.fill(nameChars, 'k');
        String longName = new String(nameChars);
        // the string/name size caps guard GGUF names; a long string VALUE below the 1 GiB cap is written
        char[] valueChars = new char[4096];
        Arrays.fill(valueChars, 'v');
        throwaway().writeKv("k", GGUFValueType.STRING, null, new String(valueChars));
        IOException oversizeKey = assertThrows(IOException.class, () -> throwaway()
                .writeKv(longName, GGUFValueType.UINT32, null, 1));
        assertTrue(oversizeKey.getMessage(), oversizeKey.getMessage().contains("128"));

        IOException longTensorName = assertThrows(IOException.class, () -> throwaway()
                .writeTensorInfo(new GGUFTensorInfo(longName, new long[]{1}, GGMLType.F32, 0), 0));
        assertTrue(longTensorName.getMessage(), longTensorName.getMessage().contains("more than the"));
        IOException rank = assertThrows(IOException.class, () -> throwaway().writeTensorInfo(new GGUFTensorInfo(
                "t", new long[]{1, 1, 1, 1, 1}, GGMLType.F32, 0), 0));
        assertTrue(rank.getMessage(), rank.getMessage().contains("dimensions"));
        IOException noType = assertThrows(IOException.class, () -> throwaway().writeTensorInfo(new GGUFTensorInfo(
                "t", new long[]{4}, null, 0), 0));
        assertTrue(noType.getMessage(), noType.getMessage().contains("no ggml type"));
        IOException negativeOffset = assertThrows(IOException.class, () -> throwaway().writeTensorInfo(
                new GGUFTensorInfo("t", new long[]{4}, GGMLType.F32, 0), -1));
        assertTrue(negativeOffset.getMessage(), negativeOffset.getMessage().contains("negative"));
    }

    @Test
    public void headerCountsAreValidated() throws IOException {
        GgufEncoder enc = throwaway();
        assertThrows(IOException.class, () -> enc.writeHeader(Channels.newChannel(new ByteArrayOutputStream()),
                -1, 0));
        assertThrows(IOException.class, () -> enc.writeHeader(Channels.newChannel(new ByteArrayOutputStream()),
                0, -5));
    }

    @Test
    public void alignPadsToTheNextMultipleAndValidatesItsArgument() throws IOException {
        GgufEncoder enc = throwaway();
        assertThrows(IOException.class, () -> enc.align(0));
        assertThrows(IOException.class, () -> enc.align(3));
        enc.align(32);
        assertEquals(0, enc.position());
        enc.writeData(new byte[1]);
        assertEquals(1, enc.position());
        enc.align(32);
        assertEquals(32, enc.position());
        enc.writeData(new byte[10]);
        enc.align(8);
        assertEquals(48, enc.position());
        enc.writeData(new byte[80]);
        enc.align(32);
        assertEquals(128, enc.position());
    }

    /** An encoder over a stream nobody reads, for the rejection calls. */
    private static GgufEncoder throwaway() {
        return new GgufEncoder(Channels.newChannel(new ByteArrayOutputStream()), "test file");
    }
}
