package com.eonmux.neuralfree.gguf;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Header checks of {@link GGUFFileParser} against llama.cpp ggml/src/gguf.cpp: version, entry counts, empty and
 * duplicate keys, nested arrays, string and array sizes against the bytes left in the file, and tensor name length.
 * Each malformed file is built byte by byte, since the test writer only produces well-formed files.
 */
public class GGUFFileParserHeaderTest {

    private static final int UINT8 = 0;
    private static final int UINT32 = 4;
    private static final int STRING = 8;
    private static final int ARRAY = 9;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** Little-endian GGUF byte builder. */
    private static final class Raw {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        static Raw header(int version, long nTensors, long nKv) {
            Raw r = new Raw();
            r.out.write('G');
            r.out.write('G');
            r.out.write('U');
            r.out.write('F');
            return r.u32(version).u64(nTensors).u64(nKv);
        }

        Raw u8(int v) {
            out.write(v & 0xFF);
            return this;
        }

        Raw u32(int v) {
            for (int i = 0; i < 4; i++) {
                out.write((v >>> (8 * i)) & 0xFF);
            }
            return this;
        }

        Raw u64(long v) {
            for (int i = 0; i < 8; i++) {
                out.write((int) (v >>> (8 * i)) & 0xFF);
            }
            return this;
        }

        Raw str(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            u64(b.length);
            out.write(b, 0, b.length);
            return this;
        }

        Raw kvU8(String key, int v) {
            return str(key).u32(UINT8).u8(v);
        }

        byte[] bytes() {
            return out.toByteArray();
        }
    }

    private String write(Raw raw) throws IOException {
        File f = tmp.newFile();
        Files.write(f.toPath(), raw.bytes());
        return f.getPath();
    }

    private void assertRejected(Raw raw, String fragment) throws IOException {
        String path = write(raw);
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue("message '" + e.getMessage() + "' should contain '" + fragment + "'",
                e.getMessage().contains(fragment));
    }

    @Test
    public void wellFormedRawHeaderParses() throws IOException {
        String path = write(Raw.header(3, 0, 2).kvU8("a", 1).kvU8("b", 2));
        GGUFFileParser p = new GGUFFileParser(path);
        try {
            assertEquals(3, p.getVersion());
            assertEquals(2L, p.getKVCount());
            assertEquals(Integer.valueOf(1), p.getInt("a"));
            assertEquals(Integer.valueOf(2), p.getInt("b"));
            assertEquals(1, p.getShards().size());
            assertFalse(p.isMultiShard());
        } finally {
            p.close();
        }
    }

    @Test
    public void versionTwoParses() throws IOException {
        GGUFFileParser p = new GGUFFileParser(write(Raw.header(2, 0, 1).kvU8("a", 1)));
        try {
            assertEquals(2, p.getVersion());
        } finally {
            p.close();
        }
    }

    @Test
    public void rejectsVersionZero() throws IOException {
        assertRejected(Raw.header(0, 0, 0), "bad GGUF version");
    }

    @Test
    public void rejectsVersionOne() throws IOException {
        assertRejected(Raw.header(1, 0, 0), "GGUFv1 is no longer supported");
    }

    @Test
    public void rejectsByteSwappedVersion() throws IOException {
        assertRejected(Raw.header(0x03000000, 0, 0), "endianness");
    }

    @Test
    public void rejectsNewerVersion() throws IOException {
        assertRejected(Raw.header(4, 0, 0), "version 4");
    }

    @Test
    public void rejectsTensorCountAboveSignedRange() throws IOException {
        assertRejected(Raw.header(3, -1L, 0), "number of tensors is 18446744073709551615");
    }

    @Test
    public void rejectsTensorCountLargerThanTheFile() throws IOException {
        assertRejected(Raw.header(3, 1000, 0), "number of tensors is 1000");
    }

    @Test
    public void rejectsKvCountLargerThanTheFile() throws IOException {
        assertRejected(Raw.header(3, 0, 1L << 40).kvU8("a", 1), "number of key/value pairs");
    }

    @Test
    public void rejectsDuplicateKey() throws IOException {
        assertRejected(Raw.header(3, 0, 2).kvU8("a", 1).kvU8("a", 2), "duplicate key 'a'");
    }

    @Test
    public void rejectsEmptyKey() throws IOException {
        assertRejected(Raw.header(3, 0, 1).kvU8("", 1), "key 0 is empty");
    }

    @Test
    public void rejectsNestedArray() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("nested").u32(ARRAY).u32(ARRAY).u64(1).u32(UINT8).u64(1).u8(0);
        assertRejected(raw, "array of arrays");
    }

    @Test
    public void rejectsStringLongerThanTheMaximum() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("s").u32(STRING).u64(1L << 40);
        assertRejected(raw, "exceeds the maximum");
    }

    @Test
    public void rejectsStringLongerThanTheRestOfTheFile() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("s").u32(STRING).u64(1000).u8('x');
        assertRejected(raw, "bytes left in the file");
    }

    @Test
    public void rejectsArrayLongerThanTheMaximum() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("arr").u32(ARRAY).u32(UINT8).u64((1L << 30) + 1);
        assertRejected(raw, "more than the maximum");
    }

    @Test
    public void rejectsArrayLongerThanTheRestOfTheFile() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("arr").u32(ARRAY).u32(UINT32).u64(1L << 29).u32(7);
        assertRejected(raw, "bytes are left in the file");
    }

    @Test
    public void rejectsStringArrayLongerThanTheRestOfTheFile() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("arr").u32(ARRAY).u32(STRING).u64(100).str("only one");
        assertRejected(raw, "bytes are left in the file");
    }

    @Test
    public void rejectsUnknownValueType() throws IOException {
        assertRejected(Raw.header(3, 0, 1).str("k").u32(99).u8(0), "invalid GGUF type 99");
    }

    @Test
    public void rejectsTensorNameOfSixtyFourBytes() throws IOException {
        StringBuilder name = new StringBuilder();
        for (int i = 0; i < GGUFHeaderReader.MAX_NAME; i++) {
            name.append('t');
        }
        Raw raw = Raw.header(3, 1, 0).str(name.toString()).u32(1).u64(32).u32(GGMLType.F32.getId()).u64(0);
        for (int i = 0; i < 32 * 4 + 64; i++) {
            raw.u8(0);
        }
        assertRejected(raw, "more than the 63 allowed");
    }

    @Test
    public void rejectsTruncatedHeader() throws IOException {
        Raw raw = Raw.header(3, 0, 1).str("a").u32(UINT32).u8(1);
        String path = write(raw);
        assertThrows(IOException.class, () -> new GGUFFileParser(path));
    }

    @Test
    public void rejectsBadMagic() throws IOException {
        File f = tmp.newFile();
        Files.write(f.toPath(), new byte[]{'G', 'G', 'M', 'L', 3, 0, 0, 0});
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(f.getPath()));
        assertTrue(e.getMessage(), e.getMessage().contains("bad magic"));
    }
}
