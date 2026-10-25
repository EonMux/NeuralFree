package com.eonmux.neuralfree.io.safetensors;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.Defect;
import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.ramp;
import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.writeMalformed;
import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.writeRaw;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SafetensorsHeader} and {@link SafetensorsFile}: a valid header read back, and one malformed file per
 * validation rule, each refused with the {@code "Invalid safetensors file '<path>': "} prefix and a message naming
 * the defect.
 */
public class SafetensorsHeaderTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path file(String name) {
        return tmp.getRoot().toPath().resolve(name);
    }

    private static SafetensorsHeader read(Path p) throws IOException {
        try (FileChannel ch = FileChannel.open(p, StandardOpenOption.READ)) {
            return SafetensorsHeader.read(ch, p.toString());
        }
    }

    /** Reads {@code p}, expects a refusal and checks its prefix and that it mentions {@code fragment}. */
    private static void assertInvalid(Path p, String fragment) {
        IOException e = assertThrows(IOException.class, () -> read(p));
        String prefix = "Invalid safetensors file '" + p + "': ";
        assertTrue(e.getMessage(), e.getMessage().startsWith(prefix));
        assertTrue(e.getMessage(), e.getMessage().contains(fragment));
    }

    private Path raw(String header, int dataBytes) throws IOException {
        Path p = file("raw.safetensors");
        writeRaw(p, header, new byte[dataBytes]);
        return p;
    }

    private static String tensor(String name, String dtype, String shape, long begin, long end) {
        return "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape + ",\"data_offsets\":[" + begin
                + "," + end + "]}";
    }

    // ------------------------------------------------------------------ valid files

    @Test
    public void readsTensorsMetadataAndLayout() throws IOException {
        Path p = file("ok.safetensors");
        SafetensorsWriter.builder()
                .add("w", StDType.BF16, new long[]{2, 3}, ramp(0, 12))
                .add("bias", StDType.F32, new long[]{3}, ramp(100, 12))
                .add("scalar", StDType.F32, new long[0], ramp(50, 4))
                .add("empty", StDType.F16, new long[]{0, 4}, new byte[0])
                .metadata("format", "pt")
                .build()
                .write(p);
        SafetensorsHeader h = read(p);
        assertEquals(0, h.headerLength() % 8);
        assertEquals(8 + h.headerLength(), h.dataStart());
        assertEquals(28, h.dataLength());
        assertEquals(Files.size(p), h.dataStart() + h.dataLength());
        assertEquals("pt", h.metadata().get("format"));
        assertEquals(Arrays.asList("bias", "empty", "scalar", "w"), names(h));
        StTensor w = h.tensor("w");
        assertEquals(StDType.BF16, w.dtype());
        assertArrayEquals(new long[]{2, 3}, w.shape());
        assertArrayEquals(new long[]{3, 2}, w.ne());
        assertEquals(6, w.numel());
        assertEquals(12, w.byteLength());
        assertEquals(1, h.tensor("scalar").numel());
        assertEquals(0, h.tensor("empty").byteLength());
        assertNull(h.tensor("missing"));
    }

    @Test
    public void fileReadsTensorBytesAndDescribesItsSegment() throws IOException {
        Path p = file("data.safetensors");
        byte[] w = ramp(7, 24);
        SafetensorsWriter.builder().add("w", StDType.F32, new long[]{2, 3}, w).build().write(p);
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            StTensor t = f.tensor("w");
            ByteBuffer all = f.readAll(t);
            byte[] got = new byte[all.remaining()];
            all.get(got);
            assertArrayEquals(w, got);
            ByteBuffer part = ByteBuffer.allocate(8);
            f.read(t, 12, part);
            assertEquals(8, part.position());
            assertArrayEquals(Arrays.copyOfRange(w, 12, 20), part.array());
            assertThrows(IllegalArgumentException.class, () -> f.read(t, 20, ByteBuffer.allocate(8)));
            assertThrows(IllegalArgumentException.class, () -> f.read(t, -1, ByteBuffer.allocate(1)));
            StTensor foreign = new StTensor("w", StDType.F32, new long[]{6}, 0, 24);
            assertThrows(IllegalArgumentException.class, () -> f.read(foreign, 0, ByteBuffer.allocate(4)));
            DataSegment s = f.segment(128);
            assertEquals(new DataSegment(p.toString(), f.dataStart(), 128, 24), s);
        }
    }

    @Test
    public void acceptsExtensionDtypesAndZeroLengthTensorsAnywhere() throws IOException {
        String h = "{" + tensor("z0", "F32", "[0]", 0, 0) + "," + tensor("a", "F8_E4M3FNUZ", "[4]", 0, 4) + ","
                + tensor("z1", "I64", "[3,0]", 4, 4) + "," + tensor("b", "F8_E5M2FNUZ", "[2]", 4, 6) + ","
                + tensor("c", "F4", "[2,2]", 6, 8) + "}";
        SafetensorsHeader header = read(raw(h, 8));
        assertEquals(5, header.size());
        assertEquals(StDType.F8_E4M3FNUZ, header.tensor("a").dtype());
        assertEquals(StDType.F4, header.tensor("c").dtype());
        assertEquals(8, header.dataLength());
    }

    @Test
    public void dtypeTableMatchesTheFormat() {
        assertEquals(22, StDType.values().length);
        assertSame(StDType.F8_E8M0, StDType.fromName("F8_E8M0"));
        assertNull(StDType.fromName("F12"));
        assertNull(StDType.fromName("f32"));
        assertEquals(4, StDType.F4.bits());
        assertEquals(6, StDType.F6_E3M2.bits());
        assertEquals(64, StDType.C64.bits());
        assertSame(GGMLType.BF16, StDType.BF16.directType());
        assertNull(StDType.U8.directType());
        assertNull(StDType.F8_E4M3.directType());
        assertSame(StDType.I64, StDType.forDirectType(GGMLType.I64));
        assertNull(StDType.forDirectType(GGMLType.Q8_0));
    }

    // ------------------------------------------------------------------ rule 1: the length field

    @Test
    public void rejectsFileShorterThanTheLengthField() throws IOException {
        Path p = file("short.safetensors");
        Files.write(p, new byte[]{2, 0, 0});
        assertInvalid(p, "fewer than the 8-byte header length field");
    }

    @Test
    public void rejectsHeaderLengthBelowMinimum() throws IOException {
        Path p = file("tiny.safetensors");
        writeRaw(p, 1, "{".getBytes(StandardCharsets.UTF_8), new byte[0]);
        assertInvalid(p, "below the minimum of 2 bytes");
    }

    @Test
    public void rejectsHeaderLengthAboveLimit() throws IOException {
        Path p = file("huge.safetensors");
        writeRaw(p, 100_000_001L, "{}".getBytes(StandardCharsets.UTF_8), new byte[0]);
        assertInvalid(p, "exceeds the limit of 100000000 bytes");
        writeRaw(p, -1L, "{}".getBytes(StandardCharsets.UTF_8), new byte[0]);
        assertInvalid(p, "header length 18446744073709551615 exceeds the limit");
    }

    @Test
    public void rejectsHeaderLengthPastTheFile() throws IOException {
        Path p = file("bad-length.safetensors");
        writeMalformed(p, Defect.BAD_LENGTH);
        assertInvalid(p, "exceeds the file size");
    }

    // ------------------------------------------------------------------ rule 2: UTF-8 and '{'

    @Test
    public void rejectsInvalidUtf8() throws IOException {
        Path p = file("utf8.safetensors");
        writeRaw(p, new byte[]{'{', '"', (byte) 0xC3, '"', ':', '1', '}', ' '}, new byte[0]);
        assertInvalid(p, "not valid UTF-8");
    }

    @Test
    public void rejectsHeaderNotStartingWithBrace() throws IOException {
        assertInvalid(raw("  [1,2,3]", 0), "does not start with '{'");
        assertInvalid(raw("        ", 0), "does not start with '{'");
    }

    // ------------------------------------------------------------------ rule 3: strict JSON

    @Test
    public void rejectsMalformedJson() throws IOException {
        assertInvalid(raw("{\"a\":{\"dtype\":", 0), "not valid JSON");
        assertInvalid(raw("{} {}", 0), "not valid JSON");
    }

    @Test
    public void rejectsDuplicateKeys() throws IOException {
        Path p = file("dup.safetensors");
        writeMalformed(p, Defect.DUPLICATE_KEY);
        assertInvalid(p, "duplicate key 'a'");
    }

    // ------------------------------------------------------------------ rule 4: metadata

    @Test
    public void rejectsMetadataThatIsNotAStringMap() throws IOException {
        assertInvalid(raw("{\"__metadata__\":[\"pt\"]}", 0), "__metadata__ is an array, not an object");
        assertInvalid(raw("{\"__metadata__\":{\"format\":1}}", 0), "__metadata__ key 'format' is an integer");
    }

    // ------------------------------------------------------------------ rule 5: tensor fields

    @Test
    public void rejectsTensorEntriesWithOtherFields() throws IOException {
        assertInvalid(raw("{\"a\":[1]}", 0), "tensor 'a' is an array, not an object");
        assertInvalid(raw("{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", 4), "expected exactly [dtype, shape,"
                + " data_offsets]");
        assertInvalid(raw("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"x\":0}}", 4),
                "tensor 'a' has the fields");
    }

    @Test
    public void rejectsUnknownDtype() throws IOException {
        Path p = file("dtype.safetensors");
        writeMalformed(p, Defect.BAD_DTYPE);
        assertInvalid(p, "tensor 'a' has the unknown dtype 'F12'");
        assertInvalid(raw("{" + tensor("a", "f32", "[1]", 0, 4) + "}", 4), "unknown dtype 'f32'");
    }

    // ------------------------------------------------------------------ rule 6: shapes

    @Test
    public void rejectsBadShapes() throws IOException {
        assertInvalid(raw("{" + tensor("a", "U8", "[1,1,1,1,1,1,1,1,1]", 0, 1) + "}", 1),
                "rank 9 exceeds the limit of 8");
        assertInvalid(raw("{" + tensor("a", "U8", "[-1]", 0, 0) + "}", 0), "shape dimension 0 is -1");
        assertInvalid(raw("{" + tensor("a", "U8", "[2.0]", 0, 2) + "}", 2), "shape dimension 0 is 2.0");
        assertInvalid(raw("{" + tensor("a", "U8", "[9007199254740992]", 0, 0) + "}", 0),
                "not an integer in [0, 2^53)");
        assertInvalid(raw("{" + tensor("a", "U8", "\"4\"", 0, 4) + "}", 4), "shape is a string, not an array");
    }

    @Test
    public void rejectsOverflowingElementCounts() throws IOException {
        assertInvalid(raw("{" + tensor("a", "F32", "[4503599627370496,4503599627370496]", 0, 0) + "}", 0),
                "overflows a 64-bit integer");
    }

    @Test
    public void rejectsSubByteTensorsOfPartialBytes() throws IOException {
        assertInvalid(raw("{" + tensor("a", "F4", "[3]", 0, 2) + "}", 2), "12 bits, not a whole number of bytes");
    }

    // ------------------------------------------------------------------ rule 7: data offsets

    @Test
    public void rejectsBadDataOffsets() throws IOException {
        assertInvalid(raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0]}}", 1),
                "not an array of two integers");
        assertInvalid(raw("{" + tensor("a", "U8", "[0]", 4, 3) + "}", 4), "is not a range [begin, end]");
        assertInvalid(raw("{" + tensor("a", "U8", "[1]", -1, 0) + "}", 1), "is not a range [begin, end]");
        assertInvalid(raw("{" + tensor("a", "F32", "[2]", 0, 4) + "}", 4), "span 4 bytes, but [2] of F32 needs 8");
    }

    // ------------------------------------------------------------------ rule 8: tiling and file size

    @Test
    public void rejectsOverlappingTensors() throws IOException {
        Path p = file("overlap.safetensors");
        writeMalformed(p, Defect.OVERLAP);
        assertInvalid(p, "overlaps tensor");
    }

    @Test
    public void rejectsGapsBetweenTensors() throws IOException {
        Path p = file("gap.safetensors");
        writeMalformed(p, Defect.GAP);
        assertInvalid(p, "leaves the bytes [8, 9) unused after tensor 'a'");
        assertInvalid(raw("{" + tensor("a", "U8", "[2]", 2, 4) + "}", 4), "at the start of the data section");
    }

    @Test
    public void rejectsTrailingBytes() throws IOException {
        Path p = file("trailing.safetensors");
        writeMalformed(p, Defect.TRAILING_BYTES);
        assertInvalid(p, "3 trailing bytes follow the data section");
    }

    @Test
    public void rejectsDataPastTheEndOfTheFile() throws IOException {
        assertInvalid(raw("{" + tensor("a", "U8", "[8]", 0, 8) + "}", 4), "runs past the end of the file");
    }

    private static List<String> names(SafetensorsHeader h) {
        List<String> out = new ArrayList<>();
        for (StTensor t : h.tensors()) {
            out.add(t.name());
        }
        return out;
    }
}
