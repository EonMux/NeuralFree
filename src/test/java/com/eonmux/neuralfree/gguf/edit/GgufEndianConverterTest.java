package com.eonmux.neuralfree.gguf.edit;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GgufEndianConverter} over a real GGUF fixture: little-endian to big-endian and back is the identity for
 * every metadata type and several tensor block layouts, the converted header carries the version in the target
 * order, in-place conversion replaces the file, an already-converted file is left untouched, and types whose block
 * layout is only defined little-endian are refused by tensor name.
 */
public class GgufEndianConverterTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void roundTripIsTheIdentity() throws Exception {
        Path file = fixture().writeTempFile();
        byte[] original = Files.readAllBytes(file);
        Path big = tmp.newFile("big.gguf").toPath();
        Path back = tmp.newFile("back.gguf").toPath();

        GgufEndianConverter.Info info = GgufEndianConverter.convert(file, GgufEndianConverter.Endian.BIG, big,
                true);
        assertEquals(GgufEndianConverter.Endian.BIG, info.fileEndian());
        assertFalse("the converted file differs from the input", java.util.Arrays.equals(original,
                Files.readAllBytes(big)));
        assertEquals(GgufEndianConverter.Endian.BIG, GgufEndianConverter.inspect(big).fileEndian());
        assertThrows("a big-endian file is not readable as little-endian", IOException.class,
                () -> ModelSources.parseGguf(big.toString()).close());

        GgufEndianConverter.convert(big, GgufEndianConverter.Endian.LITTLE, back, true);
        assertArrayEquals(original, Files.readAllBytes(back));
    }

    @Test
    public void theConvertedHeaderAndTensorDataAreByteSwapped() throws Exception {
        Path file = fixture().writeTempFile();
        byte[] original = Files.readAllBytes(file);
        GgufEndianConverter.Info info = GgufEndianConverter.inspect(file);
        Path big = tmp.newFile("big2.gguf").toPath();
        GgufEndianConverter.convert(file, GgufEndianConverter.Endian.BIG, big, true);
        byte[] converted = Files.readAllBytes(big);

        // magic keeps its bytes; the version is 3 stored big-endian
        assertEquals('G', converted[0]);
        assertEquals(0, converted[4]);
        assertEquals(0, converted[5]);
        assertEquals(0, converted[6]);
        assertEquals(3, converted[7]);

        // the first tensor is F32 at the start of the data section: every 4-byte element is reversed
        int start = (int) info.dataOffset();
        for (int i = 0; i < 16; i += 4) {
            for (int b = 0; b < 4; b++) {
                assertEquals("byte " + (i + b), original[start + i + b],
                        converted[start + i + 3 - b]);
            }
        }
        // the Q4_K tensor's d and dmin halves are the only multi-byte fields of its block
        int q4k = (int) (info.dataOffset() + q4kOffset());
        assertEquals(original[q4k], converted[q4k + 1]);
        assertEquals(original[q4k + 1], converted[q4k]);
        assertEquals(original[q4k + 2], converted[q4k + 3]);
        assertEquals(original[q4k + 3], converted[q4k + 2]);
        assertEquals(original[q4k + 4], converted[q4k + 4]);
    }

    @Test
    public void inPlaceConversionReplacesTheFileAndConvertsBack() throws Exception {
        Path file = fixture().writeTempFile();
        byte[] original = Files.readAllBytes(file);
        GgufEndianConverter.convert(file, GgufEndianConverter.Endian.BIG, file, false);
        assertThrows(IOException.class, () -> ModelSources.parseGguf(file.toString()).close());
        GgufEndianConverter.convert(file, GgufEndianConverter.Endian.LITTLE, file, false);
        assertArrayEquals(original, Files.readAllBytes(file));
    }

    @Test
    public void aFileAlreadyInTheTargetOrderIsNotWritten() throws Exception {
        Path file = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("should-not-exist.gguf");
        GgufEndianConverter.Info info = GgufEndianConverter.convert(file, GgufEndianConverter.Endian.LITTLE, out,
                false);
        assertEquals(GgufEndianConverter.Endian.LITTLE, info.fileEndian());
        assertFalse(Files.exists(out));
    }

    @Test
    public void checkConvertibleWritesNothing() throws Exception {
        Path file = fixture().writeTempFile();
        long size = Files.size(file);
        GgufEndianConverter.Info info = GgufEndianConverter.checkConvertible(file);
        assertEquals(GgufEndianConverter.Endian.LITTLE, info.fileEndian());
        assertEquals(size, Files.size(file));
    }

    @Test
    public void extensionTypesAreRefusedByTensorName() throws Exception {
        GgufTestWriter writer = new GgufTestWriter();
        writer.architecture("synthetic");
        writer.tensor("ext", GGMLType.F8_E4M3, new long[]{32}, new byte[32]);
        Path file = writer.writeTempFile();
        IOException e = assertThrows(IOException.class, () -> GgufEndianConverter.checkConvertible(file));
        assertTrue(e.getMessage(), e.getMessage().contains("'ext'"));
        assertTrue(e.getMessage(), e.getMessage().contains("little-endian only"));
    }

    @Test
    public void anExistingOutputIsReplacedOnlyWithOverwrite() throws Exception {
        Path file = fixture().writeTempFile();
        Path out = tmp.newFile("existing.gguf").toPath();
        IOException e = assertThrows(IOException.class,
                () -> GgufEndianConverter.convert(file, GgufEndianConverter.Endian.BIG, out, false));
        assertTrue(e.getMessage(), e.getMessage().contains("already exists"));
    }

    @Test
    public void aFileThatIsNotGgufIsRefused() throws Exception {
        Path file = tmp.newFile("not.gguf").toPath();
        Files.write(file, "not a gguf file".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> GgufEndianConverter.inspect(file));
        assertTrue(e.getMessage(), e.getMessage().contains("bad magic"));
    }

    @Test
    public void aModelWithoutTensorsRoundTrips() throws Exception {
        GgufTestWriter writer = new GgufTestWriter();
        writer.architecture("synthetic");
        writer.str("general.name", "metadata only");
        Path file = writer.writeTempFile();
        byte[] original = Files.readAllBytes(file);
        Path big = tmp.newFile("empty-big.gguf").toPath();
        Path back = tmp.newFile("empty-back.gguf").toPath();
        GgufEndianConverter.convert(file, GgufEndianConverter.Endian.BIG, big, true);
        assertEquals(original.length, Files.size(big));
        GgufEndianConverter.convert(big, GgufEndianConverter.Endian.LITTLE, back, true);
        assertArrayEquals(original, Files.readAllBytes(back));
    }

    /** Offset of the Q4_K tensor inside the data section: after the F32 and F16 tensors, padded to 32. */
    private static long q4kOffset() {
        return 32 + 32;
    }

    private static GgufTestWriter fixture() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.u8("k.u8", 255);
        w.i8("k.i8", -128);
        w.u16("k.u16", 65535);
        w.i16("k.i16", -32768);
        w.u32("k.u32", 4294967295L);
        w.i32("k.i32", -1);
        w.f32("k.f32", 1.5f);
        w.bool("k.bool", true);
        w.u64("k.u64", -1L);
        w.i64("k.i64", -7L);
        w.f64("k.f64", 2.5);
        w.str("k.str", "hello ä");
        w.strArray("k.strings", "a", "b", "c");
        w.u32Array("k.u32s", 1, 2, 3);
        w.f32Array("k.f32s", 1.5f, -2.5f);
        w.tensorF32("t.f32", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        w.tensorFromFloats("t.f16", GGMLType.F16, new long[]{16}, new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
        byte[] q4k = new byte[144];
        for (int i = 0; i < q4k.length; i++) {
            q4k[i] = (byte) (i * 3);
        }
        w.tensor("t.q4k", GGMLType.Q4_K, new long[]{256}, q4k);
        return w;
    }
}
