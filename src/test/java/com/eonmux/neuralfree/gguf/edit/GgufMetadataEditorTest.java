package com.eonmux.neuralfree.gguf.edit;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GgufMetadataEditor} over a real GGUF fixture: every scalar type is changed in place, the stored type and
 * byte width are kept, nothing else in the file moves, and the missing-key, non-scalar and out-of-range refusals
 * name what is wrong.
 */
public class GgufMetadataEditorTest {

    @Test
    public void aUint32ValueChangesInPlaceAndNothingElseMoves() throws Exception {
        GgufTestWriter writer = new GgufTestWriter();
        writer.architecture("synthetic");
        writer.u32("tokenizer.ggml.bos_token_id", 1);
        writer.str("general.name", "name");
        writer.tensorF32("w.a", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        Path file = writer.writeTempFile();
        byte[] before = Files.readAllBytes(file);

        long valueOffset;
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufMetadataEditor.Change change = GgufMetadataEditor.plan(src, "tokenizer.ggml.bos_token_id", "2");
            assertEquals(Integer.valueOf(1), change.oldValue());
            assertEquals(Integer.valueOf(2), change.newValue());
            assertEquals(4, change.width());
            assertFalse(change.isNoOp());
            valueOffset = change.valueOffset();
            GgufMetadataEditor.apply(src, change);
        }
        byte[] after = Files.readAllBytes(file);
        assertEquals(before.length, after.length);
        assertEquals((byte) 2, after[(int) valueOffset]);
        assertEquals((byte) 0, after[(int) valueOffset + 1]);
        byte[] expected = Arrays.copyOf(before, before.length);
        expected[(int) valueOffset] = 2;
        expected[(int) valueOffset + 1] = 0;
        expected[(int) valueOffset + 2] = 0;
        expected[(int) valueOffset + 3] = 0;
        assertArrayEquals(expected, after);
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertEquals(Integer.valueOf(2), src.getMetadata().get("tokenizer.ggml.bos_token_id"));
            assertEquals("name", src.getString("general.name"));
            assertArrayEquals(writer.tensorData("w.a"), src.readTensorData(src.getTensor("w.a")));
        }
    }

    @Test
    public void everyScalarTypeRoundTripsThroughTheEditor() throws Exception {
        GgufTestWriter writer = new GgufTestWriter();
        writer.architecture("synthetic");
        writer.u8("k.u8", 1);
        writer.i8("k.i8", 1);
        writer.u16("k.u16", 1);
        writer.i16("k.i16", 1);
        writer.u32("k.u32", 1);
        writer.i32("k.i32", 1);
        writer.f32("k.f32", 1.0f);
        writer.bool("k.bool", false);
        writer.u64("k.u64", 1);
        writer.i64("k.i64", 1);
        writer.f64("k.f64", 1.0);
        Path file = writer.writeTempFile();
        String[][] edits = {
                {"k.u8", "255"}, {"k.i8", "-128"}, {"k.u16", "65535"}, {"k.i16", "-32768"},
                {"k.u32", "4294967295"}, {"k.i32", "-2147483648"}, {"k.f32", "-1.5"},
                {"k.bool", "true"}, {"k.u64", "18446744073709551615"}, {"k.i64", "-9223372036854775808"},
                {"k.f64", "2.5"},
        };
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            for (String[] edit : edits) {
                GgufMetadataEditor.apply(src, GgufMetadataEditor.plan(src, edit[0], edit[1]));
            }
        }
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertEquals(Integer.valueOf(255), src.getMetadata().get("k.u8"));
            assertEquals(Integer.valueOf(-128), src.getMetadata().get("k.i8"));
            assertEquals(Integer.valueOf(65535), src.getMetadata().get("k.u16"));
            assertEquals(Integer.valueOf(-32768), src.getMetadata().get("k.i16"));
            assertEquals(Integer.valueOf(-1), src.getMetadata().get("k.u32"));
            assertEquals(Integer.valueOf(-2147483648), src.getMetadata().get("k.i32"));
            assertEquals(Float.valueOf(-1.5f), src.getMetadata().get("k.f32"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("k.bool"));
            assertEquals(Long.valueOf(-1L), src.getMetadata().get("k.u64"));
            assertEquals(Long.valueOf(Long.MIN_VALUE), src.getMetadata().get("k.i64"));
            assertEquals(Double.valueOf(2.5), src.getMetadata().get("k.f64"));
        }
    }

    @Test
    public void theSameValueIsANoOp() throws Exception {
        Path file = scalarFixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufMetadataEditor.Change change = GgufMetadataEditor.plan(src, "k.u32", "1");
            assertTrue(change.isNoOp());
            assertEquals("1", change.oldText());
            assertEquals("1", change.newText());
        }
    }

    @Test
    public void unsignedValuesPrintUnsigned() throws Exception {
        Path file = scalarFixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufMetadataEditor.Change change = GgufMetadataEditor.plan(src, "k.u32", "4294967295");
            assertEquals("1", change.oldText());
            assertEquals("4294967295", change.newText());
        }
    }

    @Test
    public void aMissingKeyIsRefusedByName() throws Exception {
        Path file = scalarFixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            IOException e = assertThrows(IOException.class,
                    () -> GgufMetadataEditor.plan(src, "general.nope", "1"));
            assertTrue(e.getMessage(), e.getMessage().contains("'general.nope'"));
            assertTrue(e.getMessage(), e.getMessage().contains("is not in"));
        }
    }

    @Test
    public void stringsAndArraysAreRefusedWithAPointerToNewMetadata() throws Exception {
        GgufTestWriter writer = scalarFixture();
        writer.u32Array("k.array", 1, 2);
        Path file = writer.writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            IOException text = assertThrows(IOException.class,
                    () -> GgufMetadataEditor.plan(src, "general.architecture", "other"));
            assertTrue(text.getMessage(), text.getMessage().contains("STRING"));
            assertTrue(text.getMessage(), text.getMessage().contains("nftool new-metadata"));
            IOException array = assertThrows(IOException.class,
                    () -> GgufMetadataEditor.plan(src, "k.array", "1"));
            assertTrue(array.getMessage(), array.getMessage().contains("ARRAY"));
        }
    }

    @Test
    public void generalAlignmentIsRefusedBecauseTheLayoutDependsOnIt() throws Exception {
        Path file = scalarFixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            IOException e = assertThrows(IOException.class,
                    () -> GgufMetadataEditor.plan(src, "general.alignment", "64"));
            assertTrue(e.getMessage(), e.getMessage().contains("general.alignment"));
            assertTrue(e.getMessage(), e.getMessage().contains("nftool new-metadata"));
        }
    }

    @Test
    public void outOfRangeAndMalformedValuesAreRefused() throws Exception {
        Path file = scalarFixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertRefused(src, "k.u32", "-1", "outside");
            assertRefused(src, "k.u32", "4294967296", "outside");
            assertRefused(src, "k.u32", "x", "an integer in");
            assertRefused(src, "k.bool", "yes", "true or false");
            assertRefused(src, "k.f32", "NaN", "finite");
        }
    }

    private static void assertRefused(GGUFFileParser src, String key, String value, String message) {
        IOException e = assertThrows(IOException.class, () -> GgufMetadataEditor.plan(src, key, value));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
    }

    private static GgufTestWriter scalarFixture() {
        GgufTestWriter writer = new GgufTestWriter();
        writer.architecture("synthetic");
        writer.u32("general.alignment", 32);
        writer.u32("k.u32", 1);
        writer.bool("k.bool", false);
        writer.f32("k.f32", 1.0f);
        writer.tensorF32("w.a", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        return writer;
    }
}
