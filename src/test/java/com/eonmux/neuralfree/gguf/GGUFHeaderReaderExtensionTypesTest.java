package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Locale;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code neuralfree.extension_types} gate of {@link GGUFHeaderReader}: tensors of the NeuralFree extension types
 * (ids 1000-1005) load only from a file that sets the key to BOOL true, ggml types never need it, and extension-range
 * ids without a {@link GGMLType} keep the unknown-type rejection.
 */
public class GGUFHeaderReaderExtensionTypesTest {

    private static final String WEIGHT = "ext_probe.weight";

    private static final GGMLType[] EXTENSION_TYPES = {
        GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.MXFP8, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB,
    };

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyExtensionTypeLoadsWithTheMarker() throws IOException {
        for (GGMLType type : EXTENSION_TYPES) {
            assertTrue(type + " is an extension type", type.isExtension());
            String path = write(withTensor(type).bool(GgufKeys.NF_EXTENSION_TYPES, true));
            try (GGUFFileParser parser = new GGUFFileParser(path)) {
                GGUFTensorInfo info = parser.getTensor(WEIGHT);
                assertEquals(type, info.type());
                assertEquals(type.getBytesForElements(2L * type.getBlockSize() * 3), info.sizeInBytes());
            }
        }
    }

    @Test
    public void everyExtensionTypeIsRejectedWithoutTheMarker() throws IOException {
        for (GGMLType type : EXTENSION_TYPES) {
            String path = write(withTensor(type));
            IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
            assertEquals("Invalid GGUF file '" + path + "': tensor '" + WEIGHT + "' uses NeuralFree extension type "
                    + type + " but the file lacks neuralfree.extension_types", e.getMessage());
        }
    }

    @Test
    public void markerSetToFalseIsRejected() throws IOException {
        String path = write(withTensor(GGMLType.NF4).bool(GgufKeys.NF_EXTENSION_TYPES, false));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("uses NeuralFree extension type NF4"));
        assertTrue(e.getMessage(), e.getMessage().contains("not BOOL true"));
    }

    @Test
    public void markerOfAnotherTypeIsRejected() throws IOException {
        String path = write(withTensor(GGMLType.MXFP8).u8(GgufKeys.NF_EXTENSION_TYPES, 1));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("sets neuralfree.extension_types to UINT8 1"));
        assertTrue(e.getMessage(), e.getMessage().contains("not BOOL true"));

        String text = write(withTensor(GGMLType.MXFP8).str(GgufKeys.NF_EXTENSION_TYPES, "true"));
        IOException s = assertThrows(IOException.class, () -> new GGUFFileParser(text));
        assertTrue(s.getMessage(),
                s.getMessage().endsWith("sets neuralfree.extension_types to a STRING, not BOOL true"));
    }

    @Test
    public void ggmlTypesNeedNoMarker() throws IOException {
        String path = write(withTensor(GGMLType.Q8_0));
        try (GGUFFileParser parser = new GGUFFileParser(path)) {
            assertEquals(GGMLType.Q8_0, parser.getTensor(WEIGHT).type());
            assertFalse(parser.getMetadata().containsKey(GgufKeys.NF_EXTENSION_TYPES));
        }
    }

    @Test
    public void markedFileMayMixGgmlAndExtensionTypes() throws IOException {
        GgufTestWriter w = withTensor(GGMLType.I8_B128).bool(GgufKeys.NF_EXTENSION_TYPES, true)
                .tensorZeros("plain.weight", GGMLType.Q4_0, 64, 2);
        try (GGUFFileParser parser = new GGUFFileParser(write(w))) {
            assertEquals(GGMLType.I8_B128, parser.getTensor(WEIGHT).type());
            assertEquals(GGMLType.Q4_0, parser.getTensor("plain.weight").type());
            assertEquals(Boolean.TRUE, parser.getMetadata().get(GgufKeys.NF_EXTENSION_TYPES));
        }
    }

    /** Id 1006 lies in the extension range but names no type, so it fails as any unknown ggml type does. */
    @Test
    public void unassignedExtensionIdKeepsTheUnknownTypeRejection() throws IOException {
        for (boolean marked : new boolean[]{true, false}) {
            GgufTestWriter w = withTensor(GGMLType.F8_E4M3);
            if (marked) {
                w.bool(GgufKeys.NF_EXTENSION_TYPES, true);
            }
            String path = write(withRawTypeId(w.toByteArray(), 1006));
            IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
            assertTrue(e.getMessage(), e.getMessage().contains("unknown or was removed from ggml"));
            assertFalse(e.getMessage(), e.getMessage().contains("extension type"));
        }
    }

    /** A raw id 1000 behind the writer's back reads as {@link GGMLType#F8_E4M3} and is gated like it. */
    @Test
    public void rawExtensionIdIsGatedByTheMarker() throws IOException {
        byte[] raw = withRawTypeId(withTensor(GGMLType.Q8_0).toByteArray(), GGMLType.F8_E4M3.getId());
        String path = write(raw);
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("uses NeuralFree extension type F8_E4M3 but the file lacks"
                + " neuralfree.extension_types"));
    }

    /** Each file of a split model is gated on its own key/value section. */
    @Test
    public void everySplitHoldingExtensionTensorsCarriesTheMarker() throws IOException {
        File marked = tmp.newFolder();
        String first = writeSplits(marked, splitWithPlainTensor(), splitWithExtensionTensor(true));
        try (GGUFFileParser parser = new GGUFFileParser(first)) {
            assertEquals(2, parser.getShards().size());
            assertEquals(GGMLType.NF4, parser.getTensor(WEIGHT).type());
        }

        File unmarked = tmp.newFolder();
        String head = writeSplits(unmarked, splitWithPlainTensor(), splitWithExtensionTensor(false));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(head));
        assertTrue(e.getMessage(), e.getMessage().contains("Invalid GGUF file '" + splitName(unmarked, 1)
                + "': tensor '" + WEIGHT + "' uses NeuralFree extension type NF4 but the file lacks "
                + "neuralfree.extension_types"));
    }

    // ------------------------------------------------------------------ helpers

    /** Split 0 of a two-split model: the marker and one Q4_0 tensor. */
    private static GgufTestWriter splitWithPlainTensor() {
        return new GgufTestWriter().architecture("llama")
                .u16(GgufKeys.SPLIT_NO, 0)
                .u16(GgufKeys.SPLIT_COUNT, 2)
                .i32(GgufKeys.SPLIT_TENSORS_COUNT, 2)
                .bool(GgufKeys.NF_EXTENSION_TYPES, true)
                .tensorZeros("plain.weight", GGMLType.Q4_0, 64, 2);
    }

    /** Split 1 of a two-split model: one NF4 tensor, with or without its own marker. */
    private static GgufTestWriter splitWithExtensionTensor(boolean marked) {
        GgufTestWriter w = new GgufTestWriter()
                .u16(GgufKeys.SPLIT_NO, 1)
                .u16(GgufKeys.SPLIT_COUNT, 2)
                .i32(GgufKeys.SPLIT_TENSORS_COUNT, 2)
                .tensorZeros(WEIGHT, GGMLType.NF4, 2L * GGMLType.NF4.getBlockSize(), 3);
        return marked ? w.bool(GgufKeys.NF_EXTENSION_TYPES, true) : w;
    }

    private static String splitName(File dir, int no) {
        return new File(dir, String.format(Locale.ROOT, "model-%05d-of-00002.gguf", no + 1)).getPath();
    }

    /** Writes both splits into {@code dir} and returns the path of the first. */
    private static String writeSplits(File dir, GgufTestWriter first, GgufTestWriter second) throws IOException {
        first.writeTo(new File(splitName(dir, 0)).toPath());
        second.writeTo(new File(splitName(dir, 1)).toPath());
        return splitName(dir, 0);
    }

    /** A file with one {@code [2 blocks, 3 rows]} zero tensor of {@code type} and no marker. */
    private static GgufTestWriter withTensor(GGMLType type) {
        return new GgufTestWriter().architecture("llama")
                .tensorZeros(WEIGHT, type, 2L * type.getBlockSize(), 3);
    }

    /** Overwrites the type id of the {@link #WEIGHT} tensor info, which follows its name, rank and extents. */
    private static byte[] withRawTypeId(byte[] file, int typeId) {
        byte[] name = WEIGHT.getBytes(StandardCharsets.UTF_8);
        int at = indexOf(file, name);
        ByteBuffer b = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        int rankAt = at + name.length;
        int nDims = b.getInt(rankAt);
        b.putInt(rankAt + 4 + 8 * nDims, typeId);
        return file;
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        int found = -1;
        for (int i = 0; i + needle.length <= haystack.length; i++) {
            int j = 0;
            while (j < needle.length && haystack[i + j] == needle[j]) {
                j++;
            }
            if (j == needle.length) {
                if (found >= 0) {
                    throw new IllegalStateException(WEIGHT + " occurs more than once in the file");
                }
                found = i;
            }
        }
        if (found < 0) {
            throw new IllegalStateException(WEIGHT + " is not in the file");
        }
        return found;
    }

    private String write(GgufTestWriter w) throws IOException {
        return write(w.toByteArray());
    }

    private String write(byte[] bytes) throws IOException {
        File f = tmp.newFile();
        Files.write(f.toPath(), bytes);
        return f.getPath();
    }
}
