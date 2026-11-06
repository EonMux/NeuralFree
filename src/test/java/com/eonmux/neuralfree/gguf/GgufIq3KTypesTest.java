package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The GGUF surface of the IQ3_K / IQ3_KS types: a file Avalon wrote (block layouts of 110 / 104 bytes per
 * 256 elements at ggml type ids 43 / 44) parses, decodes and sizes its tensors, next to non-colliding
 * tensors and next to NF's own F8_E4M3FN reading of the shared id 43. A file laid out at 1 byte per element
 * for the unshared id 44 is rejected by name. The collision files are built byte by byte;
 * {@link GgufTestWriter} cannot produce them because it derives every tensor size from {@link GGMLType}.
 */
public class GgufIq3KTypesTest {

    // ------------------------------------------------------------------ Avalon-layout files

    @Test
    public void avalonIq3KFileParsesAndSizes() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("qwen3");
        w.tensorZeros("blk.0.attn_q", GGMLType.IQ3_K, 256L, 2L);
        w.tensorZeros("blk.0.attn_k", GGMLType.IQ3_KS, 256L, 2L);
        Path file = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            GGUFTensorInfo q = parser.getTensors().get(0);
            GGUFTensorInfo k = parser.getTensors().get(1);
            assertEquals(GGMLType.IQ3_K, q.type());
            assertEquals(GGMLType.IQ3_KS, k.type());
            assertEquals(2L * BlockLayout.IQ3_K_BYTES, q.sizeInBytes());
            assertEquals(2L * BlockLayout.IQ3_KS_BYTES, k.sizeInBytes());
            // the parser's decode path reaches the CPU decoders: an all-zero block decodes to 0
            // (d = 0 scales every grid value to zero)
            byte[] raw = parser.readTensorData(q);
            float[] decoded = GGUFFileParser.dequantize(raw, GGMLType.IQ3_K, 512);
            assertEquals(0f, decoded[0], 0f);
            assertEquals(0f, decoded[511], 0f);
        }
    }

    // ------------------------------------------------------------------ type-id 43/44 collision

    /**
     * Raw GGUF v3 bytes: {@code tensors} tensors of type id {@code typeId} with {@code elementsPerTensor}
     * elements each, stored at 1 byte per element (the mainline llama.cpp F8_E4M3 layout), 32-byte aligned.
     */
    private static byte[] f8LayoutFile(int typeId, int tensors, int elementsPerTensor) {
        byte[][] names = new byte[tensors][];
        int headerBytes = 4 + 4 + 8 + 8;
        for (int t = 0; t < tensors; t++) {
            names[t] = ("t." + t).getBytes();
            headerBytes += 8 + names[t].length + 4 + 16 + 4 + 8; // name, rank, two ne, type, offset
        }
        int dataOffset = (headerBytes + 31) & ~31;
        ByteBuffer b = ByteBuffer.allocate(dataOffset + tensors * elementsPerTensor)
                .order(ByteOrder.LITTLE_ENDIAN);
        b.put((byte) 'G').put((byte) 'G').put((byte) 'U').put((byte) 'F');
        b.putInt(3);                       // version
        b.putLong(tensors);                // tensor count
        b.putLong(0);                      // kv count
        for (int t = 0; t < tensors; t++) {
            b.putLong(names[t].length);
            b.put(names[t]);
            b.putInt(2);                   // rank
            b.putLong(256);                // ne[0]: a multiple of the 256-element block size
            b.putLong(elementsPerTensor / 256);
            b.putInt(typeId);
            b.putLong((long) t * elementsPerTensor);
        }
        b.position(dataOffset + tensors * elementsPerTensor);
        return Arrays.copyOf(b.array(), b.position());
    }

    private static Path write(byte[] bytes) throws IOException {
        Path file = java.nio.file.Files.createTempFile("iq3k-collision", ".gguf");
        java.nio.file.Files.write(file, bytes);
        file.toFile().deleteOnExit();
        return file;
    }

    /**
     * Id 43 is shared: a file whose id-43 tensors sit on a consistent 1-byte-per-element chain is NF's own
     * F8_E4M3FN reading (mainline llama.cpp's F8_E4M3), not an IQ3_K model, and parses as F8_E4M3FN.
     */
    @Test
    public void mainlineF8LayoutFileReadsAsF8E4M3FN() throws IOException {
        Path file = write(f8LayoutFile(43, 2, 1024));
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            GGUFTensorInfo first = parser.getTensors().get(0);
            GGUFTensorInfo second = parser.getTensors().get(1);
            assertEquals(GGMLType.F8_E4M3FN, first.type());
            assertEquals(GGMLType.F8_E4M3FN, second.type());
            assertEquals(1024L, first.sizeInBytes());
            assertEquals(1024L, second.offset());
            byte[] raw = parser.readTensorData(first);
            float[] decoded = GGUFFileParser.dequantize(raw, GGMLType.F8_E4M3FN, 1024);
            assertEquals(0f, decoded[0], 0f);
            assertEquals(0f, decoded[1023], 0f);
        }
    }

    /** The guard covers IQ3_KS (id 44), whose alternative reading exists only through id 43. */
    @Test
    public void mainlineF8LayoutFileWithType44IsRejectedByName() throws IOException {
        Path file = write(f8LayoutFile(44, 1, 1024));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("F8_E4M3"));
        assertTrue(e.getMessage(), e.getMessage().contains("44"));
    }

    /**
     * An Avalon file that also carries a tensor of a non-colliding type: the id-43 tensor is re-read as
     * IQ3_K from its block layout and the other tensor keeps its own type.
     */
    @Test
    public void avalonIq3KTensorIsReReadNextToOtherTypes() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("qwen3");
        w.tensorZeros("blk.0.attn_q", GGMLType.IQ3_K, 256L, 4L);
        w.tensorZeros("blk.0.attn_norm", GGMLType.F32, 256L);
        w.tensorZeros("blk.1.attn_k", GGMLType.IQ3_KS, 256L, 2L);
        Path file = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            assertEquals(GGMLType.IQ3_K, parser.getTensors().get(0).type());
            assertEquals(GGMLType.F32, parser.getTensors().get(1).type());
            assertEquals(GGMLType.IQ3_KS, parser.getTensors().get(2).type());
            assertEquals(4L * BlockLayout.IQ3_K_BYTES, parser.getTensors().get(0).sizeInBytes());
        }
    }

    /**
     * A file whose later tensors do not fit the 1-byte-per-element chain (their data is shorter) is laid out
     * for the IQ3_K reading: it passes the guard and fails only if the block layouts disagree.
     */
    @Test
    public void blockLayoutFileIsNotMistakenForTheMainlineReading() throws IOException {
        // tensor t.0 holds 256 elements = 110 IQ3_K bytes but only 256 bytes are reserved for it
        GgufTestWriter w = new GgufTestWriter().architecture("qwen3");
        w.tensorZeros("t.0", GGMLType.IQ3_K, 256L);
        w.tensorZeros("t.1", GGMLType.IQ3_K, 256L);
        Path file = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            assertEquals(2, parser.getTensors().size());
            assertEquals(GGMLType.IQ3_K, parser.getTensors().get(1).type());
        }
    }
}
