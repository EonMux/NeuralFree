package com.eonmux.neuralfree.demo.tools;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The base-logits file of llama.cpp (tools/perplexity/perplexity.cpp): the byte layout of the header and the
 * stored rows, the round trip through {@link LogitsFile.Out} and {@link LogitsFile.In}, and the refusals for
 * a wrong magic, a truncated header and truncated rows.
 */
public class LogitsFileTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theFileLayoutIsLlamaCppByteForByte() throws IOException {
        Path path = tmp.newFile("base.bin").toPath();
        int nVocab = 5;
        int nv = PerplexityMath.rowLength(nVocab);
        float[] logits = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        short[] rows = new short[3 * nv];
        try (LogitsFile.Out out = new LogitsFile.Out(path, 4)) {
            out.writeRun(nVocab, 1, new int[] {9, 8, 7, 6}, 4);
            PerplexityMath.quantLogSoftmaxRow(logits, 2, rows, 0);
            PerplexityMath.quantLogSoftmaxRow(logits, 1, rows, nv);
            PerplexityMath.quantLogSoftmaxRow(logits, 0, rows, 2 * nv);
            out.writeRows(rows, 3, nv);
        }
        byte[] bytes = Files.readAllBytes(path);
        assertEquals(8 + 3 * 4 + 4 * 4 + 3 * nv * 2, bytes.length);
        assertEquals("_logits_", new String(bytes, 0, 8, java.nio.charset.StandardCharsets.US_ASCII));
        ByteBuffer header = ByteBuffer.wrap(bytes, 8, 20).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(4, header.getInt());
        assertEquals(nVocab, header.getInt());
        assertEquals(1, header.getInt());
        assertEquals(9, header.getInt());
        assertEquals(8, header.getInt());

        try (LogitsFile.In in = new LogitsFile.In(path)) {
            assertEquals(4, in.nCtx());
            assertEquals(nVocab, in.nVocab());
            assertEquals(1, in.nChunk());
            assertArrayEquals(new int[] {9, 8, 7, 6}, in.tokens());
            short[] read = new short[3 * nv];
            in.readRows(read);
            assertTrue(Arrays.equals(rows, read));
        }
    }

    @Test
    public void rowsStreamChunkByChunk() throws IOException {
        Path path = tmp.newFile("base.bin").toPath();
        int nv = PerplexityMath.rowLength(3);
        try (LogitsFile.Out out = new LogitsFile.Out(path, 2)) {
            out.writeRun(3, 2, new int[] {1, 2, 3, 4}, 4);
            out.writeRows(new short[nv], 1, nv);
            out.writeRows(new short[nv], 1, nv);
        }
        try (LogitsFile.In in = new LogitsFile.In(path)) {
            short[] rows = new short[nv];
            in.readRows(rows);
            in.readRows(rows);
            assertEquals("every row of both chunks was consumed", 0, in.tokens().length - 4);
        }
    }

    @Test
    public void aWrongMagicIsRefusedByName() throws IOException {
        Path path = tmp.newFile("bad.bin").toPath();
        Files.write(path, "_wrong!12345678".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        IOException e = assertThrows(IOException.class, () -> new LogitsFile.In(path));
        assertTrue(e.getMessage(), e.getMessage().contains("does not look like a file containing"
                + " log-probabilities"));
    }

    @Test
    public void aTruncatedHeaderIsRefusedByName() throws IOException {
        Path path = tmp.newFile("short.bin").toPath();
        Files.write(path, "_logits_\u0001\u0002".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        IOException e = assertThrows(IOException.class, () -> new LogitsFile.In(path));
        assertTrue(e.getMessage(), e.getMessage().contains("ends inside"));
    }

    @Test
    public void truncatedRowsAreRefusedByName() throws IOException {
        Path path = tmp.newFile("cut.bin").toPath();
        try (LogitsFile.Out out = new LogitsFile.Out(path, 2)) {
            out.writeRun(3, 1, new int[] {1, 2}, 2);
        }
        try (LogitsFile.In in = new LogitsFile.In(path)) {
            short[] rows = new short[PerplexityMath.rowLength(3)];
            IOException e = assertThrows(IOException.class, () -> in.readRows(rows));
            assertTrue(e.getMessage(), e.getMessage().contains("ends inside the stored rows"));
        }
    }

    @Test
    public void anImpossibleHeaderIsRefusedByName() throws IOException {
        Path path = tmp.newFile("huge.bin").toPath();
        ByteBuffer header = ByteBuffer.allocate(8 + 12).order(ByteOrder.LITTLE_ENDIAN);
        header.put("_logits_".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        header.putInt(-1);
        header.putInt(-1);
        header.putInt(-1);
        Files.write(path, header.array());
        IOException e = assertThrows(IOException.class, () -> new LogitsFile.In(path));
        assertTrue(e.getMessage(), e.getMessage().contains("impossible header"));
    }
}
