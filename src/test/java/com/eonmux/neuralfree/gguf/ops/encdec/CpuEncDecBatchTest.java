package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Multi-sequence decoder batches of the CPU encoder-decoder backend over the synthetic T5 (relative-position bias,
 * cross-attention in every decoder layer): every sequence encodes its own source into its own cross caches
 * ({@link SequenceEncoder#encodeSequence}), and an interleaved batch of two sequences' decoder rows, then a decode
 * round, equal the same sources and decoder tokens run one at a time on the contiguous path ({@code prefill(source,
 * 0)} and {@code forwardRows}). Re-encoding one sequence leaves the other's encoding alone; a non-causal batch keeps
 * its sequences apart; rows of a sequence without an encoding and oversize sources refuse by name.
 */
public class CpuEncDecBatchTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;
    private static final int START = SyntheticModels.T5_DECODER_START;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The decoder rows of one sequence: the start token, then {@code tail}. */
    private static int[] decoderRows(int[] tail) {
        int[] rows = new int[tail.length + 1];
        rows[0] = START;
        System.arraycopy(tail, 0, rows, 1, tail.length);
        return rows;
    }

    /** The contiguous logits of every decoder row of {@code rows} over {@code source}. */
    private static float[][] contiguous(ForwardBackend f, int[] source, int[] rows) {
        float[][] out = new float[rows.length][];
        out[0] = f.prefill(source, 0).clone();
        float[][] rest = f.forwardRows(Arrays.copyOfRange(rows, 1, rows.length), 1, null);
        System.arraycopy(rest, 0, out, 1, rest.length);
        return out;
    }

    /** Rows {@code [from, to)} of two sequences interleaved into one batch, every row an output row. */
    private static ForwardBatch interleave(int[] a, int[] b, int from, int to, boolean nonCausal) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2 * (to - from));
        for (int i = from; i < to; i++) {
            if (i < a.length) {
                batch.add(a[i], i, SEQ_A, true);
            }
            if (i < b.length) {
                batch.add(b[i], i, SEQ_B, true);
            }
        }
        if (nonCausal) {
            batch.nonCausal();
        }
        return batch.build();
    }

    private static void assertRows(String what, float[][] expected, int from, float[][] actual, int at, int n) {
        for (int i = 0; i < n; i++) {
            assertArrayEquals(what + " row " + (from + i), expected[from + i], actual[at + 2 * i], TOL);
        }
    }

    @Test
    public void twoSequencesOverTheirOwnSourcesMatchTheSingleSequenceRuns() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = tokens(1, 5);
        int[] srcB = tokens(2, 9);
        int[] rowsA = decoderRows(tokens(3, 4));
        int[] rowsB = decoderRows(tokens(4, 4));
        float[][] refA;
        float[][] refB;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false))) {
            assertNull("a contiguous memory keeps one encoding", one.forwardBackend().sequenceEncoder());
            assertEquals(1, one.forwardBackend().maxSequences());
            assertNotNull(one.forwardBackend().batchRefusal());
            refA = contiguous(one.forwardBackend(), srcA, rowsA);
            refB = contiguous(one.forwardBackend(), srcB, rowsB);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            ForwardBackend f = two.forwardBackend();
            assertNull(f.batchRefusal());
            assertEquals(2, f.maxSequences());
            SequenceEncoder encoder = f.sequenceEncoder();
            assertNotNull(encoder);
            assertEquals(START, encoder.decoderStartToken());
            encoder.encodeSequence(SEQ_A, srcA);
            encoder.encodeSequence(SEQ_B, srcB);
            float[][] prompt = f.forwardBatch(interleave(rowsA, rowsB, 0, 4, false));
            assertEquals(8, prompt.length);
            assertRows("sequence A over its own source", refA, 0, prompt, 0, 4);
            assertRows("sequence B over its own source", refB, 0, prompt, 1, 4);
            float[][] round = f.forwardBatch(interleave(rowsA, rowsB, 4, 5, false));
            assertArrayEquals("the decode round continues sequence A", refA[4], round[0], TOL);
            assertArrayEquals("the decode round continues sequence B", refB[4], round[1], TOL);
        }
    }

    /**
     * A sequence that restarts over a new source re-encodes into its own cross caches only: the other sequence keeps
     * decoding over its encoding, and the restarted one equals a fresh run over the new source.
     */
    @Test
    public void reEncodingOneSequenceLeavesTheOtherSequencesEncodingAlone() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = tokens(5, 6);
        int[] srcB = tokens(6, 7);
        int[] srcC = tokens(7, 4);
        int[] rowsA = decoderRows(tokens(8, 2));
        int[] rowsB = decoderRows(tokens(9, 4));
        int[] rowsC = decoderRows(tokens(10, 2));
        float[][] refB;
        float[][] refC;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false))) {
            refB = contiguous(one.forwardBackend(), srcB, rowsB);
            refC = contiguous(one.forwardBackend(), srcC, rowsC);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            ForwardBackend f = two.forwardBackend();
            SequenceEncoder encoder = f.sequenceEncoder();
            encoder.encodeSequence(SEQ_A, srcA);
            encoder.encodeSequence(SEQ_B, srcB);
            f.forwardBatch(interleave(rowsA, rowsB, 0, 2, false));
            HeapModelMemory mem = (HeapModelMemory) f.memory();
            mem.sequences().remove(SEQ_A, 0, -1);
            encoder.encodeSequence(SEQ_A, srcC);
            ForwardBatch.Builder batch = ForwardBatch.builder(6);
            for (int i = 0; i < 3; i++) {
                batch.add(rowsC[i], i, SEQ_A, true);
                batch.add(rowsB[2 + i], 2 + i, SEQ_B, true);
            }
            float[][] out = f.forwardBatch(batch.build());
            for (int i = 0; i < 3; i++) {
                assertArrayEquals("the restarted sequence runs over its new source, row " + i, refC[i], out[2 * i],
                        TOL);
                assertArrayEquals("sequence B keeps its own encoding, row " + (2 + i), refB[2 + i], out[2 * i + 1],
                        TOL);
            }
        }
    }

    /** Rows {@code [from, to)} of one sequence as a batch of its own, every row an output row. */
    private static ForwardBatch single(int[] rows, int seq, int from, int to) {
        ForwardBatch.Builder batch = ForwardBatch.builder(to - from);
        for (int i = from; i < to; i++) {
            batch.add(rows[i], i, seq, true);
        }
        return batch.build();
    }

    @Test
    public void aNonCausalBatchKeepsTheSequencesApart() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = tokens(11, 5);
        int[] srcB = tokens(12, 6);
        int[] rowsA = decoderRows(tokens(13, 3));
        int[] rowsB = decoderRows(tokens(14, 3));
        float[] aloneA;
        float[] aloneB;
        float[] causalA;
        try (GGUFModel one = new GGUFModel(model.toString(), config(true))) {
            ForwardBackend f = one.forwardBackend();
            f.sequenceEncoder().encodeSequence(SEQ_A, srcA);
            f.sequenceEncoder().encodeSequence(SEQ_B, srcB);
            aloneA = f.forwardBatch(nonCausal(single(rowsA, SEQ_A, 0, 4)))[0];
            f.memory().sequences().remove(SEQ_A, 0, -1);
            aloneB = f.forwardBatch(nonCausal(single(rowsB, SEQ_B, 0, 4)))[0];
            f.memory().sequences().remove(SEQ_B, 0, -1);
            causalA = f.forwardBatch(single(rowsA, SEQ_A, 0, 4))[0];
        }
        assertFalse("a non-causal step lets the start row see the later decoder rows",
                Arrays.equals(aloneA, causalA));
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            ForwardBackend f = two.forwardBackend();
            f.sequenceEncoder().encodeSequence(SEQ_A, srcA);
            f.sequenceEncoder().encodeSequence(SEQ_B, srcB);
            float[][] out = f.forwardBatch(interleave(rowsA, rowsB, 0, 4, true));
            assertArrayEquals("sequence A's non-causal rows match its own batch", aloneA, out[0], TOL);
            assertArrayEquals("sequence B's non-causal rows match its own batch", aloneB, out[1], TOL);
        }
    }

    private static ForwardBatch nonCausal(ForwardBatch b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(b.nTokens());
        for (int i = 0; i < b.nTokens(); i++) {
            batch.add(b.tokens()[i], b.positions()[i], b.seqIds()[i], b.output(i));
        }
        return batch.nonCausal().build();
    }

    @Test
    public void rowsWithoutAnEncodingAndOversizeSourcesRefuseByName() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            ForwardBackend f = two.forwardBackend();
            SequenceEncoder encoder = f.sequenceEncoder();
            encoder.encodeSequence(SEQ_A, tokens(15, 4));
            IllegalStateException unencoded = assertThrows(IllegalStateException.class,
                    () -> f.forwardBatch(interleave(decoderRows(new int[0]), decoderRows(new int[0]), 0, 1, false)));
            assertTrue(unencoded.getMessage(), unencoded.getMessage().contains("holds no encoded source"));
            assertEquals("a refused batch reserves nothing", -1, f.memory().sequences().posMax(SEQ_A));
            int limit = encoder.maxSourceTokens();
            com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan dec0 = f.memory().plan().layer(0);
            assertEquals("one cross stream per sequence", 2, dec0.crossStreams());
            assertEquals("the source limit is the ubatch within one stream's rows",
                    Math.min(f.memory().plan().ubatch(), dec0.crossKv().rows()), limit);
            assertThrows(ContextFullException.class, () -> encoder.encodeSequence(SEQ_B, tokens(16, limit + 1)));
            assertThrows(IndexOutOfBoundsException.class, () -> encoder.encodeSequence(2, tokens(17, 3)));
            assertThrows(IllegalArgumentException.class, () -> encoder.encodeSequence(SEQ_B, new int[0]));
        }
    }
}
