package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Evaluations of the pooling, scoring and classifier paths on a backend whose memory is paged
 * ({@code neuralfree.parallel > 1} or {@code neuralfree.kv.paged=true}), where the contiguous single-sequence calls
 * ({@code prefill}, {@code hiddenRows} with memory) do not address the page tables. One prompt runs as a
 * single-sequence {@link ForwardBatch} in sequence 0 from position 0 ({@link ForwardBackend#forwardBatch},
 * {@link ForwardBackend#hiddenBatch}); several prompts run as multi-sequence batches, one sequence each
 * ({@link #hiddenSequences}, llama.cpp's parallel-sequence embedding batch). The pages free afterwards, also when the
 * call fails. The caller empties the memory first (the generation state is discarded), as the contiguous paths do.
 * A backend planned without memory serves the memoryless forms instead ({@link #memorylessRows},
 * {@link #memorylessNextLogits}): the whole media prompt runs as one non-causal batch, as llama.cpp's cache-less
 * decision graphs read it.
 */
public final class PagedEvaluations {

    private PagedEvaluations() {
    }

    /** True when {@code backend} holds a paged memory, the case these evaluations serve. */
    public static boolean paged(ForwardBackend backend) {
        ModelMemory memory = Objects.requireNonNull(backend, "backend").memory();
        return memory != null && memory.pagedSequences() != null;
    }

    /**
     * The logits after the last token of {@code tokens} (a copy).
     *
     * @throws IllegalStateException when the backend's memory is not paged
     */
    public static float[] nextLogits(ForwardBackend backend, int[] tokens) {
        ForwardBatch batch = prompt(tokens, false);
        return inSequenceZero(backend, () -> backend.forwardBatch(batch)[0].clone());
    }

    /**
     * The logits after the last row of the media {@code prompt}: the chunks run as consecutive batches like
     * {@link #hiddenRows(ForwardBackend, MediaPrompt)}, and the last chunk's last row produces the logits.
     *
     * @throws IllegalStateException when the backend's memory is not paged
     */
    public static float[] nextLogits(ForwardBackend backend, MediaPrompt prompt) {
        if (!paged(backend)) {
            throw new IllegalStateException(Objects.requireNonNull(backend, "backend") + " holds no paged memory");
        }
        Objects.requireNonNull(prompt, "prompt");
        SequenceMemory sequences = backend.memory().sequences();
        float[] out = null;
        try {
            int pos = 0;
            java.util.List<MediaChunk> chunks = prompt.chunks();
            for (int c = 0; c < chunks.size(); c++) {
                MediaChunk chunk = chunks.get(c);
                if (c == chunks.size() - 1) {
                    out = backend.forwardBatch(chunkBatch(chunk, pos, chunk.nTokens() - 1))[0].clone();
                } else {
                    backend.hiddenBatch(chunkBatch(chunk, pos));
                }
                pos += chunk.nPositions();
            }
        } catch (RuntimeException e) {
            try {
                sequences.remove(0, 0, -1);
            } catch (RuntimeException cleanup) {
                e.addSuppressed(cleanup);
            }
            throw e;
        }
        sequences.remove(0, 0, -1);
        if (out == null) {
            throw new IllegalStateException("the media prompt holds no chunk to read");
        }
        return out;
    }

    /**
     * The token ids of the media {@code prompt}'s rows, {@code -1} for an embedding row (a decision head reads the
     * lexical term only at text rows, which follow the media).
     *
     * @throws IllegalStateException when the backend's memory is not paged
     */
    public static int[] tokens(ForwardBackend backend, MediaPrompt prompt) {
        if (!paged(backend)) {
            throw new IllegalStateException(Objects.requireNonNull(backend, "backend") + " holds no paged memory");
        }
        return tokens(prompt);
    }

    /**
     * The token ids of the media {@code prompt}'s rows, {@code -1} for an embedding row (a decision head reads the
     * lexical term only at text rows, which follow the media): the memoryless form of
     * {@link #tokens(ForwardBackend, MediaPrompt)}.
     */
    public static int[] tokens(MediaPrompt prompt) {
        Objects.requireNonNull(prompt, "prompt");
        int[] out = new int[prompt.nTokens()];
        int row = 0;
        for (MediaChunk chunk : prompt.chunks()) {
            if (chunk.isText()) {
                int[] tokens = chunk.tokens();
                System.arraycopy(tokens, 0, out, row, tokens.length);
            } else {
                Arrays.fill(out, row, row + chunk.nTokens(), -1);
            }
            row += chunk.nTokens();
        }
        return out;
    }

    // ------------------------------------------------------------------ memoryless evaluations

    /**
     * The final hidden rows (after the output norm) of every row of the media {@code prompt} on a backend planned
     * without memory: the whole prompt runs as one non-causal batch (the graph keeps no cache, so every row attends
     * every row, llama.cpp's cache-less decision graphs) instead of the chunk-by-chunk paged run, every row
     * producing its hidden state.
     *
     * @throws IllegalArgumentException when the backend holds a memory
     */
    public static float[][] memorylessRows(ForwardBackend backend, MediaPrompt prompt) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(prompt, "prompt");
        if (backend.memory() != null) {
            throw new IllegalArgumentException(backend + " holds a memory; its media rows run on the paged path");
        }
        return backend.hiddenBatch(memorylessBatch(prompt, -1));
    }

    /**
     * The logits after the last row of the media {@code prompt} on a backend planned without memory (a copy); the
     * memoryless form of {@link #nextLogits(ForwardBackend, MediaPrompt)}.
     *
     * @throws IllegalArgumentException when the backend holds a memory
     */
    public static float[] memorylessNextLogits(ForwardBackend backend, MediaPrompt prompt) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(prompt, "prompt");
        if (backend.memory() != null) {
            throw new IllegalArgumentException(backend + " holds a memory; its media rows run on the paged path");
        }
        return backend.forwardBatch(memorylessBatch(prompt, prompt.nTokens() - 1))[0].clone();
    }

    /**
     * True when {@code backend} is planned without memory and packs several sequences into one memoryless batch
     * ({@link ForwardBackend#memorylessSequences()} above 1), the case {@link #memorylessSequences} serves.
     */
    public static boolean packsMemoryless(ForwardBackend backend) {
        return backend.memory() == null && backend.memorylessSequences() > 1 && backend.memorylessBatchRows() > 0;
    }

    /**
     * The final hidden rows (after the output norm) of every token of every sequence on a backend that packs
     * memoryless batches ({@link #packsMemoryless}): consecutive sequences pack into batches of at most
     * {@link ForwardBackend#memorylessSequences()} sequences and {@link ForwardBackend#memorylessBatchRows()} rows,
     * each sequence a run of rows from position 0 that attends only itself, so every sequence's rows equal its own
     * single-sequence evaluation while the weights are read once per batch.
     *
     * @return per sequence, its hidden rows in token order
     * @throws IllegalStateException    when the backend does not pack memoryless batches
     * @throws IllegalArgumentException when a sequence is empty or longer than one batch's rows
     */
    public static float[][][] memorylessSequences(ForwardBackend backend, int[][] sequences) {
        if (!packsMemoryless(Objects.requireNonNull(backend, "backend"))) {
            throw new IllegalStateException(backend + " packs no memoryless sequences");
        }
        checkSequences(sequences);
        int maxRows = backend.memorylessBatchRows();
        for (int s = 0; s < sequences.length; s++) {
            if (sequences[s].length > maxRows) {
                throw new IllegalArgumentException("sequence " + s + " holds " + sequences[s].length + " tokens; one"
                        + " memoryless batch holds " + maxRows + " (neuralfree.ubatch)");
            }
        }
        int maxSequences = backend.memorylessSequences();
        float[][][] out = new float[sequences.length][][];
        int first = 0;
        while (first < sequences.length) {
            int end = first;
            int rows = 0;
            while (end < sequences.length && end - first < maxSequences && rows + sequences[end].length <= maxRows) {
                rows += sequences[end].length;
                end++;
            }
            float[][] hidden = backend.hiddenBatch(sequenceBatch(sequences, first, end, rows));
            splitRows(hidden, sequences, first, end, out);
            first = end;
        }
        return out;
    }

    /**
     * Sequences {@code first .. end - 1} ({@code total} rows) as one batch: sequence {@code s - first} from position
     * 0, every row a hidden output row.
     */
    private static ForwardBatch sequenceBatch(int[][] sequences, int first, int end, int total) {
        ForwardBatch.Builder batch = ForwardBatch.builder(total);
        for (int s = first; s < end; s++) {
            for (int i = 0; i < sequences[s].length; i++) {
                batch.add(sequences[s][i], i, s - first, true);
            }
        }
        batch.outputKinds(EnumSet.of(OutputKind.HIDDEN));
        return batch.build();
    }

    /** Hands each of sequences {@code first .. end - 1} its own rows of the batch's {@code hidden} rows. */
    private static void splitRows(float[][] hidden, int[][] sequences, int first, int end, float[][][] out) {
        int off = 0;
        for (int s = first; s < end; s++) {
            out[s] = Arrays.copyOfRange(hidden, off, off + sequences[s].length);
            off += sequences[s].length;
        }
    }

    /**
     * The whole media {@code prompt} as one single-sequence non-causal batch: the chunks' rows keep their absolute
     * positions (a text chunk's tokens at consecutive positions, a media chunk's embedding rows at the chunk's
     * positions), every row an output row, or only {@code logitsRow} producing logits when it is not negative. Rows
     * of an M-RoPE media chunk carry their sections; the other rows take their scalar position in every section.
     */
    private static ForwardBatch memorylessBatch(MediaPrompt prompt, int logitsRow) {
        List<MediaChunk> chunks = prompt.chunks();
        if (chunks.isEmpty()) {
            throw new IllegalStateException("the media prompt holds no chunk to read");
        }
        boolean outputAll = logitsRow < 0;
        ForwardBatch.Builder batch = ForwardBatch.builder(prompt.nTokens());
        batch.nonCausal();
        int nRows = prompt.nTokens();
        int[] scalarPos = new int[nRows];
        List<int[][]> sectionChunks = new ArrayList<>();
        List<Integer> sectionStarts = new ArrayList<>();
        int nSections = 0;
        int row = 0;
        int pos = 0;
        for (MediaChunk chunk : chunks) {
            int n = chunk.nTokens();
            if (chunk.isText()) {
                int[] tokens = chunk.tokens();
                for (int i = 0; i < n; i++) {
                    batch.add(tokens[i], pos + i, 0, outputAll || row + i == logitsRow);
                    scalarPos[row + i] = pos + i;
                }
            } else {
                MediaRopePositions positions = chunk.positions();
                float[][] rows = chunk.embeddings();
                for (int i = 0; i < n; i++) {
                    batch.addEmbedding(rows[i], positions.position(0, i), 0, outputAll || row + i == logitsRow);
                    scalarPos[row + i] = positions.position(0, i);
                }
                if (positions.kind() != MediaRopePositions.Kind.NORMAL) {
                    if (nSections > 0 && positions.sections().length != nSections) {
                        throw new IllegalStateException("the prompt's media rows take "
                                + positions.sections().length + " rope sections, an earlier chunk took "
                                + nSections);
                    }
                    nSections = positions.sections().length;
                    sectionChunks.add(positions.sections());
                    sectionStarts.add(row);
                }
            }
            row += n;
            pos += chunk.nPositions();
        }
        if (row != nRows) {
            throw new IllegalStateException("the prompt's chunks hold " + row + " rows, not the " + nRows
                    + " the prompt reports");
        }
        if (nSections > 0) {
            int[][] sections = new int[nSections][nRows];
            int at = 0;
            int next = 0;
            for (MediaChunk chunk : chunks) {
                int n = chunk.nTokens();
                if (next < sectionStarts.size() && sectionStarts.get(next) == at) {
                    int[][] own = sectionChunks.get(next);
                    for (int s = 0; s < nSections; s++) {
                        System.arraycopy(own[s], 0, sections[s], at, n);
                    }
                    next++;
                } else {
                    for (int i = 0; i < n; i++) {
                        for (int s = 0; s < nSections; s++) {
                            sections[s][at + i] = scalarPos[at + i];
                        }
                    }
                }
                at += n;
            }
            batch.ropePositions(sections);
        }
        batch.outputKinds(EnumSet.of(outputAll ? OutputKind.HIDDEN : OutputKind.LOGITS));
        return batch.build();
    }

    /**
     * The final hidden rows (after the output norm) of every token of {@code tokens}.
     *
     * @throws IllegalStateException when the backend's memory is not paged
     */
    public static float[][] hiddenRows(ForwardBackend backend, int[] tokens) {
        ForwardBatch batch = prompt(tokens, true);
        return inSequenceZero(backend, () -> backend.hiddenBatch(batch));
    }

    /**
     * The final hidden rows (after the output norm) of every row of the media {@code prompt}: each chunk runs as
     * its own single-sequence batch in sequence 0 at its place in the prompt — a text chunk's tokens at consecutive
     * positions, a media chunk's embedding rows at the chunk's positions with its RoPE sections and its mask rule,
     * the piece shape of the batch engine's media prefill without the batching — and the memory keeps the chunk's
     * rows between the batches, so every row attends the prompt rows before it. The sequence frees afterwards,
     * also when a chunk fails.
     *
     * @return one hidden row per prompt row, in prompt order
     * @throws IllegalStateException when the backend's memory is not paged
     */
    public static float[][] hiddenRows(ForwardBackend backend, MediaPrompt prompt) {
        if (!paged(backend)) {
            throw new IllegalStateException(Objects.requireNonNull(backend, "backend") + " holds no paged memory");
        }
        Objects.requireNonNull(prompt, "prompt");
        SequenceMemory sequences = backend.memory().sequences();
        float[][] out = new float[prompt.nTokens()][];
        try {
            int row = 0;
            int pos = 0;
            for (MediaChunk chunk : prompt.chunks()) {
                float[][] hidden = backend.hiddenBatch(chunkBatch(chunk, pos));
                for (int i = 0; i < hidden.length; i++) {
                    out[row + i] = hidden[i];
                }
                row += chunk.nTokens();
                pos += chunk.nPositions();
            }
            if (row != out.length) {
                throw new IllegalStateException("the prompt's chunks hold " + row + " rows, not the "
                        + out.length + " the prompt reports");
            }
        } catch (RuntimeException e) {
            try {
                sequences.remove(0, 0, -1);
            } catch (RuntimeException cleanup) {
                e.addSuppressed(cleanup);
            }
            throw e;
        }
        sequences.remove(0, 0, -1);
        return out;
    }

    /** One chunk of a media prompt as its own batch in sequence 0: every row an output row. */
    private static ForwardBatch chunkBatch(MediaChunk chunk, int posBase) {
        return chunkBatch(chunk, posBase, -1);
    }

    /**
     * One chunk of a media prompt as its own batch in sequence 0: every row an output row producing hidden rows, or
     * the single {@code logitsRow} producing logits when it is not negative.
     */
    private static ForwardBatch chunkBatch(MediaChunk chunk, int posBase, int logitsRow) {
        int n = chunk.nTokens();
        ForwardBatch.Builder batch = ForwardBatch.builder(n);
        boolean outputAll = logitsRow < 0;
        if (chunk.isText()) {
            int[] tokens = chunk.tokens();
            for (int i = 0; i < n; i++) {
                batch.add(tokens[i], posBase + i, 0, outputAll || i == logitsRow);
            }
        } else {
            MediaRopePositions pos = chunk.positions();
            float[][] rows = chunk.embeddings();
            for (int i = 0; i < n; i++) {
                batch.addEmbedding(rows[i], pos.position(0, i), 0, outputAll || i == logitsRow);
            }
            if (pos.kind() != MediaRopePositions.Kind.NORMAL) {
                batch.ropePositions(pos.sections());
            }
            if (chunk.nonCausal()) {
                batch.nonCausal();
            }
        }
        batch.outputKinds(EnumSet.of(outputAll ? OutputKind.HIDDEN : OutputKind.LOGITS));
        return batch.build();
    }

    /** Runs {@code call}, then frees sequence 0 (its pages and recurrent state); a failure frees it too. */
    private static <T> T inSequenceZero(ForwardBackend backend, Supplier<T> call) {
        if (!paged(backend)) {
            throw new IllegalStateException(backend + " holds no paged memory");
        }
        SequenceMemory sequences = backend.memory().sequences();
        T out;
        try {
            out = call.get();
        } catch (RuntimeException e) {
            try {
                sequences.remove(0, 0, -1);
            } catch (RuntimeException cleanup) {
                e.addSuppressed(cleanup);
            }
            throw e;
        }
        sequences.remove(0, 0, -1);
        return out;
    }

    /**
     * {@code tokens} for a paged evaluation, which carries no token types: segment types must be absent or all 0.
     *
     * @throws UnsupportedOperationException when a token type is not 0
     */
    public static int[] untyped(int[] tokens, int[] tokenTypes) {
        if (tokenTypes != null) {
            for (int i = 0; i < tokenTypes.length; i++) {
                if (tokenTypes[i] != 0) {
                    throw new UnsupportedOperationException("token_types: token " + i + " has segment type "
                            + tokenTypes[i] + "; a paged memory evaluates one segment type (0)");
                }
            }
        }
        return tokens;
    }

    /** {@code tokens} in sequence 0 from position 0; every row an output row, or the last one alone. */
    private static ForwardBatch prompt(int[] tokens, boolean everyRow) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            batch.add(tokens[i], i, 0, everyRow || i == tokens.length - 1);
        }
        return batch.build();
    }

    /**
     * Checks a batch of token sequences for the multi-sequence entry points: non-null, at least one sequence, none
     * null or empty.
     *
     * @throws NullPointerException     naming the null sequence
     * @throws IllegalArgumentException naming the empty sequence, or for an empty batch
     */
    public static void checkSequences(int[][] sequences) {
        Objects.requireNonNull(sequences, "sequences");
        if (sequences.length == 0) {
            throw new IllegalArgumentException("a batch needs at least one sequence");
        }
        for (int i = 0; i < sequences.length; i++) {
            Objects.requireNonNull(sequences[i], "sequence " + i);
            if (sequences[i].length == 0) {
                throw new IllegalArgumentException("sequence " + i + " holds no tokens");
            }
        }
    }

    /**
     * The final hidden rows (after the output norm) of every token of every sequence, each sequence run from
     * position 0 in its own memory sequence. Consecutive sequences pack into multi-sequence batches of at most
     * {@link ForwardBackend#maxSequences()} sequences whose page-rounded rows fit the memory's context; each batch
     * runs as one {@link ForwardBackend#hiddenBatch} and its sequences free afterwards, also when the call fails.
     *
     * @return per sequence, its hidden rows in token order
     * @throws IllegalStateException    when the backend's memory is not paged
     * @throws IllegalArgumentException when a sequence is empty or longer than one sequence's context
     */
    public static float[][][] hiddenSequences(ForwardBackend backend, int[][] sequences) {
        if (!paged(backend)) {
            throw new IllegalStateException(Objects.requireNonNull(backend, "backend") + " holds no paged memory");
        }
        checkSequences(sequences);
        PagedSequences paged = backend.memory().pagedSequences();
        int pageSize = paged.pageSize();
        int seqCap = paged.sequence(0).nCtx();
        long poolRows = roundUp(paged.plan().nCtx(), pageSize);
        for (int s = 0; s < sequences.length; s++) {
            if (sequences[s].length > seqCap) {
                throw new IllegalArgumentException("sequence " + s + " holds " + sequences[s].length + " tokens; one"
                        + " sequence of this memory holds " + seqCap + " (neuralfree.ctx, neuralfree.kv.unified)");
            }
        }
        int maxSequences = backend.maxSequences();
        float[][][] out = new float[sequences.length][][];
        int first = 0;
        while (first < sequences.length) {
            int end = first;
            long rows = 0;
            while (end < sequences.length && end - first < maxSequences) {
                long next = rows + roundUp(sequences[end].length, pageSize);
                if (end > first && next > poolRows) {
                    break;
                }
                rows = next;
                end++;
            }
            runPacked(backend, sequences, first, end, out);
            first = end;
        }
        return out;
    }

    /** Runs sequences {@code first .. end - 1} in memory sequences {@code 0 .. end - first - 1}, one batch. */
    private static void runPacked(ForwardBackend backend, int[][] sequences, int first, int end, float[][][] out) {
        int total = 0;
        for (int s = first; s < end; s++) {
            total = Math.addExact(total, sequences[s].length);
        }
        ForwardBatch batch = sequenceBatch(sequences, first, end, total);
        int n = end - first;
        SequenceMemory memory = backend.memory().sequences();
        float[][] hidden;
        try {
            hidden = backend.hiddenBatch(batch);
        } catch (RuntimeException e) {
            removeSequences(memory, n, e);
            throw e;
        }
        removeSequences(memory, n, null);
        splitRows(hidden, sequences, first, end, out);
    }

    private static long roundUp(int rows, int pageSize) {
        return ((long) rows + pageSize - 1) / pageSize * pageSize;
    }

    /** Frees sequences {@code 0 .. n - 1}; a refusal joins {@code failure} as suppressed, or propagates. */
    private static void removeSequences(SequenceMemory sequences, int n, RuntimeException failure) {
        for (int s = 0; s < n; s++) {
            try {
                sequences.remove(s, 0, -1);
            } catch (RuntimeException cleanup) {
                if (failure != null) {
                    failure.addSuppressed(cleanup);
                } else {
                    throw cleanup;
                }
            }
        }
    }
}
