package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.Objects;

/**
 * A generation prompt that carries precomputed embedding rows (vLLM {@code prompt_embeds}, the chat
 * {@code image_embeds}/{@code audio_embeds} content parts): alternating runs of token ids and embedding rows.
 * A rows-only prompt (completions {@code prompt_embeds}) is an empty first token run and the rows as the only
 * embedding run; a spliced prompt (chat) places its rows where the content parts sat in the rendered text. The
 * rows occupy the positions they sit at, so the prompt holds {@link #rowCount()} positions in total, and the
 * token runs alone feed the sampling history and the prefix bookkeeping.
 *
 * <p>Immutable; every constructor copies what it is given and validates it: token ids non-negative, rows
 * non-empty of one finite width. Build one per request; a prompt never changes after validation.</p>
 */
public final class EmbedsPrompt {

    private final int[][] tokenRuns;    // k + 1 token runs
    private final float[][][] embedsRuns; // k embedding runs between the token runs

    private EmbedsPrompt(int[][] tokenRuns, float[][][] embedsRuns) {
        this.tokenRuns = tokenRuns;
        this.embedsRuns = embedsRuns;
    }

    /**
     * A prompt of {@code tokenRuns} token runs with the rows of {@code embedsRuns[i]} between runs {@code i}
     * and {@code i + 1}: {@code embedsRuns.length == tokenRuns.length - 1}, runs may be empty.
     *
     * @throws IllegalArgumentException when a run shape is broken, a token id is negative, a row is null or
     *                                 of another width than the rows around it, holds a non-finite value,
     *                                 or the prompt holds no row at all
     */
    public static EmbedsPrompt of(int[][] tokenRuns, float[][][] embedsRuns) {
        Objects.requireNonNull(tokenRuns, "tokenRuns");
        Objects.requireNonNull(embedsRuns, "embedsRuns");
        if (tokenRuns.length != embedsRuns.length + 1) {
            throw new IllegalArgumentException(tokenRuns.length + " token runs cannot carry " + embedsRuns.length
                    + " embedding runs between them: one embedding run fewer than token runs");
        }
        int width = -1;
        int rows = 0;
        int[][] runs = new int[tokenRuns.length][];
        for (int i = 0; i < tokenRuns.length; i++) {
            runs[i] = checkedTokens(i, tokenRuns[i]);
        }
        float[][][] embeds = new float[embedsRuns.length][][];
        for (int i = 0; i < embedsRuns.length; i++) {
            embeds[i] = checkedRows(i, embedsRuns[i]);
            rows += embeds[i].length;
            for (float[] row : embeds[i]) {
                if (width < 0) {
                    width = row.length;
                } else if (row.length != width) {
                    throw new IllegalArgumentException("embedding rows share one width: row of " + row.length
                            + " floats after rows of " + width);
                }
            }
        }
        if (rows == 0) {
            throw new IllegalArgumentException("an embedding prompt holds at least one row");
        }
        return new EmbedsPrompt(runs, embeds);
    }

    /** The rows-only prompt of completions {@code prompt_embeds}: the rows between two empty token runs. */
    public static EmbedsPrompt ofRows(float[][] rows) {
        return of(new int[][] {new int[0], new int[0]}, new float[][][] {rows});
    }

    private static int[] checkedTokens(int run, int[] tokens) {
        Objects.requireNonNull(tokens, "token run " + run);
        int[] copy = tokens.clone();
        for (int i = 0; i < copy.length; i++) {
            if (copy[i] < 0) {
                throw new IllegalArgumentException("token run " + run + " holds the negative id " + copy[i]
                        + " at " + i);
            }
        }
        return copy;
    }

    private static float[][] checkedRows(int run, float[][] rows) {
        Objects.requireNonNull(rows, "embedding run " + run);
        float[][] copy = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            Objects.requireNonNull(rows[i], "embedding run " + run + " row " + i);
            copy[i] = rows[i].clone();
            for (float v : copy[i]) {
                if (!Float.isFinite(v)) {
                    throw new IllegalArgumentException("embedding run " + run + " row " + i
                            + " holds a value that is not finite");
                }
            }
        }
        return copy;
    }

    /** The number of token runs: one more than the embedding runs. */
    public int segments() {
        return tokenRuns.length;
    }

    /** Token ids of run {@code segment} (a copy), possibly empty. */
    public int[] tokens(int segment) {
        return tokenRuns[segment].clone();
    }

    /**
     * The embedding rows between token run {@code segment} and the next (copies); empty for the last run.
     *
     * @throws IndexOutOfBoundsException when {@code segment} is outside the runs
     */
    public float[][] embeds(int segment) {
        if (segment < 0 || segment >= embedsRuns.length) {
            throw new IndexOutOfBoundsException("segment " + segment + " outside the " + embedsRuns.length
                    + " embedding runs");
        }
        float[][] out = new float[embedsRuns[segment].length][];
        for (int i = 0; i < out.length; i++) {
            out[i] = embedsRuns[segment][i].clone();
        }
        return out;
    }

    /** Whether every row is an embedding row: no token run holds an id. */
    public boolean rowsOnly() {
        for (int[] run : tokenRuns) {
            if (run.length > 0) {
                return false;
            }
        }
        return true;
    }

    /** Positions the prompt fills: its tokens and embedding rows together. */
    public int rowCount() {
        int rows = 0;
        for (int[] run : tokenRuns) {
            rows += run.length;
        }
        for (float[][] run : embedsRuns) {
            rows += run.length;
        }
        return rows;
    }

    /** Width of every embedding row of the prompt. */
    public int rowWidth() {
        for (float[][] run : embedsRuns) {
            if (run.length > 0) {
                return run[0].length;
            }
        }
        throw new IllegalStateException("an embedding prompt holds at least one row");
    }

    /** The token ids of every token run, in order: the sampling history of the prompt. */
    public int[] allTokens() {
        int n = 0;
        for (int[] run : tokenRuns) {
            n += run.length;
        }
        int[] out = new int[n];
        int at = 0;
        for (int[] run : tokenRuns) {
            System.arraycopy(run, 0, out, at, run.length);
            at += run.length;
        }
        return out;
    }

    /**
     * The prompt as the media prompt the batch engine runs: every non-empty token run a text chunk, every
     * non-empty embedding run a prompt-embeddings chunk ({@link MediaChunk#embeds}: causal, one position per row,
     * llama.cpp's {@code set_position_normal}) starting where the runs before it end.
     */
    public MediaPrompt toMediaPrompt() {
        MediaPrompt.Builder b = MediaPrompt.builder();
        int pos = 0;
        for (int i = 0; i < tokenRuns.length; i++) {
            if (tokenRuns[i].length > 0) {
                b.addText(tokenRuns[i]);
                pos += tokenRuns[i].length;
            }
            if (i < embedsRuns.length && embedsRuns[i].length > 0) {
                b.addEmbeds(pos, embedsRuns[i]);
                pos += embedsRuns[i].length;
            }
        }
        return b.build();
    }

    @Override
    public String toString() {
        return "EmbedsPrompt{" + rowCount() + " rows, " + (tokenRuns.length - 1) + " embedding run(s), width "
                + rowWidth() + (rowsOnly() ? ", rows only" : "") + '}';
    }
}
