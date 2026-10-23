package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.ModelMemory;

/**
 * Runs the model's forward passes for one sequence on one device (GPU or CPU). Implemented by
 * {@code GpuForwardBackend}, {@code CpuForwardBackend} and the encoder-decoder backends; driven by the generation
 * loop, the speculative decoder, the diffusion generator and the embedding / classifier heads.
 *
 * <p>Contract shared by every method:
 * <ul>
 *   <li>Token ids must lie in {@code [0, nVocab)}; positions are absolute sequence positions.</li>
 *   <li>Logits rows are {@code nVocab} floats. A returned array may be reused by the next call on this backend;
 *       callers may modify it in place and must copy what they keep across calls.</li>
 *   <li>Calls that write memory throw {@link ContextFullException} when the positions they need exceed the
 *       memory's context length, before any work is enqueued.</li>
 *   <li>A backend is used by one thread at a time.</li>
 * </ul>
 */
public interface ForwardBackend extends AutoCloseable {

    /**
     * Evaluates {@code tokens} at positions {@code pos0 .. pos0 + tokens.length - 1} (causal), chunked by the
     * memory's ubatch, storing them in memory.
     *
     * @return logits of the last row
     */
    float[] prefill(int[] tokens, int pos0);

    /**
     * Evaluates one token at position {@code pos} and stores it in memory.
     *
     * @return logits of the token
     */
    float[] decode(int token, int pos);

    /**
     * {@link #decode} followed by greedy selection. GPU backends select on the device and read back only the
     * token id; the default selects on the host with {@link #argmax(float[])}.
     *
     * @return the id of the highest logit (lowest id on ties)
     */
    default int decodeGreedyOnDevice(int token, int pos) {
        return argmax(decode(token, pos));
    }

    /**
     * Evaluates {@code tokens} causally at positions {@code pos0 ..} and stores them in memory, like
     * {@link #prefill}, returning the logits of the requested rows (speculative verification).
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @return one logits row per requested row
     */
    float[][] forwardRows(int[] tokens, int pos0, int[] outputRows);

    /**
     * Evaluates the whole of {@code tokens} at positions {@code 0 .. tokens.length - 1} with non-causal attention,
     * without reading or modifying memory (masked diffusion, encoders).
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @return one logits row per requested row
     */
    float[][] forwardAll(int[] tokens, int[] outputRows);

    /**
     * Evaluates the whole of {@code tokens} from position 0 with the model's own attention mode (non-causal for
     * encoders, causal for decoder embedding models), without reading or modifying memory, and returns the final
     * hidden states after the output norm: the input of the LM or classifier head (llama.cpp {@code t_embd}).
     *
     * @param tokenTypes segment ids per token (BERT {@code token_types}), or null for all zero
     * @return {@code [tokens.length][width]} hidden rows
     */
    float[][] hiddenRows(int[] tokens, int[] tokenTypes);

    /** Empties every cache and zeroes every state ({@link ModelMemory#reset()}). */
    void reset();

    /** Whether the last {@code nTokens} positions can be dropped ({@link ModelMemory#canRollback(int)}). */
    boolean canRollback(int nTokens);

    /**
     * Drops the last {@code nTokens} positions from memory.
     *
     * @throws IllegalStateException when {@link #canRollback(int)} is false
     */
    void rollback(int nTokens);

    /**
     * Context shift (llama.cpp {@code n_keep} / {@code n_discard}): keeps positions {@code [0, nKeep)}, drops
     * {@code [nKeep, nKeep + nDiscard)} and moves every later position down by {@code nDiscard}, re-rotating the
     * stored keys.
     *
     * @throws UnsupportedOperationException when a layer's memory cannot be shifted (recurrent and sparse mixers)
     */
    void shiftContext(int nKeep, int nDiscard);

    /** The memory this backend writes; drafters that share the target's KV bind to it. */
    ModelMemory memory();

    /**
     * Installs the hidden-state observer used by every later forward call.
     *
     * @param tap the observer; {@link HiddenTap#NONE} removes it
     * @throws NullPointerException when {@code tap} is null
     */
    void setHiddenTap(HiddenTap tap);

    /** Releases every device and host resource this backend owns. */
    @Override
    void close();

    /**
     * Index of the largest value, the first one on ties; NaN never wins over an earlier value (llama.cpp
     * greedy sampler order).
     *
     * @throws IllegalArgumentException when {@code logits} is empty
     */
    static int argmax(float[] logits) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("argmax of an empty logits row");
        }
        int best = 0;
        float bestValue = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }
}
