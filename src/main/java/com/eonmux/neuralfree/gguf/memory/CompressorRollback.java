package com.eonmux.neuralfree.gguf.memory;

/**
 * Rollback bookkeeping of the DeepSeek-V4 compressor state of one memory, shared by the device and heap memories
 * (llama.cpp {@code llama_kv_cache_dsv4::rs_idx} and {@code seq_rm}, src/llama-kv-cache-dsv4.cpp:1463-1510, 1715-1730).
 * Every step writes rollback snapshot planes {@code d = 1..snapshots}: the compressor state as it stood before the
 * step's last {@code d} tokens (llama-kv-cache-dsv4.cpp:684-713). A rollback of {@code n} positions is therefore
 * possible when
 * <ul>
 *   <li>{@code 1 <= n <= snapshots} ({@code n_rs_seq}),</li>
 *   <li>no rollback happened since the last step (llama.cpp keeps a single pending rollback: stacked partial
 *       removals do not compose), and</li>
 *   <li>the last step stored at least {@code n} tokens: a plane {@code d} beyond the last step's tokens holds the
 *       state before that step, not the state {@code d} positions back. llama.cpp guarantees it by keeping the
 *       trailing {@code 1 + n_rs_seq} tokens of a batch in one ubatch (llama-kv-cache-dsv4.cpp:1414-1417).</li>
 * </ul>
 * The memory then restores plane {@code n} into the live plane before the next step reads it, as llama.cpp restores
 * plane {@code rs_idx} at the start of the next ubatch (llama-kv-cache-dsv4.cpp:657-681). Mutable runtime state owned by
 * one memory; not thread-safe.
 */
public final class CompressorRollback {

    private final int snapshots;
    private int lastStepTokens;
    private boolean pending;

    /**
     * @param snapshots rollback snapshot planes of the memory's compressed caches ({@link MemoryPlan#rollbackSnapshots()})
     * @throws IllegalArgumentException for a negative count
     */
    public CompressorRollback(int snapshots) {
        if (snapshots < 0) {
            throw new IllegalArgumentException("compressor rollback snapshots must be >= 0, got " + snapshots);
        }
        this.snapshots = snapshots;
    }

    /** Rollback snapshot planes ({@code n_rs_seq}). */
    public int snapshots() {
        return snapshots;
    }

    /** Records a step of {@code nTokens} tokens: its snapshots replace the previous ones and any rollback is done. */
    public void onStep(int nTokens) {
        if (nTokens < 1) {
            throw new IllegalArgumentException("a step stores at least one token, got " + nTokens);
        }
        lastStepTokens = nTokens;
        pending = false;
    }

    /**
     * Why the last {@code nTokens > 0} positions cannot be dropped from the compressor state, or null when the
     * snapshot plane {@code nTokens} holds the state to restore.
     */
    public String blocker(int nTokens) {
        if (snapshots == 0) {
            return "DSV4 compressor state accumulates across positions and the memory keeps no rollback snapshots"
                    + " (n_rs_seq = 0; speculative decoding with an MTP, eagle3 or dflash drafter sets it)";
        }
        if (nTokens > snapshots) {
            return "DSV4 compressor state keeps rollback snapshots for at most " + snapshots + " positions";
        }
        if (pending) {
            return "the DSV4 compressor state was already rolled back since the last step";
        }
        if (nTokens > lastStepTokens) {
            return "the last step stored " + lastStepTokens + " positions; DSV4 compressor snapshots cover only the"
                    + " positions of the last step";
        }
        return null;
    }

    /**
     * Records a rollback of {@code nTokens} positions and returns the snapshot plane to restore into the live plane.
     *
     * @throws IllegalStateException when {@link #blocker(int)} refuses it
     */
    public int onRollback(int nTokens) {
        String why = nTokens < 1 ? "a rollback drops at least one position" : blocker(nTokens);
        if (why != null) {
            throw new IllegalStateException("cannot roll back " + nTokens + " positions of DSV4 compressor state: "
                    + why);
        }
        pending = true;
        return nTokens;
    }

    /** Forgets the last step and any rollback (the memory was emptied). */
    public void reset() {
        lastStepTokens = 0;
        pending = false;
    }

    @Override
    public String toString() {
        return "CompressorRollback{snapshots=" + snapshots + ", lastStep=" + lastStepTokens
                + (pending ? ", rolled back" : "") + '}';
    }
}
