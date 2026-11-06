package com.eonmux.neuralfree.gguf.engine.lora;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * The adapter slot of every row of the forward pass a model's projection deltas currently serve (vLLM's
 * {@code token_lora_mapping} of the punica wrapper, punica_wrapper/punica_base.py): slot 0 runs the base model with
 * the startup adapters ({@code --lora}); slot {@code s >= 1} runs the per-request adapters the {@link LoraRegistry}
 * opened in that slot instead. Each delta applies only to the rows of its own slot, so a slot's adapters replace the
 * startup set rather than add to it. The forward backend sets the table before the rows' products run — the ubatch
 * rows before the trunk, the output rows before the head — and clears it after the pass.
 *
 * <p>A table is uniform when every row runs in one slot: then a product over any rows of the pass applies that
 * slot's deltas to all of them. A mixed table holds rows of several slots: a product must then run over exactly the
 * table's rows (row {@code t} of the product is row {@code t} of the table), and its deltas apply run by run over
 * the consecutive rows of one slot. {@link #mixable} names the models whose blocks keep that alignment; the
 * backends split the other models' batches into slot-uniform ubatches.</p>
 *
 * <p>The session slot ({@link #session}) is the slot of the passes whose batch sets no per-row slots: 0 by default,
 * another slot while a single-sequence run or a pooling call runs one request's adapters over passes that carry no
 * table ({@link LoraRegistry#useSlot}).</p>
 *
 * <p>Mutable and owned by one model's deltas; set and read on the forward thread.</p>
 */
public final class AdapterRows {

    /** The adapter slot of every row of a mixed pass, or null when every row runs in {@link #uniform}. */
    private int[] slots;
    /** The slot every row runs in, or -1 when the rows run in different slots. */
    private int uniform;
    /** The slot of a pass whose batch sets no per-row slots. */
    private int session;

    /**
     * Sets the slot of every row of the following products, or, for null, puts every row in the session slot.
     *
     * @throws IllegalArgumentException for a negative slot
     */
    public void set(int[] rowSlots) {
        if (rowSlots == null || rowSlots.length == 0) {
            clear();
            return;
        }
        int first = rowSlots[0];
        boolean same = true;
        for (int s : rowSlots) {
            if (s < 0) {
                throw new IllegalArgumentException("adapter slots are non-negative, got " + s);
            }
            same &= s == first;
        }
        this.slots = same ? null : rowSlots.clone();
        this.uniform = same ? first : -1;
    }

    /** Every following row runs in the session slot. */
    public void clear() {
        this.slots = null;
        this.uniform = session;
    }

    /**
     * Sets the session slot, the slot of the passes whose batch sets no per-row slots, and puts the following rows
     * in it; 0 restores the startup adapters.
     *
     * @throws IllegalArgumentException for a negative slot
     */
    public void session(int slot) {
        if (slot < 0) {
            throw new IllegalArgumentException("adapter slots are non-negative, got " + slot);
        }
        this.session = slot;
        clear();
    }

    /** The session slot. */
    public int session() {
        return session;
    }

    /** True when some row of the pass runs in a slot other than 0. */
    public boolean active() {
        return uniform != 0;
    }

    /** The slot every row runs in (0 for an inactive table), or -1 when the rows run in different slots. */
    public int uniform() {
        return uniform;
    }

    /** Rows of a mixed table, 0 for a uniform one. */
    public int rows() {
        return slots == null ? 0 : slots.length;
    }

    /** The slot of row {@code row} of the pass. */
    public int slot(int row) {
        if (uniform >= 0) {
            return uniform;
        }
        if (row < 0 || row >= slots.length) {
            throw new IndexOutOfBoundsException("row " + row + " outside the " + slots.length + " rows of the pass");
        }
        return slots[row];
    }

    /**
     * The end of the run of rows starting at {@code start} that share one slot, at most {@code n}: the next row
     * whose slot differs, or {@code n}.
     */
    public int runEnd(int start, int n) {
        if (uniform >= 0) {
            return n;
        }
        int s = slots[start];
        int end = start + 1;
        while (end < n && slots[end] == s) {
            end++;
        }
        return end;
    }

    /**
     * Checks that a product of {@code n} rows can tell the slot of each of its rows: a mixed table needs the
     * product to run over exactly its rows.
     *
     * @throws IllegalStateException naming {@code product} when a mixed table holds a different row count
     */
    public void checkSpan(String product, int n) {
        if (uniform < 0 && n != slots.length) {
            throw new IllegalStateException(product + " runs " + n + " rows of a pass whose " + slots.length
                    + " rows run in different adapter slots; the block must run the pass's rows in one product"
                    + " (the backend splits this model's batches into slot-uniform ubatches otherwise)");
        }
    }

    /**
     * The slots of a ubatch's rows in ubatch order, gathered from the batch's per-row slots by the ubatch's batch
     * rows, or null when the batch sets none.
     */
    public static int[] ofRows(int[] batchSlots, int[] batchRows) {
        Objects.requireNonNull(batchRows, "batchRows");
        if (batchSlots == null) {
            return null;
        }
        int[] out = new int[batchRows.length];
        for (int i = 0; i < batchRows.length; i++) {
            out[i] = batchSlots[batchRows[i]];
        }
        return out;
    }

    /**
     * Whether every block of {@code spec} applies its projection deltas over the pass's rows in one product, so a
     * ubatch may mix rows of several adapter slots: every layer mixes by attention (or not at all) and runs a dense
     * or mixture-of-experts FFN, and carries none of the gemma3n extras that project rows of their own (AltUp,
     * Laurel, per-layer embeddings). Recurrent, MLA, sparse and the other mixers run per sequence segment or in row
     * chunks; their batches split into slot-uniform ubatches.
     */
    public static boolean mixable(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        for (LayerSpec layer : spec.layers()) {
            MixerKind m = layer.mixerKind();
            if (m != MixerKind.ATTENTION && m != MixerKind.NONE) {
                return false;
            }
            FfnKind f = layer.ffnKind();
            if (f != FfnKind.DENSE && f != FfnKind.MOE && f != FfnKind.NONE) {
                return false;
            }
            for (String extra : PROJECTING_EXTRAS) {
                if (layer.hasExtra(extra)) {
                    return false;
                }
            }
        }
        return true;
    }

    /** The layer extras that apply projection deltas over rows they gather themselves. */
    private static final String[] PROJECTING_EXTRAS = {LayerSpec.Extras.ALTUP, LayerSpec.Extras.LAUREL,
            LayerSpec.Extras.PLE, LayerSpec.Extras.NGRAM_PLE};

    @Override
    public String toString() {
        return slots == null ? "AdapterRows{slot " + uniform + "}" : "AdapterRows{" + Arrays.toString(slots) + "}";
    }
}
