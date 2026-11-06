package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * The self-extend (group attention) of a {@link BatchEngine}: when a running sequence's request sets a
 * group-attention factor {@code grpAttnN > 1}, every {@code grpAttnW} stored positions are grouped and the group's
 * positions divided by the factor, so the sequence attends at an effective context of {@code nCtx / grpAttnN}
 * (llama.cpp's completion loop, tools/completion/completion.cpp:596-615, over {@code llama_memory_seq_add} and
 * {@code llama_memory_seq_div}).
 *
 * <p>The loop runs before a step schedules its next chunk, on the position cursor the sequence reached
 * ({@code Slot.nextPromptPos} while the prompt prefills, {@code Slot.nextPos} while it decodes): while the cursor
 * holds at least one more group, it shifts the positions from the group cursor up by {@code ib * bd}, divides the
 * window {@code [gaI + ib * bd, gaI + ib * bd + gaW)} by the factor, shifts the positions above the window down by
 * {@code dd}, and moves the cursors down by {@code bd} — llama.cpp's arithmetic, with {@code bd} the positions one
 * window retires. Rows keep their cells; only their positions and the RoPE of their keys change, through
 * {@link ForwardBackend#addSequence} and {@link ForwardBackend#divSequence}.
 *
 * <p>A compressed sequence stops serving as a donor fork ({@link Slot#selfExtended}): its positions are no longer
 * the prompt's. Its blocks above the group cursor leave the prefix cache first, since the compression moves the
 * positions their rows held. Confined to the engine's thread.
 */
final class SelfExtend {

    private static final Logger LOGGER = LogManager.getLogger(SelfExtend.class);

    private final BatchState state;
    private final ForwardBackend backend;
    private final SlotLifecycle lifecycle;
    /** Whether the backend serves per-sequence position edits at all. */
    private final boolean enabled;

    SelfExtend(BatchState state, ForwardBackend backend, SlotLifecycle lifecycle) {
        this.state = state;
        this.backend = backend;
        this.lifecycle = lifecycle;
        String refusal = backend.sequenceEditRefusal();
        this.enabled = refusal == null;
        if (!enabled) {
            LOGGER.info("self-extend stays off: {}", refusal);
        }
    }

    /** Whether the backend serves the position edits a self-extend request needs. */
    boolean supported() {
        return enabled;
    }

    /**
     * Compresses every running sequence whose request asks for group attention and whose position cursor reached
     * its next group boundary. Runs before the step schedules, so the next chunk writes at the compressed cursor.
     */
    void apply() {
        if (!enabled) {
            return;
        }
        for (Slot slot : state.active) {
            if (slot.submission.request.selfExtend()) {
                compress(slot);
            }
        }
    }

    /**
     * Runs the group loop over {@code slot} until its position cursor sits below the next boundary: the loop is
     * deterministic in position space, so a step that crosses several boundaries at once applies them all, as
     * llama.cpp's own while loop does.
     */
    private void compress(Slot slot) {
        GenerationRequest request = slot.submission.request;
        int gaN = request.grpAttnN();
        int gaW = request.grpAttnW();
        int step = gaW / gaN;
        int nPast = slot.needsPrefill ? slot.nextPromptPos : slot.nextPos;
        int groups = 0;
        try {
            while (nPast >= slot.gaI + gaW) {
                if (groups == 0 && state.cache != null) {
                    // the compressed rows no longer hold the positions their cached blocks recorded
                    state.cache.rewind(slot.chain, slot.gaI / state.pages.pageSize());
                }
                int ib = (gaN * slot.gaI) / gaW;
                int bd = step * (gaN - 1);
                int dd = step - ib * bd - gaW;
                int window = slot.gaI + ib * bd;
                if (ib * bd != 0) {
                    backend.addSequence(slot.seq, slot.gaI, nPast, ib * bd);
                }
                backend.divSequence(slot.seq, window, window + gaW, gaN);
                if (dd != 0) {
                    backend.addSequence(slot.seq, window + gaW, nPast + ib * bd, dd);
                }
                nPast -= bd;
                slot.gaI += step;
                groups++;
            }
        } catch (RuntimeException e) {
            // a half-applied compression leaves no recoverable sequence: the failure lands on it alone
            lifecycle.retire(slot, null, false, new IllegalStateException("self-extend of sequence " + slot.seq
                    + " failed (grp_attn_n " + gaN + ", grp_attn_w " + gaW + "): " + e.getMessage(), e));
            return;
        }
        if (groups == 0) {
            return;
        }
        slot.selfExtended();
        if (slot.needsPrefill) {
            slot.nextPromptPos = nPast;
        } else {
            slot.nextPos = nPast;
        }
        state.stats.grpAttnEdits += groups;
        LOGGER.debug("self-extend on sequence " + slot.seq + ": " + groups + " group(s) of " + gaW + " positions"
                + " divided by " + gaN + ", the cursor moves to position " + nPast);
    }
}
