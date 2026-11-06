package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * The slot context shift of a {@link BatchEngine} (llama.cpp server's pre_decode shift,
 * server-context.cpp:3783-3824): when a decoding sequence's next token has no position left, the sequence drops
 * the positions {@code [n_keep, n_keep + n_discard)} ({@code seq_rm}) and moves every later position down
 * ({@code seq_add}), so generation continues in the same slot instead of ending with
 * {@link StopReason#CONTEXT_FULL}. The counts come from the request's {@code n_keep}/{@code n_discard} over the
 * sequence's stored positions with llama.cpp's rules: a negative {@code n_keep} keeps the whole prompt, a leading
 * BOS is kept on top, {@code n_keep} never passes the {@code n_ctx - 4} margin, and {@code n_discard} 0 discards
 * half of what lies beyond the kept tokens, never reaching the last row. A shift serves the text prompt of a
 * decoder-only model, whose rows are one token per position; a media prompt's rows share M-RoPE positions and an
 * encoder-decoder's first row is its fixed decoder start, so those sequences stop with {@code CONTEXT_FULL}
 * instead, as a request setting {@code n_keep}/{@code n_discard} on one refuses at submit. The backend must serve
 * the move ({@code ForwardBackend#shiftSequence}, with the moved keys re-rotated): a backend that shifts no paged
 * sequence, or a memory refusal (a DeepSeek-V4 compressor beyond its rollback snapshots), turns the shift off —
 * named once at construction — or stops the sequence with {@code CONTEXT_FULL}, and a pool too full to copy the
 * sequence's shared pages fails it alone (the move would leave it inconsistent — llama.cpp's server refuses a
 * shared-prompt shift the same way). Before the move the sequence's own cached blocks above the cut leave the
 * prefix cache ({@link PrefixCache#rewind}), so its pages hold no cache reference the move would copy around, and
 * the kept prefix stays cached for the sequences after it. After a shift the slot keeps decoding with the armed
 * token, the fed tokens compact with the moved rows, and the handle reports {@code truncated}.
 */
final class ContextShift {

    private static final Logger LOGGER = LogManager.getLogger(ContextShift.class);

    /** The margin llama.cpp's server keeps below the context when clamping {@code n_keep}. */
    private static final int KEEP_MARGIN = 4;

    private final BatchState state;
    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    private final SlotLifecycle lifecycle;
    /** Whether the engine shifts; off (config, or a backend that shifts no paged sequence) leaves every full context stopping. */
    private final boolean enabled;

    ContextShift(BatchState state, ForwardBackend backend, GGUFTokenizer tokenizer, SlotLifecycle lifecycle,
                 boolean enabled) {
        this.state = state;
        this.backend = backend;
        this.tokenizer = tokenizer;
        this.lifecycle = lifecycle;
        if (enabled) {
            String refusal = backend.shiftSequenceRefusal();
            if (refusal != null) {
                LOGGER.info("the context shift stays off: {}", refusal);
                enabled = false;
            }
        }
        this.enabled = enabled;
    }

    /**
     * Shifts {@code slot} when its next token has no position left, or stops it with
     * {@link StopReason#CONTEXT_FULL}. Nothing to do while positions remain.
     */
    void shiftOrFinish(Slot slot) {
        if (slot.nextPos < slot.nCtx) {
            return;
        }
        if (!shift(slot)) {
            lifecycle.finish(slot, StopReason.CONTEXT_FULL);
        }
    }

    /**
     * Whether {@code slot} was served: shifted, or failed on its own after a move that cannot finish. False ends
     * the sequence with {@code CONTEXT_FULL} — the shift is off, cannot serve the sequence, has nothing to
     * discard, or the memory refused the move before touching it.
     */
    private boolean shift(Slot slot) {
        if (!enabled || !canShift(slot)) {
            return false;
        }
        int nPast = slot.nextPos;
        int[] prompt = slot.submission.prompt.samplerTokens();
        int nKeep = keepCount(slot.submission.request, slot.submission.prompt.nTokens(),
                prompt.length > 0 ? prompt[0] : -1, tokenizer.specialTokens().bos(), slot.nCtx, nPast);
        int nDiscard = discardCount(slot.submission.request, nKeep, nPast);
        if (nDiscard <= 0) {
            return false;
        }
        state.evictFor(nDiscard, 1); // best effort: free cached blocks so the move has pages to copy into
        // the sequence's own cached blocks above the cut share its pages: dropping their cache references first
        // lets the move write the sequence's pages in place, with no page to allocate
        if (state.cache != null) {
            state.cache.rewind(slot.chain, nKeep / state.pages.pageSize());
        }
        try {
            backend.shiftSequence(slot.seq, nKeep, nDiscard);
        } catch (UnsupportedOperationException e) {
            LOGGER.warn("context full and the context shift is unsupported: {}", e.getMessage());
            return false;
        } catch (RuntimeException e) {
            // a move that stopped halfway leaves no recoverable sequence: the failure lands on it alone
            lifecycle.retire(slot, null, false, new IllegalStateException("context shift of sequence "
                    + slot.seq + " failed (n_keep " + nKeep + ", n_discard " + nDiscard + "): " + e.getMessage(), e));
            return true;
        }
        slot.contextShifted(nKeep, nDiscard);
        slot.submission.handle.truncated(true);
        state.stats.contextShifts++;
        LOGGER.info("slot context shift on sequence " + slot.seq + ": n_keep " + nKeep + ", n_discard "
                + nDiscard + ", " + nPast + " -> " + (nPast - nDiscard) + " positions");
        return true;
    }

    /** Whether the shift can serve {@code slot}: the text prompt of a decoder-only model. */
    private boolean canShift(Slot slot) {
        return state.encoder == null && slot.submission.prompt.isTextOnly();
    }

    /**
     * The positions the shift keeps (llama.cpp's {@code n_keep} arithmetic): the request's count, a negative one
     * keeping the whole prompt, a leading BOS on top, never past the {@code n_ctx - 4} margin nor the stored
     * positions. {@code firstPromptToken} below 0 (a prompt without tokens) keeps no BOS.
     */
    static int keepCount(GenerationRequest request, int promptTokens, int firstPromptToken, int bos, int nCtx,
                         int nPast) {
        int nKeep = request.nKeep() < 0 ? promptTokens : request.nKeep();
        if (firstPromptToken >= 0 && bos != SpecialTokens.NULL && firstPromptToken == bos) {
            nKeep += 1;
        }
        nKeep = Math.min(nKeep, nCtx - KEEP_MARGIN);
        return Math.min(Math.max(nKeep, 0), nPast);
    }

    /**
     * The positions the shift discards (llama.cpp's {@code n_discard} arithmetic) after {@code nKeep} of
     * {@code nPast} stored positions: the request's count, 0 taking half of what lies beyond the kept positions,
     * never reaching the last stored row.
     */
    static int discardCount(GenerationRequest request, int nKeep, int nPast) {
        int nLeft = nPast - nKeep;
        int nDiscard = request.nDiscard();
        if (nDiscard <= 0) {
            nDiscard = nLeft / 2;
        }
        return Math.min(Math.max(nDiscard, 0), Math.max(0, nLeft - 1));
    }
}
