package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Speculative drafting with a smaller ordinary language model that shares the target's vocabulary (llama.cpp
 * {@code common_speculative_impl_draft_simple}, common/speculative.cpp:219-512, with
 * {@code common_speculative_are_compatible}, :107-176).
 *
 * <p>Per {@link #draft}: the draft model's memory is brought to the target's stored tokens (the longest common
 * prefix is kept, the rest rolled back, the missing target tokens prefilled), then {@code lastToken} is decoded at the
 * target's next position and tokens are drafted greedily ({@link DraftModelLoader#pick}): each step takes the most
 * likely token and its probability among the 10 most likely ones (the drafter's top-k 10 sampler), stops when that
 * probability is below {@code p_min}, and decodes the token to continue unless the draft is complete. Drafted tokens
 * stay in the draft memory; the next draft drops the ones the target rejected by the same prefix comparison, so
 * {@link #accept} keeps no state.
 *
 * <p>The drafter owns the draft model and closes it. Not thread-safe: it runs on the target's forward thread.
 */
public final class ModelDrafter implements DraftModel {

    /** Largest vocabulary size difference llama.cpp accepts ({@code SPEC_VOCAB_MAX_SIZE_DIFFERENCE}). */
    static final int MAX_VOCAB_DIFFERENCE = 128;
    /** First token id whose text must match ({@code SPEC_VOCAB_CHECK_START_TOKEN_ID}). */
    static final int VOCAB_CHECK_START = 5;

    private final EngineModel draft;
    private final float pMin;
    private ForwardBackend target;
    private ForwardBackend drafter;
    private int targetVocab;
    private boolean closed;

    /**
     * @param draft the draft model, owned by this drafter
     * @param pMin  drafting stops at a token whose top-10 probability is below this ({@code --draft-p-min})
     * @throws IllegalArgumentException for a {@code pMin} outside {@code [0, 1]}
     */
    public ModelDrafter(EngineModel draft, float pMin) {
        this.draft = Objects.requireNonNull(draft, "draft");
        if (!(pMin >= 0f && pMin <= 1f)) {
            throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
        }
        this.pMin = pMin;
    }

    /** The draft model. */
    public EngineModel draftModel() {
        return draft;
    }

    /**
     * @throws IllegalArgumentException when the draft model cannot draft for the target: it does not generate
     *                                  autoregressively with memory, or the vocabularies are incompatible
     */
    @Override
    public void bindTarget(EngineModel targetModel) {
        Objects.requireNonNull(targetModel, "target");
        ensureOpen();
        if (target != null) {
            throw new IllegalStateException("the draft model is already bound to a target");
        }
        if (draft.supportClass() != SupportClass.GEN) {
            throw new IllegalArgumentException("draft model " + draft.architecture() + " (" + draft.supportClass()
                    + ") does not generate text autoregressively");
        }
        ForwardBackend tb = targetModel.forwardBackend();
        ForwardBackend db = draft.forwardBackend();
        if (tb.memory() == null || db.memory() == null) {
            throw new IllegalArgumentException("speculative drafting needs model memory on both the target ("
                    + targetModel.architecture() + ") and the draft model (" + draft.architecture() + ")");
        }
        String incompatible = vocabMismatch(targetModel.tokenizer(), draft.tokenizer());
        if (incompatible != null) {
            throw new IllegalArgumentException("draft model " + draft.name() + " cannot draft for "
                    + targetModel.name() + ": " + incompatible);
        }
        this.target = tb;
        this.drafter = db;
        this.targetVocab = targetModel.vocabSize();
    }

    /**
     * Why two vocabularies cannot share token ids, or null when they can (llama.cpp
     * {@code common_speculative_are_compatible}): the same tokenizer type, the same BOS / EOS insertion and ids, sizes
     * within {@value #MAX_VOCAB_DIFFERENCE}, and the same text for every token from id {@value #VOCAB_CHECK_START}.
     */
    static String vocabMismatch(GGUFTokenizer target, GGUFTokenizer draft) {
        TokenizerVocab tv = target.vocab();
        TokenizerVocab dv = draft.vocab();
        if (tv.type() != dv.type()) {
            return "tokenizer types differ (" + tv.type() + " vs " + dv.type() + ")";
        }
        if (tv.addBos() != dv.addBos() || (tv.addBos() && target.bosId() != draft.bosId())) {
            return "BOS handling differs (add " + tv.addBos() + "/" + dv.addBos() + ", id " + target.bosId() + "/"
                    + draft.bosId() + ")";
        }
        if (tv.addEos() != dv.addEos() || (tv.addEos() && target.eosId() != draft.eosId())) {
            return "EOS handling differs (add " + tv.addEos() + "/" + dv.addEos() + ", id " + target.eosId() + "/"
                    + draft.eosId() + ")";
        }
        int nt = tv.size();
        int nd = dv.size();
        if (Math.abs(nt - nd) > MAX_VOCAB_DIFFERENCE) {
            return "vocabulary sizes " + nt + " and " + nd + " differ by more than " + MAX_VOCAB_DIFFERENCE;
        }
        for (int i = VOCAB_CHECK_START; i < Math.min(nt, nd); i++) {
            if (!tv.text(i).equals(dv.text(i))) {
                return "token " + i + " differs ('" + tv.text(i) + "' vs '" + dv.text(i) + "')";
            }
        }
        return null;
    }

    @Override
    public int[] draft(int lastToken, int n) {
        ensureBound();
        if (n <= 0) {
            return new int[0];
        }
        SequenceState ts = target.memory().sequence();
        int pos = ts.nPast();
        int count = Math.min(n, drafter.memory().sequence().nCtx() - pos);
        if (count <= 0 || lastToken < 0 || lastToken >= draft.vocabSize()) {
            return new int[0];
        }
        try {
            sync(ts, pos);
            float[] logits = drafter.decode(lastToken, pos);
            int[] out = new int[count];
            int len = 0;
            while (len < count) {
                DraftModelLoader.Pick pick = DraftModelLoader.pick(logits, logits.length);
                if (pick.p < pMin || pick.id >= targetVocab) {
                    break;
                }
                out[len++] = pick.id;
                if (len < count) {
                    logits = drafter.decode(pick.id, pos + len);
                }
            }
            return Arrays.copyOf(out, len);
        } catch (RuntimeException e) {
            try {
                drafter.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /**
     * Brings the draft memory to the target's tokens {@code [0, pos)}: keeps the common prefix, drops the rest (or
     * empties the memory when it cannot drop it) and prefills the missing target tokens.
     */
    private void sync(SequenceState ts, int pos) {
        ModelMemory dm = drafter.memory();
        SequenceState ds = dm.sequence();
        int stored = ds.nPast();
        int keep = 0;
        int limit = Math.min(stored, pos);
        while (keep < limit && ds.token(keep) == ts.token(keep)) {
            keep++;
        }
        int drop = stored - keep;
        if (drop > 0) {
            if (drafter.canRollback(drop)) {
                drafter.rollback(drop);
            } else {
                drafter.reset();
                keep = 0;
            }
        }
        if (keep < pos) {
            int[] missing = new int[pos - keep];
            for (int i = 0; i < missing.length; i++) {
                missing[i] = ts.token(keep + i);
            }
            drafter.prefill(missing, keep);
        }
    }

    /** Verification keeps no drafter state: the next {@link #draft} re-aligns the draft memory by prefix. */
    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must be non-negative, got " + nAccepted);
        }
    }

    @Override
    public void reset() {
        if (drafter != null && !closed) {
            drafter.reset();
        }
    }

    /** Closes the draft model. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        target = null;
        drafter = null;
        try {
            draft.close();
        } catch (IOException e) {
            throw new UncheckedIOException("closing the draft model " + draft.name() + " failed", e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the draft model drafter is closed");
        }
    }

    private void ensureBound() {
        ensureOpen();
        if (target == null) {
            throw new IllegalStateException("the draft model drafter is not bound to a target");
        }
    }

    @Override
    public String toString() {
        return "ModelDrafter{" + draft.name() + " (" + draft.architecture() + "), pMin=" + pMin + '}';
    }
}
