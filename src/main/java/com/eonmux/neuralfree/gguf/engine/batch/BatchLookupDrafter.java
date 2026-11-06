package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.LookupDrafterFactory;
import com.eonmux.neuralfree.gguf.engine.draft.DrafterChain;
import com.eonmux.neuralfree.gguf.engine.draft.HistoryDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixRequestCache;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The lookup drafters of a {@link BatchEngine} with per-sequence state (vLLM's per-request n-gram and suffix
 * proposers, llama.cpp server's per-slot {@code common_speculative} state): every sequence slot holds its own set of
 * the configured lookup drafters ({@code ngram-simple}, {@code ngram-map-k}, {@code ngram-map-k4v},
 * {@code ngram-mod}, {@code ngram-cache}, {@code suffix}; {@link LookupDrafterFactory#newDrafters()}), bound to that
 * slot's stored text tokens in the paged memory ({@code SequenceState.textTokens()}: media rows left out, as
 * llama.cpp's server drafts over {@code get_text_tokens}). {@code ngram-simple} searches llama.cpp's single lookup
 * size, or with {@code neuralfree.draft.ngram.simple.size.n.min} below {@code size.n} vLLM's
 * {@code prompt_lookup_min} / {@code prompt_lookup_max} range ({@link com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter.Kind#SIMPLE}). Several types
 * chain in ask order through a
 * {@link DrafterChain}: the first non-empty proposal wins and only its drafter hears {@link #accept}.
 *
 * <p>A slot's drafters are built at its first draft and kept for the slot's later sequences: {@link #reset} drops
 * their per-history state (map, hash table, context cache, suffix tree) before the slot serves another sequence.
 * The {@code ngram-cache} drafters share the factory's loaded cache files read-only; when a dynamic cache file is
 * configured, {@link #close()} writes it once with the n-grams every slot learned. The {@code suffix} drafters share
 * one request tree ({@link LookupDrafterFactory#sharedSuffixRequests()}): every sequence's response feeds it and
 * every sequence drafts from the responses of all of them, past and running, as vLLM's suffix proposer keeps one
 * {@code SuffixDecodingCache} for the batch. Not thread-safe: it runs on the engine's thread.
 */
public final class BatchLookupDrafter implements BatchDrafter {

    private final LookupDrafterFactory factory;
    private final SequenceMemory sequences;
    /** Every slot's drafters, flat, for the dynamic cache save. */
    private final List<HistoryDrafter> built = new ArrayList<>();
    private final DraftModel[] slots;
    /** The request tree the slots' suffix drafters share, or null without one. */
    private final SuffixRequestCache suffixRequests;
    private boolean closed;

    /**
     * @param factory   the lookup types and their settings, at least one type
     * @param sequences the paged memory whose sequences the drafters read
     * @throws IllegalArgumentException when the factory selects no lookup type
     */
    public BatchLookupDrafter(LookupDrafterFactory factory, SequenceMemory sequences) {
        this.factory = Objects.requireNonNull(factory, "factory");
        this.sequences = Objects.requireNonNull(sequences, "sequences");
        if (factory.isEmpty()) {
            throw new IllegalArgumentException("a batch lookup drafter needs at least one lookup type");
        }
        this.slots = new DraftModel[sequences.maxSequences()];
        this.suffixRequests = factory.sharedSuffixRequests();
    }

    @Override
    public int[] draft(int seq, int lastToken, int n) {
        return slot(seq).draft(lastToken, n);
    }

    @Override
    public void accept(int seq, int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must not be negative, got " + nAccepted);
        }
        checkSequence(seq);
        if (slots[seq] != null) {
            slots[seq].accept(nAccepted);
        }
    }

    @Override
    public void reset(int seq) {
        checkSequence(seq);
        if (slots[seq] != null && !closed) {
            slots[seq].reset();
        }
    }

    /**
     * Closes every slot's drafters, then writes the dynamic lookup cache file when one is configured.
     *
     * @throws UncheckedIOException naming the dynamic cache property when the file cannot be written
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (DraftModel d : slots) {
            if (d != null) {
                d.close();
            }
        }
        try {
            factory.saveDynamic(built);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /** The drafter of {@code seq}, built and bound at its first use. */
    private DraftModel slot(int seq) {
        checkSequence(seq);
        if (closed) {
            throw new IllegalStateException("the batch lookup drafter is closed");
        }
        DraftModel d = slots[seq];
        if (d == null) {
            List<HistoryDrafter> members = factory.newDrafters(suffixRequests);
            for (HistoryDrafter m : members) {
                m.bindHistory(() -> sequences.sequence(seq).textTokens());
            }
            built.addAll(members);
            d = members.size() == 1 ? members.get(0) : new DrafterChain(new ArrayList<DraftModel>(members));
            slots[seq] = d;
        }
        return d;
    }

    private void checkSequence(int seq) {
        if (seq < 0 || seq >= slots.length) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside the memory's " + slots.length
                    + " sequences");
        }
    }

    /** The cached requests of the shared suffix request tree, the running ones included; 0 without one. */
    public int suffixCachedRequests() {
        return suffixRequests == null ? 0 : suffixRequests.cachedRequests();
    }

    @Override
    public String toString() {
        return "BatchLookupDrafter{" + String.join(",", factory.names()) + '}';
    }
}
