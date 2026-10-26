package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Several drafter types per target, asked in order (llama.cpp's comma-separated {@code --spec-type a,b,c}, where the
 * next implementation drafts only while the previous ·gave nothing ("after the first successful draft from an
 * implementation... all drafting flags will be reset to false"): the first non-empty proposal is the chain's draft
 * and only its drafter hears {@link #accept}, llama.cpp's per-implementation accept.
 *
 * <p>A chain member is any {@link DraftModel}; typically the token-stream drafters ({@link NgramDrafter},
 * {@link SuffixDrafter}) first, the hidden-fed ones ({@code DFlash2}, eagle3, draft-simple, the MTP head) after, but
 * any order the caller picks is valid. The chain binds, resets and closes all members it holds. Not thread-safe: it
 * runs on the target's forward thread.
 */
public final class DrafterChain implements DraftModel {

    private final List<DraftModel> drafters;
    private int activeIndex = -1;
    private boolean closed;

    /**
     * @param drafters the drafter types, in ask order; at least one
     * @throws IllegalArgumentException for an empty chain or a null member
     */
    public DrafterChain(DraftModel... drafters) {
        this(toList(drafters));
    }

    /**
     * @param drafters the drafter types, in ask order; at least one
     * @throws IllegalArgumentException for an empty chain or a null member
     */
    public DrafterChain(List<DraftModel> drafters) {
        Objects.requireNonNull(drafters, "drafters");
        if (drafters.isEmpty()) {
            throw new IllegalArgumentException("a drafter chain needs at least one drafter");
        }
        List<DraftModel> copy = new ArrayList<>(drafters.size());
        for (DraftModel d : drafters) {
            Objects.requireNonNull(d, "drafter");
            copy.add(d);
        }
        this.drafters = Collections.unmodifiableList(copy);
    }

    private static List<DraftModel> toList(DraftModel[] drafters) {
        Objects.requireNonNull(drafters, "drafters");
        List<DraftModel> out = new ArrayList<>(drafters.length);
        for (DraftModel d : drafters) {
            Objects.requireNonNull(d, "drafter");
            out.add(d);
        }
        return out;
    }

    /** The chain members, in ask order. */
    public List<DraftModel> drafters() {
        return drafters;
    }

    /**
     * Binds every member to {@code target} in order; a member that fails leaves the earlier ones bound (they are
     * closed by the caller through {@link #close()}).
     *
     * @throws IllegalStateException    when the chain is closed or already bound, naming the mismatching member
     * @throws IllegalArgumentException from a member's {@link DraftModel#bindTarget}
     */
    @Override
    public void bindTarget(EngineModel target) {
        Objects.requireNonNull(target, "target");
        requireOpen();
        for (int i = 0; i < drafters.size(); i++) {
            DraftModel drafter = drafters.get(i);
            try {
                drafter.bindTarget(target);
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("drafter chain member " + i + " (" + drafter
                        + ") rejected the target: " + e.getMessage(), e);
            }
        }
    }

    /**
     * Asks the members in order and takes the first non-empty proposal; a later member drafts only the tokens the
     * earlier members could not (llama.cpp's chain draft).
     */
    @Override
    public int[] draft(int lastToken, int n) {
        requireOpen();
        if (n <= 0) {
            return new int[0];
        }
        for (int i = 0; i < drafters.size(); i++) {
            int[] proposal = drafters.get(i).draft(lastToken, n);
            if (proposal != null && proposal.length > 0) {
                activeIndex = i;
                return proposal;
            }
        }
        activeIndex = -1;
        return new int[0];
    }

    /**
     * Reports verification to the member whose proposal the chain returned; the members without a returned proposal
     * hear nothing, which llama.cpp's ``drafting'' flags mirror. The decoder's {@code accept(0)} after a draft-less
     * round reaches no member.
     *
     * @throws IllegalArgumentException for a negative count
     */
    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("the accepted count must not be negative, got " + nAccepted);
        }
        requireOpen();
        if (activeIndex != -1) {
            drafters.get(activeIndex).accept(nAccepted);
        }
    }

    /** Resets every member (a generation start drops one another's per-sequence state). */
    @Override
    public void reset() {
        requireOpen();
        for (DraftModel drafter : drafters) {
            drafter.reset();
        }
    }

    /** Closes every member. Idempotent; the members' own {@code close()} is idempotent too. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (DraftModel drafter : drafters) {
            drafter.close();
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("the drafter chain is closed");
        }
    }

    @Override
    public String toString() {
        return "DrafterChain{[" + drafters.size() + " drafters" + (closed ? ", closed" : "") + "]}";
    }
}
