package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Several drafter types per target, asked in order (llama.cpp's comma-separated {@code --spec-type a,b,c}, where the
 * next implementation drafts only while the previous gave nothing ("after the first successful draft from an
 * implementation... all drafting flags will be reset to false"): the first non-empty proposal is the chain's draft
 * and only its drafter hears {@link #accept}, llama.cpp's per-implementation accept.
 *
 * <p>A chain member is any {@link DraftModel}; typically the token-stream drafters ({@link NgramDrafter},
 * {@link SuffixDrafter}) first, the hidden-fed ones ({@code DFlash2}, eagle3, draft-simple, the MTP head) after, but
 * any order the caller picks is valid. The chain binds, resets and closes all members it holds. Not thread-safe: it
 * runs on the target's forward thread.
 */
public final class DrafterChain implements DraftModel, DraftProbs {

    private final List<DraftModel> drafters;
    /** The most tokens each member drafts per round, by position. */
    private final int[] limits;
    private int activeIndex = -1;
    /** The tokens of the last proposal, by position. */
    private int[] lastProposal = new int[0];
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
        this(drafters, unlimited(drafters));
    }

    /**
     * @param drafters the drafter types, in ask order; at least one
     * @param limits   the most tokens each member drafts per round, one positive value per member (a drafter with
     *                 its own cap, such as a DFlash block or chained MTP heads, drafts no more than that cap while
     *                 the others draft the full round)
     * @throws IllegalArgumentException for an empty chain, a null member, or limits that do not match the members
     */
    public DrafterChain(List<DraftModel> drafters, int[] limits) {
        Objects.requireNonNull(drafters, "drafters");
        Objects.requireNonNull(limits, "limits");
        if (limits.length != drafters.size()) {
            throw new IllegalArgumentException(limits.length + " limits for " + drafters.size() + " drafters");
        }
        for (int limit : limits) {
            if (limit < 1) {
                throw new IllegalArgumentException("a drafter limit must be positive, got " + limit);
            }
        }
        this.limits = limits.clone();
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

    private static int[] unlimited(List<DraftModel> drafters) {
        int[] out = new int[Objects.requireNonNull(drafters, "drafters").size()];
        Arrays.fill(out, Integer.MAX_VALUE);
        return out;
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
     * Asks the members in order, each for at most its limit, and takes the first non-empty proposal; a later member
     * drafts only the tokens the earlier members could not (llama.cpp's chain draft).
     */
    @Override
    public int[] draft(int lastToken, int n) {
        return draft(lastToken, n, 1f);
    }

    /**
     * {@link #draft(int, int)} at the request's temperature: a watermarked chain's member drawn under the draft
     * role samples at that temperature; a lookup member's stored proposal ignores it.
     */
    @Override
    public int[] draft(int lastToken, int n, float temperature) {
        requireOpen();
        if (n <= 0) {
            lastProposal = new int[0];
            return new int[0];
        }
        for (int i = 0; i < drafters.size(); i++) {
            int[] proposal = drafters.get(i).draft(lastToken, Math.min(n, limits[i]), temperature);
            if (proposal != null && proposal.length > 0) {
                activeIndex = i;
                lastProposal = proposal;
                return proposal;
            }
        }
        activeIndex = -1;
        lastProposal = new int[0];
        return new int[0];
    }

    @Override
    public boolean supportsWatermarking() {
        for (DraftModel drafter : drafters) {
            if (!drafter.supportsWatermarking()) {
                return false;
            }
        }
        return true;
    }

    /**
     * The active member's proposal distribution, or null for a deterministic proposal (a lookup match, a greedy
     * draft), which the verifier treats as one-hot.
     *
     * @throws IllegalStateException when the chain has no drafted token at {@code position}
     */
    @Override
    public float[] draftProbs(int position) {
        if (activeIndex < 0 || position < 0 || position >= lastProposal.length) {
            throw new IllegalStateException("the chain has no drafted token at position " + position);
        }
        DraftModel member = drafters.get(activeIndex);
        return member instanceof DraftProbs ? ((DraftProbs) member).draftProbs(position) : null;
    }

    @Override
    public void watermark(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig config, int promptLength,
                          long seed) {
        for (DraftModel drafter : drafters) {
            drafter.watermark(config, promptLength, seed);
        }
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

    /**
     * Starts every member's generation over the reused prefix (or the positions a context shift kept) and replays
     * as many stored positions as the member needing the most asks for (a lookup member needs none; a hidden-fed one
     * needs the boundary position).
     */
    @Override
    public int beginGeneration(int reused) {
        requireOpen();
        if (reused < 0) {
            throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
        }
        int replay = 0;
        for (DraftModel drafter : drafters) {
            replay = Math.max(replay, drafter.beginGeneration(reused));
        }
        return replay;
    }

    /** True when any member rebuilds its state from the target's hidden rows. */
    @Override
    public boolean readsTargetHidden() {
        for (DraftModel drafter : drafters) {
            if (drafter.readsTargetHidden()) {
                return true;
            }
        }
        return false;
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
