package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Several batch drafters asked in order per sequence (llama.cpp's {@code --spec-type a,b}: the next drafter drafts
 * only while the previous gave nothing): one {@link #draftBatch} call asks the first member for every request, the
 * second for the requests the first left empty, and so on, so each member still drafts its sequences in one call.
 * Per sequence, the member whose proposal the chain returned is the one that hears {@link #accept} and answers
 * {@link #draftProbs}. The chain resets and closes every member. Not thread-safe: it runs on the engine's thread.
 */
final class BatchDrafterChain implements BatchDrafter {

    private final List<BatchDrafter> members;
    /** Per sequence, the member whose proposal the last draft returned, or -1. */
    private final int[] active;

    /**
     * @param members the drafters in ask order, at least one
     * @param maxSequences the sequences the engine runs
     */
    BatchDrafterChain(List<BatchDrafter> members, int maxSequences) {
        Objects.requireNonNull(members, "members");
        if (members.isEmpty()) {
            throw new IllegalArgumentException("a batch drafter chain needs at least one drafter");
        }
        for (BatchDrafter m : members) {
            Objects.requireNonNull(m, "member");
        }
        this.members = Collections.unmodifiableList(new ArrayList<>(members));
        this.active = new int[maxSequences];
        Arrays.fill(active, -1);
    }

    @Override
    public int[] draft(int seq, int lastToken, int n) {
        checkSequence(seq);
        active[seq] = -1;
        for (int i = 0; i < members.size(); i++) {
            int[] proposal = members.get(i).draft(seq, lastToken, n);
            if (proposal != null && proposal.length > 0) {
                active[seq] = i;
                return proposal;
            }
        }
        return new int[0];
    }

    @Override
    public int[][] draftBatch(List<Request> requests) {
        int[][] out = new int[requests.size()][];
        List<Integer> open = new ArrayList<>(requests.size());
        for (int i = 0; i < out.length; i++) {
            checkSequence(requests.get(i).seq());
            active[requests.get(i).seq()] = -1;
            out[i] = new int[0];
            open.add(i);
        }
        for (int m = 0; m < members.size() && !open.isEmpty(); m++) {
            List<Request> asked = new ArrayList<>(open.size());
            for (int i : open) {
                asked.add(requests.get(i));
            }
            int[][] answers = members.get(m).draftBatch(asked);
            if (answers == null || answers.length != asked.size()) {
                throw new IllegalStateException("chain member " + m + " (" + members.get(m) + ") answered "
                        + (answers == null ? "no" : String.valueOf(answers.length)) + " proposals for "
                        + asked.size() + " requests");
            }
            List<Integer> still = new ArrayList<>();
            for (int j = 0; j < answers.length; j++) {
                int i = open.get(j);
                if (answers[j] == null || answers[j].length > 0) {
                    out[i] = answers[j];
                    active[requests.get(i).seq()] = m;
                } else {
                    still.add(i);
                }
            }
            open = still;
        }
        return out;
    }

    @Override
    public float[] draftProbs(int seq, int position) {
        checkSequence(seq);
        if (active[seq] < 0) {
            throw new IllegalStateException("sequence " + seq + " has no drafted token at position " + position);
        }
        return members.get(active[seq]).draftProbs(seq, position);
    }

    /** Reports to the member whose proposal the chain returned; a draft-less round reaches no member. */
    @Override
    public void accept(int seq, int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must not be negative, got " + nAccepted);
        }
        checkSequence(seq);
        if (active[seq] >= 0) {
            members.get(active[seq]).accept(seq, nAccepted);
            active[seq] = -1;
        }
    }

    @Override
    public void reset(int seq) {
        checkSequence(seq);
        active[seq] = -1;
        for (BatchDrafter m : members) {
            m.reset(seq);
        }
    }

    @Override
    public boolean supportsWatermarking() {
        for (BatchDrafter member : members) {
            if (!member.supportsWatermarking()) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean proposesTrees() {
        for (BatchDrafter member : members) {
            if (member.proposesTrees()) {
                return true;
            }
        }
        return false;
    }

    /** Closes every member, the later ones even when an earlier one fails; the first failure is rethrown. */
    @Override
    public void close() {
        RuntimeException first = null;
        for (BatchDrafter m : members) {
            try {
                m.close();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    private void checkSequence(int seq) {
        if (seq < 0 || seq >= active.length) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside the engine's " + active.length
                    + " sequences");
        }
    }

    @Override
    public String toString() {
        return "BatchDrafterChain" + members;
    }
}
