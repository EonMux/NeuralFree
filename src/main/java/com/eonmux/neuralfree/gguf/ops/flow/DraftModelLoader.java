package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Builds the drafters of speculative decoding, bound to their target (llama.cpp common/speculative.cpp):
 * <ul>
 *   <li>{@link #nextn(EngineModel, MtpHead, Options)}: self-speculation with the target's own NextN/MTP blocks
 *       ({@link NextnDraftModel});</li>
 *   <li>{@link #nextn(EngineModel, ModelSpec, MtpHead, Options)}: the NextN/MTP blocks of an MTP draft file of the
 *       target's model, with or without its trunk;</li>
 *   <li>{@link #fromFile}: a separate drafter model, dispatched on its architecture: eagle3
 *       ({@link Eagle3DraftModel}) and gemma4-assistant ({@link Gemma4AssistantDraftModel}).</li>
 * </ul>
 * The caller supplies the {@link MtpHead} over the target's runtime ({@link MtpHead#gpu} or {@link MtpHead#cpu}),
 * carrying the drafter file's weights when there is one. The returned drafter is bound
 * ({@link DraftModel#bindTarget}) and owns the head: closing the drafter closes it and detaches the drafter from the
 * target's hidden-state tap. When building fails, the head is closed before the exception propagates.
 *
 * <p>Also holds what every drafter shares: {@link Options}, the {@link TargetFeed} that collects the target's hidden
 * rows of each forward call, and the greedy draft pick ({@link #pick}).
 */
public final class DraftModelLoader {

    /** Candidates whose probabilities the draft pick normalizes over: llama.cpp drafters sample with top_k 10. */
    public static final int DRAFT_TOP_K = 10;

    private DraftModelLoader() {
    }

    /** Drafter settings: llama.cpp {@code common_params_speculative_draft.p_min}. Immutable. */
    public static final class Options {
        /** Defaults: drafting continues at any confidence. */
        public static final Options DEFAULT = new Options(0f);

        private final float pMin;

        private Options(float pMin) {
            if (!(pMin >= 0f && pMin <= 1f)) {
                throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
            }
            this.pMin = pMin;
        }

        /** Options with minimum draft-token probability {@code pMin}. */
        public static Options of(float pMin) {
            return new Options(pMin);
        }

        /** The settings of {@code config} ({@code neuralfree.draft.p.min}). */
        public static Options of(EngineConfig config) {
            return new Options(Objects.requireNonNull(config, "config").draftPMin());
        }

        /** Drafting stops at the first token whose top-10 probability is below this. */
        public float pMin() {
            return pMin;
        }

        @Override
        public String toString() {
            return "DraftModelLoader.Options{pMin=" + pMin + '}';
        }
    }

    /**
     * The NextN/MTP self-speculation head of {@code target}: its blocks after the trunk.
     *
     * @throws IllegalArgumentException      when the head does not run {@code target} or the target has no NextN
     *                                       blocks
     * @throws UnsupportedOperationException when the architecture's MTP head cannot run here (see
     *                                       {@link NextnDraftModel})
     */
    public static DraftModel nextn(EngineModel target, MtpHead head, Options options) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(head, "head");
        Objects.requireNonNull(options, "options");
        try {
            checkHead(target, head);
            return bound(new NextnDraftModel(head, options), target);
        } catch (RuntimeException e) {
            Closeables.close(head, e);
            throw e;
        }
    }

    /**
     * The NextN/MTP head of an MTP draft file of the target's model ({@code ModelSpecBuilder.buildMtpDraft}, with or
     * without the file's trunk, which never runs): its blocks, embedding and head over the head's drafter weights,
     * fed with the target's hidden rows, as llama.cpp runs such a file as the draft model of an MTP context
     * (common/speculative.cpp:2655-2690).
     *
     * @throws IllegalArgumentException      when the head does not run {@code target}, the file is not of the
     *                                       target's model, or it has no NextN blocks
     * @throws UnsupportedOperationException when the architecture's MTP head cannot run here (see
     *                                       {@link NextnDraftModel})
     */
    public static DraftModel nextn(EngineModel target, ModelSpec mtp, MtpHead head, Options options) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(mtp, "mtp");
        Objects.requireNonNull(head, "head");
        Objects.requireNonNull(options, "options");
        try {
            checkHead(target, head);
            return bound(new NextnDraftModel(head, mtp, MtpHead.Source.DRAFT, options), target);
        } catch (RuntimeException e) {
            Closeables.close(head, e);
            throw e;
        }
    }

    /**
     * The drafter of a separate model file whose spec is {@code draft}.
     *
     * @param memory memory settings for a drafter with caches of its own (eagle3); its context is the target's
     * @throws IllegalArgumentException      when the head does not run {@code target}, or the drafter is
     *                                       incompatible with it
     * @throws UnsupportedOperationException for a drafter architecture this loader does not build (dflash has its
     *                                       own drafter) or a non-draft model
     */
    public static DraftModel fromFile(EngineModel target, ModelSpec draft, MtpHead head, MemoryConfig memory,
                                      Options options) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(draft, "draft");
        Objects.requireNonNull(head, "head");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(options, "options");
        try {
            checkHead(target, head);
            return bound(create(draft, head, memory, options), target);
        } catch (RuntimeException e) {
            Closeables.close(head, e);
            throw e;
        }
    }

    private static DraftModel create(ModelSpec draft, MtpHead head, MemoryConfig memory, Options options) {
        Arch arch = draft.arch();
        if (arch == Arch.EAGLE3) {
            return new Eagle3DraftModel(head, draft, memory, options);
        }
        if (arch == Arch.GEMMA4_ASSISTANT) {
            return new Gemma4AssistantDraftModel(head, draft, options);
        }
        if (arch == Arch.DFLASH) {
            throw new UnsupportedOperationException("dflash drafts by block diffusion; it has its own drafter");
        }
        throw new UnsupportedOperationException(arch.ggufName() + " is not a draft model this loader builds"
                + " (eagle3, gemma4-assistant)");
    }

    private static void checkHead(EngineModel target, MtpHead head) {
        ModelSpec t = target.spec();
        if (head.target() != t && (head.target().arch() != t.arch() || head.target().nLayer() != t.nLayer())) {
            throw new IllegalArgumentException("the draft head runs " + head.target().arch().ggufName() + " ("
                    + head.target().nLayer() + " layers), not the target " + t.arch().ggufName() + " ("
                    + t.nLayer() + " layers)");
        }
    }

    private static DraftModel bound(DraftModel d, EngineModel target) {
        try {
            d.bindTarget(target);
            return d;
        } catch (RuntimeException e) {
            try {
                d.close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    // ====================================================================== draft pick

    /** A greedy draft token: the top candidate and its probability among the top {@value #DRAFT_TOP_K}. */
    public static final class Pick {
        /** The most likely token id. */
        public final int id;
        /** Its softmax probability over the top {@value #DRAFT_TOP_K} logits. */
        public final float p;

        Pick(int id, float p) {
            this.id = id;
            this.p = p;
        }

        @Override
        public String toString() {
            return "Pick{" + id + ", p=" + p + '}';
        }
    }

    /**
     * The draft token of {@code logits[0 .. n)}: the argmax, with its softmax probability over the
     * {@value #DRAFT_TOP_K} largest logits (llama.cpp drafters: top_k 10, then {@code cur_p[0]}).
     *
     * @throws IllegalStateException when no logit is finite
     */
    public static Pick pick(float[] logits, int n) {
        int k = Math.min(DRAFT_TOP_K, n);
        int[] top = new int[k];
        float[] val = new float[k];
        int filled = 0;
        for (int i = 0; i < n; i++) {
            float v = logits[i];
            if (Float.isNaN(v) || v == Float.NEGATIVE_INFINITY) {
                continue;
            }
            if (filled < k) {
                insert(top, val, filled++, i, v);
            } else if (v > val[k - 1]) {
                insert(top, val, k - 1, i, v);
            }
        }
        if (filled == 0) {
            throw new IllegalStateException("draft logits hold no finite value");
        }
        double sum = 0;
        for (int j = 0; j < filled; j++) {
            sum += Math.exp((double) val[j] - val[0]);
        }
        return new Pick(top[0], (float) (1.0 / sum));
    }

    /** Places (id, v) at or before slot {@code last}, keeping {@code val} descending. */
    private static void insert(int[] top, float[] val, int last, int id, float v) {
        int j = last;
        while (j > 0 && val[j - 1] < v) {
            top[j] = top[j - 1];
            val[j] = val[j - 1];
            j--;
        }
        top[j] = id;
        val[j] = v;
    }

    // ====================================================================== hidden rows by position

    /**
     * The target hidden rows a drafter pairs with tokens, by position: the rows of the last target call (the verify
     * rows), the pending row that the next draft starts from, and the row the last call was bridged from.
     *
     * <p>{@link #record} makes a call's rows the verify rows and its last row pending; {@link #accept} makes verify
     * row {@code min(n, rows - 1)} pending (llama.cpp common/speculative.cpp mtp/eagle3 {@code accept}). {@link #at}
     * finds the row of a position among them, which also serves a call that repeats the previous one's positions (a
     * recurrent target restores its state and replays the accepted tokens from the same start).
     */
    static final class HiddenRows {
        private final int width;
        private float[] verify;
        private int verifyPos0;
        private int verifyRows;
        private final float[] pending;
        private int pendingPos = -1;
        private final float[] bridge;
        private int bridgePos = -1;

        HiddenRows(int width) {
            this.width = width;
            this.pending = new float[width];
            this.bridge = new float[width];
        }

        int width() {
            return width;
        }

        /** The row of position {@code pos}, or null when none is kept. The array must not be modified. */
        float[] at(int pos) {
            if (pos < 0) {
                return null;
            }
            if (verify != null && pos >= verifyPos0 && pos < verifyPos0 + verifyRows) {
                return Arrays.copyOfRange(verify, (pos - verifyPos0) * width, (pos - verifyPos0 + 1) * width);
            }
            if (pos == pendingPos) {
                return pending;
            }
            if (pos == bridgePos) {
                return bridge;
            }
            return null;
        }

        /** Takes the {@code m} rows of a call at {@code pos0 ..}; the array is kept, not copied. */
        void record(int pos0, float[] rows, int m) {
            float[] before = at(pos0 - 1);
            if (before != null) {
                System.arraycopy(before, 0, bridge, 0, width);
                bridgePos = pos0 - 1;
            }
            verify = rows;
            verifyPos0 = pos0;
            verifyRows = m;
            System.arraycopy(rows, (m - 1) * width, pending, 0, width);
            pendingPos = pos0 + m - 1;
        }

        /** Makes verify row {@code min(n, rows - 1)} pending; nothing when no call was recorded. */
        void accept(int n) {
            if (verify == null) {
                return;
            }
            int i = Math.min(n, verifyRows - 1);
            System.arraycopy(verify, i * width, pending, 0, width);
            pendingPos = verifyPos0 + i;
        }

        boolean hasPending() {
            return pendingPos >= 0;
        }

        /** Position of the pending row; -1 when there is none. */
        int pendingPos() {
            return pendingPos;
        }

        /** The pending row, or zeros when there is none. The array must not be modified. */
        float[] pending() {
            return pending;
        }

        void clear() {
            verify = null;
            verifyRows = 0;
            pendingPos = -1;
            bridgePos = -1;
            Arrays.fill(pending, 0f);
            Arrays.fill(bridge, 0f);
        }

        @Override
        public String toString() {
            return "HiddenRows{verify=" + verifyPos0 + "+" + verifyRows + ", pending=" + pendingPos + ", bridge="
                    + bridgePos + '}';
        }
    }

    // ====================================================================== target feed

    /** The rows of one target forward call: positions {@code pos0 ..}, its tokens and the rows per tap point. */
    static final class Batch {
        final int pos0;
        final int[] tokens;
        /** {@code rows[i]} holds the call's rows at the i-th requested tap point, {@code [n][width]}. */
        final float[][] rows;

        Batch(int pos0, int[] tokens, float[][] rows) {
            this.pos0 = pos0;
            this.tokens = tokens;
            this.rows = rows;
        }

        int size() {
            return tokens.length;
        }

        @Override
        public String toString() {
            return "Batch{pos0=" + pos0 + ", n=" + tokens.length + '}';
        }
    }

    /**
     * The drafter's {@link HiddenTap} on the target: collects, per forward call, the rows at the requested tap
     * points with the call's positions and tokens. Forward backends deliver every wanted tap point once per call,
     * after appending the call's tokens to the sequence (CpuForwardBackend, GpuPrefillDriver), so the call covers the
     * last {@code nRows} stored positions. Batches queue until the drafter drains them. A feed may keep only the
     * last rows of each call when the drafter never reads further back.
     */
    static final class TargetFeed implements HiddenTap {
        private final int[] points;
        private final int[] distinct;
        private final int width;
        private final int maxRows;
        private final SequenceState sequence;
        private final List<Batch> pending = new ArrayList<>();
        private float[][] current;
        private int currentRows;
        private int delivered;

        /** A feed that keeps every row of every call. */
        TargetFeed(int[] points, int width, ModelMemory memory) {
            this(points, width, memory, Integer.MAX_VALUE);
        }

        /**
         * @param points  tap points in the order the drafter wants them (repeats allowed)
         * @param width   floats per row at every point
         * @param memory  the target memory whose sequence records each call's tokens
         * @param maxRows rows kept per call, the last ones
         */
        TargetFeed(int[] points, int width, ModelMemory memory, int maxRows) {
            if (points.length == 0) {
                throw new IllegalArgumentException("the feed needs at least one tap point");
            }
            if (maxRows < 1) {
                throw new IllegalArgumentException("the feed must keep at least one row per call, got " + maxRows);
            }
            this.maxRows = maxRows;
            this.points = points.clone();
            this.distinct = Arrays.stream(points).distinct().sorted().toArray();
            this.width = width;
            this.sequence = Objects.requireNonNull(memory, "target memory").sequence();
        }

        @Override
        public boolean wants(int layer) {
            return Arrays.binarySearch(distinct, layer) >= 0;
        }

        @Override
        public void onHidden(int layer, float[] rows, int nRows, int rowWidth) {
            int idx = Arrays.binarySearch(distinct, layer);
            if (idx < 0) {
                throw new IllegalStateException("tap point " + layer + " was not requested");
            }
            if (rowWidth != width) {
                throw new IllegalStateException("tap point " + layer + " delivers rows of " + rowWidth
                        + " floats; the drafter reads " + width);
            }
            if (delivered == 0) {
                current = new float[distinct.length][];
                currentRows = nRows;
            } else if (nRows != currentRows || current[idx] != null) {
                throw new IllegalStateException("tap point " + layer + " delivered out of call order (" + nRows
                        + " rows, call of " + currentRows + ")");
            }
            int kept = Math.min(nRows, maxRows);
            current[idx] = Arrays.copyOfRange(rows, (nRows - kept) * width, Math.multiplyExact(nRows, width));
            if (++delivered == distinct.length) {
                complete();
            }
        }

        private void complete() {
            int n = Math.min(currentRows, maxRows);
            int pos0 = sequence.nPast() - n;
            if (sequence.nPast() < currentRows) {
                throw new IllegalStateException("a forward call of " + currentRows + " rows left "
                        + sequence.nPast() + " stored positions");
            }
            int[] tokens = new int[n];
            for (int t = 0; t < n; t++) {
                tokens[t] = sequence.token(pos0 + t);
            }
            float[][] byPoint = new float[points.length][];
            for (int i = 0; i < points.length; i++) {
                byPoint[i] = current[Arrays.binarySearch(distinct, points[i])];
            }
            pending.add(new Batch(pos0, tokens, byPoint));
            current = null;
            delivered = 0;
        }

        /**
         * Removes and returns the collected batches, oldest first, without those a later call replaced: a call that
         * starts at or before an earlier call's start rewrote all of its positions (a recurrent target restored its
         * state and replayed the accepted tokens).
         */
        List<Batch> drain() {
            if (pending.isEmpty()) {
                return Collections.emptyList();
            }
            List<Batch> out = new ArrayList<>(pending.size());
            int earliestLater = Integer.MAX_VALUE;
            for (int i = pending.size() - 1; i >= 0; i--) {
                Batch b = pending.get(i);
                if (b.pos0 < earliestLater) {
                    out.add(b);
                }
                earliestLater = Math.min(earliestLater, b.pos0);
            }
            pending.clear();
            Collections.reverse(out);
            return out;
        }

        /** Drops every collected batch and any partial call. */
        void clear() {
            pending.clear();
            current = null;
            delivered = 0;
        }

        @Override
        public String toString() {
            return "TargetFeed{points=" + Arrays.toString(points) + ", width=" + width + ", pending=" + pending.size()
                    + '}';
        }
    }
}
