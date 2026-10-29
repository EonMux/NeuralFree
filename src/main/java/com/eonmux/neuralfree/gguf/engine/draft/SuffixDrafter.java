package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Suffix decoding (Oliaro et al., arXiv 2411.04975; vLLM {@code suffix_decoding.py}, SGLang's suffix decoding
 * worker) over the target's own token history and the optional shared {@link NgramCorpus}: the drafter builds a
 * per-request <em>local</em> suffix tree from the generation's committed tokens and drafts from the best of its and
 * the <em>global</em> tree's speculations (the corpus sequences; the corpus's revision moves rebuild the global tree,
 * and the drafter never writes it).
 *
 * <p>Per {@link #draft}: the pattern is the token history's tail up to {@code maxTreeDepth} tokens followed by the
 * just-sampled token ({@code context} of the references); both trees speculate; the draft with the higher score wins
 * (a tie prefers the local one), bounded by the requested number, the spec factor and offset against the matched
 * length, and {@code minTokenProb}. Committed tokens enter the local tree as they appear in the memory; a history
 * that shrank (rolled back or edited tokens) rebuilds it, so the tree never holds unaquired positions.
 *
 * <p>This drafter installs no hidden states and holds no {@code HiddenTapSlot} lease. Not thread-safe: it runs on the
 * target's forward thread.
 */
public final class SuffixDrafter implements DraftModel {

    /** Default of {@code suffix_decoding_max_tree_depth} (vLLM). */
    public static final int DEFAULT_MAX_TREE_DEPTH = 64;
    /** Default of {@code suffix_decoding_max_spec_factor} (vLLM). */
    public static final double DEFAULT_MAX_SPEC_FACTOR = 1.0;
    /** Default of {@code suffix_decoding_max_spec_offset} (Arctic Inference). */
    public static final double DEFAULT_MAX_SPEC_OFFSET = 0.0;
    /** Default of {@code suffix_decoding_min_token_prob} (vLLM). */
    public static final double DEFAULT_MIN_TOKEN_PROB = 0.1;

    /** The drafter's configuration; immutable after {@link #of}. */
    public static final class Options {
        final int maxTreeDepth;
        final double maxSpecFactor;
        final double maxSpecOffset;
        final double minTokenProb;
        final NgramCorpus corpus;

        private Options(Builder b) {
            this.maxTreeDepth = b.maxTreeDepth;
            this.maxSpecFactor = b.maxSpecFactor;
            this.maxSpecOffset = b.maxSpecOffset;
            this.minTokenProb = b.minTokenProb;
            this.corpus = b.corpus;
        }

        public static Builder builder() {
            return new Builder();
        }

        public int maxTreeDepth() { return maxTreeDepth; }
        public double maxSpecFactor() { return maxSpecFactor; }
        public double maxSpecOffset() { return maxSpecOffset; }
        public double minTokenProb() { return minTokenProb; }
        /** The external corpora the global tree drafts from, shared read-only, or null. */
        public NgramCorpus corpus() { return corpus; }
    }

    /** The drafter's configuration. */
    public static final class Builder {
        private int maxTreeDepth = DEFAULT_MAX_TREE_DEPTH;
        private double maxSpecFactor = DEFAULT_MAX_SPEC_FACTOR;
        private double maxSpecOffset = DEFAULT_MAX_SPEC_OFFSET;
        private double minTokenProb = DEFAULT_MIN_TOKEN_PROB;
        private NgramCorpus corpus;

        /** The pattern length cap and suffix depth ({@code suffix_decoding_max_tree_depth};
         * default {@value SuffixDrafter#DEFAULT_MAX_TREE_DEPTH}). */
        public Builder maxTreeDepth(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("maxTreeDepth must be at least 1, got " + v);
            }
            this.maxTreeDepth = v;
            return this;
        }

        /**
         * How many tokens per matched context token may be drafted ({@code suffix_decoding_max_spec_factor}; less
         * than 0 means no limit; default {@value SuffixDrafter#DEFAULT_MAX_SPEC_FACTOR}).
         */
        public Builder maxSpecFactor(double v) {
            this.maxSpecFactor = v;
            return this;
        }

        /** Draft tokens added to the factor's budget ({@code max_spec_offset}; default
         * {@value SuffixDrafter#DEFAULT_MAX_SPEC_OFFSET}). */
        public Builder maxSpecOffset(double v) {
            this.maxSpecOffset = v;
            return this;
        }

        /** The smallest child probability a draft keeps ({@code suffix_decoding_min_token_prob}; in [0, 1]). */
        public Builder minTokenProb(double v) {
            if (!(v >= 0.0 && v <= 1.0)) {
                throw new IllegalArgumentException("minTokenProb must be in [0, 1], got " + v);
            }
            this.minTokenProb = v;
            return this;
        }

        /** The external corpora the global tree drafts from (shared read-only with the other drafters). */
        public Builder corpus(NgramCorpus v) {
            this.corpus = v;
            return this;
        }

        /**
         * @throws IllegalArgumentException for a non-finite factor or offset
         */
        public Options build() {
            if (!Double.isFinite(maxSpecFactor) || maxSpecFactor < 0) {
                throw new IllegalArgumentException("maxSpecFactor must be finite and >= 0, got " + maxSpecFactor);
            }
            if (!Double.isFinite(maxSpecOffset) || maxSpecOffset < 0) {
                throw new IllegalArgumentException("maxSpecOffset must be finite and >= 0, got " + maxSpecOffset);
            }
            return new Options(this);
        }
    }

    /** The suffix ids the drafter's trees use: the local generation tree and the corpus-backed global one. */
    private static final int LOCAL_SEQUENCE = 0;
    private static final int FIRST_CORPUS_SEQUENCE = 1;

    private final Options options;
    private final SuffixTree localTree;
    private SuffixTree globalTree;
    private long globalRevision = -1;
    private EngineModel target;
    private ForwardBackend backend;
    private int lastHistoryLength = -1;
    private boolean closed;

    private SuffixDrafter(Options options) {
        this.options = options;
        this.localTree = new SuffixTree(options.maxTreeDepth);
    }

    /** A suffix drafter with {@code options}'s limits. */
    public static SuffixDrafter of(Options options) {
        Objects.requireNonNull(options, "options");
        return new SuffixDrafter(options);
    }

    /** A suffix drafter with the default limits. */
    public static SuffixDrafter of() {
        return of(Options.builder().build());
    }

    /** The drafter's limits. */
    public Options options() {
        return options;
    }

    /**
     * @throws IllegalArgumentException when the target cannot run a lookup drafter: it has no model memory
     * @throws IllegalStateException    when the drafter is closed or already bound
     */
    @Override
    public void bindTarget(EngineModel target) {
        Objects.requireNonNull(target, "target");
        if (closed) {
            throw new IllegalStateException("the suffix drafter is closed");
        }
        if (this.target != null) {
            throw new IllegalStateException("the suffix drafter is already bound to a target");
        }
        ForwardBackend fb = target.forwardBackend();
        if (fb == null || fb.memory() == null) {
            throw new IllegalArgumentException("a suffix drafter needs a target with model memory; "
                    + target.name() + " (" + target.architecture() + ") has none");
        }
        this.target = target;
        this.backend = fb;
        if (options.corpus != null) {
            globalTree = new SuffixTree(options.maxTreeDepth);
            rebuildGlobalTree();
        }
    }

    /** Extends the global tree with every corpus sequence (a read-only spread of the shared corpus). */
    private void rebuildGlobalTree() {
        globalTree.clear();
        NgramCorpus corpus = options.corpus;
        int sequenceId = FIRST_CORPUS_SEQUENCE;
        for (NgramCorpus.Corpus c : corpus.list()) {
            for (int[] sequence : corpus.sequences(c.name())) {
                if (sequence.length > options.maxTreeDepth) {
                    // Only the deepest suffixes of the depth cap would ever be queried; the shorter ones cover the
                    // matches. Enter the whole sequence: the cap handles the depth.
                    sequence = java.util.Arrays.copyOfRange(sequence, sequence.length - options.maxTreeDepth,
                            sequence.length);
                }
                globalTree.extend(sequenceId++, sequence);
            }
        }
        globalRevision = corpus.revision();
    }

    @Override
    public int[] draft(int lastToken, int n) {
        if (closed || target == null) {
            throw new IllegalStateException("the suffix drafter is not bound to a target");
        }
        if (n <= 0 || lastToken < 0) {
            return new int[0];
        }
        SequenceState seq = backend.memory().sequence();
        int len = seq.nPast();
        if (len < lastHistoryLength) {
            // The generation's history shrank: a fresh local tree from the current tokens.
            localTree.clear();
            lastHistoryLength = 0;
        }
        if (lastHistoryLength < 0) {
            // First call of a generation (or right after a reset): the tree is empty, enter the full history.
            lastHistoryLength = 0;
        }
        if (lastHistoryLength < len) {
            int[] appended = new int[len - lastHistoryLength];
            for (int i = 0; i < appended.length; i++) {
                appended[i] = seq.token(lastHistoryLength + i);
            }
            localTree.extend(LOCAL_SEQUENCE, appended);
            lastHistoryLength = len;
        }
        // The pattern: the depth-capped tail of the history, then the sampled token.
        int patternStart = Math.max(0, len + 1 - options.maxTreeDepth);
        int[] pattern = new int[len + 1 - patternStart];
        for (int i = patternStart; i < len; i++) {
            pattern[i - patternStart] = seq.token(i);
        }
        pattern[pattern.length - 1] = lastToken;

        // The limits are the requested count per tree, then the best score wins (a tie prefers local).
        SuffixTree.Draft local = localTree.speculate(pattern, n, options.maxSpecFactor, options.maxSpecOffset,
                options.minTokenProb);
        SuffixTree.Draft global = null;
        if (globalTree != null) {
            if (options.corpus.revision() != globalRevision) {
                rebuildGlobalTree();
            }
            global = globalTree.speculate(pattern, n, options.maxSpecFactor, options.maxSpecOffset,
                    options.minTokenProb);
        }
        SuffixTree.Draft chosen = global != null && global.score() > local.score() ? global : local;
        int depth = Math.min(chosen.tokenIds().length, n);
        int[] out = new int[depth];
        System.arraycopy(chosen.tokenIds(), 0, out, 0, depth);
        return out;
    }

    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("the accepted count must not be negative, got " + nAccepted);
        }
        // The tree only holds committed tokens; verification needs no correction here.
    }

    @Override
    public void reset() {
        if (closed) {
            return;
        }
        localTree.clear();
        lastHistoryLength = -1;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("the suffix drafter is closed");
        }
    }

    @Override
    public void close() {
        closed = true;
        target = null;
        backend = null;
        localTree.clear();
        if (globalTree != null) {
            globalTree.clear();
            globalTree = null;
        }
    }

    @Override
    public String toString() {
        return "SuffixDrafter{maxTreeDepth=" + options.maxTreeDepth
                + (options.corpus != null ? ", global=" + options.corpus : "") + '}';
    }
}
