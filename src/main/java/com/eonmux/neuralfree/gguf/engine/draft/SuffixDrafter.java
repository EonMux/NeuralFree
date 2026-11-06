package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;

import java.util.Arrays;
import java.util.Objects;

/**
 * Suffix decoding (Oliaro et al., arXiv 2411.04975; vLLM {@code suffix_decoding.py} over Arctic Inference's
 * {@code SuffixDecodingCache}, SGLang's suffix decoding worker) over the token history of the sequence it is bound
 * to (the target's, or one batch sequence's {@link TokenHistory}, {@link #bindHistory}), the responses of past
 * requests and the optional shared {@link NgramCorpus}. The drafter keeps three trees:
 * <ul>
 *   <li>the per-request <em>local</em> tree of the generation's committed tokens, prompt and response;</li>
 *   <li>the <em>request</em> tree ({@link SuffixRequestCache}, vLLM {@code suffix_decoding_max_cached_requests}):
 *       the responses of the last {@code maxCachedRequests} generations, the running one included, evicting the
 *       oldest; {@code maxCachedRequests = 0} turns it off. Each {@link #reset()} starts a new request; its response
 *       is every token from the first draft's sampled token on: the sampled tokens, the accepted draft tokens and
 *       the tokens committed between drafts. The tree is the drafter's own, or one shared by the drafters of a
 *       batch's sequences ({@link #of(Options, SuffixRequestCache)}), whose running requests all feed it;</li>
 *   <li>the <em>corpus</em> tree of the corpus sequences, rebuilt when the corpus's revision moves and never written
 *       by the drafter.</li>
 * </ul>
 *
 * <p>Per {@link #draft}: the pattern is the token history's tail up to {@code maxTreeDepth} tokens followed by the
 * just-sampled token ({@code context} of the references); every tree speculates; the draft with the highest score
 * wins (a tie prefers the local tree, then the request tree), bounded by the requested number, the spec factor and
 * offset against the matched length, and {@code minTokenProb}. Committed tokens enter the local tree as they appear
 * in the memory; a history that shrank (rolled back or edited tokens) rebuilds it, so the tree never holds
 * unacquired positions. A history that shrank below the response already cached starts a new request.
 *
 * <p>This drafter installs no hidden states and holds no {@code HiddenTapSlot} lease. Not thread-safe: it runs on the
 * target's forward thread.
 */
public final class SuffixDrafter implements HistoryDrafter {

    /** Default of {@code suffix_decoding_max_tree_depth} (vLLM). */
    public static final int DEFAULT_MAX_TREE_DEPTH = 64;
    /** Default of {@code suffix_decoding_max_spec_factor} (vLLM). */
    public static final double DEFAULT_MAX_SPEC_FACTOR = 1.0;
    /** Default of {@code suffix_decoding_max_spec_offset} (Arctic Inference). */
    public static final double DEFAULT_MAX_SPEC_OFFSET = 0.0;
    /** Default of {@code suffix_decoding_min_token_prob} (vLLM). */
    public static final double DEFAULT_MIN_TOKEN_PROB = 0.1;
    /** Default of {@code suffix_decoding_max_cached_requests} (vLLM). */
    public static final int DEFAULT_MAX_CACHED_REQUESTS = 10000;

    /** The drafter's configuration; immutable after {@link #of}. */
    public static final class Options {
        final int maxTreeDepth;
        final double maxSpecFactor;
        final double maxSpecOffset;
        final double minTokenProb;
        final int maxCachedRequests;
        final NgramCorpus corpus;

        private Options(Builder b) {
            this.maxTreeDepth = b.maxTreeDepth;
            this.maxSpecFactor = b.maxSpecFactor;
            this.maxSpecOffset = b.maxSpecOffset;
            this.minTokenProb = b.minTokenProb;
            this.maxCachedRequests = b.maxCachedRequests;
            this.corpus = b.corpus;
        }

        /** These options restated as a {@link Builder}. */
        public Builder toBuilder() {
            return new Builder().maxTreeDepth(maxTreeDepth).maxSpecFactor(maxSpecFactor).maxSpecOffset(maxSpecOffset)
                    .minTokenProb(minTokenProb).maxCachedRequests(maxCachedRequests).corpus(corpus);
        }

        public static Builder builder() {
            return new Builder();
        }

        public int maxTreeDepth() { return maxTreeDepth; }
        public double maxSpecFactor() { return maxSpecFactor; }
        public double maxSpecOffset() { return maxSpecOffset; }
        public double minTokenProb() { return minTokenProb; }
        /** The requests whose responses the request tree keeps, the running one included; 0 = no request tree. */
        public int maxCachedRequests() { return maxCachedRequests; }
        /** The external corpora the global tree drafts from, shared read-only, or null. */
        public NgramCorpus corpus() { return corpus; }
    }

    /** The drafter's configuration. */
    public static final class Builder {
        private int maxTreeDepth = DEFAULT_MAX_TREE_DEPTH;
        private double maxSpecFactor = DEFAULT_MAX_SPEC_FACTOR;
        private double maxSpecOffset = DEFAULT_MAX_SPEC_OFFSET;
        private double minTokenProb = DEFAULT_MIN_TOKEN_PROB;
        private int maxCachedRequests = DEFAULT_MAX_CACHED_REQUESTS;
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
         * How many tokens per matched context token may be drafted ({@code suffix_decoding_max_spec_factor};
         * {@code >= 0}; default {@value SuffixDrafter#DEFAULT_MAX_SPEC_FACTOR}).
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

        /**
         * The requests whose responses the request tree keeps, the running one included, evicting the oldest
         * ({@code suffix_decoding_max_cached_requests}; {@code >= 0}, 0 turns the request tree off; default
         * {@value SuffixDrafter#DEFAULT_MAX_CACHED_REQUESTS}).
         */
        public Builder maxCachedRequests(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("maxCachedRequests must not be negative, got " + v);
            }
            this.maxCachedRequests = v;
            return this;
        }

        /** The external corpora the corpus tree drafts from (shared read-only with the other drafters). */
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
    /** The request tree, the drafter's own or one shared between drafters; null without one. */
    private final SuffixRequestCache requests;
    /** Whether {@link #requests} is the drafter's own, cleared when it closes. */
    private final boolean ownsRequests;
    /** The id of the drafter's running request in {@link #requests}, or -1 between requests. */
    private int requestId = -1;
    private SuffixTree globalTree;
    private long globalRevision = -1;
    /** What the drafter is bound to, for messages; null while unbound. */
    private String boundTo;
    private TokenHistory history;
    private int lastHistoryLength = -1;
    /** The history position one past the running request's last response token in the request tree. */
    private int responseEnd;
    /** The last returned proposal, whose accepted head joins the response at {@link #accept}; null after it. */
    private int[] lastProposal;
    private boolean closed;

    private SuffixDrafter(Options options, SuffixRequestCache requests, boolean ownsRequests) {
        this.options = options;
        this.localTree = new SuffixTree(options.maxTreeDepth);
        this.requests = requests;
        this.ownsRequests = ownsRequests;
    }

    /** A suffix drafter with {@code options}'s limits and a request tree of its own. */
    public static SuffixDrafter of(Options options) {
        Objects.requireNonNull(options, "options");
        SuffixRequestCache own = options.maxCachedRequests == 0 ? null
                : new SuffixRequestCache(options.maxTreeDepth, options.maxCachedRequests);
        return new SuffixDrafter(options, own, true);
    }

    /**
     * A suffix drafter with {@code options}'s limits whose responses go to the shared request tree
     * {@code requests} and who drafts from every response it holds: one tree for the sequences of a batch, as
     * vLLM's proposer keeps one {@code SuffixDecodingCache}. Closing the drafter ends its request and leaves the
     * shared tree to its owner.
     *
     * @throws IllegalArgumentException when {@code options} turn the request tree off or {@code requests} does not
     *                                  match their depth and capacity
     */
    public static SuffixDrafter of(Options options, SuffixRequestCache requests) {
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(requests, "requests");
        if (options.maxCachedRequests == 0) {
            throw new IllegalArgumentException("maxCachedRequests 0 turns the request tree off; a shared request"
                    + " tree needs a capacity");
        }
        if (requests.maxTreeDepth() != options.maxTreeDepth
                || requests.maxCachedRequests() != options.maxCachedRequests) {
            throw new IllegalArgumentException("the shared request tree (depth " + requests.maxTreeDepth()
                    + ", " + requests.maxCachedRequests() + " requests) does not match the options (depth "
                    + options.maxTreeDepth + ", " + options.maxCachedRequests + " requests)");
        }
        return new SuffixDrafter(options, requests, false);
    }

    /** A request tree for drafters of {@code options} to share ({@link #of(Options, SuffixRequestCache)}). */
    public static SuffixRequestCache sharedRequests(Options options) {
        Objects.requireNonNull(options, "options");
        if (options.maxCachedRequests == 0) {
            throw new IllegalArgumentException("maxCachedRequests 0 turns the request tree off");
        }
        return new SuffixRequestCache(options.maxTreeDepth, options.maxCachedRequests);
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
        requireUnbound();
        ForwardBackend fb = target.forwardBackend();
        if (fb == null || fb.memory() == null) {
            throw new IllegalArgumentException("a suffix drafter needs a target with model memory; "
                    + target.name() + " (" + target.architecture() + ") has none");
        }
        bind(target.name(), () -> fb.memory().sequence().tokens());
    }

    /** @throws IllegalStateException when the drafter is closed or already bound */
    @Override
    public void bindHistory(TokenHistory history) {
        Objects.requireNonNull(history, "history");
        requireUnbound();
        bind("a sequence history", history);
    }

    private void requireUnbound() {
        requireOpen();
        if (history != null) {
            throw new IllegalStateException("the suffix drafter is already bound to " + boundTo);
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("the suffix drafter is closed");
        }
    }

    /** Binds the drafter to {@code history} and builds the global tree of its corpus. */
    private void bind(String name, TokenHistory history) {
        this.boundTo = name;
        this.history = history;
        if (options.corpus != null) {
            globalTree = new SuffixTree(options.maxTreeDepth);
            rebuildGlobalTree();
        }
    }

    /** Rebuilds the corpus tree from every corpus sequence (a read-only spread of the shared corpus). */
    private void rebuildGlobalTree() {
        globalTree.clear();
        NgramCorpus corpus = options.corpus;
        int sequenceId = FIRST_CORPUS_SEQUENCE;
        for (NgramCorpus.Corpus c : corpus.list()) {
            for (int[] sequence : corpus.sequences(c.name())) {
                // the whole sequence: the tree's depth cap bounds every entered suffix
                globalTree.extend(sequenceId++, sequence);
            }
        }
        globalRevision = corpus.revision();
    }

    @Override
    public int[] draft(int lastToken, int n) {
        if (closed || history == null) {
            throw new IllegalStateException("the suffix drafter is not bound to a target");
        }
        if (n <= 0 || lastToken < 0) {
            return new int[0];
        }
        int[] tokens = history.tokens();
        int len = tokens.length;
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
            int[] appended = Arrays.copyOfRange(tokens, lastHistoryLength, len);
            localTree.extend(LOCAL_SEQUENCE, appended);
            lastHistoryLength = len;
        }
        // The pattern: the depth-capped tail of the history, then the sampled token.
        int patternStart = Math.max(0, len + 1 - options.maxTreeDepth);
        int[] pattern = new int[len + 1 - patternStart];
        System.arraycopy(tokens, patternStart, pattern, 0, len - patternStart);
        pattern[pattern.length - 1] = lastToken;

        // The limits are the requested count per tree, then the best score wins (a tie prefers the earlier tree).
        SuffixTree.Draft chosen = localTree.speculate(pattern, n, options.maxSpecFactor, options.maxSpecOffset,
                options.minTokenProb);
        if (requests != null) {
            recordResponse(tokens, len, lastToken);
            chosen = better(chosen, requests.speculate(pattern, n, options.maxSpecFactor, options.maxSpecOffset,
                    options.minTokenProb));
        }
        if (globalTree != null) {
            if (options.corpus.revision() != globalRevision) {
                rebuildGlobalTree();
            }
            chosen = better(chosen, globalTree.speculate(pattern, n, options.maxSpecFactor, options.maxSpecOffset,
                    options.minTokenProb));
        }
        int[] out = Arrays.copyOf(chosen.tokenIds(), Math.min(chosen.tokenIds().length, n));
        lastProposal = requests != null ? out : null;
        return out;
    }

    private static SuffixTree.Draft better(SuffixTree.Draft kept, SuffixTree.Draft other) {
        return other.score() > kept.score() ? other : kept;
    }

    /**
     * Brings the running request's response up to the sampled {@code lastToken} at history position {@code len}:
     * the tokens committed since the response's end (decoded without a draft) and then the sampled token, as
     * Arctic Inference's {@code add_active_response} adds each step's sampled tokens before it speculates. The first
     * draft after a reset starts a new request there, and so does a history that shrank below the cached response.
     */
    private void recordResponse(int[] tokens, int len, int lastToken) {
        lastProposal = null;
        if (requestId < 0 || len < responseEnd) {
            endRequest();
            requestId = requests.start();
            responseEnd = len;
        }
        if (responseEnd < len) {
            requests.append(requestId, Arrays.copyOfRange(tokens, responseEnd, len));
        }
        requests.append(requestId, lastToken);
        responseEnd = len + 1;
    }

    private void endRequest() {
        if (requestId >= 0) {
            requests.end(requestId);
            requestId = -1;
        }
    }

    /** The requests the request tree holds, the running ones included; 0 without a request tree. */
    public int cachedRequests() {
        return requests == null ? 0 : requests.cachedRequests();
    }

    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("the accepted count must not be negative, got " + nAccepted);
        }
        // The local tree only holds committed tokens; the accepted head of the proposal joins the response.
        int[] proposal = lastProposal;
        lastProposal = null;
        int kept = proposal == null ? 0 : Math.min(nAccepted, proposal.length);
        if (kept > 0 && requestId >= 0) {
            requests.append(requestId, Arrays.copyOf(proposal, kept));
            responseEnd += kept;
        }
    }

    @Override
    public void reset() {
        if (closed) {
            return;
        }
        localTree.clear();
        lastHistoryLength = -1;
        lastProposal = null;
        if (requests != null) {
            endRequest();
        }
    }

    @Override
    public void close() {
        closed = true;
        boundTo = null;
        history = null;
        localTree.clear();
        lastProposal = null;
        if (requests != null) {
            endRequest();
            if (ownsRequests) {
                requests.clear();
            }
        }
        if (globalTree != null) {
            globalTree.clear();
            globalTree = null;
        }
    }

    @Override
    public String toString() {
        return "SuffixDrafter{maxTreeDepth=" + options.maxTreeDepth
                + (requests != null ? ", requests=" + requests : "")
                + (options.corpus != null ? ", corpus=" + options.corpus : "") + '}';
    }
}
