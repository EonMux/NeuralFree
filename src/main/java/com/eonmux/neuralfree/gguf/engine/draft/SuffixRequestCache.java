package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * The global suffix tree of suffix decoding: one {@link SuffixTree} holding the responses of the past requests and of
 * the running ones, as Arctic Inference's {@code SuffixDecodingCache} keeps its {@code _global_tree} (vLLM
 * {@code suffix_decoding_max_cached_requests}). Each request is one sequence of the tree, named by the id
 * {@link #start()} returns. Several requests may run at once: the batch engine shares one cache between the
 * {@link SuffixDrafter}s of its sequences, so every sequence drafts from the responses of all of them, as vLLM's
 * proposer keeps one cache for the whole batch. Starting a request beyond the cache's capacity evicts the oldest
 * cached requests in FIFO order, never the request just started; an evicted request that is still running stops
 * adding to the tree, as Arctic Inference extends only the cached responses. Prompts are not cached; they live in the
 * per-request tree of {@link SuffixDrafter}.
 *
 * <p>Not thread-safe: it runs on the target's forward thread with its drafters.
 */
public final class SuffixRequestCache {

    private final SuffixTree tree;
    private final int maxTreeDepth;
    private final int maxCachedRequests;
    /** The cached request ids, oldest first. */
    private final Set<Integer> cached = new LinkedHashSet<>();
    /** The ids started and not yet ended, cached or evicted. */
    private final Set<Integer> running = new HashSet<>();
    private int nextId;

    /**
     * @param maxTreeDepth      the suffix depth of the tree ({@code suffix_decoding_max_tree_depth})
     * @param maxCachedRequests the requests the tree keeps, the running ones included
     * @throws IllegalArgumentException for a capacity below 1 or a depth below 1
     */
    public SuffixRequestCache(int maxTreeDepth, int maxCachedRequests) {
        if (maxCachedRequests < 1) {
            throw new IllegalArgumentException("a request cache holds at least 1 request, got " + maxCachedRequests);
        }
        this.tree = new SuffixTree(maxTreeDepth);
        this.maxTreeDepth = maxTreeDepth;
        this.maxCachedRequests = maxCachedRequests;
    }

    /** The suffix depth of the tree. */
    public int maxTreeDepth() {
        return maxTreeDepth;
    }

    /** The requests the tree keeps, the running ones included. */
    public int maxCachedRequests() {
        return maxCachedRequests;
    }

    /**
     * Starts a new request and evicts the oldest cached requests beyond the capacity.
     *
     * @return the request's id, for {@link #append} and {@link #end}
     */
    int start() {
        int id = nextId;
        // the next non-negative id; after a wrap-around, a still cached request of the same id leaves first
        nextId = (nextId + 1) & Integer.MAX_VALUE;
        if (cached.remove(id)) {
            tree.remove(id);
        }
        cached.add(id);
        running.add(id);
        Iterator<Integer> oldest = cached.iterator();
        while (cached.size() > maxCachedRequests) {
            int evicted = oldest.next();
            oldest.remove();
            tree.remove(evicted);
        }
        return id;
    }

    /** Whether request {@code id} is running: started and not yet {@linkplain #end ended}. */
    boolean running(int id) {
        return running.contains(id);
    }

    /**
     * Appends response tokens to the running request {@code id}; a no-op once the request was evicted.
     *
     * @throws IllegalStateException when the request is not running
     */
    void append(int id, int[] tokens) {
        if (!running.contains(id)) {
            throw new IllegalStateException("request " + id + " is not running in the suffix request cache");
        }
        if (cached.contains(id)) {
            tree.extend(id, tokens);
        }
    }

    /** Appends one response token to the running request {@code id}. */
    void append(int id, int token) {
        append(id, new int[]{token});
    }

    /** Ends request {@code id}; its response stays cached until evicted. A no-op for a request not running. */
    void end(int id) {
        running.remove(id);
    }

    /** The cached requests, the running ones included. */
    public int cachedRequests() {
        return cached.size();
    }

    /** {@link SuffixTree#speculate} over the cached responses. */
    SuffixTree.Draft speculate(int[] pattern, int maxSpecTokens, double maxSpecFactor, double maxSpecOffset,
                               double minTokenProb) {
        return tree.speculate(pattern, maxSpecTokens, maxSpecFactor, maxSpecOffset, minTokenProb);
    }

    /** Drops every cached request and ends the running ones. */
    void clear() {
        tree.clear();
        cached.clear();
        running.clear();
    }

    @Override
    public String toString() {
        return "SuffixRequestCache{" + cached.size() + "/" + maxCachedRequests + " requests}";
    }
}
