package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Named, bounded corpora of token sequences the prompt-lookup drafters draft from: SGLang's external n-gram
 * corpora ({@code srt/speculative/external_corpus_manager.py}, backed by its sam-defined ngram workers). A corpus is
 * an ordered, append-only (evicting-oldest-when-full) list of token sequences under one name; the server exposes the
 * manager's {@code add}, {@code remove} and {@code list} operations.
 *
 * <p>Drafters share one corpus <em>read-only</em>: {@link #sequences} hands out copies, so callers never write
 * stored arrays, and drafters that built their own caches or trees from the corpus refresh them when
 * {@link #revision()} moves. All methods are thread-safe.
 */
public final class NgramCorpus {

    /** The default per-corpus token budget ({@value #DEFAULT_MAX_TOKENS}). */
    public static final int DEFAULT_MAX_TOKENS = 8_388_608;

    /** One named corpus: its identifier, sequence and token counts. Immutable. */
    public static final class Corpus {
        private final String name;
        private final int sequences;
        private final long tokens;
        private final int maxTokens;

        Corpus(String name, int sequences, long tokens, int maxTokens) {
            this.name = name;
            this.sequences = sequences;
            this.tokens = tokens;
            this.maxTokens = maxTokens;
        }

        /** The corpus name (its {@code corpus_id}). */
        public String name() { return name; }

        /** Number of token sequences in the corpus. */
        public int sequenceCount() { return sequences; }

        /** Number of tokens the corpus stores. */
        public long tokenCount() { return tokens; }

        /** The corpus's token budget. */
        public int maxTokens() { return maxTokens; }

        @Override
        public String toString() {
            return "Corpus{name=" + name + ", sequences=" + sequences + ", tokens=" + tokens
                    + ", maxTokens=" + maxTokens + '}';
        }
    }

    private static final class Entries {
        final Deque<int[]> sequences = new ArrayDeque<>();
        long tokens;

        void append(int[] sequence, int maxTokens) {
            while (!sequences.isEmpty() && tokens + sequence.length > maxTokens) {
                tokens -= sequences.removeFirst().length;
            }
            if (sequence.length > maxTokens) {
                throw new IllegalArgumentException("a corpus sequence of " + sequence.length + " tokens exceeds the"
                        + " corpus budget of " + maxTokens);
            }
            sequences.addLast(sequence.clone());
            tokens += sequence.length;
        }
    }

    private final Map<String, Entries> corpora = new LinkedHashMap<>();
    private final int maxTokens;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    /** Mutation counter behind {@link #revision()}. */
    private long version;

    /**
     * @param maxTokensPerCorpus token budget of every corpus; adding a sequence that does not fit evicts its oldest
     *                           sequences first (FIFO)
     * @throws IllegalArgumentException for a non-positive budget
     */
    public NgramCorpus(int maxTokensPerCorpus) {
        if (maxTokensPerCorpus <= 0) {
            throw new IllegalArgumentException("the corpus token budget must be positive, got " + maxTokensPerCorpus);
        }
        this.maxTokens = maxTokensPerCorpus;
    }

    /** A default set of corpora with the {@link #DEFAULT_MAX_TOKENS} per-corpus budget. */
    public NgramCorpus() {
        this(DEFAULT_MAX_TOKENS);
    }

    /** The per-corpus token budget. */
    public int maxTokens() {
        return maxTokens;
    }

    /**
     * Adds {@code sequence} to the corpus {@code name}, creating it. A sequence that does not fit beside the
     * corpus's current tokens evicts the oldest stored sequences first; a sequence larger than the whole budget is
     * refused instead.
     *
     * @return the corpus's total token count after the add
     * @throws IllegalArgumentException for a blank name, null tokens, an empty or negative-id sequence, or one beyond
     *                                  the budget
     */
    public long add(String name, int[] sequence) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("corpus name is empty");
        }
        requireTokens(sequence);
        lock.writeLock().lock();
        try {
            Entries entries = corpora.computeIfAbsent(name.trim(), n -> new Entries());
            entries.append(sequence, maxTokens);
            version++;
            return entries.tokens;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private static void requireTokens(int[] sequence) {
        Objects.requireNonNull(sequence, "sequence");
        if (sequence.length == 0) {
            throw new IllegalArgumentException("a corpus sequence must not be empty");
        }
        for (int t : sequence) {
            if (t < 0) {
                throw new IllegalArgumentException("a corpus sequence holds negative token id " + t);
            }
        }
    }

    /**
     * Removes the corpus {@code name}. Reports false when no such corpus exists. Drafters read the corpus through
     * their own copies; the removal changes what they see only on their next rebuild ({@link #revision()}).
     */
    public boolean remove(String name) {
        lock.writeLock().lock();
        try {
            if (corpora.remove(name) != null) {
                version++;
                return true;
            }
            return false;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** The corpora, in creation order. */
    public List<Corpus> list() {
        lock.readLock().lock();
        try {
            List<Corpus> out = new ArrayList<>();
            for (Map.Entry<String, Entries> e : corpora.entrySet()) {
                out.add(new Corpus(e.getKey(), e.getValue().sequences.size(), e.getValue().tokens, maxTokens));
            }
            return out;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Copies of the corpus's stored sequences, in add order.
     *
     * @throws IllegalArgumentException when no corpus is named {@code name}
     */
    public int[][] sequences(String name) {
        lock.readLock().lock();
        try {
            Entries entries = corpora.get(Objects.requireNonNull(name, "name"));
            if (entries == null) {
                throw new IllegalArgumentException("no corpus is named " + name + "; stored: " + corpora.keySet());
            }
            int[][] out = new int[entries.sequences.size()][];
            int i = 0;
            for (int[] s : entries.sequences) {
                out[i++] = s.clone();
            }
            return out;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** How many mutations ({@link #add}, {@link #remove}) happened, so drafters can detect changes. */
    public long revision() {
        lock.readLock().lock();
        try {
            return version;
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public String toString() {
        return "NgramCorpus{" + list() + '}';
    }
}
