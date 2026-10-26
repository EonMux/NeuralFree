package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Prompt-lookup drafting from the target's own token history (vLLM {@code ngram_proposer.py}, llama.cpp
 * {@code ngram-simple}) and SGLang's external corpora, a port of llama.cpp's no-draft-model implementations
 * ({@code common_ngram_simple_draft}, {@code common_ngram_map_draft}, {@code common_ngram_mod} and
 * {@code common_ngram_cache_draft}, common/ngram-map.cpp and common/ngram-cache.cpp: llama.cpp's
 * {@code ngram-simple}, {@code ngram-map-k}, {@code ngram-map-k4v}, {@code ngram-mod} and {@code ngram-cache}
 * speculative types).
 *
 * <p>Every {@link #draft} call reads the target memory's stored tokens, appends the just-sampled token and looks the
 * resulting n-grams up:
 * <ul>
 *   <li>{@link Kind#SIMPLE}: searches the history backwards for the pattern of the last {@code sizeN - 1} history
 *       tokens plus the sampled token (the {@code n_draft_min} n-gram) and proposes the up to {@code sizeM}
 *       ({@code size_mgram}) tokens that followed the match, at least {@code sizeN} of them or none at all.</li>
 *   <li>{@link Kind#MAP_K}: llama.cpp's n-gram map, keys only: the most recent key n-gram hit proposes the
 *       {@code sizeM} tokens that followed it.</li>
 *   <li>{@link Kind#MAP_K4V}: the map, keys and values: per key n-gram it keeps up to {@value #MAP_MAX_VALUES} value
 *       m-grams and proposes the most frequent one only when it dominates ({@code max >= 2 * sum} of the others) and
 *       the key reached {@code minHits} hits.</li>
 *   <li>{@link Kind#MOD}: the hashed n-gram index of llama.cpp's {@code common_ngram_mod}: the token that last
 *       followed the hashed key n-gram, chained one proposal after another while the table keeps offering tokens, at
 *       most {@code sizeM} of them.</li>
 *   <li>{@link Kind#CACHE}: llama.cpp's cache chain. The context part ({@code nc_context}) is built from the
 *       generation's committed tokens, the dynamic part ({@code nc_dynamic}) from an optional preloaded
 *       {@link NgramCache} and the generation's committed tokens, and the static part ({@code nc_static}) from an
 *       optional preloaded {@link NgramCache} ({@code NgramCacheFile#load}) together with the optional
 *       {@link NgramCorpus}, which is shared read-only with the other drafters and refreshed when the corpus
 *       revision moves. Every step offers the context candidates over {@code ngramMin}..{@code ngramMax}-grams with
 *       llama.cpp's lax sample-size and percentage thresholds, the dynamic ones with the strict thresholds (static
 *       cache counts weighting both, as llama.cpp weights them), then the static 2-gram's best token; a step without
 *       a candidate ends the draft.</li>
 * </ul>
 * A history that shrank between two calls (verified tokens were rolled back, or the context itself was edited) is a
 * new history for the stateful kinds: the map, the hash table and the context cache are dropped, as llama.cpp
 * documents "a complete rebuild" after a context change, so no entry references a dropped position. A history that
 * only grew adds entries ({@link NgramCache#update}), as llama.cpp's caches only append.
 *
 * <p>This drafter installs no hidden states and holds no {@code HiddenTapSlot} lease, so several n-gram drafters can
 * run beside each other and with the hidden-fed drafters through {@link DrafterChain}. Not thread-safe: it runs on the
 * target's forward thread.
 */
public final class NgramDrafter implements DraftModel {

    /** Which lookup the drafter drafts with ({@link NgramDrafter} class doc). Immutable per drafter. */
    public enum Kind {
        SIMPLE,
        MAP_K,
        MAP_K4V,
        MOD,
        CACHE
    }

    /** The m-gram values stored for one key n-gram ({@code COMMON_NGRAM_MAX_VALUES}). */
    static final int MAP_MAX_VALUES = 4;
    /** Largest counted key or value hits ({@code COMMON_NGRAM_MAX_VALUE_COUNT}). */
    static final int MAP_MAX_VALUE_COUNT = 16380;
    /** Entries of the map's hash-to-key-index array ({@code COMMON_NGRAM_HASH_MAP_SIZE}). */
    static final int MAP_HASH_MAP_SIZE = 262144;
    /** Hash prime of a map key n-gram ({@code LCG_FACTOR}, near (sqrt(5) - 1)/2 * 2^32). */
    static final long MAP_LCG_FACTOR = 2654435761L;
    /** Hash constant of {@code common_ngram_mod::idx}. */
    static final long MOD_HASH_FACTOR = 6364136223846793005L;
    /** N-gram size of the static-cache candidate and fallback ({@code LLAMA_NGRAM_STATIC}). */
    static final int CACHE_STATIC_NGRAM = 2;
    /** llama.cpp's cache draft thresholds, indexed by n-gram size - 1 (common/ngram-cache.cpp). */
    static final int[] CACHE_MIN_SAMPLE_LAX = {2, 2, 1, 1};
    static final int[] CACHE_MIN_PERCENT_LAX = {66, 50, 50, 50};
    static final int[] CACHE_MIN_SAMPLE_STRICT = {4, 3, 2, 2};
    static final int[] CACHE_MIN_PERCENT_STRICT = {75, 66, 66, 66};

    /** Defaults the drafter's sizes fall back to. */
    public static final int DEFAULT_SIZE_N = 3;
    public static final int DEFAULT_SIZE_M = 8;
    public static final int DEFAULT_MIN_HITS = 1;
    public static final int DEFAULT_MOD_TABLE_SIZE = 1 << 20;
    public static final int DEFAULT_NGRAM_MIN = 1;
    public static final int DEFAULT_NGRAM_MAX = 4;

    private static final int MAX_SIZE_N = 64;
    private static final int MAX_SIZE_M = 128;
    private static final int MAX_MOD_TABLE = 1 << 28;
    private static final int EMPTY = -1;

    /** Everything a drafter needs besides its target: immutable after {@link Builder#build()}. */
    public static final class Options {
        final Kind kind;
        final int sizeN;
        final int sizeM;
        final int minHits;
        final int ngramMin;
        final int ngramMax;
        final int modTableSize;
        final int draftMin;
        final NgramCache dynamicCache;
        final NgramCache staticCache;
        final NgramCorpus corpus;

        private Options(Builder b) {
            this.kind = b.kind;
            this.sizeN = b.sizeN;
            this.sizeM = b.sizeM;
            this.minHits = b.minHits;
            this.ngramMin = b.ngramMin;
            this.ngramMax = b.ngramMax;
            this.modTableSize = b.modTableSize;
            this.draftMin = b.draftMin;
            this.dynamicCache = b.dynamicCache;
            this.staticCache = b.staticCache;
            this.corpus = b.corpus;
        }

        public static Builder builder(Kind kind) {
            return new Builder(kind);
        }

        /** This option set restated as a {@link Builder}. */
        public Builder toBuilder() {
            return new Builder(kind).sizeN(sizeN).sizeM(sizeM).minHits(minHits).ngramMin(ngramMin)
                    .ngramMax(ngramMax).modTableSize(modTableSize).draftMin(draftMin)
                    .dynamicCache(dynamicCache).staticCache(staticCache).corpus(corpus);
        }

        public Kind kind() { return kind; }

        /** The key n-gram size: the search n-gram of {@link Kind#SIMPLE}, the key size of the map kinds, the hashed
         * n-gram size of {@link Kind#MOD}; unused by {@link Kind#CACHE}. */
        public int sizeN() { return sizeN; }

        /** The proposal size: the m-gram of {@link Kind#SIMPLE} and the map kinds, the proposal cap of
         * {@link Kind#MOD} and {@link Kind#CACHE}. */
        public int sizeM() { return sizeM; }

        /** Key hits needed before a map proposal ({@code min_hits}). */
        public int minHits() { return minHits; }

        public int ngramMin() { return ngramMin; }
        public int ngramMax() { return ngramMax; }
        public int modTableSize() { return modTableSize; }
        public int draftMin() { return draftMin; }
        public NgramCache dynamicCache() { return dynamicCache; }
        public NgramCache staticCache() { return staticCache; }
        public NgramCorpus corpus() { return corpus; }

        @Override
        public String toString() {
            return "NgramOptions{kind=" + kind + ", sizeN=" + sizeN + ", sizeM=" + sizeM + ", minHits=" + minHits
                    + (kind == Kind.CACHE ? ", ngram=[" + ngramMin + ", " + ngramMax + "]" : "")
                    + ", draftMin=" + draftMin
                    + (dynamicCache != null ? ", dynamicCache=" + dynamicCache.ngrams() + " n-grams" : "")
                    + (staticCache != null ? ", staticCache=" + staticCache.ngrams() + " n-grams" : "")
                    + (corpus != null ? ", corpus=" + corpus : "") + '}';
        }
    }

    /** The drafter's configuration. */
    public static final class Builder {
        private final Kind kind;
        private int sizeN = -1;
        private int sizeM = DEFAULT_SIZE_M;
        private int minHits = DEFAULT_MIN_HITS;
        private int ngramMin = DEFAULT_NGRAM_MIN;
        private int ngramMax = DEFAULT_NGRAM_MAX;
        private int modTableSize = DEFAULT_MOD_TABLE_SIZE;
        private int draftMin;
        private NgramCache dynamicCache;
        private NgramCache staticCache;
        private NgramCorpus corpus;

        private Builder(Kind kind) {
            this.kind = Objects.requireNonNull(kind, "kind");
        }

        /** The key n-gram size ({@code sizeN}); defaults to {@value NgramDrafter#DEFAULT_SIZE_N}. */
        public Builder sizeN(int v) {
            requireInRange(v, 1, MAX_SIZE_N, "sizeN");
            this.sizeN = v;
            return this;
        }

        /** The proposal size ({@code sizeM}); defaults to {@value NgramDrafter#DEFAULT_SIZE_M}. */
        public Builder sizeM(int v) {
            requireInRange(v, 1, MAX_SIZE_M, "sizeM");
            this.sizeM = v;
            return this;
        }

        /** Key hits before a map proposal ({@code min_hits}, {@link Kind#MAP_K4V} only; default 1). */
        public Builder minHits(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("minHits must be positive, got " + v);
            }
            this.minHits = v;
            return this;
        }

        /** The cache lookup's smallest n-gram size ({@code ngram_min}). */
        public Builder ngramMin(int v) {
            requireInRange(v, 1, NgramCache.MAX_KEY, "ngramMin");
            this.ngramMin = v;
            return this;
        }

        /** The cache lookup's largest n-gram size ({@code ngram_max}). */
        public Builder ngramMax(int v) {
            requireInRange(v, 1, NgramCache.MAX_KEY, "ngramMax");
            this.ngramMax = v;
            return this;
        }

        /** The entry count of the {@link Kind#MOD} hash table; default {@value NgramDrafter#DEFAULT_MOD_TABLE_SIZE}. */
        public Builder modTableSize(int v) {
            if (v < 1 || v > MAX_MOD_TABLE) {
                throw new IllegalArgumentException("modTableSize must be in [1, " + MAX_MOD_TABLE + "], got " + v);
            }
            this.modTableSize = v;
            return this;
        }

        /** Proposals shorter than this return an empty draft (default 0). */
        public Builder draftMin(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("draftMin must not be negative, got " + v);
            }
            this.draftMin = v;
            return this;
        }

        /**
         * Preloaded {@code nc_dynamic} cache, llama.cpp's cache of earlier generations. The drafter copies it at
         * {@link #bindTarget} and refreshes it with the generation's committed tokens.
         */
        public Builder dynamicCache(NgramCache v) {
            this.dynamicCache = v;
            return this;
        }

        /** Preloaded {@code nc_static} cache (llama.cpp {@code ngram_cache_load}, typically built by a cache file). */
        public Builder staticCache(NgramCache v) {
            this.staticCache = v;
            return this;
        }

        /**
         * External corpora (SGLang's external corpus manager) the drafter reads read-only; the corpora's token
         * sequences enter the static cache.
         */
        public Builder corpus(NgramCorpus v) {
            this.corpus = v;
            return this;
        }

        /**
         * Builds the options. {@link Kind#SIMPLE} needs {@code sizeN >= 2} (a pattern of a history token plus the
         * sampled token) and the {@link Kind#CACHE} pair needs {@code ngramMin <= ngramMax}.
         *
         * @throws IllegalArgumentException for out-of-range sizes
         */
        public Options build() {
            if (sizeN == -1 && kind != Kind.CACHE) {
                sizeN = DEFAULT_SIZE_N;
            }
            if (kind == Kind.SIMPLE && sizeN < 2) {
                throw new IllegalArgumentException(kind + " needs a lookup of at least 2 tokens (a history token"
                        + " plus the sampled token), got sizeN=" + sizeN);
            }
            if (kind == Kind.CACHE && ngramMin > ngramMax) {
                throw new IllegalArgumentException("ngramMin " + ngramMin + " exceeds ngramMax " + ngramMax);
            }
            return new Options(this);
        }
    }

    private static void requireInRange(int v, int min, int max, String name) {
        if (v < min || v > max) {
            throw new IllegalArgumentException(name + " must be in [" + min + ", " + max + "], got " + v);
        }
    }

    // ---- the map state, a port of llama.cpp's common_ngram_map (common/ngram-map.h) ----

    private static final class MapValue {
        long valueIdx;   // first position of the value m-gram in the history; 0 = unused (llama.cpp's sentinel)
        int valueNum;    // occurrences of the value m-gram after the key n-gram
        int nAccepted;   // tokens of the value accepted at the last draft it proposed, at most sizeValue
    }

    private static final class MapKey {
        long keyIdx;     // first position of the key n-gram in the history
        long statIdx;    // position the key's statistics were updated to
        int keyNum;      // occurrences of the key n-gram, saturated
        final MapValue[] values;

        MapKey() {
            this.values = new MapValue[MAP_MAX_VALUES];
        }
    }

    private static final class MapState {
        final int sizeKey;
        final int sizeValue;
        final boolean keyOnly;
        final int minHits;
        final List<MapKey> keys = new ArrayList<>();
        /** From a key n-gram's 32-bit hash ({@code % size}) to its history position, 0 = empty. */
        final int[] keyMap = new int[MAP_HASH_MAP_SIZE];
        int keyMapLastIdx;
        long sizeLastBegin = -1;
        boolean lastDraftCreated;
        int lastDraftKeyIdx;
        int lastDraftValueIdx;

        MapState(Options o) {
            this.sizeKey = o.sizeN;
            this.sizeValue = o.sizeM;
            this.keyOnly = o.kind == Kind.MAP_K;
            this.minHits = o.minHits;
        }
    }

    /** llama.cpp {@code common_ngram_map_hash}: the key n-gram hashed to 32 bits ({@code uint32_t} per step). */
    private static int hashKey(int[] inp, int offset, int size) {
        long hash = 0;
        for (int i = 0; i < size; i++) {
            hash = (hash * MAP_LCG_FACTOR + (inp[offset + i] & 0xFFFFFFFFL)) & 0xFFFFFFFFL;
        }
        return (int) hash;
    }

    private final Kind kind;
    private final Options options;
    private EngineModel target;
    private ForwardBackend backend;
    private MapState map;
    private int[] modTable;
    private NgramCache cacheContext;
    private NgramCache cacheDynamic;
    private NgramCache cacheStatic;
    private long corpusRevision;
    private int lastHistoryLength = -1;
    private boolean closed;

    private NgramDrafter(Options options) {
        this.options = options;
        this.kind = options.kind;
    }

    /** A drafter of {@code options.kind} with {@code options}'s sizes and sources. */
    public static NgramDrafter of(Options options) {
        Objects.requireNonNull(options, "options");
        return new NgramDrafter(options);
    }

    /** A drafter of {@code kind} with the default sizes ({@link Options#builder}). */
    public static NgramDrafter of(Kind kind) {
        return of(Options.builder(kind).build());
    }

    /** The drafter's kind and sizes. */
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
        requireOpen();
        if (this.target != null) {
            throw new IllegalStateException("the n-gram drafter is already bound to a target");
        }
        ForwardBackend fb = target.forwardBackend();
        if (fb == null || fb.memory() == null) {
            throw new IllegalArgumentException("an n-gram drafter needs a target with model memory; "
                    + target.name() + " (" + target.architecture() + ") has none");
        }
        this.target = target;
        this.backend = fb;
        if (kind == Kind.MAP_K || kind == Kind.MAP_K4V) {
            this.map = new MapState(options);
        } else if (kind == Kind.MOD) {
            this.modTable = new int[options.modTableSize];
            Arrays.fill(modTable, EMPTY);
        } else if (kind == Kind.CACHE) {
            this.cacheContext = new NgramCache();
            this.cacheDynamic = copied(options.dynamicCache);
            this.cacheStatic = copied(options.staticCache);
            addCorpusToStatic();
        }
    }

    private static NgramCache copied(NgramCache cache) {
        if (cache == null) {
            return null;
        }
        NgramCache copy = new NgramCache();
        copy.merge(cache);
        return copy;
    }

    /** Adds every corpus sequence to the static cache ({@code nc_static}), then pins the corpus revision. */
    private void addCorpusToStatic() {
        NgramCorpus corpus = options.corpus;
        if (corpus != null) {
            if (cacheStatic == null) {
                cacheStatic = new NgramCache();
            }
            for (NgramCorpus.Corpus c : corpus.list()) {
                for (int[] sequence : corpus.sequences(c.name())) {
                    // llama.cpp's corpus caches always use the full n-gram size range.
                    cacheStatic.addSequence(1, NgramCache.MAX_KEY, sequence);
                }
            }
        }
        corpusRevision = corpus == null ? -1 : corpus.revision();
    }

    @Override
    public int[] draft(int lastToken, int n) {
        requireBound();
        if (n <= 0 || lastToken < 0) {
            return new int[0];
        }
        SequenceState seq = backend.memory().sequence();
        int len = seq.nPast();
        // llama.cpp's inp: the history and then the just-sampled token.
        int[] inp = new int[len + 1];
        for (int i = 0; i < len; i++) {
            inp[i] = seq.token(i);
        }
        inp[len] = lastToken;

        if (len < lastHistoryLength) {
            rebuild();
        }
        int[] proposal;
        if (kind == Kind.SIMPLE) {
            proposal = simpleDraft(inp, len);
        } else if (kind == Kind.MAP_K || kind == Kind.MAP_K4V) {
            proposal = mapDraft(inp, len);
        } else if (kind == Kind.MOD) {
            proposal = modDraft(inp, len);
        } else if (kind == Kind.CACHE) {
            proposal = cacheDraft(inp, len);
        } else {
            throw new IllegalStateException("unknown kind " + kind);
        }
        lastHistoryLength = len;
        if (proposal.length > n) {
            proposal = Arrays.copyOf(proposal, n);
        }
        if (proposal.length < options.draftMin) {
            return new int[0];
        }
        return proposal;
    }

    /** llama.cpp {@code common_ngram_simple_draft} over {@code inp} ({@code cur_len} = {@code len}, sampled last). */
    private int[] simpleDraft(int[] inp, int len) {
        int sizeN = options.sizeN;
        int sizeM = options.sizeM;
        if (len <= sizeN + sizeM + 1) {
            return new int[0];
        }
        int[] pattern = new int[sizeN];
        for (int j = 0; j < sizeN - 1; j++) {
            pattern[j] = inp[len - sizeN + 1 + j];
        }
        pattern[sizeN - 1] = inp[len];

        // Search backwards, skipping the current (rightmost) match: position 0 means no match.
        int matchPos = 0;
        for (int j = len - sizeN - 1; j > 0; j--) {
            if (matchesKey(inp, j, pattern, sizeN)) {
                matchPos = j;
                break;
            }
        }
        if (matchPos == 0) {
            return new int[0];
        }
        int copyMax = Math.min(sizeM, len - (matchPos + sizeN));
        if (copyMax < sizeN) {
            return new int[0];
        }
        int[] out = new int[copyMax];
        for (int j = 0; j < copyMax; j++) {
            out[j] = inp[matchPos + sizeN + j];
        }
        return out;
    }

    /**
     * llama.cpp {@code common_ngram_map_draft}. A shrunk history was already rebuilt, so {@code size_last_begin} is
     * set at the first draft of the generation (llama.cpp's {@code common_ngram_map_begin}) and every position the
     * map references exists in the current history.
     */
    private int[] mapDraft(int[] inp, int len) {
        MapState m = map;
        if (m.sizeLastBegin < 0 || m.sizeLastBegin > len) {
            m.sizeLastBegin = len;
        }
        m.lastDraftCreated = false;
        int n = m.sizeKey;
        int mv = m.sizeValue;
        if (len < 2 * n + mv) {
            return new int[0];
        }

        int[] keyTokens = new int[n];
        for (int j = 0; j < n - 1; j++) {
            keyTokens[j] = inp[len - n + 1 + j];
        }
        keyTokens[n - 1] = inp[len];

        // The key n-gram: a hash match first, then the two descending searches llama.cpp orders.
        int matchPos = 0;
        int idxKey = m.keyMap[Integer.remainderUnsigned(hashKey(keyTokens, 0, n), MAP_HASH_MAP_SIZE)];
        if (idxKey != 0 && idxKey < len - n - mv - 1 && matchesKey(inp, idxKey, keyTokens, n)) {
            matchPos = idxKey;
        }
        if (matchPos == 0 && m.sizeLastBegin > n + mv + 1) {
            for (int j = (int) m.sizeLastBegin - n - mv - 1; j > m.keyMapLastIdx; j--) {
                if (matchesKey(inp, j, keyTokens, n)) {
                    matchPos = j;
                    break;
                }
            }
        }
        if (matchPos == 0) {
            for (int j = len - n - mv - 1; j > m.sizeLastBegin && j > m.keyMapLastIdx; j--) {
                if (matchesKey(inp, j, keyTokens, n)) {
                    matchPos = j;
                    break;
                }
            }
        }

        // Add the hashes of the new key n-grams in the same ranges, so the next steps find them in the key map.
        if (m.sizeLastBegin > n + mv + 1) {
            for (int j = (int) m.sizeLastBegin - n - mv - 1; j > m.keyMapLastIdx; j--) {
                noteHash(m, inp, j, n);
            }
        }
        for (int j = len - n - mv - 1; j > m.sizeLastBegin && j > m.keyMapLastIdx; j--) {
            noteHash(m, inp, j, n);
        }
        m.keyMapLastIdx = Math.max(len - n - mv - 1, m.keyMapLastIdx);

        if (matchPos == 0) {
            return new int[0];
        }

        int keyOffset = m.keys.size();
        for (int i = 0; i < m.keys.size(); i++) {
            if (matchesKey(inp, (int) m.keys.get(i).keyIdx, keyTokens, n)) {
                keyOffset = i;
                break;
            }
        }
        if (keyOffset == m.keys.size()) {
            MapKey newKey = new MapKey();
            newKey.keyIdx = matchPos;
            newKey.statIdx = 0;
            for (int i = 0; i < MAP_MAX_VALUES; i++) {
                MapValue v = new MapValue();
                v.nAccepted = mv;
                newKey.values[i] = v;
            }
            m.keys.add(newKey);
        }
        MapKey curr = m.keys.get(keyOffset);
        curr.keyNum = (int) Math.min(curr.keyNum + 1L, (long) MAP_MAX_VALUE_COUNT);

        if (m.keyOnly) {
            // key-only mode drafts the tokens that follow the key on this match.
            int nDraft = Math.min(mv, curr.values[0].nAccepted);
            int[] out = new int[nDraft];
            for (int i = 0; i < nDraft; i++) {
                out[i] = inp[matchPos + n + i];
            }
            m.lastDraftCreated = true;
            m.lastDraftKeyIdx = keyOffset;
            m.lastDraftValueIdx = 0;
            return out;
        }
        if (curr.keyNum < m.minHits) {
            return new int[0];
        }

        // Examine every value m-gram of the key between the key's statistics start and the match.
        for (long i = curr.statIdx; i <= matchPos; i++) {
            if (!matchesKey(inp, (int) i, keyTokens, n)) {
                continue;
            }
            int idxBeginValueKey = (int) i + n;
            int idxValue = -1;
            for (int v = 0; v < MAP_MAX_VALUES; v++) {
                MapValue value = curr.values[v];
                long idxBeginValueV = value.valueIdx;
                if (idxBeginValueV == 0) {
                    value.valueIdx = idxBeginValueKey;
                    value.valueNum = 0;
                    value.nAccepted = mv;
                    idxValue = v;
                    break;
                }
                boolean match = true;
                for (int j = 0; j < mv; j++) {
                    if (inp[idxBeginValueKey + j] != inp[(int) idxBeginValueV + j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    idxValue = v;
                    break;
                }
            }
            if (idxValue >= 0) {
                MapValue value = curr.values[idxValue];
                value.valueNum = (int) Math.min(value.valueNum + 1L, (long) MAP_MAX_VALUE_COUNT);
            }
        }
        curr.statIdx = matchPos;

        int maxOccur = 0;
        int slotMax = 0;
        for (int v = 0; v < MAP_MAX_VALUES; v++) {
            if (curr.values[v].valueNum > maxOccur) {
                maxOccur = curr.values[v].valueNum;
                slotMax = v;
            }
        }
        long sumOccur = 0;
        for (int v = 0; v < MAP_MAX_VALUES; v++) {
            if (v != slotMax) {
                sumOccur += curr.values[v].valueNum;
            }
        }
        if (sumOccur > 0 && maxOccur < 2 * sumOccur) {
            // The most frequent value is not much more frequent than the other values: no draft.
            return new int[0];
        }
        int nDraft = Math.min(mv, curr.values[slotMax].nAccepted);
        int[] out = new int[nDraft];
        for (int i = 0; i < nDraft; i++) {
            out[i] = inp[matchPos + n + i];
        }
        m.lastDraftCreated = true;
        m.lastDraftKeyIdx = keyOffset;
        m.lastDraftValueIdx = slotMax;
        return out;
    }

    private void noteHash(MapState m, int[] inp, int j, int n) {
        int slot = Integer.remainderUnsigned(hashKey(inp, j, n), MAP_HASH_MAP_SIZE);
        if (m.keyMap[slot] == 0) {
            m.keyMap[slot] = j; // collisions may occur; llama.cpp keeps the first entry
        }
    }

    private static boolean matchesKey(int[] inp, int pos, int[] pattern, int size) {
        for (int k = 0; k < size; k++) {
            if (inp[pos + k] != pattern[k]) {
                return false;
            }
        }
        return true;
    }

    /** llama.cpp {@code common_ngram_map_accept}: the last draft's value learns how many tokens the target kept. */
    private void mapAccept(int nAccepted) {
        MapState m = map;
        if (!m.lastDraftCreated) {
            return;
        }
        m.keys.get(m.lastDraftKeyIdx).values[m.lastDraftValueIdx].nAccepted = Math.min(nAccepted, m.sizeValue);
    }

    // ---- mod ----

    /** llama.cpp {@code common_ngram_mod::idx}: the n-gram window hashed to an unsigned 64-bit table index. */
    private int modIndex(int[] tokens, int offset, int size) {
        long res = 0;
        for (int i = 0; i < size; i++) {
            res = res * MOD_HASH_FACTOR + tokens[offset + i];
        }
        return (int) Long.remainderUnsigned(res, modTable.length);
    }

    /** Add the n-gram {@code tokens[offset .. offset + size)} to the table, with the token that followed it. */
    private void modAdd(int[] tokens, int offset, int size, int following) {
        modTable[modIndex(tokens, offset, size)] = following;
    }

    /**
     * llama.cpp's n-gram-{@code mod} draft: every committed token adds the key n-gram that ends at it with itself as
     * the offer; each step the window of the last {@code sizeN} virtual tokens (history tail, the sampled token, the
     * proposals so far) is looked up, its offer appended, and the window moves.
     */
    private int[] modDraft(int[] inp, int len) {
        int n = options.sizeN;
        int cap = Math.min(options.sizeM, MAX_SIZE_M);
        if (len < n) {
            return new int[0];
        }
        for (int pos = Math.max(lastHistoryLength, n); pos < len; pos++) {
            modAdd(inp, pos - n, n, inp[pos]);
        }
        int[] out = new int[cap];
        int count = 0;
        while (count < cap) {
            int end = len + count;      // the last virtual token: history, then the sampled token, then proposals
            if (end < n) {
                break;
            }
            int[] window = new int[n];
            for (int j = 0; j < n; j++) {
                int idx = end - n + 1 + j;
                window[j] = idx <= len ? inp[idx] : out[idx - len - 1];
            }
            int next = modTable[modIndex(window, 0, n)];
            if (next == EMPTY) {
                break;
            }
            out[count++] = next;
        }
        return Arrays.copyOf(out, count);
    }

    // ---- cache ----

    /**
     * llama.cpp {@code common_ngram_cache_draft}: each step offers the context candidates over
     * {@code ngramMin}..{@code ngramMax}-grams (lax thresholds), then the dynamic candidates (strict), then the
     * static 2-gram's best token; the static part validates every candidate by its counts.
     */
    private int[] cacheDraft(int[] inp, int len) {
        int nNew = len - Math.max(lastHistoryLength, 0);
        if (nNew > 0 || lastHistoryLength < 0) {
            cacheContext.update(options.ngramMin, options.ngramMax, inp, len, nNew);
        }
        if (options.corpus != null && options.corpus.revision() != corpusRevision) {
            cacheStatic = new NgramCache();
            NgramCache loaded = copied(options.staticCache);
            if (loaded != null) {
                cacheStatic.merge(loaded);
            }
            for (NgramCorpus.Corpus c : options.corpus.list()) {
                for (int[] sequence : options.corpus.sequences(c.name())) {
                    cacheStatic.addSequence(1, NgramCache.MAX_KEY, sequence);
                }
            }
            corpusRevision = options.corpus.revision();
        }
        NgramCache ncStatic = cacheStatic;
        NgramCache ncContext = cacheContext;
        int[] proposed = new int[Math.min(options.sizeM, MAX_SIZE_M)];
        int count = 0;
        while (count < proposed.length) {
            if (len + 1 + count < CACHE_STATIC_NGRAM) {
                break;
            }
            NgramCache.Key keyStatic = combinedKey(inp, proposed, count, CACHE_STATIC_NGRAM);
            int token = tryDraft(ncContext, ncStatic, keyStatic, inp, proposed, count, true);
            if (token < 0) {
                token = tryDraft(cacheDynamic, ncStatic, keyStatic, inp, proposed, count, false);
            }
            if (token < 0) {
                token = tryStatic(ncStatic, keyStatic);
            }
            if (token < 0) {
                break;
            }
            proposed[count++] = token;
        }
        return Arrays.copyOf(proposed, count);
    }

    /**
     * llama.cpp {@code try_draft} against one primary cache: every candidate's weight compares
     * {@code count * staticWeight} ({@code 100 * count} in the static cache, 1 when absent); the best token is
     * drafted when the part's sample reaches the threshold and dominates by the percentage. Iterates n-gram sizes
     * from largest to smallest, as llama.cpp does, with the thresholds indexed by size (llama.cpp indexes its lists
     * from {@code ngram_min}, equal for the default {@code ngram_min = 1}).
     */
    private static int tryDraft(NgramCache primary, NgramCache ncStatic, NgramCache.Key keyStatic,
                                int[] inp, int[] proposed, int count, boolean lax) {
        if (primary == null) {
            return -1;
        }
        int[] minSample = lax ? CACHE_MIN_SAMPLE_LAX : CACHE_MIN_SAMPLE_STRICT;
        int[] minPercent = lax ? CACHE_MIN_PERCENT_LAX : CACHE_MIN_PERCENT_STRICT;
        for (int size = NgramCache.MAX_KEY; size >= 1; size--) {
            NgramCache.Key key = combinedKey(inp, proposed, count, size);
            if (key == null || !primary.hasPart(key)) {
                continue;
            }
            int maxCount = 0;
            int maxToken = -1;
            long bestStatic = 0;
            long sum = 0;
            for (int token : primary.tokensOf(key)) {
                int c = primary.count(key, token);
                long staticWeight = 1;
                if (ncStatic != null) {
                    int staticCount = ncStatic.count(keyStatic, token);
                    staticWeight = staticCount > 0 ? 100L * staticCount : 1;
                }
                if (c * staticWeight > maxCount * bestStatic) {
                    maxCount = c;
                    maxToken = token;
                    bestStatic = staticWeight;
                }
                sum += c;
            }
            if (sum < minSample[size - 1]) {
                continue;
            }
            if (100L * maxCount < (long) minPercent[size - 1] * sum) {
                continue;
            }
            return maxToken;
        }
        return -1;
    }

    /** llama.cpp's static fallback: the best token of the 2-gram part (its thresholds live at index static - 1). */
    private static int tryStatic(NgramCache ncStatic, NgramCache.Key keyStatic) {
        if (ncStatic == null) {
            return -1;
        }
        int maxCount = 0;
        int maxToken = -1;
        long sum = 0;
        for (int token : ncStatic.tokensOf(keyStatic)) {
            int c = ncStatic.count(keyStatic, token);
            if (c > maxCount) {
                maxCount = c;
                maxToken = token;
            }
            sum += c;
        }
        if (sum < CACHE_MIN_SAMPLE_LAX[CACHE_STATIC_NGRAM - 1]) {
            return -1;
        }
        if (100L * maxCount < (long) CACHE_MIN_PERCENT_LAX[CACHE_STATIC_NGRAM - 1] * sum) {
            return -1;
        }
        return maxToken;
    }

    /** The n-gram of {@code size} tokens ending {@code count} proposals past the sampled token, null when the
     * combined sequence is still shorter than the n-gram. */
    private static NgramCache.Key combinedKey(int[] inp, int[] proposed, int count, int size) {
        int end = inp.length + count;
        if (end < size) {
            return null;
        }
        int[] window = new int[size];
        for (int j = 0; j < size; j++) {
            int idx = end - size + j;
            window[j] = idx < inp.length ? inp[idx] : proposed[idx - inp.length];
        }
        return NgramCache.Key.of(window, 0, size);
    }

    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("the accepted count must not be negative, got " + nAccepted);
        }
        if (map != null) {
            mapAccept(nAccepted);
        }
    }

    @Override
    public void reset() {
        if (closed) {
            return;
        }
        rebuild();
    }

    /** Drops every per-history state: a new or shrunk history rebuilds the stateful kinds. */
    private void rebuild() {
        if (map != null) {
            map = new MapState(options);
        }
        if (modTable != null) {
            Arrays.fill(modTable, EMPTY);
        }
        if (cacheContext != null) {
            cacheContext = new NgramCache();
        }
        lastHistoryLength = -1;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("the n-gram drafter is closed");
        }
    }

    private void requireBound() {
        requireOpen();
        if (target == null) {
            throw new IllegalStateException("the n-gram drafter is not bound to a target");
        }
    }

    @Override
    public void close() {
        closed = true;
        target = null;
        backend = null;
        map = null;
        modTable = null;
        cacheContext = null;
        cacheDynamic = null;
        cacheStatic = null;
    }

    @Override
    public String toString() {
        return "NgramDrafter{" + kind.name().toLowerCase(Locale.ROOT) + ", sizeN=" + options.sizeN + ", sizeM="
                + options.sizeM + (options.draftMin > 0 ? ", draftMin=" + options.draftMin : "")
                + (target != null ? ", bound to " + target.name() : "") + '}';
    }
}
