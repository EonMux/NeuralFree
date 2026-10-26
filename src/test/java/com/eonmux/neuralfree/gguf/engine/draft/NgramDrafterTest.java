package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.HiddenTapSlot;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The SP-1 drafter surface on token streams: {@link NgramDrafter} in llama.cpp's {@code ngram-simple},
 * {@code ngram-map-k}, {@code ngram-map-k4v}, {@code ngram-mod} and {@code ngram-cache} modes, {@link NgramCorpus},
 * {@link NgramCache}, {@link NgramCacheFile} in the {@code common_ngram_cache_save} format, {@link SuffixTree} and
 * {@link SuffixDrafter} (vLLM's suffix decoding proposer), {@link DrafterChain} (llama.cpp {@code --spec-type a,b}),
 * the several-consumer {@code HiddenTapSlot} leases and the new {@code DraftType} names. Every target is a fake whose
 * memory is a plain {@link SequenceState}, prefilled with the test's token stream, exactly the view a bound drafter
 * has of its target's history.
 */
public class NgramDrafterTest {

    // ---- fixtures -------------------------------------------------------------------------

    private static final class FakeMem implements ModelMemory {
        final SequenceState sequence = new SequenceState(64);

        void prefill(int[] tokens) {
            if (sequence.nPast() != 0) {
                throw new IllegalStateException();
            }
            sequence.append(tokens, 0, tokens.length);
        }

        int[] append(int[] tokens) {
            sequence.append(tokens, 0, tokens.length);
            return tokens;
        }

        @Override public MemoryPlan plan() {
            return null;
        }

        @Override public SequenceState sequence() {
            return sequence;
        }

        @Override public void reset() {
            sequence.clear();
        }

        @Override public boolean canRollback(int nTokens) {
            return nTokens <= sequence.nPast();
        }

        @Override public void rollback(int nTokens) {
            sequence.truncate(nTokens);
        }

        @Override public boolean hasRecurrentState() {
            return false;
        }

        @Override public boolean canSnapshot() {
            return false;
        }

        @Override public boolean ringsKeepWindow(int nWrite, int nDrop) {
            return true;
        }

        @Override public void snapshot() {
        }

        @Override public void restore() {
        }

        @Override public void close() {
        }
    }

    private static final class FakeBackend implements ForwardBackend {
        final FakeMem memory = new FakeMem();

        @Override public float[] prefill(int[] tokens, int pos0) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[] decode(int token, int pos) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void reset() {
            memory.reset();
        }

        @Override public boolean canRollback(int nTokens) {
            return memory.canRollback(nTokens);
        }

        @Override public void rollback(int nTokens) {
            memory.rollback(nTokens);
        }

        @Override public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public ModelMemory memory() {
            return memory;
        }

        @Override public void setHiddenTap(HiddenTap tap) {
        }

        @Override public void close() {
        }
    }

    private static final class FakeModel implements EngineModel {
        final FakeBackend backend = new FakeBackend();

        FakeModel prefill(int[] tokens) {
            backend.memory.prefill(tokens);
            return this;
        }

        @Override public com.eonmux.neuralfree.gguf.GGUFTokenizer tokenizer() {
            return null;
        }

        @Override public void close() {
        }

        @Override public ForwardBackend forwardBackend() {
            return backend;
        }

        @Override public com.eonmux.neuralfree.gguf.spec.ModelSpec spec() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public String name() {
            return "fake";
        }

        @Override public String architecture() {
            return "fake-llama";
        }

        @Override public int vocabSize() {
            return 16;
        }

        @Override public int contextLength() {
            return 64;
        }

        @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                                        TokenCallback callback) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                                  TokenCallback callback) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void resetKVCache() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void printInfo() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int kvCacheContext() {
            return 64;
        }

        @Override public int embeddingLength() {
            return 0;
        }

        @Override public int layerCount() {
            return 0;
        }

        @Override public int headCount() {
            return 0;
        }

        @Override public int headCountKv() {
            return 0;
        }

        @Override public int feedForwardLength() {
            return 0;
        }

        @Override public float rmsNormEps() {
            return 0.0f;
        }

        @Override public float ropeFreqBase() {
            return 0.0f;
        }

        @Override public int headDim() {
            return 0;
        }

        @Override public int kvDim() {
            return 0;
        }
    }

    /** A drafter answering from the first proposal of its script, then draftless, for the chain tests. */
    private static class ScriptedDrafter implements DraftModel {
        final List<int[]> proposals = new ArrayList<>();
        final List<Integer> accepted = new ArrayList<>();
        int resets;
        EngineModel bound;
        int boundCount;
        boolean closed;

        ScriptedDrafter(int[]... proposals) {
            this.proposals.addAll(Arrays.asList(proposals));
        }

        @Override public void bindTarget(EngineModel target) {
            this.bound = target;
            boundCount++;
        }

        @Override public int[] draft(int lastToken, int n) {
            if (proposals.isEmpty()) {
                return new int[0];
            }
            return proposals.remove(0);
        }

        @Override public void accept(int nAccepted) {
            accepted.add(nAccepted);
        }

        @Override public void reset() {
            resets++;
        }

        @Override public void close() {
            closed = true;
        }
    }

    private static NgramDrafter drafter(NgramDrafter.Kind kind, FakeModel target, int sizeN, int sizeM) {
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Options.builder(kind).sizeN(sizeN).sizeM(sizeM).build());
        d.bindTarget(target);
        return d;
    }

    // ---- NgramDrafter: simple ------------------------------------------------------------

    @Test
    public void simpleSearchesBackwardsAndProposesTheFollowingTokens() {
        FakeModel target = new FakeModel().prefill(new int[]{5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.SIMPLE, target, 2, 4);
        // pattern [2, 3]: the previous occurrence at position 4 is followed by [4, 5, 6, 7]
        assertArrayEquals(new int[]{4, 5, 6, 7}, d.draft(3, 4));
    }

    @Test
    public void simpleRefusesWhenTooFewFollowTokensRemainAndCapsTheRequest() {
        // the whole history fits too few tokens behind the match for a sizeN proposal
        FakeModel target = new FakeModel().prefill(new int[]{5, 6, 7, 1, 2, 3, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.SIMPLE, target, 3, 5);
        assertArrayEquals(new int[0], d.draft(3, 5));
        FakeModel full = new FakeModel().prefill(new int[]{5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2});
        NgramDrafter caps = drafter(NgramDrafter.Kind.SIMPLE, full, 2, 4);
        assertArrayEquals(new int[]{4}, caps.draft(3, 1));
    }

    // ---- NgramDrafter: map kinds ----------------------------------------------------------

    @Test
    public void mapKeyProposesTheTokensAfterTheHit() {
        FakeModel target = new FakeModel().prefill(new int[]{8, 9, 1, 2, 3, 8, 9, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.MAP_K, target, 2, 2);
        assertArrayEquals(new int[]{8, 9}, d.draft(3, 4));
    }

    @Test
    public void mapKeyShrinksTheProposalAfterAcceptAndResetsToFull() {
        FakeModel target = new FakeModel().prefill(new int[]{8, 9, 1, 2, 3, 8, 9, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.MAP_K, target, 2, 2);
        assertArrayEquals(new int[]{8, 9}, d.draft(3, 4));
        d.accept(1);
        // the key's value learned that the target kept one of its tokens
        assertArrayEquals(new int[]{8}, d.draft(3, 4));
        d.reset();
        assertArrayEquals(new int[]{8, 9}, d.draft(3, 4));
    }

    @Test
    public void mapK4vRefusesALengthThenAcceptsDominantValues() {
        // the key [2, 3] at positions 0 and 6 leaves values [7, 1] and [5, 5], tied 1:1 -> refusal
        FakeModel tied = new FakeModel().prefill(new int[]{2, 3, 7, 1, 1, 1, 2, 3, 5, 5, 5, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.MAP_K4V, tied, 2, 2);
        assertArrayEquals(new int[0], d.draft(3, 4));
        // [7, 7] dominates the single [4, 8] behind the key -> the tokens after the match are proposed
        FakeModel dominant = new FakeModel().prefill(new int[]{2, 3, 7, 7, 2, 3, 7, 7, 2, 3, 4, 8, 9, 2});
        NgramDrafter d2 = drafter(NgramDrafter.Kind.MAP_K4V, dominant, 2, 2);
        assertArrayEquals(new int[]{4, 8}, d2.draft(3, 4));
        // minHits = 2 refuses the same history, whose key has one hit so far
        NgramDrafter d3 = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.MAP_K4V)
                .sizeN(2).sizeM(2).minHits(2).build());
        d3.bindTarget(dominant);
        assertArrayEquals(new int[0], d3.draft(3, 4));
        // and the second hit on the same key proposes, from the earlier sweep the key's values are 2:1
        assertArrayEquals(new int[]{4, 8}, d3.draft(3, 4));
    }

    // ---- NgramDrafter: mod ----------------------------------------------------------------

    @Test
    public void modChainsTheHashedOffers() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3, 4, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.MOD, target, 3, 3);
        // [1,2,3] offers 4, [2,3,4] offers 1 and [3,4,1] offers 2
        assertArrayEquals(new int[]{4, 1, 2}, d.draft(3, 4));
    }

    @Test
    public void modRebuildsAfterARollback() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3, 4, 1, 2});
        NgramDrafter d = drafter(NgramDrafter.Kind.MOD, target, 3, 3);
        assertArrayEquals(new int[]{4, 1, 2}, d.draft(3, 4));
        target.backend.memory.rollback(2);
        // the stale [3, 4, 1] -> 2 offer is gone with the truncated history, and the rebuilt table
        // (over [1, 2, 3, 4]) has no offer for the window [3, 4, 1]
        assertArrayEquals(new int[0], d.draft(1, 4));
        // the new appended tokens fill the rebuilt table again and the chain drafts from it
        target.backend.memory.append(new int[]{1, 2});
        assertArrayEquals(new int[]{4, 1, 2}, d.draft(3, 4));
    }

    // ---- NgramDrafter: cache type --------------------------------------------------------

    @Test
    public void cacheDraftsFromTheStaticCorpusCache() {
        NgramCorpus corpus = new NgramCorpus();
        corpus.add("docs", new int[]{1, 2, 7, 1, 2, 7, 1, 2, 4});
        FakeModel target = new FakeModel().prefill(new int[]{1, 2});
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE)
                .sizeM(3).corpus(corpus).build());
        d.bindTarget(target);
        // the static 2-gram [2, 7] offers 1 twice, then [7, 2] offers 2 and [2, 7] offers 1 again
        assertArrayEquals(new int[]{1, 2, 7}, d.draft(7, 4));
    }

    @Test
    public void cacheRefusesBelowTheThresholds() {
        NgramCorpus corpus = new NgramCorpus();
        corpus.add("docs", new int[]{1, 2, 7, 1, 2, 4});
        FakeModel target = new FakeModel().prefill(new int[]{1, 2});
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE)
                .sizeM(3).corpus(corpus).build());
        d.bindTarget(target);
        // [2, 7] counts its offer once: below the lax sample size of 2
        assertArrayEquals(new int[0], d.draft(7, 4));
    }

    @Test
    public void cacheFollowsTheCorpusRevision() {
        NgramCorpus corpus = new NgramCorpus();
        FakeModel target = new FakeModel().prefill(new int[]{1, 2});
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE)
                .sizeM(3).corpus(corpus).build());
        d.bindTarget(target);
        assertArrayEquals(new int[0], d.draft(7, 4));
        corpus.add("docs", new int[]{1, 2, 7, 1, 2, 7, 1, 2, 4});
        assertArrayEquals(new int[]{1, 2, 7}, d.draft(7, 4));
        assertTrue(corpus.remove("docs"));
        assertArrayEquals(new int[0], d.draft(7, 4));
    }

    @Test
    public void cacheHonorsDraftMinAndTheNgramSizePair() {
        NgramCorpus corpus = new NgramCorpus();
        corpus.add("docs", new int[]{1, 2, 7, 1, 2, 7, 1, 2, 4});
        FakeModel target = new FakeModel().prefill(new int[]{1, 2});
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE)
                .sizeM(3).draftMin(4).corpus(corpus).build());
        d.bindTarget(target);
        // the 3-token chain stays below the draft minimum
        assertArrayEquals(new int[0], d.draft(7, 4));
        assertThrows(IllegalArgumentException.class,
                () -> NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE).ngramMin(3).ngramMax(2).build());
        NgramDrafter.Options sized = NgramDrafter.Options.builder(NgramDrafter.Kind.CACHE)
                .ngramMax(2).build();
        assertEquals(2, sized.ngramMax());
    }

    // ---- NgramCorpus -----------------------------------------------------------------------

    @Test
    public void corpusStoresRemovesListsAndEvictsFifo() {
        NgramCorpus corpus = new NgramCorpus(10);
        corpus.add("a", new int[]{1, 2, 3});
        assertFalse(corpus.remove("missing"));
        corpus.add("a", new int[]{4, 5, 6});
        corpus.add("b", new int[]{7, 8, 9, 10, 11, 12});
        assertEquals(2, corpus.list().size());
        assertArrayEquals(new int[][]{{7, 8, 9, 10, 11, 12}}, corpus.sequences("b"));
        NgramCorpus.Corpus b = corpus.list().get(1);
        assertEquals("b", b.name());
        assertEquals(1, b.sequenceCount());
        assertEquals(6, b.tokenCount());
        assertEquals(10, b.maxTokens());
        long before = corpus.revision();
        assertTrue(corpus.remove("b"));
        assertTrue(corpus.revision() > before);
        assertEquals(1, corpus.list().size());
        assertTrue(corpus.toString().contains("Corpus"));
        assertThrows(IllegalArgumentException.class, () -> corpus.sequences("b"));
        assertThrows(IllegalArgumentException.class, () -> corpus.add("", new int[]{1}));
        assertThrows(IllegalArgumentException.class, () -> corpus.add("negative", new int[]{-1}));
        assertThrows(IllegalArgumentException.class,
                () -> corpus.add("huge", new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
        // the budget eviction drops "a"'s oldest sequence when "a" overflows
        corpus.add("a", new int[]{1, 2, 3, 4, 5});
        assertArrayEquals(new int[][]{{4, 5, 6}, {1, 2, 3, 4, 5}}, corpus.sequences("a"));
        corpus.add("a", new int[]{6, 7, 8, 9, 10, 11});
        assertArrayEquals(new int[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}, corpus.sequences("a"));
    }

    @Test
    public void corpusHandsOutCopies() {
        NgramCorpus corpus = new NgramCorpus();
        corpus.add("a", new int[]{1, 2, 3});
        int[][] seqs = corpus.sequences("a");
        seqs[0][0] = 99;
        assertArrayEquals(new int[]{1, 2, 3}, corpus.sequences("a")[0]);
    }

    // ---- NgramCache and its file ----------------------------------------------------------

    @Test
    public void cacheKeysPadAndCount() {
        NgramCache cache = new NgramCache();
        int[] tokens = {5, 6, 7, 8};
        cache.addSequence(2, 2, tokens);
        assertEquals(1, cache.count(NgramCache.Key.of(tokens, 0, 2), tokens[2]));
        assertEquals(1, cache.count(NgramCache.Key.of(tokens, 1, 2), tokens[3]));
        assertEquals(0, cache.count(NgramCache.Key.of(tokens, 0, 2), tokens[3]));
        assertEquals(1, cache.count(NgramCache.Key.of(tokens, 0, 2), tokens[2]));
        assertTrue(cache.hasPart(NgramCache.Key.of(tokens, 0, 2)));
        assertTrue(cache.isEmpty() == false);
        NgramCache other = new NgramCache();
        other.addSequence(1, 1, tokens);
        cache.merge(other);
        assertEquals(1, cache.count(NgramCache.Key.of(tokens, 0, 1), tokens[1]));
        assertEquals(cache.entries().size() > 0, true);
        assertTrue(cache.tokensOf(NgramCache.Key.of(tokens, 0, 1)).contains(tokens[1]));
        assertTrue(cache.toString().contains("n-grams"));
    }

    @Test
    public void cacheFileRoundTrips() throws IOException {
        NgramCache cache = new NgramCache();
        cache.addSequence(1, 3, new int[]{3, 1, 4, 1, 5});
        cache.addCount(new NgramCache.Key(2, NgramCache.NULL, NgramCache.NULL, NgramCache.NULL), 9, 3);
        cache.addSequence(2, 2, new int[]{8, 8, 8});
        Path file = Files.createTempFile("ngram-cache", ".bin");
        try {
            NgramCacheFile.save(cache, file);
            NgramCache loaded = NgramCacheFile.load(file);
            assertEquals(cache.ngrams(), loaded.ngrams());
            for (NgramCache.Key key : cache.keys()) {
                assertEquals(cache.tokensOf(key), loaded.tokensOf(key));
                for (int token : cache.tokensOf(key)) {
                    assertEquals(cache.count(key, token), loaded.count(key, token));
                }
            }
            assertEquals(3, loaded.count(new NgramCache.Key(2, NgramCache.NULL, NgramCache.NULL,
                    NgramCache.NULL), 9));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    private static byte[] savedCache(NgramCache cache) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        NgramCacheFile.save(cache, out);
        return out.toByteArray();
    }

    @Test
    public void cacheFileRejectsTruncatedAndInvalidRecords() throws IOException {
        NgramCache cache = new NgramCache();
        cache.addSequence(1, 1, new int[]{5, 5});
        cache.addCount(new NgramCache.Key(7, NgramCache.NULL, NgramCache.NULL, NgramCache.NULL), 9, 2);
        byte[] raw = savedCache(cache);

        for (int cut : new int[]{1, 5, 17, 20, 21, 25}) {
            byte[] head = Arrays.copyOf(raw, Math.min(cut, raw.length));
            UncheckedIOException e = assertThrows(UncheckedIOException.class,
                    () -> NgramCacheFile.load(new ByteArrayInputStream(head), "trunc.bin"));
            assertTrue(e.getMessage(), e.getMessage().contains("offset"));
        }
        assertThrows(UncheckedIOException.class, () -> NgramCacheFile.load(new ByteArrayInputStream(
                Arrays.copyOf(raw, raw.length + 2)), "partial.bin"));
        byte[] withoutTokens = raw.clone();
        setInt32LE(withoutTokens, 16, 0);
        assertTrue(assertThrows(UncheckedIOException.class,
                () -> NgramCacheFile.load(new ByteArrayInputStream(withoutTokens), "zero.bin")).getMessage()
                .contains("non-positive token count"));
        byte[] badCount = raw.clone();
        setInt32LE(badCount, 20, 0);
        assertTrue(assertThrows(UncheckedIOException.class,
                () -> NgramCacheFile.load(new ByteArrayInputStream(badCount), "row.bin")).getMessage()
                .contains("non-positive count"));
        byte[] padding = raw.clone();
        setInt32LE(padding, 0, NgramCache.NULL);
        assertTrue(assertThrows(UncheckedIOException.class,
                () -> NgramCacheFile.load(new ByteArrayInputStream(padding), "pad.bin")).getMessage()
                .contains("padding"));
    }

    private static void setInt32LE(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
    }

    // ---- SuffixTree ------------------------------------------------------------------------

    @Test
    public void suffixTreeMatchesAndDraftsTheBestChain() {
        SuffixTree tree = new SuffixTree(8);
        tree.extend(1, new int[]{1, 2, 3, 4});
        tree.extend(2, new int[]{1, 2, 3, 4});
        tree.extend(3, new int[]{1, 2, 3, 5});
        // pattern [2, 3]: the child 4 counts 2 against 5 once -> chain [4], and [3, 4] ends the stored sequences
        SuffixTree.Draft d = tree.speculate(new int[]{2, 3}, 4, 1.0, 0.0, 0.05);
        assertArrayEquals(new int[]{4}, d.tokenIds());
        assertEquals(2, d.matchLength());
        assertEquals(1.0, d.score(), 1e-9);
        assertEquals(2.0 / 3.0, d.probabilities()[0], 1e-9);
        // [3, 4] matches fully but has no continuation
        assertEquals(0, tree.speculate(new int[]{3, 4}, 4, 1.0, 0.0, 0.05).tokenIds().length);
        // the spec factor of 0 keeps nothing drafted
        assertEquals(0, tree.speculate(new int[]{2, 3}, 4, 0.0, 0.0, 0.05).tokenIds().length);
        // minTokenProb 1.0 refuses the 2:1 child
        assertEquals(0, tree.speculate(new int[]{2, 3}, 4, 1.0, 0.0, 1.0).tokenIds().length);
        // a pattern the tree does not know matches nothing and reports its 0 matched tokens
        SuffixTree.Draft unknown = tree.speculate(new int[]{6}, 4, 1.0, 0.0, 0.05);
        assertEquals(0, unknown.matchLength());
        assertEquals(0, unknown.tokenIds().length);
    }

    @Test
    public void suffixTreeExtendsSequencesAndDropsThem() {
        SuffixTree tree = new SuffixTree(8);
        tree.extend(1, new int[]{1, 2, 3});
        tree.extend(1, new int[]{4, 5});
        // the committed suffix [3, 4, 5] matches the extended sequence, which ends there
        SuffixTree.Draft d = tree.speculate(new int[]{2, 3, 4, 5}, 4, 1.0, 0.0, 0.05);
        assertEquals(3, d.matchLength());
        assertEquals(0, d.tokenIds().length);
        tree.dropSequence(1);
        tree.clear();
        assertEquals(0, tree.speculate(new int[]{1, 2}, 4, 1.0, 0.0, 0.05).matchLength());
        assertEquals(0, tree.sequenceCount());
        assertThrows(IllegalArgumentException.class, () -> tree.extend(-1, new int[]{1}));
        assertThrows(IllegalArgumentException.class, () -> tree.extend(1, new int[]{2, -3}));
        assertThrows(IllegalArgumentException.class, () -> new SuffixTree(0));
        assertTrue(tree.toString().contains("maxDepth=8"));
        assertEquals(8, tree.maxDepth());
    }

    @Test
    public void suffixTreeCapsItsDepth() {
        SuffixTree deep = new SuffixTree(2);
        deep.extend(1, new int[]{1, 2, 3, 4});
        // suffix ends longer than 2 are never entered: [3, 4] matches but is never proposed from
        assertEquals(2, deep.speculate(new int[]{3, 4, 5}, 4, 1.0, 0.0, 0.05).matchLength());
        assertEquals(0, deep.speculate(new int[]{3, 4}, 9, 1.0, 0.0, 0.05).tokenIds().length);
    }

    // ---- SuffixDrafter ---------------------------------------------------------------------

    @Test
    public void suffixDrafterDraftsFromTheLocalTree() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3, 4, 5, 1, 2, 3});
        SuffixDrafter d = SuffixDrafter.of();
        d.bindTarget(target);
        // the local tree holds the committed tokens; the pattern [1, 2, 3, 4] + sampled 5 matches, and its most
        // probable chain continues into the stored sequence's own continuation [5, 1, 2, 3]
        assertArrayEquals(new int[]{5, 1, 2, 3}, d.draft(5, 4));
        // the request caps the chain
        assertArrayEquals(new int[]{5}, d.draft(5, 1));
    }

    @Test
    public void suffixDrafterDraftsFromTheCorpusAndFollowsTheRevision() {
        NgramCorpus corpus = new NgramCorpus();
        corpus.add("docs", new int[]{1, 2, 3, 4, 9, 9, 9});
        FakeModel target = new FakeModel().prefill(new int[]{2});
        SuffixDrafter d = SuffixDrafter.of(SuffixDrafter.Options.builder().corpus(corpus).build());
        d.bindTarget(target);
        // the local history [2] drafts nothing; the global tree continues the matched [2, 3] with [4, 9]
        assertArrayEquals(new int[]{4, 9}, d.draft(3, 4));
        assertTrue(corpus.remove("docs"));
        assertArrayEquals(new int[0], d.draft(3, 4));
    }

    @Test
    public void suffixDrafterResetsItsLocalTree() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3, 4, 5, 1, 2, 3});
        SuffixDrafter d = SuffixDrafter.of();
        d.bindTarget(target);
        assertArrayEquals(new int[]{5, 1, 2, 3}, d.draft(5, 4));
        d.accept(1);
        d.reset();
        // the decoder resets the target memory with the drafter: the fresh generation drafts nothing
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{1, 2});
        assertArrayEquals(new int[0], d.draft(3, 4));
        assertThrows(IllegalStateException.class, () -> d.bindTarget(new FakeModel()));
        assertFalse(d.toString().isEmpty());
    }

    @Test
    public void suffixDrafterValidatesItsTargetsAndOptions() {
        FakeModel target = new FakeModel();
        SuffixDrafter d = SuffixDrafter.of();
        assertThrows(IllegalStateException.class, () -> d.draft(0, 4));
        d.bindTarget(target);
        assertArrayEquals(new int[0], d.draft(-1, 4));
        assertArrayEquals(new int[0], d.draft(5, 0));
        assertThrows(IllegalStateException.class, () -> d.bindTarget(new FakeModel()));
        assertThrows(IllegalArgumentException.class, () -> d.accept(-1));
        assertEquals(SuffixDrafter.DEFAULT_MAX_TREE_DEPTH, SuffixDrafter.of().options().maxTreeDepth());
        assertEquals(SuffixDrafter.DEFAULT_MIN_TOKEN_PROB,
                SuffixDrafter.of().options().minTokenProb(), 0.0);
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.Options.builder().minTokenProb(2).build());
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.Options.builder().maxTreeDepth(0).build());
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.Options.builder().maxSpecFactor(-1).build());
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.Options.builder().maxSpecOffset(-1).build());
        d.close();
        assertThrows(IllegalStateException.class, () -> d.draft(0, 4));
    }

    @Test
    public void drafterValidatesItsBindingAndOptions() {
        NgramDrafter d = NgramDrafter.of(NgramDrafter.Kind.SIMPLE);
        assertThrows(IllegalStateException.class, () -> d.draft(0, 4));
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3});
        d.bindTarget(target);
        assertArrayEquals(new int[0], d.draft(-1, 4));
        assertArrayEquals(new int[0], d.draft(3, 0));
        assertThrows(IllegalStateException.class, () -> d.bindTarget(new FakeModel()));
        d.close();
        assertThrows(IllegalStateException.class, () -> d.draft(0, 4));
        assertThrows(IllegalArgumentException.class, () -> d.accept(-1));
        // options round-trip through their builder
        NgramDrafter.Options options = NgramDrafter.Options.builder(NgramDrafter.Kind.MOD)
                .sizeN(4).sizeM(9).modTableSize(1 << 12).draftMin(2).build();
        NgramDrafter.Options again = options.toBuilder().build();
        assertEquals(4, again.sizeN());
        assertEquals(9, again.sizeM());
        assertEquals(1 << 12, again.modTableSize());
        assertEquals(2, again.draftMin());
        assertThrows(IllegalArgumentException.class,
                () -> NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(1).build());
        assertThrows(IllegalArgumentException.class,
                () -> NgramDrafter.Options.builder(NgramDrafter.Kind.MOD).modTableSize(0).build());
        assertThrows(IllegalArgumentException.class,
                () -> NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(0).build());
        NgramDrafter cacheDrafter = NgramDrafter.of(NgramDrafter.Kind.CACHE);
        assertNotNull(cacheDrafter.options());
    }

    // ---- DrafterChain ----------------------------------------------------------------------

    @Test
    public void chainAsksInOrderAndRoutesAccept() {
        ScriptedDrafter a = new ScriptedDrafter(new int[0]);
        ScriptedDrafter b = new ScriptedDrafter(new int[]{1, 2});
        ScriptedDrafter c = new ScriptedDrafter(new int[]{3});
        FakeModel target = new FakeModel().prefill(new int[]{5});
        DrafterChain chain = new DrafterChain(a, b, c);
        chain.bindTarget(target);
        assertEquals(target, a.bound);
        assertEquals(target, c.bound);
        assertArrayEquals(new int[]{1, 2}, chain.draft(0, 4));
        // b's proposal won, so accept reaches only b
        chain.accept(1);
        assertEquals(1, b.accepted.size());
        assertEquals(0, c.accepted.size());
        // the next call reaches only c (a and b are draftless)
        assertArrayEquals(new int[0], a.draft(0, 4));
        assertArrayEquals(new int[]{3}, chain.draft(0, 4));
        chain.accept(1);
        assertEquals(1, c.accepted.size());
        // a draft-less chain hears nothing
        ScriptedDrafter empty = new ScriptedDrafter();
        DrafterChain silent = new DrafterChain(empty);
        silent.bindTarget(target);
        assertArrayEquals(new int[0], silent.draft(0, 4));
        silent.accept(1);
        assertEquals(0, empty.accepted.size());
        // reset and close reach every member
        chain.reset();
        assertEquals(1, a.resets);
        assertEquals(3, chain.drafters().size());
        chain.close();
        assertTrue(b.closed);
        assertThrows(IllegalStateException.class, () -> chain.draft(0, 4));
    }

    @Test
    public void chainValidatesItsMembersAndBindFailures() {
        FakeModel target = new FakeModel();
        assertThrows(IllegalArgumentException.class, () -> new DrafterChain(new DraftModel[]{}));
        ScriptedDrafter refuser = new ScriptedDrafter() {
            @Override public void bindTarget(EngineModel t) {
                throw new IllegalArgumentException("no memory");
            }
        };
        ScriptedDrafter ok = new ScriptedDrafter();
        DrafterChain chain = new DrafterChain(ok, refuser);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> chain.bindTarget(target));
        assertTrue(e.getMessage(), e.getMessage().contains("member 1"));
        assertEquals(1, ok.boundCount);
        assertThrows(IndexOutOfBoundsException.class, () -> chain.drafters().get(2).bindTarget(target));
        assertThrows(NullPointerException.class, () -> new DrafterChain((DraftModel[]) null));
        assertTrue(new DrafterChain(java.util.Collections.singletonList(ok)).drafters().size() == 1);
    }

    // ---- HiddenTapSlot ---------------------------------------------------------------------

    private static final class TapRoute implements HiddenTap {
        final List<Integer> layers = new ArrayList<>();
        final Set<Integer> wanted;
        int onHiddenCalls;

        TapRoute(int... wantedLayers) {
            Set<Integer> set = new HashSet<>();
            for (int layer : wantedLayers) {
                set.add(layer);
            }
            this.wanted = set;
        }

        @Override public void onHidden(int layer, float[] r, int nRows, int width) {
            layers.add(layer);
            onHiddenCalls++;
        }

        @Override public boolean wants(int layer) {
            return wanted.isEmpty() || wanted.contains(layer);
        }
    }

    @Test
    public void drafterFeedLeasesCoexistAndFilterTheirLayerSets() {
        ForwardBackend backend = new FakeModel().forwardBackend();
        HiddenTapSlot.Lease feedA = HiddenTapSlot.acquire(backend, "mtp", HiddenTapSlot.TapKind.DRAFTER_FEED,
                new int[]{2});
        HiddenTapSlot.Lease feedB = HiddenTapSlot.acquire(backend, "eagle3", HiddenTapSlot.TapKind.DRAFTER_FEED,
                new int[0]);
        assertEquals(HiddenTapSlot.TapKind.DRAFTER_FEED, feedA.kind());
        assertEquals("mtp, eagle3", HiddenTapSlot.holder(backend));
        assertEquals(2, HiddenTapSlot.leases(backend).size());
        assertArrayEquals(new int[]{2}, feedA.tapLayers());
        assertArrayEquals(new int[0], feedB.tapLayers());

        TapRoute route = new TapRoute();
        HiddenTapSlot.DrafterFeed tapA = feedA.drafterFeed(route);
        assertTrue(tapA.wants(2));
        assertFalse(tapA.wants(3));
        tapA.onHidden(2, new float[0], 0, 0);
        tapA.onHidden(5, new float[0], 0, 0);
        assertEquals(1, route.layers.size());
        // a delegate that does not want the layer keeps the feed silent too
        TapRoute selective = new TapRoute(3);
        assertFalse(feedA.drafterFeed(selective).wants(2));
        assertFalse(feedA.drafterFeed(selective).wants(3));
        // released feeds observe nothing
        feedA.close();
        assertFalse(tapA.wants(2));
        assertEquals("eagle3", HiddenTapSlot.holder(backend));
        feedB.close();
        assertNull(HiddenTapSlot.holder(backend));
        assertTrue(tapA.toString().contains("DrafterFeed"));
    }

    @Test
    public void observerLeasesStayExclusive() {
        ForwardBackend backend = new FakeModel().forwardBackend();
        HiddenTapSlot.Lease lease = HiddenTapSlot.acquire(backend, "pockettts");
        assertEquals(HiddenTapSlot.TapKind.OBSERVER, lease.kind());
        assertNull(lease.tapLayers());
        assertThrows(IllegalStateException.class, () -> HiddenTapSlot.acquire(backend, "another reader"));
        assertThrows(IllegalStateException.class,
                () -> HiddenTapSlot.acquire(backend, "mtp", HiddenTapSlot.TapKind.DRAFTER_FEED, new int[]{1}));
        assertThrows(IllegalStateException.class, () -> lease.drafterFeed(new TapRoute()));
        lease.close();
        lease.close();
        HiddenTapSlot.Lease again = HiddenTapSlot.acquire(backend, "pockettts");
        again.close();
        assertEquals(0, HiddenTapSlot.leases(backend).size());
    }

    // ---- EngineConfig.DraftType ------------------------------------------------------------

    @Test
    public void draftTypeParsesTheNewLookupTypes() {
        assertEquals(EngineConfig.DraftType.NGRAM, EngineConfig.DraftType.parse("ngram-simple"));
        assertEquals(EngineConfig.DraftType.NGRAM, EngineConfig.DraftType.parse("prompt-lookup"));
        assertEquals(EngineConfig.DraftType.NGRAM_MAP, EngineConfig.DraftType.parse("ngram-map-k"));
        assertEquals(EngineConfig.DraftType.NGRAM_MAP, EngineConfig.DraftType.parse("map-k4v"));
        assertEquals(EngineConfig.DraftType.NGRAM_MAP, EngineConfig.DraftType.parse("mod"));
        assertEquals(EngineConfig.DraftType.NGRAM_CACHE, EngineConfig.DraftType.parse("ngram-cache"));
        assertEquals(EngineConfig.DraftType.SUFFIX, EngineConfig.DraftType.parse("suffix"));
        assertEquals(EngineConfig.DraftType.MODEL, EngineConfig.DraftType.parse("model"));
        assertEquals("ngram-simple", EngineConfig.DraftType.NGRAM.llamaName());
        assertEquals("ngram-cache", EngineConfig.DraftType.NGRAM_CACHE.llamaName());
        assertEquals("suffix", EngineConfig.DraftType.SUFFIX.llamaName());
        assertTrue(EngineConfig.DraftType.NGRAM.lookupWithoutDraftModel());
        assertTrue(EngineConfig.DraftType.NGRAM_MAP.lookupWithoutDraftModel());
        assertTrue(EngineConfig.DraftType.NGRAM_CACHE.lookupWithoutDraftModel());
        assertTrue(EngineConfig.DraftType.SUFFIX.lookupWithoutDraftModel());
        assertFalse(EngineConfig.DraftType.MODEL.lookupWithoutDraftModel());
        assertFalse(EngineConfig.DraftType.AUTO.lookupWithoutDraftModel());
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.DraftType.parse("ngram-unknown"));
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.DraftType.parse("draft-eagle3"));
        java.util.Properties p = new java.util.Properties();
        p.setProperty(EngineConfig.PROP_DRAFT_TYPE, "pRoMpT-lOoKuP");
        assertEquals(EngineConfig.DraftType.NGRAM, EngineConfig.fromProperties(p).draftType());
        // and a config with only a lookup type is speculative
        assertTrue(EngineConfig.fromProperties(p).speculative());
        assertFalse(EngineConfig.defaults().speculative());
        assertNotNull(EngineConfig.fromProperties(p).toString());
    }
}
