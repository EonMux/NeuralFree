package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.draft.LookupTargetFixture.FakeModel;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The global request tree of suffix decoding (vLLM {@code suffix_decoding_max_cached_requests}, Arctic Inference's
 * {@code SuffixDecodingCache}): {@link SuffixTree#remove} takes a sequence back out, {@link SuffixDrafter} caches each
 * generation's response (sampled tokens, accepted draft tokens and tokens committed between drafts) beside the
 * per-request tree, drafts from past responses, and evicts the oldest requests beyond its capacity.
 */
public class SuffixRequestCacheTest {

    /** A drafter whose spec factor lets a one-token match draft several tokens. */
    private static SuffixDrafter drafter(FakeModel target, int maxCachedRequests) {
        SuffixDrafter d = SuffixDrafter.of(SuffixDrafter.Options.builder().maxSpecFactor(4)
                .maxCachedRequests(maxCachedRequests).build());
        d.bindTarget(target);
        return d;
    }

    /** One generation after {@code prompt}: each draft call samples the next response token, then commits it. */
    private static void generate(SuffixDrafter d, FakeModel target, int[] prompt, int... response) {
        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(prompt);
        for (int token : response) {
            d.draft(token, 4);
            d.accept(0);
            target.backend.memory.append(new int[]{token});
        }
    }

    @Test
    public void suffixTreeRemoveTakesASequenceBackOut() {
        SuffixTree tree = new SuffixTree(8);
        tree.extend(1, new int[]{1, 2, 3});
        tree.extend(2, new int[]{1, 2, 4});
        SuffixTree.Draft both = tree.speculate(new int[]{1, 2}, 4, 4.0, 0.0, 0.0);
        assertEquals(0.5, both.probabilities()[0], 1e-12);
        assertTrue(tree.remove(1));
        SuffixTree.Draft left = tree.speculate(new int[]{1, 2}, 4, 4.0, 0.0, 0.0);
        assertArrayEquals("only the remaining sequence continues [1, 2]", new int[]{4}, left.tokenIds());
        assertEquals(1.0, left.probabilities()[0], 0.0);
        assertArrayEquals("the removed sequence's own substrings are gone", new int[0],
                tree.speculate(new int[]{3}, 4, 4.0, 0.0, 0.0).tokenIds());
        assertFalse("an unknown sequence", tree.remove(1));
        assertTrue(tree.remove(2));
        assertEquals(0, tree.sequenceCount());
        assertArrayEquals(new int[0], tree.speculate(new int[]{1}, 4, 4.0, 0.0, 0.0).tokenIds());
        // a removed id extends afresh
        tree.extend(2, new int[]{5, 6});
        assertArrayEquals(new int[]{6}, tree.speculate(new int[]{5}, 4, 4.0, 0.0, 0.0).tokenIds());
    }

    @Test
    public void suffixTreeRemoveUndoesDepthCappedRepeats() {
        SuffixTree tree = new SuffixTree(3);
        tree.extend(7, new int[]{1, 1, 1, 1, 1, 2, 1, 1});
        tree.extend(8, new int[]{1, 1, 3});
        tree.extend(7, new int[]{1, 2});
        assertTrue(tree.remove(7));
        // what is left counts as if only sequence 8 had been entered
        SuffixTree only = new SuffixTree(3);
        only.extend(8, new int[]{1, 1, 3});
        for (int[] pattern : new int[][]{{1}, {1, 1}, {1, 1, 1}, {2}, {3}}) {
            SuffixTree.Draft a = tree.speculate(pattern, 4, 4.0, 0.0, 0.0);
            SuffixTree.Draft b = only.speculate(pattern, 4, 4.0, 0.0, 0.0);
            assertArrayEquals(a.tokenIds(), b.tokenIds());
            assertArrayEquals(a.probabilities(), b.probabilities(), 0.0);
        }
    }

    @Test
    public void aPastResponseDraftsForTheNextRequest() {
        FakeModel target = new FakeModel();
        SuffixDrafter d = drafter(target, SuffixDrafter.DEFAULT_MAX_CACHED_REQUESTS);
        generate(d, target, new int[]{1, 2}, 7, 8, 9);
        assertEquals(1, d.cachedRequests());

        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{5});
        // the new prompt never saw 7; the cached response [7, 8, 9] continues it
        assertArrayEquals(new int[]{8, 9}, d.draft(7, 4));
        assertEquals(2, d.cachedRequests());
    }

    @Test
    public void withoutTheRequestTreeOnlyTheRequestItselfDrafts() {
        FakeModel target = new FakeModel();
        SuffixDrafter d = drafter(target, 0);
        generate(d, target, new int[]{1, 2}, 7, 8, 9);
        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{5});
        assertArrayEquals(new int[0], d.draft(7, 4));
        assertEquals(0, d.cachedRequests());
        assertFalse(d.toString(), d.toString().contains("requests="));
    }

    @Test
    public void theOldestRequestsAreEvictedFirst() {
        FakeModel target = new FakeModel();
        SuffixDrafter d = drafter(target, 2);
        generate(d, target, new int[]{1}, 7, 8, 9);
        generate(d, target, new int[]{1}, 7, 3);
        assertEquals(2, d.cachedRequests());

        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{5});
        // starting the third request evicted the first: [7, 3] answers, [7, 8, 9] is gone
        assertArrayEquals(new int[]{3}, d.draft(7, 4));
        assertEquals(2, d.cachedRequests());
        assertTrue(d.toString(), d.toString().contains("2/2 requests"));
    }

    @Test
    public void acceptedDraftTokensJoinTheResponse() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2, 3, 1, 2});
        SuffixDrafter d = drafter(target, 4);
        // the local tree continues [1, 2, 3] with [1, 2]; both are accepted and end the generation
        assertArrayEquals(new int[]{1, 2}, d.draft(3, 4));
        d.accept(2);

        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{9});
        // the cached response is [3, 1, 2]: the sampled token and the two accepted ones
        assertArrayEquals(new int[]{1, 2}, d.draft(3, 4));
    }

    @Test
    public void tokensCommittedBetweenDraftsJoinTheResponse() {
        FakeModel target = new FakeModel().prefill(new int[]{1});
        SuffixDrafter d = drafter(target, 4);
        d.draft(5, 4);
        d.accept(0);
        // 5 is committed, then 6 and 7 decode without a draft call
        target.backend.memory.append(new int[]{5, 6, 7});
        d.draft(8, 4);
        d.accept(0);

        d.reset();
        target.backend.memory.reset();
        target.backend.memory.prefill(new int[]{0});
        assertArrayEquals("the cached response is [5, 6, 7, 8]", new int[]{7, 8}, d.draft(6, 4));
    }

    @Test
    public void aHistoryShrunkBelowTheCachedResponseStartsANewRequest() {
        FakeModel target = new FakeModel().prefill(new int[]{1, 2});
        SuffixDrafter d = drafter(target, 4);
        d.draft(7, 4);
        target.backend.memory.append(new int[]{7});
        d.draft(8, 4);
        assertEquals(1, d.cachedRequests());
        target.backend.memory.rollback(2);
        d.draft(4, 4);
        assertEquals(2, d.cachedRequests());
    }

    @Test
    public void theCapacityIsValidated() {
        assertThrows(IllegalArgumentException.class,
                () -> SuffixDrafter.Options.builder().maxCachedRequests(-1).build());
        assertEquals(SuffixDrafter.DEFAULT_MAX_CACHED_REQUESTS,
                SuffixDrafter.Options.builder().build().maxCachedRequests());
        assertEquals(5, SuffixDrafter.Options.builder().maxCachedRequests(5).build().toBuilder().build()
                .maxCachedRequests());
        assertThrows(IllegalArgumentException.class, () -> new SuffixRequestCache(8, 0));
        SuffixRequestCache cache = new SuffixRequestCache(8, 1);
        assertThrows("tokens need a running request", IllegalStateException.class, () -> cache.append(0, 3));
        int id = cache.start();
        cache.end(id);
        assertThrows("an ended request takes no tokens", IllegalStateException.class, () -> cache.append(id, 3));
    }

    @Test
    public void severalRequestsRunInOneTree() {
        SuffixRequestCache cache = new SuffixRequestCache(8, 4);
        int a = cache.start();
        int b = cache.start();
        cache.append(a, new int[]{1, 2, 3});
        cache.append(b, new int[]{1, 2, 4});
        assertEquals(2, cache.cachedRequests());
        assertEquals("both running responses continue [1, 2]", 0.5,
                cache.speculate(new int[]{1, 2}, 4, 4.0, 0.0, 0.0).probabilities()[0], 1e-12);
        cache.end(a);
        assertTrue(cache.running(b));
        assertFalse(cache.running(a));
        assertEquals("an ended response stays cached", 2, cache.cachedRequests());
    }

    @Test
    public void anEvictedRunningRequestStopsAddingToTheTree() {
        SuffixRequestCache cache = new SuffixRequestCache(8, 1);
        int a = cache.start();
        cache.append(a, new int[]{1, 2});
        int b = cache.start();
        assertEquals(1, cache.cachedRequests());
        assertTrue("the evicted request still runs", cache.running(a));
        cache.append(a, new int[]{1, 3});
        cache.append(b, 9);
        assertArrayEquals("neither the evicted response nor its later tokens draft", new int[0],
                cache.speculate(new int[]{1}, 4, 4.0, 0.0, 0.0).tokenIds());
    }

    @Test
    public void drafterSharingATreeDraftFromEachOthersResponses() {
        SuffixDrafter.Options options = SuffixDrafter.Options.builder().maxSpecFactor(4).build();
        SuffixRequestCache shared = SuffixDrafter.sharedRequests(options);
        FakeModel first = new FakeModel().prefill(new int[]{1});
        FakeModel second = new FakeModel().prefill(new int[]{2});
        SuffixDrafter a = SuffixDrafter.of(options, shared);
        SuffixDrafter b = SuffixDrafter.of(options, shared);
        a.bindTarget(first);
        b.bindTarget(second);
        for (int token : new int[]{7, 8, 9}) {
            a.draft(token, 4);
            a.accept(0);
            first.backend.memory.append(new int[]{token});
        }
        assertArrayEquals("the other drafter's running response continues 7", new int[]{8, 9}, b.draft(7, 4));
        assertEquals(2, a.cachedRequests());
        a.close();
        assertEquals("closing a sharing drafter keeps the shared tree", 2, b.cachedRequests());
        b.close();

        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.of(
                SuffixDrafter.Options.builder().maxCachedRequests(0).build(), shared));
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.of(
                SuffixDrafter.Options.builder().maxTreeDepth(4).build(), shared));
        assertThrows(IllegalArgumentException.class, () -> SuffixDrafter.sharedRequests(
                SuffixDrafter.Options.builder().maxCachedRequests(0).build()));
    }
}
