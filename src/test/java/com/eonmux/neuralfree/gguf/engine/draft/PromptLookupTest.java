package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.draft.LookupTargetFixture.FakeModel;
import org.junit.Test;

import java.util.Arrays;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link PromptLookup}: one lookup size is llama.cpp's {@code common_ngram_simple_draft}; a size range is vLLM's
 * {@code NgramProposer} ({@code prompt_lookup_min} / {@code prompt_lookup_max}), checked against the examples of
 * vLLM's tests and against a direct statement of its result (the longest suffix in the range that occurs earlier,
 * at its earliest occurrence) over random histories. {@link NgramDrafter} and the batch drafter draft with it.
 */
public class PromptLookupTest {

    /** {@code history} followed by the sampled token, as the drafters hand it over. */
    private static int[] range(int[] history, int sampled, int minN, int maxN, int k) {
        int[] inp = Arrays.copyOf(history, history.length + 1);
        inp[history.length] = sampled;
        return PromptLookup.draft(inp, history.length, minN, maxN, k);
    }

    @Test
    public void theRangeProposesAfterTheLongestMatchAtItsEarliestOccurrence() {
        // vLLM tests/v1/spec_decode/test_ngram.py: [1, 2, 3, 4, 1, 2, 3] with min 2, max 2 / 3, k 2
        assertArrayEquals(new int[]{4, 1}, range(new int[]{1, 2, 3, 4, 1, 2}, 3, 2, 3, 2));
        // [1, 2, 3, 4, 1, 2, 3, 5, 6] matches only the 1-gram [6]: none in [2, 3]
        assertArrayEquals(new int[0], range(new int[]{1, 2, 3, 4, 1, 2, 3, 5}, 6, 2, 3, 2));
        // the 3-gram [2, 3, 7] occurs twice: the earliest one proposes [8, 1], not the later [9, 2]
        assertArrayEquals(new int[]{8, 1}, range(new int[]{2, 3, 7, 8, 1, 2, 3, 7, 9, 2, 3}, 7, 1, 4, 2));
        // a longer match wins over a more recent shorter one
        assertArrayEquals(new int[]{5, 6},
                range(new int[]{1, 2, 3, 5, 6, 0, 0, 3, 4, 4, 0, 1, 2}, 3, 1, 4, 2));
    }

    @Test
    public void theRangeCapsTheMatchAtMaxAndTheProposalAtTheHistory() {
        // a run longer than max matches at max; its earliest occurrence proposes up to the end
        assertArrayEquals(new int[]{1, 1, 1, 1}, range(new int[]{1, 1, 1, 1, 1, 1}, 1, 1, 3, 8));
        // the proposal stops at the end of the context, the sampled token included
        assertArrayEquals(new int[]{4, 5, 1, 2, 3}, range(new int[]{1, 2, 3, 4, 5, 1, 2}, 3, 2, 3, 10));
        // a history shorter than min proposes nothing
        assertArrayEquals(new int[0], range(new int[]{4}, 4, 3, 4, 2));
    }

    @Test
    public void theRangeEqualsVllmsResultOverRandomHistories() {
        SplittableRandom rng = new SplittableRandom(20261103L);
        for (int trial = 0; trial < 3000; trial++) {
            int length = 1 + rng.nextInt(40);
            int[] tokens = new int[length];
            int vocab = 2 + rng.nextInt(4);
            for (int i = 0; i < length; i++) {
                tokens[i] = rng.nextInt(vocab);
            }
            int minN = 1 + rng.nextInt(4);
            int maxN = minN + 1 + rng.nextInt(5);
            int k = 1 + rng.nextInt(6);
            int[] got = PromptLookup.draft(tokens, length - 1, minN, maxN, k);
            int[] want = vllmReference(tokens, minN, maxN, k);
            assertArrayEquals(Arrays.toString(tokens) + " min=" + minN + " max=" + maxN + " k=" + k, want, got);
        }
    }

    /**
     * vLLM's documented result stated directly: the longest {@code n} in {@code [minN, maxN]} whose suffix
     * {@code tokens[total - n ..]} also starts at an earlier position {@code s <= total - n - 1}, the smallest such
     * {@code s}, and the up to {@code k} tokens from {@code s + n}.
     */
    private static int[] vllmReference(int[] tokens, int minN, int maxN, int k) {
        int total = tokens.length;
        for (int n = Math.min(maxN, total - 1); n >= minN; n--) {
            for (int s = 0; s <= total - n - 1; s++) {
                if (Arrays.equals(tokens, s, s + n, tokens, total - n, total)) {
                    int start = s + n;
                    return Arrays.copyOfRange(tokens, start, Math.min(total, start + k));
                }
            }
        }
        return new int[0];
    }

    @Test
    public void oneSizeIsLlamaCppsSearch() {
        int[] history = {5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2};
        // pattern [2, 3]: the previous occurrence at position 4 is followed by [4, 5, 6, 7]
        assertArrayEquals(new int[]{4, 5, 6, 7}, range(history, 3, 2, 2, 4));
        // llama.cpp searches backwards from the most recent occurrence; vLLM's range takes the earliest
        int[] twice = {9, 1, 2, 8, 8, 1, 2, 7, 7, 7, 7, 7, 1};
        assertArrayEquals(new int[]{7, 7}, range(twice, 2, 2, 2, 2));
        assertArrayEquals(new int[]{8, 8}, range(twice, 2, 1, 2, 2));
        assertThrows("a single size of 1 is no llama.cpp lookup", IllegalArgumentException.class,
                () -> range(history, 3, 1, 1, 4));
        assertThrows(IllegalArgumentException.class, () -> range(history, 3, 3, 2, 4));
    }

    @Test
    public void theNgramDrafterSearchesTheRangeOfItsOptions() {
        FakeModel target = new FakeModel().prefill(new int[]{9, 1, 2, 8, 8, 1, 2, 7, 7, 7, 7, 7, 1});
        NgramDrafter range = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE)
                .sizeNMin(1).sizeN(2).sizeM(3).build());
        range.bindTarget(target);
        assertTrue(range.options().lookupRange());
        assertTrue(range.toString(), range.toString().contains("sizeNMin=1"));
        assertArrayEquals(new int[]{8, 8, 1}, range.draft(2, 8));
        assertArrayEquals("the round's request caps the proposal", new int[]{8}, range.draft(2, 1));

        NgramDrafter single = NgramDrafter.of(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE)
                .sizeN(2).sizeM(2).build());
        single.bindTarget(target);
        assertFalse(single.options().lookupRange());
        assertEquals(2, single.options().sizeNMin());
        assertArrayEquals(new int[]{7, 7}, single.draft(2, 8));
    }
}
