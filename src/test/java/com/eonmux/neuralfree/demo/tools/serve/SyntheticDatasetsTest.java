package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The synthetic datasets: the random dataset's seeded lengths, shared prefix and allowed-token walk,
 * and the prefix-repetition dataset's shared prefixes with fresh suffixes.
 */
public class SyntheticDatasetsTest {

    private static ServeBenchOptions options(int prompts) {
        return ServeBenchOptions.builder().datasetName("random").numPrompts(prompts).seed(3)
                .randomInputLen(10).randomOutputLen(4).requestIdPrefix("r-").build();
    }

    @Test
    public void theRandomDatasetSamplesItsLengthsAndSkipsSpecialIds() {
        int[] skip = {0, 1, 2};
        RandomDataset dataset = new RandomDataset(options(5), 100, skip);
        List<BenchSample> samples = dataset.sample();
        assertEquals(5, samples.size());
        Set<Integer> skipped = new HashSet<>();
        for (int id : skip) {
            skipped.add(id);
        }
        for (BenchSample sample : samples) {
            assertTrue(sample.hasTokens());
            assertEquals(10, sample.promptLen());
            assertEquals(10, sample.tokens().length);
            assertEquals(4, sample.outputLen());
            for (int token : sample.tokens()) {
                assertFalse("a synthetic prompt never carries a special id", skipped.contains(token));
                assertTrue(token >= 0 && token < 100);
            }
        }
        assertEquals("r-", samples.get(0).requestId().substring(0, 2));
    }

    @Test
    public void theSameSeedSamplesTheSameWorkload() {
        List<BenchSample> first = new RandomDataset(options(4), 100, new int[]{0}).sample();
        List<BenchSample> second = new RandomDataset(options(4), 100, new int[]{0}).sample();
        assertEquals(first, second);
        List<BenchSample> other = new RandomDataset(ServeBenchOptions.builder().datasetName("random")
                .numPrompts(4).seed(4).randomInputLen(10).randomOutputLen(4).requestIdPrefix("r-").build(),
                100, new int[]{0}).sample();
        assertNotEquals(first, other);
    }

    @Test
    public void theRandomPrefixIsSharedAndTheBodyWalksTheAllowedTokens() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("random").numPrompts(3).seed(1)
                .randomInputLen(6).randomOutputLen(2).randomPrefixLen(4).build();
        List<BenchSample> samples = new RandomDataset(opts, 50, new int[]{}).sample();
        int[] prefix = new int[4];
        System.arraycopy(samples.get(0).tokens(), 0, prefix, 0, 4);
        for (BenchSample sample : samples) {
            for (int i = 0; i < 4; i++) {
                assertEquals("the prefix is shared", prefix[i], sample.tokens()[i]);
            }
            // the body is (offset + index + j) % allowed: consecutive body tokens differ by one modulo 50
            for (int i = 5; i < sample.tokens().length; i++) {
                assertEquals((sample.tokens()[i - 1] + 1) % 50, sample.tokens()[i]);
            }
        }
    }

    @Test
    public void aRangeRatioSpreadsTheLengthsAroundTheMean() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("random").numPrompts(20).seed(2)
                .randomInputLen(10).randomOutputLen(8)
                .randomRangeRatio(new ServeBenchOptions.RangeRatio(0.5, 0.5)).build();
        for (BenchSample sample : new RandomDataset(opts, 100, new int[]{}).sample()) {
            assertTrue("input in [5, 15]: " + sample.promptLen(),
                    sample.promptLen() >= 5 && sample.promptLen() <= 15);
            assertTrue("output in [4, 12]: " + sample.outputLen(),
                    sample.outputLen() >= 4 && sample.outputLen() <= 12);
        }
    }

    @Test
    public void aVocabularyOfSpecialIdsOnlyIsRefused() {
        try {
            new RandomDataset(options(1), 3, new int[]{0, 1, 2}).sample();
            fail("no allowed token is left");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("no allowed token"));
        }
    }

    @Test
    public void thePrefixRepetitionDatasetSharesPrefixesAndVariesSuffixes() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("prefix_repetition")
                .numPrompts(6).numPrompts(6).seed(5)
                .prefixRepetitionPrefixLen(5).prefixRepetitionSuffixLen(3)
                .prefixRepetitionNumPrefixes(3).prefixRepetitionOutputLen(7)
                .disableShuffle(true).build();
        List<BenchSample> samples = new PrefixRepetitionDataset(opts, 1000, new int[]{}).sample();
        assertEquals(6, samples.size());
        for (int pair = 0; pair < 3; pair++) {
            int[] first = samples.get(pair * 2).tokens();
            int[] second = samples.get(pair * 2 + 1).tokens();
            assertEquals(5 + 3, first.length);
            for (int i = 0; i < 5; i++) {
                assertEquals("a shared prefix", first[i], second[i]);
            }
            for (int i = 5; i < 8; i++) {
                assertNotEquals("a fresh suffix", first[i], second[i]);
            }
            assertEquals(7, samples.get(pair * 2).outputLen());
        }
    }

    @Test
    public void fewerPromptsThanPrefixesIsRefused() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("prefix_repetition")
                .numPrompts(2).prefixRepetitionNumPrefixes(5).build();
        try {
            new PrefixRepetitionDataset(opts, 100, new int[]{}).sample();
            fail("the prompts must cover every prefix");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("num prompts"));
        }
    }

    @Test
    public void anOversampledWorkloadRepeatsSampledRequests() {
        List<BenchSample> sampled = new RandomDataset(options(2), 100, new int[]{}).sample();
        List<BenchSample> grown = ServeBenchDataset.oversample(sampled, 4, "p-", false, 1);
        assertEquals(4, grown.size());
        assertEquals(sampled.get(0).promptLen(), grown.get(2).promptLen());
        assertEquals("p-2", grown.get(2).requestId());
        assertEquals(2, ServeBenchDataset.oversample(sampled, 4, "p-", true, 1).size());
    }

    @Test
    public void theSequencePruningFollowsTheReferenceBounds() {
        assertTrue(ServeBenchDataset.isValidSequence(4, 4, false));
        assertTrue(ServeBenchDataset.isValidSequence(4, 1, true));
        assertFalse(ServeBenchDataset.isValidSequence(3, 4, false));
        assertFalse(ServeBenchDataset.isValidSequence(4, 3, false));
        assertFalse(ServeBenchDataset.isValidSequence(1025, 4, false));
        assertFalse(ServeBenchDataset.isValidSequence(1024, 1025, false));
    }
}
