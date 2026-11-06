package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code random} dataset: seeded and reproducible, its lengths drawn from the range ratio's
 * integer interval, its shared prefix drawn once, and its tokens inside the vocabulary minus the
 * tokenizer's special ids.
 */
public class RandomDatasetTest {

    private static ServeBenchOptions options() {
        return ServeBenchOptions.builder()
                .datasetName("random")
                .numPrompts(20)
                .seed(11)
                .randomInputLen(64)
                .randomOutputLen(8)
                .randomRangeRatio(ServeBenchOptions.parseRangeRatio("0.25"))
                .randomPrefixLen(4)
                .requestIdPrefix("r-")
                .build();
    }

    private static RandomDataset dataset(ServeBenchOptions opts) {
        return new RandomDataset(opts, 1000, new int[]{0, 1, 2});
    }

    @Test
    public void everyRequestCarriesTheSampledLengths() {
        List<BenchSample> samples = dataset(options()).sample();
        assertEquals(20, samples.size());
        for (BenchSample sample : samples) {
            assertTrue(sample.hasTokens());
            assertTrue("input below the interval: " + sample.promptLen(), sample.promptLen() >= 4 + 48);
            assertTrue("input above the interval: " + sample.promptLen(), sample.promptLen() <= 4 + 80);
            assertTrue("output below the interval: " + sample.outputLen(), sample.outputLen() >= 6);
            assertTrue("output above the interval: " + sample.outputLen(), sample.outputLen() <= 10);
            assertEquals(sample.tokens().length, sample.promptLen());
        }
        assertEquals("r-0", samples.get(0).requestId());
    }

    @Test
    public void theTokensStayInsideTheAllowedVocabulary() {
        for (BenchSample sample : dataset(options()).sample()) {
            for (int id : sample.tokens()) {
                assertFalse("a special id reached a prompt: " + id, id == 0 || id == 1 || id == 2);
                assertTrue(id >= 0 && id < 1000);
            }
        }
    }

    @Test
    public void theSharedPrefixIsIdenticalAcrossRequests() {
        List<BenchSample> samples = dataset(options()).sample();
        int[] prefix = new int[4];
        System.arraycopy(samples.get(0).tokens(), 0, prefix, 0, 4);
        for (BenchSample sample : samples) {
            int[] head = new int[4];
            System.arraycopy(sample.tokens(), 0, head, 0, 4);
            assertArrayEquals(prefix, head);
        }
    }

    @Test
    public void aSeedReproducesTheSameWorkload() {
        List<BenchSample> first = dataset(options()).sample();
        List<BenchSample> second = dataset(options()).sample();
        assertEquals(first, second);
    }

    @Test
    public void anotherSeedSamplesAnotherWorkload() {
        List<BenchSample> first = dataset(options()).sample();
        List<BenchSample> second = dataset(ServeBenchOptions.builder()
                .numPrompts(20).seed(12).randomInputLen(64).randomOutputLen(8).randomPrefixLen(4)
                .requestIdPrefix("r-").build()).sample();
        assertFalse(first.equals(second));
    }

    @Test
    public void theRequestIdsAreUnique() {
        Set<String> ids = new HashSet<>();
        for (BenchSample sample : dataset(options()).sample()) {
            assertTrue(ids.add(sample.requestId()));
        }
    }

    @Test
    public void aVocabularyOfOnlySpecialIdsIsRefused() {
        try {
            new RandomDataset(ServeBenchOptions.builder().numPrompts(1).build(), 2, new int[]{0, 1}).sample();
            fail("a vocabulary with no allowed id must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("allowed token"));
        }
    }
}
