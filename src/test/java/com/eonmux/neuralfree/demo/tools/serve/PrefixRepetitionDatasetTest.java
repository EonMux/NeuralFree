package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code prefix_repetition} dataset: {@code numPrefixes} shared prefixes of the configured
 * length, a fresh suffix per request, and a shuffle that interleaves the groups unless disabled.
 */
public class PrefixRepetitionDatasetTest {

    private static ServeBenchOptions options(boolean shuffle) {
        return ServeBenchOptions.builder()
                .datasetName("prefix_repetition")
                .numPrompts(10)
                .seed(4)
                .prefixRepetitionPrefixLen(4)
                .prefixRepetitionSuffixLen(3)
                .prefixRepetitionNumPrefixes(2)
                .prefixRepetitionOutputLen(7)
                .disableShuffle(!shuffle)
                .requestIdPrefix("p-")
                .build();
    }

    private static PrefixRepetitionDataset dataset(ServeBenchOptions opts) {
        return new PrefixRepetitionDataset(opts, 100, new int[]{0});
    }

    @Test
    public void theWorkloadSharesExactlyTheConfiguredPrefixes() {
        List<BenchSample> samples = dataset(options(true)).sample();
        assertEquals(10, samples.size());
        Set<String> prefixes = new HashSet<>();
        for (BenchSample sample : samples) {
            assertEquals(7, sample.outputLen());
            assertEquals(7, sample.promptLen());
            int[] head = new int[4];
            System.arraycopy(sample.tokens(), 0, head, 0, 4);
            prefixes.add(java.util.Arrays.toString(head));
        }
        assertEquals(2, prefixes.size());
    }

    @Test
    public void withoutAShuffleThePrefixesArriveInGroups() {
        List<BenchSample> samples = dataset(options(false)).sample();
        int switches = 0;
        for (int i = 1; i < samples.size(); i++) {
            int[] previous = java.util.Arrays.copyOf(samples.get(i - 1).tokens(), 4);
            int[] current = java.util.Arrays.copyOf(samples.get(i).tokens(), 4);
            if (!java.util.Arrays.equals(previous, current)) {
                switches++;
            }
        }
        assertEquals("two contiguous groups", 1, switches);
    }

    @Test
    public void theSuffixesOfOnePrefixDiffer() {
        List<BenchSample> samples = dataset(options(false)).sample();
        assertTrue(!java.util.Arrays.equals(samples.get(0).tokens(), samples.get(1).tokens()));
    }

    @Test
    public void aSeedReproducesTheWorkload() {
        assertEquals(dataset(options(true)).sample(), dataset(options(true)).sample());
    }

    @Test
    public void theSuffixUsesOnlyAllowedIds() {
        for (BenchSample sample : dataset(options(true)).sample()) {
            for (int id : sample.tokens()) {
                assertTrue("token " + id + " is allowed", id > 0 && id < 100);
            }
        }
    }

    @Test
    public void onlyWholePrefixGroupsAreSampled() {
        ServeBenchOptions opts = ServeBenchOptions.builder()
                .datasetName("prefix_repetition").numPrompts(10).seed(4)
                .prefixRepetitionPrefixLen(4).prefixRepetitionSuffixLen(3)
                .prefixRepetitionNumPrefixes(3).prefixRepetitionOutputLen(7)
                .requestIdPrefix("p-").build();
        assertEquals("the reference samples whole prefix groups only", 9, dataset(opts).sample().size());
    }

    @Test
    public void fewerPromptsThanPrefixesIsRefused() {
        ServeBenchOptions opts = ServeBenchOptions.builder()
                .numPrompts(1).prefixRepetitionNumPrefixes(5).build();
        try {
            dataset(opts).sample();
            fail("one prompt cannot carry five prefixes");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("num prompts"));
        }
    }
}
