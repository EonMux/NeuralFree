package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code burstgpt} dataset: a BurstGPT CSV trace filtered to the GPT-4 rows with a positive
 * response length, replayed as synthetic token-id prompts of the trace's request lengths and the
 * trace's response lengths.
 */
public class BurstGptDatasetTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private File csv(String content) throws Exception {
        File file = tmp.newFile("burstgpt.csv");
        Files.write(file.toPath(), content.getBytes("UTF-8"));
        return file;
    }

    private static ServeBenchOptions.Builder builder(File file) {
        return ServeBenchOptions.builder()
                .datasetName("burstgpt")
                .datasetPath(file.getAbsolutePath())
                .numPrompts(4)
                .seed(1)
                .requestIdPrefix("b-");
    }

    @Test
    public void theGpt4RowsWithAResponseBecomeTheWorkload() {
        try {
            File file = csv("Timestamp,Model,Request tokens,Response tokens\n"
                    + "2024-01-01 00:00:00,GPT-4,120,40\n"
                    + "2024-01-01 00:00:01,GPT-4,80,20\n"
                    + "2024-01-01 00:00:02,GPT-3.5,10,10\n"
                    + "2024-01-01 00:00:03,GPT-4,50,0\n");
            List<BenchSample> samples = new BurstGptDataset(builder(file).build(), 64).sample();
            assertEquals(4, samples.size());
            for (BenchSample sample : samples) {
                assertTrue("request tokens from the trace: " + sample.promptLen(),
                        sample.promptLen() == 120 || sample.promptLen() == 80);
                assertTrue("response tokens from the trace: " + sample.outputLen(),
                        sample.outputLen() == 40 || sample.outputLen() == 20);
                assertEquals(sample.promptLen(), sample.tokens().length);
            }
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void theTokensStayInsideTheVocabulary() {
        try {
            File file = csv("Timestamp,Model,Request tokens,Response tokens\n"
                    + "2024-01-01 00:00:00,GPT-4,7,3\n");
            List<BenchSample> samples = new BurstGptDataset(builder(file).numPrompts(2).build(), 5).sample();
            for (BenchSample sample : samples) {
                for (int id : sample.tokens()) {
                    assertTrue("token " + id + " in [0, 5)", id >= 0 && id < 5);
                }
            }
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aWrongHeaderIsRefusedWithItsLine() throws Exception {
        File file = csv("Timestamp,Name,Request,Response\n2024-01-01,GPT-4,1,1\n");
        try {
            new BurstGptDataset(builder(file).build(), 64).sample();
            fail("a wrong header must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("header"));
        }
    }

    @Test
    public void aTraceWithoutGpt4RowsIsRefused() throws Exception {
        File file = csv("Timestamp,Model,Request tokens,Response tokens\n"
                + "2024-01-01,GPT-3.5,10,10\n");
        try {
            new BurstGptDataset(builder(file).build(), 64).sample();
            fail("an empty trace must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no GPT-4 row"));
        }
    }

    @Test
    public void aShortRowIsRefusedWithItsLine() throws Exception {
        File file = csv("Timestamp,Model,Request tokens,Response tokens\n2024-01-01,GPT-4,10\n");
        try {
            new BurstGptDataset(builder(file).build(), 64).sample();
            fail("a short row must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("line 2"));
        }
    }

    @Test
    public void aDatasetPathIsRequired() {
        try {
            new BurstGptDataset(ServeBenchOptions.builder().datasetName("burstgpt").build(), 64).sample();
            fail("--dataset-path is required");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--dataset-path"));
        }
    }
}
