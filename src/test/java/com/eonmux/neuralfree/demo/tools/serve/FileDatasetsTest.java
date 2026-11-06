package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The file datasets: ShareGPT's first/second turns and the reference's pruning, the custom JSONL
 * records and their {@code output_tokens}, and the BurstGPT CSV trace filtered to its GPT-4 rows.
 */
public class FileDatasetsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A tokenizer that counts one token per character, so the tests can predict every length. */
    private static final BenchTokenizer CHARS = text -> new int[text.length()];

    private Path write(String name, String content) throws IOException {
        Path file = tmp.newFile(name).toPath();
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private ServeBenchOptions.Builder options(String dataset, Path path) {
        return ServeBenchOptions.builder().datasetName(dataset).datasetPath(path.toString())
                .seed(0).disableShuffle(true);
    }

    @Test
    public void sharegptUsesTheFirstTurnAsPromptAndTheSecondAsOutputLength() throws IOException {
        Path file = write("sharegpt.json", "["
                + "{\"conversations\":[{\"from\":\"human\",\"value\":\"hello\"},"
                + "{\"from\":\"gpt\",\"value\":\"world!\"}]},"
                + "{\"conversations\":[{\"from\":\"human\",\"value\":\"solo\"}]},"
                + "{\"conversations\":[{\"from\":\"human\",\"value\":\"abcd\"},"
                + "{\"from\":\"gpt\",\"value\":\"defgh\"}]}]");
        List<BenchSample> samples = new ShareGptDataset(options("sharegpt", file).numPrompts(2).build(),
                CHARS).sample();
        assertEquals(2, samples.size());
        assertEquals("hello", samples.get(0).text());
        assertEquals(5, samples.get(0).promptLen());
        assertEquals(6, samples.get(0).outputLen());
        assertEquals("abcd", samples.get(1).text());
        assertEquals(5, samples.get(1).outputLen());
        assertEquals("bench-", samples.get(0).requestId().substring(0, 6));
    }

    @Test
    public void sharegptOutputLengthCanBeFixed() throws IOException {
        Path file = write("sharegpt2.json", "["
                + "{\"conversations\":[{\"from\":\"human\",\"value\":\"hello\"},"
                + "{\"from\":\"gpt\",\"value\":\"world!\"}]}]");
        List<BenchSample> samples = new ShareGptDataset(options("sharegpt", file).numPrompts(1)
                .sharegptOutputLen(32).build(), CHARS).sample();
        assertEquals(32, samples.get(0).outputLen());
    }

    @Test
    public void sharegptNeedsItsPathAndAnArray() {
        try {
            new ShareGptDataset(ServeBenchOptions.builder().datasetName("sharegpt").build(), CHARS).sample();
            fail("the sharegpt dataset needs --dataset-path");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("--dataset-path"));
        }
        try {
            Path bad = write("bad.json", "{\"not\":\"an array\"}");
            new ShareGptDataset(options("sharegpt", bad).numPrompts(1).build(), CHARS).sample();
            fail("the sharegpt file must be an array");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("JSON array"));
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void theCustomDatasetReadsItsRecords() throws IOException {
        Path file = write("custom.jsonl", "{\"prompt\":\"abcd\",\"output_tokens\":5}\n"
                + "{\"prompt\":\"efghij\",\"output_tokens\":7}\n");
        List<BenchSample> samples = new CustomDataset(options("custom", file).numPrompts(2)
                .customOutputLen(-1).build(), CHARS).sample();
        assertEquals(2, samples.size());
        assertEquals("abcd", samples.get(0).text());
        assertEquals(4, samples.get(0).promptLen());
        assertEquals(5, samples.get(0).outputLen());
        assertEquals(7, samples.get(1).outputLen());
    }

    @Test
    public void aFixedCustomOutputLengthOverridesTheRecord() throws IOException {
        Path file = write("custom2.jsonl", "{\"prompt\":\"abcd\",\"output_tokens\":5}\n");
        List<BenchSample> samples = new CustomDataset(options("custom", file).numPrompts(1)
                .customOutputLen(12).build(), CHARS).sample();
        assertEquals(12, samples.get(0).outputLen());
    }

    @Test
    public void aRecordWithoutItsOutputLengthIsRefusedWhereOneIsNeeded() throws IOException {
        Path file = write("custom3.jsonl", "{\"prompt\":\"abcd\"}\n");
        try {
            new CustomDataset(options("custom", file).numPrompts(1).customOutputLen(-1).build(), CHARS)
                    .sample();
            fail("the record needs output_tokens");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("output_tokens"));
        }
    }

    @Test
    public void theCustomDatasetTakesJsonlOnly() throws IOException {
        Path file = write("custom.json", "{\"prompt\":\"abcd\",\"output_tokens\":5}\n");
        try {
            new CustomDataset(options("custom", file).numPrompts(1).build(), CHARS).sample();
            fail("the custom dataset takes the JSONL format");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("JSONL"));
        }
    }

    @Test
    public void burstgptReplaysTheGpt4RowsWithPositiveResponses() throws IOException {
        Path file = write("burstgpt.csv", "Timestamp,Model,Request tokens,Response tokens\n"
                + "0,GPT-4,4,5\n"
                + "1,gpt-3.5,6,7\n"
                + "2,GPT-4,8,0\n"
                + "3,GPT-4,9,10\n");
        List<BenchSample> samples = new BurstGptDataset(options("burstgpt", file).numPrompts(6).build(), 100)
                .sample();
        assertEquals(6, samples.size());
        for (BenchSample sample : samples) {
            assertTrue(sample.hasTokens());
            int len = sample.tokens().length;
            assertEquals(sample.promptLen(), len);
            if (len == 4) {
                assertEquals(5, sample.outputLen());
            } else if (len == 9) {
                assertEquals(10, sample.outputLen());
            } else {
                fail("a GPT-4 row with a positive response is expected, got length " + len);
            }
        }
    }

    @Test
    public void burstgptNeedsItsHeader() throws IOException {
        Path file = write("bad.csv", "a,b,c,d\n0,GPT-4,4,5\n");
        try {
            new BurstGptDataset(options("burstgpt", file).numPrompts(1).build(), 100).sample();
            fail("the trace needs its header");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("header"));
        }
    }

    @Test
    public void theDatasetFactoryRoutesEveryName() {
        assertTrue(ServeBenchDataset.create("random", ServeBenchOptions.builder().numPrompts(1).build(),
                CHARS, 10, new int[]{}) instanceof RandomDataset);
        assertTrue(ServeBenchDataset.create("prefix_repetition",
                ServeBenchOptions.builder().numPrompts(1).build(), CHARS, 10, new int[]{})
                instanceof PrefixRepetitionDataset);
        assertTrue(ServeBenchDataset.create("hf", ServeBenchOptions.builder().datasetPath(".").build(),
                CHARS, 10, new int[]{}) instanceof HfDataset);
        try {
            ServeBenchDataset.create("bogus", ServeBenchOptions.builder().build(), CHARS, 10, new int[]{});
            fail("an unknown dataset name is refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("unknown dataset name"));
        }
    }
}
