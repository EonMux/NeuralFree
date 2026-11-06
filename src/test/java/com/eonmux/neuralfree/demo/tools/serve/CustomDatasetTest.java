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
 * The {@code custom} dataset: the JSONL records' prompts counted through the tokenizer, their
 * {@code output_tokens} used when no fixed length is given, the extension gate, and the refusal of a
 * record that lacks the output length it needs.
 */
public class CustomDatasetTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final BenchTokenizer WORDS = text -> new int[text.trim().split("\\s+").length];

    private File jsonl(String lines) throws Exception {
        File file = tmp.newFile("custom.jsonl");
        Files.write(file.toPath(), lines.getBytes("UTF-8"));
        return file;
    }

    private static ServeBenchOptions.Builder builder(File file) {
        return ServeBenchOptions.builder()
                .datasetName("custom")
                .datasetPath(file.getAbsolutePath())
                .numPrompts(2)
                .seed(5)
                .requestIdPrefix("c-");
    }

    @Test
    public void theRecordsCarryTheirPromptsAndOutputLengths() {
        try {
            File file = jsonl("{\"prompt\": \"a b c d\", \"output_tokens\": 9}\n"
                    + "{\"prompt\": \"e f g h\", \"output_tokens\": 3}\n");
            ServeBenchOptions opts = builder(file).customOutputLen(-1).build();
            List<BenchSample> samples = new CustomDataset(opts, WORDS).sample();
            assertEquals(2, samples.size());
            for (BenchSample sample : samples) {
                assertEquals(4, sample.promptLen());
                assertTrue(sample.outputLen() == 9 || sample.outputLen() == 3);
            }
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aFixedOutputLengthOverridesTheRecord() {
        try {
            File file = jsonl("{\"prompt\": \"a b c d\", \"output_tokens\": 9}\n");
            ServeBenchOptions opts = builder(file).numPrompts(1).customOutputLen(17).build();
            List<BenchSample> samples = new CustomDataset(opts, WORDS).sample();
            assertEquals(17, samples.get(0).outputLen());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aRecordWithoutItsOutputTokensIsRefusedByRecordNumber() throws Exception {
        try {
            File file = jsonl("{\"prompt\": \"a b c d\"}\n");
            new CustomDataset(builder(file).customOutputLen(-1).build(), WORDS).sample();
            fail("a record without output_tokens must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("output_tokens"));
            assertTrue(e.getMessage(), e.getMessage().contains("record 0"));
        }
    }

    @Test
    public void onlyJsonlIsAccepted() throws Exception {
        File file = tmp.newFile("custom.json");
        Files.write(file.toPath(), "[]".getBytes("UTF-8"));
        try {
            new CustomDataset(builder(file).build(), WORDS).sample();
            fail("a non-JSONL file must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("JSONL"));
        }
    }

    @Test
    public void aRecordWithoutAPromptIsRefused() throws Exception {
        File file = jsonl("{\"output_tokens\": 4}\n");
        try {
            new CustomDataset(builder(file).build(), WORDS).sample();
            fail("a record without a prompt must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("prompt"));
        }
    }

    @Test
    public void anEmptyFileIsRefused() throws Exception {
        File file = jsonl("\n\n");
        try {
            new CustomDataset(builder(file).build(), WORDS).sample();
            fail("an empty dataset must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no record"));
        }
    }

    @Test
    public void theWorkloadOversamplesWithSequentialIds() {
        try {
            File file = jsonl("{\"prompt\": \"a b c d\", \"output_tokens\": 4}\n");
            List<BenchSample> samples = new CustomDataset(builder(file).build(), WORDS).sample();
            assertEquals(2, samples.size());
            assertEquals("c-0", samples.get(0).requestId());
            assertEquals("c-1", samples.get(1).requestId());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aFixedOutputLengthSkipsTheRecordField() {
        try {
            File file = jsonl("{\"prompt\": \"a b c d\"}\n");
            List<BenchSample> samples = new CustomDataset(builder(file).numPrompts(1).customOutputLen(6)
                    .build(), WORDS).sample();
            assertEquals(6, samples.get(0).outputLen());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }
}
