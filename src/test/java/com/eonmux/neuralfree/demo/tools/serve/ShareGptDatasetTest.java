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
 * The {@code sharegpt} dataset: the first turn becomes the prompt, the second turn's token count (or
 * the fixed {@code --sharegpt-output-len}) the output, the reference's pruning bounds drop short and
 * long conversations, and the workload oversamples to {@code --num-prompts} unless told not to.
 */
public class ShareGptDatasetTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A whitespace tokenizer, so the test controls the token counts exactly. */
    private static final BenchTokenizer WORDS = text -> {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) {
            return new int[0];
        }
        return new int[trimmed.split("\\s+").length];
    };

    private File datasetFile(String json) throws Exception {
        File file = tmp.newFile("sharegpt.json");
        Files.write(file.toPath(), json.getBytes("UTF-8"));
        return file;
    }

    private static ServeBenchOptions.Builder builder(File file) {
        return ServeBenchOptions.builder()
                .datasetName("sharegpt")
                .datasetPath(file.getAbsolutePath())
                .numPrompts(4)
                .seed(2)
                .requestIdPrefix("s-");
    }

    @Test
    public void theFirstTurnIsThePromptAndTheSecondSetsTheOutput() {
        try {
            File file = datasetFile("[{\"conversations\": ["
                    + "{\"from\": \"human\", \"value\": \"one two three four\"},"
                    + "{\"from\": \"gpt\", \"value\": \"five six seven eight nine\"}]}]");
            ServeBenchOptions opts = builder(file).numPrompts(1).build();
            List<BenchSample> samples = new ShareGptDataset(opts, WORDS).sample();
            assertEquals(1, samples.size());
            assertEquals("one two three four", samples.get(0).text());
            assertEquals(4, samples.get(0).promptLen());
            assertEquals(5, samples.get(0).outputLen());
            assertEquals("s-0", samples.get(0).requestId());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aFixedOutputLengthOverridesTheSecondTurn() {
        try {
            File file = datasetFile("[{\"conversations\": ["
                    + "{\"from\": \"human\", \"value\": \"one two three four\"},"
                    + "{\"from\": \"gpt\", \"value\": \"five six seven eight nine\"}]}]");
            ServeBenchOptions opts = builder(file).numPrompts(1).sharegptOutputLen(2).build();
            List<BenchSample> samples = new ShareGptDataset(opts, WORDS).sample();
            assertEquals(2, samples.get(0).outputLen());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void conversationsOutsideThePruningBoundsAreDropped() {
        try {
            File file = datasetFile("["
                    + "{\"conversations\": [{\"from\": \"human\", \"value\": \"hi\"},"
                    + "{\"from\": \"gpt\", \"value\": \"a b c d\"}]},"
                    + "{\"conversations\": [{\"from\": \"human\", \"value\": \"one two three four\"},"
                    + "{\"from\": \"gpt\", \"value\": \"a b c d\"}]},"
                    + "{\"conversations\": [{\"from\": \"human\", \"value\": \"only one turn\"}]}]");
            ServeBenchOptions opts = builder(file).numPrompts(4).noOversample(true).build();
            List<BenchSample> samples = new ShareGptDataset(opts, WORDS).sample();
            assertEquals("only the valid conversation survives", 1, samples.size());
            assertEquals("one two three four", samples.get(0).text());
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void theWorkloadOversamplesToTheRequestedCount() {
        try {
            File file = datasetFile("[{\"conversations\": ["
                    + "{\"from\": \"human\", \"value\": \"one two three four\"},"
                    + "{\"from\": \"gpt\", \"value\": \"a b c d\"}]}]");
            List<BenchSample> samples = new ShareGptDataset(builder(file).build(), WORDS).sample();
            assertEquals(4, samples.size());
            for (int i = 0; i < samples.size(); i++) {
                assertEquals("s-" + i, samples.get(i).requestId());
            }
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    @Test
    public void aDatasetPathIsRequired() {
        try {
            new ShareGptDataset(ServeBenchOptions.builder().datasetName("sharegpt").build(), WORDS).sample();
            fail("--dataset-path is required");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--dataset-path"));
        }
    }

    @Test
    public void aFileThatIsNotAnArrayIsRefused() throws Exception {
        File file = datasetFile("{\"conversations\": []}");
        try {
            new ShareGptDataset(builder(file).build(), WORDS).sample();
            fail("a non-array dataset must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("JSON array"));
        }
    }

    @Test
    public void anEntryThatIsNotAnObjectIsRefused() throws Exception {
        File file = datasetFile("[1, 2]");
        try {
            new ShareGptDataset(builder(file).build(), WORDS).sample();
            fail("a non-object entry must be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("object"));
        }
    }
}
