package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@code BatchedDemo} over the synthetic two-layer llama: {@code -np} sequences continue one prompt through
 * the batch engine, the streams report as llama.cpp's {@code sequence i:} blocks, a greedy run gives every
 * stream the same tokens (the reference's one-seed chains), {@code -n} counts the whole sequence so the
 * decoded total follows the budget, and the budget and {@code n_kv_req} refusals name their flags. The load
 * flags LLMDemo shares reach the engine configuration over the demo's paged memory.
 */
public class BatchedDemoTest {

    private static final int CTX = 64;
    /** The CPU engine the runs open on. */
    private static final EngineConfig CPU = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();

    private static Path model() throws IOException {
        return SyntheticModels.llama().writeTempFile();
    }

    private static BatchedDemo.Plan plan(Path model, String prompt, int maxTokens, int parallel) {
        return new BatchedDemo.Plan(model, CPU, prompt, maxTokens, parallel, CTX, 8, 7L, 0f, 0, 1f);
    }

    private static int decodedCount(String out) {
        Matcher m = Pattern.compile("decoded (\\d+) tokens").matcher(out);
        assertTrue("the decoded line: " + out, m.find());
        return Integer.parseInt(m.group(1));
    }

    @Test
    public void parallelSequencesContinueOnePromptIntoIdenticalGreedyStreams() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        BatchedDemo.simulate(plan(model, "Hello", 12, 2), out);
        String text = out.toString();
        assertTrue("the run header: " + text, text.contains("generating 2 sequences"));
        assertTrue("the header names the backend: " + text, text.contains(", backend = cpu\n"));
        assertTrue(text.contains("sequence 0:"));
        assertTrue(text.contains("sequence 1:"));
        String stream0 = stream(text, 0);
        String stream1 = stream(text, 1);
        assertEquals("greedy chains with one seed decode identically, as llama.cpp's do", stream0, stream1);
        assertTrue("the stream carries the prompt and the generation", stream0.contains("Hello"));

        // -n counts the whole sequence: per stream the engine decodes n_predict - P tokens, and the
        // header's n_kv_req = P + (n_predict - P)*np gives P, so the total is checkable exactly
        Matcher kv = Pattern.compile("n_kv_req = (\\d+)").matcher(text);
        assertTrue(kv.find());
        int promptTokens = 12 * 2 - Integer.parseInt(kv.group(1));
        assertEquals(2 * (12 - promptTokens), decodedCount(text));
    }

    /** The body after {@code sequence i:} up to the next block (identical streams cut identically). */
    private static String stream(String text, int i) {
        int start = text.indexOf("sequence " + i + ":\n\n");
        assertTrue(start >= 0);
        start += ("sequence " + i + ":\n\n").length();
        int end = text.indexOf("\n\n", start);
        assertTrue(end > start);
        return text.substring(start, end);
    }

    @Test
    public void aSingleStreamPrintsItsTokensInsteadOfSequenceBlocks() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        BatchedDemo.simulate(plan(model, "Hello", 12, 1), out);
        String text = out.toString();
        assertFalse("no stream blocks for one sequence", text.contains("sequence 0:"));
        assertTrue("the streamed response", text.contains("Hello"));
        assertTrue(decodedCount(text) > 0);
    }

    @Test
    public void aBudgetThatDoesNotExceedThePromptRefusesByName() throws IOException {
        Path model = model();
        IOException e = null;
        try {
            BatchedDemo.simulate(plan(model, "Hello", 1, 2), new StringBuilder());
        } catch (IOException ex) {
            e = ex;
        }
        assertTrue("refused: " + e, e != null && e.getMessage().contains("counts the whole sequence")
                && e.getMessage().contains("--max-tokens"));
    }

    @Test
    public void aNkvReqBeyondTheContextRefusesByName() throws IOException {
        Path model = model();
        IOException e = null;
        try {
            // n_kv_req = P + (60 - P)*2 = 120 - P > 64 for every prompt shorter than 56 tokens
            BatchedDemo.simulate(plan(model, "Hello", 60, 2), new StringBuilder());
        } catch (IOException ex) {
            e = ex;
        }
        assertTrue("refused: " + e, e != null && e.getMessage().contains("n_kv_req")
                && e.getMessage().contains("reduce --parallel"));
    }

    @Test
    public void theLoadFlagsReachTheEngineConfigOverThePagedMemory() throws Exception {
        String saved = System.getProperty(EngineConfig.PROP_BACKEND);
        System.clearProperty(EngineConfig.PROP_BACKEND);
        try {
            EngineConfig defaults = BatchedDemo.parse(new String[] {"m.gguf", "-np", "3"}).engineConfig();
            assertEquals("LLMDemo's default: the best GPU backend", EngineConfig.BackendMode.AUTO,
                    defaults.backend());
            assertEquals(3, defaults.memory().parallel());
            assertTrue(defaults.memory().kvPaged());

            EngineConfig cfg = BatchedDemo.parse(new String[] {"m.gguf", "--backend", "vulkan", "-ngl", "7",
                    "-dev", "Arc", "--n-cpu-moe", "2", "-fa", "on", "--kv-type", "q8_0", "-c", "256", "-ub", "32",
                    "-np", "2"}).engineConfig();
            assertEquals(EngineConfig.BackendMode.VULKAN, cfg.backend());
            assertEquals(7, cfg.gpuLayers());
            assertEquals("Arc", cfg.deviceFilter());
            assertEquals(2, cfg.nCpuMoe());
            assertEquals(KvDType.Q8_0, cfg.memory().typeK());
            assertEquals(KvDType.Q8_0, cfg.memory().typeV());
            assertEquals(256, cfg.memory().nCtx());
            assertEquals(32, cfg.memory().ubatch());
            assertEquals(2, cfg.memory().parallel());
        } finally {
            if (saved != null) {
                System.setProperty(EngineConfig.PROP_BACKEND, saved);
            }
        }
    }

    @Test
    public void parseRefusesBadInputByName() throws Exception {
        BatchedDemo.UsageException e = null;
        try {
            BatchedDemo.parse(new String[] {});
        } catch (BatchedDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(e != null && e.getMessage().contains("needs a model path"));

        e = null;
        try {
            BatchedDemo.parse(new String[] {"m.gguf", "--frobozz"});
        } catch (BatchedDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(e != null && e.getMessage().contains("--frobozz"));

        e = null;
        try {
            BatchedDemo.parse(new String[] {"m.gguf", "-p"});
        } catch (BatchedDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(e != null && e.getMessage().contains("-p needs a value"));

        e = null;
        try {
            BatchedDemo.parse(new String[] {"m.gguf", "-np", "0"});
        } catch (BatchedDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(e != null && e.getMessage().contains("-np"));

        e = null;
        try {
            BatchedDemo.parse(new String[] {"m.gguf", "-fa", "off"});
        } catch (BatchedDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(e != null && e.getMessage().contains("-fa off is not supported"));
    }
}
