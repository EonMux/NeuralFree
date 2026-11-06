package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@code ParallelDemo} over the synthetic two-layer llama: simulated clients run through the batch engine
 * with one sequence slot each, every request reports llama.cpp's client line with its input and response,
 * the totals carry the prompt, generated and combined throughputs plus the prefix-cache reuse, an oversize
 * system prompt refuses by name against the context, and the parser refuses bad flags by name. The load flags
 * LLMDemo shares reach the engine configuration over the demo's paged memory.
 */
public class ParallelDemoTest {

    private static Path model() throws IOException {
        // the synthetic tokenizer spends about a token per byte: the built-in system text takes ~1000 rows
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nCtxTrain(CTX).build())
                .writeTempFile();
    }

    /** The context of the runs: two requests of ~1050 rows each fit at once. */
    private static final int CTX = 4096;
    /** The CPU engine the runs open on. */
    private static final EngineConfig CPU = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();

    private static ParallelDemo.Plan plan(Path model, int clients, int sequences, int maxTokens, int ctx) {
        return new ParallelDemo.Plan(model, CPU, clients, sequences, maxTokens, 1, ctx, 64, 3L, 0f, 0, 1f);
    }

    private static int count(String text, String marker) {
        int n = 0;
        int at = 0;
        while ((at = text.indexOf(marker, at)) >= 0) {
            n++;
            at += marker.length();
        }
        return n;
    }

    private static long total(String text, String label) {
        Matcher m = Pattern.compile(Pattern.quote(label) + "\\s*:?\\s*(\\d+)").matcher(text);
        assertTrue("the totals line of " + label + ": " + text, m.find());
        return Long.parseLong(m.group(1));
    }

    @Test
    public void clientsRunTheirRequestsThroughTheBatchEngine() throws Exception {
        Path model = model();
        StringBuilder out = new StringBuilder();
        ParallelDemo.simulate(plan(model, 2, 4, 5, CTX), out);
        String text = out.toString();
        assertTrue("the run header names the backend: " + text,
                text.contains("simulating parallel requests from clients (backend = cpu):"));
        assertTrue(text.contains("system tokens = "));
        assertTrue(text, text.contains("cont_batching = 1"));
        assertTrue(text.contains("External prompt file: " + ParallelDemo.BUILT_IN_PROMPTS));
        assertTrue(text.contains("Preemptions:"));
        assertEquals("one start line per request", 4, count(text, "started decoding..."));
        assertEquals("one report per request", 4, count(text, "Input:"));
        assertEquals(4, count(text, "Response:"));
        List<String> builtIn = ParallelDemo.parse(new String[] {model.toString()}).prompts;
        Matcher input = Pattern.compile("Input:    (.*)\n").matcher(text);
        int inputs = 0;
        while (input.find()) {
            assertTrue("every request asks a built-in question: " + input.group(1), builtIn.contains(input.group(1)));
            inputs++;
        }
        assertEquals(4, inputs);

        assertTrue(text.contains("Total prompt tokens:"));
        assertTrue(text.contains("Total gen tokens:"));
        assertTrue(text.contains("Total speed (AVG):"));
        assertTrue(text.contains("Prefix-cache tokens:"));
        assertTrue("every request generated at least its first token", total(text, "Total gen tokens") >= 4);
        assertTrue(total(text, "Total prompt tokens") > 0);
    }

    @Test
    public void anOversizeSystemPromptRefusesByName() throws IOException {
        Path model = model();
        IOException e = null;
        try {
            ParallelDemo.simulate(plan(model, 1, 1, 5, 64), new StringBuilder());
        } catch (IOException ex) {
            e = ex;
        }
        assertTrue("refused: " + e, e != null && e.getMessage().contains("more than the context of 64 rows"));
    }

    @Test
    public void parseDefaultsAndRefusals() throws Exception {
        Path model = model();
        ParallelDemo.Plan p = ParallelDemo.parse(new String[] {model.toString(),
                "-np", "3", "-ns", "7", "-n", "64", "--junk", "2", "-c", "256", "-ub", "16", "-s", "9",
                "--temp", "0.5", "--top-k", "10", "--top-p", "0.9"});
        assertEquals(model.toString(), p.model.toString());
        assertEquals(3, p.clients);
        assertEquals(7, p.sequences);
        assertEquals(64, p.maxTokens);
        assertEquals(2, p.junk);
        assertEquals(256, p.ctx);
        assertEquals(16, p.ubatch);
        assertEquals(9L, p.seed);
        assertEquals(0.5f, p.temperature, 1e-6f);
        assertEquals(10, p.topK);
        assertEquals(0.9f, p.topP, 1e-6f);

        assertRefusal(new String[] {}, "needs a model path");
        assertRefusal(new String[] {model.toString(), "--frobozz"}, "--frobozz");
        assertRefusal(new String[] {model.toString(), "-np", "0"}, "-np");
        assertRefusal(new String[] {model.toString(), "--junk", "-1"}, "--junk");
        assertRefusal(new String[] {model.toString(), "-pps"}, "-pps");
        assertRefusal(new String[] {model.toString(), "-p", " \n "}, "-p holds no prompt line");
        assertRefusal(new String[] {model.toString(), "-f", model.resolveSibling("absent.txt").toString()},
                "-f cannot read");

        ParallelDemo.Plan defaults = ParallelDemo.parse(new String[] {model.toString()});
        assertTrue("llama.cpp batches continuously by default", defaults.contBatching);
        assertEquals(9, defaults.prompts.size());
        assertEquals(ParallelDemo.BUILT_IN_PROMPTS, defaults.promptSource);
        assertEquals(0, ParallelDemo.parse(new String[] {model.toString(), "--junk", "0"}).junk);
        assertTrue(!ParallelDemo.parse(new String[] {model.toString(), "-nocb"}).contBatching);

        ParallelDemo.Plan inline = ParallelDemo.parse(new String[] {model.toString(), "-p", "first?\n\nsecond?\n"});
        assertEquals(java.util.Arrays.asList("first?", "second?"), inline.prompts);
        Path file = model.resolveSibling(model.getFileName() + ".prompts.txt");
        java.nio.file.Files.write(file, "one?\r\ntwo?\nthree?\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        ParallelDemo.Plan fromFile = ParallelDemo.parse(new String[] {model.toString(), "-f", file.toString()});
        assertEquals(java.util.Arrays.asList("one?", "two?", "three?"), fromFile.prompts);
        assertEquals(file.toString(), fromFile.promptSource);
        assertRefusal(new String[] {model.toString(), "--temp", "-1"}, "--temp");
    }

    @Test
    public void theLoadFlagsReachTheEngineConfigOverThePagedMemory() throws Exception {
        String saved = System.getProperty(EngineConfig.PROP_BACKEND);
        System.clearProperty(EngineConfig.PROP_BACKEND);
        try {
            EngineConfig defaults = ParallelDemo.parse(new String[] {"m.gguf", "-np", "4"}).engineConfig();
            assertEquals("LLMDemo's default: the best GPU backend", EngineConfig.BackendMode.AUTO,
                    defaults.backend());
            assertEquals(EngineConfig.GPU_LAYERS_AUTO, defaults.gpuLayers());
            assertEquals("one sequence slot per client", 4, defaults.memory().parallel());
            assertTrue(defaults.memory().kvPaged());

            EngineConfig cfg = ParallelDemo.parse(new String[] {"m.gguf", "--backend", "cuda", "-ngl", "auto",
                    "--cpu-moe", "-ot", "exps=cpu", "-fa", "1", "-ctk", "bf16", "-ctv", "q5_0", "-np", "2",
                    "-c", "512"}).engineConfig();
            assertEquals(EngineConfig.BackendMode.CUDA, cfg.backend());
            assertEquals(EngineConfig.GPU_LAYERS_AUTO, cfg.gpuLayers());
            assertTrue(cfg.cpuMoe());
            assertEquals("exps", cfg.tensorOverrides().get(0).pattern());
            assertEquals(KvDType.BF16, cfg.memory().typeK());
            assertEquals(KvDType.Q5_0, cfg.memory().typeV());
            assertEquals(512, cfg.memory().nCtx());
            assertEquals(2, cfg.memory().parallel());
            assertEquals(EngineConfig.BackendMode.CPU,
                    ParallelDemo.parse(new String[] {"m.gguf", "--cpu"}).engineConfig().backend());
            assertRefusal(new String[] {"m.gguf", "-fa", "false"}, "-fa false is not supported");
            assertRefusal(new String[] {"m.gguf", "-ngl"}, "-ngl needs a value");
        } finally {
            if (saved != null) {
                System.setProperty(EngineConfig.PROP_BACKEND, saved);
            }
        }
    }

    @Test
    public void negativeSeedsCountDownPerClientInTheThirtyTwoBitSeedSpace() {
        assertEquals("a non-negative seed is shared", 9L, ParallelDemo.clientSeed(9L, 0));
        assertEquals(9L, ParallelDemo.clientSeed(9L, 3));
        // -1 is LLAMA_DEFAULT_SEED (0xFFFFFFFF): random for client 0, then 0xFFFFFFFE, 0xFFFFFFFD, ...
        assertEquals(com.eonmux.neuralfree.gguf.engine.SamplerConfig.RANDOM_SEED, ParallelDemo.clientSeed(-1L, 0));
        assertEquals(0xFFFFFFFEL, ParallelDemo.clientSeed(-1L, 1));
        assertEquals(0xFFFFFFFDL, ParallelDemo.clientSeed(-1L, 2));
        assertEquals(0xFFFFFFFBL, ParallelDemo.clientSeed(-5L, 0));
        assertEquals(0xFFFFFFFAL, ParallelDemo.clientSeed(-5L, 1));
    }

    @Test
    public void theRequestTextFollowsParallelCpp() {
        String prompt = ParallelDemo.requestPrompt("Which?", 0, new java.util.Random(1));
        assertTrue(prompt, prompt.startsWith("Transcript of a never ending dialog"));
        assertTrue(prompt, prompt.endsWith("User:\nWhich?\nAssistant:\n"));
        String junk = ParallelDemo.requestPrompt("Which?", 2, new java.util.Random(1));
        assertEquals("two filler blocks, each a User: turn and its answer",
                count(prompt, "User:\n") + 2, count(junk, "User:\n"));
        assertEquals(count(prompt, "\nAssistant:\n ") + 2, count(junk, "\nAssistant:\n "));
        assertTrue("no doubled colon", !junk.contains("User::"));
    }

    @Test
    public void withoutContinuousBatchingRequestsEnterInWaves() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        ParallelDemo.Plan p = new ParallelDemo.Plan(model, CPU, 2, 3, 3, 1, CTX, 64, 3L, 0f, 0, 1f, false,
                java.util.Arrays.asList("Why?"), "-p");
        ParallelDemo.simulate(p, out);
        String text = out.toString();
        assertTrue(text, text.contains("cont_batching = 0"));
        assertEquals(3, count(text, "started decoding..."));
        assertEquals(3, count(text, "Input:    Why?"));
        // the third request starts only after both clients of the first wave reported
        int third = text.indexOf("seq    2,");
        assertTrue(text, third > 0);
        assertEquals("both first-wave reports precede the third start", 2,
                count(text.substring(0, third), "Response:"));
        assertTrue(text.contains("External prompt file: -p"));
    }

    private static void assertRefusal(String[] args, String messagePart) {
        ParallelDemo.UsageException e = null;
        try {
            ParallelDemo.parse(args);
        } catch (ParallelDemo.UsageException ex) {
            e = ex;
        }
        assertTrue(messagePart, e != null && e.getMessage().contains(messagePart));
    }
}
