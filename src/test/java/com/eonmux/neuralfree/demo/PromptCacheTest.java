package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CLI prompt cache ({@code PromptCache} over {@code --prompt-cache}): the first turn saves the session file,
 * later turns leave it alone unless {@code --prompt-cache-all} saves every one, and {@code --prompt-cache-ro}
 * never writes.
 */
public class PromptCacheTest {

    @Rule
    public final TemporaryFolder temp = new TemporaryFolder();

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(4)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** A sink that keeps nothing: the loop requires one, the test reads only the file and result. */
    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    /** {@code real}, counting every {@code saveSession} the cache performs. */
    private static EngineModel countingSaves(EngineModel real, AtomicInteger saves) {
        return (EngineModel) Proxy.newProxyInstance(EngineModel.class.getClassLoader(),
                new Class<?>[]{EngineModel.class}, (proxy, method, args) -> {
                    if (method.getName().equals("saveSession")) {
                        saves.incrementAndGet();
                    }
                    try {
                        return method.invoke(real, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
    }

    /** Runs the completion REPL over {@code input} with the cache's generator, as the demo does. */
    private static void runCompletionRepl(GGUFModel model, PromptCache cache, LLMDemoOptions opts, String input)
            throws IOException {
        PrintStream out = new PrintStream(new ByteArrayOutputStream(), true);
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        CompletionRepl.run(model, cache.wrap(TurnGenerator.loop(model, config())), opts,
                new BufferedReader(new StringReader(input)), out, null, interrupts);
    }

    @Test
    public void thePlainCacheSavesAfterTheFirstTurnAlone() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("turns.session");
        LLMDemoOptions opts = parse("--prompt-cache", file.toString());
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache cache = PromptCache.open(m, opts);
            assertNotNull("an absent file opens a new session", cache);
            TurnGenerator generator = cache.wrap(TurnGenerator.loop(m, config()));
            generator.generate(m.tokenizer().encode("alpha"), request(), DISCARD, null);
            long afterFirst = Files.size(file);
            assertTrue("the first turn's state saved", afterFirst > 0);
            generator.generate(m.tokenizer().encode("beta"), request(), DISCARD, null);
            assertEquals("a later turn leaves the plain cache alone", afterFirst, Files.size(file));
        }
    }

    @Test
    public void theAllCacheSavesAfterEveryTurn() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("all.session");
        LLMDemoOptions opts = parse("--prompt-cache", file.toString(), "--prompt-cache-all");
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache cache = PromptCache.open(m, opts);
            TurnGenerator generator = cache.wrap(TurnGenerator.loop(m, config()));
            generator.generate(m.tokenizer().encode("alpha"), request(), DISCARD, null);
            long afterFirst = Files.size(file);
            assertTrue(afterFirst > 0);
            generator.generate(m.tokenizer().encode("a much longer second turn"), request(), DISCARD, null);
            assertTrue("the second turn's tokens saved too", Files.size(file) != afterFirst);
        }
    }

    @Test
    public void theReadOnlyCacheNeverWrites() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("ro.session");
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            m.generate(m.tokenizer().encode("alpha"), request(), null);
            m.saveSession(file);
            long saved = Files.size(file);
            LLMDemoOptions opts = parse("--prompt-cache", file.toString(), "--prompt-cache-ro");
            PromptCache cache = PromptCache.open(m, opts);
            TurnGenerator loop = TurnGenerator.loop(m, config());
            assertSame("no write runs, so no wrapper sits between", loop, cache.wrap(loop));
            cache.wrap(loop).generate(m.tokenizer().encode("beta"), request(), DISCARD, null);
            assertEquals("the file stays untouched", saved, Files.size(file));
        }
    }

    @Test
    public void anExistingSessionLoadsAtOpen() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("load.session");
        int[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            expected = m.generate(m.tokenizer().encode("alpha"), request(), null).tokens();
            m.saveSession(file);
        }
        LLMDemoOptions opts = parse("--prompt-cache", file.toString());
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache.open(m, opts);
            GenerationResult turn = TurnGenerator.loop(m, config())
                    .generate(m.tokenizer().encode("alpha"), request(), DISCARD, null);
            assertTrue("the loaded session serves the prompt", turn.timings().promptTokens() <= 2);
            assertArrayEquals("the loaded session decodes the original run", expected, turn.tokens());
        }
    }

    @Test
    public void aSavedTurnReloadsIntoAFreshModel() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("roundtrip.session");
        int[] expected;
        LLMDemoOptions opts = parse("--prompt-cache", file.toString());
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            TurnGenerator generator = PromptCache.open(m, opts).wrap(TurnGenerator.loop(m, config()));
            expected = generator.generate(m.tokenizer().encode("alpha"), request(), DISCARD, null).tokens();
        }
        assertTrue("the turn wrote its session", Files.size(file) > 0);
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache.open(m, opts);
            GenerationResult turn = TurnGenerator.loop(m, config())
                    .generate(m.tokenizer().encode("alpha"), request(), DISCARD, null);
            assertTrue("the saved turn's prefix is reused", turn.timings().promptTokens() <= 2);
            assertArrayEquals("the reloaded session continues identically", expected, turn.tokens());
        }
    }

    @Test
    public void aDirectoryInPlaceOfTheSessionFileIsRefused() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = temp.newFolder("cache-dir").toPath();
        LLMDemoOptions opts = parse("--prompt-cache", dir.toString());
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            IOException refused = assertThrows(IOException.class, () -> PromptCache.open(m, opts));
            assertTrue(refused.getMessage(), refused.getMessage().contains("--prompt-cache"));
        }
    }

    @Test
    public void theAllCacheSavesEveryCompletionReplTurn() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("repl-all.session");
        LLMDemoOptions opts = parse("--prompt-cache", file.toString(), "--prompt-cache-all",
                "-i", "-p", "alpha", "-n", "2");
        AtomicInteger saves = new AtomicInteger();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache cache = PromptCache.open(countingSaves(m, saves), opts);
            runCompletionRepl(m, cache, opts, "beta\ngamma\n");
            assertEquals("one save per completion REPL turn", 3, saves.get());
            assertTrue("the last turn's state stays in the file", Files.size(file) > 0);
        }
    }

    @Test
    public void thePlainCacheSavesTheFirstCompletionReplTurnAlone() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = temp.getRoot().toPath().resolve("repl-first.session");
        LLMDemoOptions opts = parse("--prompt-cache", file.toString(), "-i", "-p", "alpha", "-n", "2");
        AtomicInteger saves = new AtomicInteger();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            PromptCache cache = PromptCache.open(countingSaves(m, saves), opts);
            runCompletionRepl(m, cache, opts, "beta\ngamma\n");
            assertEquals("later turns leave the plain cache alone", 1, saves.get());
        }
    }
}
