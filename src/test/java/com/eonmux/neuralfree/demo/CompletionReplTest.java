package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CompletionRepl} over a real synthetic llama model, checked token by token against llama.cpp
 * tools/completion/completion.cpp: the prompt carries the BOS, each input enters the context as the prefix, the
 * input and the suffix (no BOS of their own), {@code --in-prefix-bos} adds one, {@code -if} waits before any
 * generation, escapes are processed unless {@code --no-escape}, the context shifts its older half out, and the
 * non-interactive form completes once.
 */
public class CompletionReplTest {

    private static final int N_CTX = 64;

    private static GGUFModel model;
    private static EngineConfig config;
    private static GGUFTokenizer tokenizer;

    @BeforeClass
    public static void open() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        config = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        model = new GGUFModel(file.toString(), config);
        tokenizer = model.tokenizer();
    }

    @AfterClass
    public static void close() throws IOException {
        model.close();
    }

    /** The engine loop, recording every prompt and result. */
    private static final class Recording implements TurnGenerator {
        final TurnGenerator real = TurnGenerator.loop(model, config);
        final List<int[]> prompts = new ArrayList<>();
        final List<GenerationResult> results = new ArrayList<>();

        @Override
        public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                         String rendered) throws IOException {
            prompts.add(prompt.clone());
            GenerationResult r = real.generate(prompt, request, sink, rendered);
            results.add(r);
            return r;
        }

        @Override
        public GenerationRequest effective(GenerationRequest request) {
            return real.effective(request);
        }

        @Override
        public void reset() {
            real.reset();
        }
    }

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();
    private final Recording generator = new Recording();

    private void run(String input, String... args) throws Exception {
        run(new StringReader(input), null, args);
    }

    private void run(Reader input, Transcript transcript, String... args) throws Exception {
        PrintStream out = new PrintStream(console, true);
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        CompletionRepl.run(model, generator, LLMDemoOptions.parse(args), new BufferedReader(input), out,
                transcript, interrupts);
    }

    private String output() {
        return new String(console.toByteArray(), StandardCharsets.UTF_8);
    }

    private static int[] concat(int[]... parts) {
        int n = 0;
        for (int[] p : parts) {
            n += p.length;
        }
        int[] out = new int[n];
        int at = 0;
        for (int[] p : parts) {
            System.arraycopy(p, 0, out, at, p.length);
            at += p.length;
        }
        return out;
    }

    @Test
    public void inputsEnterTheContextBetweenPrefixAndSuffix() throws Exception {
        run("hello\n", "-i", "-p", "Once upon", "-n", "3", "--temp", "0", "--in-prefix", " Q:", "--in-suffix",
                " A:");
        assertEquals(2, generator.prompts.size());
        int[] prompt = tokenizer.encode("Once upon", true, true);
        assertEquals("the prompt starts with the BOS", tokenizer.bosId(), prompt[0]);
        assertArrayEquals(prompt, generator.prompts.get(0));
        int[] expected = concat(prompt, generator.results.get(0).tokens(), tokenizer.encode(" Q:", false, true),
                tokenizer.encode("hello", false, false), tokenizer.encode(" A:", false, true));
        assertArrayEquals(expected, generator.prompts.get(1));
        String out = output();
        assertTrue(out.contains("== Running in interactive mode. =="));
        assertTrue(out.contains("Once upon"));
        assertTrue("the prefix and suffix are shown around the input", out.contains(" Q: A:"));
        assertTrue(out.endsWith("EOF by user\n"));
    }

    @Test
    public void interactiveFirstWaitsForInputAndPrefixBosAddsOne() throws Exception {
        run("hi\n", "-if", "-p", "Story", "-n", "2", "--temp", "0", "--in-prefix-bos");
        assertEquals("no generation before the first input, then one per input", 1, generator.prompts.size());
        int[] expected = concat(tokenizer.encode("Story", true, true), new int[]{tokenizer.bosId()},
                tokenizer.encode("hi", false, false));
        assertArrayEquals(expected, generator.prompts.get(0));
    }

    @Test
    public void anEmptyInputPassesControlBackWithoutTokens() throws Exception {
        run("\n", "-i", "-p", "Go", "-n", "2", "--temp", "0");
        assertEquals(2, generator.prompts.size());
        assertArrayEquals(concat(generator.prompts.get(0), generator.results.get(0).tokens()),
                generator.prompts.get(1));
    }

    @Test
    public void escapesAreProcessedUnlessTurnedOff() throws Exception {
        run("", "-i", "-p", "a\\nb\\x41", "-n", "1", "--temp", "0");
        assertArrayEquals(tokenizer.encode("a\nbA", true, true), generator.prompts.get(0));
        generator.prompts.clear();
        run("", "-i", "-p", "a\\nb", "-n", "1", "--temp", "0", "--no-escape");
        assertArrayEquals(tokenizer.encode("a\\nb", true, true), generator.prompts.get(0));
        assertEquals("tab\t'q\" \\ \\z \\x4", CompletionRepl.processEscapes("tab\\t\\'q\\\" \\\\ \\z \\x4"));
    }

    @Test
    public void withoutInteractiveTheModeCompletesOnce() throws Exception {
        Reader unreadable = new Reader() {
            @Override
            public int read(char[] buf, int off, int len) {
                throw new AssertionError("the non-interactive form reads no input");
            }

            @Override
            public void close() {
            }
        };
        StringWriter log = new StringWriter();
        run(unreadable, new Transcript(log), "--in-suffix", "x", "-p", "Once", "-n", "2", "--temp", "0");
        assertEquals(1, generator.prompts.size());
        assertTrue(log.toString().startsWith("Assistant:\n"));
    }

    @Test
    public void theContextShiftsItsOlderHalfOutKeepingTheBos() throws Exception {
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < 80; i++) {
            words.append("hello ");
        }
        run(words + "\n", "-i", "-p", "Once upon a time", "-n", "2", "--temp", "0");
        assertEquals(2, generator.prompts.size());
        int[] second = generator.prompts.get(1);
        int[] unshifted = concat(generator.prompts.get(0), generator.results.get(0).tokens(),
                tokenizer.encode(words.toString(), false, false));
        assertTrue("the unshifted context would overflow: " + unshifted.length,
                unshifted.length > N_CTX - CompletionRepl.CONTEXT_MARGIN);
        assertTrue(second.length <= N_CTX - CompletionRepl.CONTEXT_MARGIN);
        assertEquals("the leading BOS is kept", tokenizer.bosId(), second[0]);
        int[] tail = Arrays.copyOfRange(unshifted, unshifted.length - (second.length - 1), unshifted.length);
        assertArrayEquals("the newest tokens stay", tail, Arrays.copyOfRange(second, 1, second.length));
        assertTrue(output().contains("[context shift: "));
    }

    @Test
    public void aPromptOverTheContextIsRefused() {
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < N_CTX; i++) {
            words.append("hello ");
        }
        IllegalArgumentException e = org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> run("", "-i", "-p", words.toString(), "-n", "1"));
        assertTrue(e.getMessage(), e.getMessage().contains("the prompt is too long"));
    }
}
