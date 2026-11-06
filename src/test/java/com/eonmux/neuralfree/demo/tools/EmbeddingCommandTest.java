package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EmbeddingNorm;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EmbeddingCommand} against llama.cpp's embedding example: the line split of {@code split_lines}, the flag
 * parsing (llama.cpp's option names, prompt files and escapes, every invalid flag refused by name), the reranker
 * prompt of a classification pair, the last-token warning, the {@code --verbose-prompt} dump, and the embed loop
 * over synthetic bert models: Euclidean-normalized pooled rows, per-token rows for pooling NONE, and labelled
 * scores for RANK pooling.
 */
public class EmbeddingCommandTest {

    private static final EngineConfig CPU = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The synthetic bert with {@code pooling_type} replaced. */
    private static GgufTestWriter bertPooling(int pooling) {
        GgufTestWriter w = SyntheticModels.bert();
        String key = w.archKey("pooling_type");
        return w.remove(key).u32(key, pooling);
    }

    /** A RANK bert: a two-class {@code cls.output} head labelled "yes" and "" (the empty label prints its index). */
    private static GgufTestWriter rankBert() {
        GgufTestWriter w = bertPooling(4);
        int nEmbd = SyntheticModels.Config.bert().nEmbd();
        w.tensorRandom("cls.output.weight", GGMLType.F32, 7L, -0.5f, 0.5f, nEmbd, 2);
        return w.strArray(w.archKey("classifier.output_labels"), "yes", "");
    }

    private static EngineModel open(GgufTestWriter w) throws IOException {
        return LLMModelFactory.open(w.writeTempFile().toString(), CPU);
    }

    // ------------------------------------------------------------------ the line split

    @Test
    public void splitLinesKeepsEmptyLinesAndTheTailLikeLlamaCpp() {
        assertEquals(Arrays.asList("a", "b", "c"), EmbeddingCommand.splitLines("a\nb\nc", "\n"));
        assertEquals(Arrays.asList("a", "b", ""), EmbeddingCommand.splitLines("a\nb\n", "\n"));
        assertEquals(Arrays.asList("", ""), EmbeddingCommand.splitLines("\n", "\n"));
        assertEquals(Collections.singletonList("one line"), EmbeddingCommand.splitLines("one line", "\n"));
        assertEquals(Arrays.asList("a", "b"), EmbeddingCommand.splitLines("a<#sep#>b", "<#sep#>"));
        assertThrows(IllegalArgumentException.class, () -> EmbeddingCommand.splitLines("a", ""));
    }

    // ------------------------------------------------------------------ the flags

    @Test
    public void parseTakesLlamaCppDefaultsAndThePromptAlias() throws Exception {
        EmbeddingCommand.Options o = EmbeddingCommand.parse(new String[]{"m.gguf", "--prompt", "hi"});
        assertEquals("m.gguf", o.model());
        assertEquals("hi", o.prompt());
        assertEquals(2, o.normalize());
        assertEquals("", o.format());
        assertEquals("\n", o.separator());
        assertEquals("\t", o.clsSeparator());
        assertEquals(false, o.verbosePrompt());
        assertEquals("hey", EmbeddingCommand.parse(new String[]{"m.gguf", "-p", "hey"}).prompt());
        EmbeddingCommand.Options custom = EmbeddingCommand.parse(new String[]{"m.gguf", "--prompt", "a",
                "--embd-normalize", "1", "--embd-output-format", "json+", "--embd-separator", "<#s#>",
                "--cls-separator", "<#q#>", "--verbose-prompt"});
        assertEquals(1, custom.normalize());
        assertEquals("json+", custom.format());
        assertEquals("<#s#>", custom.separator());
        assertEquals("<#q#>", custom.clsSeparator());
        assertEquals(true, custom.verbosePrompt());
        // The value consumes the next argv token even when it starts with '-' (llama.cpp's --embd-normalize -1).
        assertEquals(-1, EmbeddingCommand.parse(
                new String[]{"m.gguf", "--prompt", "a", "--embd-normalize", "-1"}).normalize());
    }

    /** The prompt's escapes process by default and stay with --no-escape; the separators never process. */
    @Test
    public void escapesProcessThePromptUnlessTurnedOff() throws Exception {
        assertEquals("a\nb\t\"c\\", EmbeddingCommand.parse(
                new String[]{"m.gguf", "-p", "a\\nb\\t\\\"c\\\\"}).prompt());
        assertEquals("a\\nb", EmbeddingCommand.parse(new String[]{"m.gguf", "-p", "a\\nb", "--no-escape"}).prompt());
        assertEquals("a\nb", EmbeddingCommand.parse(
                new String[]{"m.gguf", "--no-escape", "-e", "-p", "a\\nb"}).prompt());
        assertEquals("\\t", EmbeddingCommand.parse(
                new String[]{"m.gguf", "-p", "a", "--cls-separator", "\\t"}).clsSeparator());
    }

    /** --file reads the prompt, drops one trailing newline and processes escapes; the last prompt flag wins. */
    @Test
    public void theFileFlagReadsThePrompt() throws Exception {
        Path f = tmp.newFile("prompts.txt").toPath();
        Files.write(f, "first\nsecond\\tpart\n\n".getBytes(StandardCharsets.UTF_8));
        EmbeddingCommand.Options o = EmbeddingCommand.parse(new String[]{"m.gguf", "--file", f.toString()});
        assertEquals("first\nsecond\tpart\n", o.prompt());
        assertEquals("inline", EmbeddingCommand.parse(
                new String[]{"m.gguf", "-f", f.toString(), "-p", "inline"}).prompt());
        assertEquals("first\nsecond\tpart\n", EmbeddingCommand.parse(
                new String[]{"m.gguf", "-p", "inline", "-f", f.toString()}).prompt());
        EmbeddingCommand.UsageException e = assertThrows(EmbeddingCommand.UsageException.class,
                () -> EmbeddingCommand.parse(new String[]{"m.gguf", "-f", tmp.getRoot() + "/missing.txt"}));
        assertTrue(e.getMessage(), e.getMessage().startsWith("-f '"));
    }

    @Test
    public void parseRefusesEveryInvalidFlagByName() {
        assertRefused("embedding needs a model path");
        assertRefused("embedding needs --prompt or --file", "m.gguf");
        assertRefused("embedding needs a model path", "--prompt", "hi");
        assertRefused("--prompt needs a value", "m.gguf", "--prompt");
        assertRefused("too many file arguments", "m.gguf", "--prompt", "a", "b.gguf");
        assertRefused("--embd-normalize -2 is invalid", "m.gguf", "--prompt", "a", "--embd-normalize", "-2");
        assertRefused("--embd-normalize needs an integer", "m.gguf", "--prompt", "a", "--embd-normalize", "x");
        assertRefused("--embd-output-format 'csv' is invalid", "m.gguf", "--prompt", "a",
                "--embd-output-format", "csv");
        assertRefused("--embd-separator needs a non-empty string", "m.gguf", "--prompt", "a",
                "--embd-separator", "");
        assertRefused("--cls-separator needs a non-empty string", "m.gguf", "--prompt", "a",
                "--cls-separator", "");
        assertRefused("unknown option --embd-out", "m.gguf", "--prompt", "a", "--embd-out", "json");
        assertRefused("--pooling: pooling junk is invalid", "m.gguf", "--prompt", "a", "--pooling", "junk");
        assertRefused("--parallel 0 is outside [1, 256]", "m.gguf", "--prompt", "a", "--parallel", "0");
        assertRefused("--parallel 257 is outside [1, 256]", "m.gguf", "--prompt", "a", "--parallel", "257");
        assertRefused("--batch-size 0 is outside [1, ", "m.gguf", "--prompt", "a", "--batch-size", "0");
        assertRefused("-ub needs an integer, got 'x'", "m.gguf", "--prompt", "a", "-ub", "x");
        assertRefused("--pooling needs a value", "m.gguf", "--prompt", "a", "--pooling");
    }

    /** The pooling override and the batching flags parse into the options and the load configuration. */
    @Test
    public void poolingAndBatchingFlagsParse() throws Exception {
        EmbeddingCommand.Options o = EmbeddingCommand.parse(new String[]{"m.gguf", "--prompt", "a",
                "--pooling", "cls", "--parallel", "4", "--batch-size", "512", "--ubatch-size", "128",
                "--ctx-size", "2048", "-ngl", "0"});
        assertEquals(PoolingKind.CLS, o.pooling());
        assertEquals(4, o.parallel());
        assertEquals(512, o.batchSize());
        com.eonmux.neuralfree.gguf.memory.MemoryConfig mem = o.load().memory();
        assertEquals(128, mem.ubatch());
        assertEquals(2048, mem.nCtx());
        assertEquals(4, mem.parallel());
        assertEquals("the CPU engine stays unless a GPU flag asks", EngineConfig.BackendMode.CPU, o.load().backend());
        EmbeddingCommand.Options gpu = EmbeddingCommand.parse(new String[]{"m.gguf", "--prompt", "a",
                "--gpu", "-ngl", "9"});
        assertEquals("-ngl over 0 asks for the GPU", true, gpu.load().backend().usesGpu());
        EmbeddingCommand.Options nonePooling = EmbeddingCommand.parse(
                new String[]{"m.gguf", "--prompt", "a", "--pooling", "ALL"});
        assertEquals("vLLM's ALL names per-token pooling", PoolingKind.NONE, nonePooling.pooling());
        EmbeddingCommand.Options plain = EmbeddingCommand.parse(new String[]{"m.gguf", "--prompt", "a"});
        assertEquals("no override leaves the GGUF pooling", null, plain.pooling());
        assertEquals("--parallel absent", 0, plain.parallel());
        assertEquals(EmbeddingBatchFlags.DEFAULT_BATCH_SIZE, plain.batchSize());
        assertEquals("--backend cpu keeps the CPU engine", EngineConfig.BackendMode.CPU, EmbeddingCommand.parse(
                new String[]{"m.gguf", "--prompt", "a", "--backend", "cpu"}).load().backend());
    }

    private static void assertRefused(String message, String... args) {
        EmbeddingCommand.UsageException e = assertThrows(EmbeddingCommand.UsageException.class,
                () -> EmbeddingCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().startsWith(message));
    }

    // ------------------------------------------------------------------ the reranker prompt and the warnings

    /** Without a rerank prompt the pair joins with the SEP text a WPM vocabulary adds (it adds no EOS). */
    @Test
    public void aClassificationPairJoinsWithTheAddedSpecialTokens() throws IOException {
        try (EngineModel bert = open(rankBert())) {
            GGUFTokenizer tok = bert.tokenizer();
            assertEquals(true, tok.vocab().addSep());
            assertEquals(false, tok.vocab().addEos());
            assertEquals("query[SEP]doc", EmbeddingCommand.rankPrompt(tok, "query\tdoc", "\t"));
            assertEquals("a[SEP]b[SEP]c", EmbeddingCommand.rankPrompt(tok, "a<#q#>b<#q#>c", "<#q#>"));
        }
    }

    /** With a rerank prompt the first two pieces fill its query and document placeholders. */
    @Test
    public void aClassificationPairFillsTheRerankPrompt() throws IOException {
        GgufTestWriter w = rankBert().str(GGUFTokenizer.KEY_RERANK_TEMPLATE, "Q: {query} D: {document} ({query})");
        try (EngineModel bert = open(w)) {
            assertEquals("Q: what D: this (what)",
                    EmbeddingCommand.rankPrompt(bert.tokenizer(), "what\tthis\tignored", "\t"));
        }
    }

    @Test
    public void theLastTokenWarningNamesTheMissingEos() throws IOException {
        try (EngineModel bert = open(SyntheticModels.bert())) {
            GGUFTokenizer tok = bert.tokenizer();
            int sep = tok.specialTokens().sep();
            assertEquals("", EmbeddingCommand.lastTokenWarnings(tok, Collections.singletonList(new int[]{2, 9, sep})));
            String warning = EmbeddingCommand.lastTokenWarnings(tok, Arrays.asList(new int[]{2, 9}, new int[0]));
            assertEquals(2, warning.split("main: last token in the prompt is not SEP or EOS\n", -1).length - 1);
            assertTrue(warning, warning.contains("'tokenizer.ggml.add_eos_token' should be set to 'true'"));
        }
    }

    // ------------------------------------------------------------------ the embed loop

    /** A constant batch size, so the loop tests below exercise one prompt per window like the old runs did. */
    private static final int BATCH = 2048;

    @Test
    public void embeddingRunsPromptsOnTheSyntheticBertAndNormalizes() throws IOException {
        try (EngineModel bert = open(SyntheticModels.bert())) {
            List<String> prompts = Arrays.asList("hello world", "goodbye world");
            List<int[]> inputs = EmbeddingCommand.tokenizePrompts(bert, prompts, "\t", BATCH);
            // Special tokens are added: the WPM vocabulary wraps every prompt in [CLS] ... [SEP].
            for (int[] tokens : inputs) {
                assertEquals(bert.tokenizer().specialTokens().cls(), tokens[0]);
                assertEquals(bert.tokenizer().specialTokens().sep(), tokens[tokens.length - 1]);
            }
            assertEquals("", EmbeddingCommand.lastTokenWarnings(bert.tokenizer(), inputs));
            List<float[]> rows = EmbeddingCommand.embedRows(bert, inputs, 2, BATCH);
            assertEquals(2, rows.size());
            for (float[] row : rows) {
                assertEquals(bert.embeddingLength(), row.length);
                assertEquals(1.0, norm(row), 1e-4);
            }
            // The rows are the model's pooled embeddings scaled, not something else of unit length.
            float[] raw = bert.encode(inputs.get(0));
            assertArrayEquals(EmbeddingNorm.normalize(raw, EmbeddingNorm.EUCLIDEAN), rows.get(0), 0f);
            // -1 leaves the pooled embedding as the model returns it.
            assertArrayEquals(raw, EmbeddingCommand.embedRows(bert, inputs.subList(0, 1), -1, BATCH).get(0), 0f);
        }
    }

    @Test
    public void nonePoolingEmbedsOneRowPerTokenAcrossPrompts() throws IOException {
        try (EngineModel bert = open(bertPooling(0))) {
            assertEquals(LLMModel.PoolingType.NONE, bert.poolingType());
            List<int[]> inputs = EmbeddingCommand.tokenizePrompts(bert, Arrays.asList("hello", "hello world"),
                    "\t", BATCH);
            List<float[]> rows = EmbeddingCommand.embedRows(bert, inputs, 2, BATCH);
            assertEquals(inputs.get(0).length + inputs.get(1).length, rows.size());
            for (float[] row : rows) {
                assertEquals(1.0, norm(row), 1e-4);
            }
        }
    }

    /** RANK pooling scores every prompt over the joined pair, and the scores print with the model's labels. */
    @Test
    public void rankPoolingScoresClassificationPairs() throws IOException {
        try (EngineModel bert = open(rankBert())) {
            assertEquals(LLMModel.PoolingType.RANK, bert.poolingType());
            List<String> prompts = Arrays.asList("query\tdoc", "query\tother doc");
            List<int[]> inputs = EmbeddingCommand.tokenizePrompts(bert, prompts, "\t", BATCH);
            int sep = bert.tokenizer().specialTokens().sep();
            assertEquals("the pair holds an inner SEP and the trailing SEP", 2,
                    Arrays.stream(inputs.get(0)).filter(t -> t == sep).count());
            assertArrayEquals(bert.tokenizer().encode("query[SEP]doc", true, true), inputs.get(0));
            List<float[]> rows = EmbeddingCommand.embedRows(bert, inputs, -1, BATCH);
            assertEquals(2, rows.get(0).length);
            assertArrayEquals(bert.classify(inputs.get(0)), rows.get(0), 1e-6f);
            assertNotEquals(rows.get(0)[0], rows.get(1)[0], 0f);
            StringBuilder out = new StringBuilder();
            EmbeddingOutput.render(out, rows, prompts, bert.poolingType(), bert.classLabels(), "", -1);
            String printed = out.toString();
            assertTrue(printed, printed.contains("rerank score 0: "));
            assertTrue(printed, printed.matches("(?s).*rerank score 1: +-?\\d+\\.\\d{3} \\[yes\\]\n.*"));
            assertTrue(printed, printed.matches("(?s).*rerank score 1: +-?\\d+\\.\\d{3} \\[1\\]\n.*"));
        }
    }

    @Test
    public void verbosePromptDumpsEveryTokenWithItsPiece() throws IOException {
        try (EngineModel bert = open(SyntheticModels.bert())) {
            GGUFTokenizer tok = bert.tokenizer();
            int[] tokens = tok.encode("hello", true, true);
            StringBuilder out = new StringBuilder();
            new PromptDump(tok).append(out, 3, "hello", tokens);
            String[] lines = out.toString().split("\n", -1);
            assertEquals("main: prompt 3: 'hello'", lines[0]);
            assertEquals("main: number of tokens in prompt = " + tokens.length, lines[1]);
            // token_to_piece(special = true): the fixture's BOS (id 101, a normal letter token) prints its text with
            // the phantom space unescaped, where the BOS role hides it from decoded output; the control [SEP] prints
            // its text.
            assertEquals(101, tokens[0]);
            assertEquals("", tok.decode(tokens[0]));
            assertEquals(String.format("%6d -> '%s'", tokens[0], tok.getToken(tokens[0]).replace('\u2581', ' ')),
                    lines[2]);
            assertEquals(String.format("%6d -> '[SEP]'", tokens[tokens.length - 1]), lines[1 + tokens.length]);
            assertTrue(out.toString().endsWith("'\n\n\n"));
            assertThrows(IllegalArgumentException.class, () -> new PromptDump(tok).piece(tok.vocabSize()));
        }
    }

    /** A generative model without pooling refuses by name, as llama.cpp refuses its encoder-decoders. */
    @Test
    public void aGenerativeModelRefusesToEmbed() throws IOException {
        try (EngineModel llama = open(SyntheticModels.llama())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> EmbeddingCommand.tokenizePrompts(llama, Collections.singletonList("hello"), "\t", BATCH));
            assertTrue(e.getMessage(), e.getMessage().contains("encode needs an embedding model or a pooling type"));
        }
    }

    @Test
    public void aPromptLongerThanTheContextRefuses() throws IOException {
        try (EngineModel bert = open(SyntheticModels.bert())) {
            StringBuilder longPrompt = new StringBuilder();
            for (int i = 0; i <= bert.contextLength(); i++) {
                longPrompt.append("hello ");
            }
            IOException e = assertThrows(IOException.class, () -> EmbeddingCommand.tokenizePrompts(bert,
                    Arrays.asList("short", longPrompt.toString()), "\t", BATCH));
            assertTrue(e.getMessage(), e.getMessage().startsWith("prompt 1 tokenizes to "));
        }
    }

    /** A pooled prompt longer than {@code --batch-size} refuses, as llama.cpp's exceeds-batch error does. */
    @Test
    public void aPromptLongerThanTheBatchRefuses() throws IOException {
        try (EngineModel bert = open(SyntheticModels.bert())) {
            IOException e = assertThrows(IOException.class, () -> EmbeddingCommand.tokenizePrompts(bert,
                    Collections.singletonList("hello world"), "\t", 1));
            assertTrue(e.getMessage(), e.getMessage().contains("exceeds batch size (1)"));
            List<int[]> inputs = EmbeddingCommand.tokenizePrompts(bert,
                    Collections.singletonList("hello world"), "\t", bert.contextLength());
            assertEquals("the raise to the context keeps the prompt", 1, inputs.size());
        }
    }

    /**
     * The whole command over a model file: parse, open, tokenize, embed and print json+ to stdout; a pooling
     * override on the command line changes the rows the same run prints without one.
     */
    @Test
    public void theCommandPrintsJsonPlusEndToEnd() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        String meanRows = stdoutOf(EmbeddingCommand.parse(new String[]{model.toString(),
                "-p", "hello world<#>goodbye", "--embd-separator", "<#>", "--embd-output-format", "json+"}));
        assertTrue(meanRows, meanRows.startsWith("{\n  \"object\": \"list\",\n  \"data\": [\n"));
        assertTrue(meanRows, meanRows.contains("\"index\": 1,"));
        assertTrue(meanRows, !meanRows.contains("\"index\": 2,"));
        assertTrue(meanRows, meanRows.contains("  \"cosineSimilarity\": [\n    [  1.00, "));
        assertTrue(meanRows, meanRows.endsWith(" ]\n  ]\n}\n\n"));
        String firstRow = meanRows.substring(meanRows.indexOf("\"embedding\": [") + 14,
                meanRows.indexOf("]\n    }"));
        assertEquals(SyntheticModels.Config.bert().nEmbd(), firstRow.split(",").length);

        String clsRows = stdoutOf(EmbeddingCommand.parse(new String[]{model.toString(),
                "-p", "hello world<#>goodbye", "--embd-separator", "<#>", "--embd-output-format", "json+",
                "--pooling", "cls"}));
        assertNotEquals("CLS pooling answers other rows than the GGUF's MEAN", meanRows, clsRows);
    }

    /** {@code --parallel 2} runs both prompts as one multi-sequence batch and answers the sequential rows. */
    @Test
    public void parallelBatchingAnswersTheSequentialRowsEndToEnd() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        String[] flags = {"-p", "hello world<#>goodbye world<#>third prompt", "--embd-separator", "<#>",
                "--embd-output-format", "json"};
        String sequential = stdoutOf(EmbeddingCommand.parse(
                concat(new String[]{model.toString()}, flags)));
        String batched = stdoutOf(EmbeddingCommand.parse(
                concat(new String[]{model.toString(), "--parallel", "2", "-b", "64", "-ub", "32"}, flags)));
        assertEquals("the batched windows embed the same vectors", sequential, batched);
    }

    /** {@code --pooling rank} on a model without a classifier head refuses by name before any weight moves. */
    @Test
    public void rankOverrideOnAClassifierLessModelRefusesEndToEnd() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        EmbeddingCommand.Options o = EmbeddingCommand.parse(new String[]{model.toString(),
                "-p", "hello", "--pooling", "rank"});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EmbeddingCommand.execute(o));
        assertTrue(e.getMessage(), e.getMessage().contains("rank"));
        assertTrue(e.getMessage(), e.getMessage().contains("classifier"));
    }

    /**
     * Without {@code --parallel} every prompt takes a memory sequence (llama.cpp's embedding tool takes
     * {@code n_seq_max} sequences for {@code -np 1}), capped at 256; {@code --parallel} wins; a per-token model runs
     * one.
     */
    @Test
    public void theSequencesFollowLlamaCppsDefault() throws IOException {
        com.eonmux.neuralfree.gguf.spec.ModelSpec mean = LLMModelFactory.inspect(
                SyntheticModels.bert().writeTempFile().toString(), CPU);
        com.eonmux.neuralfree.gguf.spec.ModelSpec none = LLMModelFactory.inspect(
                bertPooling(0).writeTempFile().toString(), CPU);
        assertEquals(3, EmbeddingCommand.sequences(0, 3, mean));
        assertEquals(256, EmbeddingCommand.sequences(0, 1000, mean));
        assertEquals(2, EmbeddingCommand.sequences(2, 3, mean));
        assertEquals(1, EmbeddingCommand.sequences(0, 3, none));
    }

    /**
     * A causal model with a pooling override embeds its prompts through the paged multi-sequence path by default
     * (one batch over every prompt) and through {@code --parallel 2} (two batches), answering the rows the
     * contiguous one-prompt-per-forward run ({@code --parallel 1}) answers.
     */
    @Test
    public void aPooledCausalModelBatchesThroughThePagedMemoryEndToEnd() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        String[] flags = {"-p", "hello world<#>goodbye world<#>a third prompt", "--embd-separator", "<#>",
                "--embd-output-format", "raw", "--pooling", "last", "-c", "64", "--kv-type", "f32"};
        float[][] sequential = rawRows(stdoutOf(EmbeddingCommand.parse(
                concat(new String[]{model.toString(), "--parallel", "1"}, flags))));
        float[][] batched = rawRows(stdoutOf(EmbeddingCommand.parse(concat(new String[]{model.toString()}, flags))));
        float[][] pairs = rawRows(stdoutOf(EmbeddingCommand.parse(
                concat(new String[]{model.toString(), "--parallel", "2"}, flags))));
        assertEquals(3, sequential.length);
        for (int i = 0; i < sequential.length; i++) {
            assertArrayEquals("prompt " + i + " in one batch", sequential[i], batched[i], 1e-4f);
            assertArrayEquals("prompt " + i + " in batches of two", sequential[i], pairs[i], 1e-4f);
        }
    }

    /** llama.cpp's batch windows: a window closes at the batch size or at the sequence count. */
    @Test
    public void theWindowsCloseAtTheBatchSizeAndTheSequenceCount() {
        List<int[]> inputs = Arrays.asList(new int[3], new int[3], new int[3], new int[5]);
        assertEquals("[[0, 2], [2, 3], [3, 4]]", windowsOf(EmbeddingBatchFlags.windows(inputs, 6, 8)));
        assertEquals("[[0, 1], [1, 2], [2, 3], [3, 4]]", windowsOf(EmbeddingBatchFlags.windows(inputs, 100, 1)));
        assertEquals("[[0, 3], [3, 4]]", windowsOf(EmbeddingBatchFlags.windows(inputs, 100, 3)));
        assertEquals("[[0, 4]]", windowsOf(EmbeddingBatchFlags.windows(inputs, 14, 4)));
        assertEquals("[]", windowsOf(EmbeddingBatchFlags.windows(Collections.emptyList(), 4, 4)));
    }

    private static String windowsOf(List<int[]> windows) {
        StringBuilder sb = new StringBuilder("[");
        for (int[] w : windows) {
            sb.append(sb.length() > 1 ? ", " : "").append(Arrays.toString(w));
        }
        return sb.append(']').toString();
    }

    /** The rows of the {@code raw} format: one whitespace-delimited embedding per line. */
    private static float[][] rawRows(String raw) {
        String[] lines = raw.trim().split("\n");
        float[][] rows = new float[lines.length][];
        for (int i = 0; i < lines.length; i++) {
            String[] values = lines[i].trim().split("\\s+");
            rows[i] = new float[values.length];
            for (int j = 0; j < values.length; j++) {
                rows[i][j] = Float.parseFloat(values[j]);
            }
        }
        return rows;
    }

    /** Runs {@code execute} with stdout captured and answers what it printed. */
    private static String stdoutOf(EmbeddingCommand.Options o) throws Exception {
        PrintStream saved = System.out;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(bytes, true, StandardCharsets.UTF_8));
            EmbeddingCommand.execute(o);
        } finally {
            System.setOut(saved);
        }
        return bytes.toString(StandardCharsets.UTF_8.name());
    }

    private static String[] concat(String[] a, String[] b) {
        String[] out = new String[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static double norm(float[] row) {
        double sum = 0;
        for (float v : row) {
            sum += (double) v * v;
        }
        return Math.sqrt(sum);
    }
}
