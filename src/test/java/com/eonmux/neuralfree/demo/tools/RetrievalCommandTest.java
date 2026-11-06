package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link RetrievalCommand} against llama.cpp's retrieval example: the {@code chunk_file} chunking with its byte
 * counted sizes and file positions and its leftover rules, the flag parsing ({@code parse_csv_row} context files,
 * every invalid flag refused by name), the top-k ordering, and the embed-plus-query loop over the synthetic bert,
 * whose chunks come out Euclidean-normalized and whose queries print llama.cpp's top-k block.
 */
public class RetrievalCommandTest {

    private static final EngineConfig CPU = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path write(String name, String text) throws IOException {
        Path f = tmp.getRoot().toPath().resolve(name);
        Files.write(f, text.getBytes(StandardCharsets.UTF_8));
        return f;
    }

    // ------------------------------------------------------------------ the chunking

    @Test
    public void chunkFilePushesChunksPastTheSizeAndAccountsFilepos() throws IOException {
        Path f = write("a.txt", "aaaa.bbbb.");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 4, ".");
        assertEquals(2, chunks.size());
        assertEquals("aaaa.", chunks.get(0).textdata);
        assertEquals(0, chunks.get(0).filepos);
        assertEquals("bbbb.", chunks.get(1).textdata);
        assertEquals(5, chunks.get(1).filepos);
        assertEquals(f.toString(), chunks.get(1).filename);
    }

    @Test
    public void theUnpushedTailJoinsTheLastChunkLikeLlamaCpp() throws IOException {
        // At size 5 no single separator piece passes the limit, so one chunk accumulates "aaaa." + "bbbb."
        // (10 > 5) and pushes, and the unpushed "cccc." tail joins its text.
        Path f = write("b.txt", "aaaa.bbbb.cccc.");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 5, ".");
        assertEquals(1, chunks.size());
        assertEquals("aaaa.bbbb.cccc.", chunks.get(0).textdata);
        assertEquals(0, chunks.get(0).filepos);
    }

    @Test
    public void theTailAfterTheLastSeparatorDropsLikeLlamaCpp() throws IOException {
        // "aa." is the only separator-terminated piece; "bb" has no separator, so llama.cpp drops it.
        Path f = write("c.txt", "aa.bb");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 64, ".");
        assertEquals(1, chunks.size());
        assertEquals("aa.", chunks.get(0).textdata);
        // No separator at all leaves no chunks.
        Path none = write("d.txt", "no separator here");
        assertEquals(Collections.emptyList(), RetrievalCommand.chunkFile(none, 64, "."));
    }

    @Test
    public void aMultiLineFileChunksAtEveryNewlinePastTheSize() throws IOException {
        Path f = write("e.txt", "first line\nsecond line\nthird line\n");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 10, "\n");
        // "first line\n" (11), "second line\n" (12) and "third line\n" (11) each pass the size, so each pushes.
        assertEquals(3, chunks.size());
        assertEquals("first line\n", chunks.get(0).textdata);
        assertEquals(0, chunks.get(0).filepos);
        assertEquals("second line\n", chunks.get(1).textdata);
        assertEquals(11, chunks.get(1).filepos);
        assertEquals("third line\n", chunks.get(2).textdata);
        assertEquals(23, chunks.get(2).filepos);
    }

    @Test
    public void chunkAllWalksTheFilesInOrder() throws IOException {
        Path a = write("f1.txt", "one.");
        Path b = write("f2.txt", "two.");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkAll(Arrays.asList(a, b), 1, ".");
        assertEquals(2, chunks.size());
        assertEquals(a.toString(), chunks.get(0).filename);
        assertEquals(b.toString(), chunks.get(1).filename);
    }

    /** Sizes and file positions count UTF-8 bytes, as the C {@code std::string} does. */
    @Test
    public void chunkSizesAndPositionsCountUtf8Bytes() throws IOException {
        // "\u00e9\u00e9." is 5 bytes (3 characters): it passes a size of 4 only when bytes count.
        Path f = write("u.txt", "\u00e9\u00e9.ab.");
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 4, ".");
        assertEquals(1, chunks.size());
        assertEquals("the unpushed 'ab.' joins the first chunk", "\u00e9\u00e9.ab.", chunks.get(0).textdata);
        Path g = write("v.txt", "\u00e9\u00e9.abcd.xy.");
        List<RetrievalCommand.Chunk> two = RetrievalCommand.chunkFile(g, 4, ".");
        assertEquals(2, two.size());
        assertEquals("abcd.xy.", two.get(1).textdata);
        assertEquals("the byte offset after five bytes", 5, two.get(1).filepos);
        // A multi-byte separator matches as bytes.
        Path h = write("w.txt", "one\u2022two\u2022");
        List<RetrievalCommand.Chunk> bullets = RetrievalCommand.chunkFile(h, 1, "\u2022");
        assertEquals(2, bullets.size());
        assertEquals("two\u2022", bullets.get(1).textdata);
        assertEquals(6, bullets.get(1).filepos);
    }

    /** A file larger than llama.cpp's 1024-byte read block chunks whole, its final partial block included. */
    @Test
    public void aLargeFileChunksWhole() throws IOException {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 300; i++) {
            text.append("line ").append(i).append('\n');
        }
        Path f = write("big.txt", text.toString());
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(f, 64, "\n");
        StringBuilder joined = new StringBuilder();
        long expectedPos = 0;
        for (RetrievalCommand.Chunk c : chunks) {
            assertEquals(expectedPos, c.filepos);
            expectedPos += c.textdata.getBytes(StandardCharsets.UTF_8).length;
            joined.append(c.textdata);
        }
        assertEquals(text.toString(), joined.toString());
    }

    @Test
    public void chunkFileRefusesABadSizeOrSeparator() throws IOException {
        Path f = write("r.txt", "a.");
        assertThrows(IllegalArgumentException.class, () -> RetrievalCommand.chunkFile(f, 0, "."));
        assertThrows(IllegalArgumentException.class, () -> RetrievalCommand.chunkFile(f, 4, ""));
    }

    // ------------------------------------------------------------------ the flags

    @Test
    public void parseTakesLlamaCppDefaults() throws Exception {
        Path a = write("pa.txt", "a.");
        Path b = write("pb.txt", "b.");
        RetrievalCommand.Options o = RetrievalCommand.parse(
                new String[]{"m.gguf", "--context-file", a.toString()});
        assertEquals("m.gguf", o.model());
        assertEquals(1, o.contextFiles().size());
        assertEquals(RetrievalCommand.DEFAULT_CHUNK_SIZE, o.chunkSize());
        assertEquals("\n", o.chunkSeparator());
        assertEquals(RetrievalCommand.DEFAULT_TOP_K, o.topK());
        assertEquals(false, o.verbosePrompt());
        RetrievalCommand.Options full = RetrievalCommand.parse(new String[]{"m.gguf", "--context-file",
                a.toString(), "--context-file", b.toString(), "--chunk-size", "32", "--chunk-separator", ".",
                "--top-k", "3", "--verbose-prompt"});
        assertEquals(2, full.contextFiles().size());
        assertEquals(32, full.chunkSize());
        assertEquals(".", full.chunkSeparator());
        assertEquals(3, full.topK());
        assertEquals(true, full.verbosePrompt());
        // One --context-file value lists several files, comma-separated (llama.cpp's parse_csv_row).
        RetrievalCommand.Options csv = RetrievalCommand.parse(
                new String[]{"m.gguf", "--context-file", a + "," + b, "--context-file", a.toString()});
        assertEquals(Arrays.asList(a, b, a), csv.contextFiles());
    }

    @Test
    public void parseCsvRowFollowsLlamaCpp() {
        assertEquals(Arrays.asList("a", "b", ""), RetrievalCommand.parseCsvRow("a,b,"));
        assertEquals(Collections.singletonList(""), RetrievalCommand.parseCsvRow(""));
        assertEquals(Arrays.asList("x,y", "z"), RetrievalCommand.parseCsvRow("\"x,y\",z"));
        assertEquals(Collections.singletonList("say \"hi\""), RetrievalCommand.parseCsvRow("\"say \"\"hi\"\"\""));
        assertEquals(Collections.singletonList("a\"b"), RetrievalCommand.parseCsvRow("a\"b"));
    }

    @Test
    public void parseRefusesEveryInvalidFlagByName() throws Exception {
        String a = write("ok.txt", "a.").toString();
        assertRefused("retrieval needs a model path");
        assertRefused("retrieval needs at least one --context-file", "m.gguf");
        assertRefused("retrieval needs a model path", "--context-file", a);
        assertRefused("--context-file needs a value", "m.gguf", "--context-file");
        assertRefused("--chunk-size 0 must be positive", "m.gguf", "--context-file", a, "--chunk-size", "0");
        assertRefused("--chunk-size needs an integer", "m.gguf", "--context-file", a, "--chunk-size", "x");
        assertRefused("--top-k 0 must be at least 1", "m.gguf", "--context-file", a, "--top-k", "0");
        assertRefused("--chunk-separator needs a non-empty string", "m.gguf", "--context-file", a,
                "--chunk-separator", "");
        assertRefused("unknown option --chunks", "m.gguf", "--context-file", a, "--chunks", "4");
        assertRefused("--context-file 'missing.txt' is not a readable file", "m.gguf", "--context-file",
                "missing.txt");
        assertRefused("--context-file '' is not a readable file", "m.gguf", "--context-file", a + ",");
        assertRefused("--context-file '" + tmp.getRoot() + "' is not a readable file", "m.gguf",
                "--context-file", tmp.getRoot().toString());
        assertRefused("too many file arguments", "m.gguf", "--context-file", a, "b.gguf");
    }

    private static void assertRefused(String message, String... args) {
        RetrievalCommand.UsageException e = assertThrows(RetrievalCommand.UsageException.class,
                () -> RetrievalCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().startsWith(message));
    }

    // ------------------------------------------------------------------ the ranking

    @Test
    public void topIndicesOrderBySimilarityThenSmallerIndex() {
        int[] top = RetrievalCommand.topIndices(new float[]{0.1f, 0.9f, 0.5f}, 2);
        assertTrue(Arrays.toString(top), Arrays.equals(new int[]{1, 2}, top));
        int[] all = RetrievalCommand.topIndices(new float[]{0.5f, 0.9f, 0.5f, 0.9f}, 4);
        assertTrue(Arrays.toString(all), Arrays.equals(new int[]{1, 3, 0, 2}, all));
        int[] over = RetrievalCommand.topIndices(new float[]{0.2f}, 40);
        assertTrue(Arrays.toString(over), Arrays.equals(new int[]{0}, over));
    }

    // ------------------------------------------------------------------ the engine loop

    @Test
    public void chunksEmbedNormalizedAndTheQueryPrintsLlamaCppBlock() throws IOException {
        Path context = write("ctx.txt", "The sky is blue.\nThe grass is green.\n");
        Path model = SyntheticModels.bert().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
        try (EngineModel bert = LLMModelFactory.open(model.toString(), cfg)) {
            List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(context, 10, "\n");
            assertTrue(chunks.size() >= 2);
            StringBuilder dump = new StringBuilder();
            List<RetrievalCommand.Chunk> embedded = RetrievalCommand.embedChunks(bert, chunks, dump, 2048);
            assertEquals(chunks.size(), embedded.size());
            assertThrows("the input chunks stay unembedded", IllegalStateException.class,
                    () -> chunks.get(0).embedding());
            int width = bert.embeddingLength();
            for (RetrievalCommand.Chunk chunk : embedded) {
                float[] e = chunk.embedding();
                assertEquals(width, e.length);
                double sum = 0;
                for (float v : e) {
                    sum += v * v;
                }
                assertEquals(1.0, Math.sqrt(sum), 1e-4);
            }
            // The verbose dump lists every chunk's tokens, ending on the vocabulary's EOS-free SEP.
            assertTrue(dump.toString(), dump.toString().startsWith("main: prompt 0: 'The sky is blue.\n'\n"));
            assertTrue(dump.toString(), dump.toString().contains("main: prompt 1: 'The grass is green.\n'\n"));
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
            BufferedReader in = new BufferedReader(new StringReader("blue sky\n"));
            RetrievalCommand.queryLoop(bert, embedded, 1, in, out);
            String printed = bytes.toString(StandardCharsets.UTF_8.name());
            assertTrue(printed, printed.startsWith("Enter query: "));
            assertTrue(printed, printed.contains("Top 1 similar chunks:"));
            assertTrue(printed, printed.contains("filename: " + context.toString()));
            assertTrue(printed, printed.contains("filepos: 0"));
            assertTrue(printed, printed.contains("similarity: "));
            assertTrue(printed, printed.contains("textdata:"));
            assertTrue(printed, printed.contains("The sky is blue."));
            assertTrue(printed, printed.contains("--------------------"));
            // End of input ends the loop: one query, one top-k block.
            assertEquals(1, printed.split("Top 1 similar chunks:", -1).length - 1);
        }
    }

    /** The query ranking reads the chunk embeddings: the chunk whose text is the query ranks first. */
    @Test
    public void theQueryMatchingAChunkRanksItFirst() throws IOException {
        Path context = write("rank.txt", "alpha beta\ngamma delta\nepsilon zeta\n");
        try (EngineModel bert = LLMModelFactory.open(SyntheticModels.bert().writeTempFile().toString(), CPU)) {
            List<RetrievalCommand.Chunk> embedded = RetrievalCommand.embedChunks(bert,
                    RetrievalCommand.chunkFile(context, 1, "\n"), null, 2048);
            assertEquals(3, embedded.size());
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
            RetrievalCommand.queryLoop(bert, embedded, 3, new BufferedReader(new StringReader("gamma delta\n")), out);
            String printed = bytes.toString(StandardCharsets.UTF_8.name());
            String[] blocks = printed.split("filename: ", -1);
            assertEquals("three ranked chunks", 4, blocks.length);
            assertTrue(printed, blocks[1].contains("textdata:\ngamma delta\n"));
            assertTrue(printed, blocks[1].contains("filepos: 11\n"));
            assertTrue(printed, printed.endsWith("--------------------\nEnter query: "));
        }
    }

    /** The whole command over a model file: comma-listed context files, two queries, then the end of input. */
    @Test
    public void theCommandAnswersQueriesEndToEnd() throws Exception {
        Path first = write("e2e1.txt", "red apples\ngreen pears\n");
        Path second = write("e2e2.txt", "blue sea\n");
        Path model = SyntheticModels.bert().writeTempFile();
        RetrievalCommand.Options o = RetrievalCommand.parse(new String[]{model.toString(), "--context-file",
                first + "," + second, "--chunk-size", "1", "--top-k", "2"});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        RetrievalCommand.execute(o, new BufferedReader(new StringReader("blue sea\ngreen pears\n")), out);
        String printed = bytes.toString(StandardCharsets.UTF_8.name());
        String[] queries = printed.split("Enter query: ", -1);
        assertEquals("two answered queries and the final prompt", 4, queries.length);
        assertTrue(printed, queries[1].startsWith("Top 2 similar chunks:\nfilename: " + second + "\nfilepos: 0\n"
                + "similarity: 1.000000\ntextdata:\nblue sea\n\n--------------------\n"));
        assertTrue(printed, queries[2].startsWith("Top 2 similar chunks:\nfilename: " + first + "\nfilepos: 11\n"));
        assertEquals(2, queries[1].split("--------------------", -1).length - 1);
        assertEquals("", queries[3]);
    }

    @Test
    public void aModelWithoutPooledEmbeddingsRefusesRetrievalByName() throws IOException {
        try (EngineModel llama = LLMModelFactory.open(SyntheticModels.llama().writeTempFile().toString(), CPU)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> RetrievalCommand.requirePooling(llama));
            assertTrue(e.getMessage(), e.getMessage().contains("encode needs an embedding model or a pooling type"));
        }
        GgufTestWriter none = SyntheticModels.bert();
        String key = none.archKey("pooling_type");
        none.remove(key).u32(key, 0);
        try (EngineModel bert = LLMModelFactory.open(none.writeTempFile().toString(), CPU)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> RetrievalCommand.requirePooling(bert));
            assertTrue(e.getMessage(), e.getMessage().contains("pools by NONE"));
        }
    }

    // ------------------------------------------------------------------ batching and the pooling override

    /**
     * {@code --parallel 3} batches the chunks of a causal model with a pooling override through the paged
     * multi-sequence path and answers the embeddings the contiguous one-chunk-per-forward run answers; a chunk
     * longer than {@code --batch-size} refuses, as llama.cpp's exceeds-batch error does.
     */
    @Test
    public void parallelBatchingAnswersTheSequentialEmbeddings() throws Exception {
        Path context = write("batch.txt", "alpha beta\ngamma delta\nepsilon zeta\neta theta\n");
        Path model = SyntheticModels.llama().writeTempFile();
        String[] common = {model.toString(), "--context-file", context.toString(), "--chunk-size", "1",
                "--pooling", "mean", "-c", "64", "--kv-type", "f32"};
        RetrievalCommand.Options sequential = RetrievalCommand.parse(common);
        RetrievalCommand.Options batched = RetrievalCommand.parse(concat(common, "--parallel", "3"));
        assertEquals("the parsed options carry the flags", 3, batched.parallel());
        ModelSpec spec = LLMModelFactory.inspect(model.toString(), batched.load());
        List<RetrievalCommand.Chunk> chunks = RetrievalCommand.chunkFile(context, 1, "\n");
        List<RetrievalCommand.Chunk> expected;
        try (EngineModel m = LLMModelFactory.open(model.toString(), sequential.batch().forModel(spec, 1))) {
            assertEquals("contiguous memory: one chunk per forward", 1, m.forwardBackend().maxSequences());
            expected = RetrievalCommand.embedChunks(m, chunks, null, 64);
        }
        try (EngineModel m = LLMModelFactory.open(model.toString(), batched.batch().forModel(spec, 3))) {
            assertEquals("paged memory: three chunks per forward", 3, m.forwardBackend().maxSequences());
            List<RetrievalCommand.Chunk> got = RetrievalCommand.embedChunks(m, chunks, null, 64);
            assertEquals(chunks.size(), got.size());
            for (int i = 0; i < chunks.size(); i++) {
                org.junit.Assert.assertArrayEquals("chunk " + i + " embeds the same vector",
                        expected.get(i).embedding(), got.get(i).embedding(), 1e-4f);
            }
            IOException tooLong = assertThrows(IOException.class,
                    () -> RetrievalCommand.embedChunks(m, chunks, null, 1));
            assertTrue(tooLong.getMessage(), tooLong.getMessage().contains("exceeds batch size (1)"));
        }
        // The whole command runs batched end to end and finds the query's own chunk first.
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        RetrievalCommand.execute(batched, new BufferedReader(new StringReader("gamma delta\n")), out);
        String printed = bytes.toString(StandardCharsets.UTF_8.name());
        assertTrue(printed, printed.contains("textdata:\ngamma delta\n"));
    }

    private static String[] concat(String[] a, String... b) {
        String[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** {@code --pooling mean} rescues a pooling-NONE model, which refuses without the override. */
    @Test
    public void thePoolingOverrideServesANoneModel() throws Exception {
        GgufTestWriter none = SyntheticModels.bert();
        String key = none.archKey("pooling_type");
        Path model = none.remove(key).u32(key, 0).writeTempFile();
        Path context = write("none.txt", "alpha beta\ngamma delta\n");
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        RetrievalCommand.Options o = RetrievalCommand.parse(new String[]{model.toString(),
                "--context-file", context.toString(), "--chunk-size", "1", "--pooling", "mean",
                "--top-k", "1"});
        RetrievalCommand.execute(o, new BufferedReader(new StringReader("alpha beta\n")), out);
        String printed = bytes.toString(StandardCharsets.UTF_8.name());
        assertTrue(printed, printed.contains("Top 1 similar chunks:"));
        assertTrue(printed, printed.contains("textdata:\nalpha beta\n"));
    }

    /** The pooling and batching flags parse, and an unknown pooling name refuses naming the flag. */
    @Test
    public void poolingAndBatchingFlagsParse() throws Exception {
        Path a = write("pf.txt", "a.");
        RetrievalCommand.Options o = RetrievalCommand.parse(new String[]{"m.gguf", "--context-file", a.toString(),
                "--pooling", "last", "--parallel", "3", "--batch-size", "256", "--ubatch-size", "64",
                "--ctx-size", "512"});
        assertEquals(com.eonmux.neuralfree.gguf.arch.PoolingKind.LAST, o.pooling());
        assertEquals(3, o.parallel());
        assertEquals(256, o.batchSize());
        assertEquals(64, o.load().memory().ubatch());
        assertEquals(512, o.load().memory().nCtx());
        assertEquals("the CPU engine stays without a GPU flag", EngineConfig.BackendMode.CPU, o.load().backend());
        assertRefused("--pooling: pooling avg is invalid", "m.gguf", "--context-file", a.toString(),
                "--pooling", "avg");
        assertRefused("--parallel 0 is outside [1, 256]", "m.gguf", "--context-file", a.toString(),
                "--parallel", "0");
        assertRefused("--batch-size 0 is outside [1, ", "m.gguf", "--context-file", a.toString(),
                "--batch-size", "0");
    }
}
