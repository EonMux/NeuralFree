package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lookup cache tools over the synthetic llama fixture: {@code lookup-create} writes the static 2-gram cache
 * {@code common_ngram_cache_save} defines, {@code lookup-merge} sums the counts of the files it loads, and
 * {@code lookup-stats} replays a prompt through the cache drafter and reports drafted and accepted tokens.
 */
public class LookupCommandsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. ";

    /** A prompt tokenizing to at least {@code minTokens} tokens over the fixture's SPM vocabulary. */
    private static String prompt(GGUFTokenizer tok, int minTokens) {
        StringBuilder sb = new StringBuilder();
        while (tok.encode(sb.toString()).length < minTokens) {
            sb.append(SENTENCE);
        }
        return sb.toString();
    }

    /** The tokenizer of {@code model}; the tools read tokenizer metadata only, so the source closes at once. */
    private static GGUFTokenizer tokenizer(Path model) throws IOException {
        try (ModelSource source = ModelSources.open(model.toString())) {
            return new GGUFTokenizer(source);
        }
    }

    /** The 2-grams a sequence adds, keyed by "{@code first,second}", as key and following token counts. */
    private static Map<String, Map<Integer, Integer>> expectedTwoGrams(int[] tokens) {
        Map<String, Map<Integer, Integer>> out = new TreeMap<>();
        for (int i = 2; i < tokens.length; i++) {
            String key = tokens[i - 2] + "," + tokens[i - 1];
            Map<Integer, Integer> part = out.get(key);
            if (part == null) {
                part = new TreeMap<>();
                out.put(key, part);
            }
            part.merge(tokens[i], 1, Integer::sum);
        }
        return out;
    }

    private static Map<String, Map<Integer, Integer>> cacheTwoGrams(NgramCache cache) {
        Map<String, Map<Integer, Integer>> out = new TreeMap<>();
        for (Map.Entry<NgramCache.Key, Map<Integer, Integer>> e : cache.entries().entrySet()) {
            NgramCache.Key key = e.getKey();
            assertEquals("only static 2-grams are written", 2, key.size());
            out.put(key.t0() + "," + key.t1(), new TreeMap<>(e.getValue()));
        }
        return out;
    }

    @Test
    public void createWritesTheStaticTwoGramCache() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path out = tmp.newFile("static.bin").toPath();
        String text = prompt(tokenizer(model), 40);

        Path written = LookupCreateCommand.execute(new String[]{model.toString(), "-p", text, "-lcs",
                out.toString()});
        assertEquals(out, written);

        NgramCache cache = NgramCacheFile.load(out);
        assertEquals(expectedTwoGrams(tokenizer(model).encode(text, true, true)), cacheTwoGrams(cache));
        assertTrue(cache.ngrams() > 1);
    }

    @Test
    public void createReadsThePromptFileAndDropsOneTrailingNewline() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path out = tmp.newFile("static.bin").toPath();
        Path promptFile = tmp.newFile("prompt.txt").toPath();
        Files.write(promptFile, (SENTENCE + SENTENCE + "\n").getBytes(StandardCharsets.UTF_8));

        LookupCreateCommand.execute(new String[]{model.toString(), "-f", promptFile.toString(), "-lcs",
                out.toString()});
        NgramCache cache = NgramCacheFile.load(out);
        int[] tokens = tokenizer(model).encode(SENTENCE + SENTENCE, true, true);
        assertEquals(expectedTwoGrams(tokens), cacheTwoGrams(cache));

        assertEquals("one trailing newline is dropped", SENTENCE + SENTENCE,
                LookupToolSupport.readPrompt(promptFile));
    }

    @Test
    public void createRefusesAMissingPromptFileAndOutput() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        assertThrows(LookupCreateCommand.UsageException.class, () -> LookupCreateCommand.execute(
                new String[]{model.toString(), "-f", tmp.getRoot().toPath().resolve("absent.txt").toString(),
                        "-lcs", tmp.newFile("out.bin").toString()}));
        assertThrows(LookupCreateCommand.UsageException.class,
                () -> LookupCreateCommand.execute(new String[]{model.toString(), "-p", "text"}));
        assertThrows(LookupCreateCommand.UsageException.class,
                () -> LookupCreateCommand.execute(new String[]{"-p", "text", "-lcs", "out.bin"}));
    }

    @Test
    public void createHelpPrintsTheSurface() throws Exception {
        assertNull(LookupCreateCommand.execute(new String[]{"--help"}));
    }

    @Test
    public void mergeSumsTheCountsOfEveryInput() throws Exception {
        NgramCache first = new NgramCache();
        first.addCount(NgramCache.Key.of(new int[]{1, 2}, 0, 2), 3, 2);
        NgramCache second = new NgramCache();
        second.addCount(NgramCache.Key.of(new int[]{1, 2}, 0, 2), 3, 1);
        second.addCount(NgramCache.Key.of(new int[]{5, 6}, 0, 2), 7, 4);
        Path a = tmp.newFile("a.bin").toPath();
        Path b = tmp.newFile("b.bin").toPath();
        Path out = tmp.newFile("out.bin").toPath();
        NgramCacheFile.save(first, a);
        NgramCacheFile.save(second, b);

        assertEquals(out, LookupMergeCommand.execute(new String[]{a.toString(), b.toString(), out.toString()}));
        NgramCache merged = NgramCacheFile.load(out);
        assertEquals(3, merged.count(NgramCache.Key.of(new int[]{1, 2}, 0, 2), 3));
        assertEquals(4, merged.count(NgramCache.Key.of(new int[]{5, 6}, 0, 2), 7));
        assertEquals(2, merged.ngrams());
    }

    @Test
    public void mergeRewritesASingleInput() throws Exception {
        NgramCache cache = new NgramCache();
        cache.addSequence(1, 3, new int[]{4, 4, 1, 4, 1, 5});
        Path in = tmp.newFile("in.bin").toPath();
        Path out = tmp.newFile("out.bin").toPath();
        NgramCacheFile.save(cache, in);

        LookupMergeCommand.execute(new String[]{in.toString(), out.toString()});
        NgramCache rewritten = NgramCacheFile.load(out);
        assertEquals(cache.ngrams(), rewritten.ngrams());
        for (NgramCache.Key key : cache.keys()) {
            assertEquals(cache.tokensOf(key), rewritten.tokensOf(key));
            for (int token : cache.tokensOf(key)) {
                assertEquals(cache.count(key, token), rewritten.count(key, token));
            }
        }
    }

    @Test
    public void mergeRefusesTooFewPathsAndUnknownFlags() throws Exception {
        assertThrows(LookupMergeCommand.UsageException.class,
                () -> LookupMergeCommand.execute(new String[]{"only-one.bin"}));
        assertThrows(LookupMergeCommand.UsageException.class,
                () -> LookupMergeCommand.execute(new String[]{"-x", "a.bin", "out.bin"}));
        assertNull(LookupMergeCommand.execute(new String[]{"--help"}));
    }

    @Test
    public void statsReplayThePromptAndAcceptStaticDrafts() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = tokenizer(model);
        String text = prompt(tok, 4 * 8 + 1);
        Path staticCache = tmp.newFile("static.bin").toPath();
        LookupCreateCommand.execute(new String[]{model.toString(), "-p", text, "-lcs",
                staticCache.toString()});

        StringBuilder err = new StringBuilder();
        LookupStatsCommand.Stats stats = LookupStatsCommand.execute(new String[]{model.toString(), "-p", text,
                "-c", "8", "-lcs", staticCache.toString()}, err);

        int tokens = tok.encode(text, true, true).length;
        assertEquals(LookupStatsCommand.DEFAULT_DRAFT_MAX, stats.draftMax);
        assertEquals(tokens - tokens % 8, stats.nPredict);
        assertTrue("drafted at least one token", stats.nDrafted > 0);
        assertTrue("the static 2-grams were accepted", stats.nAccepted > 0);
        assertTrue(stats.acceptPercent > 0.0 && stats.acceptPercent <= 100.0);
        String printed = err.toString();
        assertTrue(printed, printed.contains("n_draft      = 3"));
        assertTrue(printed, printed.contains("n_drafted    = " + stats.nDrafted));
        assertTrue(printed, printed.contains("n_accept     = " + stats.nAccepted));
        assertTrue(printed, printed.contains("accept       = "));
    }

    @Test
    public void statsStartsAMissingDynamicCacheEmptyAndWritesNothing() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        String text = prompt(tokenizer(model), 20);
        Path dynamic = tmp.getRoot().toPath().resolve("absent-dynamic.bin");
        StringBuilder err = new StringBuilder();

        LookupStatsCommand.Stats stats = LookupStatsCommand.execute(new String[]{model.toString(), "-p", text,
                "-c", "8", "-lcd", dynamic.toString()}, err);
        assertEquals("no cache offers a draft", 0, stats.nDrafted);
        assertTrue(err.toString(), err.toString().contains("does not exist; it starts empty"));
        assertTrue("the dynamic file is not created", !Files.exists(dynamic));
    }

    @Test
    public void statsRequiresAnExistingStaticCacheAndAUsableContext() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path missing = tmp.getRoot().toPath().resolve("absent-static.bin");
        assertThrows(IOException.class, () -> LookupStatsCommand.execute(new String[]{model.toString(),
                "-p", "text", "-c", "8", "-lcs", missing.toString()}, new StringBuilder()));
        assertThrows(LookupStatsCommand.UsageException.class,
                () -> LookupStatsCommand.parse(new String[]{model.toString(), "-c", "many"}));
        assertThrows(LookupStatsCommand.UsageException.class,
                () -> LookupStatsCommand.parse(new String[]{"-c", "8"}));
        assertNull(LookupStatsCommand.parse(new String[]{"--help"}));
    }
}
