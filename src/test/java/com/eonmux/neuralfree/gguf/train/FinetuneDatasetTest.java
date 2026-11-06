package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The windowing and the train/validation split of a corpus: half-context stride, next-token labels over every
 * position but the last, and the truncated split llama-finetune uses.
 */
public class FinetuneDatasetTest {

    private static Path file;
    private static GGUFModel model;

    @BeforeClass
    public static void open() throws IOException {
        file = SyntheticModels.llama().writeTempFile();
        model = new GGUFModel(file.toString(), config());
    }

    @AfterClass
    public static void close() throws IOException {
        if (model != null) {
            model.close();
        }
        if (file != null) {
            Files.deleteIfExists(file);
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A token stream whose value at {@code i} is {@code i % 251}; the corpus is longer than one window. */
    private static int[] tokens(int length) {
        int[] t = new int[length];
        for (int i = 0; i < length; i++) {
            t[i] = i % 251;
        }
        return t;
    }

    @Test
    public void windowsStrideByHalfAContextAndPredictTheNextToken() {
        int[] tokens = tokens(29);
        List<FinetuneDataset.Window> windows = FinetuneDataset.windows(tokens, 8);
        // starts 0, 4, 8, 12, 16, 20; 24 + 8 > 29
        assertEquals(6, windows.size());
        int[] expectedTokens = {0, 1, 2, 3, 4, 5, 6};
        int[] expectedLabels = {1, 2, 3, 4, 5, 6, 7};
        assertArrayEquals(expectedTokens, windows.get(0).tokens());
        assertArrayEquals(expectedLabels, windows.get(0).labels());
        assertEquals(7, windows.get(0).positions());
        assertArrayEquals(new int[] {4, 5, 6, 7, 8, 9, 10}, windows.get(1).tokens());
        assertArrayEquals(new int[] {20, 21, 22, 23, 24, 25, 26}, windows.get(5).tokens());
        assertArrayEquals(new int[] {21, 22, 23, 24, 25, 26, 27}, windows.get(5).labels());
    }

    @Test
    public void aWindowHandsOutCopies() {
        FinetuneDataset.Window window = FinetuneDataset.windows(tokens(10), 4).get(0);
        window.tokens()[0] = 200;
        window.labels()[0] = 200;
        assertArrayEquals(new int[] {0, 1, 2}, window.tokens());
        assertArrayEquals(new int[] {1, 2, 3}, window.labels());
    }

    @Test
    public void theValidationPartIsTheTruncatedTrailingFraction() {
        FinetuneDataset data = FinetuneDataset.of(tokens(88), 8, 0.2);
        // (88 - 8) / 4 + 1 = 21 windows; 21 * 0.2 = 4.2 truncates to 4
        assertEquals(21, data.training().size() + data.validation().size());
        assertEquals(4, data.validation().size());
        assertEquals(17, data.training().size());
        assertArrayEquals(new int[] {80, 81, 82, 83, 84, 85, 86}, data.validation().get(3).tokens());
    }

    @Test
    public void aSmallFractionTruncatesToNoValidationWindows() {
        int[] tokens = tokens(24);
        FinetuneDataset data = FinetuneDataset.of(tokens, 8, 0.05);
        assertEquals(5, data.training().size());
        assertTrue(data.validation().isEmpty());
    }

    @Test
    public void theCorpusTextIsTokenizedWithItsSpecials() {
        String text = "the quick brown fox jumps over the lazy dog and the slow turtle";
        int[] encoded = model.tokenizer().encode(text, true, true);
        FinetuneDataset data = FinetuneDataset.ofText(text, model.tokenizer(), 16, 0.0);
        assertEquals(encoded.length, data.corpusTokens());
        // the SPM vocabulary adds the BOS
        assertEquals(model.tokenizer().bosId(), data.training().get(0).tokens()[0]);
        assertEquals((encoded.length - 16) / 8 + 1, data.training().size() + data.validation().size());
        assertArrayEquals(java.util.Arrays.copyOfRange(encoded, 0, 15), data.training().get(0).tokens());
    }

    @Test
    public void aCorpusShorterThanOneWindowIsRefused() {
        try {
            FinetuneDataset.of(tokens(7), 8, 0.0);
            throw new AssertionError("a corpus shorter than a window was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("fewer than one window"));
        }
        try {
            FinetuneDataset.ofText("hi", model.tokenizer(), 16, 0.0);
            throw new AssertionError("a text shorter than a window was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("tokenizes to"));
        }
    }

    @Test
    public void aSplitThatLeavesNothingToTrainOrLeavesTheUnitIntervalIsRefused() {
        try {
            FinetuneDataset.of(tokens(24), 8, 1.0);
            throw new AssertionError("a validation split of the whole corpus was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--val-split"));
        }
        int[] corpus = tokens(40);
        try {
            FinetuneDataset.of(corpus, 8, -0.5);
            throw new AssertionError("a negative validation split was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--val-split"));
        }
        try {
            FinetuneDataset.of(corpus, 2, 0.0);
            throw new AssertionError("a context of 2 was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("at least 3"));
        }
    }
}
