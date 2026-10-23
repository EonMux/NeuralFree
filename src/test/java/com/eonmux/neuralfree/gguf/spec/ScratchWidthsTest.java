package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * {@link ScratchWidths} over the synthetic GGUF files of {@code SyntheticModels}: the {@code LOGITS} slot is as wide
 * as the LM head's vocabulary rows, and a model without an LM head (a BERT classifier) gets the 1-wide minimum
 * whatever its {@code cls} tensors hold, as only the logits heads write that slot.
 */
public class ScratchWidthsTest {

    private static ModelSpec build(GgufTestWriter w) throws IOException {
        Path path = w.writeTempFile();
        GGUFFileParser parser = null;
        try {
            parser = new GGUFFileParser(path.toString());
            return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    @Test
    public void logitsAreTheVocabularyRowsOfTheLmHead() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        ScratchWidths w = ScratchWidths.of(build(SyntheticModels.llama(c)));

        assertEquals(c.nVocab(), w.logits());
        assertEquals(c.nEmbd(), w.stream());
    }

    @Test
    public void classifierWithoutLmHeadKeepsTheMinimumLogitsWidth() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter file = SyntheticModels.bert(c);
        file.tensorZeros("cls.weight", GGMLType.F32, c.nEmbd(), c.nEmbd());
        file.tensorZeros("cls.output.weight", GGMLType.F32, c.nEmbd(), 1);
        ModelSpec spec = build(file);
        assertNotNull(spec.head().cls());
        assertEquals(0L, spec.head().vocabRows());

        ScratchWidths w = ScratchWidths.of(spec);

        assertEquals(1L, w.logits());
        assertEquals(c.nEmbd(), w.stream());
    }

    @Test
    public void rowsCountAWidthBelowOneAsOne() {
        assertEquals(8L, ScratchWidths.rows(8, 0));
        assertEquals(24L, ScratchWidths.rows(8, 3));
    }
}
