package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The SPLADE head mapping (bert.py:772-783): {@code cls.predictions.transform.dense.*} to {@code cls.*}, the
 * LayerNorm to {@code cls.norm.*}, {@code cls.predictions.decoder.*} to {@code cls.output.*}; the tied {@code
 * cls.predictions.bias} and every other head tensor drop beside their {@code decoder.bias} copy.
 */
public class HfSpladeHooksTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theMlmHeadMapsToTheClassifierChain() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "BertSpladeSparseEmbeddingModel", "");
        RecordingSink sink = new RecordingSink();
        HfSpladeHooks.tensors(ctx, "cls.predictions.transform.dense.weight", seq(3, 4), sink);
        HfSpladeHooks.tensors(ctx, "cls.predictions.transform.dense.bias", seq(3), sink);
        HfSpladeHooks.tensors(ctx, "cls.predictions.transform.LayerNorm.weight", seq(3), sink);
        HfSpladeHooks.tensors(ctx, "cls.predictions.transform.LayerNorm.bias", seq(3), sink);
        HfSpladeHooks.tensors(ctx, "cls.predictions.decoder.weight", seq(4, 3), sink);
        HfSpladeHooks.tensors(ctx, "cls.predictions.decoder.bias", seq(4), sink);
        assertEquals(Arrays.asList("cls.weight", "cls.bias", "cls.norm.weight", "cls.norm.bias",
                "cls.output.weight", "cls.output.bias"), new ArrayList<>(sink.tensors.keySet()));
        assertTrue("no head tensor drops", sink.dropped.isEmpty());
    }

    @Test
    public void theTiedPredictionBiasDrops() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "BertSpladeSparseEmbeddingModel", "");
        RecordingSink sink = new RecordingSink();
        HfSpladeHooks.tensors(ctx, "cls.predictions.bias", seq(4), sink);
        assertTrue(sink.tensors.isEmpty());
        assertTrue("the tied bias drops", sink.dropped.containsKey("cls.predictions.bias"));
    }

    @Test
    public void aBackboneTensorPassesToTheBertMapping() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "BertSpladeSparseEmbeddingModel", "");
        RecordingSink sink = new RecordingSink();
        HfSpladeHooks.tensors(ctx, "bert.embeddings.word_embeddings.weight", seq(4, 514), sink);
        assertEquals("the backbone tensor keeps its own mapping",
                java.util.Collections.singleton("token_embd.weight"), sink.tensors.keySet());
    }
}
