package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * The ColBERT projector rename (colbert.py:133-134): the checkpoint's {@code linear.*} / {@code
 * colbert_linear.*} become the GGUF {@code dense_2} projection applied to every row of NONE pooling, the weight
 * recording {@code embedding_length_out} from its outer width; every other tensor passes to the backbone's own
 * mapping, and a checkpoint without the projector keeps plain per-token rows.
 */
public class HfColBertHooksTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theProjectorBecomesTheDense2Projection() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "HF_ColBERT", "");
        RecordingSink sink = new RecordingSink();
        HfColBertHooks.bertTensor(ctx, "linear.weight", seq(128, 64), sink);
        assertEquals(java.util.Collections.singleton("dense_2.weight"), sink.tensors.keySet());
        assertEquals(128L, sink.kv.get("bert.embedding_length_out"));
        assertEquals(GGUFValueType.UINT32, sink.kvTypes.get("bert.embedding_length_out"));
    }

    @Test
    public void theColbertLinearPrefixAndBiasTakeTheSameNames() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "HF_ColBERT", "");
        RecordingSink sink = new RecordingSink();
        HfColBertHooks.bertTensor(ctx, "colbert_linear.weight", seq(32, 64), sink);
        HfColBertHooks.bertTensor(ctx, "colbert_linear.bias", seq(32), sink);
        assertEquals(java.util.Arrays.asList("dense_2.weight", "dense_2.bias"),
                new java.util.ArrayList<>(sink.tensors.keySet()));
    }

    @Test
    public void aBackboneTensorPassesToTheFamilyMapping() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder().toPath(), Arch.BERT, "HF_ColBERT", "");
        RecordingSink sink = new RecordingSink();
        HfColBertHooks.bertTensor(ctx, "bert.embeddings.word_embeddings.weight", seq(64, 514), sink);
        assertNull("the projector's metadata key is absent", sink.kv.get("bert.embedding_length_out"));
        assertEquals("the backbone tensor keeps its own mapping",
                java.util.Collections.singleton("token_embd.weight"), sink.tensors.keySet());
    }

    @Test
    public void theModernAndXlmrBackbonesComposeTheirFamilies() throws IOException {
        RecordingSink modern = new RecordingSink();
        HfColBertHooks.modernTensor(context(tmp.newFolder().toPath(), Arch.MODERN_BERT,
                "ColBERTModernBertModel", ""), "colbert_linear.weight", seq(16, 64), modern);
        assertEquals(java.util.Collections.singleton("dense_2.weight"), modern.tensors.keySet());

        RecordingSink xlmr = new RecordingSink();
        HfColBertHooks.xlmRobertaTensor(context(tmp.newFolder().toPath(), Arch.BERT,
                "ColBERTJinaRobertaModel", "'num_position_embeddings': 514"), "linear.weight", seq(16, 64), xlmr);
        assertEquals(java.util.Collections.singleton("dense_2.weight"), xlmr.tensors.keySet());
    }
}
