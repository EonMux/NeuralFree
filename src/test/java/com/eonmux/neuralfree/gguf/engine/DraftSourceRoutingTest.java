package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory.DraftRoute;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A Hugging Face draft of a drafter architecture routes through {@link LLMModelFactory#draftRoute} from a real
 * {@link ModelSources} source without running a model: a synthetic DFlash checkpoint directory ({@code HfDraftHooksA}
 * over LC/conversion/qwen.py:757-1029 — the config.json of the DFlash draft with its {@code dflash_config}) opened as
 * a draft of its gguf target resolves to the {@code dflash} architecture and drafts with {@link DraftRoute#DFLASH}
 * when the target is passed to the draft source, and refuses to open without one. The route needs only the
 * architecture and its metadata, so the fixture carries none of the gate, {@code d2t} or candidate-selector tensors a
 * bound drafter would read.
 */
public class DraftSourceRoutingTest {

    private static final int DIM = 32;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 10;

    /** The draft checkpoint's configuration: the Qwen base fields DFlashModel reads, with its dflash_config. */
    private static final String DFLASH_CONFIG = SyntheticHf.json("{'architectures':['DFlashDraftModel'],"
            + "'hidden_size':" + DIM + ",'intermediate_size':" + 2 * DIM + ",'num_hidden_layers':1,"
            + "'num_attention_heads':" + HEADS + ",'num_key_value_heads':" + KV_HEADS
            + ",'max_position_embeddings':128,'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':" + VOCAB
            + ",'bos_token_id':0,'eos_token_id':1,'dflash_config':{'block_size':8},'target_layer_ids':[3]}");

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<AutoCloseable> closing = new ArrayList<>();

    @After
    public void close() throws Exception {
        for (int i = closing.size() - 1; i >= 0; i--) {
            closing.get(i).close();
        }
    }

    @Test
    public void anHfDflashDraftRoutesToDflashWithItsTargetPassed() throws IOException {
        Path dir = dflashDir();
        Path target = target();
        ModelSource dp = ModelSources.open(dir.toString(), ModelSourceOptions.defaults(), target);
        closing.add(dp);

        // the explicit target reached the draft source, so the draft load keeps it
        assertTrue(dp instanceof HfModelSource);
        assertEquals(Optional.of(target), ((HfModelSource) dp).context().targetPath());
        assertEquals(Arch.DFLASH, Arch.fromGgufName(dp.getString("general.architecture")));
        // a DFLASH route consults no target specification: draftRoute dereferences it only on the NextN paths
        assertEquals(DraftRoute.DFLASH,
                LLMModelFactory.draftRoute(Arch.fromGgufName(dp.getString("general.architecture")), GgufKv.of(dp),
                        TensorCatalog.of(dp), null, DraftType.AUTO, dir.toString()));
    }

    @Test
    public void anHfDflashDraftWithoutItsTargetRefusesToOpen() throws IOException {
        Path dir = dflashDir();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.defaults(), null));
        assertTrue(e.getMessage(), e.getMessage().contains("open it as a draft of its target model"));
    }

    /** A minimal DFlash draft directory: its own token embedding (the mask patch stays out, so no target tensor is
     * read) under the Qwen base configuration. */
    private static SyntheticHf.Builder dflash() {
        return SyntheticHf.builder().config(DFLASH_CONFIG)
                .floats(StDType.BF16, "model.embed_tokens.weight", VOCAB, DIM);
    }

    /** The gguf target of the DRAFT: a gguf file whose tokenizer.* metadata the draft's target dispatch hands over. */
    private Path dflashDir() throws IOException {
        return dflash().write(tmp.newFolder().toPath());
    }

    private Path target() throws IOException {
        Path p = tmp.newFolder().toPath().resolve("target.gguf");
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.u32(w.archKey("embedding_length"), DIM).u32(w.archKey("block_count"), 4)
                .u32(w.archKey("vocab_size"), VOCAB)
                .str(GgufKeys.TOKENIZER_MODEL, "gpt2")
                .strArray(GgufKeys.TOKENIZER_LIST,
                        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                .writeTo(p);
        return p;
    }
}
