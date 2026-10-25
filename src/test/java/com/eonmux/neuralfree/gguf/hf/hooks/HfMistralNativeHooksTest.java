package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfMistralNativeHooks} on Mistral native {@code params.json} checkpoints: LLAMA versus MISTRAL3 selection,
 * the {@code yarn} and {@code llama_4_scaling} metadata, native names mapped without the Q/K permutation, and the
 * FP8 scale renames of the index hook.
 */
public class HfMistralNativeHooksTest {

    private static final String PARAMS = "{\"dim\": 8, \"n_layers\": 1, \"head_dim\": 4, \"hidden_dim\": 16,"
            + " \"n_heads\": 2, \"n_kv_heads\": 1, \"norm_eps\": 1e-05, \"vocab_size\": 10,"
            + " \"rope_theta\": 1000000.0%s}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private HfConfig params(String extra) throws IOException {
        Path dir = tmp.newFolder().toPath();
        Files.write(dir.resolve(HfConfig.PARAMS_JSON), String.format(PARAMS, extra).getBytes(StandardCharsets.UTF_8));
        HfConfig cfg = HfConfigLoader.load(dir, Collections.<String, Object>emptyMap());
        assertTrue(cfg.mistralNative());
        return cfg;
    }

    private static HfArchEntry entry(HfConfig cfg) {
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
    }

    private static HfTensorIndex empty() {
        return HfTensorIndex.of(Collections.<String, RowSource>emptyMap());
    }

    @Test
    public void llama4ScalingSelectsMistral3() throws IOException {
        HfConfig plain = params("");
        assertEquals(Arch.LLAMA, entry(plain).arch().resolve(plain, empty()));
        HfConfig scaled = params(", \"llama_4_scaling\": {\"beta\": 0.1, \"original_max_position_embeddings\": 8192}");
        assertEquals(Arch.MISTRAL3, entry(scaled).arch().resolve(scaled, empty()));
        assertSame(entry(plain), entry(scaled));
    }

    @Test
    public void metadataAddsYarnAndTemperatureScaleWithoutVocabSize() throws IOException {
        HfConfig cfg = params(", \"yarn\": {\"factor\": 4.0, \"beta\": 32.0, \"alpha\": 1.0,"
                + " \"original_max_position_embeddings\": 4096, \"apply_scale\": true},"
                + " \"llama_4_scaling\": {\"beta\": 0.1, \"original_max_position_embeddings\": 8192}");
        HfLoadContext ctx = context(cfg, entry(cfg), empty());
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals("yarn", m.get("mistral3.rope.scaling.type"));
        assertEquals(4f, m.get("mistral3.rope.scaling.factor"));
        assertEquals(32f, m.get("mistral3.rope.scaling.yarn_beta_fast"));
        assertEquals(1f, m.get("mistral3.rope.scaling.yarn_beta_slow"));
        assertEquals(0f, m.get("mistral3.rope.scaling.yarn_log_multiplier"));
        assertEquals(4096, m.get("mistral3.rope.scaling.original_context_length"));
        assertEquals(0.1f, m.get("mistral3.attention.temperature_scale"));
        assertEquals(4, m.get("mistral3.rope.dimension_count"));
        assertEquals(8, m.get("mistral3.embedding_length"));
        assertFalse(m.has("mistral3.vocab_size"));

        HfConfig noScale = params(", \"yarn\": {\"factor\": 4.0, \"beta\": 32.0, \"alpha\": 1.0,"
                + " \"original_max_position_embeddings\": 4096, \"apply_scale\": false}");
        HfLoadContext llama = context(noScale, entry(noScale), empty());
        HfMetadataBuilder kv2 = new HfMetadataBuilder(llama.arch().ggufName());
        llama.entry().meta().apply(llama, kv2, null);
        assertEquals(1f, kv2.build().get("llama.rope.scaling.yarn_log_multiplier"));
    }

    @Test
    public void nativeNamesMapWithoutThePermutation() throws IOException {
        ckpt = new Checkpoint(tmp.newFolder().toPath());
        HfTensorIndex index = ckpt.f32("tok_embeddings.weight", new long[]{10, 8}, seq(80))
                .f32("layers.0.attention.wq.weight", new long[]{8, 8}, seq(64))
                .f32("layers.0.attention.wk.weight", new long[]{4, 8}, seq(32)).index();
        HfConfig cfg = params("");
        Sink out = plan(context(cfg, entry(cfg), index));
        assertArrayEquals(seq(64), floats(out.get("blk.0.attn_q.weight")), 0f);
        assertArrayEquals(seq(32), floats(out.get("blk.0.attn_k.weight")), 0f);
        assertArrayEquals(seq(80), floats(out.get("token_embd.weight")), 0f);
    }

    @Test
    public void fp8ScalesAreRenamedInPlace() throws IOException {
        ckpt = new Checkpoint(tmp.newFolder().toPath());
        HfTensorIndex index = ckpt.f32("layers.0.attention.wq.weight", new long[]{8, 8}, seq(64))
                .f32("layers.0.attention.wq.qscale_weight", new long[]{}, 0.5f)
                .f32("layers.0.attention.wq.qscale_act", new long[]{}, 0.25f)
                .f32("norm.weight", new long[]{8}, seq(8)).index();
        HfConfig cfg = params("");
        HfLoadContext ctx = context(cfg, entry(cfg), index);
        HfTensorIndex out = ctx.entry().index().apply(ctx, index);
        assertEquals(Arrays.asList("layers.0.attention.wq.weight", "layers.0.attention.wq.weight_scale",
                "layers.0.attention.wq.input_scale", "norm.weight"), out.names());
        assertSame(index.get("layers.0.attention.wq.qscale_weight"), out.get("layers.0.attention.wq.weight_scale"));
        HfTensorIndex unscaled = HfTensorIndex.of(Collections.singletonMap("norm.weight", index.get("norm.weight")));
        assertSame(unscaled, ctx.entry().index().apply(ctx, unscaled));

        ckpt.close();
        ckpt = new Checkpoint(tmp.newFolder().toPath());
        HfTensorIndex clash = ckpt.f32("w.qscale_weight", new long[]{}, 0.5f).f32("w.weight_scale", new long[]{}, 1f)
                .index();
        assertThrows(GgufFormatException.class, () -> HfMistralNativeHooks.index(ctx, clash));
    }
}
