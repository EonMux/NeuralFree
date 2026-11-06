package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek EAGLE-3 HF draft route ({@link HfArchTableLlama}, {@link HfEagle3MlaHooks}): the
 * {@code Eagle3DeepseekV2ForCausalLM} / {@code Eagle3DeepseekV3ForCausalLM} names route to the EAGLE3 row, the
 * configuration fixup defaults the draft vocabulary and the one decoder layer, the metadata carries the DeepSeek MLA
 * geometry beside the EAGLE-3 target keys, and the MLA projections map to the GGUF names the MLA mixer reads, with
 * {@code kv_b_proj} split into the absorbed per-head {@code attn_k_b} / {@code attn_v_b}.
 */
public class HfEagle3MlaHooksTest {

    private static final String ARCH = "Eagle3DeepseekV2ForCausalLM";
    private static final String DRAFT = "{'architectures': ['" + ARCH + "'], 'hidden_size': 64,"
            + " 'intermediate_size': 128, 'num_attention_heads': 4, 'num_hidden_layers': 1, 'vocab_size': 320,"
            + " 'rms_norm_eps': 1e-06, 'draft_vocab_size': 320, 'q_lora_rank': 32, 'kv_lora_rank': 32,"
            + " 'qk_nope_head_dim': 16, 'qk_rope_head_dim': 16, 'v_head_dim': 32,"
            + " 'eagle_aux_hidden_state_layer_ids': [0, 1, 2], 'target_hidden_size': 64%s}";
    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = 32;
    private static final int ROT = 16;
    private static final int Q_LORA = 32;
    private static final int KV_LORA = 32;
    private static final int TARGET_VOCAB = 320;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<AutoCloseable> closing = new ArrayList<>();

    @After
    public void close() throws Exception {
        for (AutoCloseable c : closing) {
            c.close();
        }
    }

    private Path target() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("target.gguf");
        if (!p.toFile().exists()) {
            GgufTestWriter w = new GgufTestWriter().architecture("llama");
            w.u32(w.archKey("embedding_length"), N_EMBD).u32(w.archKey("block_count"), 3)
                    .u32(w.archKey("vocab_size"), TARGET_VOCAB).writeTo(p);
        }
        return p;
    }

    private HfLoadContext draft(String extra, HfTensorIndex index) throws IOException {
        return draftOf(String.format(DRAFT, extra), index);
    }

    private HfLoadContext draftOf(String json, HfTensorIndex index) throws IOException {
        HfConfig cfg = config(json);
        HfArchEntry e = entry(cfg.architecture());
        Arch arch = e.arch().resolve(cfg, index);
        HfResources resources = new HfResources();
        closing.add(resources);
        return HfLoadContext.builder(e.configFixup().apply(cfg), e, arch).index(index)
                .resources(resources).targetPath(target()).build();
    }

    private Checkpoint checkpoint() throws IOException {
        Checkpoint c = new Checkpoint(tmp.newFolder().toPath());
        closing.add(c);
        return c;
    }

    private static HfTensorIndex empty() {
        return HfTensorIndex.of(Collections.<String, RowSource>emptyMap());
    }

    /** A deterministic value sequence of {@code n} elements. */
    private static float[] vals(long n) {
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor too large for a test: " + n);
        }
        return seq((int) n);
    }

    @Test
    public void bothNamesRouteToTheEagle3Row() {
        HfArchEntry v2 = entry(ARCH);
        assertSame(v2, entry("Eagle3DeepseekV3ForCausalLM"));
        assertEquals(Arrays.asList(ARCH, "Eagle3DeepseekV3ForCausalLM"), v2.hfNames());
        assertEquals(Arch.EAGLE3, v2.arch().resolve(config("{'architectures': ['" + ARCH + "'],"
                + " 'hidden_size': 64, 'num_hidden_layers': 1}"), HfTensorIndex.of(Collections.emptyMap())));
        assertTrue(HfArchRegistry.has(ARCH, HfArchEntry.Kind.TEXT));
    }

    @Test
    public void fixupDefaultsTheDraftVocabularyAndTheSingleDecoderLayer() {
        HfConfig fixed = HfEagle3MlaHooks.fixup(config("{'architectures': ['" + ARCH + "'], 'hidden_size': 64,"
                + " 'num_attention_heads': 4, 'vocab_size': 320, 'kv_lora_rank': 32, 'qk_nope_head_dim': 16,"
                + " 'qk_rope_head_dim': 16, 'v_head_dim': 32}"));
        assertEquals(320, fixed.requireInt("draft_vocab_size"));
        assertEquals(1, fixed.requireInt("num_hidden_layers"));
    }

    @Test
    public void unsupportedVariantsAreRefusedByName() {
        assertRefused("'norm_output': true", "norm_output");
        assertRefused("'use_aux_hidden_state': false", "use_aux_hidden_state");
        assertRefused("'num_aux_layers': 4", "num_aux_layers");
    }

    private void assertRefused(String extra, String named) {
        HfConfig cfg = config("{'architectures': ['" + ARCH + "'], 'hidden_size': 64,"
                + " 'num_attention_heads': 4, 'vocab_size': 320, 'kv_lora_rank': 32, 'qk_nope_head_dim': 16,"
                + " 'qk_rope_head_dim': 16, 'v_head_dim': 32, " + extra + "}");
        GgufFormatException e = assertThrows(extra, GgufFormatException.class, () -> HfEagle3MlaHooks.fixup(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains(named));
    }

    @Test
    public void metadataCarriesTheMlaGeometryBesideTheEagle3TargetKeys() throws Exception {
        HfMetadata md = meta(draft("", empty()));
        assertEquals(Arrays.asList(0, 1, 2), md.get("eagle3.target_layers"));
        assertEquals(N_EMBD, md.get("eagle3.target_hidden_size"));
        assertEquals(TARGET_VOCAB, md.get("eagle3.vocab_size"));
        assertEquals(1, md.get("eagle3.attention.head_count_kv"));
        assertEquals(KV_LORA + ROT, md.get("eagle3.attention.key_length"));
        assertEquals(KV_LORA, md.get("eagle3.attention.value_length"));
        assertEquals(HEAD_DIM, md.get("eagle3.attention.key_length_mla"));
        assertEquals(HEAD_DIM, md.get("eagle3.attention.value_length_mla"));
        assertEquals(Q_LORA, md.get("eagle3.attention.q_lora_rank"));
        assertEquals(KV_LORA, md.get("eagle3.attention.kv_lora_rank"));
        assertEquals(ROT, md.get("eagle3.rope.dimension_count"));
    }

    private static HfMetadata meta(HfLoadContext ctx) throws IOException {
        assertEquals(Arch.EAGLE3, ctx.arch());
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }

    @Test
    public void theMlaProjectionsMapAndKvBIsSplitIntoTheAbsorbedHeads() throws IOException {
        long input = 2L * N_EMBD;
        long qWidth = (long) N_HEAD * HEAD_DIM;
        HfTensorIndex index = checkpoint()
                .f32("midlayer.self_attn.q_a_proj.weight", new long[]{Q_LORA, input}, vals(Q_LORA * input))
                .f32("midlayer.self_attn.q_a_layernorm.weight", new long[]{Q_LORA}, vals(Q_LORA))
                .f32("midlayer.self_attn.q_b_proj.weight", new long[]{qWidth, Q_LORA}, vals(qWidth * Q_LORA))
                .f32("midlayer.self_attn.kv_a_proj_with_mqa.weight", new long[]{KV_LORA + ROT, input},
                        vals((KV_LORA + ROT) * input))
                .f32("midlayer.self_attn.kv_a_layernorm.weight", new long[]{KV_LORA}, vals(KV_LORA))
                .f32("midlayer.self_attn.kv_b_proj.weight", new long[]{
                        (long) N_HEAD * (HEAD_DIM - ROT + HEAD_DIM), KV_LORA},
                        vals((long) N_HEAD * (HEAD_DIM - ROT + HEAD_DIM) * KV_LORA))
                .f32("midlayer.self_attn.o_proj.weight", new long[]{N_EMBD, qWidth}, vals(N_EMBD * qWidth))
                .f32("midlayer.input_layernorm.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .f32("midlayer.post_attention_layernorm.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .f32("midlayer.hidden_norm.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .f32("midlayer.mlp.gate_proj.weight", new long[]{128, N_EMBD}, vals(128 * N_EMBD))
                .f32("midlayer.mlp.up_proj.weight", new long[]{128, N_EMBD}, vals(128 * N_EMBD))
                .f32("midlayer.mlp.down_proj.weight", new long[]{N_EMBD, 128}, vals(N_EMBD * 128))
                .f32("norm.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .f32("embed_tokens.weight", new long[]{TARGET_VOCAB, N_EMBD}, vals(TARGET_VOCAB * N_EMBD))
                .f32("lm_head.weight", new long[]{TARGET_VOCAB, N_EMBD}, vals(TARGET_VOCAB * N_EMBD))
                .f32("fc.weight", new long[]{N_EMBD, 3L * N_EMBD}, vals(N_EMBD * 3L * N_EMBD))
                .index();
        Sink out = plan(draft("", index));
        for (String name : new String[]{"blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight",
                "blk.0.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight",
                "blk.0.attn_output.weight", "blk.0.attn_norm.weight", "blk.0.attn_norm_2.weight",
                "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                "output_norm.weight", "token_embd.weight", "output.weight", "fc.weight"}) {
            out.get(name);
        }
        assertArrayEquals("k_b is the absorbed [n_head, kv_lora, nope] tensor",
                new long[]{N_HEAD, KV_LORA, HEAD_DIM - ROT}, out.get("blk.0.attn_k_b.weight").shape());
        assertArrayEquals("v_b is the absorbed [n_head, v_head, kv_lora] tensor",
                new long[]{N_HEAD, HEAD_DIM, KV_LORA}, out.get("blk.0.attn_v_b.weight").shape());
    }

    @Test
    public void aDraftWithoutQaLoraMapsItsPlainQueryProjection() throws IOException {
        HfTensorIndex index = checkpoint()
                .f32("midlayer.self_attn.q_proj.weight", new long[]{(long) N_HEAD * HEAD_DIM, 2L * N_EMBD},
                        vals(2L * N_EMBD * N_HEAD * HEAD_DIM))
                .index();
        String json = String.format(DRAFT, "").replace("'q_lora_rank': 32, ", "");
        Sink out = plan(draftOf(json, index));
        out.get("blk.0.attn_q.weight");
    }

    @Test
    public void theEncoderNormsKeepTheirEagle3Names() throws IOException {
        HfTensorIndex index = checkpoint()
                .f32("fc_norm.0.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .f32("input_norm.weight", new long[]{N_EMBD}, vals(N_EMBD))
                .index();
        Sink out = plan(draft("", index));
        out.get("fc_norm.0.weight");
        out.get("enc.output_norm.weight");
    }
}