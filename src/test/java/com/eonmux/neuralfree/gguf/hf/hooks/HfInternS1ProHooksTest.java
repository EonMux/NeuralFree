package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.key;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfInternS1ProHooks} on a synthetic InternS1-Pro checkpoint: the grouped router fixup (vLLM's
 * {@code router_n_groups}), the Fourier position embedding metadata, and the coefficient matrices the text name map
 * does not cover.
 */
public class HfInternS1ProHooksTest {

    private static final String HF_ARCH = "InternS1ProForConditionalGeneration";

    private static final String MOE = "'num_experts': 4, 'num_experts_per_tok': 2, 'moe_intermediate_size': 6";

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private HfLoadContext context(String extra) throws IOException {
        return HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.INTERN_S1_PRO, HF_ARCH,
                MOE + (extra.isEmpty() ? "" : ", " + extra));
    }

    @Test
    public void groupedRouterCopiesTheGroupCounts() throws IOException {
        HfLoadContext ctx = context("'router_n_groups': 2");
        HfConfig fixed = HfInternS1ProHooks.fixup(ctx.config());
        assertEquals(2, fixed.findInt("n_group").getAsInt());
        assertEquals(2, fixed.findInt("topk_group").getAsInt());
    }

    /** The Qwen3-VL MoE text graph has expert FFNs on every block; a checkpoint that alternates is refused by name. */
    @Test
    public void aCheckpointWithDenseBlocksIsRefusedByName() {
        GgufFormatException step = assertThrows(GgufFormatException.class,
                () -> HfInternS1ProHooks.fixup(fixupConfig("'decoder_sparse_step': 2")));
        assertTrue(step.getMessage(), step.getMessage().contains("decoder_sparse_step"));
        GgufFormatException only = assertThrows(GgufFormatException.class,
                () -> HfInternS1ProHooks.fixup(fixupConfig("'mlp_only_layers': [0]")));
        assertTrue(only.getMessage(), only.getMessage().contains("mlp_only_layers"));
        // the defaults of the family (step 1, no dense-only block) pass through
        assertTrue(HfInternS1ProHooks.fixup(fixupConfig("'decoder_sparse_step': 1, 'mlp_only_layers': []"))
                .findInt("n_group").isEmpty());
    }

    private static HfConfig fixupConfig(String extra) {
        return HfConfigLoader.fromJson(Path.of("."), ("{'architectures':['" + HF_ARCH + "']," + MOE + ","
                + extra + "}").replace('\'', '"'));
    }

    @Test
    public void withoutGroupedRouterTheConfigurationIsUntouched() throws IOException {
        HfLoadContext ctx = context("");
        HfConfig fixed = HfInternS1ProHooks.fixup(ctx.config());
        assertFalse(fixed.findInt("n_group").isPresent());
        assertFalse(fixed.findInt("topk_group").isPresent());
    }

    private static HfMetadata meta(HfLoadContext ctx) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        HfInternS1ProHooks.meta(ctx, kv, planOf());
        return kv.build();
    }

    @Test
    public void fourierMetadataAndPerGroupRouterAreWritten() throws IOException {
        // the entry's config fixup has copied router_n_groups to the generic group keys by the time meta runs
        HfLoadContext ctx = context("'router_n_groups': 2, 'n_group': 2, 'topk_group': 2,"
                + " 'rope_parameters': {'rope_type': 'default', 'fope_init_factor': 0.1, 'fope_sep_head': true,"
                + " 'num_inv_freq': 16}");
        HfMetadata md = meta(ctx);
        assertEquals(true, md.get(key(GgufKeys.ROPE_FOPE_USE, Arch.INTERN_S1_PRO)));
        assertEquals(true, md.get(key(GgufKeys.ROPE_FOPE_SEP_HEAD, Arch.INTERN_S1_PRO)));
        assertEquals(16, md.get(key(GgufKeys.ROPE_FOPE_NUM_INV_FREQ, Arch.INTERN_S1_PRO)));
        assertEquals(0.1f, md.get(key(GgufKeys.ROPE_FOPE_INIT_FACTOR, Arch.INTERN_S1_PRO)));
        assertEquals(true, md.get(key(GgufKeys.EXPERT_PER_GROUP_TOP_K, Arch.INTERN_S1_PRO)));
        assertEquals(true, md.get(key(GgufKeys.EXPERT_WEIGHTS_NORM, Arch.INTERN_S1_PRO)));
        assertEquals(2, md.get("interns1-pro.expert_group_count"));
        assertEquals(2, md.get("interns1-pro.expert_group_used_count"));
    }

    @Test
    public void aCheckpointWithoutFopeForbidsTheFlagAndTheRouter() throws IOException {
        HfLoadContext ctx = context("");
        HfMetadata md = meta(ctx);
        assertEquals(false, md.get(key(GgufKeys.ROPE_FOPE_USE, Arch.INTERN_S1_PRO)));
        assertEquals(false, md.get(key(GgufKeys.ROPE_FOPE_SEP_HEAD, Arch.INTERN_S1_PRO)));
        assertEquals(0, md.get(key(GgufKeys.ROPE_FOPE_NUM_INV_FREQ, Arch.INTERN_S1_PRO)));
        assertEquals(false, md.get(key(GgufKeys.EXPERT_PER_GROUP_TOP_K, Arch.INTERN_S1_PRO)));
        // renormalization follows the checkpoint's norm_topk_prob, which the architecture defaults to true
        assertEquals(true, md.get(key(GgufKeys.EXPERT_WEIGHTS_NORM, Arch.INTERN_S1_PRO)));
        HfLoadContext unnormalized = context("'norm_topk_prob': false");
        assertEquals(false, meta(unnormalized).get(key(GgufKeys.EXPERT_WEIGHTS_NORM, Arch.INTERN_S1_PRO)));
    }

    @Test
    public void theCoefficientMatricesBecomeModelLevelTensors() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        HfInternS1ProHooks.tensors(ctx, "model.language_model.rotary_emb.cos_coef", seq(4, 4, 2), sink);
        HfInternS1ProHooks.tensors(ctx, "model.language_model.rotary_emb.sin_coef", seq(4, 4, 2), sink);
        assertEquals(new HashSet<>(Arrays.asList(TensorNames.weight(TensorNames.ROPE_COS_COEF),
                TensorNames.weight(TensorNames.ROPE_SIN_COEF))), sink.tensors.keySet());
    }

    /**
     * The text backbone flows through the Qwen3-VL MoE tensor hook and the text name map: the architecture's
     * {@code arch_tensors.tsv} rows resolve the per-layer projections.
     */
    @Test
    public void theTextBackboneMapsThroughTheQwen3VlMoeHook() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        HfInternS1ProHooks.tensors(ctx, "model.language_model.layers.0.self_attn.q_proj.weight", seq(4, 8), sink);
        HfInternS1ProHooks.tensors(ctx, "model.language_model.norm.weight", seq(8), sink);
        assertEquals(new HashSet<>(Arrays.asList("blk.0.attn_q.weight", "output_norm.weight")),
                sink.tensors.keySet());
    }

    /** The rotation's own arithmetic stays F32 under every output type, like the vLLM float projection it feeds. */
    @Test
    public void theCoefficientMatricesStayF32() {
        ElementFormat f16 = ElementFormat.of(GGMLType.F16);
        for (String name : Arrays.asList(TensorNames.weight(TensorNames.ROPE_COS_COEF),
                TensorNames.weight(TensorNames.ROPE_SIN_COEF))) {
            assertEquals(name, GGMLType.F32, HfInternS1ProHooks.forceType(name, 3, f16,
                    ModelSourceOptions.OutType.Q8_0));
            assertEquals(name, GGMLType.F32, HfInternS1ProHooks.forceType(name, 3, f16,
                    ModelSourceOptions.OutType.AUTO));
        }
        assertNull(HfInternS1ProHooks.forceType("blk.0.attn_q.weight", 2, f16,
                ModelSourceOptions.OutType.Q8_0));
    }

    /** The one module of the checkpoint's family the engine has no graph for is refused by name. */
    @Test
    public void theTimeSeriesTowerIsRefusedByName() throws IOException {
        HfLoadContext ctx = context("");
        RecordingSink sink = new RecordingSink();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfInternS1ProHooks.tensors(ctx, "model.time_series.encoder.proj.weight", seq(4, 8), sink));
        assertTrue(e.getMessage(), e.getMessage().contains("model.time_series."));
        assertTrue("nothing is planned", sink.tensors.isEmpty());
    }

    private static HfPlanView planOf() {
        return new HfPlanView() {
            @Override
            public List<String> names() {
                return new ArrayList<>();
            }

            @Override
            public boolean has(String ggufName) {
                return false;
            }

            @Override
            public long[] ne(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public GGMLType type(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public float[] readF32(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public long[] readI64(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }
        };
    }
}
