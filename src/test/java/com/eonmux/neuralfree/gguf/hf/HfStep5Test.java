package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Step5 checkpoint loading: vLLM's {@code Step5ForCausalLM} / {@code Step5MTP} geometry (GQA with a per-head output
 * gate, zero-centered RMS norms, the weighted-ReLU sparse indexer over 8-token region summaries, dense and
 * routed-plus-shared-expert FFNs and the appended MTP block) converts to the NF GGUF layout, the export round trip
 * preserves it, the hparams hook reads the sparse geometry and the per-block MoE mask, and unsupported sparse
 * configurations are refused by name.
 */
public class HfStep5Test {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 32;
    private static final int V = 7;
    private static final int HEADS = 8;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int ROTARY = 4;
    private static final int FF = 6;
    private static final int EXPERTS = 2;
    private static final int MOE_FF = 6;
    private static final int SHARE_FF = 4;
    private static final int PROXY = 256;
    private static final int IDX_HEADS = 4;
    private static final int IDX_ROPE = 32;
    private static final float EPS = 1e-5f;
    private static final float THETA = 10000f;

    private static final String SPARSE = "'sparse_config':{'enabled':true,'proxy_dim':" + PROXY
            + ",'sparse_indexer_rope_dim':" + IDX_ROPE + ",'sparse_indexer_use_rope':true"
            + ",'sparse_indexer_num_heads':" + IDX_HEADS + ",'sparse_indexer_num_k_heads':1"
            + ",'sparse_indexer_q_norm_type':'rmsnorm','sparse_indexer_k_norm_type':'layernorm'"
            + ",'sparse_indexer_csa_z_norm_type':'none','index_tp_size':" + KV_HEADS
            + ",'topk':8,'region_block_size':8,'compression_method':'csa_block_compress'"
            + ",'attention_impl':'sparse_gqa','sparse_indexer_softmax_variant':'softmax'"
            + ",'sparse_indexer_ssmax_s_granularity':'q_head','apply_to_layer_types':['full_attention']}";

    private static final String CONFIG_RAW = "{'architectures':['Step5ForCausalLM'],"
            + "'model_type':'step5','hidden_size':" + D + ",'intermediate_size':" + FF
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'num_attention_groups':" + KV_HEADS
            + ",'head_dim':" + HEAD_DIM + ",'partial_rotary_factor':" + ((float) ROTARY / HEAD_DIM)
            + ",'max_position_embeddings':16,'rms_norm_eps':" + EPS + ",'vocab_size':" + V
            + ",'rope_theta':" + THETA + ",'hidden_act':'silu','tie_word_embeddings':false"
            + ",'zero_centered':true,'fp32_residual_connection':true,'att_impl_type':'GQA'"
            + ",'use_head_wise_attn_gate':true,'num_nextn_predict_layers':1"
            + ",'moe_num_experts':" + EXPERTS + ",'moe_top_k':1,'moe_intermediate_size':" + MOE_FF
            + ",'share_expert_dim':" + SHARE_FF + ",'moe_layer_list':[1],'use_moe_router_bias':true"
            + ",'need_fp32_gate':true,'moe_router_activation':'sigmoid','norm_expert_weight':false"
            + ",'moe_router_scaling_factor':2.5," + SPARSE + "}";
    private static final String CONFIG = SyntheticHf.json(CONFIG_RAW);
    @Test
    public void conversionWritesTheStep5Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        try (ModelSource src = open(dir, "layout-cache")) {
            Map<String, Object> md = src.getMetadata();
            assertEquals("step5", md.get("general.architecture"));
            assertEquals(3, ((Number) md.get("step5.block_count")).intValue());
            assertEquals(1, ((Number) md.get("step5.nextn_predict_layers")).intValue());
            assertEquals(IDX_HEADS, ((Number) md.get("step5.attention.indexer.head_count")).intValue());
            assertEquals(PROXY, ((Number) md.get("step5.attention.indexer.key_length")).intValue());
            assertEquals(8, ((Number) md.get("step5.attention.indexer.top_k")).intValue());
            assertEquals(8, ((Number) md.get("step5.attention.indexer.block_size")).intValue());
            assertEquals(IDX_ROPE, ((Number) md.get("step5.attention.indexer.rope_dim")).intValue());
            assertEquals(KV_HEADS, ((Number) md.get("step5.attention.indexer.group_count")).intValue());
            assertEquals(Boolean.TRUE, md.get("step5.zero_centered_norms"));
            assertEquals(Boolean.TRUE, md.get("step5.fp32_residual_connection"));
            assertEquals(HEAD_DIM, ((Number) md.get("step5.attention.key_length")).intValue());
            assertEquals(ROTARY, ((Number) md.get("step5.rope.dimension_count")).intValue());
            assertEquals(EXPERTS, ((Number) md.get("step5.expert_count")).intValue());
            assertEquals(1, ((Number) md.get("step5.expert_shared_count")).intValue());
            assertEquals(2.5, ((Number) md.get("step5.expert_weights_scale")).doubleValue(), 1e-6);
            Set<String> tensors = tensorNames(src);
            for (String expected : new String[]{
                    "token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                    "blk.0.attn_output.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
                    "blk.0.attn_gate.weight",
                    "blk.0.indexer.attn_q_b.weight", "blk.0.indexer.attn_k.weight", "blk.0.indexer.z.weight",
                    "blk.0.indexer.proj.weight", "blk.0.indexer.q_norm.weight", "blk.0.indexer.k_norm.weight",
                    "blk.0.indexer.k_norm.bias",
                    "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias",
                    "blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight",
                    "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight",
                    "blk.2.nextn.eh_proj.weight", "blk.2.nextn.enorm.weight", "blk.2.nextn.hnorm.weight",
                    "blk.2.nextn.shared_head_norm.weight", "blk.2.nextn.shared_head_head.weight",
                    "blk.2.attn_q.weight", "blk.2.ffn_gate.weight"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
            assertArrayEquals(new long[]{D, IDX_HEADS * PROXY}, dims(src, "blk.0.indexer.attn_q_b.weight"));
            assertArrayEquals(new long[]{D, PROXY}, dims(src, "blk.0.indexer.z.weight"));
            assertArrayEquals(new long[]{D, MOE_FF, EXPERTS}, dims(src, "blk.1.ffn_gate_exps.weight"));
            assertArrayEquals(new long[]{2L * D, D}, dims(src, "blk.2.nextn.eh_proj.weight"));
            for (String t : tensors) {
                assertFalse("ssmax_s is not a NeuralFree tensor: " + t, t.contains("ssmax"));
            }
        }
    }

    @Test
    public void slidingBlocksConvertWithoutIndexerTensors() throws Exception {
        String config = CONFIG_RAW.replace("'num_nextn_predict_layers':1",
                "'num_nextn_predict_layers':1,'layer_types':['sliding_attention','full_attention']"
                        + ",'sliding_window':4");
        Path dir = builder(SyntheticHf.json(config), false).write(tmp.newFolder("sliding").toPath());
        try (ModelSource src = open(dir, "sliding-cache")) {
            Map<String, Object> md = src.getMetadata();
            assertEquals(4, ((Number) md.get("step5.attention.sliding_window")).intValue());
            assertEquals(Arrays.asList(true, false, false),
                    md.get("step5.attention.sliding_window_pattern"));
            Set<String> tensors = tensorNames(src);
            for (String t : tensors) {
                assertFalse("sliding block 0 has no indexer: " + t, t.startsWith("blk.0.indexer."));
            }
            assertTrue(tensors.contains("blk.1.indexer.z.weight"));
            assertTrue(tensors.contains("blk.1.indexer.k_norm.bias"));
        }
    }

    @Test
    public void exportedGgufReloadsWithTheSameLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step5.gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("export-cache"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (ModelSource reloaded = open(out, "reload-cache")) {
            assertEquals("step5", reloaded.getMetadata().get("general.architecture"));
            assertEquals(3, ((Number) reloaded.getMetadata().get("step5.block_count")).intValue());
            Set<String> tensors = tensorNames(reloaded);
            assertTrue(tensors.contains("blk.0.indexer.z.weight"));
            assertTrue(tensors.contains("blk.1.ffn_gate_exps.weight"));
            assertTrue(tensors.contains("blk.2.nextn.eh_proj.weight"));
        }
    }

    @Test
    public void hparamsReadTheSparseGeometryAndMoeMask() throws Exception {
        Path dir = builder().write(tmp.newFolder("hparams").toPath());
        try (ModelSource src = open(dir, "hparams-cache")) {
            ModelHparams hp = HparamsLoader.load(GgufKv.of(src), Arch.STEP5);
            assertEquals(3, hp.nLayerAll());
            assertEquals(2, hp.nLayer());
            assertEquals(1, hp.nextnLayers());
            assertEquals(IDX_HEADS, hp.indexerNHead());
            assertEquals(PROXY, hp.indexerHeadSize());
            assertEquals(8, hp.indexerTopK());
            assertEquals(8, hp.indexerBlockSize());
            assertEquals(IDX_ROPE, hp.indexerRopeDim());
            assertEquals(KV_HEADS, hp.indexerGroupCount());
            assertEquals(ROTARY, hp.nRotFull());
            assertTrue(hp.fp32Residual());
            assertTrue(hp.zeroCenteredNorms());
            assertFalse(hp.isMoeLayer(0));
            assertTrue(hp.isMoeLayer(1));
            assertFalse(hp.isMoeLayer(2));
        }
    }

    @Test
    public void mtpOnlyKeepsTheDraftBlockAndSharedRoots() throws Exception {
        Path dir = builder().write(tmp.newFolder("mtp-only").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("mtp-only-cache").toPath())
                .draftMode(ModelSourceOptions.DraftMode.MTP_ONLY).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Set<String> tensors = tensorNames(src);
            assertTrue(tensors.contains("blk.2.nextn.eh_proj.weight"));
            assertTrue(tensors.contains("blk.2.attn_q.weight"));
            assertTrue(tensors.contains("token_embd.weight"));
            assertFalse(tensors.contains("blk.0.attn_q.weight"));
        }
    }

    @Test
    public void mtpDraftingKeepsThePlainGreedyTokens() throws Exception {
        Path dir = builder().write(tmp.newFolder("mtp-run-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step5-mtp.gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("mtp-run-cache"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        int[] prompt = {1, 2, 3};
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                .sampler(SamplerConfig.greedyDecoding()).build();
        // the Step5 summary/ring state snapshots for a rejected draft, so the target is planned with rollbacks
        EngineConfig speculative = engineConfig().mtp(true).draftMax(2).build();
        assertTrue(speculative.memory().rollbackSnapshots() >= 2);
        try (EngineModel model = LLMModelFactory.open(out.toString(), speculative)) {
            int[] plain = model.generate(prompt, request, null).tokens();
            model.resetKVCache();
            try (SpeculativeSession session = LLMModelFactory.speculate(model, speculative)) {
                GenerationResult drafted = session.generate(prompt, request, null);
                assertArrayEquals("the Step5 MTP draft keeps the target's greedy tokens", plain,
                        drafted.tokens());
                SpeculativeDecoder.Stats stats = session.lastStats();
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
                assertTrue("the drafter ran verify rounds: " + stats, stats.rounds() > 0);
                assertTrue(stats.accepted() <= stats.drafted());
            }
        }
    }

    private static EngineConfig.Builder engineConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).typeK(KvDType.BF16).typeV(KvDType.BF16)
                        .build());
    }

    @Test
    public void pagedServingRunsTheStep5Mixer() throws Exception {
        Path dir = builder().write(tmp.newFolder("paged-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step5-paged.gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("paged-cache"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(4).parallel(2).typeK(KvDType.BF16)
                        .typeV(KvDType.BF16).build()).build();
        try (GGUFModel model = new GGUFModel(out.toString(), cfg)) {
            assertTrue("the serving memory is paged",
                    model.forwardBackend().memory() instanceof com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory);
            ForwardBatch.Builder batch = ForwardBatch.builder(4);
            batch.add(1, 0, 0, false);
            batch.add(2, 0, 1, false);
            batch.add(3, 1, 0, true);
            batch.add(4, 1, 1, true);
            float[][] logits = model.forwardBackend().forwardBatch(batch.build());
            assertEquals(2, logits.length);
            for (float[] row : logits) {
                assertEquals(V, row.length);
                for (float value : row) {
                    assertTrue("logit " + value + " is not finite", Float.isFinite(value));
                }
            }
        }
    }

    @Test
    public void slidingLayersRunTheWindowedPath() throws Exception {
        String config = CONFIG_RAW.replace("'num_nextn_predict_layers':1",
                "'num_nextn_predict_layers':1,'layer_types':['sliding_attention','full_attention']"
                        + ",'sliding_window':4");
        Path dir = builder(SyntheticHf.json(config), false).write(tmp.newFolder("sliding-run-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step5-sliding.gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("sliding-run-cache"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (EngineModel model = LLMModelFactory.open(out.toString(), engineConfig().build())) {
            GenerationResult r = model.generate(new int[]{1, 2, 3}, GenerationRequest.builder()
                    .maxNewTokens(2).sampler(SamplerConfig.greedyDecoding()).build(), null);
            assertEquals(2, r.tokens().length);
            for (int token : r.tokens()) {
                assertTrue("token " + token + " is outside the vocabulary", token >= 0 && token < V);
            }
        }
    }

    @Test
    public void engineRunsTheConvertedGraph() throws Exception {
        Path dir = builder().write(tmp.newFolder("graph-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step5-graph.gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("graph-cache"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).typeK(KvDType.BF16).typeV(KvDType.BF16)
                        .build()).build();
        try (EngineModel model = LLMModelFactory.open(out.toString(), cfg)) {
            GenerationResult r = model.generate(new int[]{1, 2, 3}, GenerationRequest.builder()
                    .maxNewTokens(2).sampler(SamplerConfig.greedyDecoding()).build(), null);
            assertEquals(2, r.tokens().length);
            for (int token : r.tokens()) {
                assertTrue("token " + token + " is outside the vocabulary", token >= 0 && token < V);
            }
        }
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        for (String name : new String[]{"Step5ForCausalLM", "Step5MTP"}) {
            assertTrue(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
        }
        assertEquals(RopeType.NEOX, ArchRegistry.traits(Arch.STEP5).ropeType());
    }

    @Test
    public void refusesAMissingSparseConfig() throws Exception {
        String config = CONFIG_RAW.replace("," + SPARSE, "");
        assertRefused(config, "sparse_config", "missing-sparse");
    }

    @Test
    public void refusesADenseOnlySparseConfig() throws Exception {
        assertRefused(CONFIG_RAW.replace("'enabled':true", "'enabled':false"), "enabled", "dense-only");
    }

    @Test
    public void refusesANonProxyIndexerHead() throws Exception {
        assertRefused(CONFIG_RAW.replace("'proxy_dim':256", "'proxy_dim':128"), "proxy_dim", "proxy");
    }

    @Test
    public void refusesANonRegionBlock() throws Exception {
        assertRefused(CONFIG_RAW.replace("'region_block_size':8", "'region_block_size':4"), "region_block_size",
                "region");
    }

    @Test
    public void refusesAnOutOfRangeTopK() throws Exception {
        assertRefused(CONFIG_RAW.replace("'topk':8", "'topk':2048"), "topk", "topk");
    }

    @Test
    public void refusesAMismatchedProviderGroupCount() throws Exception {
        assertRefused(CONFIG_RAW.replace("'index_tp_size':2", "'index_tp_size':4"), "index_tp_size", "groups");
    }

    @Test
    public void refusesANonBooleanIndexerSwitch() throws Exception {
        assertRefused(CONFIG_RAW.replace("'sparse_indexer_use_rope':true",
                "'sparse_indexer_use_rope':'yes'"), "sparse_indexer_use_rope", "rope-bool");
    }

    @Test
    public void refusesAMoeMtpBlock() throws Exception {
        assertRefused(CONFIG_RAW.replace("'moe_layer_list':[1]", "'moe_layer_list':[1,2]"), "MTP", "moe-mtp");
    }

    @Test
    public void moeMtpBlockIsAcceptedWhenTheMtpBlockIsNotExported() throws Exception {
        Path dir = builder(SyntheticHf.json(
                CONFIG_RAW.replace("'moe_layer_list':[1]", "'moe_layer_list':[1,2]")))
                .write(tmp.newFolder("moe-mtp-base").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmpCache("moe-mtp-base-cache"))
                .draftMode(ModelSourceOptions.DraftMode.NO_NEXTN).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertEquals(2, ((Number) src.getMetadata().get("step5.block_count")).intValue());
        }
    }

    @Test
    public void refusesANonSiluActivation() throws Exception {
        assertRefused(CONFIG_RAW.replace("'hidden_act':'silu'", "'hidden_act':'gelu'"), "hidden_act", "silu");
    }

    @Test
    public void refusesANoPeLayer() throws Exception {
        assertRefused(CONFIG_RAW.replace("'use_head_wise_attn_gate':true",
                "'use_head_wise_attn_gate':true,'use_rope_layers':[true,false]"), "use_rope_layers", "nope");
    }

    private void assertRefused(String config, String field, String folder) throws Exception {
        Path dir = builder(SyntheticHf.json(config)).write(tmp.newFolder(folder).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> open(dir, folder + "-cache").close());
        assertTrue(e.getMessage(), names(e, field));
    }

    private static boolean names(Throwable e, String field) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains(field)) {
                return true;
            }
        }
        return false;
    }

    private static Set<String> tensorNames(ModelSource src) {
        Set<String> out = new HashSet<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static long[] dims(ModelSource src, String name) {
        for (GGUFTensorInfo t : src.getTensors()) {
            if (t.name().equals(name)) {
                return t.dims();
            }
        }
        throw new AssertionError("missing tensor " + name);
    }

    private static ModelSource open(Path dir, String cache) throws Exception {
        return ModelSources.open(dir.toString(), options(cache), null);
    }

    private static ModelSourceOptions options(String cache) throws Exception {
        return ModelSourceOptions.builder().cacheDir(tmpCache(cache)).build();
    }

    private static Path tmpCache(String name) {
        try {
            return java.nio.file.Files.createTempDirectory(name);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private static SyntheticHf.Builder builder() {
        return builder(CONFIG);
    }

    private static SyntheticHf.Builder builder(String config) {
        return builder(config, true);
    }

    private static SyntheticHf.Builder builder(String config, boolean firstLayerIndexer) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        layer(b, 0, false, firstLayerIndexer);
        layer(b, 1, true, true);
        // the appended MTP block: fusion projections, shared head and a dense decoder body
        String p = "model.layers.2.";
        tensor(b, p + "enorm.weight", D, 1, ones(D));
        tensor(b, p + "hnorm.weight", D, 1, ones(D));
        tensor(b, p + "eh_proj.weight", D, 2 * D, matrix(D, 2 * D, 200));
        tensor(b, p + "shared_head.norm.weight", D, 1, ones(D));
        // the checkpoint spells the shared head "output"; the converter renames it to "head"
        tensor(b, p + "shared_head.output.weight", V, D, matrix(V, D, 201));
        layerBody(b, 2, false, 300);
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 90));
        return b;
    }

    private static void layer(SyntheticHf.Builder b, int il, boolean moe, boolean indexer) {
        layerBody(b, il, moe, 2 + 40 * il, indexer);
    }

    private static void layerBody(SyntheticHf.Builder b, int il, boolean moe, int seed) {
        layerBody(b, il, moe, seed, true);
    }

    private static void layerBody(SyntheticHf.Builder b, int il, boolean moe, int seed, boolean indexer) {
        String p = "model.layers." + il + (il >= 2 ? ".mtp_block." : ".");
        tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "self_attn.q_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, seed));
        tensor(b, p + "self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, D, matrix(KV_HEADS * HEAD_DIM, D, seed + 1));
        tensor(b, p + "self_attn.v_proj.weight", KV_HEADS * HEAD_DIM, D, matrix(KV_HEADS * HEAD_DIM, D, seed + 2));
        tensor(b, p + "self_attn.o_proj.weight", D, HEADS * HEAD_DIM, matrix(D, HEADS * HEAD_DIM, seed + 3));
        tensor(b, p + "self_attn.q_norm.weight", HEAD_DIM, 1, ones(HEAD_DIM));
        tensor(b, p + "self_attn.k_norm.weight", HEAD_DIM, 1, ones(HEAD_DIM));
        tensor(b, p + "self_attn.g_proj.weight", HEADS, D, matrix(HEADS, D, seed + 4));
        // vLLM registers but never consumes ssmax_s; the converter drops the checkpoint tensor
        tensor(b, p + "self_attn.ssmax_s", HEADS, 1, new float[HEADS]);
        if (indexer) {
            tensor(b, p + "self_attn.sparse_indexer_q.weight", IDX_HEADS * PROXY, D,
                    matrix(IDX_HEADS * PROXY, D, seed + 5));
            tensor(b, p + "self_attn.sparse_indexer_k.weight", PROXY, D, matrix(PROXY, D, seed + 6));
            tensor(b, p + "self_attn.sparse_indexer_z.weight", PROXY, D, matrix(PROXY, D, seed + 7));
            tensor(b, p + "self_attn.sparse_indexer_w.weight", IDX_HEADS, D, matrix(IDX_HEADS, D, seed + 8));
            tensor(b, p + "self_attn.sparse_indexer_q_norm.weight", PROXY, 1, ones(PROXY));
            tensor(b, p + "self_attn.sparse_indexer_k_norm.weight", PROXY, 1, ones(PROXY));
            tensor(b, p + "self_attn.sparse_indexer_k_norm.bias", PROXY, 1, new float[PROXY]);
        }
        if (!moe) {
            tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, seed + 9));
            tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 10));
            tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 11));
            return;
        }
        tensor(b, p + "moe.gate.weight", EXPERTS, D, matrix(EXPERTS, D, seed + 9));
        tensor(b, p + "moe.router_bias", EXPERTS, 1, new float[EXPERTS]);
        tensor(b, p + "moe.gate_proj.weight", EXPERTS, MOE_FF, D, matrix3(EXPERTS, MOE_FF, D, seed + 10));
        tensor(b, p + "moe.up_proj.weight", EXPERTS, MOE_FF, D, matrix3(EXPERTS, MOE_FF, D, seed + 11));
        tensor(b, p + "moe.down_proj.weight", EXPERTS, D, MOE_FF, matrix3(EXPERTS, D, MOE_FF, seed + 12));
        tensor(b, p + "moe.share_expert.gate_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 13));
        tensor(b, p + "moe.share_expert.up_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 14));
        tensor(b, p + "moe.share_expert.down_proj.weight", D, SHARE_FF, matrix(D, SHARE_FF, seed + 15));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int experts, int rows, int cols, float[] data) {
        b.tensor(name, StDType.F32, new long[]{experts, rows, cols}, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 5) % 19 - 9) * 0.025f;
        }
        return x;
    }

    private static float[] matrix3(int first, int rows, int cols, int seed) {
        return matrix(first * rows, cols, seed);
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1.0f);
        return x;
    }
}
