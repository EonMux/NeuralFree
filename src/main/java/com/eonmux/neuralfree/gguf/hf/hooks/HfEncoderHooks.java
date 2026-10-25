package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.UnaryOperator;

/**
 * Hooks of the encoder checkpoints that are not BERT sub-families (LC/conversion/{gemma,llama,clef,
 * vllm_decision2}.py): the EmbeddingGemma embedders on the HK-B2 Gemma hooks ({@code EmbeddingGemma},
 * gemma.py:186-256, and {@code EmbeddingGemma2Model}, gemma.py:813-832), the sentence-transformers
 * {@code 2_Dense}/{@code 3_Dense} side files both include when the load option
 * {@code st_dense_modules} is on (NF default on, llama.cpp converter flag default off — recorded in
 * LLAMACPP-PARITY), and the Qwen3 decision heads: {@code VllmDecision2Model} (vllm_decision2.py:15-107) and
 * {@code ClefModel} (clef.py:19-145). {@code LlamaEmbedNemotronModel} (llama.py:420-423) needs no hooks of
 * its own: it is the HK-0 Llama hooks on the LLAMA_EMBED arch.
 */
public final class HfEncoderHooks {

    /** The modules.json of a sentence-transformers checkpoint. */
    private static final String MODULES_JSON = "modules.json";
    /** The marker of the dense modules the embedders include (gemma.py:200). */
    private static final String DENSE_TYPE = "Dense";
    /** The side tensors file of a dense module (gemma.py:220). */
    private static final String DENSE_TENSORS = "model.safetensors";
    /** The configuration file of a dense module (gemma.py:207). */
    private static final String DENSE_CONFIG = "config.json";
    /** The tensor prefixes dense weights get (gemma.py:230-234). */
    private static final Map<String, String> DENSE_PREFIXES;
    private static final String SUBDIR_BACKBONE = "backbone";                                   // vllm_decision2.py:52
    private static final String UNPACK_CONFIG = "config.json";
    private static final String SCORE_BIAS_CONFIG = "score_bias.json";                          // vllm_decision2.py:31
    private static final String JOINT_HEAD_CONFIG = "joint_head_config.json";                   // clef.py:20
    private static final String JOINT_HEAD_CONFIG_TENSORS = "joint_head.safetensors";
    private static final String DECISION_HEAD_TENSORS = "decision_head.safetensors";            // vllm_decision2.py:61
    /** The layer-norm epsilon of the decision heads (clef.py:44, vllm_decision2.py:42). */
    static final double HEAD_NORM_EPS = 1e-5;
    /** The decision head types of the two converters. */
    static final String CLEF_TYPE = "clef";                                                     // clef.py:101
    static final String DECISION2_TYPE = "vllm-decision2";                                      // vllm_decision2.py:104
    /** The decision formats the loaders detect. */
    static final String DECISION_FORMAT = "vllm-sr-decision";
    /** The one supported backbone of a vllm-sr Decision 2.0 checkpoint (vllm_decision2.py:47). */
    private static final String DECISION2_ARCHITECTURE = "qwen3-text-endpoints-global-query-shared-bilinear-mlp";
    /** The one supported head variant (vllm_decision2.py:47). */
    private static final String DECISION2_VARIANT = "shared";

    static {
        Map<String, String> prefixes = new LinkedHashMap<>();
        prefixes.put("2_Dense", "dense_2");
        prefixes.put("3_Dense", "dense_3");
        DENSE_PREFIXES = Collections.unmodifiableMap(prefixes);
    }

    private HfEncoderHooks() {
    }

    // ================================================================= EmbeddingGemma

    /**
     * {@code EmbeddingGemma.set_gguf_parameters} (gemma.py:236-256) after the Gemma 3 metadata
     * ({@link HfGemmaHooks#meta}): the original {@code sliding_window} of config.json is required (:242-250),
     * the dense module in/out features (gemma.py:251-254) and the sentence-transformers pooling (:256).
     */
    public static void embeddingGemmaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        HfGemmaHooks.meta(ctx, kv, plan);
        if (cfg.raw().raw("sliding_window") == null) {
            throw new GgufFormatException(cfg.source() + ": sliding_window not found in model config - this"
                    + " is required for the model (gemma.py:246)");                      // gemma.py:242-246
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        writeDenseFeatureDims(ctx, kv);                                                  // gemma.py:251-254
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        HfQwenHooks.pooling(ctx, kv);                                                    // gemma.py:256
    }

    /** EmbeddingGemma2Model needs nothing but the register to write its dense weights. */
    public static HfConfig embeddingGemma2Fixup(HfConfig cfg) {
        return cfg;
    }

    // ================================================================= dense side files

    /**
     * The sentence-transformers dense side files of an EmbeddingGemma checkpoint (gemma.py:191-214,
     * :216-228): when the load option {@code st_dense_modules} is on (NF default on, llama.cpp's
     * {@code --sentence-transformers-modules-dense} flag default off — recorded in LLAMACPP-PARITY), the
     * {@code 2_Dense}/{@code 3_Dense} model files become the {@code dense_2}/{@code dense_3} input tensors.
     *
     * @throws IOException when a side file cannot be read or its shape is invalid
     */
    public static HfTensorIndex index(HfLoadContext ctx, HfTensorIndex index) throws IOException {
        if (!ctx.options().stDenseModules()) {
            return index;
        }
        Path modules = ctx.config().sideFile(MODULES_JSON);
        if (!Files.isRegularFile(modules)) {
            return index;
        }
        Map<String, RowSource> add = new LinkedHashMap<>();
        Map<String, long[]> dims = new LinkedHashMap<>();
        for (Map.Entry<String, String> prefix : DENSE_PREFIXES.entrySet()) {
            Path dir = ctx.config().sideFile(prefix.getKey());
            Path tensorsFile = dir.resolve(DENSE_TENSORS);
            if (!Files.isRegularFile(tensorsFile)) {
                continue;
            }
            CheckpointContainer container = ctx.resources().register(
                    CheckpointContainers.open(dir, dir, false));
            Path denseConfig = dir.resolve(DENSE_CONFIG);
            if (Files.isRegularFile(denseConfig)) {
                JsonView conf = JsonView.of(parse(denseConfig), denseConfig.toString());
                if (conf.has("in_features") && conf.has("out_features")) {
                    dims.put(prefix.getValue(), new long[]{conf.i64("in_features"),
                            conf.i64("out_features")});                                    // gemma.py:213-214
                }
            }
            for (StTensor t : container.tensors()) {                                        // gemma.py:222
                String name = t.name();                                                     // gemma.py:222
                if (!name.endsWith(".weight")) {                                            // gemma.py:223-225
                    continue;
                }
                add.put(prefix.getValue() + name.substring(name.lastIndexOf('.')), container.source(name));
            }
        }
        if (add.isEmpty()) {
            return index;
        }
        ctx.state(DENSE_STATE, DenseState.class, DenseState::new).dims = dims;
        Map<String, RowSource> merged = new LinkedHashMap<>();
        for (String name : index.names()) {
            merged.put(name, index.get(name));
        }
        merged.putAll(add);
        return HfTensorIndex.of(merged);
    }

    /** The in/out features the dense tensors carry, in gemma.py:251-254's format. */
    private static void writeDenseFeatureDims(HfLoadContext ctx, HfMetadataBuilder kv) {
        DenseState st = ctx.state(DENSE_STATE, DenseState.class, DenseState::new);
        for (Map.Entry<String, long[]> e : st.dims.entrySet()) {
            if ("dense_2".equals(e.getKey())) {
                kv.set(GgufKeys.DENSE_2_FEAT_IN, e.getValue()[0]);
                kv.set(GgufKeys.DENSE_2_FEAT_OUT, e.getValue()[1]);
            } else if ("dense_3".equals(e.getKey())) {
                kv.set(GgufKeys.DENSE_3_FEAT_IN, e.getValue()[0]);
                kv.set(GgufKeys.DENSE_3_FEAT_OUT, e.getValue()[1]);
            }
        }
    }

    // ================================================================= EmbeddingGemma2

    /**
     * {@code EmbeddingGemma2Model.set_gguf_parameters} (gemma.py:822-828) after the Gemma 4 metadata
     * ({@link HfGemma4Hooks#meta}): HF's bidirectional sliding window doubles, the output embedding length and
     * the non-causal attention; the pooling step closes it.
     */
    public static void embeddingGemma2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfGemma4Hooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, (long) 2 * cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.EMBEDDING_LENGTH_OUT, cfg.requireInt("embedding_dim"));           // gemma.py:826
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);                                          // gemma.py:827
        HfQwenHooks.pooling(ctx, kv);                                                      // gemma.py:828
        writeDenseFeatureDims(ctx, kv);
    }

    // ================================================================= vllm-sr Decision 2.0

    /**
     * {@code _load_vllm_decision2_hparams} (vllm_decision2.py:22-33): the backbone configuration replaces the
     * loader's members, the model config's object is the decision head and its {@code score_bias.json}
     * offsets. NeuralFree resolves the checkpoint by the root {@code config.json} naming
     * {@code VllmDecision2Model}; this fixup rejects such a checkpoint without the package and side files.
     *
     * @throws IOException when a side file cannot be read
     */
    public static HfConfig vllmDecision2Fixup(HfConfig cfg) {
        if (!"VllmDecision2Model".equals(cfg.architecture())) {
            return cfg;
        }
        try {
            return vllmDecision2FixupChecked(cfg);
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": the decision checkpoint side files cannot be read: "
                    + e.getMessage(), e);
        }
    }

    private static HfConfig vllmDecision2FixupChecked(HfConfig cfg) throws IOException {
        if (!"VllmDecision2Model".equals(cfg.architecture())) {
            return cfg;
        }
        JsonView root = cfg.raw();
        JsonView backbone = root.obj("backbone");                                          // vllm_decision2.py:27
        String backboneConfig = backbone.str("config");
        Path backbonePath = cfg.dir().resolve(backboneConfig).getParent();                  // :27 parent
        Map<String, Object> backboneCfg = readObject(cfg.sideFile(backboneConfig));        // :27 read
        Path modelConfigPath = cfg.dir().resolve(root.str("model_config"));
        Map<String, Object> decision = readObject(cfg.sideFile(root.str("model_config")));  // :29
        Map<String, Object> scoreBias = readObject(cfg.dir().resolve(SCORE_BIAS_CONFIG));
        decision.put("score_bias", scoreBias.get("offsets"));                               // :31-32
        Map<String, Object> merged = new LinkedHashMap<>(cfg.flat());
        merged.putAll(backboneCfg);
        merged.put("decision", decision);
        merged.put("backbone_dir", backbonePath.toString());
        return cfg.withFlat(merged);
    }

    /** The step-1 arch of a vllm-sr Decision 2.0 checkpoint: {@link Arch#VLLM_DECISION2}, or nothing. */
    public static Arch vllmDecision2Arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.VLLM_DECISION2;
    }

    /**
     * {@code VllmDecision2Model.set_gguf_parameters} (vllm_decision2.py:102-107) after the HK-B1 Qwen3
     * metadata: the decision head type, the layer-norm epsilon of the head, and one score-bias key per token.
     */
    public static void vllmDecision2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfQwenHooks.qwen3Meta(ctx, kv, plan);
        kv.set(GgufKeys.DECISION_TYPE, DECISION2_TYPE);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, HEAD_NORM_EPS);                            // vllm_decision2.py:105
        writeScoreBias(ctx.config(), kv);                                                   // :106-107
        kv.putAll(HfChatTemplates.toGguf(Collections.singletonList(systemOneTemplate())));
    }

    /** The score-bias keys of vllm_decision2.py:106-107: one per token ({@code n} is the count). */
    static void writeScoreBias(HfConfig cfg, HfMetadataBuilder kv) {
        Map<?, ?> decision = decisionOf(cfg);
        Object bias = decision.get("score_bias");
        if (!(bias instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": the decision head needs a score_bias object");
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) bias).entrySet()) {
            Object offsets = e.getValue();
            if (!(offsets instanceof List)) {
                throw new GgufFormatException(cfg.source() + ": score_bias offsets must be a list");
            }
            List<?> values = (List<?>) offsets;
            for (int i = 0; i < values.size(); i++) {
                kv.setAt(GgufKeys.DECISION_SCORE_BIAS, asDouble(values.get(i), "score_bias offset"),
                        values.size(), i);
            }
        }
    }

    /** The systemone chat template of vllm_decision2.py:75-100 (the pieces tokenized one by one). */
    private static Map<String, String> systemOneTemplate() {
        Map<String, String> template = new LinkedHashMap<>();
        template.put("name", "systemone");
        template.put("template", TEMPLATE_DECISION2);
        return template;
    }

    private static final String TEMPLATE_DECISION2 =
            "{% for q in questions %}{% set d = o.description if o is defined else none %}"
            + "Context:\n{{ state | tojson }}\n\nTask type: {{ q.type }}\nQuestion:\n"
            + "{{ q.instructions | tojson }}\nOptions:"
            + "{% for o in q.options %}{{ sep }}\n<option>\n{{ sep }}{{ mark_option }}"
            + "{{ {'description': o.description, 'key': o.key} | tojson }}\n</option>{% endfor %}"
            + "{{ sep }}{{ mark_question }}\n\nSelect the single option best supported by the context and"
            + " instructions.\nDecision:{% endfor %}";

    /**
     * {@code VllmDecision2Model.index_tensors} (vllm_decision2.py:50-62) and {@code get_tensors} (:58-62):
     * the backbone weights live in their own directory and the decision head is a root side file; both merge
     * into the checkpoint index, the head tensors under their {@code decision_head.} names.
     */
    public static HfTensorIndex vllmDecision2Index(HfLoadContext ctx, HfTensorIndex index) throws IOException {
        HfConfig cfg = ctx.config();
        Path backbone = cfg.dir().resolve(SUBDIR_BACKBONE);
        if (!Files.isDirectory(backbone)) {
            throw new GgufFormatException(cfg.source() + ": the backbone directory '" + backbone
                    + "' is missing");
        }
        CheckpointContainer backboneCkpt = ctx.resources().register(
                CheckpointContainers.open(backbone, backbone, false));
        Map<String, RowSource> merged = new LinkedHashMap<>();
        for (StTensor t : backboneCkpt.tensors()) {
            merged.put(t.name(), backboneCkpt.source(t.name()));
        }
        Path decisionFile = cfg.dir().resolve(DECISION_HEAD_TENSORS);
        if (Files.isRegularFile(decisionFile)) {
            CheckpointContainer decisionCkpt = ctx.resources().register(
                    CheckpointContainers.open(decisionFile, cfg.dir(), false));
            for (StTensor t : decisionCkpt.tensors()) {
                merged.put("decision_head." + t.name(), decisionCkpt.source(t.name()));
            }
        }
        return HfTensorIndex.of(merged);
    }

    /**
     * {@code VllmDecision2Model}'s modify path: the {@code decision_head.} names keep theirs and rely on the
     * converter map, the backbone tensors follow the HK-B1 Qwen3 step.
     */
    public static void vllmDecision2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.startsWith("decision_head.")) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        }
        HfQwenHooks.qwen3Tensors(ctx, hfName, src, out);
    }

    /** {@code tensor_force_quant} (vllm_decision2.py:64-68): the decision head is computed in F32. */
    public static GGMLType decisionHeadForceType(String ggufName, int nDims,
                                                 com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat source,
                                                 ModelSourceOptions.OutType outType) {
        if (ggufName.startsWith("decision.")) {
            return GGMLType.F32;
        }
        return null;
    }

    // ================================================================= Clef

    /**
     * {@code _load_clef_hparams} (clef.py:23-30): the joint head configuration is stored as {@code decision};
     * everything else stays. NeuralFree resolves the checkpoint by the root {@code config.json} naming
     * {@code ClefModel}.
     */
    public static HfConfig clefFixup(HfConfig cfg) {
        if (!"ClefModel".equals(cfg.architecture())) {
            return cfg;
        }
        try {
            return clefFixupChecked(cfg);
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": the Clef checkpoint side files cannot be read: "
                    + e.getMessage(), e);
        }
    }

    private static HfConfig clefFixupChecked(HfConfig cfg) throws IOException {
        if (!"ClefModel".equals(cfg.architecture())) {
            return cfg;
        }
        Path headConfig = cfg.sideFile(JOINT_HEAD_CONFIG);
        if (!Files.isRegularFile(headConfig)) {
            throw new GgufFormatException(cfg.source() + ": the Clef checkpoint has no " + JOINT_HEAD_CONFIG);
        }
        Map<String, Object> decision = readObject(headConfig);
        Map<String, Object> merged = new LinkedHashMap<>(cfg.flat());
        merged.put("decision", decision);
        return cfg.withFlat(merged);
    }

    private static int decisionRouting(Map<?, ?> decision) {
        Object v = decision.get("routing_layers");
        if (!(v instanceof Number)) {
            throw new GgufFormatException("the joint head configuration needs the int routing_layers");
        }
        return ((Number) v).intValue();
    }

    /**
     * {@code ClefModel.set_gguf_parameters} (clef.py:98-105) after the Qwen3.5 metadata: the decision head
     * keys, its layer-norm epsilon and the systemone chat template.
     */
    public static void clefMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        Map<?, ?> decision = decisionOf(ctx.config());
        kv.set(GgufKeys.DECISION_TYPE, CLEF_TYPE);                                          // clef.py:101
        kv.set(GgufKeys.DECISION_ROUTING_BLOCK_COUNT, decisionRouting(decision));           // clef.py:102
        kv.set(GgufKeys.DECISION_BLOCK_COUNT, decisionInt(decision, "layers"));             // clef.py:103
        kv.set(GgufKeys.DECISION_HEAD_COUNT, decisionInt(decision, "heads"));               // clef.py:104
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, HEAD_NORM_EPS);                            // clef.py:105
        kv.putAll(HfChatTemplates.toGguf(Collections.singletonList(clefSystemOne())));
    }

    /** The systemone chat template of clef.py:59-96. */
    private static Map<String, String> clefSystemOne() {
        Map<String, String> template = new LinkedHashMap<>();
        template.put("name", "systemone");
        template.put("template", TEMPLATE_CLEF);
        return template;
    }

    private static final String TEMPLATE_CLEF =
            "{{ '<|im_start|>system\\nRead the complete state and schema. Decide every field jointly. Each "
            + "answer must be exactly one of that field's allowed options.<|im_end|>\\n<|im_start|>user\\n"
            + "STATE:\\n' }}"
            + "{% if images %}{{ sep }}{% for image in images %}{{ image }}{% endfor %}{{ '\\n' }}{% endif %}"
            + "{{ sep }}{{ state | tojson }}{{ sep }}{{ '\\n\\nSCHEMA FIELDS:\\n' }}"
            + "{% for q in questions %}{{ sep }}{{ '\\nFIELD ' }}{{ loop.index }}{{ '\\nID: ' }}"
            + "{{ q.id | tojson }}{{ '\\nTYPE: ' }}{{ q.type | tojson }}{{ '\\nINSTRUCTION: ' }}"
            + "{{ sep }}{{ mark_question }}{{ q.instructions | tojson }}{{ sep }}"
            + "{{ '\\nALLOWED OPTIONS:\\n' }}{% for o in q.options %}{{ sep }}"
            + "{{ 'OPTION ' }}{{ loop.index }}{{ ': ' }}{{ sep }}{{ mark_option }}"
            + "{{ ({'option_id': o.key} if o.description is none else {'description': o.description,"
            + " 'option_id': o.key}) | tojson }}{{ sep }}{{ '\\n' }}{% endfor %}"
            + "{{ sep }}{{ 'END FIELD\\n' }}{% endfor %}{{ sep }}{{ '\\n<|im_end|>\\n"
            + "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA DECISIONS:' }}";

    /**
     * {@code ClefModel.index_tensors} and {@code get_tensors} (clef.py:107-111): the joint head tensors file
     * becomes the {@code joint_head.*} checkpoint names ahead of the backbone tensors.
     */
    public static HfTensorIndex clefIndex(HfLoadContext ctx, HfTensorIndex index) throws IOException {
        HfConfig cfg = ctx.config();
        Path headFile = cfg.dir().resolve(JOINT_HEAD_CONFIG_TENSORS);
        if (!Files.isRegularFile(headFile)) {
            throw new GgufFormatException(cfg.source() + ": the Clef checkpoint has no "
                    + JOINT_HEAD_CONFIG_TENSORS);
        }
        CheckpointContainer head = ctx.resources().register(
                CheckpointContainers.open(headFile, cfg.dir(), false));
        Map<String, RowSource> merged = new LinkedHashMap<>();
        for (String name : index.names()) {
            merged.put(name, index.get(name));
        }
        for (StTensor t : head.tensors()) {
            merged.put("joint_head." + t.name(), head.source(t.name()));
        }
        return HfTensorIndex.of(merged);
    }

    /**
     * {@code ClefModel.modify_tensors} (clef.py:113-145): the {@code joint_head.} names with the learned
     * scalar decision scales, the routing-block renumber and the {@code in_proj_*} chunk-3 split, then the
     * Qwen3.5 step for the backbone tensors.
     */
    public static void clefTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (!hfName.startsWith("joint_head.")) {
            HfQwenLinearHooks.qwen3NextTensors(ctx, hfName, src, out);
            return;
        }
        String[] parts = hfName.split("\\.", -1);
        if (parts.length == 2 && src.shape().length == 0) {                                  // clef.py:121
            Scales s = ctx.state(SCALES_STATE, Scales.class, Scales::new);
            double value = readScalar(src);
            if ("residual_gate".equals(parts[1])) {
                s.byName.put(parts[1], 1.0 / (1.0 + Math.exp(-value)));                     // clef.py:124
            } else {
                s.byName.put(parts[1], Math.exp(Math.min(value, Math.log(100.0))));          // clef.py:126
            }
            out.consume(hfName);
            if (s.byName.size() == 3) {
                float[] scales = new float[3];
                for (int i = 0; i < SCALES_ORDER.length; i++) {
                    Double v = s.byName.get(SCALES_ORDER[i]);
                    if (v == null) {
                        throw new GgufFormatException("the Clef joint head needs the " + SCALES_ORDER[i]
                                + " scalar");
                    }
                    scales[i] = (float) v.doubleValue();
                }
                out.add(ctx.names().ggufName("DECISION_SCALES", -1, ".weight"),
                        RopeTensors.fromFloats(new long[]{3}, scales));                     // clef.py:129
            }
            return;
        }
        Map<String, RowSource> renamed = new LinkedHashMap<>();
        String name = hfName;
        if (parts.length > 2 && "layers".equals(parts[1])) {                                  // clef.py:133
            parts[2] = String.valueOf((long) Integer.parseInt(parts[2])
                    + decisionRouting(decisionOf(ctx.config())));
            name = String.join(".", parts);
        }
        for (String suffix : new String[]{"weight", "bias"}) {                                // clef.py:138-143
            String inProj = ".in_proj_" + suffix;
            if (name.endsWith(inProj)) {
                String prefix = name.substring(0, name.length() - inProj.length());
                for (int i = 0; i < 3; i++) {
                    renamed.put(prefix + "qkv".charAt(i) + "." + suffix,
                            RowTransforms.chunkRows(src, 3, i));
                }
                break;
            }
        }
        if (!renamed.isEmpty()) {
            for (Map.Entry<String, RowSource> e : renamed.entrySet()) {
                out.add(ctx.names().mapOrThrow(e.getKey()), e.getValue());
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /** A scalar RowSource read: the learned decision scalars stored as plain float values. */
    private static double readScalar(RowSource src) throws IOException {
        if (src.rowBytes() != 4) {
            throw new GgufFormatException("a scalar decision tensor must store one float, got " + src.rowBytes()
                    + " bytes");
        }
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, 1, b);
        b.flip();
        return b.getFloat();
    }

    /** The scale aggregate of clef.py:120-130. */
    private static final String[] SCALES_ORDER = {"prior_logit_scale", "joint_logit_scale", "residual_gate"};
    /** The state keys the per-load hooks register their private holders under. */
    private static final String DENSE_STATE = HfEncoderHooks.class.getName() + ".denseModules";
    private static final String SCALES_STATE = HfEncoderHooks.class.getName() + ".clefScales";

    /** The dense features collected by {@link #index}. */
    private static final class DenseState {
        Map<String, long[]> dims = new LinkedHashMap<>();
    }

    /** The learned decision scalars of {@link #clefTensors}. */
    private static final class Scales {
        Map<String, Double> byName = new LinkedHashMap<>();
    }

    // ------------------------------------------------------------------ shared helpers

    private static Map<?, ?> decisionOf(HfConfig cfg) {
        Object d = cfg.find("decision").orElse(null);
        if (d instanceof Map) {
            return (Map<?, ?>) d;
        }
        throw new GgufFormatException(cfg.source() + ": decision checkpoints need a decision configuration"
                + " object");
    }

    private static int decisionInt(Map<?, ?> decision, String key) {
        Object v = decision.get(key);
        if (!(v instanceof Number)) {
            throw new GgufFormatException("the decision configuration needs the int member '" + key + "'");
        }
        return ((Number) v).intValue();
    }

    private static double asDouble(Object v, String where) {
        if (v instanceof Number) {
            return ((Number) v).doubleValue();
        }
        throw new GgufFormatException(where + " must be a number, got " + v);
    }

    private static Object parse(Path file) throws IOException {
        try {
            return Json.parse(file, HfConfigLoader.MAX_CONFIG_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw new GgufFormatException(e.getMessage(), e);
        }
    }

    private static Map<String, Object> readObject(Path file) throws IOException {
        Object parsed = parse(file);
        if (!(parsed instanceof Map)) {
            throw new GgufFormatException(file + ": expected a JSON object");
        }
        return (Map<String, Object>) parsed;
    }

    private static int asInt(Object v, String where) {
        if (v instanceof Number) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(where + " must be an int, got " + v);
    }
}
