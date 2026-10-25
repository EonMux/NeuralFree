package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reads the configuration of a Hugging Face checkpoint directory into an {@link HfConfig}. Every JSON file is
 * parsed with {@link Json.Limits#LENIENT} (duplicate keys: the last wins, logged).
 * <ol>
 *   <li>Mistral native: {@code params.json} without {@code config.json} is translated as LC/conversion/mistral.py
 *       does ({@code MistralModel} / {@code MistralMoeModel}); the architecture is {@value #MISTRAL_NATIVE} or
 *       {@value #MISTRAL_NATIVE_MOE}.</li>
 *   <li>Otherwise the first {@link HfCheckpointDetectors checkpoint detector} that recognizes the directory
 *       synthesizes the root.</li>
 *   <li>Otherwise {@code config.json} is required. Like {@code load_hparams}, its sub-configs {@code llm_config},
 *       {@code lm_config}, {@code thinker_config.text_config}, {@code language_config} and {@code lfm} are copied
 *       to {@code text_config}, in that order; the source keys stay.</li>
 *   <li>The overrides are deep-merged into the root: objects merge recursively, any other value replaces. An
 *       override of the sub-config that {@code text_config} was copied from is merged into {@code text_config}
 *       too, before the overrides' own {@code text_config}.</li>
 *   <li>{@code generation_config.json} is read when present.</li>
 * </ol>
 * Directory and file problems (including malformed JSON) throw {@code IOException("Invalid Hugging Face checkpoint
 * '<dir>': ...")}; configuration semantics throw {@link GgufFormatException}.
 */
public final class HfConfigLoader {

    private static final Logger LOG = LogManager.getLogger(HfConfigLoader.class);

    /** Largest accepted configuration file, in bytes. */
    public static final long MAX_CONFIG_BYTES = 512L << 20;
    /** Architecture name of a dense Mistral native checkpoint. */
    public static final String MISTRAL_NATIVE = "__mistral_native__";
    /** Architecture name of a mixture-of-experts Mistral native checkpoint. */
    public static final String MISTRAL_NATIVE_MOE = "__mistral_native_moe__";

    /** Mistral key to HF key (mistral.py MistralMoeModel config_mapping). */
    private static final String[][] MISTRAL_KEYS = {
            {"dim", "hidden_size"}, {"norm_eps", "rms_norm_eps"}, {"n_kv_heads", "num_key_value_heads"},
            {"n_layers", "num_hidden_layers"}, {"n_heads", "num_attention_heads"}, {"hidden_dim", "intermediate_size"}};
    /** Mistral MoE key to HF key (mistral.py moe_config_map). */
    private static final String[][] MISTRAL_MOE_KEYS = {
            {"route_every_n", "moe_layer_freq"}, {"first_k_dense_replace", "first_k_dense_replace"},
            {"num_experts_per_tok", "num_experts_per_tok"}, {"num_experts", "n_routed_experts"},
            {"expert_hidden_dim", "moe_intermediate_size"}, {"routed_scale", "routed_scaling_factor"},
            {"num_shared_experts", "n_shared_experts"}, {"num_expert_groups", "n_group"},
            {"num_expert_groups_per_tok", "topk_group"}};

    private HfConfigLoader() {
    }

    /** As {@link #load(Path, Map, Path)} with the overrides and base directory of {@code options}. */
    public static HfConfig load(Path configDir, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(options, "options");
        return load(configDir, options.configOverrides(), options.baseDir());
    }

    /** As {@link #load(Path, Map, Path)} without a base model directory. */
    public static HfConfig load(Path configDir, Map<String, Object> overrides) throws IOException {
        return load(configDir, overrides, null);
    }

    /**
     * Loads the configuration of {@code configDir}.
     *
     * @param overrides a JSON object deep-merged into the root, or null/empty for none
     * @param baseDir   the local base model of a LoRA-style decision checkpoint, or null
     * @throws IOException         when the directory or a configuration file is missing, unreadable or malformed
     * @throws GgufFormatException when the configuration content is semantically invalid
     */
    public static HfConfig load(Path configDir, Map<String, Object> overrides, Path baseDir) throws IOException {
        Objects.requireNonNull(configDir, "configDir");
        if (!Files.isDirectory(configDir)) {
            throw new IOException("Invalid Hugging Face checkpoint '" + configDir + "': not a directory");
        }
        Map<String, Object> root;
        String source;
        boolean mistralNative = false;
        Path baseModelDir = null;
        if (!Files.isRegularFile(configDir.resolve(HfConfig.CONFIG_JSON))
                && Files.isRegularFile(configDir.resolve(HfConfig.PARAMS_JSON))) {
            root = mistralRoot(readObject(configDir, configDir.resolve(HfConfig.PARAMS_JSON)));
            source = HfConfig.PARAMS_JSON;
            mistralNative = true;
        } else {
            HfCheckpointDetectors.Detection d = HfCheckpointDetectors.detect(configDir, baseDir).orElse(null);
            if (d != null) {
                root = d.root;
                source = d.source;
                baseModelDir = d.baseModelDir;
            } else {
                root = readConfigJson(configDir, configDir);
                source = HfConfig.CONFIG_JSON;
            }
        }
        if (overrides != null && !overrides.isEmpty()) {
            applyOverrides(root, overrides);
            LOG.info("Applied configuration overrides to {} of {}: {}", source, configDir,
                    Json.toString(overrides, Json.Style.CANONICAL));
        }
        Path generationFile = configDir.resolve(HfConfig.GENERATION_CONFIG_JSON);
        Map<String, Object> generation = Files.isRegularFile(generationFile)
                ? readObject(configDir, generationFile) : null;
        return HfConfig.create(configDir, source, root, generation, mistralNative, baseModelDir);
    }

    /**
     * A configuration over {@code configJson}, the text of a {@code config.json}, with the sub-config renames
     * applied and no overrides; side files resolve against {@code dir}.
     *
     * @throws GgufFormatException when the text is not a valid JSON object or the configuration is invalid
     */
    public static HfConfig fromJson(Path dir, String configJson) {
        Objects.requireNonNull(configJson, "configJson");
        Object parsed;
        try {
            parsed = Json.parse(configJson, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw new GgufFormatException(HfConfig.CONFIG_JSON + ": " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new GgufFormatException(HfConfig.CONFIG_JSON + ": expected a JSON object, got "
                    + HfConfig.typeName(parsed));
        }
        Map<String, Object> root = HfConfig.asObject(parsed);
        applyRenames(root, HfConfig.CONFIG_JSON);
        return HfConfig.create(dir, HfConfig.CONFIG_JSON, root, null, false, null);
    }

    /**
     * The {@code config.json} of {@code configDir} with the sub-config renames applied, as a mutable object
     * (LC base.py load_hparams with {@code guess=False}).
     *
     * @param modelDir the checkpoint directory named in errors
     */
    static Map<String, Object> readConfigJson(Path modelDir, Path configDir) throws IOException {
        Path file = configDir.resolve(HfConfig.CONFIG_JSON);
        if (!Files.isRegularFile(file)) {
            throw new IOException("Invalid Hugging Face checkpoint '" + modelDir + "': " + file
                    + " is missing (no config.json, params.json or recognized checkpoint layout)");
        }
        Map<String, Object> root = readObject(modelDir, file);
        applyRenames(root, String.valueOf(modelDir.normalize().relativize(file.normalize())));
        return root;
    }

    /**
     * The JSON object in {@code file}, mutable.
     *
     * @throws IOException naming {@code modelDir} when the file cannot be read, is too large, is malformed or is
     *                     not an object
     */
    static Map<String, Object> readObject(Path modelDir, Path file) throws IOException {
        Object parsed;
        try {
            parsed = Json.parse(file, MAX_CONFIG_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw new IOException("Invalid Hugging Face checkpoint '" + modelDir + "': " + e.getMessage(), e);
        } catch (IOException e) {
            throw new IOException("Invalid Hugging Face checkpoint '" + modelDir + "': cannot read " + file + ": "
                    + e, e);
        }
        if (!(parsed instanceof Map)) {
            throw new IOException("Invalid Hugging Face checkpoint '" + modelDir + "': " + file
                    + ": expected a JSON object, got " + HfConfig.typeName(parsed));
        }
        return HfConfig.asObject(parsed);
    }

    /** load_hparams: copy the language model sub-config of composite models to {@code text_config}. */
    static void applyRenames(Map<String, Object> root, String source) {
        copyToText(root, "llm_config", source);
        copyToText(root, "lm_config", source);
        if (root.containsKey("thinker_config")) {
            Object thinker = root.get("thinker_config");
            if (!(thinker instanceof Map) || !((Map<?, ?>) thinker).containsKey("text_config")) {
                throw new GgufFormatException(source + ": 'thinker_config' must be an object with 'text_config'");
            }
            putText(root, "thinker_config.text_config", ((Map<?, ?>) thinker).get("text_config"), source);
        }
        copyToText(root, "language_config", source);
        copyToText(root, "lfm", source);
    }

    private static void copyToText(Map<String, Object> root, String key, String source) {
        if (root.containsKey(key)) {
            putText(root, key, root.get(key), source);
        }
    }

    private static void putText(Map<String, Object> root, String key, Object value, String source) {
        if (!(value instanceof Map)) {
            throw new GgufFormatException(source + ": '" + key + "' must be an object, got "
                    + HfConfig.typeName(value));
        }
        root.put("text_config", HfConfig.mutableCopy(value));
    }

    /**
     * Deep-merges {@code overrides} into {@code root}; the override of the sub-config {@code text_config} was copied
     * from (the last of the rename sources present in {@code root}) is merged into {@code text_config} first.
     */
    static void applyOverrides(Map<String, Object> root, Map<String, ?> overrides) {
        Object sourceOverride = null;
        if (root.containsKey("lfm")) {
            sourceOverride = overrides.get("lfm");
        } else if (root.containsKey("language_config")) {
            sourceOverride = overrides.get("language_config");
        } else if (root.containsKey("thinker_config")) {
            Object thinker = overrides.get("thinker_config");
            sourceOverride = thinker instanceof Map ? ((Map<?, ?>) thinker).get("text_config") : null;
        } else if (root.containsKey("lm_config")) {
            sourceOverride = overrides.get("lm_config");
        } else if (root.containsKey("llm_config")) {
            sourceOverride = overrides.get("llm_config");
        }
        if (sourceOverride instanceof Map && root.get("text_config") instanceof Map) {
            Map<String, Object> text = HfConfig.asObject(HfConfig.mutableCopy(root.get("text_config")));
            @SuppressWarnings("unchecked")
            Map<String, ?> nested = (Map<String, ?>) sourceOverride;
            deepMerge(text, nested);
            root.put("text_config", text);
        }
        deepMerge(root, overrides);
    }

    /** Deep merge: objects merge recursively, any other override value replaces the target member. */
    static void deepMerge(Map<String, Object> target, Map<String, ?> overrides) {
        for (Map.Entry<String, ?> e : overrides.entrySet()) {
            Object current = target.get(e.getKey());
            Object value = e.getValue();
            if (current instanceof Map && value instanceof Map) {
                Map<String, Object> merged = HfConfig.asObject(HfConfig.mutableCopy(current));
                @SuppressWarnings("unchecked")
                Map<String, ?> nested = (Map<String, ?>) value;
                deepMerge(merged, nested);
                target.put(e.getKey(), merged);
            } else {
                target.put(e.getKey(), HfConfig.mutableCopy(value));
            }
        }
    }

    // ------------------------------------------------------------------ Mistral native

    /**
     * The HF-shaped root of Mistral native {@code params}: the params keys are kept, the HF names of
     * {@code config_mapping} are added, and a {@code moe} block gets the remaps and defaults of
     * {@code MistralMoeModel}. The {@code yarn}, {@code llama_4_scaling}, {@code vision_encoder} and
     * {@code quantization} blocks stay as they are, after validation.
     */
    static Map<String, Object> mistralRoot(Map<String, Object> params) {
        String src = HfConfig.PARAMS_JSON;
        Map<String, Object> root = HfConfig.asObject(HfConfig.mutableCopy(params));
        Map<String, Object> yarn = optionalObject(root, "yarn", src);
        if (yarn != null) {
            requireNumber(yarn, "factor", "yarn", src);
            requireNumber(yarn, "beta", "yarn", src);
            requireNumber(yarn, "alpha", "yarn", src);
            HfConfig.toLong(src, "yarn.original_max_position_embeddings",
                    requireMember(yarn, "original_max_position_embeddings", "yarn", src));
            if (!(requireMember(yarn, "apply_scale", "yarn", src) instanceof Boolean)) {
                throw new GgufFormatException(src + ": 'yarn.apply_scale' must be a boolean");
            }
        }
        Map<String, Object> llama4 = optionalObject(root, "llama_4_scaling", src);
        if (llama4 != null) {
            requireNumber(llama4, "beta", "llama_4_scaling", src);
        }
        optionalObject(root, "vision_encoder", src);
        optionalObject(root, "quantization", src);
        for (String[] m : MISTRAL_KEYS) {
            if (root.containsKey(m[0])) {
                root.put(m[1], root.get(m[0]));
            }
        }
        Map<String, Object> moe = optionalObject(root, "moe", src);
        if (moe != null) {
            if (yarn == null) {
                throw new GgufFormatException(src + ": a Mistral mixture-of-experts checkpoint requires 'yarn'");
            }
            Object maxPos = root.containsKey("max_position_embeddings")
                    ? root.get("max_position_embeddings") : Long.valueOf(128_000L);
            root.put("model_type", root.containsKey("model_type") ? root.get("model_type") : "transformer");
            root.put("hidden_act", root.containsKey("activation") ? root.get("activation") : "silu");
            root.put("tie_word_embeddings", root.containsKey("tied_embeddings")
                    ? root.get("tied_embeddings") : Boolean.FALSE);
            root.put("max_seq_len", root.containsKey("max_seq_len") ? root.get("max_seq_len") : maxPos);
            root.put("max_position_embeddings", maxPos);
            for (String[] m : MISTRAL_MOE_KEYS) {
                if (moe.containsKey(m[0])) {
                    root.put(m[1], HfConfig.mutableCopy(moe.get(m[0])));
                }
            }
            root.put("topk_method", null);
            root.put("norm_topk_prob", Boolean.TRUE);
            root.put("scoring_func", "softmax");
        }
        List<Object> arch = new ArrayList<>(1);
        arch.add(moe != null ? MISTRAL_NATIVE_MOE : MISTRAL_NATIVE);
        root.put("architectures", arch);
        return root;
    }

    private static Map<String, Object> optionalObject(Map<String, Object> map, String key, String src) {
        Object v = map.get(key);
        if (v == null) {
            return null;
        }
        if (!(v instanceof Map)) {
            throw new GgufFormatException(src + ": '" + key + "' must be an object, got " + HfConfig.typeName(v));
        }
        return HfConfig.asObject(v);
    }

    private static Object requireMember(Map<String, Object> map, String key, String path, String src) {
        Object v = map.get(key);
        if (v == null) {
            throw new GgufFormatException(src + ": required key '" + path + "." + key + "' is missing");
        }
        return v;
    }

    private static void requireNumber(Map<String, Object> map, String key, String path, String src) {
        HfConfig.toDouble(src, path + "." + key, requireMember(map, key, path, src));
    }
}
