package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.GgufKeyTable;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The EAGLE-3 draft parts of llama.cpp's {@code LlamaModel} (LC/conversion/llama.py:49-104, 106-113, 208-253,
 * 329-355) and the draft layouts vLLM serves with {@code llama_eagle3.Eagle3LlamaForCausalLM}
 * (V/model_executor/models/registry.py:676-682): the {@code transformer_layer_config} flattening, the draft test, the
 * target-derived metadata, the {@code midlayer.} renames and special tensors, the absolute {@code d2t} token map and
 * the target's tokenizer. {@link HfLlamaHooks} calls these hooks when a checkpoint resolves to EAGLE3.
 *
 * <p>The target is the model the draft serves ({@link HfLoadContext#targetPath()}): a Hugging Face checkpoint, read as
 * the converter reads {@code --target-model-dir} (its {@code config.json} and tokenizer files, without converting
 * it), or a GGUF file, opened through {@link HfLoadContext#target()} and read through its metadata.</p>
 *
 * <p>The draft settings are read from the configuration root as llama.cpp does; where the root has none, the
 * {@code eagle_config} object vLLM drafts nest them in supplies {@code eagle_aux_hidden_state_layer_ids} and
 * {@code norm_before_fc} (V/v1/worker/gpu_model_runner.py:5236-5250, V/model_executor/models/llama_eagle3.py:179-187),
 * and a draft without {@code draft_vocab_size} has its {@code vocab_size} as draft vocabulary
 * (llama_eagle3.py:312-316).</p>
 */
public final class HfEagle3Hooks {

    private static final Logger LOG = LogManager.getLogger(HfEagle3Hooks.class);

    /**
     * vLLM's EAGLE-3 draft architectures that share the {@code llama_eagle3} layout (registry.py:676-682) and resolve
     * to EAGLE3 by name.
     */
    public static final List<String> VLLM_ALIASES = Collections.unmodifiableList(Arrays.asList(
            "Eagle3MiniMaxM2ForCausalLM", "Eagle3Qwen2_5vlForCausalLM", "Eagle3Qwen3vlForCausalLM",
            "PEagleDraftModel", "PeagleLlamaForCausalLM"));

    /** The nested layer configuration of Eagle3Speculator checkpoints (llama.py:212-214). */
    public static final String TRANSFORMER_LAYER_CONFIG = "transformer_layer_config";

    private static final String MIDLAYER_PREFIX = "midlayer.";
    private static final String LAYER0_PREFIX = "layers.0.";
    private static final String MODEL_LAYER0_PREFIX = "model.layers.0.";
    private static final String FC_WEIGHT = "fc.weight";
    private static final String INPUT_NORM_WEIGHT = "input_norm.weight";
    private static final String HIDDEN_NORM_SUFFIX = ".hidden_norm.weight";
    private static final String D2T = "d2t";
    private static final String T2D = "t2d";
    /** The parallel-drafting mask of PEagle drafts, used only for parallel drafting (llama_eagle3.py:431-442). */
    private static final String MASK_HIDDEN = "mask_hidden";
    /** vLLM's nested EAGLE-3 settings object. */
    private static final String EAGLE_CONFIG = "eagle_config";
    private static final String AUX_LAYER_IDS = "eagle_aux_hidden_state_layer_ids";
    private static final String TOKENIZER_KEY_PREFIX = GgufKeys.TOKENIZER_PREFIX;
    private static final String TARGET_STATE = HfEagle3Hooks.class.getName() + ".target";
    /** The aux layers of llama.py:77-78 when the draft names none: {@code [2, n // 2, n - 3]}. */
    private static final int FIRST_AUX_LAYER = 2;
    private static final int LAST_AUX_LAYER_OFFSET = 3;

    private HfEagle3Hooks() {
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code LlamaModel.index_tensors}' flattening of {@code transformer_layer_config} (llama.py:212-214), which runs
     * before {@code TextModel.__init__} moves {@code text_config} to the root (LC/conversion/base.py:1521-1523): the
     * nested keys override the root, and {@code text_config} keys still override them.
     *
     * @throws GgufFormatException when {@code transformer_layer_config} is not an object
     */
    public static HfConfig fixup(HfConfig cfg) {
        JsonView root = cfg.raw();
        if (!root.has(TRANSFORMER_LAYER_CONFIG)) {
            return cfg;
        }
        Object tlc = root.raw(TRANSFORMER_LAYER_CONFIG);
        if (!(tlc instanceof Map)) {
            throw new GgufFormatException(root.where(TRANSFORMER_LAYER_CONFIG) + " must be an object");
        }
        JsonView text = cfg.text();
        Map<String, Object> replacements = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) tlc).entrySet()) {
            String key = String.valueOf(e.getKey());
            if (!text.has(key)) {
                replacements.put(key, e.getValue());
            }
        }
        return replacements.isEmpty() ? cfg : cfg.withFlat(replacements);
    }

    /**
     * True for an EAGLE-3 draft: one of {@link #VLLM_ALIASES}, or a configuration that, flattened by {@link #fixup},
     * has {@code draft_vocab_size} and one hidden layer (llama.py:49-50, 216-217).
     *
     * @throws GgufFormatException when a configuration with {@code draft_vocab_size} has no {@code num_hidden_layers}
     */
    public static boolean isEagle3(HfConfig cfg) {
        if (VLLM_ALIASES.contains(cfg.architecture())) {
            return true;
        }
        HfConfig flat = fixup(cfg);
        if (!flat.flat().containsKey("draft_vocab_size")) {
            return false;
        }
        if (!flat.flat().containsKey("num_hidden_layers")) {
            throw new GgufFormatException(cfg.source() + ": an EAGLE-3 configuration with 'draft_vocab_size' needs"
                    + " 'num_hidden_layers'");
        }
        return flat.findInt("num_hidden_layers").orElse(-1) == 1;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The EAGLE-3 keys of {@code LlamaModel.__init__} (llama.py:73-104), read from the draft's own
     * {@code config.json}: {@code target_layers} ({@code eagle_aux_hidden_state_layer_ids} of the root, else of
     * {@code eagle_config}, else {@code [2, n // 2, n - 3]} for the target's {@code n} layers),
     * {@code target_hidden_size} (the draft's, else the target's hidden size), {@code norm_before_residual} and
     * {@code norm_before_fc} (the root's, else {@code eagle_config}'s; default false).
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when a value has the wrong type or the target lacks what the keys need
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        JsonView raw = ctx.config().raw();
        Target target = target(ctx);
        List<Integer> layers = auxLayers(raw);
        if (layers == null) {
            int n = target.numLayers;
            layers = Arrays.asList(FIRST_AUX_LAYER, Math.floorDiv(n, 2), n - LAST_AUX_LAYER_OFFSET);
        }
        ctx.log("EAGLE-3: target_layers = " + layers + " (target model has " + target.numLayers + " layers)");
        kv.set(GgufKeys.TARGET_LAYERS, layers);
        Object own = raw.raw("target_hidden_size");
        int hidden = own != null ? positiveInt(raw.where("target_hidden_size"), own, HfConfig.MAX_WIDTH)
                : target.hiddenSize;
        ctx.log("EAGLE-3: target_hidden_size = " + hidden + " (from " + (own != null ? "EAGLE-3 config"
                : "target model config") + ")");
        kv.set(GgufKeys.TARGET_HIDDEN_SIZE, hidden);
        kv.set(GgufKeys.NORM_BEFORE_RESIDUAL, flag(raw, "norm_before_residual"));
        JsonView fcSource = raw.raw("norm_before_fc") != null ? raw : eagleConfig(raw);
        kv.set(GgufKeys.NORM_BEFORE_FC, flag(fcSource, "norm_before_fc"));
    }

    /**
     * {@code eagle_aux_hidden_state_layer_ids} of the root when truthy (llama.py:76-80), else of {@code eagle_config}
     * when truthy, else null.
     */
    private static List<Integer> auxLayers(JsonView raw) {
        List<Integer> own = auxLayerIds(raw);
        return own != null ? own : auxLayerIds(eagleConfig(raw));
    }

    /**
     * The {@code eagle_config} object of the draft configuration, or an empty view when it has none.
     *
     * @throws com.eonmux.neuralfree.utils.json.JsonException when it is not an object
     */
    private static JsonView eagleConfig(JsonView raw) {
        return raw.optObj(EAGLE_CONFIG);
    }

    /** {@code view}'s {@code eagle_aux_hidden_state_layer_ids} when truthy, else null. */
    private static List<Integer> auxLayerIds(JsonView view) {
        Object v = view.raw(AUX_LAYER_IDS);
        if (v == null || (v instanceof List && ((List<?>) v).isEmpty())) {
            return null;
        }
        if (!(v instanceof List)) {
            throw new GgufFormatException(view.where(AUX_LAYER_IDS) + " must be a list of layer indices");
        }
        List<Integer> out = new ArrayList<>();
        for (Object o : (List<?>) v) {
            if (!(o instanceof Long || o instanceof Integer)) {
                throw new GgufFormatException(view.where(AUX_LAYER_IDS) + " must hold integers, got " + o);
            }
            long x = ((Number) o).longValue();
            if (x < 0 || x >= HfConfig.MAX_LAYERS) {
                throw new GgufFormatException(view.where(AUX_LAYER_IDS) + ": layer " + x + " is outside [0, "
                        + HfConfig.MAX_LAYERS + ")");
            }
            out.add((int) x);
        }
        return Collections.unmodifiableList(out);
    }

    private static boolean flag(JsonView raw, String key) {
        Object v = raw.raw(key);
        if (v == null) {
            return false;
        }
        if (!(v instanceof Boolean)) {
            throw new GgufFormatException(raw.where(key) + " must be a boolean");
        }
        return (Boolean) v;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code LlamaModel.index_tensors}' EAGLE-3 renames (llama.py:216-229): {@code midlayer.*} and
     * {@code layers.0.*} become {@code model.layers.0.*}; other names are kept.
     */
    public static String rename(String hfName) {
        if (hfName.startsWith(MIDLAYER_PREFIX)) {
            return MODEL_LAYER0_PREFIX + hfName.substring(MIDLAYER_PREFIX.length());
        }
        if (hfName.startsWith(LAYER0_PREFIX)) {
            return "model." + hfName;
        }
        return hfName;
    }

    /**
     * The EAGLE-3 tensors of {@code LlamaModel.modify_tensors} (llama.py:235-253) that bypass the Llama mapping:
     * {@code fc.weight} kept as {@code fc.weight}, {@code input_norm.weight} as {@code ENC_OUTPUT_NORM},
     * {@code *.hidden_norm.weight} as {@code ATTN_NORM_2} of its block, {@code d2t} as the absolute I64 token map
     * ({@link #d2t}), {@code t2d} and the PEagle {@code mask_hidden} dropped. Returns false for any other tensor.
     *
     * @param hfName the index name of the tensor
     * @param name   the name after {@link #rename}
     * @throws IOException         when {@code d2t} or the target cannot be read
     * @throws GgufFormatException when {@code d2t} is invalid
     */
    public static boolean tensor(HfLoadContext ctx, String hfName, String name, RowSource src, TensorSink out)
            throws IOException {
        if (name.equals(FC_WEIGHT)) {
            out.add(ctx.names().ggufName("FC", -1, ".weight"), src);
        } else if (name.equals(INPUT_NORM_WEIGHT)) {
            out.add(ctx.names().ggufName("ENC_OUTPUT_NORM", -1, ".weight"), src);
        } else if (name.equals(D2T)) {
            out.add(ctx.names().ggufName("D2T", -1), d2t(ctx, src));
        } else if (name.equals(T2D)) {
            out.drop(hfName, "EAGLE-3 t2d is not used at runtime");
        } else if (name.equals(MASK_HIDDEN)) {
            out.drop(hfName, "the parallel-drafting mask is not used by EAGLE-3 decoding");
        } else if (name.endsWith(HIDDEN_NORM_SUFFIX)) {
            out.add(ctx.names().ggufName("ATTN_NORM_2", HfHookSupport.layerOf(name), ".weight"), src);
        } else {
            return false;
        }
        return true;
    }

    /**
     * {@code prepare_tensors}' {@code d2t} (llama.py:339-355): the draft-to-target offsets plus their index, as I64
     * absolute target token ids, kept integer by {@link ElementTransforms#keepInteger}.
     *
     * @throws GgufFormatException when the map does not have one entry per draft token ({@code draft_vocab_size},
     *                             else {@code vocab_size}), an id falls outside the target vocabulary, or two draft
     *                             tokens map to one target token
     * @throws IOException         when the tensor or the target cannot be read
     */
    public static RowSource d2t(HfLoadContext ctx, RowSource src) throws IOException {
        HfConfig cfg = ctx.config();
        OptionalInt own = cfg.findInt("draft_vocab_size");
        String key = own.isPresent() ? "draft_vocab_size" : "vocab_size";
        int draftVocab = positiveInt(cfg.source() + ": " + key,
                own.isPresent() ? own.getAsInt() : cfg.requireInt("vocab_size"), HfConfig.MAX_VOCAB);
        long[] ids = readIntegers(src, draftVocab);
        long targetVocab = target(ctx).vocabSize;
        for (int i = 0; i < ids.length; i++) {
            long abs = ids[i] > Long.MAX_VALUE - i ? Long.MAX_VALUE : ids[i] + i;
            if (abs < 0 || abs >= targetVocab) {
                throw new GgufFormatException("EAGLE-3 d2t target ids out of range for target vocab size "
                        + targetVocab + " (entry " + i + " maps to " + abs + ")");
            }
            ids[i] = abs;
        }
        BitSet seen = new BitSet((int) targetVocab);
        for (int i = 0; i < ids.length; i++) {
            int id = (int) ids[i];
            if (seen.get(id)) {
                throw new GgufFormatException("EAGLE-3 d2t contains duplicate target ids (" + id + ")");
            }
            seen.set(id);
        }
        return ElementTransforms.keepInteger(RopeTensors.fromLongs(new long[]{ids.length}, ids));
    }

    /**
     * The elements of {@code src} as {@code long}s, as {@code data_torch.to(torch.int64)} reads them: integer
     * formats exactly, float formats truncated toward zero.
     *
     * @throws GgufFormatException when the tensor does not hold exactly {@code expected} elements, is a block format,
     *                             or holds a value an {@code int64} cannot represent
     */
    static long[] readIntegers(RowSource src, int expected) throws IOException {
        long numel = 1;
        for (long d : src.shape()) {
            numel = Math.multiplyExact(numel, d);
        }
        if (numel != expected) {
            throw new GgufFormatException(src.describe() + ": EAGLE-3 d2t has " + numel + " entries, expected one"
                    + " per draft token (" + expected + ")");
        }
        ElementFormat f = src.format();
        int width = elementBytes(f);
        if (width <= 0) {
            throw new GgufFormatException(src.describe() + ": EAGLE-3 d2t must be an integer or float tensor, not "
                    + f);
        }
        long bytes = src.rows() * src.rowBytes();
        if (bytes != numel * width) {
            throw new GgufFormatException(src.describe() + ": EAGLE-3 d2t holds " + bytes + " bytes for " + numel
                    + " elements of " + f);
        }
        ByteBuffer buf = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, Math.toIntExact(src.rows()), buf);
        long[] out = new long[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = element(src, f, buf, i * width);
        }
        return out;
    }

    /** The byte width of an element of {@code f}, or -1 when {@code f} is not a plain integer or float format. */
    private static int elementBytes(ElementFormat f) {
        if (f.isBlock()) {
            return -1;
        }
        GGMLType g = f.ggmlType();
        if (g != null) {
            switch (g) {
                case I8: return 1;
                case I16: case F16: case BF16: return 2;
                case I32: case F32: return 4;
                case I64: case F64: return 8;
                default: return -1;
            }
        }
        StDType d = f.stDType();
        if (d == null) {
            return -1;
        }
        switch (d) {
            case BOOL: case U8: return 1;
            case U16: return 2;
            case U32: return 4;
            case U64: return 8;
            default: return -1;
        }
    }

    private static long element(RowSource src, ElementFormat f, ByteBuffer b, int at) {
        GGMLType g = f.ggmlType();
        if (g != null) {
            switch (g) {
                case I8: return b.get(at);
                case I16: return b.getShort(at);
                case I32: return b.getInt(at);
                case I64: return b.getLong(at);
                case F16: return truncate(src, FpBlockDecoders.fp16ToFp32(b.getShort(at) & 0xFFFF));
                case BF16: return truncate(src, FpBlockDecoders.bf16ToFp32(b.getShort(at) & 0xFFFF));
                case F32: return truncate(src, b.getFloat(at));
                case F64: return truncate(src, b.getDouble(at));
                default: break;
            }
        } else {
            switch (f.stDType()) {
                case BOOL: case U8: return b.get(at) & 0xFFL;
                case U16: return b.getShort(at) & 0xFFFFL;
                case U32: return b.getInt(at) & 0xFFFF_FFFFL;
                case U64:
                    long v = b.getLong(at);
                    if (v < 0) {
                        throw new GgufFormatException(src.describe() + ": EAGLE-3 d2t value "
                                + new BigInteger(Long.toUnsignedString(v)) + " does not fit int64");
                    }
                    return v;
                default: break;
            }
        }
        throw new IllegalStateException("unhandled d2t format " + f);
    }

    private static long truncate(RowSource src, double v) {
        if (Double.isNaN(v) || v >= 0x1p63 || v < -0x1p63) {
            throw new GgufFormatException(src.describe() + ": EAGLE-3 d2t value " + v + " is not an int64");
        }
        return (long) v;
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code LlamaModel.set_vocab} for an EAGLE-3 draft (llama.py:107-113): the target's tokenizer. A Hugging Face
     * target goes through {@link HfLlamaHooks#llamaVocab} on its tokenizer files, with its {@code config.json} for
     * the special-token ids and the draft's vocabulary size, as the converter's {@code dir_model} swap does; a GGUF
     * target's {@code tokenizer.*} keys are copied. A tokenizer source given in the load options takes precedence.
     *
     * @throws IOException         when the target or its tokenizer files cannot be read
     * @throws GgufFormatException when a GGUF target holds no tokenizer
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.options().tokenizerDir() != null) {
            return HfLlamaHooks.llamaVocab(ctx, pre, ctx.tokenizerFiles(), ctx.config());
        }
        Target target = target(ctx);
        ctx.log("EAGLE-3: using the tokenizer of the target model " + target.path);
        if (target.config == null) {
            if (target.tokenizerKeys.isEmpty()) {
                throw new GgufFormatException("EAGLE-3 target '" + target.path + "' holds no tokenizer.* metadata");
            }
            return target.tokenizerKeys;
        }
        Map<String, Object> sizes = new LinkedHashMap<>();
        Map<String, Object> draft = ctx.config().flat();
        Object vocabSize = draft.get("vocab_size");
        sizes.put("vocab_size", vocabSize);
        sizes.put("vocab_size_per_layer_input", draft.containsKey("vocab_size_per_layer_input")
                ? draft.get("vocab_size_per_layer_input") : vocabSize);
        return HfLlamaHooks.llamaVocab(ctx, pre, HfTokenizerFiles.locate(target.path),
                target.config.withFlat(sizes));
    }

    // ------------------------------------------------------------------ target

    /**
     * What the draft needs from its target, read once per load.
     *
     * @throws IOException         when the target cannot be opened or its configuration read
     * @throws GgufFormatException when the target lacks its hidden size, layer count or vocabulary size
     */
    public static Target target(HfLoadContext ctx) throws IOException {
        return ctx.state(TARGET_STATE, TargetHolder.class, TargetHolder::new).get(ctx);
    }

    /** The target facts of one load. Immutable. */
    public static final class Target {
        /** The target directory (Hugging Face) or file (GGUF). */
        final Path path;
        /** The target's {@code config.json}, null for a GGUF target. */
        final HfConfig config;
        /** A GGUF target's {@code tokenizer.*} metadata (unmodifiable), empty for a Hugging Face target. */
        final Map<String, Object> tokenizerKeys;
        final int hiddenSize;
        final int numLayers;
        final long vocabSize;

        Target(Path path, HfConfig config, Map<String, Object> tokenizerKeys, int hiddenSize, int numLayers,
               long vocabSize) {
            this.path = path;
            this.config = config;
            this.tokenizerKeys = tokenizerKeys;
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.vocabSize = vocabSize;
        }
    }

    /** Loads the {@link Target} on first use; shared by every context of one load. */
    static final class TargetHolder {
        private Target target;

        synchronized Target get(HfLoadContext ctx) throws IOException {
            if (target == null) {
                Path path = ctx.targetPath().orElse(null);
                target = path != null && ModelSources.detect(path) == ModelFormat.SAFETENSORS
                        ? loadCheckpoint(path) : load(ctx.target(), ctx.arch().ggufName());
            }
            return target;
        }
    }

    /**
     * Reads the target facts of the Hugging Face checkpoint at {@code path} (its directory, or one of its
     * {@code .safetensors} files) from its {@code config.json}, as the converter reads {@code --target-model-dir};
     * the checkpoint is not converted.
     */
    private static Target loadCheckpoint(Path path) throws IOException {
        Path p = path.toAbsolutePath().normalize();
        Path dir = Files.isDirectory(p) ? p : p.getParent();
        HfConfig cfg = HfConfigLoader.load(dir, Collections.<String, Object>emptyMap());
        return new Target(dir, cfg, Collections.<String, Object>emptyMap(),
                positiveInt(cfg.source() + ": hidden_size", cfg.requireInt("hidden_size"), HfConfig.MAX_WIDTH),
                positiveInt(cfg.source() + ": num_hidden_layers", cfg.requireInt("num_hidden_layers"),
                        HfConfig.MAX_LAYERS),
                positiveInt(cfg.source() + ": vocab_size", cfg.requireInt("vocab_size"), HfConfig.MAX_VOCAB));
    }

    /**
     * Reads the target facts from the GGUF {@code source}; its {@code tokenizer.*} keys are kept when the
     * converter's key table types them (the vocabulary hook's values are typed through it), others are logged and
     * left out.
     */
    private static Target load(ModelSource source, String draftArch) throws IOException {
        Path path = Paths.get(source.getFilePath());
        switch (source.format()) {
            case GGUF: {
                GgufKv kv = GgufKv.of(source);
                Map<String, Object> keys = new LinkedHashMap<>();
                for (Map.Entry<String, Object> e : source.getMetadata().entrySet()) {
                    if (!e.getKey().startsWith(TOKENIZER_KEY_PREFIX)) {
                        continue;
                    }
                    if (GgufKeyTable.resolve(e.getKey(), draftArch) == null) {
                        LOG.warn("EAGLE-3 target '{}': tokenizer key '{}' has no type in the converter's key table;"
                                + " it is not copied", path, e.getKey());
                        continue;
                    }
                    keys.put(e.getKey(), e.getValue());
                }
                String where = "EAGLE-3 target '" + path + "'";
                return new Target(path, null, Collections.unmodifiableMap(keys),
                        positiveInt(where + " " + kv.key(GgufKeys.EMBEDDING_LENGTH),
                                kv.u32Required(GgufKeys.EMBEDDING_LENGTH), HfConfig.MAX_WIDTH),
                        positiveInt(where + " " + kv.key(GgufKeys.BLOCK_COUNT), kv.u32Required(GgufKeys.BLOCK_COUNT),
                                HfConfig.MAX_LAYERS),
                        ggufVocabSize(source, kv, where));
            }
            default:
                throw new GgufFormatException("EAGLE-3 target '" + path + "' has the unsupported format "
                        + source.format());
        }
    }

    /**
     * A GGUF target's vocabulary size: {@code <arch>.vocab_size}, else the rows of {@code token_embd.weight}, else the
     * length of {@code tokenizer.ggml.tokens}.
     */
    private static long ggufVocabSize(ModelSource source, GgufKv kv, String where) {
        if (kv.has(GgufKeys.VOCAB_SIZE)) {
            return positiveInt(where + " " + kv.key(GgufKeys.VOCAB_SIZE), kv.u32Long(GgufKeys.VOCAB_SIZE, 0),
                    HfConfig.MAX_VOCAB);
        }
        String embd = TensorNames.weight(TensorNames.TOKEN_EMBD);
        for (GGUFTensorInfo t : source.getTensors()) {
            if (t.name().equals(embd) && t.rank() >= 2) {
                return positiveInt(where + " " + embd + " rows", t.dim(1), HfConfig.MAX_VOCAB);
            }
        }
        if (kv.has(GgufKeys.TOKENIZER_LIST)) {
            return positiveInt(where + " " + GgufKeys.TOKENIZER_LIST, kv.arrayLength(GgufKeys.TOKENIZER_LIST),
                    HfConfig.MAX_VOCAB);
        }
        throw new GgufFormatException(where + " has no vocabulary size (" + kv.key(GgufKeys.VOCAB_SIZE) + ", "
                + embd + " or " + GgufKeys.TOKENIZER_LIST + ")");
    }

    private static int positiveInt(String where, Object v, long max) {
        if (!(v instanceof Long || v instanceof Integer)) {
            throw new GgufFormatException(where + " must be an integer, got " + v);
        }
        long x = ((Number) v).longValue();
        if (x <= 0 || x > max) {
            throw new GgufFormatException(where + " " + x + " is outside [1, " + max + "]");
        }
        return (int) x;
    }
}
