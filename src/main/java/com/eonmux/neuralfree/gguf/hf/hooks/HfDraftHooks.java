package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The shared draft load of Appendix A table HK-SG: the target model the drafts serve (the converter's
 * {@code --target-model-dir}, a Hugging Face checkpoint or a GGUF file), the {@code mask_embedding} patch and the
 * {@code dflash_config} numbers every draft reads. The draft classes themselves live in {@link HfDraftHooksA}
 * (DFlash, DSpark, MuseGlimmerAssistant) and {@link HfDraftHooksB} (DeepseekV4DSpark, LiLiCorr, the Gemma 4 drafts
 * and the Gemma 4 assistant).
 *
 * <p>Every draft needs {@link HfLoadContext#target()} and reports its named error without one. The target serves the
 * converter's {@code --target-model-dir}: a Hugging Face target gives the target's tokenizer files (with the
 * draft's {@code vocab_size} padding) and the tensors the draft patches; a GGUF target gives its
 * {@code tokenizer.*} metadata. {@code target_layers} shift by one because the configuration counts the layers
 * <em>producing</em> the draft inputs (qwen.py:832-835), and the non-NeoX rope checkpoints arrange each head's rows
 * even-then-odd for the q/k projections and norms (qwen.py:932-955).</p>
 */
public final class HfDraftHooks {

    /** The draft names the target dispatch falls back on, as {@code target_cls is type(self)} would recurse. */
    static final Set<String> DRAFT_NAMES = Set.of("DFlashDraftModel", "DFlash2DraftModel", "Qwen3DSparkModel",
            "DSparkDraftModel", "DSparkSpeculator", "Lfm2DSparkDraftModel", "LingDSparkModel",
            "DeepseekV4DSparkModel", "Gemma4DSparkModel", "LiLiCorrDraftModel", "MuseGlimmerAssistantModel");

    /** The {@code .weight} suffix the draft name maps add. */
    static final String WEIGHT = ".weight";
    static final String DFLASH_CONFIG = "dflash_config";
    private static final String MASK_PT = "mask_embedding.pt";
    private static final String MASK_ST = "mask_embedding.safetensors";
    private static final String MASK_STATE = HfDraftHooks.class.getName() + ".mask";
    private static final String TARGET_STATE = HfDraftHooks.class.getName() + ".target";

    private HfDraftHooks() {
    }

    // ------------------------------------------------------------------ shared helpers

    /** {@code dflash_config} of the flattened configuration, or the empty map. Nested objects are frozen. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> dflashConfig(HfConfig cfg) {
        Object v = cfg.flat().get(DFLASH_CONFIG);
        if (v == null) {
            return Collections.emptyMap();
        }
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": '" + DFLASH_CONFIG
                    + "' must be an object, got " + typeName(v));
        }
        return (Map<String, Object>) v;
    }

    /** {@code dflash_config.mask_token_id}, or -1. */
    static int maskTokenId(HfConfig cfg) {
        Object v = dflashConfig(cfg).get("mask_token_id");
        return v == null ? -1 : (int) longOf(cfg, DFLASH_CONFIG + ".mask_token_id", v);
    }

    /** {@code v} as a long; booleans and non-numbers fail. */
    static long longOf(HfConfig cfg, String where, Object v) {
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be an integer, got " + typeName(v));
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d > Integer.MAX_VALUE || d < Integer.MIN_VALUE) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' value " + d + " is not an int32");
        }
        return (long) d;
    }

    /** {@code v} as a double; non-numbers fail. */
    static double doubleOf(HfConfig cfg, String where, Object v) {
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be a number, got " + typeName(v));
        }
        return ((Number) v).doubleValue();
    }

    private static String typeName(Object v) {
        if (v == null) {
            return "nothing (null)";
        }
        if (v instanceof Boolean) {
            return "boolean";
        }
        if (v instanceof Number) {
            return "a number";
        }
        if (v instanceof String) {
            return "string";
        }
        if (v instanceof List) {
            return "array";
        }
        if (v instanceof Map) {
            return "object";
        }
        return v.getClass().getSimpleName();
    }

    /** True for the Python truthiness of numbers, booleans and lists. */
    static boolean truthy(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        return v != null;
    }

    /** Reads a single element of {@code src} as an exact {@code int32}. */
    static int scalarInt(RowSource src, String where) throws IOException {
        long n = 1;
        for (long d : src.shape()) {
            n = Math.multiplyExact(n, d);
        }
        if (n != 1) {
            throw new GgufFormatException(where + " holds " + n + " elements, expected one");
        }
        if (src.format().isBlock()) {
            throw new GgufFormatException(where + " is " + src.format() + ", not an integer or float scalar");
        }
        ByteBuffer b = ByteBuffer.allocate((int) src.rowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, 1, b);
        float f = b.getFloat(b.remaining() - 4);
        if (Math.rint(f) != f || f < Integer.MIN_VALUE || f > Integer.MAX_VALUE) {
            throw new GgufFormatException(where + " value " + f + " is not an int32");
        }
        return (int) f;
    }

    // ------------------------------------------------------------------ the target model

    /**
     * The target the draft serves: a Hugging Face checkpoint read as the converter's {@code --target-model-dir}
     * (its configuration, with the draft's {@code vocab_size} for the padding), or a GGUF file with its
     * {@code tokenizer.*} metadata. A draft without a target fails in the target open.
     */
    static final class Target {
        final Path dir;
        final HfConfig config;
        final Map<String, Object> tokenizerKeys;
        CheckpointContainer container;

        Target(Path dir, HfConfig config, Map<String, Object> tokenizerKeys) {
            this.dir = dir;
            this.config = config;
            this.tokenizerKeys = tokenizerKeys;
        }
    }

    private static final class TargetHolder {
        Target value;
    }

    /**
     * The target of {@code ctx}, opened once per load; also opened directly by {@link HfLoadContext#target}, so a
     * target-less load fails there. The Hugging Face checkpoint container stays registered in
     * {@link HfLoadContext#resources}.
     */
    static Target target(HfLoadContext ctx) throws IOException {
        ctx.target();
        TargetHolder holder = ctx.state(TARGET_STATE, TargetHolder.class, TargetHolder::new);
        synchronized (holder) {
            if (holder.value == null) {
                holder.value = openTarget(ctx);
            }
            return holder.value;
        }
    }

    private static Target openTarget(HfLoadContext ctx) throws IOException {
        Path path = ctx.targetPath().get();
        if (ModelSources.detect(path) == ModelFormat.GGUF) {
            return new Target(path, null, tokenizerKeysOf(ctx.target()));
        }
        Path dir = (Files.isDirectory(path) ? path : path.getParent()).toAbsolutePath().normalize();
        HfConfig cfg = HfConfigLoader.load(dir,
                Collections.singletonMap("vocab_size", ctx.config().requireInt("vocab_size")));
        Target t = new Target(dir, cfg, Collections.emptyMap());
        ctx.resources().register(() -> {
            if (t.container != null) {
                t.container.close();
                t.container = null;
            }
        });
        return t;
    }

    private static Map<String, Object> tokenizerKeysOf(ModelSource target) {
        Map<String, Object> keys = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : target.getMetadata().entrySet()) {
            if (e.getKey().startsWith(GgufKeys.TOKENIZER_PREFIX)) {
                keys.put(e.getKey(), e.getValue());
            }
        }
        return Collections.unmodifiableMap(keys);
    }

    /**
     * The target's tokenizer files. Callers needing the target's tensors check {@link Target#config} first: a GGUF
     * target has no safetensors to read tensors from.
     */
    static HfTokenizerFiles targetTokenizer(HfLoadContext ctx) throws IOException {
        Target t = target(ctx);
        if (t.config == null) {
            throw new GgufFormatException(ctx.config().architecture() + " target '" + t.dir + "' is a GGUF file; its"
                    + " tokenizer comes from its tokenizer.* metadata");
        }
        return HfTokenizerFiles.locate(t.dir);
    }

    /** The target tensor {@code name}, read from the target's safetensors files (qwen.py:874-889). */
    static RowSource targetTensor(HfLoadContext ctx, String name) throws IOException {
        Target t = target(ctx);
        if (t.config == null) {
            throw new GgufFormatException(ctx.config().architecture() + " target '" + t.dir + "' is a GGUF file; the"
                    + " tensors patched into the draft cannot be read from it");
        }
        if (t.container == null) {
            t.container = CheckpointContainers.open(t.dir, t.dir, false);
        }
        return t.container.source(name);
    }

    /**
     * Whether the target ropes its text inputs with M-RoPE (qwen.py:914-921): the target's flattened
     * {@code rope_parameters} or {@code rope_scaling} names {@code mrope_section}.
     */
    static boolean targetUsesMrope(HfLoadContext ctx) throws IOException {
        Target t = target(ctx);
        if (t.config == null) {
            return false;
        }
        Object rope = t.config.raw().raw("rope_parameters");
        if (!(rope instanceof Map)) {
            rope = t.config.raw().raw("rope_scaling");
        }
        return rope instanceof Map && ((Map<?, ?>) rope).containsKey("mrope_section");
    }

    /**
     * {@code set_vocab}'s target dispatch (qwen.py:766-785, muse_glimmer.py:141-155, nemotron.py:59-77): the
     * target's own conversion writes the vocabulary over the target's tokenizer files; when the target belongs to
     * this draft family, the conversion would recurse into itself (LC's {@code target_cls is type(self)} check), so
     * {@code fallback} writes the draft family's inherited vocabulary. A GGUF target hands over its
     * {@code tokenizer.*} keys, the GGUF-side equivalent of the swap.
     */
    static Map<String, Object> targetVocab(HfLoadContext ctx, PreNameResolver pre,
                                           VocabFallback fallback) throws IOException {
        ctx.target();
        Target t = target(ctx);
        if (t.config == null) {
            if (t.tokenizerKeys.isEmpty()) {
                throw new GgufFormatException(ctx.config().architecture() + " target '" + t.dir
                        + "' holds no tokenizer.* metadata");
            }
            ctx.log("using the GGUF tokenizer of the target model " + t.dir);
            return new LinkedHashMap<>(t.tokenizerKeys);
        }
        if (DRAFT_NAMES.contains(t.config.architecture())) {
            return fallback.apply(ctx, pre);
        }
        HfArchEntry entry = HfArchRegistry.resolve(t.config, HfArchEntry.Kind.TEXT);
        Arch arch = entry.arch().resolve(t.config, HfTensorIndex.of(Collections.emptyMap()));
        HfLoadContext tctx = HfLoadContext.builder(t.config, entry, arch)
                .options(ctx.options())
                .tokenizerFiles(HfTokenizerFiles.locate(t.dir))
                .build();
        Map<String, Object> md = new LinkedHashMap<>(entry.vocab().build(tctx, pre));
        int maskId = maskTokenId(ctx.config());
        if (maskId >= 0) {
            md.put(GgufKeys.TOKENIZER_MASK_ID, maskId);
        }
        return Collections.unmodifiableMap(md);
    }

    /** The vocabulary writer of the fallback path of a draft family. */
    @FunctionalInterface
    interface VocabFallback {
        Map<String, Object> apply(HfLoadContext ctx, PreNameResolver pre) throws IOException;
    }

    // ------------------------------------------------------------------ mask_embedding

    /** One load's mask data. */
    static final class Mask {
        final RowSource embedding;
        final int tokenId;

        Mask(RowSource embedding, int tokenId) {
            this.embedding = embedding;
            this.tokenId = tokenId;
        }
    }

    private static final class MaskState {
        boolean resolved;
        Mask mask;
    }

    /**
     * The mask vector of the load, or null. {@code mask_embedding.pt} is a pickle, which NF never unpickles: a
     * {@code mask_embedding.safetensors} sibling provides {@code mask_token_id} and {@code embedding}; without one
     * {@code .pt} is offered to {@link CheckpointContainers#forFile} (the restricted pickle reader of PAR FX-1 can
     * accept it) and otherwise the pickle error is thrown. A checkpoint without {@code mask_embedding.pt} keeps the
     * unpatched embeddings.
     */
    static Mask maskOf(HfLoadContext ctx) throws IOException {
        MaskState st = ctx.state(MASK_STATE, MaskState.class, MaskState::new);
        synchronized (st) {
            if (!st.resolved) {
                Path dir = ctx.config().dir();
                Path pt = dir.resolve(MASK_PT);
                if (Files.isRegularFile(pt)) {
                    st.mask = readMask(ctx, dir, pt, dir.resolve(MASK_ST));
                }
                st.resolved = true;
            }
            return st.mask;
        }
    }

    /** qwen.py:864-871: the mask token id, the embedding shape and the vocabulary range. */
    private static Mask readMask(HfLoadContext ctx, Path dir, Path pt, Path st) throws IOException {
        HfConfig cfg = ctx.config();
        CheckpointContainer container;
        if (Files.isRegularFile(st)) {
            container = CheckpointContainers.open(st, dir, false);
        } else {
            CheckpointContainers.Factory factory = CheckpointContainers.forFile(pt);
            if (factory == null) {
                throw new IOException("Invalid checkpoint '" + pt
                        + "': PyTorch pickle checkpoints are not supported; use the model's safetensors files");
            }
            container = factory.open(pt, dir, false);
        }
        ctx.resources().register(container);
        HfTensorIndex index = HfTensorIndex.of(container);
        if (!index.has("mask_token_id") || !index.has("embedding")) {
            throw new IOException("Invalid checkpoint '" + st
                    + "': mask_embedding.pt holds the tensors 'mask_token_id' and 'embedding'");
        }
        int fileMaskId = scalarInt(index.get("mask_token_id"), "mask_embedding mask_token_id of '" + st + "'");
        int maskId = maskTokenId(cfg);
        if (maskId < 0 || fileMaskId != maskId) {
            throw new GgufFormatException(cfg.source() + ": mask_embedding.pt mask_token_id " + fileMaskId
                    + " does not match dflash_config " + maskId);
        }
        RowSource embedding = index.get("embedding");
        long[] shape = embedding.shape();
        int hidden = cfg.requireInt("hidden_size");
        if (shape.length != 1 || shape[0] != hidden) {
            throw new GgufFormatException(cfg.source() + ": mask_embedding.pt has an unexpected embedding shape "
                    + Arrays.toString(shape) + ", expected [" + hidden + "]");
        }
        int vocab = cfg.requireInt("vocab_size");
        if (fileMaskId < 0 || fileMaskId >= vocab) {
            throw new GgufFormatException(cfg.source() + ": mask_embedding.pt mask_token_id " + fileMaskId
                    + " is outside the vocabulary of " + vocab);
        }
        return new Mask(embedding, fileMaskId);
    }

    /**
     * {@code embeddings[mask_id] = mask["embedding"].to(embeddings.dtype)} (qwen.py:906-910): the masked token's
     * row of {@code embeddings} is replaced by the mask vector in the table's element format.
     */
    static RowSource patch(RowSource embeddings, Mask mask) throws IOException {
        long[] shape = embeddings.shape();
        if (shape.length < 2 || mask.tokenId >= shape[0]) {
            throw new GgufFormatException(embeddings.describe() + " has no row for mask token " + mask.tokenId);
        }
        GGMLType type = embeddings.format().ggmlType();
        if (type == null) {
            throw new GgufFormatException(embeddings.describe() + " holds " + embeddings.format()
                    + "; the mask row needs a ggml float element format to be converted into");
        }
        List<RowSource> parts = new ArrayList<>(3);
        if (mask.tokenId > 0) {
            parts.add(RowTransforms.sliceRows(embeddings, 0, mask.tokenId));
        }
        parts.add(ElementTransforms.convert(mask.embedding, type));
        if (mask.tokenId + 1 < shape[0]) {
            parts.add(RowTransforms.sliceRows(embeddings, mask.tokenId + 1, shape[0] - mask.tokenId - 1));
        }
        return parts.size() == 1 ? parts.get(0) : RowTransforms.concatRows(parts);
    }

    /** The values of {@code src} as F32 ({@code .float().cpu().numpy()}, nemotron.py:39-40). */
    static float[] rowFloats(RowSource src, String name) throws IOException {
        RowSource f = ElementTransforms.convert(src, GGMLType.F32);
        long total = f.rows() * f.rowBytes();
        if (total > Integer.MAX_VALUE - 8) {
            throw new GgufFormatException("tensor '" + name + "' is too large to read as float32");
        }
        ByteBuffer b = ByteBuffer.allocate((int) total).order(ByteOrder.LITTLE_ENDIAN);
        f.readRows(0, (int) f.rows(), b);
        float[] values = new float[Math.toIntExact(total / 4L)];
        b.flip();
        b.asFloatBuffer().get(values);
        return values;
    }
}
