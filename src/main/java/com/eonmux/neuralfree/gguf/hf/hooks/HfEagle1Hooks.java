package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * The EAGLE-1 draft hooks: the HF-side load of the {@code Eagle*ForCausalLM} and
 * {@code EagleDeepSeekMTPModel} checkpoints vLLM serves with {@code sglang srt/models/llama_eagle.py} — the fused
 * {@code cnets} drafter of one decoder layer over the target's last hidden row, the layout the GGUF-side
 * {@code ArchTableDraft} row of {@link com.eonmux.neuralfree.gguf.arch.Arch#EAGLE} loads. The draft routes to
 * {@link com.eonmux.neuralfree.gguf.engine.DraftRoute#HEAD} beside its target, exactly as {@code eagle3} and
 * {@code gemma4-assistant} do ({@code LLMModelFactory.draftRoute}); the engine-side {@code Eagle1DraftModel} runs
 * the GGUF metadata this class writes.
 *
 * <p>The {@code EagleDeepSeekMTPModel} name is shared: a checkpoint that declares DeepSeek multi-head latent
 * attention ({@link HfEagleMlaHooks#isMla}) is vLLM's {@code deepseek_eagle.EagleDeepseekV3ForCausalLM} and converts
 * through {@link HfEagleMlaHooks}, while the llama- and mistral-shaped drafts keep the mapping below.</p>
 *
 * <p>The configuration keys and tensor names follow llama_eagle.py:42-115 ({@code cnets.py:427}) and
 * mistral_eagle.py:51-129: the drafter applies no norm around the {@code fc} fusion — the attention runs on the
 * {@code fc} output directly (the llama shape) or on its norm (mistral_eagle.py keeps {@code input_layernorm}),
 * the target hidden row is the residual, and the FFN keeps its {@code ffn_norm} — so the metadata's
 * {@code norm_before_residual} and {@code norm_before_fc} are always false. The target serves the converter's
 * {@code --target-model-dir} ({@link HfLoadContext#targetPath()}): its Hugging Face {@code config.json} is read
 * without being converted, and a GGUF target keeps its {@code tokenizer.*} metadata. The draft vocabulary is the
 * target's (there is no d2t), so the vocabulary dispatch reads the target like {@code HfEagle3Hooks.vocab} does; a
 * load without a target is refused there, as {@link HfDraftHooks} refuses every draft without one.</p>
 */
public final class HfEagle1Hooks {

    /** The fused fusion weight of the cnets layout (llama_eagle.py:73). */
    static final String FC_WEIGHT = "fc.weight";

    private static final String TARGET_STATE = HfEagle1Hooks.class.getName() + ".target";

    private HfEagle1Hooks() {
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code __init__}'s configuration fixes: a checkpoint's {@code architectures[0]} alias name
     * ({@code EagleModel}, {@code LlamaEagleForCausalLM}, ...) becomes the class name the converter's key and name
     * tables resolve ({@code EagleForCausalLM}); {@code num_hidden_layers} then names the drafter's one decoder
     * layer, so a missing count reads as 1. A DeepSeek-MTP-shaped checkpoint is checked for the MLA geometry its
     * mapping writes ({@link HfEagleMlaHooks#check}).
     *
     * @throws GgufFormatException when {@code num_hidden_layers} is present but is not a positive int32, or an MLA
     *                             draft is missing its head geometry
     */
    public static HfConfig fixup(HfConfig cfg) {
        if (!"EagleForCausalLM".equals(cfg.architecture())) {
            cfg = cfg.withArchitecture("EagleForCausalLM");
        }
        if (cfg.flat().containsKey("num_hidden_layers")) {
            int layers = cfg.requireInt("num_hidden_layers");
            if (layers != 1) {
                throw new GgufFormatException(cfg.source() + ": an EAGLE-1 drafter holds one decoder layer, got "
                        + layers);
            }
        } else {
            cfg = cfg.withFlat(Collections.singletonMap("num_hidden_layers", 1));
        }
        if (HfEagleMlaHooks.isMla(cfg)) {
            HfEagleMlaHooks.check(cfg);
        }
        return cfg;
    }

    /**
     * {@code filter_tensors}: the target's own weights are refused where the converter's draft classes read the
     * draft directory alone — a name the checkpoint carries beyond the drafter's tensors names a target file inside
     * a draft directory.
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        return HfNameFilter.INHERITED;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The metadata of the standard llama-family keys ({@link HfGenericMetadata#apply} over the draft's own
     * configuration, then {@link HfLlamaHooks#llamaKv} for the vocabulary size and the RoPE dimensions) followed by
     * the EAGLE-1 keys of llama_eagle.py:64-115, read from the draft's own {@code config.json} and its target:
     * {@code target_layers} is the target's last hidden layer — the single layer whose hidden rows the engine's
     * target feed records (llama_eagle.py:64-69, exactly one entry, which the {@code ArchTableDraft} hparams
     * require) — {@code target_hidden_size} is the target's hidden size, the fusion input the {@code fc} weight is
     * built for, and {@code norm_before_residual} and {@code norm_before_fc} are false because the drafter applies
     * no norm around the fc fusion (llama_eagle.py:42-61, cnets.py:427).
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when the target lacks its hidden size or layer count, or the draft configuration
     *                             misses a standard key
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        if (HfEagleMlaHooks.isMla(ctx.config())) {
            HfEagleMlaHooks.meta(ctx, kv, plan);
            return;
        }
        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        targetKeys(ctx, kv);
    }

    /**
     * The EAGLE-1 keys of llama_eagle.py:64-115, read from the draft's own {@code config.json} and its target:
     * {@code target_layers} is the target's last hidden layer — the single layer whose hidden rows the engine's
     * target feed records (exactly one entry, which the {@code ArchTableDraft} hparams require) —
     * {@code target_hidden_size} is the target's hidden size, the fusion input the {@code fc} weight is built for,
     * and {@code norm_before_residual} and {@code norm_before_fc} are false because the drafter applies no norm
     * around the fc fusion (llama_eagle.py:42-61, cnets.py:427).
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when the target lacks its hidden size or layer count
     */
    static void targetKeys(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        Target target = target(ctx);
        int last = target.numLayers - 1;
        ctx.log("EAGLE-1: target_layers = [" + last + "] (target model has " + target.numLayers + " layers)");
        kv.set(GgufKeys.TARGET_LAYERS, Collections.singletonList(last));
        ctx.log("EAGLE-1: target_hidden_size = " + target.hiddenSize + " (from the target model config)");
        kv.set(GgufKeys.TARGET_HIDDEN_SIZE, target.hiddenSize);
        kv.set(GgufKeys.NORM_BEFORE_RESIDUAL, false);
        kv.set(GgufKeys.NORM_BEFORE_FC, false);
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The tensors of the fused cnets layout (llama_eagle.py:42-115): {@code fc.weight} is kept as {@code fc.weight}
     * (the {@code FC} row of the tensor name table, as EAGLE-3 keeps its fusion weight), and the standard Llama
     * tensor names — the drafter's own embeddings and LM head when the checkpoint carries them, and its one decoder
     * layer without an input norm, whose optional {@code attn_norm.weight}, {@code attn_norm_2.weight} and
     * {@code output_norm.weight} map as EAGLE-3's do — go through {@link HfLlamaHooks#llamaTensor}. The block
     * indices of the checkpoint name their GGUF blocks one to one, so {@code layers.0.*} lands at {@code blk.0.*}.
     * A DeepSeek-MTP-shaped checkpoint maps through {@link HfEagleMlaHooks}.
     *
     * @throws IOException         when an expert tensor or the target cannot be read
     * @throws GgufFormatException when a tensor name has no mapping
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (HfEagleMlaHooks.isMla(ctx.config())) {
            HfEagleMlaHooks.tensors(ctx, hfName, src, out);
            return;
        }
        if (hfName.equals(FC_WEIGHT)) {
            out.add(ctx.names().ggufName("FC", -1, ".weight"), src);
            return;
        }
        HfLlamaHooks.llamaTensor(ctx, hfName, hfName, src, out);
    }

    /**
     * {@code prepare_tensors}' completeness checks of a DeepSeek-MTP-shaped draft: every routed-expert set that was
     * started must have been stacked ({@link HfEagleMlaHooks#extra}). A llama-shaped draft generates no extra
     * tensors.
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        if (HfEagleMlaHooks.isMla(ctx.config())) {
            HfEagleMlaHooks.extra(ctx, out);
        }
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * The target's tokenizer ({@code set_vocab}'s target dispatch: the draft shares the target's vocabulary, there
     * is no d2t and no draft vocabulary of its own). A GGUF target hands over its {@code tokenizer.*} metadata; a
     * Hugging Face target is read from its tokenizer files with the draft's vocabulary size.
     *
     * @throws IOException         when the target or its tokenizer files cannot be read
     * @throws GgufFormatException when the target is a GGUF file that holds no tokenizer
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Target target = target(ctx);
        if (target.config == null) {
            if (target.tokenizerKeys.isEmpty()) {
                throw new GgufFormatException(ctx.config().architecture() + " target '" + target.path
                        + "' holds no tokenizer.* metadata");
            }
            ctx.log("EAGLE-1: using the GGUF tokenizer of the target model " + target.path);
            return target.tokenizerKeys;
        }
        ctx.log("EAGLE-1: using the tokenizer of the target model " + target.path);
        return HfLlamaHooks.llamaVocab(ctx, pre, HfTokenizerFiles.locate(target.path),
                target.config.withFlat(Collections.singletonMap("vocab_size", ctx.config().flat()
                        .getOrDefault("vocab_size", target.config.flat().get("vocab_size")))));
    }

    // ------------------------------------------------------------------ target

    /**
     * What the draft needs from its target, read once per load: the hidden size and the layer count the metadata
     * keys take, and the tokenizer source the vocabulary dispatch reads. The target is the converter's
     * {@code --target-model-dir}: a Hugging Face checkpoint read as its {@code config.json} alone, or a GGUF file
     * read through its metadata.
     *
     * @throws IOException         when the target cannot be opened or its configuration read
     * @throws GgufFormatException when the target lacks its hidden size, layer count or vocabulary size
     */
    static Target target(HfLoadContext ctx) throws IOException {
        return ctx.state(TARGET_STATE, TargetHolder.class, TargetHolder::new).get(ctx);
    }

    /** The target facts of one load. Immutable. */
    static final class Target {
        /** The target directory (Hugging Face) or file (GGUF). */
        final Path path;
        /** The target's {@code config.json}, null for a GGUF target. */
        final HfConfig config;
        /** A GGUF target's {@code tokenizer.*} metadata (unmodifiable), empty for a Hugging Face target. */
        final Map<String, Object> tokenizerKeys;
        final int hiddenSize;
        final int numLayers;

        Target(Path path, HfConfig config, Map<String, Object> tokenizerKeys, int hiddenSize, int numLayers) {
            this.path = path;
            this.config = config;
            this.tokenizerKeys = tokenizerKeys;
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
        }
    }

    /** Loads the {@link Target} on first use; shared by every context of one load. */
    static final class TargetHolder {
        private Target target;

        synchronized Target get(HfLoadContext ctx) throws IOException {
            if (target == null) {
                target = load(ctx);
            }
            return target;
        }

        private static Target load(HfLoadContext ctx) throws IOException {
            if (ctx.targetPath().isEmpty()) {
                throw new GgufFormatException(ctx.config().architecture()
                        + " is a draft model; open it as a draft of its target model");
            }
            // the reference reads the target's config.json (llama.py:64-74); its weights are never
            // converted, so the target is not opened as a model source here
            Path path = ctx.targetPath().get().toAbsolutePath().normalize();
            String arch = ctx.config().architecture();
            if (ModelSources.detect(path) == ModelFormat.GGUF) {
                try (ModelSource target = ModelSources.open(path.toString())) {
                    Map<String, Object> metadata = target.getMetadata();
                    Map<String, Object> tokenizerKeys = ggufTokenizerKeys(ctx, path, metadata);
                    // the target GGUF carries its own architecture; the keys are prefixed with it
                    String targetArch = String.valueOf(metadata.getOrDefault(GgufKeys.GENERAL_ARCHITECTURE, arch));
                    return new Target(path, null, tokenizerKeys,
                            ggufInt(path, metadata, targetArch, GgufKeys.EMBEDDING_LENGTH),
                            ggufInt(path, metadata, targetArch, GgufKeys.BLOCK_COUNT));
                }
            }
            Path dir = Files.isDirectory(path) ? path : path.getParent();
            HfConfig cfg = HfConfigLoader.load(dir.toAbsolutePath().normalize(),
                    Collections.singletonMap("vocab_size", ctx.config().requireInt("vocab_size")));
            return new Target(dir.toAbsolutePath().normalize(), cfg, Collections.emptyMap(),
                    cfg.requireInt("hidden_size"), cfg.requireInt("num_hidden_layers"));
        }

        /** A GGUF target's {@code tokenizer.*} metadata, as the draft's vocabulary inherits it. */
        private static Map<String, Object> ggufTokenizerKeys(HfLoadContext ctx, Path path,
                                                             Map<String, Object> metadata) throws IOException {
            Map<String, Object> keys = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                if (e.getKey().startsWith(GgufKeys.TOKENIZER_PREFIX)) {
                    keys.put(e.getKey(), e.getValue());
                }
            }
            if (keys.isEmpty()) {
                throw new GgufFormatException(ctx.config().architecture() + " target '" + path
                        + "' holds no tokenizer.* metadata");
            }
            return Collections.unmodifiableMap(keys);
        }

        /** A GGUF target's {@code <arch>.<key>} hparam as an exact int32. */
        private static int ggufInt(Path path, Map<String, Object> metadata, String arch, String keyFmt)
                throws IOException {
            Object v = metadata.get(String.format(Locale.ROOT, keyFmt, arch));
            if (!(v instanceof Number)) {
                throw new GgufFormatException("EAGLE-1 target '" + path + "' holds no " + keyFmt.replace("%s", arch));
            }
            double d = ((Number) v).doubleValue();
            if (d != Math.rint(d) || d < 1 || d > HfConfig.MAX_LAYERS) {
                throw new GgufFormatException("EAGLE-1 target '" + path + "' " + keyFmt.replace("%s", arch)
                        + " value " + d + " is not a positive layer or width count");
            }
            return (int) d;
        }
    }
}
