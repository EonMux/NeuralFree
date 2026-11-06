package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.mm.ClipFamily;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * The Hugging Face {@code CLIPModel} and {@code SiglipModel} dual encoders as the two-tower clip file, a load-time
 * port of llama.cpp's legacy {@code convert_image_encoder_to_gguf.py} (tools/mtmd/legacy-models): the
 * {@code clip.has_text_encoder} / {@code clip.has_vision_encoder} towers, the {@code clip.text.*} and
 * {@code clip.vision.*} hyperparameters (the HF config class defaults where the configuration omits a field, as
 * {@code from_pretrained} fills them), the image mean and std of {@code preprocessor_config.json} (the processor
 * defaults without one), the converter's
 * {@link #ggufName tensor names} (the text and visual projections kept, the swapped {@code ffn_down} / {@code ffn_up}
 * names), and the text tower's vocabulary. A {@code SiglipModel} adds {@link ClipFamily#KEY} {@code siglip}, the text
 * {@code pad_token_id} its input pads with (the tokenizer's pad token when the vocabulary names one), and keeps
 * the converter's names for its text head ({@code t.head}), attention pooling head ({@code v.head.*}) and
 * {@code logit_scale} / {@code logit_bias}. The file opens as {@link com.eonmux.neuralfree.gguf.mm.ClipDualEncoder}.
 * Every method is a public static hook reference.
 */
public final class HfClipHooks {

    /** The Hugging Face architecture name of the CLIP dual encoder. */
    public static final String CLIP_MODEL = "CLIPModel";
    /** The Hugging Face architecture name of the SigLIP dual encoder. */
    public static final String SIGLIP_MODEL = "SiglipModel";
    /** Every dual-encoder architecture this entry converts. */
    public static final Set<String> DUAL_ENCODERS =
            Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(CLIP_MODEL, SIGLIP_MODEL)));

    static final String TEXT_TOKEN_EMBD = "text_model.embeddings.token_embedding.weight";
    static final String VISION_PATCH_EMBD = "vision_model.embeddings.patch_embedding.weight";
    static final String TEXT_PROJECTION = "text_projection.weight";
    static final String VISUAL_PROJECTION = "visual_projection.weight";
    static final String PREPROCESSOR_CONFIG = "preprocessor_config.json";
    private static final String BLOCK_COUNT = "num_hidden_layers";
    private static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};
    private static final float[] SIGLIP_MEAN_STD = {0.5f, 0.5f, 0.5f};

    /**
     * The converter's names as they stand: no tower is dropped, so the inherited {@code TextModel} filters (which drop
     * vision towers from language models) do not apply.
     */
    public static final HfNameFilter FILTER = (name, base) -> name;

    private HfClipHooks() {
    }

    /** One tower's hyperparameters with the HF config class defaults of its family. Immutable. */
    static final class Tower {
        final int hidden;
        final int ff;
        final int heads;
        final int layers;
        final double eps;
        final String act;
        final int positions;
        final int imageSize;
        final int patchSize;
        final int padId;

        private Tower(JsonView v, boolean siglip, boolean text) {
            this.hidden = v.optI32("hidden_size", siglip || !text ? 768 : 512);
            this.ff = v.optI32("intermediate_size", siglip || !text ? 3072 : 2048);
            this.heads = v.optI32("num_attention_heads", siglip || !text ? 12 : 8);
            this.layers = v.optI32(BLOCK_COUNT, 12);
            this.eps = v.optF64("layer_norm_eps", siglip ? 1e-6 : 1e-5);
            this.act = v.optStr("hidden_act", siglip ? "gelu_pytorch_tanh" : "quick_gelu");
            this.positions = v.optI32("max_position_embeddings", siglip ? 64 : 77);
            this.imageSize = v.optI32("image_size", 224);
            this.patchSize = v.optI32("patch_size", siglip ? 16 : 32);
            this.padId = v.optI32("pad_token_id", 1);
        }

        /** {@code text_config} of {@code cfg}. */
        static Tower text(HfConfig cfg) {
            return new Tower(cfg.text(), isSiglip(cfg), true);
        }

        /** {@code vision_config} of {@code cfg}. */
        static Tower vision(HfConfig cfg) {
            JsonView v = cfg.vision().orElse(JsonView.empty(cfg.source(), "vision_config"));
            return new Tower(v, isSiglip(cfg), false);
        }
    }

    static boolean isSiglip(HfConfig cfg) {
        return SIGLIP_MODEL.equals(cfg.architecture());
    }

    /**
     * The file's block count is the deeper tower's, so the layer filters of the load keep every block of both
     * towers (the configuration's flat {@code num_hidden_layers} is the text tower's).
     */
    public static HfConfig fixup(HfConfig cfg) {
        int blocks = Math.max(Tower.text(cfg).layers, Tower.vision(cfg).layers);
        return cfg.withFlat(Collections.singletonMap(BLOCK_COUNT, (long) blocks));
    }

    /**
     * The two-tower keys of the converter for the towers the checkpoint carries, the family of a
     * {@code SiglipModel}, and {@code clip.use_gelu} from the shared activation.
     *
     * @throws GgufFormatException when the towers use different activations, or one the clip file cannot name
     */
    public static void clipMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfTensorIndex index = ctx.index();
        boolean siglip = isSiglip(cfg);
        boolean hasText = index.has(TEXT_TOKEN_EMBD);
        boolean hasVision = index.has(VISION_PATCH_EMBD);
        kv.setTyped("clip.has_text_encoder", GGUFValueType.BOOL, hasText);
        kv.setTyped("clip.has_vision_encoder", GGUFValueType.BOOL, hasVision);
        kv.setTyped("clip.has_llava_projector", GGUFValueType.BOOL, false);
        if (siglip) {
            kv.setTyped(ClipFamily.KEY, GGUFValueType.STRING, ClipFamily.SIGLIP.keyName());
        }
        Tower text = Tower.text(cfg);
        Tower vision = Tower.vision(cfg);
        if (hasText) {
            kv.setTyped("clip.text.context_length", GGUFValueType.UINT32, text.positions);
            if (siglip) {
                kv.setTyped(GgufKeys.TOKENIZER_PAD_ID, GGUFValueType.UINT32, text.padId);
            }
            tower(kv, "clip.text.", text, siglip ? 0 : projectionDim(index, TEXT_PROJECTION));
        }
        if (hasVision) {
            kv.setTyped("clip.vision.image_size", GGUFValueType.UINT32, vision.imageSize);
            kv.setTyped("clip.vision.patch_size", GGUFValueType.UINT32, vision.patchSize);
            tower(kv, "clip.vision.", vision, siglip ? 0 : projectionDim(index, VISUAL_PROJECTION));
            JsonView pre = HfBertVariantHooks.readJsonView(cfg.sideFile(PREPROCESSOR_CONFIG));
            float[] mean = siglip ? SIGLIP_MEAN_STD : CLIP_MEAN;
            float[] std = siglip ? SIGLIP_MEAN_STD : CLIP_STD;
            kv.setTyped("clip.vision.image_mean", GGUFValueType.ARRAY, GGUFValueType.FLOAT32,
                    pre.has("image_mean") ? pre.list("image_mean") : mean);
            kv.setTyped("clip.vision.image_std", GGUFValueType.ARRAY, GGUFValueType.FLOAT32,
                    pre.has("image_std") ? pre.list("image_std") : std);
        }
        kv.setTyped("clip.use_gelu", GGUFValueType.BOOL, useGelu(cfg, hasText ? text : null,
                hasVision ? vision : null));
    }

    private static void tower(HfMetadataBuilder kv, String prefix, Tower t, int projectionDim) {
        kv.setTyped(prefix + "embedding_length", GGUFValueType.UINT32, t.hidden);
        kv.setTyped(prefix + "feed_forward_length", GGUFValueType.UINT32, t.ff);
        kv.setTyped(prefix + "projection_dim", GGUFValueType.UINT32, projectionDim);
        kv.setTyped(prefix + "attention.head_count", GGUFValueType.UINT32, t.heads);
        kv.setTyped(prefix + "attention.layer_norm_epsilon", GGUFValueType.FLOAT32, t.eps);
        kv.setTyped(prefix + "block_count", GGUFValueType.UINT32, t.layers);
    }

    /** The output rows of the projection {@code name}, 0 without one. */
    private static int projectionDim(HfTensorIndex index, String name) {
        if (!index.has(name)) {
            return 0;
        }
        long[] shape = index.get(name).shape();
        if (shape.length != 2 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
            throw new GgufFormatException("tensor '" + name + "' has the unusable shape " + Arrays.toString(shape));
        }
        return (int) shape[0];
    }

    /**
     * {@code clip.use_gelu}: true for the GELU activations ({@code gelu}, as the converter maps it, and
     * {@code gelu_pytorch_tanh} / {@code gelu_new}, the tanh form the clip file runs), false for
     * {@code quick_gelu}, the clip file's default.
     */
    private static boolean useGelu(HfConfig cfg, Tower text, Tower vision) {
        String act = vision != null ? vision.act : text.act;
        if (text != null && vision != null && !text.act.equals(vision.act)) {
            throw new GgufFormatException(cfg.source() + ": the text tower uses hidden_act '" + text.act
                    + "' and the vision tower '" + vision.act + "'; a two-tower clip file names one activation");
        }
        switch (act) {
            case "quick_gelu":
                return false;
            case "gelu":
            case "gelu_new":
            case "gelu_pytorch_tanh":
                return true;
            default:
                throw new GgufFormatException(cfg.source() + ": hidden_act '" + act + "' has no two-tower clip"
                        + " activation (quick_gelu, gelu, gelu_new, gelu_pytorch_tanh)");
        }
    }

    /**
     * The converter's {@code should_skip_tensor} and {@code get_tensor_name}: the {@code position_ids} buffers and
     * CLIP's {@code logit_scale} are skipped, every other tensor lands under {@link #ggufName}.
     */
    public static void clipTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (hfName.endsWith(".position_ids") || !isSiglip(ctx.config()) && "logit_scale".equals(hfName)) {
            return;
        }
        out.add(ggufName(hfName), src);
    }

    /** {@code get_tensor_name} of the legacy converter for a dual-encoder tensor. */
    static String ggufName(String name) {
        if (name.contains("projection")) {
            return name;
        }
        return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk")
                .replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_")
                .replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down")
                .replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post")
                .replace("layrnorm", "ln");
    }

    /**
     * The text tower's tokenizer: CLIP's byte-level BPE as the GPT-2 family converts it (the end-of-word suffix,
     * normalizer and post-processor included); SigLIP's Unigram SentencePiece model ({@code spiece.model}) as T5's
     * converts it. A SigLIP checkpoint without a SentencePiece model writes none, so text inputs refuse by name while
     * token ids and the tokenizer source option are still served.
     */
    public static Map<String, Object> clipVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTensorIndex index = ctx.index();
        if (!index.has(TEXT_TOKEN_EMBD)) {
            return Collections.emptyMap();
        }
        HfVocabWriters.Input in = HfVocabWriters.Input.of(ctx.config(), ctx.tokenizerFiles(),
                        PreTokenizerCanonicalizer::toSpec).withGgufArch(ctx.arch().ggufName())
                .withEmbeddingRows(HfHookSupport.embeddingRows(TEXT_TOKEN_EMBD, index.get(TEXT_TOKEN_EMBD)));
        if (!isSiglip(ctx.config())) {
            return HfVocabWriters.gpt2(in, pre);
        }
        boolean sentencePiece = in.files().hasGgufTokenizer() || in.files().sentencePieceModel() != null;
        return sentencePiece ? HfVocabWriters.t5Ugm(in, pre) : Collections.emptyMap();
    }
}
