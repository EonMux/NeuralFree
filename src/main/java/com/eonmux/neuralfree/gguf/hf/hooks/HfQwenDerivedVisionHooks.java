package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.List;

/**
 * The projector classes whose vision tower is a Qwen-VL ViT: the subclasses llama.cpp adds over
 * {@code Qwen3VLVisionModel} (LC/conversion/qwen3vl.py:17-146) for Clef, H2O-Lightning, the PPLX and vLLM decision
 * models, Qwen4Exp and MiniCPM-OCR's sequence classifier, over {@code Qwen2VLVisionModel} (LC/conversion/qwenvl.py:45-141)
 * for EXAONE 4.5, plus the Ling tower of {@code BailingMoeV3VLVisionModel} (LC/conversion/bailingmoe3.py:237-304).
 * Each class writes the base {@code MmprojModel} keys and then its own projector type, activation, pixel bounds and
 * merge settings; Bailing-MoE renames its top-level projector to {@code mm.0} / {@code mm.2} and its merger norm to
 * the projector's input norm.
 */
public final class HfQwenDerivedVisionHooks {

    /** {@code VisionProjectorType.EXAONE4_5}. */
    public static final String EXAONE4_5 = "exaone4_5";
    /** {@code VisionProjectorType.LING3VL}. */
    public static final String LING3VL = "ling3vl";

    /** The default RMS epsilon of the EXAONE and Ling towers (exaone.py:300, bailingmoe3.py:276). */
    private static final double DEFAULT_EPS = 1e-6;

    /** The image pixel bounds the vLLM Decision 3.0 runtime reads (vllm_d3.py:58-61). */
    private static final int VLLM_D3_MIN_PIXELS = 65_536;
    private static final int VLLM_D3_MAX_PIXELS = 1_638_400;

    private HfQwenDerivedVisionHooks() {
    }

    // ------------------------------------------------------------------ pass-through Qwen3-VL towers

    /** The tower of Clef, H2O-Lightning, Pplx-Decider, vLLM-D3 and Qwen4Exp: the {@code qwen3vl_merger} keys. */
    public static void qwen3vlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenVlVisionHooks.qwen3vlMeta(ctx, kv, plan);
    }

    /**
     * {@code PplxDeciderVisionModel.set_gguf_parameters} (pplx_decider.py:96-106): the Qwen3-VL keys and the image
     * pixel bounds of the processor, or the fixed bounds of the Decision 3.0 class.
     */
    public static void decisionMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen3vlMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        if ("VllmD3Model".equals(cfg.architecture())) {
            kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, VLLM_D3_MIN_PIXELS);
            kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, VLLM_D3_MAX_PIXELS);
            return;
        }
        JsonView size = HfMmprojHooks.object(cfg, HfMmprojHooks.preprocessor(cfg), "size");
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, HfMmprojHooks.intOf(cfg, size, "size", "shortest_edge"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, HfMmprojHooks.intOf(cfg, size, "size", "longest_edge"));
    }

    /**
     * {@code MiniCPMOCRSeqClsVisionModel.set_gguf_parameters} (qwen3vl.py:148-158): the Qwen3-VL keys and the image
     * budget of its processor, wider than the Qwen-VL default.
     */
    public static void seqClsMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen3vlMeta(ctx, kv, plan);
        JsonView size = HfMmprojHooks.object(ctx.config(), HfMmprojHooks.preprocessor(ctx.config()), "size");
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS,
                HfMmprojHooks.intOf(ctx.config(), size, "size", "shortest_edge"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS,
                HfMmprojHooks.intOf(ctx.config(), size, "size", "longest_edge"));
    }

    // ------------------------------------------------------------------ EXAONE 4.5

    /** {@code Exaone4_5VisionModel.filter_tensors} (exaone.py:280-285): {@code model.visual.} strips, {@code visual.} keeps. */
    public static final HfNameFilter EXAONE_FILTER = (name, base) -> {
        String renamed = name.startsWith("model.visual.")
                ? "visual." + name.substring("model.visual.".length()) : name;
        return renamed.startsWith("visual.") ? base.apply(renamed) : null;
    };

    /**
     * {@code Exaone4_5VisionModel.set_gguf_parameters} (exaone.py:287-309): the base vision keys of a Qwen2-VL styled
     * config, the {@code exaone4_5} projector, SiLU, the preprocessor pixel bounds, the optional KV head count, the
     * vision (or text) RMS epsilon, the optional window and the window-attention period.
     *
     * @throws GgufFormatException naming the missing or unsupported key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void exaoneMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, exaoneImageSize(cfg), HfMmprojHooks.visionInt(cfg, "patch_size"),
                exaoneHidden(cfg), exaoneFfn(cfg), exaoneHeads(cfg), exaoneBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, EXAONE4_5);
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, HfMmprojHooks.preInt(cfg, "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, HfMmprojHooks.preInt(cfg, "max_pixels"));
        HfMmprojHooks.visionIntOpt(cfg, "num_key_value_heads")
                .ifPresent(v -> kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT_KV, v));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, exaoneEps(cfg));
        Object window = HfMmprojHooks.visionFind(cfg, "window_size").orElse(null);
        if (window != null) {
            kv.set(GgufKeys.CLIP_VISION_WINDOW_SIZE, HfMmprojHooks.visionInt(cfg, "window_size"));
        }
        List<Integer> full = HfMmprojHooks.intList(cfg, vision(cfg), "fullatt_block_indexes");
        if (!full.isEmpty()) {
            int period = full.get(0) + 1;
            for (int i = 1; i < full.size(); i++) {
                if (full.get(i) - full.get(i - 1) != period) {
                    throw new GgufFormatException(cfg.source() + ": Invalid EXAONE4.5 fullatt_block_indexes: "
                            + full + " (exaone.py:305-308)");
                }
            }
            kv.set(GgufKeys.CLIP_VISION_N_WA_PATTERN, period);
        }
    }

    /**
     * {@code Exaone4_5VisionModel.modify_tensors} (exaone.py:311-317): the fused {@code qkv} maps directly, everything
     * else runs the Qwen2-VL path.
     */
    public static void exaoneTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains(".qkv.")) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        }
        HfQwenVlVisionHooks.qwen2vlTensors(ctx, hfName, src, out);
    }

    private static JsonView vision(HfConfig cfg) {
        return cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no 'vision_config'"));
    }

    /** The tower width: {@code embed_dim} when the config renames it, else the base aliases. */
    private static int exaoneHidden(HfConfig cfg) {
        return HfMmprojHooks.visionFind(cfg, "embed_dim").isPresent()
                ? HfMmprojHooks.visionInt(cfg, "embed_dim")
                : HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
    }

    /** The FFN width: the config's {@code hidden_size} when it carries {@code embed_dim}, else the base aliases. */
    private static int exaoneFfn(HfConfig cfg) {
        return HfMmprojHooks.visionFind(cfg, "embed_dim").isPresent()
                ? HfMmprojHooks.visionInt(cfg, "hidden_size")
                : HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS);
    }

    private static int exaoneHeads(HfConfig cfg) {
        return HfMmprojHooks.visionInt(cfg, "num_attention_heads", "num_heads", "heads");
    }

    private static int exaoneBlocks(HfConfig cfg) {
        return HfMmprojHooks.blocksOf(cfg, vision(cfg), "vision_config");
    }

    private static int exaoneImageSize(HfConfig cfg) {
        return HfMmprojHooks.visionInt(cfg, "image_size");
    }

    /** {@code hparams.get("rms_norm_eps", global_config.get("rms_norm_eps", 1e-6))} (exaone.py:300). */
    private static double exaoneEps(HfConfig cfg) {
        double eps = HfMmprojHooks.visionF64Opt(cfg, "rms_norm_eps").orElse(DEFAULT_EPS);
        if (eps == DEFAULT_EPS && !HfMmprojHooks.visionFind(cfg, "rms_norm_eps").isPresent()) {
            Object root = cfg.raw().raw("rms_norm_eps");
            if (root instanceof Number) {
                return ((Number) root).doubleValue();
            }
        }
        return eps;
    }

    // ------------------------------------------------------------------ Bailing-MoE (Ling-3.0-VL)

    /** The top-level projector MLP keeps {@code mm.<index>}; otherwise the {@code Qwen3VLVisionModel} filter applies. */
    public static final HfNameFilter BAILING_FILTER = (name, base) -> {
        if (name.startsWith("lm_head.")) {
            return null;
        }
        if (name.startsWith("linear_proj")) {
            String[] parts = name.split("\\.");
            if (parts.length != 3) {
                throw new GgufFormatException("Unexpected linear_proj tensor: " + name + " (bailingmoe3.py:287)");
            }
            return base.apply("mm." + parts[1] + "." + parts[2]);
        }
        String renamed = name.startsWith("model.visual.")
                ? "visual." + name.substring("model.visual.".length()) : name;
        return renamed.startsWith("visual.") ? HfQwenVlVisionHooks.QWEN3VL_FILTER.filter(renamed, base) : null;
    };

    /**
     * {@code BailingMoeV3VLVisionModel.set_gguf_parameters} (bailingmoe3.py:255-276): the base vision keys of the
     * Qwen3-VL tower, the {@code ling3vl} projector, GELU, the spatial merge and the text model's RMS epsilon.
     *
     * @throws GgufFormatException when the config lacks the required Ling keys
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void bailingMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        if (!Boolean.TRUE.equals(vision.raw("disable_merger_proj"))) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3VL requires disable_merger_proj=true"
                    + " (bailingmoe3.py:246)");
        }
        if (!vision.has("out_hidden_size")) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3VL vision config requires out_hidden_size"
                    + " (bailingmoe3.py:251)");
        }
        HfMmprojHooks.visionMeta(ctx, kv, HfQwenVlVisionHooks.imageSize(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, LING3VL);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        HfMmprojHooks.visionIntOpt(cfg, "spatial_merge_size")
                .ifPresent(v -> kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, v));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                cfg.text().has("rms_norm_eps") ? cfg.text().f64("rms_norm_eps") : DEFAULT_EPS);
    }

    /**
     * {@code BailingMoeV3VLVisionModel.modify_tensors} (bailingmoe3.py:294-304): the top-level projector and the
     * merger norm take their own names, everything else runs the Qwen3-VL path.
     */
    public static void bailingTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("mm.0.") || hfName.startsWith("mm.2.")) {
            out.add(hfName, src);
            return;
        }
        if ("visual.merger.norm.weight".equals(hfName) || "visual.merger.norm.bias".equals(hfName)) {
            String suffix = hfName.substring(hfName.lastIndexOf('.'));
            out.add("mm.input_norm" + suffix, src);
            return;
        }
        HfQwenVlVisionHooks.qwen3vlTensors(ctx, hfName, src, out);
    }
}
