package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Every on-disk {@code quant_method} name the loader recognizes, mapped to its {@link HfQuantMethod}: the method
 * names, the kernel-choice and engine aliases of vLLM and SGLang (V/model_executor/layers/quantization/__init__.py,
 * SG/srt/layers/quantization/__init__.py), the online-quantization names, and the rejected methods with their
 * reasons. Names are compared after {@link #normalize}.
 */
public final class HfQuantAliases {

    /** vLLM's bare online method, whose quantization_config selects the layers and their methods. */
    public static final String ONLINE = "online";

    private static final Map<String, HfQuantMethod> METHODS;
    private static final Map<String, String> REJECTED;

    static {
        Map<String, HfQuantMethod> m = new LinkedHashMap<>();
        m.put("fp8", HfQuantMethod.FP8);
        m.put("compressed-tensors", HfQuantMethod.COMPRESSED_TENSORS);
        m.put("modelopt", HfQuantMethod.MODELOPT);
        m.put("modelopt_fp4", HfQuantMethod.MODELOPT);
        m.put("modelopt_fp8", HfQuantMethod.MODELOPT);
        m.put("modelopt_mxfp8", HfQuantMethod.MODELOPT);
        m.put("modelopt_mixed", HfQuantMethod.MODELOPT);
        m.put("petit_nvfp4", HfQuantMethod.MODELOPT);
        m.put("gptq", HfQuantMethod.GPTQ);
        m.put("auto_gptq", HfQuantMethod.GPTQ);
        m.put("gptq_marlin", HfQuantMethod.GPTQ);
        m.put("moe_wna16", HfQuantMethod.GPTQ);
        m.put("awq", HfQuantMethod.AWQ);
        m.put("auto_awq", HfQuantMethod.AWQ);
        m.put("awq_marlin", HfQuantMethod.AWQ);
        m.put("auto-round", HfQuantMethod.AUTO_ROUND);
        m.put("inc", HfQuantMethod.AUTO_ROUND);
        m.put("humming", HfQuantMethod.AUTO_ROUND);
        m.put("bitsandbytes", HfQuantMethod.BITSANDBYTES);
        m.put("mxfp4", HfQuantMethod.MXFP4);
        m.put("gpt_oss_mxfp4", HfQuantMethod.MXFP4);
        m.put("mxfp8", HfQuantMethod.MXFP8);
        m.put("quark", HfQuantMethod.QUARK);
        m.put("quark_mxfp4", HfQuantMethod.QUARK);
        m.put("mlx_q4", HfQuantMethod.MLX);
        m.put("mlx_q8", HfQuantMethod.MLX);
        m.put("bitnet", HfQuantMethod.BITNET);
        m.put("fbgemm_fp8", HfQuantMethod.FBGEMM_FP8);
        m.put("w8a8_int8", HfQuantMethod.W8A8_INT8);
        m.put("auto-round-int8", HfQuantMethod.W8A8_INT8);
        m.put("w8a8_fp8", HfQuantMethod.W8A8_FP8);
        m.put("blockwise_int8", HfQuantMethod.BLOCKWISE_INT8);
        m.put("deepseek_v4_fp8", HfQuantMethod.DEEPSEEK_V4_FP8);
        m.put("hqq", HfQuantMethod.HQQ);
        m.put("fp_quant", HfQuantMethod.FP_QUANT);
        m.put("w4afp8", HfQuantMethod.W4AFP8);
        m.put("torchao", HfQuantMethod.TORCHAO);
        for (String online : new String[]{"experts_int8", "quark_int4fp8_moe", "mxfp_w4a8", "ptpc_fp8", ONLINE,
                "fp8_per_tensor", "fp8_per_block", "fp8_per_channel", "int8_per_channel_weight_only",
                "nvfp4_per_token", "nvfp4_online"}) {
            m.put(online, HfQuantMethod.ONLINE);
        }
        Map<String, String> r = new LinkedHashMap<>();
        r.put("modelslim", "Ascend ModelSlim checkpoints are not supported");
        for (String removed : new String[]{"aqlm", "eetq", "vptq", "spqr", "higgs", "quanto"}) {
            r.put(removed, "the method is deprecated or removed in vLLM and is not supported");
        }
        r.put("gguf", "GGUF checkpoints load from their .gguf file");
        for (String name : r.keySet()) {
            m.put(name, HfQuantMethod.REJECTED);
        }
        METHODS = Collections.unmodifiableMap(m);
        REJECTED = Collections.unmodifiableMap(r);
    }

    private HfQuantAliases() {
    }

    /** A classified name: the normalized name, its method and, for {@link HfQuantMethod#REJECTED}, the reason. */
    public static final class Resolution {
        private final String name;
        private final HfQuantMethod method;
        private final String reason;

        Resolution(String name, HfQuantMethod method, String reason) {
            this.name = name;
            this.method = method;
            this.reason = reason;
        }

        /** The normalized on-disk name. */
        public String name() {
            return name;
        }

        public HfQuantMethod method() {
            return method;
        }

        /** Why a rejected method is unsupported; null for every other method. */
        public String reason() {
            return reason;
        }

        @Override
        public String toString() {
            return name + " -> " + method + (reason == null ? "" : " (" + reason + ")");
        }
    }

    /** {@code raw} lower-cased, with {@code compressed_tensors} spelled {@code compressed-tensors} (vLLM). */
    public static String normalize(String raw) {
        Objects.requireNonNull(raw, "raw");
        return raw.toLowerCase(Locale.ROOT).replace("compressed_tensors", "compressed-tensors");
    }

    /**
     * Classifies quant_method {@code raw}.
     *
     * @throws GgufFormatException {@code "unsupported quant_method '<x>'"} when the name is in no table
     */
    public static Resolution resolve(String raw) {
        String name = normalize(raw);
        HfQuantMethod method = METHODS.get(name);
        if (method == null) {
            throw new GgufFormatException("unsupported quant_method '" + raw + "'");
        }
        return new Resolution(name, method, REJECTED.get(name));
    }

    /** Whether {@code raw} names a method that quantizes float weights at load. */
    public static boolean isOnline(String raw) {
        return METHODS.get(normalize(raw)) == HfQuantMethod.ONLINE;
    }

    /**
     * The online-quantization target the normalized method name quantizes at load with: the online shorthands of
     * vLLM's {@code --quantization} ({@code fp8_per_tensor}, {@code fp8_per_block}, {@code fp8_per_channel},
     * {@code ptpc_fp8}, {@code int8_per_channel_weight_only}, {@code nvfp4_per_token}, {@code nvfp4_online} and
     * {@code experts_int8}). Null for the names with no single target: bare {@code online} quantizes as its
     * quantization_config selects ({@link HfQuantConfig#onlineConfig()}), and {@code quark_int4fp8_moe} and
     * {@code mxfp_w4a8} have no online int4 target. {@code mxfp4} and {@code mxfp8} name checkpoint methods.
     */
    public static ModelSourceOptions.OnlineQuant onlineTarget(String normalized) {
        Objects.requireNonNull(normalized, "normalized");
        switch (normalized) {
            case "fp8_per_tensor":
                return ModelSourceOptions.OnlineQuant.FP8_PER_TENSOR;
            case "fp8_per_block":
                return ModelSourceOptions.OnlineQuant.FP8_PER_BLOCK;
            case "fp8_per_channel":
            case "ptpc_fp8":
                return ModelSourceOptions.OnlineQuant.FP8_PER_CHANNEL;
            case "int8_per_channel_weight_only":
                return ModelSourceOptions.OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY;
            case "nvfp4_per_token":
            case "nvfp4_online":
                return ModelSourceOptions.OnlineQuant.NVFP4_PER_TOKEN;
            case "experts_int8":
                return ModelSourceOptions.OnlineQuant.EXPERTS_INT8;
            default:
                return null;
        }
    }

    /** Every recognized normalized name with its method, in table order. */
    public static Map<String, HfQuantMethod> table() {
        return METHODS;
    }
}
