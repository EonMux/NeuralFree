package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.Assembly;
import com.eonmux.neuralfree.gguf.hf.config.QuantRule.ModuleMatcher;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.firstOf;
import static com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.nullableObject;
import static com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.object;
import static com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.optObject;
import static com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig.strings;

/**
 * The quantization method of a checkpoint, after alias normalization ({@link HfQuantAliases}) and the vLLM method
 * overrides. The constant name is the normalized method name recorded in metadata. Each method reads its own
 * configuration dialect into the ignore list and rules of an {@link HfQuantConfig}, following the vLLM / SGLang
 * config class cited at its reader.
 */
public enum HfQuantMethod {
    /** Not quantized. */
    NONE,
    /**
     * A method that stores float weights and quantizes at load in vLLM/SGLang: the vLLM online shorthands
     * ({@link HfQuantConfig#onlineTarget()}) and bare {@code online} with its layer selection
     * ({@link HfQuantConfig#onlineConfig()}) quantize at load, the other names load their weights as stored.
     */
    ONLINE,
    /** {@code fp8}: FP8 E4M3 per-tensor or block weights. */
    FP8,
    /** {@code compressed-tensors} (llm-compressor). */
    COMPRESSED_TENSORS,
    /** NVIDIA ModelOpt ({@code modelopt*}, {@code hf_quant_config.json}). */
    MODELOPT,
    /** GPTQ / GPTQModel. */
    GPTQ,
    /** AutoAWQ. */
    AWQ,
    /** Intel AutoRound / INC formats other than its GPTQ and AWQ packings. */
    AUTO_ROUND,
    /** bitsandbytes 4-bit or 8-bit. */
    BITSANDBYTES,
    /** {@code mxfp4} (gpt-oss {@code _blocks}/{@code _scales}). */
    MXFP4,
    /** {@code mxfp8}. */
    MXFP8,
    /** AMD Quark. */
    QUARK,
    /** MLX affine quantization. */
    MLX,
    /** BitNet b1.58. */
    BITNET,
    /** {@code fbgemm_fp8}: per-channel FP8 weights, dynamic per-token activations. */
    FBGEMM_FP8,
    /** SGLang {@code w8a8_int8}. */
    W8A8_INT8,
    /** SGLang {@code w8a8_fp8}. */
    W8A8_FP8,
    /** SGLang {@code blockwise_int8}. */
    BLOCKWISE_INT8,
    /** DeepSeek-V4 FP8 linears with MXFP4 or FP8 experts. */
    DEEPSEEK_V4_FP8,
    /** HQQ. */
    HQQ,
    /** fp_quant (QuTLASS MXFP4/NVFP4 with Hadamard rotations). */
    FP_QUANT,
    /** SGLang {@code w4afp8}: int4 expert weights with FP8 activations, FP8 block linears. */
    W4AFP8,
    /**
     * torchao, whose checkpoints store torchao tensor subclasses (safetensors tensor-data descriptors, or
     * {@code _rebuild_wrapper_subclass} wrappers in a PyTorch pickle). The configuration is read into the group
     * size, weight scale format and activation quantization of the layouts the containers decode.
     */
    TORCHAO,
    /** A known method that is not supported; {@link HfQuantAliases.Resolution#reason()} says why. */
    REJECTED;

    private static final Logger LOG = LogManager.getLogger(HfQuantMethod.class);
    private static final Set<String> MLX_MODES = Set.of("affine");
    private static final String EXPERTS = "(?:.*\\.)?experts\\.\\d+\\.";
    /** A fused or per-expert MoE module: {@code ...experts} itself or any module below it. */
    private static final String ANY_EXPERTS = "(?:.*\\.)?experts(?:\\.|$)";
    private static final Pattern REGEX_CHARS = Pattern.compile("[*+?^$()\\[\\]{}|\\\\]");

    /** Reads the method's configuration {@code a.q} into the ignore list and rules of {@code a}. */
    void read(HfConfig cfg, Assembly a) {
        switch (this) {
            case NONE: case ONLINE: case REJECTED:
                break;
            case FP8: fp8(a); break;
            case DEEPSEEK_V4_FP8: deepseekV4(cfg, a); break;
            case COMPRESSED_TENSORS: compressedTensors(a); break;
            case MODELOPT: modelOpt(a); break;
            case GPTQ: gptq(a, a.name.equals("moe_wna16")); break;
            case AWQ: awq(a); break;
            case AUTO_ROUND: autoRound(a); break;
            case BITSANDBYTES: bitsandbytes(a); break;
            case MXFP4: mxfp4(a); break;
            case MXFP8: mxfp8(a); break;
            case QUARK: quark(a); break;
            case MLX: mlx(a); break;
            case BITNET: bitnet(a); break;
            case FBGEMM_FP8:
                a.ignore(a.q.get("modules_to_not_convert"), "modules_to_not_convert", ModuleMatcher::exactOrRegex);
                a.all(QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL)
                        .actScheme(ActQuantScheme.FP8_TOKEN).build());
                break;
            case W8A8_INT8:
                a.ignore(a.q.get("ignore"), "ignore", ModuleMatcher::exactOrRegex);
                a.all(QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL)
                        .actScheme(ActQuantScheme.INT8_TOKEN).build());
                break;
            case W8A8_FP8:
                a.ignore(a.q.get("ignore"), "ignore", ModuleMatcher::exactOrRegex);
                a.all(QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL)
                        .actScheme(ActQuantScheme.FP8_TOKEN).build());
                break;
            case BLOCKWISE_INT8: blockwiseInt8(a); break;
            case HQQ: hqq(a); break;
            case FP_QUANT: fpQuant(a); break;
            case W4AFP8: w4afp8(a); break;
            case TORCHAO: torchao(a); break;
            default: throw new IllegalStateException("unhandled quantization method " + this);
        }
    }

    /** V fp8.py Fp8Config.from_config. */
    private static void fp8(Assembly a) {
        fp8Ignore(a);
        String scheme = activationScheme(a);
        int[] block = blockSize(a, null);
        if (block != null && !scheme.equals("dynamic")) {
            throw new GgufFormatException(a.where + ": block FP8 (weight_block_size) requires activation_scheme "
                    + "'dynamic', got '" + scheme + "'");
        }
        a.all(block != null ? blockFp8(block)
                : QuantArgs.builder(QuantFormat.FP8).actScheme(scheme.equals("static") ? ActQuantScheme.FP8_STATIC
                : ActQuantScheme.FP8_TOKEN).build());
    }

    private static QuantArgs blockFp8(int[] block) {
        QuantArgs.Builder b = QuantArgs.builder(QuantFormat.FP8).block(block[0], block[1]);
        return (block[1] == 128 ? b.actScheme(ActQuantScheme.FP8_GROUP128)
                : b.actUnemulated("dynamic FP8 activations in groups of " + block[1])).build();
    }

    /**
     * V fp8.py Fp8Config.from_config: {@code ignored_layers}, or {@code modules_to_not_convert} when that is absent
     * or empty, matched exactly or as {@code re:} regexes.
     */
    private static void fp8Ignore(Assembly a) {
        Object ignored = a.q.get("ignored_layers");
        boolean empty = ignored == null || (ignored instanceof List && ((List<?>) ignored).isEmpty());
        a.ignore(empty ? a.q.get("modules_to_not_convert") : ignored,
                empty ? "modules_to_not_convert" : "ignored_layers", ModuleMatcher::exactOrRegex);
    }

    /**
     * V deepseek_v4/quant_config.py: block FP8 linears; with {@code expert_dtype} fp4, MXFP4 experts (NVFP4 when
     * {@code moe_quant_algo} is NVFP4); with fp8, block FP8 experts.
     */
    private static void deepseekV4(HfConfig cfg, Assembly a) {
        fp8Ignore(a);
        int[] block = blockSize(a, new int[]{128, 128});
        String moeAlgo = QuantArgs.strOr(a.q, "moe_quant_algo", "", a.where).toUpperCase(Locale.ROOT);
        String expertDtype = cfg.findStringOr("fp4", "expert_dtype");
        if (expertDtype.equals("fp4")) {
            a.rules.add(QuantRule.of(ModuleMatcher.regex(EXPERTS), moeAlgo.equals("NVFP4") ? modelOptArgs("NVFP4", 0)
                    : QuantArgs.builder(QuantFormat.MXFP4_DSV4).build()));
        } else if (!expertDtype.equals("fp8")) {
            throw new GgufFormatException(cfg.source() + ": unsupported DeepSeek-V4 expert_dtype '" + expertDtype
                    + "'; expected fp4 or fp8");
        }
        // DeepSeek-V4.1 Engram tables are E4M3 rows with one E8M0 scale per 32 elements, not block-FP8 linears:
        // deepseek41.py:181-193 dequantizes them itself and keeps the stored tensors out of the rewrite
        a.rules.add(QuantRule.exclude(ModuleMatcher.regex("(?:.*\\.)?layers\\.\\d+\\.engram\\.embed$")));
        a.all(blockFp8(block));
    }

    /** compressed-tensors: config groups in order, name and regex targets before class targets. */
    private static void compressedTensors(Assembly a) {
        Object sparsity = a.q.get("sparsity_config");
        if (sparsity instanceof Map && ((Map<?, ?>) sparsity).get("targets") instanceof List
                && !((List<?>) ((Map<?, ?>) sparsity).get("targets")).isEmpty()) {
            throw new GgufFormatException("unsupported quant_method 'compressed-tensors': " + a.where
                    + ".sparsity_config declares sparse layers; compressed-tensors sparsity is not supported");
        }
        a.ignore(a.q.get("ignore"), "ignore", ModuleMatcher::exactOrRegex);
        if (a.q.get("transform_config") != null) {
            a.transforms = HfTransformConfig.parse(a.q.get("transform_config"), a.where + ".transform_config");
        }
        if (a.q.get("kv_cache_scheme") != null) {
            LOG.info("{}.kv_cache_scheme is not applied; the KV cache keeps its runtime type", a.where);
        }
        String format = QuantArgs.strOr(a.q, "format", null, a.where);
        Map<String, Object> groups = optObject(a.q.get("config_groups"), a.where + ".config_groups");
        List<QuantRule> classRules = new ArrayList<>();
        for (Map.Entry<String, Object> g : groups.entrySet()) {
            String where = a.where + ".config_groups." + g.getKey();
            Map<String, Object> group = optObject(g.getValue(), where);
            List<String> targets = strings(group.get("targets"), where + ".targets");
            if (targets.stream().allMatch(t -> t.endsWith("Attention")) || group.get("weights") == null) {
                continue;
            }
            QuantArgs args = QuantArgs.compressedTensors(optObject(group.get("weights"), where + ".weights"),
                    nullableObject(group.get("input_activations"), where + ".input_activations"),
                    nullableObject(group.get("output_activations"), where + ".output_activations"),
                    QuantArgs.strOr(group, "format", format, where), where);
            for (String t : targets) {
                ModuleMatcher m = ModuleMatcher.target(t);
                (m.kind() == ModuleMatcher.Kind.CLASS ? classRules : a.rules).add(QuantRule.of(m, args));
            }
        }
        a.rules.addAll(classRules);
    }

    /** V modelopt.py: legacy {@code quantization} block or config.json form; MIXED_PRECISION per layer. */
    private static void modelOpt(Assembly a) {
        Object legacy = a.q.get("quantization");
        Map<String, Object> inner = legacy instanceof Map ? HfConfig.asObject(legacy) : a.q;
        String algo = QuantArgs.strOr(inner, "quant_algo", null, a.where);
        if (algo == null || algo.isEmpty()) {
            throw new GgufFormatException(a.where + ": ModelOpt configuration has no quant_algo");
        }
        a.algo = algo.toUpperCase(Locale.ROOT);
        modelOptExclude(a, inner == a.q ? firstOf(a.q, "ignore", "exclude_modules") : inner.get("exclude_modules"),
                "exclude_modules");
        if (a.algo.equals("MIXED_PRECISION")) {
            Object layers = inner.get("quantized_layers") != null ? inner.get("quantized_layers")
                    : a.q.get("quantized_layers");
            modelOptMixed(a, optObject(layers, a.where + ".quantized_layers"), groupSize(inner, a.where));
        } else {
            a.all(modelOptArgs(a.algo, a.algo.endsWith("NVFP4") ? groupSize(inner, a.where) : 0));
        }
    }

    /** V modelopt.py is_layer_excluded: each entry excludes the modules containing it or matching it as a wildcard. */
    private static void modelOptExclude(Assembly a, Object entries, String key) {
        if (entries == null) {
            return;
        }
        for (String e : strings(entries, a.where + "." + key)) {
            a.ignoreList.add(e);
            a.ignore.add(ModuleMatcher.substring(e));
            a.ignore.add(ModuleMatcher.glob(e));
        }
    }

    /**
     * V modelopt.py ModelOptMixedPrecisionConfig: one rule per {@code quantized_layers} entry; modules it does not
     * list stay unquantized. Every NVFP4-family layer uses the configured {@code group_size}, else that of the first
     * NVFP4-family entry, else 16; the block-FP8 layers share one block size (their {@code group_size}, default 128).
     */
    private static void modelOptMixed(Assembly a, Map<String, Object> perLayer, int configuredGroup) {
        if (perLayer.isEmpty()) {
            throw new GgufFormatException(a.where + ": ModelOpt MIXED_PRECISION needs a non-empty quantized_layers");
        }
        int nvfp4Group = configuredGroup;
        int fp8Block = 0;
        for (Map.Entry<String, Object> e : perLayer.entrySet()) {
            String where = a.where + ".quantized_layers." + e.getKey();
            Map<String, Object> recipe = optObject(e.getValue(), where);
            String layerAlgo = QuantArgs.strOr(recipe, "quant_algo", "", where).toUpperCase(Locale.ROOT);
            if (nvfp4Group <= 0 && (layerAlgo.equals("NVFP4") || layerAlgo.equals("W4A16_NVFP4"))) {
                nvfp4Group = recipe.get("group_size") == null ? 16 : groupSize(recipe, where);
            }
            if (layerAlgo.equals("FP8_PB_WO") || layerAlgo.equals("FP8_BLOCK_SCALES")) {
                int block = recipe.get("group_size") == null ? 128 : groupSize(recipe, where);
                if (fp8Block != 0 && fp8Block != block) {
                    throw new GgufFormatException(a.where + ": ModelOpt MIXED_PRECISION block-FP8 layers use one "
                            + "group_size, got " + fp8Block + " and " + block);
                }
                fp8Block = block;
            }
        }
        for (Map.Entry<String, Object> e : perLayer.entrySet()) {
            String where = a.where + ".quantized_layers." + e.getKey();
            String layerAlgo = QuantArgs.strOr(optObject(e.getValue(), where), "quant_algo", "", where)
                    .toUpperCase(Locale.ROOT);
            int group = layerAlgo.endsWith("NVFP4") ? nvfp4Group : fp8Block;
            a.rules.add(QuantRule.of(ModuleMatcher.prefix(e.getKey()), modelOptArgs(layerAlgo, group)));
        }
    }

    /**
     * The arguments of ModelOpt {@code quant_algo} (V modelopt.py LINEAR_ALGOS; {@code FP8_BLOCK_SCALES} is the
     * compatibility alias of {@code FP8_PB_WO}).
     *
     * @param groupSize the configured {@code group_size} (NVFP4 groups, or the square block of a MIXED_PRECISION
     *                  block-FP8 layer), or 0 for the algorithm's default
     * @throws GgufFormatException {@code "unsupported ModelOpt quant_algo '<x>'"} for any other name
     */
    static QuantArgs modelOptArgs(String algo, int groupSize) {
        switch (algo.toUpperCase(Locale.ROOT)) {
            case "FP8":
                return QuantArgs.builder(QuantFormat.FP8).actScheme(ActQuantScheme.FP8_STATIC).build();
            case "FP8_PER_CHANNEL_PER_TOKEN":
                return QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL)
                        .actScheme(ActQuantScheme.FP8_TOKEN).build();
            case "FP8_PB_WO": case "FP8_BLOCK_SCALES":
                int block = groupSize > 0 ? groupSize : 128;
                return QuantArgs.builder(QuantFormat.FP8).block(block, block).build();
            case "NVFP4":
                return QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).groupSize(groupSize > 0 ? groupSize : 16)
                        .actScheme(ActQuantScheme.NVFP4).build();
            case "W4A16_NVFP4":
                return QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).groupSize(groupSize > 0 ? groupSize : 16).build();
            case "MXFP8":
                return QuantArgs.builder(QuantFormat.MXFP8).actScheme(ActQuantScheme.MXFP8).build();
            default:
                throw new GgufFormatException("unsupported ModelOpt quant_algo '" + algo + "'");
        }
    }

    private static int groupSize(Map<String, Object> m, String where) {
        Object v = m.get("group_size");
        if (v instanceof String) {
            try {
                return Integer.parseInt(((String) v).trim());
            } catch (NumberFormatException e) {
                throw new GgufFormatException(where + ".group_size '" + v + "' is not an integer", e);
            }
        }
        return v == null ? 0 : QuantArgs.toInt(v, where + ".group_size");
    }

    /**
     * V auto_gptq.py and gptq_utils.py: {@code lm_head} stays unquantized unless {@code lm_head} is true; with
     * {@code modules_in_block_to_quantize}, modules containing none of its names stay unquantized; then the GPTQModel
     * {@code dynamic} overrides in order, then the base rule.
     */
    private static void gptq(Assembly a, boolean moeWna16) {
        if ("marlin".equals(a.q.get("checkpoint_format")) || Boolean.TRUE.equals(a.q.get("is_marlin_format"))) {
            throw new GgufFormatException("unsupported quant_method '" + a.name + "': " + a.where
                    + " declares checkpoint_format 'marlin'; Marlin-repacked GPTQ checkpoints are not supported");
        }
        int bits = required(a.q, moeWna16 && a.q.containsKey("weight_bits") ? "weight_bits" : "bits", a.where);
        int group = required(a.q, "group_size", a.where);
        boolean descAct = QuantArgs.boolOr(a.q, "desc_act", false, a.where);
        QuantArgs.Builder base = QuantArgs.builder(QuantFormat.INT_GPTQ).bits(bits).group(group)
                .symmetric(QuantArgs.boolOr(a.q, "sym", true, a.where)).actOrder(descAct && group != QuantArgs.CHANNEL)
                .gptqVersion("gptq_v2".equals(a.q.get("checkpoint_format")) ? 2 : 1);
        Map<String, Object> dynamic = optObject(a.q.get("dynamic"), a.where + ".dynamic");
        QuantArgs baseArgs = base.build();
        excludeLmHead(a);
        List<String> inBlock = flatStrings(a.q.get("modules_in_block_to_quantize"),
                a.where + ".modules_in_block_to_quantize");
        if (!inBlock.isEmpty()) {
            StringBuilder any = new StringBuilder();
            for (String m : inBlock) {
                any.append(any.length() == 0 ? "" : "|").append(ModuleMatcher.escape(m));
            }
            a.rules.add(QuantRule.exclude(ModuleMatcher.regex("(?!.*(?:" + any + "))")));
        }
        for (Map.Entry<String, Object> e : dynamic.entrySet()) {
            String key = e.getKey();
            if (key.startsWith("-:")) {
                a.rules.add(QuantRule.exclude(ModuleMatcher.regex(key.substring(2))));
                continue;
            }
            String where = a.where + ".dynamic." + key;
            Map<String, Object> o = optObject(e.getValue(), where);
            int g = QuantArgs.intOr(o, "group_size", baseArgs.groupSize(), where);
            boolean desc = QuantArgs.boolOr(o, "desc_act", descAct, where);
            a.rules.add(QuantRule.of(ModuleMatcher.regex(key.startsWith("+:") ? key.substring(2) : key),
                    baseArgs.toBuilder().bits(QuantArgs.intOr(o, "bits", baseArgs.bits(), where)).group(g)
                            .symmetric(QuantArgs.boolOr(o, "sym", baseArgs.symmetric(), where))
                            .actOrder(desc && g != QuantArgs.CHANNEL).build()));
        }
        a.all(baseArgs);
    }

    /**
     * SG awq.py AWQConfig.from_config: {@code modules_to_not_convert} by substring; {@code lm_head} unquantized unless
     * {@code lm_head} is true (AWQMarlinConfig).
     */
    private static void awq(Assembly a) {
        a.ignore(a.q.get("modules_to_not_convert"), "modules_to_not_convert", ModuleMatcher::substring);
        excludeLmHead(a);
        Object bits = firstOf(a.q, "w_bit", "bits");
        Object group = firstOf(a.q, "q_group_size", "group_size");
        if (bits == null || group == null) {
            throw new GgufFormatException(a.where + ": AWQ needs w_bit|bits and q_group_size|group_size");
        }
        a.all(QuantArgs.builder(QuantFormat.INT_AWQ).bits(QuantArgs.toInt(bits, a.where + ".bits"))
                .group(QuantArgs.toInt(group, a.where + ".group_size"))
                .symmetric(!QuantArgs.boolOr(a.q, "zero_point", true, a.where)).build());
    }

    /**
     * V inc/inc.py and inc/config_parser.py: AutoRound / INC. Integer weights are GPTQ-packed when the packing format
     * or backend names gptq, else AWQ-packed when either names awq; {@code extra_config} names (exact or dotted
     * suffix) win over its regex keys; {@code lm_head} and modules outside {@code block_name_to_quantize} stay
     * unquantized unless {@code extra_config} names them.
     */
    private static void autoRound(Assembly a) {
        String dataType = QuantArgs.strOr(a.q, "data_type", "int", a.where);
        dataType = dataType.equals("fp8") ? "fp" : dataType;
        int bits = required(a.q, "bits", a.where);
        Object groupValue = a.q.get("group_size");
        boolean sym = QuantArgs.boolOr(a.q, "sym", true, a.where);
        QuantArgs base;
        if (dataType.equals("int")) {
            String packing = QuantArgs.strOr(a.q, "packing_format", "auto_round:auto_gptq", a.where);
            String backend = Objects.toString(firstOf(a.q, "backend", "vllm_backend"), "auto");
            boolean gptq = packing.contains("gptq") || backend.contains("gptq");
            boolean awq = !gptq && (packing.contains("awq") || backend.contains("awq"));
            if (!gptq && !awq) {
                throw new GgufFormatException(a.where + ": AutoRound integer packing_format '" + packing
                        + "' with backend '" + backend + "' is neither GPTQ nor AWQ packed");
            }
            a.method = awq ? AWQ : GPTQ;
            base = QuantArgs.builder(awq ? QuantFormat.INT_AWQ : QuantFormat.INT_GPTQ).bits(bits)
                    .group(QuantArgs.toInt(groupValue, a.where + ".group_size")).symmetric(sym).build();
        } else if (dataType.equals("fp") && bits == 8 && groupValue instanceof List) {
            List<?> g = (List<?>) groupValue;
            if (g.size() != 2 || !sym) {
                throw new GgufFormatException(a.where + ": AutoRound block FP8 needs sym and group_size [N, K], got "
                        + g);
            }
            base = blockFp8(new int[]{QuantArgs.toInt(g.get(0), a.where + ".group_size[0]"),
                    QuantArgs.toInt(g.get(1), a.where + ".group_size[1]")});
        } else if (dataType.contains("mx_fp") && (bits == 4 || bits == 8)) {
            base = QuantArgs.builder(bits == 4 ? QuantFormat.MXFP4_CT : QuantFormat.MXFP8)
                    .actScheme(bits == 4 ? ActQuantScheme.MXFP4 : ActQuantScheme.MXFP8).build();
        } else {
            throw new GgufFormatException(a.where + ": unsupported AutoRound data_type '" + dataType + "' with bits "
                    + bits + " and group_size " + groupValue);
        }
        Map<String, Object> extra = optObject(a.q.get("extra_config"), a.where + ".extra_config");
        List<QuantRule> regexRules = new ArrayList<>();
        for (Map.Entry<String, Object> e : extra.entrySet()) {
            String where = a.where + ".extra_config." + e.getKey();
            Map<String, Object> o = optObject(e.getValue(), where);
            boolean regex = REGEX_CHARS.matcher(e.getKey()).find();
            ModuleMatcher m = regex ? ModuleMatcher.search(e.getKey()) : ModuleMatcher.suffix(e.getKey());
            int layerBits = QuantArgs.intOr(o, "bits", base.bits(), where);
            QuantRule rule = layerBits >= 16 ? QuantRule.exclude(m) : QuantRule.of(m, base.toBuilder().bits(layerBits)
                    .group(QuantArgs.intOr(o, "group_size", base.groupSize(), where))
                    .symmetric(QuantArgs.boolOr(o, "sym", base.symmetric(), where)).build());
            (regex ? regexRules : a.rules).add(rule);
        }
        a.rules.addAll(regexRules);
        Object blocks = firstOf(a.q, "block_name_to_quantize", "to_quant_block_names");
        if (blocks == null) {
            a.rules.add(QuantRule.exclude(ModuleMatcher.suffix("lm_head")));
            a.all(base);
            return;
        }
        List<String> names = blocks instanceof String ? List.of(((String) blocks).split(","))
                : strings(blocks, a.where + ".block_name_to_quantize");
        for (String n : names) {
            a.rules.add(QuantRule.of(ModuleMatcher.regex(Pattern.quote(n)), base));
        }
        a.rules.add(QuantRule.exclude(ModuleMatcher.anyLinear()));
    }

    /** V bitsandbytes.py: 4-bit NF4/FP4 or LLM.int8, skipping {@code llm_int8_skip_modules}. */
    private static void bitsandbytes(Assembly a) {
        a.ignore(a.q.get("llm_int8_skip_modules"), "llm_int8_skip_modules", ModuleMatcher::component);
        if (QuantArgs.boolOr(a.q, "load_in_4bit", false, a.where)) {
            String type = QuantArgs.strOr(a.q, "bnb_4bit_quant_type", "fp4", a.where);
            if (!type.equals("nf4") && !type.equals("fp4")) {
                throw new GgufFormatException(a.where + ": unsupported bnb_4bit_quant_type '" + type + "'");
            }
            Object dq = a.q.get("bnb_4bit_use_double_quant");
            a.all(QuantArgs.builder(type.equals("nf4") ? QuantFormat.NF4_BNB : QuantFormat.FP4_BNB)
                    .bnbDoubleQuant(dq == null ? null : QuantArgs.boolOr(a.q, "bnb_4bit_use_double_quant", false,
                            a.where)).build());
        } else if (QuantArgs.boolOr(a.q, "load_in_8bit", false, a.where)) {
            a.all(QuantArgs.builder(QuantFormat.INT8_BNB).group(QuantArgs.CHANNEL).build());
        } else {
            throw new GgufFormatException(a.where + ": bitsandbytes configuration sets neither load_in_4bit nor "
                    + "load_in_8bit");
        }
    }

    /** V quark.py: layer_quant_config globs, then layer_type_quant_config, then global_quant_config. */
    private static void quark(Assembly a) {
        Map<String, Object> export = optObject(a.q.get("export"), a.where + ".export");
        String pack = QuantArgs.strOr(export, "pack_method", null, a.where + ".export");
        a.ignore(a.q.get("exclude"), "exclude", ModuleMatcher::exactOrRegex);
        Map<String, Object> layers = optObject(a.q.get("layer_quant_config"), a.where + ".layer_quant_config");
        for (Map.Entry<String, Object> e : layers.entrySet()) {
            quarkRule(a, ModuleMatcher.glob(e.getKey()), e.getValue(), pack, "layer_quant_config." + e.getKey());
        }
        Map<String, Object> types = optObject(a.q.get("layer_type_quant_config"),
                a.where + ".layer_type_quant_config");
        for (Map.Entry<String, Object> e : types.entrySet()) {
            quarkRule(a, ModuleMatcher.className(e.getKey()), e.getValue(), pack,
                    "layer_type_quant_config." + e.getKey());
        }
        if (a.q.get("global_quant_config") != null) {
            quarkRule(a, ModuleMatcher.anyLinear(), a.q.get("global_quant_config"), pack, "global_quant_config");
        }
    }

    private static void quarkRule(Assembly a, ModuleMatcher m, Object layer, String pack, String path) {
        String where = a.where + "." + path;
        Map<String, Object> cfg = optObject(layer, where);
        a.rules.add(cfg.get("weight") == null ? QuantRule.exclude(m) : QuantRule.of(m, QuantArgs.quark(cfg, pack, where)));
    }

    /** MLX affine: per-module entries (an object of overrides, or {@code false}) before the base rule. */
    private static void mlx(Assembly a) {
        String mode = QuantArgs.strOr(a.q, "mode", "affine", a.where);
        if (!MLX_MODES.contains(mode)) {
            throw new GgufFormatException(a.where + ": unsupported MLX quantization mode '" + mode + "'");
        }
        int bits = required(a.q, "bits", a.where);
        QuantArgs base = QuantArgs.builder(QuantFormat.INT_MLX).bits(bits).mlxBits(bits)
                .group(required(a.q, "group_size", a.where)).symmetric(false).build();
        for (Map.Entry<String, Object> e : a.q.entrySet()) {
            String where = a.where + "." + e.getKey();
            if (Boolean.FALSE.equals(e.getValue())) {
                a.rules.add(QuantRule.exclude(ModuleMatcher.exact(e.getKey())));
            } else if (e.getValue() instanceof Map) {
                Map<String, Object> o = HfConfig.asObject(e.getValue());
                int b = QuantArgs.intOr(o, "bits", bits, where);
                a.rules.add(QuantRule.of(ModuleMatcher.exact(e.getKey()), base.toBuilder().bits(b).mlxBits(b)
                        .group(QuantArgs.intOr(o, "group_size", base.groupSize(), where)).build()));
            }
        }
        a.all(base);
    }

    /**
     * transformers BitNet: offline checkpoints store packed ternary weights, online ones store floats;
     * {@code modules_to_not_convert} follows {@code should_convert_module}.
     */
    private static void bitnet(Assembly a) {
        if ("online".equals(QuantArgs.strOr(a.q, "quantization_mode", "offline", a.where))) {
            a.method = ONLINE;
            HfQuantConfig.logOnline(a.name);
            return;
        }
        a.ignore(a.q.get("modules_to_not_convert"), "modules_to_not_convert", ModuleMatcher::modulePattern);
        a.all(QuantArgs.builder(QuantFormat.BITNET).build());
    }

    /** SG blockwise_int8.py: int8 block weights with dynamic per-group int8 activations. */
    private static void blockwiseInt8(Assembly a) {
        a.ignore(a.q.get("ignored_layers"), "ignored_layers", ModuleMatcher::exactOrRegex);
        if (!activationScheme(a).equals("dynamic")) {
            throw new GgufFormatException(a.where + ": blockwise_int8 supports only activation_scheme 'dynamic'");
        }
        int[] block = blockSize(a, null);
        if (block == null) {
            throw new GgufFormatException(a.where + ": blockwise_int8 needs weight_block_size [N, K]");
        }
        QuantArgs.Builder b = QuantArgs.builder(QuantFormat.INT8_BLOCK).block(block[0], block[1]);
        a.all((block[1] == 128 ? b.actScheme(ActQuantScheme.INT8_GROUP128)
                : b.actUnemulated("dynamic int8 activations in groups of " + block[1])).build());
    }

    /** vLLM HQQ (hqq_marlin.py before its removal): {@code quant_config.weight_quant_params}, {@code skip_modules}. */
    private static void hqq(Assembly a) {
        a.ignore(a.q.get("skip_modules"), "skip_modules", ModuleMatcher::component);
        Map<String, Object> inner = optObject(a.q.get("quant_config"), a.where + ".quant_config");
        String where = a.where + ".quant_config.weight_quant_params";
        Map<String, Object> wq = optObject(inner.get("weight_quant_params"), where);
        if (wq.isEmpty()) {
            throw new GgufFormatException(where + " is missing");
        }
        a.all(QuantArgs.builder(QuantFormat.INT_HQQ).bits(required(wq, "nbits", where))
                .group(required(wq, "group_size", where)).symmetric(false)
                .hqqAxis(QuantArgs.intOr(wq, "axis", 1, where)).build());
    }

    /**
     * transformers FPQuantConfig.post_init / V fp_quant.py: MXFP4 or NVFP4 (the default) behind a Hadamard rotation,
     * or pseudo-quantized weights; {@code forward_method} defaults to abs_max, the only method NVFP4 accepts.
     */
    private static void fpQuant(Assembly a) {
        Object skip = a.q.containsKey("modules_to_not_convert") ? a.q.get("modules_to_not_convert") : List.of("lm_head");
        a.ignore(skip, "modules_to_not_convert", ModuleMatcher::suffix);
        String dtype = QuantArgs.strOr(a.q, "forward_dtype", "nvfp4", a.where);
        if (!dtype.equals("mxfp4") && !dtype.equals("nvfp4")) {
            throw new GgufFormatException(a.where + ": unsupported fp_quant forward_dtype '" + dtype + "'");
        }
        int hg = QuantArgs.intOr(a.q, "hadamard_group_size", dtype.equals("nvfp4") ? 16 : 32, a.where);
        Set<Integer> allowed = dtype.equals("mxfp4") ? Set.of(32, 64, 128) : Set.of(16, 32, 64, 128);
        if (!allowed.contains(hg)) {
            throw new GgufFormatException(a.where + ": fp_quant hadamard_group_size " + hg + " is not one of "
                    + allowed + " for " + dtype);
        }
        String init = QuantArgs.strOr(a.q, "transform_init", "hadamard", a.where).toUpperCase(Locale.ROOT);
        QuantArgs.FpQuantTransform transform;
        try {
            transform = QuantArgs.FpQuantTransform.valueOf(init);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(a.where + ": unknown fp_quant transform_init '" + init + "'", e);
        }
        String forwardMethod = QuantArgs.strOr(a.q, "forward_method", "abs_max", a.where);
        if (!forwardMethod.equals("quest") && !forwardMethod.equals("abs_max")) {
            throw new GgufFormatException(a.where + ": unknown fp_quant forward_method '" + forwardMethod + "'");
        }
        if (dtype.equals("nvfp4") && !forwardMethod.equals("abs_max")) {
            throw new GgufFormatException(a.where + ": fp_quant nvfp4 requires forward_method 'abs_max', got '"
                    + forwardMethod + "'");
        }
        boolean pseudo = QuantArgs.boolOr(a.q, "pseudoquantization", false, a.where);
        QuantFormat f = pseudo ? QuantFormat.FPQUANT_PSEUDO
                : dtype.equals("mxfp4") ? QuantFormat.FPQUANT_MXFP4 : QuantFormat.FPQUANT_NVFP4;
        a.all(QuantArgs.builder(f).fpQuantHadamardGroup(hg).fpQuantTransform(transform)
                .actUnemulated("fp_quant " + dtype + " activations (forward_method '" + forwardMethod + "')").build());
    }

    /** SG w4afp8.py: int4 expert weights with static FP8 activations; FP8 128x128 block linears. */
    private static void w4afp8(Assembly a) {
        a.ignore(a.q.get("ignored_layers"), "ignored_layers", ModuleMatcher::exactOrRegex);
        a.rules.add(QuantRule.of(ModuleMatcher.regex(EXPERTS), QuantArgs.builder(QuantFormat.INT4_IN_INT8)
                .group(QuantArgs.intOr(a.q, "group_size", 128, a.where)).actScheme(ActQuantScheme.FP8_STATIC).build()));
        a.all(blockFp8(new int[]{128, 128}));
    }

    /**
     * transformers {@code TorchAoConfig} (V torchao.py TorchAOConfig.from_config): {@code modules_to_not_convert}
     * skips modules; {@code quant_type} holds exactly one {@code "default"} entry whose {@code _type} names the
     * torchao configuration and whose {@code _data} carries its arguments. A {@code module_fqn_to_config} inside
     * that entry gives per-module configurations (exact names or {@code re:} full-match patterns, with
     * {@code _default}), as V torchao.py does. The supported configurations map onto the checkpoint layouts the
     * readers decode ({@link QuantFormat#INT4_TORCHAO}, {@link QuantFormat#INT8_CHANNEL},
     * {@link QuantFormat#FP8}, {@link QuantFormat#MXFP4_CT}, {@link QuantFormat#MXFP8},
     * {@link QuantFormat#NVFP4_MODELOPT}); any other configuration is refused by name.
     */
    private static void torchao(Assembly a) {
        a.ignore(a.q.get("modules_to_not_convert"), "modules_to_not_convert", ModuleMatcher::component);
        Object quantType = a.q.get("quant_type");
        if (quantType == null) {
            throw new GgufFormatException(a.where + ".quant_type is missing; torchao needs the quant_type to apply");
        }
        Map<String, Object> byName = optObject(quantType, a.where + ".quant_type");
        if (byName.size() != 1 || !byName.containsKey("default")) {
            throw new GgufFormatException(a.where + ".quant_type must hold exactly the 'default' entry, got "
                    + byName.keySet());
        }
        Map<String, Object> type = optObject(byName.get("default"), a.where + ".quant_type.default");
        String name = QuantArgs.strOr(type, "_type", "", a.where + ".quant_type.default");
        if (name.isEmpty()) {
            throw new GgufFormatException(a.where + ".quant_type.default has no '_type' naming its torchao"
                    + " configuration");
        }
        Map<String, Object> data = optObject(type.get("_data"), a.where + ".quant_type.default._data");
        Map<String, Object> perModule = nullableObject(data.get("module_fqn_to_config"),
                a.where + ".quant_type.default._data.module_fqn_to_config");
        if (perModule == null || perModule.isEmpty()) {
            a.all(torchaoArgs(name, data, a.where + ".quant_type.default"));
            return;
        }
        for (Map.Entry<String, Object> e : perModule.entrySet()) {
            String fqn = e.getKey();
            if (fqn.equals("_default")) {
                continue;
            }
            if (fqn.isEmpty() || fqn.equals("re:")) {
                throw new GgufFormatException(a.where + ".quant_type.default._data.module_fqn_to_config: '" + fqn
                        + "' is not a module name or a 're:' pattern");
            }
            ModuleMatcher matcher = fqn.startsWith("re:")
                    ? ModuleMatcher.regex("(?:" + fqn.substring(3) + ")\\Z")
                    : ModuleMatcher.exact(fqn);
            if (e.getValue() == null) {
                a.rules.add(QuantRule.exclude(matcher));
            } else {
                Map<String, Object> cfg = object(e.getValue(), a.where + ".module_fqn_to_config['" + fqn + "']");
                a.rules.add(QuantRule.of(matcher, torchaoArgs(requiredType(cfg, a.where + ".module_fqn_to_config['"
                        + fqn + "']"), optObject(cfg.get("_data"), a.where + "." + fqn + "._data"),
                        a.where + ".module_fqn_to_config['" + fqn + "']")));
            }
        }
        Object dflt = perModule.get("_default");
        if (dflt != null) {
            Map<String, Object> cfg = object(dflt, a.where + ".module_fqn_to_config._default");
            a.rules.add(QuantRule.of(ModuleMatcher.regex(".*"), torchaoArgs(requiredType(cfg, a.where
                    + ".module_fqn_to_config._default"), optObject(cfg.get("_data"), a.where + "._default._data"),
                    a.where + ".module_fqn_to_config._default")));
        }
    }

    private static String requiredType(Map<String, Object> cfg, String where) {
        String type = QuantArgs.strOr(cfg, "_type", "", where);
        if (type.isEmpty()) {
            throw new GgufFormatException(where + " has no '_type' naming its torchao configuration");
        }
        return type;
    }

    /**
     * The storage arguments of one torchao configuration (V quant_api.py and the prototype MX configs). The
     * checkpoint's serialized tensor subclass is authoritative for the packing layout; the configuration supplies
     * the group size, the weight scale format and the activation quantization.
     */
    private static QuantArgs torchaoArgs(String type, Map<String, Object> d, String where) {
        switch (type) {
            case "Int4WeightOnlyConfig": {
                int group = QuantArgs.intOr(d, "group_size", 128, where);
                if (group <= 0) {
                    throw new GgufFormatException(where + ": Int4WeightOnlyConfig group_size must be positive, got "
                            + group);
                }
                String packing = serializedValue(d.get("int4_packing_format"));
                if (packing != null) {
                    String format = packing.toUpperCase(Locale.ROOT);
                    if (!format.equals("PLAIN") && !format.equals("PRESHUFFLED")
                            && !format.equals("PLAIN_INT32") && !format.equals("TILE_PACKED_TO_4D")) {
                        throw new GgufFormatException(where + ": Int4WeightOnlyConfig int4_packing_format '" + packing
                                + "' has no NeuralFree mapping; PLAIN, PRESHUFFLED, PLAIN_INT32 and"
                                + " TILE_PACKED_TO_4D are supported");
                    }
                    if (format.equals("TILE_PACKED_TO_4D")) {
                        int ntile = QuantArgs.intOr(d, "int4_tile_packed_ntile", 8, where);
                        if (ntile != 8 && ntile != 16) {
                            throw new GgufFormatException(where + ": Int4WeightOnlyConfig int4_tile_packed_ntile "
                                    + ntile + " has no NeuralFree mapping; 8 (CUDA) and 16 (ROCm) are supported");
                        }
                    }
                }
                return QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(group).build();
            }
            case "Float8DynamicActivationInt4WeightConfig": {
                String packing = serializedValue(d.get("int4_packing_format"));
                if (packing != null && !packing.equalsIgnoreCase("preshuffled")
                        && !packing.equalsIgnoreCase("plain")) {
                    throw new GgufFormatException(where + ": Float8DynamicActivationInt4WeightConfig"
                            + " int4_packing_format '" + packing + "' has no NeuralFree mapping; preshuffled and"
                            + " plain are supported");
                }
                return QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(128)
                        .actScheme(ActQuantScheme.FP8_TOKEN).build();
            }
            case "Int8WeightOnlyConfig":
                return QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(weightGroup(d.get("granularity"), where))
                        .build();
            case "Int8StaticActivationInt8WeightConfig": {
                List<Object> granularities = granularityList(d.get("granularity"));
                Object weightGranularity = granularities.isEmpty() ? null
                        : granularities.get(granularities.size() == 1 ? 0 : 1);
                QuantArgs.Builder b = QuantArgs.builder(QuantFormat.INT8_CHANNEL)
                        .group(weightGroup(weightGranularity, where));
                if (Boolean.TRUE.equals(d.get("reduce_range"))) {
                    b.actUnemulated("torchao reduced-range int8 activations");
                } else {
                    String mapping = serializedValue(d.get("act_mapping_type"));
                    b.actScheme(mapping != null && !mapping.equals("SYMMETRIC")
                            ? ActQuantScheme.INT8_STATIC_ASYM : ActQuantScheme.INT8_STATIC);
                }
                return b.build();
            }
            case "IntxWeightOnlyConfig": {
                return intx(d, where, false);
            }
            case "Int8DynamicActivationIntxWeightConfig": {
                return intx(d, where, true);
            }
            case "Int8DynamicActivationInt8WeightConfig": {
                List<Object> granularities = granularityList(d.get("granularity"));
                Object actGranularity = granularities.isEmpty() ? null : granularities.get(0);
                Object weightGranularity = granularities.isEmpty() ? null
                        : granularities.get(granularities.size() == 1 ? 0 : 1);
                QuantArgs.Builder b = QuantArgs.builder(QuantFormat.INT8_CHANNEL)
                        .group(weightGroup(weightGranularity, where));
                String mapping = serializedValue(d.get("act_mapping_type"));
                if (Boolean.TRUE.equals(d.get("reduce_range"))) {
                    b.actUnemulated("torchao reduced-range int8 activations");
                } else if (mapping != null && !mapping.equals("SYMMETRIC")) {
                    b.actUnemulated("torchao " + mapping + " int8 activations");
                } else if (actGranularity != null && !isPerRow(actGranularity)) {
                    b.actUnemulated("torchao int8 activations at " + granularityName(actGranularity)
                            + " granularity");
                } else {
                    b.actScheme(ActQuantScheme.INT8_TOKEN);
                }
                return b.build();
            }
            case "Float8WeightOnlyConfig": {
                return fp8Builder(weightGroup(d.get("granularity"), where)).build();
            }
            case "Float8DynamicActivationFloat8WeightConfig": {
                String packing = serializedValue(d.get("packing_format"));
                if (packing != null && !packing.equals("PLAIN")) {
                    throw new GgufFormatException(where + ": Float8DynamicActivationFloat8WeightConfig"
                            + " packing_format '" + packing + "' (a sparse tensor subclass) is not supported;"
                            + " PLAIN is");
                }
                List<Object> granularities = granularityList(d.get("granularity"));
                boolean defaults = granularities.isEmpty();
                Object actGranularity = defaults ? null : granularities.get(0);
                Object weightGranularity = defaults ? null
                        : granularities.get(granularities.size() == 1 ? 0 : 1);
                QuantArgs.Builder b = fp8Builder(defaults ? QuantArgs.UNGROUPED : weightGroup(weightGranularity,
                        where));
                String actDtype = serializedValue(d.get("activation_dtype"));
                if (actDtype != null && !actDtype.equals("float8_e4m3fn")) {
                    b.actUnemulated("torchao " + actDtype + " activations, which NeuralFree does not quantize");
                } else if (actGranularity == null) {
                    b.actUnemulated("torchao FP8 per-tensor dynamic activations, whose scale is computed from the"
                            + " whole activation tensor");
                } else if (isPerRow(actGranularity)) {
                    b.actScheme(ActQuantScheme.FP8_TOKEN);
                } else {
                    b.actUnemulated("torchao FP8 activations at " + granularityName(actGranularity)
                            + " granularity");
                }
                return b.build();
            }
            case "MXDynamicActivationMXWeightConfig": {
                int block = QuantArgs.intOr(d, "block_size", 32, where);
                if (block != 32) {
                    throw new GgufFormatException(where + ": MXDynamicActivationMXWeightConfig block_size must be"
                            + " 32, got " + block);
                }
                String dtype = serializedValue(d.get("weight_dtype"));
                if (dtype == null) {
                    dtype = "float8_e4m3fn";
                }
                if ("float4_e2m1fn_x2".equals(dtype)) {
                    return QuantArgs.builder(QuantFormat.MXFP4_CT).actScheme(ActQuantScheme.MXFP4).build();
                }
                if ("float8_e4m3fn".equals(dtype)) {
                    return QuantArgs.builder(QuantFormat.MXFP8).actScheme(ActQuantScheme.MXFP8).build();
                }
                if ("float8_e5m2".equals(dtype)) {
                    throw new GgufFormatException(where + ": MXDynamicActivationMXWeightConfig weight_dtype"
                            + " 'float8_e5m2' has no MXFP8 GGML mapping; only float8_e4m3fn does");
                }
                throw new GgufFormatException(where + ": MXDynamicActivationMXWeightConfig weight_dtype '" + dtype
                        + "' has no NeuralFree mapping; float4_e2m1fn_x2 and float8_e4m3fn do");
            }
            case "NVFP4DynamicActivationNVFP4WeightConfig": {
                QuantArgs.Builder b = QuantArgs.builder(QuantFormat.NVFP4_MODELOPT);
                if (QuantArgs.boolOr(d, "use_dynamic_per_tensor_scale", true, where)) {
                    b.actUnemulated("torchao NVFP4 dynamic per-tensor activations, whose global scale is computed"
                            + " from the whole activation tensor");
                } else {
                    b.actScheme(ActQuantScheme.NVFP4);
                }
                return b.build();
            }
            default:
                throw new GgufFormatException("unsupported quant_method 'torchao': quant_type '" + type
                        + "' has no NeuralFree mapping; supported configurations are Int4WeightOnlyConfig (PLAIN,"
                        + " PRESHUFFLED, PLAIN_INT32, TILE_PACKED_TO_4D), Float8DynamicActivationInt4WeightConfig,"
                        + " Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig,"
                        + " Int8StaticActivationInt8WeightConfig, IntxWeightOnlyConfig,"
                        + " Int8DynamicActivationIntxWeightConfig, Float8WeightOnlyConfig,"
                        + " Float8DynamicActivationFloat8WeightConfig, MXDynamicActivationMXWeightConfig and"
                        + " NVFP4DynamicActivationNVFP4WeightConfig");
        }
    }

    /**
     * The storage arguments of an {@code IntxWeightOnlyConfig} or {@code Int8DynamicActivationIntxWeightConfig}.
     * The unpacked-to-int8 format stores one code per byte. An int4 target packs into nibbles ({@link
     * QuantFormat#INT4_TORCHAO}); every other target width keeps its int8 bytes, which the int8 channel scheme
     * carries exactly with the group scale and zero point ({@link QuantFormat#INT8_CHANNEL}).
     */
    private static QuantArgs intx(Map<String, Object> d, String where, boolean dynamic) {
        String packing = serializedValue(d.get("intx_packing_format"));
        if (packing != null && !packing.equalsIgnoreCase("UNPACKED_TO_INT8")) {
            throw new GgufFormatException(where + ": intx_packing_format '" + packing + "' is an opaque packed"
                    + " tensor; only UNPACKED_TO_INT8 is decoded");
        }
        int bits = weightBits(d.get("weight_dtype"), where);
        Object granularity = d.get("weight_granularity");
        if (granularity == null) {
            granularity = d.get("granularity");
        }
        QuantArgs.Builder b = bits == 4
                ? QuantArgs.builder(QuantFormat.INT4_TORCHAO)
                : QuantArgs.builder(QuantFormat.INT8_CHANNEL);
        b.group(weightGroup(granularity, where));
        if (dynamic) {
            String mapping = serializedValue(d.get("act_mapping_type"));
            if (mapping != null && !mapping.equals("SYMMETRIC")) {
                b.actUnemulated("torchao " + mapping + " intx activations");
            } else {
                b.actScheme(ActQuantScheme.INT8_TOKEN);
            }
        }
        return b.build();
    }

    /** The bits of a torch {@code int1}..{@code int8} dtype marker; 8 when absent. */
    private static int weightBits(Object v, String where) {
        String dtype = serializedValue(v);
        if (dtype == null) {
            return 8;
        }
        if (!dtype.startsWith("int") || dtype.length() <= 3) {
            throw new GgufFormatException(where + ": weight_dtype '" + dtype + "' is not a torch.int1..int8 dtype");
        }
        try {
            int bits = Integer.parseInt(dtype.substring(3));
            if (bits < 1 || bits > 8) {
                throw new GgufFormatException(where + ": weight_dtype '" + dtype + "' is out of int1..int8");
            }
            return bits;
        } catch (NumberFormatException e) {
            throw new GgufFormatException(where + ": weight_dtype '" + dtype + "' is not a torch.int1..int8 dtype");
        }
    }

    /** The {@code _type} of a serialized torchao enum, dataclass or dtype marker; null when absent. */
    private static String nestedName(Object v) {
        if (!(v instanceof Map)) {
            return null;
        }
        Object type = ((Map<?, ?>) v).get("_type");
        return type instanceof String ? (String) type : null;
    }

    /**
     * The value a serialized torchao enum or dtype marker names: its string {@code _data}
     * ({@code {"_type":"Int4PackingFormat","_data":"PRESHUFFLED"}}), else its {@code _type} (a dataclass such as
     * {@code PerRow}, whose {@code _data} is an object).
     */
    private static String serializedValue(Object v) {
        if (!(v instanceof Map)) {
            return null;
        }
        Map<?, ?> m = (Map<?, ?>) v;
        Object data = m.get("_data");
        if (data instanceof String) {
            return (String) data;
        }
        return nestedName(v);
    }

    /** A granularity value or list of granularity values; one entry when a single gap is given. */
    private static List<Object> granularityList(Object v) {
        if (v instanceof List) {
            return new ArrayList<>((List<?>) v);
        }
        List<Object> one = new ArrayList<>();
        if (v != null) {
            one.add(v);
        }
        return one;
    }

    private static String granularityName(Object v) {
        String type = nestedName(v);
        return type == null ? "an unknown granularity" : type;
    }

    private static boolean isPerRow(Object v) {
        if (!(v instanceof Map) || !"PerRow".equals(nestedName(v))) {
            return false;
        }
        Map<String, Object> data = optObject(((Map<?, ?>) v).get("_data"), "granularity");
        Object dim = data.get("dim");
        return dim == null || (dim instanceof Number && ((Number) dim).longValue() == -1L);
    }

    /** FP8 per-tensor ({@link QuantArgs#UNGROUPED}), per-channel or per-group (a 1x group block) weight scales. */
    private static QuantArgs.Builder fp8Builder(int group) {
        QuantArgs.Builder b = QuantArgs.builder(QuantFormat.FP8);
        return group > 0 ? b.block(1, group) : b.group(group);
    }

    /** The weight group size of a torchao granularity: per row (channel), per tensor or a group size. */
    private static int weightGroup(Object v, String where) {
        if (v == null) {
            return QuantArgs.CHANNEL;
        }
        String type = nestedName(v);
        if (type == null) {
            throw new GgufFormatException(where + ": granularity must be a torchao PerRow, PerTensor or PerGroup"
                    + " object, got " + HfConfig.typeName(v));
        }
        Map<String, Object> data = optObject(((Map<?, ?>) v).get("_data"), where + ".granularity");
        switch (type) {
            case "PerRow":
                if (!isPerRow(v)) {
                    throw new GgufFormatException(where + ": weight PerRow granularity must use dim -1");
                }
                return QuantArgs.CHANNEL;
            case "PerTensor":
                return QuantArgs.UNGROUPED;
            case "PerAxis":
                if (QuantArgs.intOr(data, "axis", 0, where + ".granularity") != 0) {
                    throw new GgufFormatException(where + ": intx PerAxis granularity must use axis 0");
                }
                return QuantArgs.CHANNEL;
            case "PerGroup": {
                int group = QuantArgs.intOr(data, "group_size", 0, where + ".granularity");
                if (group <= 0) {
                    throw new GgufFormatException(where + ": PerGroup granularity needs a positive group_size, got "
                            + group);
                }
                return group;
            }
            default:
                throw new GgufFormatException(where + ": granularity '" + type + "' has no NeuralFree mapping");
        }
    }

    /** V/SG {@code lm_head_quantized}: {@code lm_head} stays unquantized unless the config sets {@code lm_head: true}. */
    private static void excludeLmHead(Assembly a) {
        if (!QuantArgs.boolOr(a.q, "lm_head", false, a.where)) {
            a.rules.add(QuantRule.exclude(ModuleMatcher.suffix("lm_head")));
        }
    }

    /** The strings of a string list, nested lists flattened (gptq_utils flatten_list); empty when absent. */
    private static List<String> flatStrings(Object v, String where) {
        List<String> out = new ArrayList<>();
        if (v == null) {
            return out;
        }
        if (!(v instanceof List)) {
            throw new GgufFormatException(where + " must be an array, got " + HfConfig.typeName(v));
        }
        for (Object item : (List<?>) v) {
            if (item instanceof List) {
                out.addAll(flatStrings(item, where));
            } else {
                out.addAll(strings(item, where));
            }
        }
        return out;
    }

    /**
     * V mxfp4.py: only the routed experts are MXFP4 ({@code _blocks}/{@code _scales}); every other linear is stored
     * unquantized. {@code modules_to_not_convert} follows transformers {@code should_convert_module}.
     */
    private static void mxfp4(Assembly a) {
        a.ignore(a.q.get("modules_to_not_convert"), "modules_to_not_convert", ModuleMatcher::modulePattern);
        a.rules.add(QuantRule.of(ModuleMatcher.regex(ANY_EXPERTS), QuantArgs.builder(QuantFormat.MXFP4_GPTOSS).build()));
    }

    /**
     * V modelopt.py ModelOptMxFp8Config.from_config: a config in ModelOpt form is read as ModelOpt; the bare
     * {@code mxfp8} form excludes its {@code ignored_layers} with ModelOpt matching.
     */
    private static void mxfp8(Assembly a) {
        if (a.q.get("quantization") != null || a.q.get("quant_algo") != null) {
            modelOpt(a);
            return;
        }
        modelOptExclude(a, a.q.get("ignored_layers"), "ignored_layers");
        a.all(QuantArgs.builder(QuantFormat.MXFP8).actScheme(ActQuantScheme.MXFP8).build());
    }

    private static int required(Map<String, Object> m, String key, String where) {
        if (m.get(key) == null) {
            throw new GgufFormatException(where + ": required member '" + key + "' is missing");
        }
        return QuantArgs.toInt(m.get(key), where + "." + key);
    }

    private static String activationScheme(Assembly a) {
        String s = QuantArgs.strOr(a.q, "activation_scheme", null, a.where);
        if (!"static".equals(s) && !"dynamic".equals(s)) {
            throw new GgufFormatException(a.where + ".activation_scheme must be 'static' or 'dynamic', got " + s);
        }
        return s;
    }

    private static int[] blockSize(Assembly a, int[] dflt) {
        Object v = a.q.get("weight_block_size");
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof List) || ((List<?>) v).size() != 2) {
            throw new GgufFormatException(a.where + ".weight_block_size must be [N, K], got " + v);
        }
        List<?> l = (List<?>) v;
        return new int[]{QuantArgs.toInt(l.get(0), a.where + ".weight_block_size[0]"),
                QuantArgs.toInt(l.get(1), a.where + ".weight_block_size[1]")};
    }
}
