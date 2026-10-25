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
    /** A method that stores float weights and quantizes at load in vLLM/SGLang; weights load as stored. */
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
