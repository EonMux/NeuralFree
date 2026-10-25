package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * How the modules of one quantization rule are stored: the {@link QuantFormat} layout and its weight, activation
 * and format-specific parameters, normalized from the checkpoint's configuration. Immutable; equal arguments mean
 * the modules can be rewritten as one batch. The package-private factories parse the argument dialects that
 * several methods share: compressed-tensors {@code QuantizationArgs}, Quark tensor specs and ModelOpt
 * {@code quant_algo} names.
 */
public final class QuantArgs {

    /** What one weight scale covers. */
    public enum Strategy { TENSOR, CHANNEL, GROUP, BLOCK, TENSOR_GROUP }

    /** The stored dtype of the weight scales; {@link #AUTO} means the scale tensor's own dtype. */
    public enum ScaleFormat { F32, BF16, F16, E8M0, E4M3, AUTO }

    /** fp_quant {@code transform_init}. */
    public enum FpQuantTransform { HADAMARD, IDENTITY, GSR }

    /** {@link #groupSize()} of per-channel quantization. */
    public static final int CHANNEL = -1;
    /** {@link #groupSize()} of ungrouped (per-tensor or block) quantization. */
    public static final int UNGROUPED = 0;

    private final QuantFormat format;
    private final int bits;
    private final int groupSize;
    private final boolean symmetric;
    private final int gptqVersion;
    private final boolean actOrder;
    private final Strategy strategy;
    private final int blockN;
    private final int blockK;
    private final ScaleFormat scaleFormat;
    private final ActQuantScheme actScheme;
    private final boolean actStatic;
    private final String actUnemulated;
    private final String outputActivations;
    private final int bnbBlocksize;
    private final Boolean bnbDoubleQuant;
    private final boolean quarkPackReorder;
    private final int mlxBits;
    private final int hqqAxis;
    private final int fpQuantHadamardGroup;
    private final FpQuantTransform fpQuantTransform;

    private QuantArgs(Builder b) {
        this.format = b.format;
        this.bits = b.bits;
        this.groupSize = b.groupSize;
        this.symmetric = b.symmetric;
        this.gptqVersion = b.gptqVersion;
        this.actOrder = b.actOrder;
        this.strategy = b.strategy;
        this.blockN = b.blockN;
        this.blockK = b.blockK;
        this.scaleFormat = b.scaleFormat;
        this.actScheme = b.actScheme;
        this.actStatic = b.actScheme != null && (b.actScheme.isStatic() || b.actScheme.needsGlobalScale());
        this.actUnemulated = b.actUnemulated;
        this.outputActivations = b.outputActivations;
        this.bnbBlocksize = b.bnbBlocksize;
        this.bnbDoubleQuant = b.bnbDoubleQuant;
        this.quarkPackReorder = b.quarkPackReorder;
        this.mlxBits = b.mlxBits;
        this.hqqAxis = b.hqqAxis;
        this.fpQuantHadamardGroup = b.fpQuantHadamardGroup;
        this.fpQuantTransform = b.fpQuantTransform;
    }

    /** A builder for {@code format} with that format's natural bit width and scale format. */
    public static Builder builder(QuantFormat format) {
        return new Builder(format);
    }

    /** A builder holding these arguments. */
    public Builder toBuilder() {
        return new Builder(this);
    }

    public QuantFormat format() { return format; }

    /** Weight bits per element. */
    public int bits() { return bits; }

    /** Elements per weight scale along K: positive for groups, {@link #CHANNEL}, or {@link #UNGROUPED}. */
    public int groupSize() { return groupSize; }

    public boolean symmetric() { return symmetric; }

    /** GPTQ zero-point convention: 1 (stored {@code z - 1}) or 2 (stored {@code z}). */
    public int gptqVersion() { return gptqVersion; }

    /** Whether activation reordering ({@code desc_act}/{@code actorder}) is declared. */
    public boolean actOrder() { return actOrder; }

    public Strategy strategy() { return strategy; }

    /** Block rows (N) of {@link Strategy#BLOCK} scales, else 0. */
    public int blockN() { return blockN; }

    /** Block columns (K) of {@link Strategy#BLOCK} scales, else 0. */
    public int blockK() { return blockK; }

    public ScaleFormat scaleFormat() { return scaleFormat; }

    /** The emulated input activation quantization, or null for weight-only modules. */
    public ActQuantScheme actScheme() { return actScheme; }

    /** Whether {@code input_scale} tensors are expected (static or global activation scales). */
    public boolean actStatic() { return actStatic; }

    /** A description of declared input activation quantization that no {@link ActQuantScheme} emulates, or null. */
    public String actUnemulated() { return actUnemulated; }

    /** A description of configured output activation quantization, or null. */
    public String outputActivations() { return outputActivations; }

    /** bitsandbytes block size, or 0 when it comes from the weight's quant_state. */
    public int bnbBlocksize() { return bnbBlocksize; }

    /** bitsandbytes double quantization, or null when it comes from the weight's quant_state. */
    public Boolean bnbDoubleQuant() { return bnbDoubleQuant; }

    /** Quark {@code pack_method: reorder} (AWQ nibble order). */
    public boolean quarkPackReorder() { return quarkPackReorder; }

    /** MLX bits, or 0 for other formats. */
    public int mlxBits() { return mlxBits; }

    /** HQQ quantization axis, or 0 for other formats. */
    public int hqqAxis() { return hqqAxis; }

    /** fp_quant Hadamard group size, or 0 for other formats. */
    public int fpQuantHadamardGroup() { return fpQuantHadamardGroup; }

    /** fp_quant {@code transform_init}, or null for other formats. */
    public FpQuantTransform fpQuantTransform() { return fpQuantTransform; }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof QuantArgs)) {
            return false;
        }
        QuantArgs a = (QuantArgs) o;
        return format == a.format && bits == a.bits && groupSize == a.groupSize && symmetric == a.symmetric
                && gptqVersion == a.gptqVersion && actOrder == a.actOrder && strategy == a.strategy
                && blockN == a.blockN && blockK == a.blockK && scaleFormat == a.scaleFormat
                && actScheme == a.actScheme && Objects.equals(actUnemulated, a.actUnemulated)
                && Objects.equals(outputActivations, a.outputActivations) && bnbBlocksize == a.bnbBlocksize
                && Objects.equals(bnbDoubleQuant, a.bnbDoubleQuant) && quarkPackReorder == a.quarkPackReorder
                && mlxBits == a.mlxBits && hqqAxis == a.hqqAxis && fpQuantHadamardGroup == a.fpQuantHadamardGroup
                && fpQuantTransform == a.fpQuantTransform;
    }

    @Override
    public int hashCode() {
        return Objects.hash(format, bits, groupSize, symmetric, gptqVersion, actOrder, strategy, blockN, blockK,
                scaleFormat, actScheme, actUnemulated, outputActivations, bnbBlocksize, bnbDoubleQuant,
                quarkPackReorder, mlxBits, hqqAxis, fpQuantHadamardGroup, fpQuantTransform);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder(format.name()).append(" b").append(bits).append(" g").append(groupSize)
                .append(symmetric ? " sym" : " asym").append(' ').append(strategy);
        if (strategy == Strategy.BLOCK) s.append(' ').append(blockN).append('x').append(blockK);
        if (scaleFormat != ScaleFormat.AUTO) s.append(" scale=").append(scaleFormat);
        if (format == QuantFormat.INT_GPTQ) s.append(" v").append(gptqVersion);
        if (actOrder) s.append(" act-order");
        if (actScheme != null) s.append(" act=").append(actScheme);
        if (actUnemulated != null) s.append(" act-unemulated=").append(actUnemulated);
        if (outputActivations != null) s.append(" out=").append(outputActivations);
        if (bnbBlocksize > 0) s.append(" blocksize=").append(bnbBlocksize);
        if (bnbDoubleQuant != null) s.append(" double-quant=").append(bnbDoubleQuant);
        if (quarkPackReorder) s.append(" reorder");
        if (hqqAxis != 0) s.append(" axis=").append(hqqAxis);
        if (fpQuantTransform != null) s.append(" hadamard=").append(fpQuantHadamardGroup).append('/')
                .append(fpQuantTransform);
        return s.toString();
    }

    // ------------------------------------------------------------------ shared dialects

    /**
     * compressed-tensors {@code QuantizationArgs} of one config group (CT quantization_args.py; the scheme
     * predicates of V compressed_tensors.py).
     *
     * @param format the group's {@code format}, else the config's, or null
     * @param where  the group as named in messages
     */
    static QuantArgs compressedTensors(Map<String, Object> weights, Map<String, Object> input,
                                       Map<String, Object> output, String format, String where) {
        String w = where + ".weights";
        int bits = intOr(weights, "num_bits", 8, w);
        String type = strOr(weights, "type", "int", w);
        Strategy strategy = ctStrategy(weights, w);
        int group = ctGroup(weights, strategy, w);
        ScaleFormat scale = scaleFormatOf(strOr(weights, "scale_dtype", null, w));
        boolean pack = "pack-quantized".equals(format);
        QuantFormat f;
        if ("int".equals(type)) {
            if (format != null && (format.startsWith("marlin") || format.contains("sparse") || format.contains("24"))) {
                throw new GgufFormatException(where + ": compressed-tensors format '" + format
                        + "' (sparse) is not supported");
            }
            if (pack || (format == null && bits < 8)) {
                f = QuantFormat.INT_PACKED_CT;
            } else if (bits == 8) {
                f = strategy == Strategy.BLOCK ? QuantFormat.INT8_BLOCK : QuantFormat.INT8_CHANNEL;
            } else if (bits == 4) {
                f = QuantFormat.INT4_IN_INT8;
            } else {
                throw unsupported(where, "int" + bits + " weights in format '" + format + "'");
            }
        } else if ("float".equals(type) && bits == 8) {
            boolean mx = "mxfp8-quantized".equals(format) || (group == 32 && scale == ScaleFormat.E8M0);
            f = mx ? QuantFormat.MXFP8 : QuantFormat.FP8;
        } else if ("float".equals(type) && bits == 4) {
            if (strategy == Strategy.TENSOR_GROUP || group == 16 || "nvfp4-pack-quantized".equals(format)) {
                f = QuantFormat.NVFP4_CT;
            } else if (group == 32 || "mxfp4-pack-quantized".equals(format)) {
                f = QuantFormat.MXFP4_CT;
            } else {
                throw unsupported(where, "float4 weights with group size " + group);
            }
        } else {
            throw unsupported(where, type + bits + " weights");
        }
        Object actorder = weights.get("actorder");
        Builder b = builder(f).bits(bits).symmetric(boolOr(weights, "symmetric", true, w))
                .actOrder(actorder != null && !Boolean.FALSE.equals(actorder));
        if (strategy == Strategy.BLOCK) {
            int[] block = ctBlock(weights, w);
            b.block(block[0], block[1]);
        } else {
            b.strategy(strategy).groupSize(group);
        }
        if (scale != ScaleFormat.AUTO) {
            b.scaleFormat(scale);
        }
        if (input != null) {
            ActQuantScheme s = ctActivation(input, where + ".input_activations");
            if (s != null) {
                b.actScheme(s);
            } else {
                b.actUnemulated("compressed-tensors input_activations " + describe(input));
            }
        }
        if (output != null) {
            b.outputActivations("compressed-tensors output_activations " + describe(output));
        }
        return b.build();
    }

    private static Strategy ctStrategy(Map<String, Object> a, String where) {
        String s = strOr(a, "strategy", null, where);
        if (s == null) {
            if (a.get("block_structure") != null) {
                return Strategy.BLOCK;
            }
            Object g = a.get("group_size");
            if (g == null) {
                return Strategy.TENSOR;
            }
            long group = HfConfig.toLong(where + ".group_size", g);
            if (group > 0) {
                return Strategy.GROUP;
            }
            if (group == -1) {
                return Strategy.CHANNEL;
            }
            throw unsupported(where, "group_size " + group);
        }
        switch (s) {
            case "tensor": return Strategy.TENSOR;
            case "channel": return Strategy.CHANNEL;
            case "group": return Strategy.GROUP;
            case "block": return Strategy.BLOCK;
            case "tensor_group": return Strategy.TENSOR_GROUP;
            default: throw unsupported(where, "weight strategy '" + s + "'");
        }
    }

    private static int ctGroup(Map<String, Object> a, Strategy strategy, String where) {
        switch (strategy) {
            case GROUP: case TENSOR_GROUP:
                int g = intOr(a, "group_size", 0, where);
                if (g <= 0) {
                    throw unsupported(where, strategy + " strategy without a positive group_size");
                }
                return g;
            case CHANNEL:
                return CHANNEL;
            default:
                return UNGROUPED;
        }
    }

    private static int[] ctBlock(Map<String, Object> a, String where) {
        Object v = a.get("block_structure");
        if (v instanceof String) {
            String[] parts = ((String) v).toLowerCase(Locale.ROOT).split("x");
            if (parts.length == 2) {
                try {
                    return new int[]{Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim())};
                } catch (NumberFormatException e) {
                    throw new GgufFormatException(where + ": block_structure '" + v + "' is not 'NxK'", e);
                }
            }
        } else if (v instanceof List && ((List<?>) v).size() == 2) {
            List<?> l = (List<?>) v;
            return new int[]{toInt(l.get(0), where + ".block_structure[0]"),
                    toInt(l.get(1), where + ".block_structure[1]")};
        }
        throw new GgufFormatException(where + ": block strategy needs block_structure [N, K], got " + v);
    }

    /** The emulated scheme of compressed-tensors input activations, or null when none emulates them. */
    private static ActQuantScheme ctActivation(Map<String, Object> a, String where) {
        int bits = intOr(a, "num_bits", 8, where);
        String type = strOr(a, "type", "int", where);
        Object dyn = a.get("dynamic");
        boolean dynamic = Boolean.TRUE.equals(dyn) || "local".equals(dyn);
        String strategy = strOr(a, "strategy", null, where);
        int group = a.get("group_size") == null ? 0 : intOr(a, "group_size", 0, where);
        boolean perRow = strategy == null ? group == 0 : strategy.equals("token") || strategy.equals("tensor");
        if (bits == 8 && "int".equals(type)) {
            if (dynamic) {
                return perRow ? ActQuantScheme.INT8_TOKEN : group == 128 ? ActQuantScheme.INT8_GROUP128 : null;
            }
            return perRow ? (boolOr(a, "symmetric", true, where) ? ActQuantScheme.INT8_STATIC
                    : ActQuantScheme.INT8_STATIC_ASYM) : null;
        }
        if (bits == 8 && "float".equals(type)) {
            if (dynamic) {
                return perRow ? ActQuantScheme.FP8_TOKEN : group == 128 ? ActQuantScheme.FP8_GROUP128
                        : group == 32 ? ActQuantScheme.MXFP8 : null;
            }
            return perRow ? ActQuantScheme.FP8_STATIC : null;
        }
        if (bits == 4 && "float".equals(type)) {
            return group == 16 ? ActQuantScheme.NVFP4 : group == 32 ? ActQuantScheme.MXFP4 : null;
        }
        return null;
    }

    /**
     * One Quark layer config ({@code weight}, {@code input_tensors}, {@code output_tensors}), as V quark.py
     * dispatches it. A single-element spec list is unwrapped; a longer one (two-level weight specs) is rejected.
     * FP8 {@code per_block} weights with {@code [R, 32]} blocks and {@code float8_e8m0fnu} scales are the 2-D MXFP8
     * spelling (V ocp_mx_utils.ocp_mx_weight_dtype_and_rows): one E8M0 scale per R rows and 32 columns.
     */
    static QuantArgs quark(Map<String, Object> layer, String packMethod, String where) {
        Map<String, Object> w = quarkSpec(layer.get("weight"), where + ".weight");
        if (w == null) {
            throw new GgufFormatException(where + ": Quark layer config has no 'weight' spec");
        }
        String ww = where + ".weight";
        String dtype = strOr(w, "dtype", null, ww);
        String qscheme = strOr(w, "qscheme", null, ww);
        boolean ocpMx = "per_group".equals(qscheme) && intOr(w, "group_size", 0, ww) == 32
                && "e8m0".equals(strOr(w, "scale_format", null, ww));
        Builder b;
        if ("int4".equals(dtype) || "uint4".equals(dtype)) {
            b = builder(QuantFormat.INT_QUARK).bits(4).symmetric(boolOr(w, "symmetric", "int4".equals(dtype), ww))
                    .quarkPackReorder("reorder".equals(packMethod));
            b.group("per_group".equals(qscheme) ? intOr(w, "group_size", 0, ww) : CHANNEL);
        } else if ("int8".equals(dtype)) {
            b = builder(QuantFormat.INT8_CHANNEL).symmetric(boolOr(w, "symmetric", true, ww))
                    .group("per_channel".equals(qscheme) ? CHANNEL : UNGROUPED);
        } else if ("fp8_e4m3".equals(dtype) && ocpMx) {
            b = builder(QuantFormat.MXFP8);
        } else if ("fp8_e4m3".equals(dtype)) {
            boolean e8m0 = "float8_e8m0fnu".equals(strOr(w, "scale_type", null, ww));
            int[] block = "per_block".equals(qscheme) ? quarkBlock(w, ww) : null;
            boolean mx2d = block != null && block[1] == 32 && e8m0 && boolOr(w, "symmetric", false, ww);
            b = builder(mx2d ? QuantFormat.MXFP8 : QuantFormat.FP8);
            if (mx2d && block[0] > 1) {
                b.block(block[0], block[1]);
            } else if (block != null && !mx2d) {
                b.block(block[0], block[1]).scaleFormat(e8m0 ? ScaleFormat.E8M0 : ScaleFormat.AUTO);
            } else if (block == null) {
                b.group("per_channel".equals(qscheme) ? CHANNEL : UNGROUPED);
            }
        } else if ("fp4".equals(dtype) && ocpMx) {
            b = builder(QuantFormat.MXFP4_CT);
        } else if (("fp6_e2m3".equals(dtype) || "fp6_e3m2".equals(dtype)) && ocpMx) {
            b = builder(QuantFormat.MXFP6);
        } else {
            throw unsupported(where, "Quark weight dtype '" + dtype + "' with qscheme '" + qscheme + "'");
        }
        if (qscheme != null && !qscheme.startsWith("per_")) {
            throw unsupported(where, "Quark qscheme '" + qscheme + "'");
        }
        Map<String, Object> in = quarkSpec(layer.get("input_tensors"), where + ".input_tensors");
        if (in != null) {
            ActQuantScheme s = quarkActivation(in, where + ".input_tensors");
            if (s != null) {
                b.actScheme(s);
            } else {
                b.actUnemulated("Quark input_tensors " + describe(in));
            }
        }
        Map<String, Object> out = quarkSpec(layer.get("output_tensors"), where + ".output_tensors");
        if (out != null) {
            b.outputActivations("Quark output_tensors " + describe(out));
        }
        return b.build();
    }

    private static Map<String, Object> quarkSpec(Object v, String where) {
        if (v instanceof List) {
            List<?> l = (List<?>) v;
            if (l.size() > 1) {
                throw new GgufFormatException("unsupported quant_method 'quark': " + where
                        + " is a two-level spec list; Quark multi-entry weight specs are not supported");
            }
            v = l.isEmpty() ? null : l.get(0);
        }
        if (v != null && !(v instanceof Map)) {
            throw new GgufFormatException(where + ": expected an object, got " + HfConfig.typeName(v));
        }
        return v == null ? null : HfConfig.asObject(v);
    }

    private static int[] quarkBlock(Map<String, Object> w, String where) {
        Object v = w.get("block_size");
        if (v instanceof List && ((List<?>) v).size() == 2) {
            List<?> l = (List<?>) v;
            return new int[]{toInt(l.get(0), where + ".block_size[0]"), toInt(l.get(1), where + ".block_size[1]")};
        }
        throw new GgufFormatException(where + ": per_block quantization needs block_size [N, K], got " + v);
    }

    private static ActQuantScheme quarkActivation(Map<String, Object> a, String where) {
        String dtype = strOr(a, "dtype", null, where);
        String qscheme = strOr(a, "qscheme", null, where);
        boolean dynamic = boolOr(a, "is_dynamic", false, where);
        boolean mx = "per_group".equals(qscheme) && a.get("group_size") != null
                && intOr(a, "group_size", 0, where) == 32;
        if ("fp8_e4m3".equals(dtype)) {
            if (mx) {
                return ActQuantScheme.MXFP8;
            }
            if (dynamic) {
                return "per_tensor".equals(qscheme) || "per_token".equals(qscheme) || "per_channel".equals(qscheme)
                        ? ActQuantScheme.FP8_TOKEN : null;
            }
            return "per_tensor".equals(qscheme) ? ActQuantScheme.FP8_STATIC : null;
        }
        if ("int8".equals(dtype)) {
            if (dynamic) {
                return "per_tensor".equals(qscheme) || "per_token".equals(qscheme) || "per_channel".equals(qscheme)
                        ? ActQuantScheme.INT8_TOKEN : null;
            }
            if (!"per_tensor".equals(qscheme)) {
                return null;
            }
            return boolOr(a, "symmetric", true, where) ? ActQuantScheme.INT8_STATIC : ActQuantScheme.INT8_STATIC_ASYM;
        }
        if ("fp4".equals(dtype) && mx) {
            return ActQuantScheme.MXFP4;
        }
        return null;
    }

    /** The weight scale format of a {@code scale_dtype} string; AUTO when absent or unrecognized. */
    static ScaleFormat scaleFormatOf(String dtype) {
        if (dtype == null) {
            return ScaleFormat.AUTO;
        }
        String d = dtype.toLowerCase(Locale.ROOT);
        if (d.contains("uint8") || d.contains("e8m0")) return ScaleFormat.E8M0;
        if (d.contains("e4m3")) return ScaleFormat.E4M3;
        if (d.contains("bfloat16")) return ScaleFormat.BF16;
        if (d.contains("float16") || d.equals("half")) return ScaleFormat.F16;
        if (d.contains("float32") || d.equals("float")) return ScaleFormat.F32;
        return ScaleFormat.AUTO;
    }

    private static GgufFormatException unsupported(String where, String what) {
        return new GgufFormatException(where + ": unsupported " + what);
    }

    private static String describe(Map<String, Object> a) {
        StringBuilder s = new StringBuilder("{");
        for (String k : new String[]{"type", "dtype", "num_bits", "strategy", "qscheme", "group_size", "dynamic",
                "is_dynamic", "symmetric"}) {
            if (a.get(k) != null) {
                s.append(s.length() > 1 ? ", " : "").append(k).append('=').append(a.get(k));
            }
        }
        return s.append('}').toString();
    }

    static String strOr(Map<String, Object> m, String key, String dflt, String where) {
        Object v = m.get(key);
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof String)) {
            throw new GgufFormatException(where + "." + key + " must be a string, got " + HfConfig.typeName(v));
        }
        return (String) v;
    }

    static int intOr(Map<String, Object> m, String key, int dflt, String where) {
        Object v = m.get(key);
        return v == null ? dflt : toInt(v, where + "." + key);
    }

    static boolean boolOr(Map<String, Object> m, String key, boolean dflt, String where) {
        Object v = m.get(key);
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof Boolean)) {
            throw new GgufFormatException(where + "." + key + " must be a boolean, got " + HfConfig.typeName(v));
        }
        return (Boolean) v;
    }

    static int toInt(Object v, String where) {
        long l = HfConfig.toLong(where, v);
        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
            throw new GgufFormatException(where + " value " + l + " is out of the 32-bit integer range");
        }
        return (int) l;
    }

    /** Mutable builder; {@link #build()} validates and copies. */
    public static final class Builder {
        private final QuantFormat format;
        private int bits;
        private int groupSize = UNGROUPED;
        private boolean symmetric = true;
        private int gptqVersion = 1;
        private boolean actOrder;
        private Strategy strategy = Strategy.TENSOR;
        private int blockN;
        private int blockK;
        private ScaleFormat scaleFormat = ScaleFormat.AUTO;
        private ActQuantScheme actScheme;
        private String actUnemulated;
        private String outputActivations;
        private int bnbBlocksize;
        private Boolean bnbDoubleQuant;
        private boolean quarkPackReorder;
        private int mlxBits;
        private int hqqAxis;
        private int fpQuantHadamardGroup;
        private FpQuantTransform fpQuantTransform;

        private Builder(QuantFormat format) {
            this.format = Objects.requireNonNull(format, "format");
            switch (format) {
                case FP8: case MXFP8: case INT8_CHANNEL: case INT8_BLOCK: case INT8_BNB:
                    bits = 8;
                    break;
                case MXFP6:
                    bits = 6;
                    break;
                case BITNET:
                    bits = 2;
                    break;
                case NVFP4_CT: case NVFP4_MODELOPT: case FPQUANT_NVFP4:
                    bits = 4;
                    groupSize = 16;
                    strategy = Strategy.GROUP;
                    scaleFormat = ScaleFormat.E4M3;
                    break;
                case MXFP4_CT: case MXFP4_GPTOSS: case MXFP4_DSV4: case FPQUANT_MXFP4:
                    bits = 4;
                    groupSize = 32;
                    strategy = Strategy.GROUP;
                    scaleFormat = ScaleFormat.E8M0;
                    break;
                default:
                    bits = 4;
                    break;
            }
            if (format == QuantFormat.MXFP8 || format == QuantFormat.MXFP6) {
                groupSize = 32;
                strategy = Strategy.GROUP;
                scaleFormat = ScaleFormat.E8M0;
            }
        }

        private Builder(QuantArgs a) {
            this.format = a.format;
            this.bits = a.bits;
            this.groupSize = a.groupSize;
            this.symmetric = a.symmetric;
            this.gptqVersion = a.gptqVersion;
            this.actOrder = a.actOrder;
            this.strategy = a.strategy;
            this.blockN = a.blockN;
            this.blockK = a.blockK;
            this.scaleFormat = a.scaleFormat;
            this.actScheme = a.actScheme;
            this.actUnemulated = a.actUnemulated;
            this.outputActivations = a.outputActivations;
            this.bnbBlocksize = a.bnbBlocksize;
            this.bnbDoubleQuant = a.bnbDoubleQuant;
            this.quarkPackReorder = a.quarkPackReorder;
            this.mlxBits = a.mlxBits;
            this.hqqAxis = a.hqqAxis;
            this.fpQuantHadamardGroup = a.fpQuantHadamardGroup;
            this.fpQuantTransform = a.fpQuantTransform;
        }

        public Builder bits(int v) { this.bits = v; return this; }

        /** Sets the group size and the matching strategy (GROUP, CHANNEL or TENSOR). */
        public Builder group(int v) {
            this.groupSize = v;
            this.strategy = v > 0 ? Strategy.GROUP : v == CHANNEL ? Strategy.CHANNEL : Strategy.TENSOR;
            return this;
        }

        public Builder groupSize(int v) { this.groupSize = v; return this; }

        public Builder symmetric(boolean v) { this.symmetric = v; return this; }

        public Builder gptqVersion(int v) { this.gptqVersion = v; return this; }

        public Builder actOrder(boolean v) { this.actOrder = v; return this; }

        public Builder strategy(Strategy v) { this.strategy = Objects.requireNonNull(v, "strategy"); return this; }

        /** Sets {@link Strategy#BLOCK} with {@code n x k} blocks. */
        public Builder block(int n, int k) {
            this.strategy = Strategy.BLOCK;
            this.groupSize = UNGROUPED;
            this.blockN = n;
            this.blockK = k;
            return this;
        }

        public Builder scaleFormat(ScaleFormat v) { this.scaleFormat = Objects.requireNonNull(v, "scaleFormat"); return this; }

        public Builder actScheme(ActQuantScheme v) { this.actScheme = v; return this; }

        public Builder actUnemulated(String v) { this.actUnemulated = v; return this; }

        public Builder outputActivations(String v) { this.outputActivations = v; return this; }

        public Builder bnbBlocksize(int v) { this.bnbBlocksize = v; return this; }

        public Builder bnbDoubleQuant(Boolean v) { this.bnbDoubleQuant = v; return this; }

        public Builder quarkPackReorder(boolean v) { this.quarkPackReorder = v; return this; }

        public Builder mlxBits(int v) { this.mlxBits = v; return this; }

        public Builder hqqAxis(int v) { this.hqqAxis = v; return this; }

        public Builder fpQuantHadamardGroup(int v) { this.fpQuantHadamardGroup = v; return this; }

        public Builder fpQuantTransform(FpQuantTransform v) { this.fpQuantTransform = v; return this; }

        /**
         * The arguments.
         *
         * @throws GgufFormatException when a value is out of its range
         */
        public QuantArgs build() {
            check(bits >= 1 && bits <= 16, "bits " + bits + " is outside [1, 16]");
            check(groupSize >= CHANNEL && groupSize <= HfConfig.MAX_WIDTH, "group size " + groupSize
                    + " is outside [-1, " + HfConfig.MAX_WIDTH + "]");
            check(gptqVersion == 1 || gptqVersion == 2, "GPTQ version " + gptqVersion + " is not 1 or 2");
            check(strategy != Strategy.BLOCK || (blockN > 0 && blockK > 0 && blockN <= HfConfig.MAX_WIDTH
                    && blockK <= HfConfig.MAX_WIDTH), "block " + blockN + "x" + blockK + " is not positive");
            check(strategy == Strategy.BLOCK || (blockN == 0 && blockK == 0), "block sizes need the BLOCK strategy");
            check((strategy != Strategy.GROUP && strategy != Strategy.TENSOR_GROUP) || groupSize > 0,
                    strategy + " quantization needs a positive group size");
            check(bnbBlocksize >= 0 && mlxBits >= 0 && fpQuantHadamardGroup >= 0, "negative format parameter");
            return new QuantArgs(this);
        }

        private void check(boolean ok, String message) {
            if (!ok) {
                throw new GgufFormatException(format + " quantization: " + message);
            }
        }
    }
}
