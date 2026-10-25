package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Geometry;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Module;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Notes;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Produced;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Rewrite;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.RowWriter;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Rewrites the 4-bit and 6-bit float formats into the §1.5 targets:
 * <ul>
 *   <li>NVFP4 (compressed-tensors {@code weight_packed}/{@code weight_global_scale}, ModelOpt and petit
 *       {@code weight}/{@code weight_scale_2}): NVFP4 blocks when {@code K % 64 == 0}, with the global multiplier as
 *       the output scale (compressed-tensors stores its divisor) and the activation global scale as the input
 *       multiplier; otherwise the float fallback.</li>
 *   <li>MXFP4 (compressed-tensors and Quark {@code weight_packed}/{@code weight} with E8M0 {@code weight_scale},
 *       DeepSeek-V4 experts with {@code scale}, gpt-oss {@code _blocks}/{@code _scales}): MXFP4 blocks, the E8M0
 *       byte copied.</li>
 *   <li>MXFP6 (Quark OCP-MX E2M3 or E3M2, four 6-bit codes per three bytes): BF16, exact while every block
 *       exponent keeps the values inside BF16's range; a larger exponent is rejected naming the tensor and block.</li>
 * </ul>
 * Packed E2M1 codes are stored low nibble first in every source format (gpt-oss included).
 */
public final class Fp4Scheme {

    private static final Logger LOG = LogManager.getLogger(Fp4Scheme.class);

    private static final int MX = 32;
    private static final int NV = 64;
    private static final int NV_SUB = 16;
    private static final int E2M3_MAX_EXP = 252;
    private static final int E3M2_MAX_EXP = 250;

    private Fp4Scheme() {
    }

    /** True for the NVFP4, MXFP4 and MXFP6 formats. */
    public static boolean handles(QuantFormat f) {
        switch (f) {
            case NVFP4_CT: case NVFP4_MODELOPT: case MXFP4_CT: case MXFP4_DSV4: case MXFP4_GPTOSS: case MXFP6:
                return true;
            default:
                return false;
        }
    }

    /**
     * True when modules quantized with {@code args} are ModelOpt NVFP4 weights with 16-bit activations
     * (W4A16_NVFP4), which the metadata records as {@code general.tensor_extra.prec_a4 = false}.
     */
    public static boolean weightOnlyNvfp4(QuantArgs args) {
        return args.format() == QuantFormat.NVFP4_MODELOPT && args.actScheme() == null;
    }

    /**
     * Rewrites every module of {@code modules} (HF module prefixes; for gpt-oss the fused expert parameter such as
     * {@code model.layers.0.mlp.experts.gate_up_proj}) in {@code in}.
     *
     * @throws GgufFormatException when a module's tensors are missing, have the wrong dtype or inconsistent shapes
     * @throws IOException         when a tensor cannot be read
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(modules, "modules");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(o, "o");
        if (!handles(args.format())) {
            throw new IllegalArgumentException("Fp4Scheme does not handle " + args.format());
        }
        Ctx ctx = new Ctx(args, cfg, in, o);
        for (String module : modules) {
            Module m = new Module(in, module);
            switch (args.format()) {
                case NVFP4_CT: nvfp4(ctx, m, true); break;
                case NVFP4_MODELOPT: nvfp4(ctx, m, false); break;
                case MXFP4_CT: mxfp4(ctx, m, "weight_scale", "weight_packed", "weight"); break;
                case MXFP4_DSV4: mxfp4(ctx, m, "scale", "weight"); break;
                case MXFP4_GPTOSS: gptOss(ctx, m); break;
                default: mxfp6(ctx, m); break;
            }
        }
        if (args.actUnemulated() != null && !modules.isEmpty()) {
            LOG.warn("{} are not emulated; {} {} module(s) load weight-only", args.actUnemulated(), modules.size(),
                    args.format());
        }
        ctx.notes.log(LOG);
        return ctx.rw.apply(in);
    }

    /** The state of one batch. */
    private static final class Ctx {
        final QuantArgs args;
        final HfConfig cfg;
        final HfTensorIndex in;
        final ModelSourceOptions o;
        final Rewrite rw = new Rewrite();
        final Notes notes = new Notes();
        private Boolean fp6E2m3;

        Ctx(QuantArgs args, HfConfig cfg, HfTensorIndex in, ModelSourceOptions o) {
            this.args = args;
            this.cfg = cfg;
            this.in = in;
            this.o = o;
        }

        boolean fp6E2m3() throws IOException {
            if (fp6E2m3 == null) {
                fp6E2m3 = quarkFp6E2m3(HfQuantConfig.of(cfg).view());
            }
            return fp6E2m3;
        }

        String describe(Module m, Object target) {
            return Fp4Packing.describe(m.prefix(), args.format(), args.bits(), args.groupSize(), target);
        }
    }

    /** Reads {@code count} rows of a packed tensor as bytes. */
    @FunctionalInterface
    interface Bytes {
        byte[] read(long first, int count) throws IOException;
    }

    private static boolean isFloat(StDType t) {
        return t == StDType.F32 || t == StDType.F16 || t == StDType.BF16;
    }

    /**
     * The packed weight of suffix {@code wSuffix}, or null when the module is stored unquantized (a float weight and
     * no scale), which loads as stored.
     */
    private static RowSource packedWeight(Ctx ctx, Module m, String wSuffix, String sSuffix) {
        RowSource w = m.get(wSuffix);
        if (!m.has(sSuffix)) {
            RowSource plain = w != null ? w : m.get("weight");
            if (plain != null && isFloat(Fp4Packing.dtype(plain, m.prefix()))) {
                ctx.notes.info(ctx.args.format() + " modules stored unquantized load as stored; --quant-online"
                        + " quantizes them", m.prefix());
                return null;
            }
            throw new GgufFormatException(ctx.args.format() + " module " + m.prefix() + " has no tensor "
                    + m.prefix() + "." + sSuffix);
        }
        if (w == null) {
            throw new GgufFormatException(ctx.args.format() + " module " + m.prefix() + " has no tensor "
                    + m.prefix() + "." + wSuffix);
        }
        StDType t = Fp4Packing.dtype(w, m.name(wSuffix));
        if (t != StDType.U8 && t != StDType.I8) {
            throw new GgufFormatException(ctx.args.format() + " weight " + m.name(wSuffix) + " has dtype "
                    + t.headerName() + "; expected packed U8");
        }
        return w;
    }

    private static void checkScaleDtype(RowSource s, String sName, StDType a, StDType b) {
        StDType t = Fp4Packing.dtype(s, sName);
        if (t != a && t != b) {
            throw new GgufFormatException("scale " + sName + " has dtype " + t.headerName() + "; expected "
                    + a.headerName() + " or " + b.headerName());
        }
    }

    // ------------------------------------------------------------------ NVFP4

    private static void nvfp4(Ctx ctx, Module m, boolean ct) throws IOException {
        String wSuffix = ct ? "weight_packed" : "weight";
        RowSource w = packedWeight(ctx, m, wSuffix, "weight_scale");
        if (w == null) {
            return;
        }
        if (ctx.args.groupSize() != NV_SUB) {
            throw new GgufFormatException("NVFP4 module " + m.prefix() + " has group size " + ctx.args.groupSize()
                    + "; NVFP4 scales cover 16 elements");
        }
        String wName = m.name(wSuffix);
        Geometry g = Geometry.packed(w.shape(), 1, 2, wName);
        if (g.k % NV_SUB != 0) {
            throw new GgufFormatException("NVFP4 weight " + wName + " has K = " + g.k + ", not a multiple of 16");
        }
        String sName = m.name("weight_scale");
        RowSource s = ctx.in.get(sName);
        checkScaleDtype(s, sName, StDType.F8_E4M3, StDType.U8);
        g.checkRows(s, sName, g.k / NV_SUB);
        String globalSuffix = ct ? "weight_global_scale" : "weight_scale_2";
        RowSource global = m.require(globalSuffix);
        String gName = m.name(globalSuffix);
        float[] stored = Fp4Packing.readPerExpert(global, gName, g.experts);
        float[] mult = ct ? Fp4Packing.reciprocal(stored, gName) : stored;
        Bytes packed = (f, c) -> Fp4Packing.readRows(w, f, c);
        Bytes scales = (f, c) -> Fp4Packing.readRows(s, f, c);
        boolean rowLocal = w.rowLocal() && s.rowLocal();
        StDType fb = Fp4Packing.fallbackType(StDType.F8_E4M3, ctx.o);
        Produced src;
        if (g.k % NV == 0) {
            src = Produced.of(ctx.describe(m, GGMLType.NVFP4), ElementFormat.of(GGMLType.NVFP4), g.shape, rowLocal,
                    nvfp4Writer(g, packed, scales, sName)).withOutputScale(mult, () -> {
                        LOG.warn("{}: the NVFP4 global scale has no sibling slot; the weight loads as {}", m.prefix(),
                                fb);
                        return nvfp4Floats(ctx.describe(m, fb), g, packed, scales, sName, mult, fb, rowLocal);
                    });
        } else {
            ctx.notes.fallback("NVFP4 weights with K % 64 != 0 load as the float fallback", m.prefix());
            src = nvfp4Floats(ctx.describe(m, fb), g, packed, scales, sName, mult, fb, rowLocal);
        }
        String out = m.outputName(wSuffix);
        ctx.rw.put(out, Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, ct,
                ct ? "input_global_scale" : "input_scale"));
        if (!wName.equals(out)) {
            ctx.rw.consume(wName);
        }
        ctx.rw.consume(sName);
        ctx.rw.consume(gName);
    }

    /** NVFP4 blocks: four UE4M3 sub-scales and 64 codes per block. */
    static RowWriter nvfp4Writer(Geometry g, Bytes packed, Bytes scales, String sName) {
        final int kb = g.k / 2;
        final int sc = g.k / NV_SUB;
        return (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] p = packed.read(first + done, c);
                byte[] s = scales.read(first + done, c);
                for (int i = 0; i < c; i++) {
                    long row = first + done + i;
                    for (int b = 0; b < g.k / NV; b++) {
                        int sOff = i * sc + 4 * b;
                        for (int sub = 0; sub < 4; sub++) {
                            Fp4Packing.checkE4m3Scale(s[sOff + sub], sName, row, 4L * b + sub);
                        }
                        Fp4Packing.putNvfp4(dst, s, sOff, p, i * kb + 32 * b);
                    }
                }
            }
        };
    }

    /** The dequantized NVFP4 weight {@code e2m1 * e4m3 * mult[expert]} in {@code type}. */
    static Produced nvfp4Floats(String describe, Geometry g, Bytes packed, Bytes scales, String sName,
                                float[] mult, StDType type, boolean rowLocal) {
        final int kb = g.k / 2;
        final int sc = g.k / NV_SUB;
        return Fp4Packing.floats(describe, g.shape, type, rowLocal, (first, count, out) -> {
            byte[] p = packed.read(first, count);
            byte[] s = scales.read(first, count);
            for (int i = 0; i < count; i++) {
                long row = first + i;
                float gm = mult.length == 1 ? mult[0] : mult[(int) (row / g.n)];
                for (int col = 0; col < g.k; col++) {
                    int code = s[i * sc + col / NV_SUB];
                    if (col % NV_SUB == 0) {
                        Fp4Packing.checkE4m3Scale(code, sName, row, col / NV_SUB);
                    }
                    out[i * g.k + col] = FpBlockDecoders.e2m1ToFp32(Fp4Packing.lowFirst(p, i * kb, col))
                            * FpBlockDecoders.e4m3ToFp32(code) * gm;
                }
            }
        });
    }

    // ------------------------------------------------------------------ MXFP4

    private static void mxfp4(Ctx ctx, Module m, String sSuffix, String... wSuffixes) throws IOException {
        String wSuffix = m.first(wSuffixes);
        RowSource w = packedWeight(ctx, m, wSuffix == null ? wSuffixes[0] : wSuffix, sSuffix);
        if (w == null) {
            return;
        }
        String wName = m.name(wSuffix);
        Geometry g = Geometry.packed(w.shape(), 1, 2, wName);
        if (g.k % MX != 0) {
            throw new GgufFormatException("MXFP4 weight " + wName + " has K = " + g.k + ", not a multiple of 32");
        }
        String sName = m.name(sSuffix);
        RowSource s = ctx.in.get(sName);
        checkScaleDtype(s, sName, StDType.U8, StDType.F8_E8M0);
        g.checkRows(s, sName, g.k / MX);
        Produced src = Produced.of(ctx.describe(m, GGMLType.MXFP4), ElementFormat.of(GGMLType.MXFP4), g.shape,
                w.rowLocal() && s.rowLocal(), mxfp4Writer(g, (f, c) -> Fp4Packing.readRows(w, f, c),
                        (f, c) -> Fp4Packing.readRows(s, f, c), sName));
        String out = m.outputName(wSuffix);
        ctx.rw.put(out, Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, false, "input_scale"));
        if (!wName.equals(out)) {
            ctx.rw.consume(wName);
        }
        ctx.rw.consume(sName);
    }

    /** gpt-oss {@code <p>_blocks [E, R, K/32, 16]} and {@code <p>_scales [E, R, K/32]} into MXFP4 {@code <p>}. */
    private static void gptOss(Ctx ctx, Module m) throws IOException {
        if (!m.has("blocks")) {
            if (ctx.in.has(m.prefix()) || m.has("weight")) {
                ctx.notes.info("MXFP4 expert parameters stored unquantized load as stored", m.prefix());
                return;
            }
            throw new GgufFormatException("gpt-oss MXFP4 parameter " + m.prefix() + " has no tensor " + m.prefix()
                    + "_blocks");
        }
        String bName = m.name("blocks");
        RowSource blocks = ctx.in.get(bName);
        RowSource scales = m.require("scales");
        String sName = m.name("scales");
        if (Fp4Packing.dtype(blocks, bName) != StDType.U8) {
            throw new GgufFormatException("gpt-oss MXFP4 blocks " + bName + " have dtype "
                    + Fp4Packing.dtype(blocks, bName).headerName() + "; expected packed U8");
        }
        checkScaleDtype(scales, sName, StDType.U8, StDType.F8_E8M0);
        long[] bs = blocks.shape();
        if (bs.length < 3 || bs[bs.length - 1] != 16) {
            throw new GgufFormatException("gpt-oss MXFP4 blocks " + bName + " have shape " + Arrays.toString(bs)
                    + "; expected [E, R, K/32, 16]");
        }
        long[] logical = Arrays.copyOf(bs, bs.length - 1);
        long groups = logical[logical.length - 1];
        logical[logical.length - 1] = Math.multiplyExact(groups, MX);
        Geometry g = Geometry.of(logical, bName);
        g.checkRows(scales, sName, groups);
        final int perRow = (int) groups;
        Produced src = Produced.of(ctx.describe(m, GGMLType.MXFP4), ElementFormat.of(GGMLType.MXFP4), g.shape,
                blocks.rowLocal() && scales.rowLocal(), mxfp4Writer(g,
                        (f, c) -> Fp4Packing.readRows(blocks, f * perRow, Math.multiplyExact(c, perRow)),
                        (f, c) -> Fp4Packing.readRows(scales, f, c), sName));
        ctx.rw.put(m.outputName("blocks"), Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, false,
                "input_scale"));
        ctx.rw.consume(bName);
        ctx.rw.consume(sName);
    }

    /** MXFP4 blocks from codes packed low nibble first, {@code K/2} bytes per row, and one E8M0 byte per 32. */
    static RowWriter mxfp4Writer(Geometry g, Bytes packed, Bytes scales, String sName) {
        final int kb = g.k / 2;
        final int sc = g.k / MX;
        return (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] p = packed.read(first + done, c);
                byte[] s = scales.read(first + done, c);
                for (int i = 0; i < c; i++) {
                    long row = first + done + i;
                    for (int b = 0; b < sc; b++) {
                        int e = Fp4Packing.checkE8m0(s[i * sc + b], sName, row, b);
                        Fp4Packing.putMxfp4(dst, e, p, i * kb + 16 * b);
                    }
                }
            }
        };
    }

    /** The dequantized MXFP4 weight {@code e2m1 * 2^(e-127) * mult[expert]} in {@code type}. */
    static Produced mxfp4Floats(String describe, Geometry g, Bytes packed, Bytes scales, String sName,
                                float[] mult, StDType type, boolean rowLocal) {
        final int kb = g.k / 2;
        final int sc = g.k / MX;
        return Fp4Packing.floats(describe, g.shape, type, rowLocal, (first, count, out) -> {
            byte[] p = packed.read(first, count);
            byte[] s = scales.read(first, count);
            for (int i = 0; i < count; i++) {
                long row = first + i;
                float gm = mult.length == 1 ? mult[0] : mult[(int) (row / g.n)];
                for (int b = 0; b < sc; b++) {
                    float scale = FpBlockDecoders.e8m0ToFp32(Fp4Packing.checkE8m0(s[i * sc + b], sName, row, b));
                    for (int j = 0; j < MX; j++) {
                        int col = b * MX + j;
                        out[i * g.k + col] = FpBlockDecoders.e2m1ToFp32(Fp4Packing.lowFirst(p, i * kb, col)) * scale
                                * gm;
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ MXFP6

    private static void mxfp6(Ctx ctx, Module m) throws IOException {
        RowSource w = packedWeight(ctx, m, "weight", "weight_scale");
        if (w == null) {
            return;
        }
        String wName = m.name("weight");
        if (Fp4Packing.dtype(w, wName) != StDType.U8) {
            throw new GgufFormatException("MXFP6 weight " + wName + " must be packed U8");
        }
        Geometry g = Geometry.packed(w.shape(), 3, 4, wName);
        if (g.k % MX != 0) {
            throw new GgufFormatException("MXFP6 weight " + wName + " has K = " + g.k + ", not a multiple of 32");
        }
        String sName = m.name("weight_scale");
        RowSource s = ctx.in.get(sName);
        checkScaleDtype(s, sName, StDType.U8, StDType.F8_E8M0);
        g.checkRows(s, sName, g.k / MX);
        final boolean e2m3 = ctx.fp6E2m3();
        final int maxExp = e2m3 ? E2M3_MAX_EXP : E3M2_MAX_EXP;
        final int kb = g.k / 4 * 3;
        final int sc = g.k / MX;
        Produced src = Fp4Packing.floats(ctx.describe(m, StDType.BF16), g.shape, StDType.BF16,
                w.rowLocal() && s.rowLocal(), (first, count, out) -> {
                    byte[] p = Fp4Packing.readRows(w, first, count);
                    byte[] e = Fp4Packing.readRows(s, first, count);
                    for (int i = 0; i < count; i++) {
                        long row = first + i;
                        for (int b = 0; b < sc; b++) {
                            int exp = Fp4Packing.checkE8m0(e[i * sc + b], sName, row, b);
                            if (exp > maxExp) {
                                throw new IOException("tensor " + sName + " row " + row + " block " + b
                                        + ": exponent " + exp + " overflows BF16 for fp6_" + (e2m3 ? "e2m3" : "e3m2")
                                        + " (at most " + maxExp + ")");
                            }
                            int base = i * kb + b * 24;
                            for (int q = 0; q < 8; q++) {
                                int word = (p[base + 3 * q] & 0xFF) | (p[base + 3 * q + 1] & 0xFF) << 8
                                        | (p[base + 3 * q + 2] & 0xFF) << 16;
                                for (int j = 0; j < 4; j++) {
                                    int code = (word >>> (6 * j)) & 0x3F;
                                    float v = e2m3 ? FpBlockDecoders.e2m3ToFp32(code)
                                            : FpBlockDecoders.e3m2ToFp32(code);
                                    out[i * g.k + b * MX + 4 * q + j] = Math.scalb(v, exp - 127);
                                }
                            }
                        }
                    }
                });
        String out = m.outputName("weight");
        ctx.rw.put(out, Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, false, "input_scale"));
        if (!wName.equals(out)) {
            ctx.rw.consume(wName);
        }
        ctx.rw.consume(sName);
    }

    /**
     * The FP6 element dtype of a Quark configuration's weight specs: true for {@code fp6_e2m3}, false for
     * {@code fp6_e3m2}. The arguments of a module do not carry it, so the configuration must use one of them.
     */
    static boolean quarkFp6E2m3(JsonView q) {
        Set<String> found = new TreeSet<>();
        collectFp6(q.raw("global_quant_config"), found);
        for (String key : List.of("layer_quant_config", "layer_type_quant_config")) {
            Object v = q.raw(key);
            if (v instanceof Map) {
                for (Object layer : ((Map<?, ?>) v).values()) {
                    collectFp6(layer, found);
                }
            }
        }
        if (found.size() != 1) {
            throw new GgufFormatException(q.source() + ": Quark MXFP6 weights need exactly one FP6 element dtype"
                    + " across the weight specs; found " + found);
        }
        return found.contains("fp6_e2m3");
    }

    private static void collectFp6(Object layer, Set<String> found) {
        if (!(layer instanceof Map)) {
            return;
        }
        Object w = ((Map<?, ?>) layer).get("weight");
        if (w instanceof List && ((List<?>) w).size() == 1) {
            w = ((List<?>) w).get(0);
        }
        if (w instanceof Map) {
            Object d = ((Map<?, ?>) w).get("dtype");
            if (d instanceof String && ((String) d).startsWith("fp6_")) {
                found.add((String) d);
            }
        }
    }
}
