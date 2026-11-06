package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantMethod;
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
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Rewrites FP8 modules (fp8, DeepSeek-V4, compressed-tensors, ModelOpt, fbgemm, w8a8_fp8 and Quark FP8 rows) and
 * MXFP8 modules into their GGUF targets:
 * <ul>
 *   <li>E4M3 with a per-tensor or per-expert scale: the stored bytes as ggml's per-element F8_E4M3FN (llama.cpp
 *       {@code GGML_TYPE_F8_E4M3}, zero-copy, any K) with the scale as the output multiplier; folded into the
 *       weight it becomes F8_E4M3_B128.</li>
 *   <li>E4M3 with a per-channel scale, or block scales whose K block is a multiple of 128: F8_E4M3_B128 with the
 *       scale of each 128-run; E8M0 block scales with a K block of 32 (or a multiple): MXFP8.</li>
 *   <li>E5M2 with a per-tensor scale: F16 ({@code code << 8}) with the output multiplier.</li>
 *   <li>E4M3FNUZ and E5M2FNUZ: F32 with the scale applied.</li>
 *   <li>MXFP8: E4M3 codes with one E8M0 scale per 32 elements (per {@code r} rows in the repeat form): MXFP8.</li>
 *   <li>Anything else: the float fallback, logged once per batch.</li>
 * </ul>
 * Scales are discovered per module, first present wins: {@code weight_scale_inv}, {@code weight_scale},
 * {@code qscale_weight}, {@code scale_inv} (the {@code <p>_scale_inv} of transformers' fused FP8 experts, whose
 * weight is the bare {@code <p>}, finegrained_fp8.py:673-680), and {@code scale} only for DeepSeek-V4 FP8
 * checkpoints with F8 weights (128x128 blocks).
 * Every scale is a dequant multiplier; E8M0 scales decode as {@code 2^(e-127)} and the NaN byte is rejected.
 */
public final class Fp8Scheme {

    private static final Logger LOG = LogManager.getLogger(Fp8Scheme.class);

    private static final String[] SCALES = {"weight_scale_inv", "weight_scale", "qscale_weight", "scale_inv"};
    private static final String[] INPUT_SCALES = {"input_scale", "qscale_act"};
    private static final int DSV4_BLOCK = 128;
    private static final int B128 = 128;
    private static final int MX = 32;

    private enum Grid { TENSOR, CHANNEL, BLOCK }

    private Fp8Scheme() {
    }

    /** True for {@link QuantFormat#FP8} and {@link QuantFormat#MXFP8}. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.FP8 || f == QuantFormat.MXFP8;
    }

    /**
     * Rewrites every module of {@code modules} (HF module prefixes) in {@code in}.
     *
     * @throws GgufFormatException when a module's tensors are missing, have the wrong dtype or inconsistent shapes
     * @throws IOException         when a scale tensor cannot be read or holds a NaN code
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(modules, "modules");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(o, "o");
        if (!handles(args.format())) {
            throw new IllegalArgumentException("Fp8Scheme does not handle " + args.format());
        }
        Ctx ctx = new Ctx(args, cfg, in, o);
        for (String module : modules) {
            Module m = new Module(in, module);
            if (args.format() == QuantFormat.MXFP8) {
                mxfp8(ctx, m);
            } else {
                fp8(ctx, m);
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
        private Boolean dsv4;

        Ctx(QuantArgs args, HfConfig cfg, HfTensorIndex in, ModelSourceOptions o) {
            this.args = args;
            this.cfg = cfg;
            this.in = in;
            this.o = o;
        }

        boolean dsv4() throws IOException {
            if (dsv4 == null) {
                dsv4 = HfQuantConfig.of(cfg).method() == HfQuantMethod.DEEPSEEK_V4_FP8;
            }
            return dsv4;
        }
    }

    /** A decoded scale grid over {@code [E,] N, K}. */
    private static final class Scales {
        final Grid grid;
        final float[] v;
        final boolean e8m0;
        final StDType dtype;
        final long n;
        final int bn;
        final int bk;
        final long rb;
        final long cb;

        Scales(Grid grid, float[] v, boolean e8m0, StDType dtype, long n, int bn, int bk, long rb, long cb) {
            this.grid = grid;
            this.v = v;
            this.e8m0 = e8m0;
            this.dtype = dtype;
            this.n = n;
            this.bn = bn;
            this.bk = bk;
            this.rb = rb;
            this.cb = cb;
        }

        /** The multiplier of weight row {@code row} (expert-major) at column {@code col}. */
        float at(long row, int col) {
            switch (grid) {
                case TENSOR:
                    return v.length == 1 ? v[0] : v[(int) (row / n)];
                case CHANNEL:
                    return v[(int) row];
                default:
                    long e = row / n;
                    return v[(int) ((e * rb + (row % n) / bn) * cb + col / bk)];
            }
        }

        /** The per-expert (or single) values of a TENSOR grid. */
        float[] perTensor() {
            return v.clone();
        }
    }

    private static boolean isF8(StDType t) {
        return t == StDType.F8_E4M3 || t == StDType.F8_E5M2 || t == StDType.F8_E4M3FNUZ || t == StDType.F8_E5M2FNUZ;
    }

    /**
     * The note of a {@code scheme} module stored unquantized: it loads as stored, unless the load's online
     * quantization requantizes it.
     */
    private static String unquantizedNote(String scheme, ModelSourceOptions o) {
        if (o.onlineQuant() == ModelSourceOptions.OnlineQuant.NONE) {
            return scheme + " modules stored unquantized load as stored; --quant-online quantizes them";
        }
        return scheme + " modules stored unquantized load as stored where online quantization ("
                + o.onlineQuant().name().toLowerCase(Locale.ROOT) + ") leaves them";
    }

    private static boolean isFloat(StDType t) {
        return t == StDType.F32 || t == StDType.F16 || t == StDType.BF16;
    }

    // ------------------------------------------------------------------ FP8

    private static void fp8(Ctx ctx, Module m) throws IOException {
        String wName = m.name("weight");
        if (wName == null) {
            throw new GgufFormatException("FP8 module " + m.prefix() + " has no weight tensor");
        }
        RowSource w = ctx.in.get(wName);
        StDType wt = Fp4Packing.dtype(w, wName);
        String scaleSuffix = m.first(SCALES);
        if (scaleSuffix == null && isF8(wt) && m.has("scale") && ctx.dsv4()) {
            scaleSuffix = "scale";
        }
        if (isFloat(wt) && scaleSuffix == null && m.first(INPUT_SCALES) == null) {
            ctx.notes.info(unquantizedNote("FP8", ctx.o), m.prefix());
            return;
        }
        if (!isF8(wt)) {
            throw new GgufFormatException("FP8 module " + m.prefix() + ": weight " + wName + " has dtype "
                    + wt.headerName() + "; expected an FP8 dtype");
        }
        if (scaleSuffix == null) {
            throw new GgufFormatException("FP8 module " + m.prefix() + " has no weight scale (weight_scale_inv,"
                    + " weight_scale or qscale_weight)");
        }
        Geometry g = Geometry.of(w.shape(), wName);
        String sName = m.name(scaleSuffix);
        Scales s = scales(ctx, g, ctx.in.get(sName), sName, scaleSuffix.equals("scale"));
        String out = m.outputName("weight");
        Produced src = fp8Target(ctx, m, w, wName, wt, g, s);
        ctx.rw.put(out, Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, false, INPUT_SCALES));
        if (!wName.equals(out)) {
            ctx.rw.consume(wName);
        }
        ctx.rw.consume(sName);
    }

    private static Produced fp8Target(Ctx ctx, Module m, RowSource w, String wName, StDType wt, Geometry g,
                                      Scales s) {
        QuantArgs a = ctx.args;
        if (wt == StDType.F8_E4M3FNUZ || wt == StDType.F8_E5M2FNUZ) {
            return fallback(ctx, m, w, wName, wt, g, s, StDType.F32);
        }
        if (wt == StDType.F8_E5M2) {
            if (s.grid == Grid.TENSOR) {
                Produced f16 = Produced.of(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(),
                        "F16 (E5M2)"), ElementFormat.of(StDType.F16), g.shape, w.rowLocal(), e5m2ToF16(w, wName, g));
                return f16.withOutputScale(s.perTensor(), () -> foldedFallback(ctx, m, w, wName, wt, g, s));
            }
            ctx.notes.fallback("FP8 E5M2 weights without a per-tensor scale load as the float fallback",
                    m.prefix());
            return fallback(ctx, m, w, wName, wt, g, s, Fp4Packing.fallbackType(s.dtype, ctx.o));
        }
        if (s.grid == Grid.TENSOR) {
            Produced raw = Produced.of(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(),
                    GGMLType.F8_E4M3FN), ElementFormat.of(GGMLType.F8_E4M3FN), g.shape, w.rowLocal(), w::readRows)
                    .withDirect(w.directVirtualOffset());
            return raw.withOutputScale(s.perTensor(), () -> g.k % B128 == 0 ? b128(ctx, m, w, wName, g, s)
                    : foldedFallback(ctx, m, w, wName, wt, g, s));
        }
        boolean blockRuns = s.grid != Grid.BLOCK || s.bk % B128 == 0;
        if (s.grid != Grid.TENSOR && blockRuns && g.k % B128 == 0) {
            return b128(ctx, m, w, wName, g, s);
        }
        if (s.grid == Grid.BLOCK && s.e8m0 && s.bk % MX == 0 && g.k % MX == 0) {
            return blockMxfp8(ctx, m, w, wName, g, s);
        }
        ctx.notes.fallback("FP8 weights whose K or scale blocks fit neither 32- nor 128-element blocks load as the"
                + " float fallback", m.prefix());
        return fallback(ctx, m, w, wName, wt, g, s, Fp4Packing.fallbackType(s.dtype, ctx.o));
    }

    /** The float fallback of a per-tensor weight whose output multiplier is folded into it (no sibling slot). */
    private static Produced foldedFallback(Ctx ctx, Module m, RowSource w, String wName, StDType wt, Geometry g,
                                           Scales s) {
        LOG.warn("{}: the per-tensor scale of this {} weight (K = {}) has no sibling slot and no block type"
                + " carries it; it is folded into the float fallback", m.prefix(), wt.headerName(), g.k);
        return fallback(ctx, m, w, wName, wt, g, s, Fp4Packing.fallbackType(s.dtype, ctx.o));
    }

    private static Scales scales(Ctx ctx, Geometry g, RowSource s, String sName, boolean dsv4Scale)
            throws IOException {
        QuantArgs a = ctx.args;
        StDType st = Fp4Packing.dtype(s, sName);
        boolean u8E8m0 = dsv4Scale || a.scaleFormat() == QuantArgs.ScaleFormat.E8M0;
        boolean e8m0 = st == StDType.F8_E8M0 || st == StDType.U8 && u8E8m0;
        float[] v = Fp4Packing.readScales(s, sName, u8E8m0);
        long[] sh = s.shape();
        int e = Math.max(1, g.experts);
        if (v.length == 1 && allOnes(sh, 0)) {
            return new Scales(Grid.TENSOR, v, e8m0, st, g.n, 0, 0, 0, 0);
        }
        if (g.experts > 0 && v.length == g.experts && sh.length >= 1 && sh[0] == g.experts && allOnes(sh, 1)) {
            return new Scales(Grid.TENSOR, v, e8m0, st, g.n, 0, 0, 0, 0);
        }
        int lead = g.experts > 0 ? 1 : 0;
        if (g.experts > 0 && (sh.length < 1 || sh[0] != g.experts)) {
            throw badScale(sName, sh, g, "a leading expert dimension of " + g.experts);
        }
        if (v.length == e * g.n && sh.length - lead >= 1 && sh[lead] == g.n && allOnes(sh, lead + 1)) {
            return new Scales(Grid.CHANNEL, v, e8m0, st, g.n, 0, 0, 0, 0);
        }
        if (sh.length - lead != 2) {
            throw badScale(sName, sh, g, "a per-tensor, per-channel or 2-D block grid");
        }
        int bn;
        int bk;
        if (a.strategy() == QuantArgs.Strategy.BLOCK) {
            bn = a.blockN();
            bk = a.blockK();
        } else if (dsv4Scale) {
            bn = DSV4_BLOCK;
            bk = DSV4_BLOCK;
        } else {
            throw badScale(sName, sh, g, "a per-tensor or per-channel scale (the configuration declares no block"
                    + " size)");
        }
        long rb = (g.n + bn - 1) / bn;
        long cb = (g.k + bk - 1) / bk;
        if (sh[lead] != rb || sh[lead + 1] != cb) {
            throw badScale(sName, sh, g, "[" + rb + ", " + cb + "] blocks of " + bn + "x" + bk);
        }
        return new Scales(Grid.BLOCK, v, e8m0, st, g.n, bn, bk, rb, cb);
    }

    private static boolean allOnes(long[] shape, int from) {
        for (int i = from; i < shape.length; i++) {
            if (shape[i] != 1) {
                return false;
            }
        }
        return true;
    }

    private static GgufFormatException badScale(String sName, long[] sh, Geometry g, String expected) {
        return new GgufFormatException("scale " + sName + " has shape " + Arrays.toString(sh) + " for weight "
                + Arrays.toString(g.shape) + "; expected " + expected);
    }

    // ------------------------------------------------------------------ producers

    /** F8_E4M3_B128: per 128-run, {@code d} = the run's scale, then the 128 E4M3 codes. */
    private static Produced b128(Ctx ctx, Module m, RowSource w, String wName, Geometry g, Scales s) {
        QuantArgs a = ctx.args;
        RowWriter writer = (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] codes = Fp4Packing.readRows(w, first + done, c);
                for (int r = 0; r < c; r++) {
                    long row = first + done + r;
                    int base = r * g.k;
                    for (int col = 0; col < g.k; col += B128) {
                        Fp4Packing.putF32(dst, s.at(row, col));
                        for (int j = 0; j < B128; j++) {
                            int code = codes[base + col + j];
                            Fp4Packing.checkE4m3(code, wName, row, col + j);
                            dst.put((byte) code);
                        }
                    }
                }
            }
        };
        return Produced.of(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(),
                GGMLType.F8_E4M3_B128), ElementFormat.of(GGMLType.F8_E4M3_B128), g.shape, w.rowLocal(), writer);
    }

    /** MXFP8 from E8M0 block scales: per 32-run, the run's exponent byte, then the 32 E4M3 codes. */
    private static Produced blockMxfp8(Ctx ctx, Module m, RowSource w, String wName, Geometry g, Scales s) {
        QuantArgs a = ctx.args;
        RowWriter writer = (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] codes = Fp4Packing.readRows(w, first + done, c);
                for (int r = 0; r < c; r++) {
                    long row = first + done + r;
                    putMxfp8Row(dst, codes, r * g.k, g.k, wName, row, col -> e8m0Byte(s.at(row, col)));
                }
            }
        };
        return Produced.of(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(), GGMLType.MXFP8),
                ElementFormat.of(GGMLType.MXFP8), g.shape, w.rowLocal(), writer);
    }

    /** The exponent byte of a power-of-two scale decoded from E8M0. */
    private static int e8m0Byte(float v) {
        int bits = Float.floatToRawIntBits(v);
        return bits == 0x00400000 ? 0 : bits >>> 23;
    }

    private interface ExpAt {
        int at(int col) throws IOException;
    }

    private static void putMxfp8Row(ByteBuffer dst, byte[] codes, int base, int k, String wName, long row, ExpAt exp)
            throws IOException {
        for (int col = 0; col < k; col += MX) {
            dst.put((byte) exp.at(col));
            for (int j = 0; j < MX; j++) {
                int code = codes[base + col + j];
                Fp4Packing.checkE4m3(code, wName, row, col + j);
                dst.put((byte) code);
            }
        }
    }

    /** E5M2 codes as IEEE halves: the code is the half's high byte. */
    private static RowWriter e5m2ToF16(RowSource w, String wName, Geometry g) {
        return (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] codes = Fp4Packing.readRows(w, first + done, c);
                for (int i = 0; i < c * g.k; i++) {
                    int code = codes[i] & 0xFF;
                    if ((code & 0x7C) == 0x7C) {
                        throw new IOException("tensor " + wName + " row " + (first + done + i / g.k) + " column "
                                + (i % g.k) + ": E5M2 code 0x" + Integer.toHexString(code) + " is not finite");
                    }
                    dst.put((byte) 0).put((byte) code);
                }
            }
        };
    }

    /** The dequantized weight {@code code * scale} in {@code type}. */
    private static Produced fallback(Ctx ctx, Module m, RowSource w, String wName, StDType wt, Geometry g, Scales s,
                                     StDType type) {
        QuantArgs a = ctx.args;
        return Fp4Packing.floats(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(), type),
                g.shape, type, w.rowLocal(), (first, count, out) -> {
                    byte[] codes = Fp4Packing.readRows(w, first, count);
                    for (int r = 0; r < count; r++) {
                        long row = first + r;
                        for (int col = 0; col < g.k; col++) {
                            int i = r * g.k + col;
                            float v = decode(wt, codes[i]);
                            if (!Float.isFinite(v)) {
                                throw new IOException("tensor " + wName + " row " + row + " column " + col + ": "
                                        + wt.headerName() + " code 0x" + Integer.toHexString(codes[i] & 0xFF)
                                        + " is not finite");
                            }
                            out[i] = v * s.at(row, col);
                        }
                    }
                });
    }

    private static float decode(StDType wt, byte code) {
        switch (wt) {
            case F8_E4M3:
                return FpBlockDecoders.e4m3ToFp32(code);
            case F8_E5M2:
                return FpBlockDecoders.e5m2ToFp32(code);
            case F8_E4M3FNUZ:
                return e4m3FnuzToFp32(code);
            default:
                return e5m2FnuzToFp32(code);
        }
    }

    /** E4M3FNUZ code to float (bias 8, no infinities); {@code 0x80} is NaN. */
    private static float e4m3FnuzToFp32(int code) {
        return fnuz(code & 0xFF, 3, 8);
    }

    /** E5M2FNUZ code to float (bias 16, no infinities); {@code 0x80} is NaN. */
    private static float e5m2FnuzToFp32(int code) {
        return fnuz(code & 0xFF, 2, 16);
    }

    private static float fnuz(int c, int mantBits, int bias) {
        if (c == 0x80) {
            return Float.NaN;
        }
        int exp = (c & 0x7F) >>> mantBits;
        int man = c & ((1 << mantBits) - 1);
        float v = exp == 0 ? Math.scalb((float) man, 1 - bias - mantBits)
                : Math.scalb((float) ((1 << mantBits) + man), exp - bias - mantBits);
        return (c & 0x80) != 0 ? -v : v;
    }

    // ------------------------------------------------------------------ MXFP8

    private static void mxfp8(Ctx ctx, Module m) throws IOException {
        String wName = m.name("weight");
        if (wName == null) {
            throw new GgufFormatException("MXFP8 module " + m.prefix() + " has no weight tensor");
        }
        RowSource w = ctx.in.get(wName);
        StDType wt = Fp4Packing.dtype(w, wName);
        String sName = m.name("weight_scale");
        if (isFloat(wt) && sName == null) {
            ctx.notes.info(unquantizedNote("MXFP8", ctx.o), m.prefix());
            return;
        }
        if (wt != StDType.F8_E4M3) {
            throw new GgufFormatException("MXFP8 module " + m.prefix() + ": weight " + wName + " has dtype "
                    + wt.headerName() + "; expected F8_E4M3");
        }
        if (sName == null) {
            throw new GgufFormatException("MXFP8 module " + m.prefix() + " has no weight_scale");
        }
        Geometry g = Geometry.of(w.shape(), wName);
        if (g.k % MX != 0) {
            throw new GgufFormatException("MXFP8 weight " + wName + " has K = " + g.k + ", not a multiple of 32");
        }
        RowSource s = ctx.in.get(sName);
        StDType st = Fp4Packing.dtype(s, sName);
        if (st != StDType.U8 && st != StDType.F8_E8M0) {
            throw new GgufFormatException("MXFP8 scale " + sName + " has dtype " + st.headerName()
                    + "; expected U8 or F8_E8M0 exponents");
        }
        long[] sh = s.shape();
        int lead = g.experts > 0 ? 1 : 0;
        if (sh.length != 2 + lead || lead == 1 && sh[0] != g.experts || sh[lead + 1] != g.k / MX) {
            throw badScale(sName, sh, g, "[" + (g.experts > 0 ? "E, " : "") + "rows, " + g.k / MX + "]");
        }
        long scaleRows = sh[lead];
        int r = mxRows(ctx.args, g.n, scaleRows, sName);
        QuantArgs a = ctx.args;
        RowWriter writer = (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                long f = first + done;
                long s0 = scaleRow(f, g.n, scaleRows, r);
                long s1 = scaleRow(f + c - 1, g.n, scaleRows, r);
                byte[] exps = Fp4Packing.readRows(s, s0, Math.toIntExact(s1 - s0 + 1));
                byte[] codes = Fp4Packing.readRows(w, f, c);
                int perRow = g.k / MX;
                for (int i = 0; i < c; i++) {
                    long row = f + i;
                    int sBase = Math.toIntExact((scaleRow(row, g.n, scaleRows, r) - s0) * perRow);
                    putMxfp8Row(dst, codes, i * g.k, g.k, wName, row,
                            col -> Fp4Packing.checkE8m0(exps[sBase + col / MX], sName, row, col / MX));
                }
            }
        };
        Produced src = Produced.of(Fp4Packing.describe(m.prefix(), a.format(), a.bits(), a.groupSize(),
                GGMLType.MXFP8), ElementFormat.of(GGMLType.MXFP8), g.shape, w.rowLocal() && s.rowLocal(), writer);
        String out = m.outputName("weight");
        ctx.rw.put(out, Fp4Packing.activation(ctx.args, m, ctx.in, ctx.rw, src, g.experts, false, INPUT_SCALES));
        if (!wName.equals(out)) {
            ctx.rw.consume(wName);
        }
        ctx.rw.consume(sName);
    }

    /** Rows per scale row: 1, the configured block rows of 2-D MXFP8, or the ModelOpt repeat factor. */
    private static int mxRows(QuantArgs a, long n, long scaleRows, String sName) {
        int r;
        if (a.strategy() == QuantArgs.Strategy.BLOCK) {
            if (a.blockK() != MX) {
                throw new GgufFormatException("MXFP8 block " + a.blockN() + "x" + a.blockK() + " of " + sName
                        + " does not have 32 columns");
            }
            r = a.blockN();
        } else {
            r = scaleRows <= 0 ? 0 : Math.toIntExact((n + scaleRows - 1) / scaleRows);
        }
        if (r <= 0 || (n + r - 1) / r != scaleRows) {
            throw new GgufFormatException("MXFP8 scale " + sName + " has " + scaleRows + " rows for " + n
                    + " weight rows" + (r > 0 ? " and " + r + " rows per scale" : ""));
        }
        return r;
    }

    private static long scaleRow(long row, long n, long scaleRows, int r) {
        return row / n * scaleRows + row % n / r;
    }
}
