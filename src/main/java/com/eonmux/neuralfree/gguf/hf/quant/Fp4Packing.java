package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Shared machinery of the float-format rewriters: the E2M1 nibble shuffles into the ggml MXFP4, NVFP4 and
 * bitsandbytes 4-bit block layouts, OCP scale checks that reject the NaN codes, reads of small auxiliary tensors, the
 * module tensor lookup, and the lazy row sources ({@link Produced}) the schemes return.
 *
 * <p>Packed E2M1 runs come in two nibble orders: low nibble first (element {@code 2j} in the low nibble of byte
 * {@code j}: compressed-tensors, ModelOpt, Quark, gpt-oss, DeepSeek-V4 and fp_quant) and high nibble first
 * (bitsandbytes). The ggml MXFP4 block holds element {@code j} in the low nibble and element {@code j+16} in the high
 * nibble of byte {@code j}; the ggml NVFP4 sub-block {@code s} holds elements {@code 16s+j} and {@code 16s+j+8} in
 * byte {@code 8s+j}; the NF4/FP4_BNB block holds elements {@code j} and {@code j+32} in byte {@code j}.</p>
 */
public final class Fp4Packing {

    /** Largest auxiliary tensor (scales, global scales, quant maps, transforms) read whole, in bytes. */
    static final long MAX_AUX_BYTES = 1L << 28;

    /** Producers work through a request in chunks of about this many weight elements. */
    static final int CHUNK_ELEMENTS = 1 << 20;

    private Fp4Packing() {
    }

    // ------------------------------------------------------------------ E2M1 nibble shuffles

    /** Code {@code i} of a run packed two per byte from {@code packed[off]}, element {@code 2j} in the low nibble. */
    public static int lowFirst(byte[] packed, int off, int i) {
        return (packed[off + (i >>> 1)] >>> ((i & 1) << 2)) & 0x0F;
    }

    /** Code {@code i} of a run packed two per byte from {@code packed[off]}, element {@code 2j} in the high nibble. */
    public static int highFirst(byte[] packed, int off, int i) {
        return (packed[off + (i >>> 1)] >>> ((~i & 1) << 2)) & 0x0F;
    }

    /**
     * Writes one ggml MXFP4 block: the E8M0 byte {@code e}, then 16 bytes holding the 32 codes packed low nibble
     * first at {@code packed[off, off + 16)}, byte {@code j} = {@code code(j) | code(j + 16) << 4}.
     */
    public static void putMxfp4(ByteBuffer dst, int e, byte[] packed, int off) {
        dst.put((byte) e);
        for (int j = 0; j < 16; j++) {
            dst.put((byte) (lowFirst(packed, off, j) | lowFirst(packed, off, j + 16) << 4));
        }
    }

    /**
     * Writes one ggml NVFP4 block of 64 codes packed low nibble first at {@code packed[off, off + 32)}: four UE4M3
     * sub-scales {@code scales[sOff + s] & 0x7F}, then byte {@code 8s + j} = {@code code(16s+j) | code(16s+j+8) << 4}.
     * The scale bytes must already be checked ({@link #checkE4m3Scale}).
     */
    public static void putNvfp4(ByteBuffer dst, byte[] scales, int sOff, byte[] packed, int off) {
        for (int s = 0; s < 4; s++) {
            dst.put((byte) (scales[sOff + s] & 0x7F));
        }
        for (int s = 0; s < 4; s++) {
            for (int j = 0; j < 8; j++) {
                dst.put((byte) (lowFirst(packed, off, 16 * s + j) | lowFirst(packed, off, 16 * s + j + 8) << 4));
            }
        }
    }

    /**
     * Writes one NF4/FP4_BNB block: {@code absmax} as a little-endian f32, then byte {@code j} =
     * {@code codes[cOff + j] | codes[cOff + j + 32] << 4} for 64 codes held one per byte.
     */
    public static void putBnb4(ByteBuffer dst, float absmax, byte[] codes, int cOff) {
        putF32(dst, absmax);
        for (int j = 0; j < 32; j++) {
            dst.put((byte) ((codes[cOff + j] & 0x0F) | (codes[cOff + j + 32] & 0x0F) << 4));
        }
    }

    // ------------------------------------------------------------------ code and scale checks

    /** Fails with the NaN byte {@code 0xFF} of an E8M0 scale; returns {@code e & 0xFF} otherwise. */
    static int checkE8m0(int e, String tensor, long row, long block) throws IOException {
        int u = e & 0xFF;
        if (u == 0xFF) {
            throw new IOException("tensor " + tensor + " row " + row + " block " + block
                    + ": E8M0 scale 0xFF is NaN");
        }
        return u;
    }

    /** Fails with the E4M3FN NaN codes {@code 0x7F} and {@code 0xFF}. */
    static void checkE4m3(int code, String tensor, long row, long col) throws IOException {
        if ((code & 0x7F) == 0x7F) {
            throw new IOException("tensor " + tensor + " row " + row + " column " + col + ": E4M3 code 0x"
                    + Integer.toHexString(code & 0xFF) + " is NaN");
        }
    }

    /** Fails with an E4M3 sub-scale that is NaN or negative and nonzero (an NVFP4 sub-scale is unsigned). */
    static void checkE4m3Scale(int code, String tensor, long row, long block) throws IOException {
        int u = code & 0xFF;
        if ((u & 0x7F) == 0x7F || (u & 0x80) != 0 && (u & 0x7F) != 0) {
            throw new IOException("tensor " + tensor + " row " + row + " block " + block + ": E4M3 scale 0x"
                    + Integer.toHexString(u) + " is " + ((u & 0x7F) == 0x7F ? "NaN" : "negative"));
        }
    }

    // ------------------------------------------------------------------ auxiliary tensor reads

    /** The number of elements of {@code shape}. */
    static long numel(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    /** All bytes of a small tensor, at most {@link #MAX_AUX_BYTES}. */
    static byte[] readAll(RowSource src, String name) throws IOException {
        long total = Math.multiplyExact(src.rows(), src.rowBytes());
        if (total > MAX_AUX_BYTES) {
            throw new GgufFormatException("tensor " + name + " of " + total + " bytes is too large for an auxiliary"
                    + " quantization tensor (at most " + MAX_AUX_BYTES + ")");
        }
        ByteBuffer buf = ByteBuffer.allocate((int) total);
        src.readRows(0, (int) src.rows(), buf);
        return buf.array();
    }

    /** Rows {@code [first, first + count)} of {@code src} as bytes. */
    static byte[] readRows(RowSource src, long first, int count) throws IOException {
        long n = Math.multiplyExact((long) count, src.rowBytes());
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(count + " rows of " + src.describe() + " do not fit one buffer");
        }
        ByteBuffer buf = ByteBuffer.allocate((int) n);
        src.readRows(first, count, buf);
        return buf.array();
    }

    /** The safetensors dtype of {@code src}, or a named error for a produced block type. */
    static StDType dtype(RowSource src, String name) {
        StDType t = src.format().stDType();
        if (t == null) {
            throw new GgufFormatException("tensor " + name + " has the block format " + src.format()
                    + "; a quantized checkpoint tensor was expected");
        }
        return t;
    }

    /**
     * The values of a small scale tensor: F32, F16 and BF16 decode exactly, F8_E8M0 (and U8 when
     * {@code u8IsE8m0}) as {@code 2^(e-127)} with the NaN byte rejected. Every value must be finite.
     */
    static float[] readScales(RowSource src, String name, boolean u8IsE8m0) throws IOException {
        StDType t = dtype(src, name);
        byte[] b = readAll(src, name);
        ByteBuffer le = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        int n = Math.toIntExact(numel(src.shape()));
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            switch (t) {
                case F32:
                    out[i] = le.getFloat(4 * i);
                    break;
                case F16:
                    out[i] = FpBlockDecoders.fp16ToFp32(le.getShort(2 * i) & 0xFFFF);
                    break;
                case BF16:
                    out[i] = FpBlockDecoders.bf16ToFp32(le.getShort(2 * i) & 0xFFFF);
                    break;
                case F8_E8M0:
                    out[i] = FpBlockDecoders.e8m0ToFp32(checkE8m0(b[i], name, 0, i));
                    break;
                case U8:
                    if (u8IsE8m0) {
                        out[i] = FpBlockDecoders.e8m0ToFp32(checkE8m0(b[i], name, 0, i));
                        break;
                    }
                    throw unsupportedScale(name, t);
                default:
                    throw unsupportedScale(name, t);
            }
            if (!Float.isFinite(out[i])) {
                throw new GgufFormatException("tensor " + name + " element " + i + " is " + out[i]
                        + "; quantization parameters must be finite");
            }
        }
        return out;
    }

    private static GgufFormatException unsupportedScale(String name, StDType t) {
        return new GgufFormatException("tensor " + name + ": unsupported scale dtype " + t.headerName());
    }

    /** One value of a tensor that holds 1 element, or 1 per expert when {@code experts > 0}. */
    static float[] readPerExpert(RowSource src, String name, int experts) throws IOException {
        float[] v = readScales(src, name, false);
        if (v.length != 1 && (experts <= 0 || v.length != experts)) {
            throw new GgufFormatException("tensor " + name + " " + Arrays.toString(src.shape()) + " must hold 1 value"
                    + (experts > 0 ? " or 1 per expert (" + experts + ")" : ""));
        }
        return v;
    }

    /** The reciprocal of every value, each a positive finite number. */
    static float[] reciprocal(float[] v, String name) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            if (!(v[i] > 0) || Float.isInfinite(v[i])) {
                throw new GgufFormatException("tensor " + name + ": global scale " + v[i] + " is not a positive finite"
                        + " number");
            }
            out[i] = 1.0f / v[i];
        }
        return out;
    }

    /** {@code [E,] N, K} of a rank-2 or rank-3 logical weight; {@code experts} is 0 for rank 2. Immutable. */
    static final class Geometry {
        final long[] shape;
        final int experts;
        final long n;
        final int k;

        private Geometry(long[] shape, int experts, long n, int k) {
            this.shape = shape;
            this.experts = experts;
            this.n = n;
            this.k = k;
        }

        /** The geometry of a weight of logical {@code shape}, which must be a non-empty [N, K] or [E, N, K]. */
        static Geometry of(long[] shape, String name) {
            if (shape.length != 2 && shape.length != 3) {
                throw new GgufFormatException("quantized weight " + name + " has shape " + Arrays.toString(shape)
                        + "; expected [N, K] or [E, N, K]");
            }
            for (long d : shape) {
                if (d <= 0) {
                    throw new GgufFormatException("quantized weight " + name + " has the empty shape "
                            + Arrays.toString(shape));
                }
            }
            if (shape[shape.length - 1] > Integer.MAX_VALUE / 8 || shape.length == 3 && shape[0] > Integer.MAX_VALUE) {
                throw new GgufFormatException("quantized weight " + name + " has the oversized shape "
                        + Arrays.toString(shape));
            }
            return new Geometry(shape.clone(), shape.length == 3 ? (int) shape[0] : 0, shape[shape.length - 2],
                    (int) shape[shape.length - 1]);
        }

        /** The geometry of a packed tensor whose innermost dimension holds {@code num/den} bytes per element. */
        static Geometry packed(long[] packedShape, int num, int den, String name) {
            long[] s = packedShape.clone();
            if (s.length == 0) {
                throw new GgufFormatException("packed weight " + name + " is a scalar");
            }
            long last = s[s.length - 1];
            if (Math.multiplyExact(last, den) % num != 0) {
                throw new GgufFormatException("packed weight " + name + " has " + last + " bytes per row, not a whole"
                        + " number of elements");
            }
            s[s.length - 1] = Math.multiplyExact(last, den) / num;
            return of(s, name);
        }

        /** The number of logical rows, experts times N. */
        long rows() {
            return Math.max(1, experts) * n;
        }

        /** The shape with K replaced. */
        long[] withK(long kk) {
            long[] s = shape.clone();
            s[s.length - 1] = kk;
            return s;
        }

        /** Checks that {@code aux} has shape {@code [E,] N, cols}. */
        void checkRows(RowSource aux, String auxName, long cols) {
            long[] sh = aux.shape();
            long[] want = withK(cols);
            if (!Arrays.equals(sh, want)) {
                throw new GgufFormatException("tensor " + auxName + " has shape " + Arrays.toString(sh) + "; expected "
                        + Arrays.toString(want) + " for the weight " + Arrays.toString(shape));
            }
        }
    }

    // ------------------------------------------------------------------ module tensors

    /**
     * The tensors of one quantized module in an index. A suffix resolves in dotted form ({@code <module>.<suffix>}),
     * then in the underscore form of fused expert parameters ({@code <module>_<suffix>}); a fused expert parameter
     * stores its weight under the bare parameter name ({@code <module>}), which {@code weight} also resolves to. The
     * rewritten weight is {@code <module>.weight} in dotted form and {@code <module>} otherwise.
     */
    static final class Module {
        private static final String WEIGHT = "weight";

        private final HfTensorIndex in;
        private final String prefix;

        Module(HfTensorIndex in, String prefix) {
            if (prefix == null || prefix.isEmpty()) {
                throw new IllegalArgumentException("empty module name");
            }
            this.in = in;
            this.prefix = prefix;
        }

        String prefix() {
            return prefix;
        }

        /** The tensor name of {@code suffix}, or null when the module has none. */
        String name(String suffix) {
            String dotted = prefix + "." + suffix;
            if (in.has(dotted)) {
                return dotted;
            }
            String underscore = prefix + "_" + suffix;
            if (in.has(underscore)) {
                return underscore;
            }
            return WEIGHT.equals(suffix) && in.has(prefix) ? prefix : null;
        }

        boolean has(String suffix) {
            return name(suffix) != null;
        }

        /** The first suffix of {@code suffixes} the module has, or null. */
        String first(String... suffixes) {
            for (String s : suffixes) {
                if (has(s)) {
                    return s;
                }
            }
            return null;
        }

        RowSource get(String suffix) {
            String n = name(suffix);
            return n == null ? null : in.get(n);
        }

        RowSource require(String suffix) {
            String n = name(suffix);
            if (n == null) {
                throw new GgufFormatException("quantized module " + prefix + " has no tensor " + prefix + "." + suffix);
            }
            return in.get(n);
        }

        /** The name of the rewritten weight, by the form of {@code suffix}'s tensor. */
        String outputName(String suffix) {
            String dotted = prefix + "." + WEIGHT;
            String n = name(suffix);
            return n == null || n.equals(prefix + "." + suffix) ? dotted : prefix;
        }
    }

    /**
     * {@code src} with the activation scheme of {@code args}. The first present tensor of {@code suffixes} is
     * consumed; a static or global-scale scheme takes its 1 (or 1 per expert) values as the input multiplier,
     * inverted when {@code invert} (compressed-tensors NVFP4 stores the divisor), and fails without one.
     */
    static Produced activation(QuantArgs args, Module m, HfTensorIndex in, Rewrite rw, Produced src, int experts,
                               boolean invert, String... suffixes) throws IOException {
        ActQuantScheme act = args.actScheme();
        String suffix = m.first(suffixes);
        float[] scale = null;
        if (suffix != null) {
            String n = m.name(suffix);
            rw.consume(n);
            if (act != null && (act.isStatic() || act.needsGlobalScale())) {
                float[] v = readPerExpert(in.get(n), n, experts);
                scale = invert ? reciprocal(v, n) : v;
            }
        }
        if (act == null) {
            return src;
        }
        if ((act.isStatic() || act.needsGlobalScale()) && scale == null) {
            throw new GgufFormatException(args.format() + " module " + m.prefix() + ": " + act + " activations need "
                    + m.prefix() + "." + suffixes[0]);
        }
        return src.withActivation(act, scale);
    }

    /** The replacements and removals of one rewrite batch, applied together. */
    static final class Rewrite {
        private final Map<String, RowSource> replace = new LinkedHashMap<>();
        private final Set<String> remove = new LinkedHashSet<>();

        void put(String name, RowSource src) {
            remove.remove(name);
            replace.put(name, src);
        }

        void consume(String name) {
            if (name != null && !replace.containsKey(name)) {
                remove.add(name);
            }
        }

        void consume(Module m, String... suffixes) {
            for (String s : suffixes) {
                consume(m.name(s));
            }
        }

        HfTensorIndex apply(HfTensorIndex in) {
            return replace.isEmpty() && remove.isEmpty() ? in : in.withReplaced(replace, remove);
        }
    }

    /** Fallbacks and notices of one rewrite batch, each logged once with its modules. */
    static final class Notes {
        private final Map<String, List<String>> warn = new LinkedHashMap<>();
        private final Map<String, List<String>> info = new LinkedHashMap<>();

        void fallback(String reason, String module) {
            warn.computeIfAbsent(reason, k -> new ArrayList<>()).add(module);
        }

        void info(String message, String module) {
            info.computeIfAbsent(message, k -> new ArrayList<>()).add(module);
        }

        void log(Logger log) {
            for (Map.Entry<String, List<String>> e : warn.entrySet()) {
                log.warn("{} ({} module(s), first {})", e.getKey(), e.getValue().size(), e.getValue().get(0));
            }
            for (Map.Entry<String, List<String>> e : info.entrySet()) {
                log.info("{} ({} module(s), first {})", e.getKey(), e.getValue().size(), e.getValue().get(0));
            }
        }
    }

    // ------------------------------------------------------------------ produced row sources

    /** Writes rows {@code [first, first + count)} of a produced tensor at {@code dst}'s position. */
    @FunctionalInterface
    interface RowWriter {
        void write(long first, int count, ByteBuffer dst) throws IOException;
    }

    /** Decodes rows {@code [first, first + count)} of a logical float tensor into {@code out}, row-major. */
    @FunctionalInterface
    interface FloatRows {
        void decode(long first, int count, float[] out) throws IOException;
    }

    /** The float fallback dtype: F32 under {@code exactQuant}, else the scales' float dtype, else BF16. */
    static StDType fallbackType(StDType scaleType, ModelSourceOptions o) {
        if (o.exactQuant()) {
            return StDType.F32;
        }
        return scaleType == StDType.F32 || scaleType == StDType.F16 || scaleType == StDType.BF16
                ? scaleType : StDType.BF16;
    }

    /** Rows per chunk for rows of {@code rowElements} elements. */
    static int chunkRows(long rowElements) {
        return (int) Math.max(1L, CHUNK_ELEMENTS / Math.max(1L, rowElements));
    }

    /** A float tensor of {@code type} whose values {@code dec} decodes; rows hold {@code shape[last]} elements. */
    static Produced floats(String describe, long[] shape, StDType type, boolean rowLocal, FloatRows dec) {
        final int k = Math.toIntExact(shape[shape.length - 1]);
        RowWriter w = (first, count, dst) -> {
            int step = chunkRows(k);
            float[] buf = new float[Math.toIntExact((long) Math.min(step, count) * k)];
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                dec.decode(first + done, c, buf);
                for (int i = 0, n = c * k; i < n; i++) {
                    putFloat(dst, type, buf[i]);
                }
            }
        };
        return Produced.of(describe, ElementFormat.of(type), shape, rowLocal, w);
    }

    /** Writes {@code v} in {@code type} (F32, F16 or BF16, rounding to nearest even), little-endian. */
    static void putFloat(ByteBuffer dst, StDType type, float v) {
        switch (type) {
            case F32:
                putF32(dst, v);
                break;
            case F16:
                putU16(dst, FpBlockDecoders.fp32ToFp16(v));
                break;
            case BF16:
                putU16(dst, HalfFloats.floatToBFloat16(v));
                break;
            default:
                throw new IllegalArgumentException("not a float fallback dtype: " + type);
        }
    }

    static void putF32(ByteBuffer dst, float v) {
        int b = Float.floatToRawIntBits(v);
        dst.put((byte) b).put((byte) (b >>> 8)).put((byte) (b >>> 16)).put((byte) (b >>> 24));
    }

    static void putU16(ByteBuffer dst, int v) {
        dst.put((byte) v).put((byte) (v >>> 8));
    }

    /** {@code describe()} of a rewritten module: {@code <module> <format> b<bits> g<group> -> <target>}. */
    static String describe(String module, Object format, int bits, int group, Object target) {
        return module + " " + format + " b" + bits + " g" + group + " -> " + target;
    }

    /**
     * An immutable lazy tensor: its rows come from a {@link RowWriter}, and it carries the sibling values of
     * {@link RowSource}. The {@code with*} methods return modified copies.
     */
    static final class Produced implements RowSource {
        private final String describe;
        private final ElementFormat format;
        private final long[] shape;
        private final long rows;
        private final long rowBytes;
        private final boolean rowLocal;
        private final RowWriter writer;
        private final long direct;
        private final float[] outputScale;
        private final Supplier<RowSource> foldOutput;
        private final float[] inputScale;
        private final ActQuantScheme act;

        private Produced(String describe, ElementFormat format, long[] shape, boolean rowLocal, RowWriter writer,
                         long direct, float[] outputScale, Supplier<RowSource> foldOutput, float[] inputScale,
                         ActQuantScheme act) {
            this.describe = describe;
            this.format = format;
            this.shape = shape.clone();
            long r = 1;
            for (int i = 0; i + 1 < shape.length; i++) {
                r = Math.multiplyExact(r, shape[i]);
            }
            this.rows = r;
            this.rowBytes = format.rowBytes(shape.length == 0 ? 1 : shape[shape.length - 1]);
            this.rowLocal = rowLocal;
            this.writer = writer;
            this.direct = direct;
            this.outputScale = outputScale;
            this.foldOutput = foldOutput;
            this.inputScale = inputScale;
            this.act = act;
        }

        static Produced of(String describe, ElementFormat format, long[] shape, boolean rowLocal, RowWriter writer) {
            return new Produced(describe, format, shape, rowLocal, writer, -1, null, null, null, null);
        }

        /** This tensor stored unchanged at {@code offset} of the view's virtual address space. */
        Produced withDirect(long offset) {
            return new Produced(describe, format, shape, rowLocal, writer, offset, outputScale, foldOutput, inputScale,
                    act);
        }

        /** This tensor with an output multiplier that {@code fold} multiplies into the weight. */
        Produced withOutputScale(float[] scale, Supplier<RowSource> fold) {
            return new Produced(describe, format, shape, rowLocal, writer, direct, scale.clone(), fold, inputScale,
                    act);
        }

        /** This tensor with an activation scheme and its dequant multiplier (null when dynamic). */
        Produced withActivation(ActQuantScheme scheme, float[] scale) {
            return new Produced(describe, format, shape, rowLocal, writer, direct, outputScale, foldOutput,
                    scale == null ? null : scale.clone(), scheme);
        }

        @Override
        public String describe() {
            return describe;
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            return rowBytes;
        }

        @Override
        public boolean rowLocal() {
            return rowLocal;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                        + " rows of " + describe);
            }
            long bytes = count * rowBytes;
            if (dst.remaining() < bytes) {
                throw new IllegalArgumentException(count + " rows of " + describe + " need " + bytes + " bytes; the"
                        + " buffer has " + dst.remaining());
            }
            int start = dst.position();
            writer.write(first, count, dst);
            if (dst.position() - start != bytes) {
                throw new IllegalStateException(describe + " wrote " + (dst.position() - start) + " bytes for "
                        + count + " rows; expected " + bytes);
            }
        }

        @Override
        public long directVirtualOffset() {
            return direct;
        }

        @Override
        public float[] outputScale() {
            return outputScale == null ? null : outputScale.clone();
        }

        @Override
        public float[] inputScale() {
            return inputScale == null ? null : inputScale.clone();
        }

        @Override
        public ActQuantScheme actQuant() {
            return act;
        }

        @Override
        public RowSource foldOutputScale() {
            if (outputScale == null) {
                return RowSource.super.foldOutputScale();
            }
            RowSource folded = foldOutput.get();
            if (folded instanceof Produced && act != null) {
                return ((Produced) folded).withActivation(act, inputScale);
            }
            return folded;
        }

        @Override
        public String toString() {
            return "Produced{" + describe + "}";
        }
    }
}
