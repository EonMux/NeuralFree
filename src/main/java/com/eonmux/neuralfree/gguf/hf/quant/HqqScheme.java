package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.CodeTable;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

/**
 * HQQ checkpoints ({@link QuantFormat#INT_HQQ}, vLLM hqq_marlin.py before its removal) rewritten into ggml types.
 * With {@code axis = 1} the weight {@code [N, K]} is quantized as {@code R = N*K/g} rows of {@code g} values
 * (row-major), {@code w = (q - zero) * scale} with {@code scale} and {@code zero} {@code [R, 1]}
 * (equivalently {@code [N, K/g]}).
 *
 * <p>{@code W_q} packs the {@code R} rows in slabs (hqq bitpack): with {@code s} slots per stored element, packed
 * row {@code p} holds unpacked rows {@code p + i * ceil(R/s)} for {@code i < s}, slab {@code i} at bit
 * {@code (s - 1 - i) * b}. Four bits use U8 with 2 slots (the high nibble is the first half, as
 * {@code unpack_4bit_u8}), two bits 4 slots, one bit 8 slots, eight bits U8 unpacked, three bits I32 with 10 slots and
 * {@code R} padded to a multiple of 10.</p>
 *
 * <p>The per-module metadata tensors HQQ saves ({@code shape}, {@code axis}, {@code nbits}, {@code group_size} and
 * the flags vLLM ignored) are consumed; {@code shape} gives {@code [N, K]}, else {@code bias} gives N. Four-bit
 * weights become Q4_1 ({@code d = scale}, {@code m = fp16(-zero*scale)}); exact quantization takes Q8_0 when every
 * zero is an integer, else F32.</p>
 */
public final class HqqScheme {

    /** Metadata tensors of an HQQ module that carry no weight data (vLLM hqq_marlin.py ignore list). */
    private static final List<String> META = List.of("channel_wise", "compute_dtype", "encoded_state_dict",
            "offload_meta", "optimize", "packing", "round_zero", "stores_quant_config", "unpack_view_dtype");

    private HqqScheme() {
    }

    /** True for the formats this scheme rewrites. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.INT_HQQ;
    }

    /**
     * Replaces each module's quantized tensors in {@code in} by one virtual weight source, as
     * {@link IntGroupScheme#rewriteModules} does.
     *
     * @throws IOException              when a tensor is missing, has the wrong dtype or shape, or holds invalid values
     * @throws com.eonmux.neuralfree.gguf.hparams.GgufFormatException for a configuration this scheme cannot honour
     * @throws IllegalArgumentException when {@code args} is not a format of this scheme
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        IntRewriteBatch.requireFormat(args, handles(args.format()), HqqScheme.class);
        return IntRewriteBatch.rewrite(args, modules, cfg, in, o, HqqScheme::hqq, "W_q");
    }

    private static RowSource hqq(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        QuantArgs a = b.args();
        int bits = a.bits();
        if (bits != 1 && bits != 2 && bits != 3 && bits != 4 && bits != 8) {
            throw b.unsupported("HQQ " + bits + "-bit weights");
        }
        if (a.hqqAxis() != 1) {
            throw b.unsupported("HQQ axis " + a.hqqAxis() + " (groups along the output dimension)");
        }
        int g = a.groupSize();
        if (g <= 0) {
            throw b.unsupported("HQQ quantization without a positive group size");
        }
        checkMeta(m, "axis", 1);
        checkMeta(m, "nbits", bits);
        checkMeta(m, "group_size", g);
        checkMeta(m, "view_as_float", 0);
        checkMeta(m, "quant_scale", 0);
        checkMeta(m, "quant_zero", 0);
        for (String meta : META) {
            m.takeOptional(meta);
        }
        long[] nk = geometry(m, bits, g);
        long n = nk[0];
        int k = m.width(nk[1], "shape");
        if (k % g != 0) {
            throw m.invalid("K " + k + " is not a multiple of the group size " + g);
        }
        int groups = k / g;
        long rows = Math.multiplyExact(n, (long) groups);
        RowSource wq = m.take("W_q");
        int slots = slots(bits);
        m.requireDtype(wq, "W_q", bits == 3 ? new StDType[]{StDType.I32, StDType.U32} : new StDType[]{StDType.U8});
        long step = (rows + slots - 1) / slots;
        long[] ws = wq.shape();
        if (ws.length != 2 || ws[0] != step || ws[1] != g) {
            throw m.invalid(m.tensorName("W_q") + " has shape " + Arrays.toString(ws) + "; " + rows + " rows of " + g
                    + " " + bits + "-bit values pack into [" + step + ", " + g + "]");
        }
        RowSource sc = m.take("scale");
        RowSource zr = m.take("zero");
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, 0, n, k, bits, g, groups);
        s.codes = slabs(wq, bits, slots, step, g, k);
        s.scaleDtype = m.requireFloat(sc, "scale");
        s.scales = flatParams(m, sc, n, groups, "scale");
        m.requireFloat(zr, "zero");
        s.zeros = flatParams(m, zr, n, groups, "zero");
        return b.finish(m, s.build());
    }

    /** Stored elements per packed element: 32/3 rounded down for I32 three-bit packing, else 8/b in U8. */
    private static int slots(int bits) {
        return bits == 3 ? 10 : 8 / bits;
    }

    /** {@code [N, K]} from the {@code shape} metadata, else N from {@code bias} and K from the packed size. */
    private static long[] geometry(IntModuleTensors m, int bits, int g) throws IOException {
        RowSource shape = m.takeOptional("shape");
        if (shape != null) {
            long[] v = m.longs(shape, "shape");
            if (v.length != 2 || v[0] <= 0 || v[1] <= 0) {
                throw m.invalid(m.tensorName("shape") + " holds " + Arrays.toString(v) + "; expected [N, K]");
            }
            return v;
        }
        if (bits == 3 || !m.has("bias")) {
            throw m.invalid("the weight shape is unknown: there is no " + m.tensorName("shape")
                    + (bits == 3 ? " (three-bit packing pads its rows)" : " and no " + m.tensorName("bias")));
        }
        long[] bs = m.peek("bias").shape();
        long[] ws = m.peek("W_q").shape();
        if (bs.length != 1 || ws.length != 2 || bs[0] <= 0) {
            throw m.invalid("the weight shape is unknown: " + m.tensorName("bias") + " " + Arrays.toString(bs)
                    + " and " + m.tensorName("W_q") + " " + Arrays.toString(ws) + " give no [N, K]");
        }
        long total = Math.multiplyExact(Math.multiplyExact(ws[0], (long) slots(bits)), ws[1]);
        if (total % bs[0] != 0) {
            throw m.invalid(m.tensorName("W_q") + " " + Arrays.toString(ws) + " holds no whole number of rows of "
                    + bs[0] + " outputs");
        }
        return new long[]{bs[0], total / bs[0]};
    }

    /** Checks the single value of metadata tensor {@code suffix}, when present, against {@code expected}. */
    private static void checkMeta(IntModuleTensors m, String suffix, long expected) throws IOException {
        RowSource t = m.takeOptional(suffix);
        if (t == null) {
            return;
        }
        long v = metaValue(m, t, suffix);
        if (v != expected) {
            throw m.invalid(m.tensorName(suffix) + " is " + v + "; this loader supports " + expected);
        }
    }

    private static long metaValue(IntModuleTensors m, RowSource t, String suffix) throws IOException {
        StDType d = t.format().stDType();
        if (d == StDType.BOOL) {
            ByteBuffer b = IntPacking.rows(t, 0, Math.toIntExact(t.rows()));
            if (b.capacity() != 1) {
                throw m.invalid(m.tensorName(suffix) + " has " + b.capacity() + " values; expected 1");
            }
            return b.get(0) != 0 ? 1 : 0;
        }
        long[] v = d != null && IntPacking.isFloat(d) ? toLongs(m, m.floats(t, suffix), suffix) : m.longs(t, suffix);
        if (v.length != 1) {
            throw m.invalid(m.tensorName(suffix) + " has " + v.length + " values; expected 1");
        }
        return v[0];
    }

    private static long[] toLongs(IntModuleTensors m, float[] f, String suffix) throws IOException {
        long[] out = new long[f.length];
        for (int i = 0; i < f.length; i++) {
            if (f[i] != Math.rint(f[i])) {
                throw m.invalid(m.tensorName(suffix) + " holds " + f[i] + "; expected an integer");
            }
            out[i] = (long) f[i];
        }
        return out;
    }

    /** {@code scale} or {@code zero}: {@code [N*G, 1]} or {@code [N*G]}, row-major {@code [N, G]}. */
    private static IntPacking.ParamTable flatParams(IntModuleTensors m, RowSource src, long n, int groups, String suffix)
            throws IOException {
        long[] s = src.shape();
        long rows = n * groups;
        boolean ok = (s.length == 2 && s[0] == rows && s[1] == 1) || (s.length == 1 && s[0] == rows)
                || (s.length == 2 && s[0] == n && s[1] == groups);
        if (!ok) {
            throw m.invalid(m.tensorName(suffix) + " has shape " + Arrays.toString(s) + "; expected [" + rows
                    + ", 1]");
        }
        return IntPacking.rowMajor(src, n, groups);
    }

    /**
     * The codes of output rows {@code [n0, n0 + count)}: unpacked rows {@code [n0*K/g, (n0+count)*K/g)}, each read
     * from packed row {@code r % step} at the bit offset of slab {@code r / step}.
     */
    private static CodeTable slabs(RowSource wq, int bits, int slots, long step, int g, int k) {
        boolean words = bits == 3;
        int mask = IntPacking.mask(bits);
        return new CodeTable(true, (e, n0, count, out) -> {
            long r = n0 * k / g;
            long end = (n0 + count) * k / g;
            int idx = 0;
            while (r < end) {
                long slab = r / step;
                long p0 = r - slab * step;
                int len = (int) Math.min(end - r, step - p0);
                ByteBuffer buf = IntPacking.rows(wq, p0, len);
                int shift = (int) (slots - 1 - slab) * bits;
                for (int i = 0; i < len * g; i++) {
                    int v = words ? buf.getInt(i * 4) : buf.get(i) & 0xFF;
                    out[idx++] = (byte) ((v >>> shift) & mask);
                }
                r += len;
            }
        });
    }
}
