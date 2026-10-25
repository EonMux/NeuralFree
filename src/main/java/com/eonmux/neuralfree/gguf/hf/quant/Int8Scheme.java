package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.CodeTable;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.ParamTable;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

/**
 * Byte-stored integer weights rewritten into exact ggml repacks:
 *
 * <ul>
 *   <li>{@link QuantFormat#INT8_CHANNEL} (compressed-tensors int8, Quark int8, w8a8_int8): {@code weight} I8
 *       {@code [N, K]} with {@code weight_scale} per tensor ({@code []}, {@code [1]}), per channel ({@code [N]},
 *       {@code [N, 1]}) or per group ({@code [N, K/g]}); an optional {@code weight_zero_point} of the same shape
 *       makes it asymmetric, {@code w = (q - zp) * s}.</li>
 *   <li>{@link QuantFormat#INT8_BLOCK} (blockwise_int8, compressed-tensors block int8): {@code weight} I8
 *       {@code [N, K]} with {@code weight_scale_inv} (or {@code weight_scale}) {@code [ceil(N/bn), ceil(K/bk)]}, a
 *       multiplier per {@code bn x bk} block.</li>
 *   <li>{@link QuantFormat#INT4_IN_INT8}: compressed-tensors W4A8 int, one signed int4 per I8 byte
 *       ({@code weight} or {@code weight_packed} {@code [N, K]}, {@code weight_scale}); and SGLang W4AFP8 experts,
 *       two signed int4 per I8 byte, the even element in the low nibble ({@code weight} {@code [N, K/2]},
 *       {@code weight_scale_inv} {@code [N, K/g]}, w4afp8.py and its pack_int4_values_to_int8 test).</li>
 * </ul>
 *
 * <p>Codes are offset to unsigned ({@code q + 128}, {@code q + 8}) so the shared engine sees the zero point at
 * {@code 2^(b-1)}. Int8 prefers I8_B128 (an f32 multiplier per 128-element run, exact for any f32 scale) when K and
 * the scale runs split into 128-element blocks; otherwise Q8_0 (an fp16 multiplier) or the float fallback.</p>
 */
public final class Int8Scheme {

    private Int8Scheme() {
    }

    /** True for the formats this scheme rewrites. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.INT8_CHANNEL || f == QuantFormat.INT8_BLOCK || f == QuantFormat.INT4_IN_INT8;
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
        IntRewriteBatch.requireFormat(args, handles(args.format()), Int8Scheme.class);
        switch (args.format()) {
            case INT8_CHANNEL:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, Int8Scheme::channel, "weight");
            case INT8_BLOCK:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, Int8Scheme::block, "weight");
            default:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, Int8Scheme::int4, "weight_packed", "weight");
        }
    }

    // ------------------------------------------------------------------ int8

    private static RowSource channel(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        RowSource w = int8Weight(b, m);
        int experts = m.experts(w, 2, "weight");
        long[] ws = m.tail(w, experts, 2, "weight");
        int k = m.width(ws[1], "weight");
        int g = b.groupSize(k);
        int groups = IntWeightSpec.groupCount(k, g);
        RowSource sc = m.take("weight_scale");
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, ws[0], k, 8, g, groups);
        s.codes = bytes(w, ws[0], k, 0x80);
        s.scaleDtype = m.requireFloat(sc, "weight_scale");
        s.scales = m.rowMajor(sc, experts, ws[0], groups, "weight_scale");
        s.zeroDefault = 128;
        s.preferI8B128 = true;
        RowSource zp = m.takeOptional("weight_zero_point");
        if (zp != null) {
            requireIntegerOrFloat(m, zp, "weight_zero_point");
            s.zeros = IntPacking.shifted(m.rowMajor(zp, experts, ws[0], groups, "weight_zero_point"), 128);
        }
        return b.finish(m, s.build());
    }

    private static RowSource block(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        QuantArgs a = b.args();
        RowSource w = int8Weight(b, m);
        int experts = m.experts(w, 2, "weight");
        long[] ws = m.tail(w, experts, 2, "weight");
        long n = ws[0];
        int k = m.width(ws[1], "weight");
        int bn = a.blockN();
        int bk = a.blockK();
        if (bn <= 0 || bk <= 0) {
            throw b.unsupported("int8 block scales without a positive block size (" + bn + "x" + bk + ")");
        }
        String sName = m.has("weight_scale_inv") ? "weight_scale_inv" : "weight_scale";
        RowSource sc = m.take(sName);
        long[] ss = m.tail(sc, experts, 2, sName);
        long blockRows = (n + bn - 1) / bn;
        int groups = IntWeightSpec.groupCount(k, bk);
        if (ss[0] != blockRows || ss[1] != groups) {
            throw m.invalid(m.tensorName(sName) + " has shape " + Arrays.toString(sc.shape()) + "; " + bn + "x" + bk
                    + " blocks of [" + n + ", " + k + "] need [" + blockRows + ", " + groups + "]");
        }
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, 8, bk, groups);
        s.codes = bytes(w, n, k, 0x80);
        s.scaleDtype = m.requireFloat(sc, sName);
        s.scales = blockScales(sc, s.scaleDtype, bn, blockRows, groups);
        s.zeroDefault = 128;
        s.preferI8B128 = true;
        return b.finish(m, s.build());
    }

    private static RowSource int8Weight(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        if (b.args().bits() != 8) {
            throw b.unsupported(b.args().bits() + "-bit weights stored one per byte");
        }
        RowSource w = m.take("weight");
        m.requireDtype(w, "weight", StDType.I8);
        return w;
    }

    /** Block scales {@code [E?, ceil(N/bn), G]}: row {@code n} uses block row {@code n / bn}. */
    private static ParamTable blockScales(RowSource src, StDType dtype, int bn, long blockRows, int groups) {
        return new ParamTable(true, (e, n0, count, out) -> {
            long b0 = n0 / bn;
            long b1 = (n0 + count - 1) / bn;
            float[] buf = new float[(int) (b1 - b0 + 1) * groups];
            IntPacking.readFloats(src, dtype, (e * blockRows + b0) * groups, buf.length, buf, 0);
            for (int i = 0; i < count; i++) {
                System.arraycopy(buf, (int) ((n0 + i) / bn - b0) * groups, out, i * groups, groups);
            }
        });
    }

    // ------------------------------------------------------------------ int4 stored in int8

    private static RowSource int4(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        if (b.args().bits() != 4) {
            throw b.unsupported(b.args().bits() + "-bit values stored in int8");
        }
        boolean twoPerByte = m.has("weight_scale_inv");
        String wName = m.has("weight_packed") ? "weight_packed" : "weight";
        String sName = twoPerByte ? "weight_scale_inv" : "weight_scale";
        RowSource w = m.take(wName);
        m.requireDtype(w, wName, StDType.I8);
        int experts = m.experts(w, 2, wName);
        long[] ws = m.tail(w, experts, 2, wName);
        long n = ws[0];
        int k = m.width(twoPerByte ? ws[1] * 2 : ws[1], wName);
        int g = b.groupSize(k);
        int groups = IntWeightSpec.groupCount(k, g);
        RowSource sc = m.take(sName);
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, 4, g, groups);
        s.codes = twoPerByte ? signedNibbles(w, n, k) : bytes(w, n, k, 0x08);
        s.scaleDtype = m.requireFloat(sc, sName);
        s.scales = m.rowMajor(sc, experts, n, groups, sName);
        s.zeroDefault = 8;
        RowSource zp = m.takeOptional("weight_zero_point");
        if (zp != null) {
            requireIntegerOrFloat(m, zp, "weight_zero_point");
            s.zeros = IntPacking.shifted(m.rowMajor(zp, experts, n, groups, "weight_zero_point"), 8);
        }
        return b.finish(m, s.build());
    }

    /**
     * One signed value per I8 byte, offset by {@code offset} (128 for int8, 8 for int4): a byte outside the
     * {@code b}-bit range yields a code the engine rejects.
     */
    private static CodeTable bytes(RowSource w, long n, int k, int offset) {
        return new CodeTable(true, (e, n0, count, out) -> {
            ByteBuffer buf = IntPacking.rows(w, e * n + n0, count);
            for (int i = 0; i < count * k; i++) {
                out[i] = (byte) (buf.get(i) + offset);
            }
        });
    }

    /** Two signed int4 per I8 byte, element {@code 2j} in the low nibble; code {@code nibble XOR 8}. */
    private static CodeTable signedNibbles(RowSource w, long n, int k) {
        return new CodeTable(true, (e, n0, count, out) -> {
            ByteBuffer buf = IntPacking.rows(w, e * n + n0, count);
            for (int i = 0; i < count * k; i += 2) {
                int v = buf.get(i / 2);
                out[i] = (byte) ((v & 0xF) ^ 0x8);
                out[i + 1] = (byte) (((v >>> 4) & 0xF) ^ 0x8);
            }
        });
    }

    private static void requireIntegerOrFloat(IntModuleTensors m, RowSource src, String suffix) throws IOException {
        StDType d = src.format().stDType();
        if (d == null || !(IntPacking.isInteger(d) || IntPacking.isFloat(d))) {
            throw m.invalid(m.tensorName(suffix) + " is " + src.format() + "; expected an integer or float dtype");
        }
    }
}
