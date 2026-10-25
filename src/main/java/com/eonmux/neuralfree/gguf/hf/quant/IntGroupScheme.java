package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.Layout;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * GPTQ, compressed-tensors {@code pack-quantized}, AutoAWQ and Quark int4 checkpoints rewritten into exact ggml
 * repacks. Like the other integer schemes, this one describes each quantized module as an {@link IntWeightSpec} and
 * hands it to the shared rewrite ({@link IntRewriteBatch}), which chooses the target and makes the module one lazy
 * {@link IntGroupSource} of {@code w[n][k] = (q - z) * s} (or {@code q * s + bias}) over the stored codes, scales and
 * zero points.
 *
 * <ul>
 *   <li>GPTQ ({@link QuantFormat#INT_GPTQ}; AutoRound {@code auto_gptq}, {@code moe_wna16}): {@code qweight} I32
 *       {@code [Kw, N]} packed along K, {@code qzeros} I32 {@code [G, Nw]} packed along N, {@code scales}
 *       {@code [G, N]}, optional {@code g_idx} I32 {@code [K]}; {@code z = qzeros + 1} for GPTQ v1 and
 *       {@code qzeros} for v2 (llama.cpp conversion/base.py:366-407).</li>
 *   <li>compressed-tensors ({@link QuantFormat#INT_PACKED_CT}): {@code weight_packed} I32 rows packed along K with
 *       values {@code u = q + 2^(b-1)}, {@code weight_scale} {@code [N, G]}, {@code [N, 1]} or {@code [1]},
 *       {@code weight_zero_point} I32 packed along N {@code [Nw, G]}, {@code weight_shape} {@code [N, K]}
 *       (authoritative, trims padding) and optional {@code weight_g_idx} (base.py:409-442).</li>
 *   <li>AutoAWQ ({@link QuantFormat#INT_AWQ}): {@code qweight} I32 {@code [K, N/8]} and {@code qzeros} I32
 *       {@code [G, N/8]} in AWQ nibble order, {@code scales} {@code [G, N]} (auto_awq.py:72-76).</li>
 *   <li>Quark int4 ({@link QuantFormat#INT_QUARK}): {@code weight} I32 {@code [K, N/8]} in AWQ order under
 *       {@code pack_method: reorder}, else sequential; symmetric ({@code int4}) codes and zero points are signed
 *       nibbles, so {@code nibble XOR 8} is the offset code; {@code weight_scale} {@code [G, N]}.</li>
 * </ul>
 *
 * <p>Targets, chosen per module from the stored values: with every zero point at {@code 2^(b-1)} (symmetric),
 * Q4_0 for 4 bits, Q5_0 for 5, Q2_0 for 2 when groups split into 64-element blocks, I8_B128 for int8 checkpoints
 * whose scale runs split into 128-element blocks, and Q8_0 otherwise; asymmetric 4 and 5 bits become Q4_1 and Q5_1
 * ({@code m = fp16(-z*s)}, approximate), and under exact quantization, like the other asymmetric widths, Q8_0 when
 * every zero point is an integer keeping {@code q - z} in int8. K or groups that do not split into 32-element blocks,
 * zero points outside that range, and (under exact quantization) scales that fp16 cannot hold take the float
 * fallback: F32 under exact quantization, else the scales' float type, else BF16. Each fallback is logged once per
 * reason.</p>
 *
 * <p>Symmetry is decided by the stored zero points, not by the configuration flag. Act-order ({@code g_idx} other
 * than {@code k / g}) yields a target on the input dimension reordered by the stable argsort of {@code g_idx}, with
 * {@link RowSource#inputPerm()} set, also for modules behind an online input transform: the runtime applies the
 * input transform, then the activation quantization, then the permutation (vLLM's order). Stacked experts, which have
 * no input permutation, take the float fallback in the original order instead.</p>
 */
public final class IntGroupScheme {

    private IntGroupScheme() {
    }

    /** True for the formats this scheme rewrites. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.INT_GPTQ || f == QuantFormat.INT_PACKED_CT || f == QuantFormat.INT_AWQ
                || f == QuantFormat.INT_QUARK;
    }

    /**
     * Replaces each module's quantized tensors in {@code in} by one virtual weight source: {@code <module>.weight}
     * for dotted checkpoint names, {@code <module>} for fused-expert underscore names. {@code <module>.bias} is kept.
     *
     * @param modules HF module prefixes, for example {@code model.layers.3.self_attn.q_proj}
     * @throws IOException                when a tensor is missing, has the wrong dtype or shape, or holds invalid values
     * @throws GgufFormatException        for a configuration this scheme cannot honour
     * @throws IllegalArgumentException   when {@code args} is not a format of this scheme
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        IntRewriteBatch.requireFormat(args, handles(args.format()), IntGroupScheme.class);
        switch (args.format()) {
            case INT_GPTQ:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, IntGroupScheme::gptq, "qweight");
            case INT_PACKED_CT:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, IntGroupScheme::packed, "weight_packed");
            case INT_AWQ:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, (b, m) -> nibbles(b, m, false), "qweight");
            default:
                return IntRewriteBatch.rewrite(args, modules, cfg, in, o, (b, m) -> nibbles(b, m, true), "weight");
        }
    }

    // ------------------------------------------------------------------ GPTQ

    private static RowSource gptq(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        int bits = b.args().bits();
        b.requireBits(2, 8);
        RowSource qw = m.take("qweight");
        m.requireDtype(qw, "qweight", StDType.I32);
        int experts = m.experts(qw, 2, "qweight");
        long[] qs = m.tail(qw, experts, 2, "qweight");
        long n = qs[1];
        RowSource gi = m.takeOptional("g_idx");
        int k = gi != null ? m.width(m.tail(gi, experts, 1, "g_idx")[0], "g_idx") : inferK(m, qs[0], bits);
        Layout lk = m.layout(k, qs[0], bits, "qweight", "K");
        int g = b.groupSize(k);
        int groups = IntWeightSpec.groupCount(k, g);
        RowSource sc = m.take("scales");
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, bits, g, groups);
        s.codes = IntPacking.columnPacked(qw, Math.toIntExact(qs[0]), n, k, bits, lk);
        s.scaleDtype = m.requireFloat(sc, "scales");
        s.scales = m.groupMajor(sc, experts, n, groups, "scales");
        s.zeroDefault = 1 << (bits - 1);
        RowSource qz = m.takeOptional("qzeros");
        if (qz != null) {
            m.requireDtype(qz, "qzeros", StDType.I32);
            long[] zs = m.tail(qz, experts, 2, "qzeros");
            m.requireDim(zs[0], groups, "qzeros", "groups");
            Layout lz = m.layout(n, zs[1], bits, "qzeros", "N");
            s.zeros = IntPacking.packedRows(qz, groups, Math.toIntExact(zs[1]), bits, lz,
                    b.args().gptqVersion() == 1 ? 1 : 0);
        }
        if (gi != null) {
            s.gidx = m.groupIndex(gi, experts, k, groups, "g_idx");
        }
        return b.finish(m, s.build());
    }

    /**
     * K of a GPTQ {@code qweight} without {@code g_idx}: {@code rows * 32 / b}, exact for the per-word layout when
     * {@code b} divides 32 and for GPTQ's bit-stream layout otherwise.
     */
    private static int inferK(IntModuleTensors m, long rows, int bits) throws IOException {
        long total = Math.multiplyExact(rows, 32L);
        if (total % bits != 0) {
            throw m.invalid(m.tensorName("qweight") + " has " + rows + " packed rows, which hold no whole number of "
                    + bits + "-bit values, and there is no g_idx giving K");
        }
        return m.width(total / bits, "qweight");
    }

    // ------------------------------------------------------------------ compressed-tensors pack-quantized

    private static RowSource packed(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        int bits = b.args().bits();
        b.requireBits(1, 8);
        RowSource wp = m.take("weight_packed");
        m.requireDtype(wp, "weight_packed", StDType.I32);
        int experts = m.experts(wp, 2, "weight_packed");
        long[] ps = m.tail(wp, experts, 2, "weight_packed");
        long[] shape = weightShape(m, m.take("weight_shape"), experts);
        long capacityK = Math.max(ps[1] * IntPacking.perWord(bits), ps[1] * 32 / bits);
        if (shape[0] <= 0 || shape[1] <= 0 || shape[0] > ps[0] || shape[1] > capacityK) {
            throw m.invalid(m.tensorName("weight_shape") + " " + Arrays.toString(shape) + " is outside the capacity ["
                    + ps[0] + ", " + capacityK + "] of weight_packed " + Arrays.toString(wp.shape()));
        }
        long n = shape[0];
        int k = m.width(shape[1], "weight_shape");
        Layout lk = m.layout(k, ps[1], bits, "weight_packed", "K");
        int g = b.groupSize(k);
        int groups = IntWeightSpec.groupCount(k, g);
        RowSource sc = m.take("weight_scale");
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, bits, g, groups);
        s.codes = IntPacking.rowPacked(wp, ps[0], k, bits, lk);
        s.scaleDtype = m.requireFloat(sc, "weight_scale");
        s.scales = m.rowMajor(sc, experts, n, groups, "weight_scale");
        s.zeroDefault = 1 << (bits - 1);
        RowSource zp = m.takeOptional("weight_zero_point");
        if (zp != null) {
            m.requireDtype(zp, "weight_zero_point", StDType.I32);
            long[] zs = m.tail(zp, experts, 2, "weight_zero_point");
            m.requireDim(zs[1], groups, "weight_zero_point", "groups");
            Layout lz = m.layout(n, zs[0], bits, "weight_zero_point", "N");
            s.zeros = IntPacking.packedColumns(zp, zs[0], groups, bits, lz);
        } else if (!b.args().symmetric()) {
            throw m.invalid("asymmetric quantization without " + m.tensorName("weight_zero_point"));
        }
        RowSource gi = m.takeOptional("weight_g_idx");
        if (gi != null) {
            m.requireDim(m.tail(gi, experts, 1, "weight_g_idx")[0], k, "weight_g_idx", "K");
            s.gidx = m.groupIndex(gi, experts, k, groups, "weight_g_idx");
        }
        return b.finish(m, s.build());
    }

    /** {@code weight_shape}: {@code [N, K]}, or one identical {@code [N, K]} per expert. */
    private static long[] weightShape(IntModuleTensors m, RowSource ws, int experts) throws IOException {
        long[] v = m.longs(ws, "weight_shape");
        int rows = Math.max(1, experts);
        if (v.length != 2 * rows) {
            throw m.invalid(m.tensorName("weight_shape") + " has " + v.length + " values; expected " + 2 * rows);
        }
        for (int e = 1; e < rows; e++) {
            if (v[2 * e] != v[0] || v[2 * e + 1] != v[1]) {
                throw m.invalid(m.tensorName("weight_shape") + " differs between experts 0 and " + e);
            }
        }
        return new long[]{v[0], v[1]};
    }

    // ------------------------------------------------------------------ AWQ and Quark int4

    private static RowSource nibbles(IntRewriteBatch b, IntModuleTensors m, boolean quark) throws IOException {
        QuantArgs a = b.args();
        if (a.bits() != 4) {
            throw b.unsupported((quark ? "Quark" : "AWQ") + " packing of " + a.bits() + "-bit weights");
        }
        String wName = quark ? "weight" : "qweight";
        String sName = quark ? "weight_scale" : "scales";
        String zName = quark ? "weight_zero_point" : "qzeros";
        RowSource qw = m.take(wName);
        m.requireDtype(qw, wName, StDType.I32);
        int experts = m.experts(qw, 2, wName);
        long[] qs = m.tail(qw, experts, 2, wName);
        int k = m.width(qs[0], wName);
        RowSource sc = m.take(sName);
        long[] scaleShape = sc.shape();
        if (scaleShape.length == 0) {
            throw m.invalid(m.tensorName(sName) + " is a scalar; expected [G, N]");
        }
        long n = scaleShape[scaleShape.length - 1];
        if (n <= 0) {
            throw m.invalid(m.tensorName(sName) + " has shape " + Arrays.toString(scaleShape) + "; expected [G, N]");
        }
        m.requireDim(qs[1], (n + 7) / 8, wName, "words along N");
        int g = b.groupSize(k);
        int groups = IntWeightSpec.groupCount(k, g);
        boolean awqOrder = !quark || a.quarkPackReorder();
        int xor = quark && a.symmetric() ? 0x8 : 0;
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, 4, g, groups);
        s.codes = IntPacking.nibblesAlongN(qw, k, Math.toIntExact(qs[1]), awqOrder, xor);
        s.scaleDtype = m.requireFloat(sc, sName);
        s.scales = m.groupMajor(sc, experts, n, groups, sName);
        s.zeroDefault = 8;
        RowSource zp = m.takeOptional(zName);
        if (zp != null) {
            m.requireDtype(zp, zName, StDType.I32);
            long[] zShape = zp.shape();
            boolean perChannel = groups == 1 && zShape.length == (experts > 0 ? 2 : 1);
            if (perChannel && experts > 0) {
                m.requireDim(zShape[0], experts, zName, "experts");
            }
            long[] zs = perChannel ? new long[]{1, zShape[zShape.length - 1]} : m.tail(zp, experts, 2, zName);
            m.requireDim(zs[0], groups, zName, "groups");
            m.requireDim(zs[1], qs[1], zName, "words along N");
            s.zeros = IntPacking.nibbleRows(zp, groups, Math.toIntExact(zs[1]), awqOrder, xor);
        } else if (!a.symmetric()) {
            throw m.invalid("asymmetric quantization without " + m.tensorName(zName));
        }
        return b.finish(m, s.build());
    }
}
