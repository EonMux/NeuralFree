package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntWeightSpec.Formula;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.Layout;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.util.List;

/**
 * MLX affine quantization ({@link QuantFormat#INT_MLX}) rewritten into ggml types: {@code weight} U32
 * {@code [N, K*b/32]} with the {@code b}-bit codes packed least significant first along K, {@code scales} and
 * {@code biases} {@code [N, K/g]}, and {@code w = q * scale + bias} (mlx affine_dequantize). Four-bit weights become
 * Q4_1 ({@code d = scale}, {@code m = bias}), exact when both are representable in fp16 (ggml computes
 * {@code q * d + m} in f32 as MLX does); other widths take the float fallback.
 */
public final class MlxScheme {

    private MlxScheme() {
    }

    /** True for the formats this scheme rewrites. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.INT_MLX;
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
        IntRewriteBatch.requireFormat(args, handles(args.format()), MlxScheme.class);
        return IntRewriteBatch.rewrite(args, modules, cfg, in, o, MlxScheme::affine, "weight");
    }

    private static RowSource affine(IntRewriteBatch b, IntModuleTensors m) throws IOException {
        QuantArgs a = b.args();
        int bits = a.bits();
        b.requireBits(1, 8);
        if (a.groupSize() <= 0) {
            throw b.unsupported("MLX quantization without a positive group size");
        }
        RowSource w = m.take("weight");
        m.requireDtype(w, "weight", StDType.U32, StDType.I32);
        int experts = m.experts(w, 2, "weight");
        long[] ws = m.tail(w, experts, 2, "weight");
        long n = ws[0];
        long total = Math.multiplyExact(ws[1], 32L);
        if (total % bits != 0) {
            throw m.invalid(m.tensorName("weight") + " has " + ws[1] + " words per row, which hold no whole number of "
                    + bits + "-bit values");
        }
        int k = m.width(total / bits, "weight");
        Layout layout = m.layout(k, ws[1], bits, "weight", "K");
        int g = a.groupSize();
        if (k % g != 0) {
            throw m.invalid("K " + k + " is not a multiple of the group size " + g);
        }
        int groups = k / g;
        RowSource sc = m.take("scales");
        RowSource bi = m.take("biases");
        IntWeightSpec.Builder s = IntWeightSpec.builder(m, experts, n, k, bits, g, groups);
        s.codes = IntPacking.rowPacked(w, n, k, bits, layout);
        s.scaleDtype = m.requireFloat(sc, "scales");
        s.scales = m.rowMajor(sc, experts, n, groups, "scales");
        m.requireFloat(bi, "biases");
        s.zeros = m.rowMajor(bi, experts, n, groups, "biases");
        s.formula = Formula.MUL_ADD;
        return b.finish(m, s.build());
    }
}
