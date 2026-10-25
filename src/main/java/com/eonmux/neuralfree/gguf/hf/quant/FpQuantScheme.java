package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Geometry;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Module;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Notes;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Produced;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Rewrite;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Scheme.Bytes;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Rewrites fp_quant modules. The weights were quantized after a Hadamard rotation of each
 * {@code hadamard_group_size} block of the input dimension, so every module also needs that rotation of its input
 * ({@link HadamardTransforms#fpQuant}); this class validates the stored {@code forward_hadamard_matrix} against it.
 * <ul>
 *   <li>Real MXFP4: {@code qweight} U8 {@code [N, K/2]} (low nibble first) and E8M0 {@code scales [N, K/32]}: MXFP4
 *       with the output multiplier {@code 1 / weight_global_scale}.</li>
 *   <li>Real NVFP4: E4M3 {@code scales [N, K/16]}: NVFP4 with the output multiplier
 *       {@code 1 / weight_global_scale}.</li>
 *   <li>Pseudo-quantized: the dequantized rotated weight {@code dqweight} loads as stored.</li>
 * </ul>
 * One quantizer turns both the weight and the activation into codes that hold the value times its global scale
 * (explicitly for NVFP4; for MXFP4 by the method's fixed factor, which the global scales record), and the GEMM
 * multiplies by {@code 1 / (weight_global_scale * act_global_scale)} (vLLM fp_quant.py:358-395). A 16-bit activation
 * carries no factor, so the weight alone needs {@code 1 / weight_global_scale}.
 * fp_quant activation quantization ({@code quest}, {@code abs_max}) is not emulated: these modules compute
 * {@code W * A16} on the rotated input.
 */
public final class FpQuantScheme {

    private static final Logger LOG = LogManager.getLogger(FpQuantScheme.class);

    private static final String[] AUX = {"weight_global_scale", "act_global_scale", "forward_hadamard_matrix",
        "backward_hadamard_matrix"};

    private FpQuantScheme() {
    }

    /** True for the fp_quant formats. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.FPQUANT_MXFP4 || f == QuantFormat.FPQUANT_NVFP4 || f == QuantFormat.FPQUANT_PSEUDO;
    }

    /**
     * Rewrites every module of {@code modules} (HF module prefixes) in {@code in}.
     *
     * @throws GgufFormatException for {@code transform_init: gsr}, a stored rotation that is not the configured one,
     *                             or missing and inconsistent tensors
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
            throw new IllegalArgumentException("FpQuantScheme does not handle " + args.format());
        }
        if (args.fpQuantTransform() == QuantArgs.FpQuantTransform.GSR) {
            throw new GgufFormatException("fp_quant transform_init 'gsr' is not supported");
        }
        if (args.fpQuantHadamardGroup() <= 0) {
            throw new GgufFormatException("fp_quant modules need a positive hadamard_group_size, got "
                    + args.fpQuantHadamardGroup());
        }
        Rewrite rw = new Rewrite();
        Notes notes = new Notes();
        for (String module : modules) {
            Module m = new Module(in, module);
            checkRotation(args, m, in);
            if (args.format() == QuantFormat.FPQUANT_PSEUDO) {
                pseudo(args, m, in, rw);
            } else {
                real(args, m, in, o, rw, notes);
            }
            rw.consume(m, AUX);
        }
        if (args.actUnemulated() != null && !modules.isEmpty()) {
            LOG.warn("{} are not emulated; {} {} module(s) compute W x A16 on the rotated input",
                    args.actUnemulated(), modules.size(), args.format());
        }
        notes.log(LOG);
        return rw.apply(in);
    }

    /** Validates the stored rotation matrices against {@code transform_init} and the group size. */
    private static void checkRotation(QuantArgs args, Module m, HfTensorIndex in) throws IOException {
        int hg = args.fpQuantHadamardGroup();
        boolean hadamard = args.fpQuantTransform() != QuantArgs.FpQuantTransform.IDENTITY;
        String fName = m.name("forward_hadamard_matrix");
        if (fName == null) {
            throw new GgufFormatException("fp_quant module " + m.prefix() + " has no forward_hadamard_matrix");
        }
        RowSource f = in.get(fName);
        if (!Arrays.equals(f.shape(), new long[]{hg, hg})) {
            throw new GgufFormatException("fp_quant matrix " + fName + " has shape " + Arrays.toString(f.shape())
                    + "; expected [" + hg + ", " + hg + "]");
        }
        HadamardTransforms.checkFpQuantMatrix(Fp4Packing.readScales(f, fName, false), hg, hadamard, fName);
        String bName = m.name("backward_hadamard_matrix");
        if (bName != null && !Arrays.equals(in.get(bName).shape(), new long[]{hg, hg})) {
            throw new GgufFormatException("fp_quant matrix " + bName + " has shape "
                    + Arrays.toString(in.get(bName).shape()) + "; expected [" + hg + ", " + hg + "]");
        }
    }

    private static void checkWidth(QuantArgs args, Geometry g, String wName) {
        if (g.k % args.fpQuantHadamardGroup() != 0) {
            throw new GgufFormatException("fp_quant weight " + wName + " has K = " + g.k + ", not a multiple of the"
                    + " hadamard_group_size " + args.fpQuantHadamardGroup());
        }
    }

    private static void pseudo(QuantArgs args, Module m, HfTensorIndex in, Rewrite rw) {
        String dName = m.name("dqweight");
        if (dName == null) {
            throw new GgufFormatException("fp_quant pseudo-quantized module " + m.prefix() + " has no dqweight");
        }
        RowSource d = in.get(dName);
        StDType t = Fp4Packing.dtype(d, dName);
        if (t != StDType.BF16 && t != StDType.F16 && t != StDType.F32) {
            throw new GgufFormatException("fp_quant dqweight " + dName + " has dtype " + t.headerName()
                    + "; expected a float dtype");
        }
        Geometry g = Geometry.of(d.shape(), dName);
        checkWidth(args, g, dName);
        String out = m.prefix() + ".weight";
        if (in.has(out)) {
            throw new GgufFormatException("fp_quant module " + m.prefix() + " has both " + out + " and " + dName);
        }
        rw.put(out, d);
        rw.consume(dName);
    }

    private static void real(QuantArgs args, Module m, HfTensorIndex in, ModelSourceOptions o, Rewrite rw,
                             Notes notes) throws IOException {
        boolean mx = args.format() == QuantFormat.FPQUANT_MXFP4;
        String wName = m.name("qweight");
        String sName = m.name("scales");
        if (wName == null || sName == null) {
            throw new GgufFormatException("fp_quant module " + m.prefix() + " needs qweight and scales");
        }
        RowSource w = in.get(wName);
        RowSource s = in.get(sName);
        if (Fp4Packing.dtype(w, wName) != StDType.U8) {
            throw new GgufFormatException("fp_quant qweight " + wName + " must be packed U8");
        }
        StDType st = Fp4Packing.dtype(s, sName);
        if (st != StDType.U8 && st != (mx ? StDType.F8_E8M0 : StDType.F8_E4M3)) {
            throw new GgufFormatException("fp_quant scales " + sName + " have dtype " + st.headerName());
        }
        Geometry g = Geometry.packed(w.shape(), 1, 2, wName);
        if (g.experts != 0) {
            throw new GgufFormatException("fp_quant qweight " + wName + " must be [N, K/2]");
        }
        checkWidth(args, g, wName);
        int group = mx ? 32 : 16;
        if (g.k % group != 0) {
            throw new GgufFormatException("fp_quant weight " + wName + " has K = " + g.k + ", not a multiple of "
                    + group);
        }
        g.checkRows(s, sName, g.k / group);
        float wgs = scalar(m, in, "weight_global_scale");
        scalar(m, in, "act_global_scale");
        float[] mult = {1.0f / wgs};
        if (!Float.isFinite(mult[0])) {
            throw new GgufFormatException("fp_quant module " + m.prefix() + ": the weight global scale " + wgs
                    + " gives the non-finite output scale " + mult[0]);
        }
        Bytes packed = (f, c) -> Fp4Packing.readRows(w, f, c);
        Bytes scales = (f, c) -> Fp4Packing.readRows(s, f, c);
        boolean rowLocal = w.rowLocal() && s.rowLocal();
        StDType fb = Fp4Packing.fallbackType(StDType.U8, o);
        String prefix = m.prefix();
        GGMLType target = mx ? GGMLType.MXFP4 : GGMLType.NVFP4;
        Produced src;
        if (mx) {
            src = Produced.of(describe(args, prefix, target), ElementFormat.of(target), g.shape, rowLocal,
                    Fp4Scheme.mxfp4Writer(g, packed, scales, sName)).withOutputScale(mult, () -> {
                        LOG.warn("{}: the fp_quant output scale has no sibling slot; the weight loads as {}", prefix,
                                fb);
                        return Fp4Scheme.mxfp4Floats(describe(args, prefix, fb), g, packed, scales, sName, mult, fb,
                                rowLocal);
                    });
        } else if (g.k % 64 == 0) {
            src = Produced.of(describe(args, prefix, target), ElementFormat.of(target), g.shape, rowLocal,
                    Fp4Scheme.nvfp4Writer(g, packed, scales, sName)).withOutputScale(mult, () -> {
                        LOG.warn("{}: the fp_quant output scale has no sibling slot; the weight loads as {}", prefix,
                                fb);
                        return Fp4Scheme.nvfp4Floats(describe(args, prefix, fb), g, packed, scales, sName, mult, fb,
                                rowLocal);
                    });
        } else {
            notes.fallback("fp_quant NVFP4 weights with K % 64 != 0 load as the float fallback", prefix);
            src = Fp4Scheme.nvfp4Floats(describe(args, prefix, fb), g, packed, scales, sName, mult, fb, rowLocal);
        }
        String out = prefix + ".weight";
        if (in.has(out)) {
            throw new GgufFormatException("fp_quant module " + prefix + " has both " + out + " and " + wName);
        }
        rw.put(out, src);
        rw.consume(wName);
        rw.consume(sName);
    }

    private static String describe(QuantArgs args, String module, Object target) {
        return Fp4Packing.describe(module, args.format(), args.bits(), args.groupSize(), target);
    }

    private static float scalar(Module m, HfTensorIndex in, String suffix) throws IOException {
        RowSource t = m.require(suffix);
        float v = Fp4Packing.readPerExpert(t, m.name(suffix), 0)[0];
        if (!(v > 0) || Float.isInfinite(v)) {
            throw new GgufFormatException("fp_quant " + m.name(suffix) + " is " + v + "; expected a positive finite"
                    + " number");
        }
        return v;
    }
}
