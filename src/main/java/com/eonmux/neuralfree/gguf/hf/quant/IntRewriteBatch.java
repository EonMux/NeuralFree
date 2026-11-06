package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantRule.ModuleMatcher;
import com.eonmux.neuralfree.gguf.hf.quant.IntGroupSource.Act;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.ParamTable;
import com.eonmux.neuralfree.gguf.hf.quant.IntWeightSpec.Formula;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The rewrite engine the integer schemes share ({@link IntGroupScheme}, {@link Int8Scheme}, {@link MlxScheme},
 * {@link HqqScheme}, {@link BitnetScheme}). One instance is one {@code rewriteModules} call: its arguments, the
 * replacements and removals it applies to the index together, the ggml target chosen per module from the stored
 * values, and the fallbacks and notices it logs once each with the modules they concern.
 */
final class IntRewriteBatch {

    private static final Logger LOG = LogManager.getLogger(IntRewriteBatch.class);

    /** Values read per pass when a rewrite scans scales or zero points. */
    private static final int SCAN_VALUES = 1 << 20;

    private final QuantArgs args;
    private final HfConfig cfg;
    private final HfTensorIndex in;
    private final ModelSourceOptions options;
    private final List<HfTransformConfig.Scheme> transforms;
    private final Map<String, RowSource> replace = new LinkedHashMap<>();
    private final Set<String> remove = new LinkedHashSet<>();
    private final Map<String, List<String>> warnings = new LinkedHashMap<>();

    private IntRewriteBatch(QuantArgs args, HfConfig cfg, HfTensorIndex in, ModelSourceOptions options,
                  List<HfTransformConfig.Scheme> transforms) {
        this.args = args;
        this.cfg = cfg;
        this.in = in;
        this.options = options;
        this.transforms = transforms;
    }

    /** @throws IllegalArgumentException when {@code handled} is false */
    static void requireFormat(QuantArgs args, boolean handled, Class<?> scheme) {
        Objects.requireNonNull(args, "args");
        if (!handled) {
            throw new IllegalArgumentException(scheme.getSimpleName() + " does not rewrite " + args.format());
        }
    }

    /**
     * Resolves each module by the first of its {@code primaries} tensors present, builds its weight source with
     * {@code builder}, and returns {@code in} with every module's consumed tensors replaced by its source.
     */
    static HfTensorIndex rewrite(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                 ModelSourceOptions o, WeightBuilder builder, String... primaries)
            throws IOException {
        Objects.requireNonNull(modules, "modules");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(o, "options");
        List<HfTransformConfig.Scheme> transforms = HfQuantConfig.of(cfg).transforms()
                .map(HfTransformConfig::schemes).orElse(Collections.emptyList());
        IntRewriteBatch b = new IntRewriteBatch(args, cfg, in, o, transforms);
        for (String module : modules) {
            IntModuleTensors m = IntModuleTensors.resolve(b, module, primaries);
            RowSource src = builder.build(b, m);
            b.replace.put(m.outputName(), src);
            b.remove.addAll(m.consumed());
        }
        b.remove.removeAll(b.replace.keySet());
        b.logWarnings();
        return b.replace.isEmpty() ? in : in.withReplaced(b.replace, b.remove);
    }

    QuantArgs args() {
        return args;
    }

    ModelSourceOptions options() {
        return options;
    }

    HfTensorIndex index() {
        return in;
    }

    /** A container error naming the checkpoint directory and the module. */
    IOException invalid(String module, String message) {
        return new IOException("Invalid Hugging Face checkpoint '" + cfg.dir() + "': " + args.format()
                + " module " + module + ": " + message);
    }

    /** A configuration this scheme cannot honour. */
    GgufFormatException unsupported(String what) {
        return new GgufFormatException(args.format() + " quantization: " + what + " is not supported");
    }

    void requireBits(int lo, int hi) {
        if (args.bits() < lo || args.bits() > hi) {
            throw unsupported(args.bits() + "-bit weights");
        }
    }

    /** The group size along K: the configured one, or K for channel and tensor quantization. */
    int groupSize(int k) {
        return args.groupSize() > 0 ? args.groupSize() : k;
    }

    /** Records a fallback or notice, logged once per reason when the batch ends. */
    void warn(String reason, String module) {
        warnings.computeIfAbsent(reason, r -> new ArrayList<>()).add(module);
    }

    private void logWarnings() {
        for (Map.Entry<String, List<String>> e : warnings.entrySet()) {
            LOG.warn("{}: {} ({} module(s), first {})", args.format(), e.getKey(), e.getValue().size(),
                    e.getValue().get(0));
        }
    }

    /**
     * Checks the online compressed-tensors transforms that apply to {@code module}. One at the module's input needs
     * nothing from the rewrite: the runtime applies it before the activation quantization and the act-order
     * permutation ({@code InputTransform}), so act-order modules keep their permuted target. Every module is checked
     * when it is resolved.
     *
     * @throws GgufFormatException when an online transform applies at its output, the key cache or the
     *                             attention query, which the runtime does not apply
     */
    void checkOnlineTransforms(String module) {
        for (HfTransformConfig.Scheme s : transforms) {
            for (HfTransformConfig.Apply a : s.applies()) {
                if (!a.location().isOnline() || !matchesAny(a.targets(), module) || matchesAny(a.ignore(), module)) {
                    continue;
                }
                if (a.location() != HfTransformConfig.Location.INPUT) {
                    throw new GgufFormatException("compressed-tensors transform " + s.type() + "@"
                            + a.location().name().toLowerCase(Locale.ROOT) + " is not supported");
                }
            }
        }
    }

    private static boolean matchesAny(List<String> targets, String module) {
        for (String t : targets) {
            if (ModuleMatcher.target(t).matches(module)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The source of {@code spec}: the act-order permutation, the ggml target chosen from the stored values, and
     * the activation siblings of the module.
     */
    RowSource finish(IntModuleTensors m, IntWeightSpec spec) throws IOException {
        return finish(m, spec, null, null);
    }

    /**
     * {@link #finish(IntModuleTensors, IntWeightSpec)} with the per-column activation pre-scale {@code preScale}, which
     * the runtime applies to the matmul input before its quantization ({@link RowSource#inputPreScale()}).
     */
    RowSource finish(IntModuleTensors m, IntWeightSpec spec, float[] preScale) throws IOException {
        return finish(m, spec, preScale, null);
    }

    /** {@code act} when the caller already resolved the activation siblings, else they are read here. */
    private RowSource finish(IntModuleTensors m, IntWeightSpec spec, float[] preScale, Act act) throws IOException {
        int[] perm = null;
        String reason = null;
        if (spec.gidx != null) {
            int[][] perms = new int[spec.gidx.length][];
            boolean permuted = false;
            for (int e = 0; e < perms.length; e++) {
                perms[e] = IntPacking.stableArgsort(spec.gidx[e], spec.groups);
                permuted |= !isIdentity(perms[e]);
            }
            if (permuted && spec.experts > 0) {
                reason = "act-order (g_idx) on stacked experts, which have no input permutation";
            } else if (permuted) {
                perm = perms[0];
            }
        }
        GGMLType fallback = fallbackType(spec.scaleDtype);
        Choice c = reason != null ? new Choice(fallback, reason) : choose(spec, Stats.scan(this, m, spec), perm);
        if (c.reason != null) {
            warn("float fallback " + c.target + ": " + c.reason, m.name());
        }
        Act a = act != null ? act : activation(m, spec.experts);
        return new IntGroupSource(spec, c.target, fallback, c.target.isQuantized() ? perm : null, a,
                cfg.dir().toString(), null, preScale);
    }

    /** The float fallback type: F32 under exact quantization, else the scales' float type, else BF16. */
    GGMLType fallbackType(StDType scaleDtype) {
        if (options.exactQuant() || scaleDtype == StDType.F32 || scaleDtype == StDType.F64) {
            return GGMLType.F32;
        }
        return scaleDtype == StDType.F16 ? GGMLType.F16 : GGMLType.BF16;
    }

    /**
     * The source of a weight carrying a per-column activation pre-scale. Without activation quantization the scale
     * is folded into an F32 weight, for which {@code q * s + z} times the column multiplier is exact. With an
     * activation scheme the weight keeps its own quantization and the pre-scale travels as
     * {@link RowSource#inputPreScale()}, because the reference scales the activation before quantizing it
     * (torchao's linear multiplies {@code activation_tensor * act_pre_scale} and then quantizes).
     */
    RowSource finishPreScaled(IntModuleTensors m, IntWeightSpec spec, float[] preScale, String reason)
            throws IOException {
        long want = Math.multiplyExact(Math.max(1, spec.experts), spec.k);
        if (preScale.length != spec.k && preScale.length != want) {
            throw new GgufFormatException(args.format() + " module " + m.name() + ": " + preScale.length
                    + " activation pre-scale values for " + spec.k + " columns");
        }
        Act act = activation(m, spec.experts);
        if (act.scheme != null || act.scale() != null || act.zeroPoint() != null) {
            return finish(m, spec, preScale, act);
        }
        warn("float fallback " + GGMLType.F32 + ": " + reason, m.name());
        return new IntGroupSource(spec, GGMLType.F32, GGMLType.F32, null, Act.NONE, cfg.dir().toString(),
                preScale);
    }

    private Choice choose(IntWeightSpec s, Stats st, int[] perm) {
        boolean exact = options.exactQuant();
        GGMLType fb = fallbackType(s.scaleDtype);
        if (!aligned(s, perm, 32)) {
            return new Choice(fb, "K " + s.k + " with groups of " + s.groupSize
                    + " does not split into 32-element blocks of one group");
        }
        if (s.formula == Formula.MUL_ADD) {
            if (s.bits != 4) {
                return new Choice(fb, s.bits + "-bit affine weights have no ggml block type");
            }
            return exact && !(st.scalesExact && st.zerosExact)
                    ? new Choice(GGMLType.F32, "scales or biases are not exact in fp16")
                    : new Choice(GGMLType.Q4_1, null);
        }
        if (st.symmetric) {
            if (s.preferI8B128 && s.bits == 8 && aligned(s, perm, 128)) {
                return new Choice(GGMLType.I8_B128, null);
            }
            if (exact && !st.scalesExact) {
                return new Choice(GGMLType.F32, "scales are not exact in fp16");
            }
            switch (s.bits) {
                case 4: return new Choice(GGMLType.Q4_0, null);
                case 5: return new Choice(GGMLType.Q5_0, null);
                case 2: return new Choice(aligned(s, perm, 64) ? GGMLType.Q2_0 : GGMLType.Q8_0, null);
                default: return new Choice(GGMLType.Q8_0, null);
            }
        }
        if (!exact && (s.bits == 4 || s.bits == 5)) {
            return new Choice(s.bits == 4 ? GGMLType.Q4_1 : GGMLType.Q5_1, null);
        }
        if (!st.zerosFitInt8(s.bits)) {
            return new Choice(fb, "zero points are not integers in [" + ((1 << s.bits) - 128) + ", 128]");
        }
        if (exact && !st.scalesExact) {
            return new Choice(GGMLType.F32, "scales are not exact in fp16");
        }
        return new Choice(GGMLType.Q8_0, null);
    }

    private static boolean aligned(IntWeightSpec s, int[] perm, int block) {
        if (s.k % block != 0) {
            return false;
        }
        if (s.gidx == null) {
            return s.groups == 1 || s.groupSize % block == 0;
        }
        for (int[] gi : s.gidx) {
            for (int j0 = 0; j0 < s.k; j0 += block) {
                int g0 = gi[perm == null ? j0 : perm[j0]];
                for (int j = j0 + 1; j < j0 + block; j++) {
                    if (gi[perm == null ? j : perm[j]] != g0) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static boolean isIdentity(int[] p) {
        for (int i = 0; i < p.length; i++) {
            if (p[i] != i) {
                return false;
            }
        }
        return true;
    }

    /**
     * The activation scheme and its static parameters ({@code input_scale}, {@code input_zero_point}, already
     * dequant multipliers). A scheme the runtime does not emulate is logged and the weight loads weight-only.
     */
    private Act activation(IntModuleTensors m, int experts) throws IOException {
        ActQuantScheme scheme = args.actScheme();
        if (scheme == null && args.actUnemulated() == null) {
            return Act.NONE;
        }
        RowSource is = m.takeOptional("input_scale");
        RowSource iz = m.takeOptional("input_zero_point");
        if (args.actUnemulated() != null) {
            warn("activation quantization (" + args.actUnemulated() + ") is not emulated; weights load"
                    + " weight-only", m.name());
            return Act.NONE;
        }
        if (!scheme.isStatic()) {
            return new Act(scheme, null, null);
        }
        if (is == null) {
            throw m.invalid(scheme + " activations need " + m.tensorName("input_scale"));
        }
        float[] scale = m.perExpertFloats(is, experts, "input_scale");
        for (float v : scale) {
            if (!(v > 0) || !Float.isFinite(v)) {
                throw m.invalid(m.tensorName("input_scale") + " holds " + v + "; expected a positive finite scale");
            }
        }
        int[] zp = null;
        if (scheme.needsZeroPoint()) {
            if (iz == null) {
                throw m.invalid(scheme + " activations need " + m.tensorName("input_zero_point"));
            }
            zp = m.perExpertInts(iz, experts, "input_zero_point");
        }
        return new Act(scheme, scale, zp);
    }

    // ------------------------------------------------------------------ helpers

    /** Builds the weight source of one module. */
    @FunctionalInterface
    interface WeightBuilder {
        RowSource build(IntRewriteBatch b, IntModuleTensors m) throws IOException;
    }

    /** A chosen target and, for a fallback, why. */
    private static final class Choice {
        final GGMLType target;
        final String reason;

        Choice(GGMLType target, String reason) {
            this.target = target;
            this.reason = reason;
        }
    }

    /** What the stored zero points and scales of a weight allow, scanned once at rewrite. */
    private static final class Stats {
        boolean symmetric = true;
        boolean integral = true;
        float zMin = Float.POSITIVE_INFINITY;
        float zMax = Float.NEGATIVE_INFINITY;
        boolean zerosExact = true;
        boolean scalesExact = true;

        /** True when every {@code q - z} of a {@code bits}-bit code fits int8. */
        boolean zerosFitInt8(int bits) {
            return integral && zMin >= (1 << bits) - 128 && zMax <= 128;
        }

        static Stats scan(IntRewriteBatch b, IntModuleTensors m, IntWeightSpec s) throws IOException {
            Stats st = new Stats();
            boolean exact = b.options().exactQuant();
            float half = 1 << (s.bits - 1);
            if (s.zeros == null) {
                st.note(s.zeroDefault, half, exact && s.formula == Formula.MUL_ADD);
            } else {
                visit(s, s.zeros, (e, row, g, v) -> {
                    if (!Float.isFinite(v)) {
                        throw m.invalid("zero point of expert " + e + " row " + row + " group " + g + " is " + v);
                    }
                    st.note(v, half, exact && s.formula == Formula.MUL_ADD);
                });
            }
            if (exact && s.scaleDtype != StDType.F16) {
                visit(s, s.scales, (e, row, g, v) -> st.scalesExact &= IntPacking.fp16Exact(v));
            }
            return st;
        }

        private void note(float z, float half, boolean checkFp16) {
            symmetric &= z == half;
            integral &= z == Math.rint(z);
            zMin = Math.min(zMin, z);
            zMax = Math.max(zMax, z);
            zerosExact &= !checkFp16 || IntPacking.fp16Exact(z);
        }

        @FunctionalInterface
        private interface Visitor {
            void accept(int expert, long row, int group, float v) throws IOException;
        }

        private static void visit(IntWeightSpec s, ParamTable t, Visitor v) throws IOException {
            int chunk = (int) Math.max(1, Math.min(s.n, SCAN_VALUES / s.groups));
            float[] buf = new float[chunk * s.groups];
            for (int e = 0; e < Math.max(1, s.experts); e++) {
                for (long n0 = 0; n0 < s.n; n0 += chunk) {
                    int count = (int) Math.min(chunk, s.n - n0);
                    t.read(e, n0, count, buf);
                    for (int i = 0; i < count * s.groups; i++) {
                        v.accept(e, n0 + i / s.groups, i % s.groups, buf[i]);
                    }
                }
            }
        }
    }
}
