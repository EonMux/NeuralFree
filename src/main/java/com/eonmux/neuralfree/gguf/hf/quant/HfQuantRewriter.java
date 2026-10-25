package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.KvScaleCollector;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantMethod;
import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
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
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;

/**
 * Rewrites the quantized modules of a checkpoint's tensor index into virtual weight sources, in this order:
 * <ol>
 *   <li>{@link KvScaleCollector} takes out the KV-cache scales (for any quantization method);</li>
 *   <li>{@link QuantModuleIndex} groups the auxiliary tensors into modules, and each module gets the
 *       {@link QuantArgs} of the first rule that matches it ({@link HfQuantConfig#argsFor});</li>
 *   <li>under {@link ActivationQuant#CHECKPOINT}, an activation scheme that is not emulated fails;</li>
 *   <li>output-activation quantization ({@code output_scale} that is not a KV-cache scale, compressed-tensors
 *       {@code output_activations}, Quark {@code output_tensors}) fails under CHECKPOINT and is dropped with one WARN
 *       per description under WEIGHT_ONLY;</li>
 *   <li>the modules are batched per {@link QuantArgs} and rewritten by the first scheme that handles the format;</li>
 *   <li>the compressed-tensors {@code transform_config} and the fp_quant rotations become one Hadamard plan;</li>
 *   <li>an auxiliary tensor still in the index is an error.</li>
 * </ol>
 * A method that is not quantized ({@link HfQuantConfig#isQuantized()} false) only goes through the first step.
 */
public final class HfQuantRewriter {

    private static final Logger LOG = LogManager.getLogger(HfQuantRewriter.class);

    private static final String OUTPUT_SCALE = "output_scale";

    /** The rewritten index and what later steps derive from the rewrite. Immutable. */
    public static final class Result {
        private final HfTensorIndex index;
        private final Map<String, QuantArgs> moduleArgs;
        private final HadamardPlan hadamard;
        private final List<KvScaleCollector.Scale> kvScales;

        Result(HfTensorIndex index, Map<String, QuantArgs> moduleArgs, HadamardPlan hadamard,
               List<KvScaleCollector.Scale> kvScales) {
            this.index = index;
            this.moduleArgs = moduleArgs;
            this.hadamard = hadamard;
            this.kvScales = kvScales;
        }

        /** The index with every quantized module rewritten and the KV-cache tensors taken out. */
        public HfTensorIndex index() { return index; }

        /** The arguments of every rewritten module (HF module prefix), in checkpoint order. Unmodifiable. */
        public Map<String, QuantArgs> moduleArgs() { return moduleArgs; }

        /** The online Hadamard input transform in checkpoint weight names, when the checkpoint has one. */
        public Optional<HadamardPlan> hadamard() { return Optional.ofNullable(hadamard); }

        /** The KV-cache scales, as GGUF tensors. Unmodifiable. */
        public List<KvScaleCollector.Scale> kvScales() { return kvScales; }
    }

    private HfQuantRewriter() {
    }

    /**
     * {@link #rewrite(HfQuantConfig, HfLoadContext, HfTensorIndex, ModelSourceOptions, ActivationQuant)} for
     * {@link ActivationQuant#WEIGHT_ONLY}.
     */
    public static Result rewrite(HfQuantConfig q, HfLoadContext ctx, HfTensorIndex in, ModelSourceOptions o)
            throws IOException {
        return rewrite(q, ctx, in, o, ActivationQuant.WEIGHT_ONLY);
    }

    /**
     * Rewrites {@code in}.
     *
     * @param mode how the engine treats the checkpoint's activation quantization
     * @throws GgufFormatException for an unemulated or output activation scheme under CHECKPOINT, an unconsumed
     *                             auxiliary tensor, a format no scheme handles, or a scheme's own format error
     * @throws IOException         when a tensor cannot be read
     */
    public static Result rewrite(HfQuantConfig q, HfLoadContext ctx, HfTensorIndex in, ModelSourceOptions o,
                                 ActivationQuant mode) throws IOException {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(o, "o");
        Objects.requireNonNull(mode, "mode");
        HfConfig cfg = ctx.config();
        KvScaleCollector.Result kv = q.method() == HfQuantMethod.NONE ? null : KvScaleCollector.collect(cfg, in);
        HfTensorIndex index = kv == null ? in : kv.index();
        List<KvScaleCollector.Scale> scales = kv == null ? Collections.emptyList() : kv.scales();
        if (!q.isQuantized()) {
            return new Result(index, Collections.emptyMap(), null, scales);
        }
        QuantModuleIndex mods = QuantModuleIndex.of(index, m -> q.argsFor(m).orElse(null),
                q.method() == HfQuantMethod.DEEPSEEK_V4_FP8);
        Map<QuantArgs, List<String>> batches = new LinkedHashMap<>();
        Map<String, QuantArgs> moduleArgs = new LinkedHashMap<>();
        for (String module : mods.modules().keySet()) {
            QuantArgs args = q.argsFor(module).orElse(null);
            if (args != null) {
                batches.computeIfAbsent(args, a -> new ArrayList<>()).add(module);
                moduleArgs.put(module, args);
            }
        }
        if (mode == ActivationQuant.CHECKPOINT) {
            requireEmulated(batches.keySet());
        }
        index = outputActivations(q, index, mods, moduleArgs, kv.outputScaleModules(), mode);
        for (Map.Entry<QuantArgs, List<String>> b : batches.entrySet()) {
            index = dispatch(q, b.getKey(), b.getValue(), cfg, index, o);
        }
        HadamardPlan online = null;
        Optional<HfTransformConfig> transforms = q.transforms();
        if (transforms.isPresent()) {
            HadamardTransforms.Result r = HadamardTransforms.compressedTensors(transforms.get(), index);
            index = r.index();
            online = merge(online, r.plan().orElse(null));
        }
        for (Map.Entry<QuantArgs, List<String>> b : batches.entrySet()) {
            if (FpQuantScheme.handles(b.getKey().format())) {
                online = merge(online, HadamardTransforms.fpQuant(b.getKey(), b.getValue()).orElse(null));
            }
        }
        for (String aux : mods.auxiliary()) {
            if (index.has(aux)) {
                throw new GgufFormatException("unconsumed quantization tensor " + aux + " (" + label(q) + ")");
            }
        }
        LOG.info("Quantization {}: {} module(s) in {} batch(es) rewritten", label(q), moduleArgs.size(),
                batches.size());
        return new Result(index, Collections.unmodifiableMap(moduleArgs), online, scales);
    }

    private static void requireEmulated(Set<QuantArgs> args) {
        for (QuantArgs a : args) {
            if (a.actUnemulated() != null) {
                throw new GgufFormatException(a.actUnemulated() + " is not emulated; use --act-quant weight-only");
            }
        }
    }

    /**
     * Applies the output-activation rule: every {@code output_scale} tensor (of a module that is not a KV-cache
     * projection) and every module whose arguments quantize its output fail under CHECKPOINT; under WEIGHT_ONLY the
     * tensors are dropped with one WARN per description.
     */
    private static HfTensorIndex outputActivations(HfQuantConfig q, HfTensorIndex index, QuantModuleIndex mods,
                                                   Map<String, QuantArgs> moduleArgs, Set<String> kvModules,
                                                   ActivationQuant mode) {
        Map<String, Integer> warnings = new LinkedHashMap<>();
        Set<String> drop = new LinkedHashSet<>();
        for (Map.Entry<String, List<String>> m : mods.modules().entrySet()) {
            if (kvModules.contains(m.getKey())) {
                continue;
            }
            QuantArgs args = moduleArgs.get(m.getKey());
            String desc = args != null ? args.outputActivations() : null;
            for (String name : m.getValue()) {
                if (name.endsWith("." + OUTPUT_SCALE) || name.endsWith("_" + OUTPUT_SCALE)) {
                    drop.add(name);
                    if (desc == null) {
                        desc = label(q) + " " + OUTPUT_SCALE;
                    }
                }
            }
            if (desc != null) {
                if (mode == ActivationQuant.CHECKPOINT) {
                    throw new GgufFormatException("output activation quantization (" + desc
                            + ") is not emulated; use --act-quant weight-only");
                }
                warnings.merge(desc, 1, Integer::sum);
            }
        }
        for (Map.Entry<String, Integer> w : warnings.entrySet()) {
            LOG.warn("output activation quantization ({}) is not emulated; {} module(s) load weight-only and their"
                    + " output scales are dropped", w.getKey(), w.getValue());
        }
        return drop.isEmpty() ? index : index.withReplaced(Collections.emptyMap(), drop);
    }

    /** Rewrites one batch with the first scheme that handles its format. */
    private static HfTensorIndex dispatch(HfQuantConfig q, QuantArgs args, List<String> modules, HfConfig cfg,
                                          HfTensorIndex in, ModelSourceOptions o) throws IOException {
        QuantFormat f = args.format();
        if (IntGroupScheme.handles(f)) {
            return IntGroupScheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (Int8Scheme.handles(f)) {
            return Int8Scheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (MlxScheme.handles(f)) {
            return MlxScheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (BitnetScheme.handles(f)) {
            return BitnetScheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (HqqScheme.handles(f)) {
            return HqqScheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (Fp8Scheme.handles(f)) {
            return Fp8Scheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (Fp4Scheme.handles(f)) {
            return Fp4Scheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (BnbScheme.handles(f)) {
            return BnbScheme.rewriteModules(args, modules, cfg, in, o);
        }
        if (FpQuantScheme.handles(f)) {
            return FpQuantScheme.rewriteModules(args, modules, cfg, in, o);
        }
        throw new GgufFormatException("no rewrite for " + f + " weights (" + label(q) + ")");
    }

    /**
     * {@code a} and {@code b} as one plan (either may be null).
     *
     * @throws GgufFormatException when the block sizes or sign modes differ, two sign vectors share a width, or a
     *                             weight is in both
     */
    static HadamardPlan merge(HadamardPlan a, HadamardPlan b) {
        if (a == null || b == null) {
            return a == null ? b : a;
        }
        if (a.blockSize() != b.blockSize()) {
            throw new GgufFormatException("the checkpoint's Hadamard transforms use blocks of " + a.blockSize()
                    + " and " + b.blockSize() + "; a model carries one Hadamard block size");
        }
        if (!a.signMode().equals(b.signMode())) {
            throw new GgufFormatException("the checkpoint's Hadamard transforms mix " + a.signMode() + " and "
                    + b.signMode() + " signs; a model carries one sign mode");
        }
        Map<Integer, float[]> signs = new TreeMap<>();
        for (HadamardPlan p : new HadamardPlan[]{a, b}) {
            for (int w : p.signWidths()) {
                if (signs.put(w, p.signs(w)) != null) {
                    throw new GgufFormatException("the checkpoint's Hadamard transforms give two sign vectors of"
                            + " width " + w);
                }
            }
        }
        Set<String> names = new LinkedHashSet<>(a.weightNames());
        for (String n : b.weightNames()) {
            if (!names.add(n)) {
                throw new GgufFormatException("weight " + n + " has two Hadamard input transforms");
            }
        }
        return HadamardPlan.of(a.blockSize(), signs, new ArrayList<>(names));
    }

    /** The method as the configuration names it, for messages. */
    static String label(HfQuantConfig q) {
        String name = q.methodName();
        return name == null || name.isEmpty() ? q.method().name().toLowerCase(Locale.ROOT) : name;
    }
}
