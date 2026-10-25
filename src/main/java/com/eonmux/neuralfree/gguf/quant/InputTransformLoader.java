package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Binds one {@link InputTransform} per projection weight from everything a model declares about its matmul inputs:
 * <ul>
 *   <li>the PRISM Hadamard transforms ({@link PrismHadamardLoader}), unchanged;</li>
 *   <li>the act-order permutations {@code <base>.input_perm} (I32 {@code [K]}, {@code K = ne[0]} of
 *       {@code <base>.weight}), whatever the activation mode, since the weight is stored in permuted input order;</li>
 *   <li>under {@link ActivationQuant#CHECKPOINT}, the activation schemes of the {@code neuralfree.act_quant.*}
 *       carrier ({@link ActQuantPlan}) with their static siblings {@code <base>.input_scale} (F32 {@code [1]} or
 *       {@code [E]}) and {@code <base>.input_zero_point} (I32 {@code [1]} or {@code [E]}), and the model's
 *       {@code scale_ub} for {@link ActQuantScheme#FP8_TOKEN}. {@link ActivationQuant#WEIGHT_ONLY} ignores the
 *       carrier.</li>
 * </ul>
 * A model without act-quant metadata and without {@code .input_perm} tensors yields the PRISM forward map, each
 * transform holding only its {@link HadamardSpec}. Every malformed input throws {@link GgufFormatException} naming the
 * tensor or key.
 */
public final class InputTransformLoader {

    private static final String WEIGHT_SUFFIX = ".weight";
    private static final String PERM_SUFFIX = "." + TensorClaims.INPUT_PERM;

    /** Reads the values of an I32 tensor of the model. */
    @FunctionalInterface
    public interface I32Reader {
        /**
         * @throws IOException when the tensor's data cannot be read
         * @throws GgufFormatException when the tensor is not stored as I32
         */
        int[] read(String tensorName) throws IOException;
    }

    /** Reads the values of an F32 tensor of the model. */
    @FunctionalInterface
    public interface F32Reader {
        /**
         * @throws IOException when the tensor's data cannot be read
         * @throws GgufFormatException when the tensor is not stored as F32
         */
        float[] read(String tensorName) throws IOException;
    }

    /** The bound input transforms of one model. Immutable. */
    public static final class Result {
        private static final Result NONE = new Result(Collections.<String, InputTransform>emptyMap(),
                Collections.<String, HadamardSpec>emptyMap(), false);

        private final Map<String, InputTransform> forward;
        private final Map<String, HadamardSpec> inverse;
        private final boolean tiedOutput;

        private Result(Map<String, InputTransform> forward, Map<String, HadamardSpec> inverse, boolean tiedOutput) {
            this.forward = forward;
            this.inverse = inverse;
            this.tiedOutput = tiedOutput;
        }

        /** Weight name to the transform of its matmul input; weights without one are absent. Unmodifiable. */
        public Map<String, InputTransform> forward() {
            return forward;
        }

        /** The transform of {@code weightName}'s matmul input; {@link InputTransform#IDENTITY} when it has none. */
        public InputTransform forward(String weightName) {
            InputTransform t = forward.get(weightName);
            return t != null ? t : InputTransform.IDENTITY;
        }

        /** The input rotation of {@code weightName}, or null. */
        public HadamardSpec hadamard(String weightName) {
            InputTransform t = forward.get(weightName);
            return t != null ? t.hadamard() : null;
        }

        /** Latent lookup table name to the PRISM transform of its looked-up rows. Unmodifiable. */
        public Map<String, HadamardSpec> inverse() {
            return inverse;
        }

        /** True when the output head is the latent token embedding ({@code prism.hadamard.tied_output}). */
        public boolean tiedOutput() {
            return tiedOutput;
        }

        /** True when no weight has an input transform and no table an inverse one. */
        public boolean isEmpty() {
            return forward.isEmpty() && inverse.isEmpty();
        }

        /** True when every forward transform is a rotation alone: no input permutation and no activation scheme. */
        public boolean hadamardOnly() {
            for (InputTransform t : forward.values()) {
                if (!t.isHadamardOnly()) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public String toString() {
            return "InputTransformLoader.Result{forward=" + forward.size() + ", inverse=" + inverse.size()
                    + (tiedOutput ? ", tied output" : "") + '}';
        }
    }

    private InputTransformLoader() {
    }

    /**
     * Binds the input transforms of an open model, reading the small sibling tensors through the source's
     * {@link TensorReadMode#READ} regions, which are opened only when a sibling is read and closed before this
     * returns.
     *
     * @param mode how the model's activation quantization is applied
     * @throws IOException         when a sibling tensor cannot be read
     * @throws GgufFormatException when the metadata or a sibling tensor is malformed or does not match the model
     */
    public static Result load(ModelSource source, GgufKv kv, TensorCatalog catalog, ActivationQuant mode)
            throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(catalog, "catalog");
        try (SourceTensors tensors = new SourceTensors(source, catalog)) {
            return load(kv, catalog.names(), name -> {
                TensorRef r = catalog.get(name);
                return r != null ? r.ne() : null;
            }, kv.u32(GgufKeys.SSM_TIME_STEP_RANK, 0), kv.u32(GgufKeys.SSM_GROUP_COUNT, 0), mode,
                    tensors::i32, tensors::f32);
        }
    }

    /**
     * Binds the input transforms of a model.
     *
     * @param kv          model metadata
     * @param tensorNames the stored names of every tensor of the model (the {@code .input_perm} tensors are found
     *                    among them)
     * @param shapes      tensor dimensions ({@code ne[0]} first) by name, null for an absent tensor
     * @param ssmDtRank   {@code ssm_dt_rank}, see {@link PrismHadamardLoader#load}
     * @param ssmNGroup   {@code ssm_n_group}, see {@link PrismHadamardLoader#load}
     * @param mode        how the model's activation quantization is applied
     * @param i32         reads I32 sibling tensors ({@code .input_perm}, {@code .input_zero_point})
     * @param f32         reads F32 sibling tensors ({@code .input_scale})
     * @throws IOException         when a reader fails
     * @throws GgufFormatException when the metadata or a sibling tensor is malformed or does not match the model
     */
    public static Result load(GgufKv kv, Collection<String> tensorNames, PrismHadamardLoader.TensorShapes shapes,
                              int ssmDtRank, int ssmNGroup, ActivationQuant mode, I32Reader i32, F32Reader f32)
            throws IOException {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tensorNames, "tensorNames");
        Objects.requireNonNull(shapes, "shapes");
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(i32, "i32");
        Objects.requireNonNull(f32, "f32");
        PrismHadamardLoader.Result prism = PrismHadamardLoader.load(kv, shapes, ssmDtRank, ssmNGroup);
        Map<String, InputTransform> forward = new LinkedHashMap<>();
        for (Map.Entry<String, HadamardSpec> e : prism.forward().entrySet()) {
            forward.put(e.getKey(), InputTransform.IDENTITY.withHadamard(e.getValue()));
        }
        for (String name : tensorNames) {
            if (name.endsWith(PERM_SUFFIX) && name.length() > PERM_SUFFIX.length()) {
                bindPerm(name, shapes, i32, forward);
            }
        }
        if (mode == ActivationQuant.CHECKPOINT) {
            ActQuantPlan plan = ActQuantPlan.load(kv);
            for (Map.Entry<String, ActQuantScheme> e : plan.schemes().entrySet()) {
                bindActQuant(e.getKey(), e.getValue(), plan.scaleUpperBound(), shapes, i32, f32, forward);
            }
        }
        if (forward.isEmpty() && prism.isEmpty()) {
            return Result.NONE;
        }
        return new Result(Collections.unmodifiableMap(forward), prism.inverse(), prism.tiedOutput());
    }

    /** Adds the act-order permutation {@code permName} to its weight's transform. */
    private static void bindPerm(String permName, PrismHadamardLoader.TensorShapes shapes, I32Reader i32,
                                 Map<String, InputTransform> forward) throws IOException {
        String weight = permName.substring(0, permName.length() - PERM_SUFFIX.length()) + WEIGHT_SUFFIX;
        int k = inputWidth(weight, shapes, "input permutation " + permName);
        long[] dims = shapes.dims(permName);
        if (dims == null || dims.length != 1 || dims[0] != k) {
            throw new GgufFormatException("input permutation " + permName + " has shape "
                    + (dims == null ? "null" : Arrays.toString(dims)) + "; " + weight + " needs [" + k + "]");
        }
        int[] perm = i32.read(permName);
        if (perm == null || perm.length != k) {
            throw new GgufFormatException("input permutation " + permName + " holds "
                    + (perm == null ? "no" : String.valueOf(perm.length)) + " values; " + weight + " needs " + k);
        }
        InputTransform t = forward.getOrDefault(weight, InputTransform.IDENTITY);
        try {
            forward.put(weight, t.withPerm(perm).requireWidth(k));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(permName + ": " + e.getMessage(), e);
        }
    }

    /** Adds the activation scheme of {@code weight}, with its static parameters, to the weight's transform. */
    private static void bindActQuant(String weight, ActQuantScheme scheme, float scaleUpperBound,
                                     PrismHadamardLoader.TensorShapes shapes, I32Reader i32, F32Reader f32,
                                     Map<String, InputTransform> forward) throws IOException {
        String what = "activation scheme " + scheme + " of " + weight;
        int cols = inputWidth(weight, shapes, what);
        long[] dims = shapes.dims(weight);
        long experts = dims.length >= 3 ? dims[2] : 0;
        String base = weight.substring(0, weight.length() - WEIGHT_SUFFIX.length());
        float[] scales = null;
        int[] zeroPoints = null;
        if (scheme.isStatic() || scheme.needsGlobalScale()) {
            String name = base + "." + TensorClaims.INPUT_SCALE;
            requireCount(name, shapes, experts, what);
            scales = f32.read(name);
            requireValues(name, scales == null ? -1 : scales.length, shapes);
        }
        if (scheme.needsZeroPoint()) {
            String name = base + "." + TensorClaims.INPUT_ZERO_POINT;
            requireCount(name, shapes, experts, what);
            zeroPoints = i32.read(name);
            requireValues(name, zeroPoints == null ? -1 : zeroPoints.length, shapes);
        }
        InputTransform t = forward.getOrDefault(weight, InputTransform.IDENTITY);
        try {
            InputTransform.ActQuant a = InputTransform.ActQuant.of(scheme, scales, zeroPoints, scaleUpperBound);
            forward.put(weight, t.withActQuant(a).requireWidth(cols));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(what + ": " + e.getMessage(), e);
        }
    }

    /** {@code ne[0]} of {@code weight} as an int; {@code what} names the entry that needs the weight. */
    private static int inputWidth(String weight, PrismHadamardLoader.TensorShapes shapes, String what) {
        long[] dims = shapes.dims(weight);
        if (dims == null || dims.length == 0) {
            throw new GgufFormatException(what + " names a weight the model does not have: " + weight);
        }
        if (dims[0] <= 0 || dims[0] > Integer.MAX_VALUE) {
            throw new GgufFormatException(weight + " has an input width of " + dims[0] + ", outside the int range");
        }
        return (int) dims[0];
    }

    /** Requires the sibling {@code name} to hold 1 value, or {@code experts} values for an expert weight. */
    private static void requireCount(String name, PrismHadamardLoader.TensorShapes shapes, long experts,
                                     String what) {
        long[] dims = shapes.dims(name);
        if (dims == null) {
            throw new GgufFormatException(what + " needs the tensor " + name);
        }
        long n = elements(dims);
        if (n != 1 && (experts == 0 || n != experts)) {
            throw new GgufFormatException(name + " has " + n + " values; " + what + " needs 1"
                    + (experts > 0 ? " or " + experts + " (one per expert)" : ""));
        }
    }

    private static void requireValues(String name, int read, PrismHadamardLoader.TensorShapes shapes) {
        long n = elements(shapes.dims(name));
        if (read != n) {
            throw new GgufFormatException(name + " read " + (read < 0 ? "no" : String.valueOf(read))
                    + " values; its shape holds " + n);
        }
    }

    private static long elements(long[] dims) {
        long n = 1;
        for (long d : dims) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    /** The small sibling tensors of a {@link ModelSource}, read through regions opened on first use. */
    private static final class SourceTensors implements AutoCloseable {
        private final ModelSource source;
        private final TensorCatalog catalog;
        private TensorRegions regions;

        SourceTensors(ModelSource source, TensorCatalog catalog) {
            this.source = source;
            this.catalog = catalog;
        }

        int[] i32(String name) throws IOException {
            ByteBuffer b = read(name, GGMLType.I32);
            int[] v = new int[b.remaining() / Integer.BYTES];
            b.asIntBuffer().get(v);
            return v;
        }

        float[] f32(String name) throws IOException {
            ByteBuffer b = read(name, GGMLType.F32);
            float[] v = new float[b.remaining() / Float.BYTES];
            b.asFloatBuffer().get(v);
            return v;
        }

        private ByteBuffer read(String name, GGMLType type) throws IOException {
            GGUFTensorInfo info = catalog.info(catalog.get(name));
            if (info == null) {
                throw new GgufFormatException("tensor " + name + " is not stored in " + source.getFilePath());
            }
            if (info.type() != type) {
                throw new GgufFormatException("tensor " + name + " is " + info.type() + "; expected " + type);
            }
            long bytes = Math.multiplyExact(info.numElements(), 4L);
            if (bytes > Integer.MAX_VALUE - 8L) {
                throw new GgufFormatException("tensor " + name + " has " + info.numElements()
                        + " values, too many for one array");
            }
            if (regions == null) {
                regions = source.openRegions(TensorReadMode.READ);
            }
            return regions.region(info, 0L, bytes);
        }

        @Override
        public void close() throws IOException {
            if (regions != null) {
                regions.close();
            }
        }
    }
}
