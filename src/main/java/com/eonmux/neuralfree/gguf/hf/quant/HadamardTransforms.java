package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.function.Function;

/**
 * Turns the online Hadamard input transforms of compressed-tensors ({@code transform_config}) and fp_quant into a
 * {@link HadamardPlan}: the PRISM {@code prism.hadamard.*} inputs (one block size, a sign mode, one sign vector per
 * input width, the transformed weights). The PRISM runtime computes {@code H * (D * x)} per block, with {@code H} the
 * normalized Sylvester matrix and {@code D} a +-1 diagonal applied before the rotation.
 *
 * <p>compressed-tensors: a scheme of type {@code hadamard} or {@code random-hadamard} applied at {@code input} stores
 * one unnormalized matrix {@code R} per transformed linear as {@code <module>.<scheme>_input.weight}, and the input
 * becomes {@code x R / sqrt(n)} per block of {@code head_dim} (or the full width). {@code hadamard} is the Sylvester
 * matrix ({@code D = I}); {@code random-hadamard} is {@code R[i][j] = d_i * (-1)^popcount(i & j)}, so
 * {@code x R / sqrt(n) = H (D x)} with {@code d_i = R[i][0]}, validated entry by entry. {@code weight_input} and
 * {@code weight_output} transforms were folded into the weights offline and are ignored; every other location,
 * matrix type, randomized permutation and inverse random transform is rejected by name.</p>
 *
 * <p>fp_quant: the weights were quantized after a normalized Sylvester rotation of {@code hadamard_group_size}
 * blocks ({@code transform_init: hadamard}); {@code identity} needs no transform and {@code gsr} is rejected.</p>
 *
 * <p>The PRISM architecture gate ({@code PrismHadamardLoader} verified architectures) is applied where the plan
 * becomes metadata.</p>
 */
public final class HadamardTransforms {

    /**
     * The largest rotation block, the limit {@code PrismHadamardLoader} enforces on {@code prism.hadamard.block_size};
     * GPU uploads additionally refuse blocks above {@link HadamardSpec#MAX_KERNEL_BLOCK}.
     */
    public static final int MAX_BLOCK = 1 << 30;

    private static final String HADAMARD = "hadamard";
    private static final String RANDOM_HADAMARD = "random-hadamard";

    private HadamardTransforms() {
    }

    /** The PRISM transform of one model. Immutable. */
    public static final class HadamardPlan {
        private final int blockSize;
        private final Map<Integer, float[]> signs;
        private final List<String> weightNames;

        private HadamardPlan(int blockSize, Map<Integer, float[]> signs, List<String> weightNames) {
            this.blockSize = blockSize;
            this.signs = signs;
            this.weightNames = weightNames;
        }

        /**
         * A plan of {@code blockSize} rotations over the inputs of {@code weightNames}, with {@code signs} mapping
         * an input width to its +-1 vector (empty for identity signs).
         *
         * @throws GgufFormatException      when the block is not a power of two at most {@link #MAX_BLOCK}, or a
         *                                  sign vector's width is not a positive multiple of the block, differs from
         *                                  its length or holds a value other than +-1
         * @throws IllegalArgumentException when {@code weightNames} is empty or holds a null or repeated name
         */
        public static HadamardPlan of(int blockSize, Map<Integer, float[]> signs, List<String> weightNames) {
            checkBlock(blockSize, "Hadamard block");
            Map<Integer, float[]> copy = new TreeMap<>();
            for (Map.Entry<Integer, float[]> e : Objects.requireNonNull(signs, "signs").entrySet()) {
                int width = e.getKey();
                float[] v = e.getValue();
                if (width <= 0 || width % blockSize != 0 || v == null || v.length != width) {
                    throw new GgufFormatException("Hadamard sign vector of width " + width + " must have " + width
                            + " values and a width that is a positive multiple of block " + blockSize);
                }
                for (float x : v) {
                    if (x != 1f && x != -1f) {
                        throw new GgufFormatException("Hadamard sign vector of width " + width + " holds " + x);
                    }
                }
                copy.put(width, v.clone());
            }
            Objects.requireNonNull(weightNames, "weightNames");
            if (weightNames.isEmpty() || weightNames.stream().anyMatch(Objects::isNull)
                    || new LinkedHashSet<>(weightNames).size() != weightNames.size()) {
                throw new IllegalArgumentException("Hadamard weight names must be non-empty and distinct: "
                        + weightNames);
            }
            return new HadamardPlan(blockSize, Collections.unmodifiableMap(copy),
                    Collections.unmodifiableList(new ArrayList<>(weightNames)));
        }

        /** The rotation block, a power of two at most {@link #MAX_BLOCK}. */
        public int blockSize() {
            return blockSize;
        }

        /** {@code identity} when every sign is +1, else {@code explicit}. */
        public String signMode() {
            return signs.isEmpty() ? "identity" : "explicit";
        }

        /** The input widths that have a sign vector, ascending; empty in identity mode. */
        public int[] signWidths() {
            return signs.keySet().stream().mapToInt(Integer::intValue).toArray();
        }

        /** The sign vectors of {@link #signWidths()} concatenated, each value +1 or -1; empty in identity mode. */
        public int[] signValues() {
            int total = signs.values().stream().mapToInt(v -> v.length).sum();
            int[] out = new int[total];
            int off = 0;
            for (float[] v : signs.values()) {
                for (float x : v) {
                    out[off++] = (int) x;
                }
            }
            return out;
        }

        /** The sign vector of inputs of {@code width}, or null in identity mode or for an unknown width. */
        public float[] signs(int width) {
            float[] v = signs.get(width);
            return v == null ? null : v.clone();
        }

        /** The names of the transformed weights, in checkpoint order. Unmodifiable. */
        public List<String> weightNames() {
            return weightNames;
        }

        /**
         * This plan with every weight name mapped by {@code rename} (for example to GGUF names).
         *
         * @throws GgufFormatException when {@code rename} maps a name to null or two names to one
         */
        public HadamardPlan renamed(Function<String, String> rename) {
            Set<String> out = new LinkedHashSet<>();
            for (String n : weightNames) {
                String r = rename.apply(n);
                if (r == null || !out.add(r)) {
                    throw new GgufFormatException("Hadamard-transformed weight " + n + " maps to "
                            + (r == null ? "no tensor" : "the duplicate name " + r));
                }
            }
            return new HadamardPlan(blockSize, signs, Collections.unmodifiableList(new ArrayList<>(out)));
        }

        @Override
        public String toString() {
            return "HadamardPlan{block " + blockSize + ", " + signMode() + ", " + weightNames.size() + " weights}";
        }
    }

    /** The index without the consumed transform tensors, and the plan when any weight is transformed. Immutable. */
    public static final class Result {
        private final HfTensorIndex index;
        private final HadamardPlan plan;

        private Result(HfTensorIndex index, HadamardPlan plan) {
            this.index = index;
            this.plan = plan;
        }

        public HfTensorIndex index() {
            return index;
        }

        public Optional<HadamardPlan> plan() {
            return Optional.ofNullable(plan);
        }
    }

    /** {@code (-1)^popcount(i & j)}: the sign of entry {@code (i, j)} of a Sylvester Hadamard matrix. */
    public static int sylvesterSign(int i, int j) {
        return (Integer.bitCount(i & j) & 1) == 0 ? 1 : -1;
    }

    /**
     * The plan of fp_quant modules quantized with {@code args}: none for {@code transform_init: identity}, a
     * normalized Sylvester rotation of {@code hadamard_group_size} blocks over every module's {@code .weight} for
     * {@code hadamard}.
     *
     * @throws GgufFormatException for {@code transform_init: gsr} or a group size outside the PRISM kernel
     */
    public static Optional<HadamardPlan> fpQuant(QuantArgs args, List<String> modules) {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(modules, "modules");
        QuantFormat f = args.format();
        if (f != QuantFormat.FPQUANT_MXFP4 && f != QuantFormat.FPQUANT_NVFP4 && f != QuantFormat.FPQUANT_PSEUDO) {
            throw new IllegalArgumentException(f + " is not an fp_quant format");
        }
        QuantArgs.FpQuantTransform t = args.fpQuantTransform();
        if (t == QuantArgs.FpQuantTransform.GSR) {
            throw new GgufFormatException("fp_quant transform_init 'gsr' is not supported");
        }
        if (t == QuantArgs.FpQuantTransform.IDENTITY || modules.isEmpty()) {
            return Optional.empty();
        }
        int block = args.fpQuantHadamardGroup();
        checkBlock(block, "fp_quant hadamard_group_size");
        List<String> names = new ArrayList<>();
        for (String m : modules) {
            names.add(m + ".weight");
        }
        return Optional.of(new HadamardPlan(block, Collections.emptyMap(), Collections.unmodifiableList(names)));
    }

    /**
     * Consumes the transform tensors of a compressed-tensors {@code transform_config} and returns the plan of its
     * input transforms. {@code in} must hold every transformed module's rewritten {@code <module>.weight} (logical
     * shape), whose innermost dimension is the input width.
     *
     * @throws GgufFormatException for an unsupported type, location or option, a missing or malformed transform
     *                             tensor, more than one block size, or two sign vectors for one width
     * @throws IOException         when a transform tensor cannot be read
     */
    public static Result compressedTensors(HfTransformConfig config, HfTensorIndex in) throws IOException {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(in, "in");
        Set<String> remove = new LinkedHashSet<>();
        Map<String, String> transformed = new LinkedHashMap<>();
        Map<Integer, float[]> signs = new TreeMap<>();
        boolean explicit = false;
        int block = 0;
        for (HfTransformConfig.Scheme s : config.schemes()) {
            for (HfTransformConfig.Apply a : s.applies()) {
                String loc = a.location().name().toLowerCase(Locale.ROOT);
                String suffix = "." + s.name() + "_" + loc;
                if (!a.location().isOnline()) {
                    consumeAll(in, suffix, remove);
                    continue;
                }
                checkSupported(s, a);
                List<String> found = matrices(in, suffix + ".weight");
                if (found.isEmpty()) {
                    throw new GgufFormatException("compressed-tensors transform " + s.name() + " at input has no "
                            + "<module>" + suffix + ".weight tensor");
                }
                boolean random = s.type().equals(RANDOM_HADAMARD);
                for (String mName : found) {
                    String module = mName.substring(0, mName.length() - (suffix + ".weight").length());
                    String weight = module + ".weight";
                    String prior = transformed.putIfAbsent(weight, s.name());
                    if (prior != null) {
                        throw new GgufFormatException("module " + module + " has the input transforms " + prior
                                + " and " + s.name() + "; at most one is supported");
                    }
                    int width = inputWidth(in, weight, mName);
                    int b = s.headDim().isPresent() ? s.headDim().getAsInt() : width;
                    checkBlock(b, "compressed-tensors transform " + s.name() + " block");
                    if (width % b != 0) {
                        throw new GgufFormatException("compressed-tensors transform " + s.name() + " block " + b
                                + " does not divide the input width " + width + " of " + weight);
                    }
                    if (block != 0 && block != b) {
                        throw new GgufFormatException("compressed-tensors transforms use the block sizes " + block
                                + " and " + b + "; PRISM supports one block size per model");
                    }
                    block = b;
                    float[] d = random ? randomSigns(in.get(mName), mName, b) : matrixShape(in.get(mName), mName, b);
                    explicit |= random;
                    float[] tiled = new float[width];
                    for (int i = 0; i < width; i++) {
                        tiled[i] = d[i % b];
                    }
                    float[] prev = signs.putIfAbsent(width, tiled);
                    if (prev != null && !Arrays.equals(prev, tiled)) {
                        throw new GgufFormatException("compressed-tensors transforms give inputs of width " + width
                                + " two different sign vectors (" + weight + "); PRISM shares one per width");
                    }
                    remove.add(mName);
                    String perm = module + suffix + ".perm";
                    if (in.has(perm)) {
                        remove.add(perm);
                    }
                }
            }
        }
        HfTensorIndex out = remove.isEmpty() ? in : in.withReplaced(Collections.emptyMap(), remove);
        if (transformed.isEmpty()) {
            return new Result(out, null);
        }
        Map<Integer, float[]> planSigns = explicit ? Collections.unmodifiableMap(signs) : Collections.emptyMap();
        return new Result(out, new HadamardPlan(block, planSigns,
                Collections.unmodifiableList(new ArrayList<>(transformed.keySet()))));
    }

    private static void checkSupported(HfTransformConfig.Scheme s, HfTransformConfig.Apply a) {
        String loc = a.location().name().toLowerCase(Locale.ROOT);
        boolean hadamardType = s.type().equals(HADAMARD) || s.type().equals(RANDOM_HADAMARD);
        if (a.location() != HfTransformConfig.Location.INPUT || !hadamardType) {
            throw new GgufFormatException("compressed-tensors transform " + s.type() + "@" + loc + " is not supported");
        }
        if (s.randomize()) {
            throw new GgufFormatException("compressed-tensors transform " + s.type() + "@" + loc
                    + " with randomize is not supported");
        }
        if (a.inverse() && s.type().equals(RANDOM_HADAMARD)) {
            throw new GgufFormatException("compressed-tensors transform " + s.type() + "@" + loc
                    + " with inverse is not supported");
        }
    }

    private static void checkBlock(int b, String what) {
        if (b <= 0 || (b & (b - 1)) != 0 || b > MAX_BLOCK) {
            throw new GgufFormatException(what + " " + b + " is not a power of two at most " + MAX_BLOCK);
        }
    }

    private static List<String> matrices(HfTensorIndex in, String suffix) {
        List<String> out = new ArrayList<>();
        for (String n : in.names()) {
            if (n.endsWith(suffix) && n.length() > suffix.length()) {
                out.add(n);
            }
        }
        return out;
    }

    private static void consumeAll(HfTensorIndex in, String suffix, Set<String> remove) {
        for (String n : in.names()) {
            if (n.endsWith(suffix + ".weight") || n.endsWith(suffix + ".perm")) {
                remove.add(n);
            }
        }
    }

    private static int inputWidth(HfTensorIndex in, String weight, String matrix) {
        if (!in.has(weight)) {
            throw new GgufFormatException("transform " + matrix + " has no weight " + weight);
        }
        long[] shape = in.get(weight).shape();
        if (shape.length < 2 || shape[shape.length - 1] > Integer.MAX_VALUE) {
            throw new GgufFormatException("transformed weight " + weight + " has shape " + Arrays.toString(shape));
        }
        return (int) shape[shape.length - 1];
    }

    /** Checks the {@code [b, b]} shape of a deterministic transform and returns its all-ones diagonal. */
    private static float[] matrixShape(RowSource t, String name, int b) {
        long[] shape = t.shape();
        if (shape.length != 2 || shape[0] != b || shape[1] != b) {
            throw new GgufFormatException("transform " + name + " has shape " + Arrays.toString(shape) + "; expected ["
                    + b + ", " + b + "]");
        }
        float[] d = new float[b];
        Arrays.fill(d, 1f);
        return d;
    }

    /** The diagonal {@code d} of a stored random Hadamard matrix {@code R[i][j] = d_i * (-1)^popcount(i & j)}. */
    private static float[] randomSigns(RowSource t, String name, int b) throws IOException {
        matrixShape(t, name, b);
        float[] r = Fp4Packing.readScales(t, name, false);
        float[] d = new float[b];
        for (int i = 0; i < b; i++) {
            float di = r[i * b];
            if (di != 1f && di != -1f) {
                throw new GgufFormatException("transform " + name + " row " + i + " starts with " + di
                        + "; a random Hadamard matrix holds +-1 entries");
            }
            for (int j = 0; j < b; j++) {
                if (r[i * b + j] != di * sylvesterSign(i, j)) {
                    throw new GgufFormatException("transform " + name + " entry (" + i + ", " + j + ") is "
                            + r[i * b + j] + "; it is not a signed Sylvester Hadamard matrix");
                }
            }
            d[i] = di;
        }
        return d;
    }

    /**
     * Checks an fp_quant {@code forward_hadamard_matrix}: within 1e-3 of {@code H / sqrt(n)} for {@code hadamard},
     * of the identity for {@code identity}.
     */
    static void checkFpQuantMatrix(float[] m, int n, boolean hadamard, String name) {
        if (m.length != n * n) {
            throw new GgufFormatException("fp_quant matrix " + name + " has " + m.length + " entries; expected " + n
                    + "x" + n);
        }
        float scale = (float) (1.0 / Math.sqrt(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float want = hadamard ? scale * sylvesterSign(i, j) : i == j ? 1f : 0f;
                if (!(Math.abs(m[i * n + j] - want) <= 1e-3f)) {
                    throw new GgufFormatException("fp_quant " + name + " entry (" + i + ", " + j + ") is "
                            + m[i * n + j] + "; expected " + want + (hadamard ? " (the normalized Sylvester matrix)"
                            : " (the identity)"));
                }
            }
        }
    }
}
