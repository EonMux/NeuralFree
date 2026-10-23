package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;

/**
 * Every tensor of one physical weight layer that is streamed to the device, with heterogeneous types and shapes,
 * planned for upload: per tensor the device type (stored, or F32 when the device cannot consume the stored type), the
 * row shards that keep each buffer within the device's allocation and binding limits, the PRISM input transform and
 * the per-tensor output scale.
 *
 * <p>A layer's tensors are the file's {@code <stack>.<weightLayer>.*} tensors (for the main stack also the Hugging Face
 * alias {@code model.layers.<weightLayer>.*}), where stack and weight layer come from the {@link LayerSpec}s of the
 * trunk. Model-level tensors, encoder layers and NextN layers are never streamed. Immutable.</p>
 */
public final class LayerWeightSet {

    private static final String WEIGHT_SUFFIX = ".weight";
    private static final String SCALE_SUFFIX = ".scale";

    /**
     * Device type of a tensor. It must be the same per-tensor decision the resident uploader makes (for the engine:
     * {@code t -> uploader.deviceType(t.name())} of {@code GpuWeightUploader}), so a block reads a streamed tensor in
     * the type it would read the resident one: projection, lookup and expert weights keep their stored type only
     * where the backend has the kernels, element-wise and special tensors (norms, biases, convolution and recurrent
     * parameters, LoRA factors, ...) are F32 whatever their rank, integer tables stay as stored.
     */
    @FunctionalInterface
    public interface UploadPolicy {
        /** The type the device buffer holds: the stored type, or {@link GGMLType#F32} to dequantize on the host. */
        GGMLType uploadType(TensorRef tensor);
    }

    /** Planning options. Immutable. */
    public static final class Options {
        private final UploadPolicy policy;
        private final Predicate<TensorRef> filter;
        private final long maxShardBytes;
        private final Map<String, HadamardSpec> hadamard;

        private Options(Builder b) {
            this.policy = b.policy;
            this.filter = b.filter;
            this.maxShardBytes = b.maxShardBytes;
            this.hadamard = Collections.unmodifiableMap(new TreeMap<>(b.hadamard));
        }

        /** Options with the given per-tensor device type decision, every layer tensor and no shard limit. */
        public static Builder builder(UploadPolicy policy) {
            return new Builder(policy);
        }

        /**
         * Options for a backend: the given device type decision, every layer tensor, and shards no larger than
         * {@code min(maxAllocSize, maxStorageBufferRange)}.
         */
        public static Options forBackend(ComputeBackend backend, UploadPolicy policy) {
            return builder(policy)
                    .maxShardBytes(maxShardBytes(Objects.requireNonNull(backend, "backend").capabilities()))
                    .build();
        }

        /** Largest single buffer the device both allocates and binds; {@link Long#MAX_VALUE} when unknown. */
        public static long maxShardBytes(DeviceCapabilities caps) {
            return WeightShards.maxBufferBytes(caps);
        }

        public UploadPolicy policy() { return policy; }
        /** Which layer tensors are streamed (the others are expected among the resident weights). */
        public Predicate<TensorRef> filter() { return filter; }
        public long maxShardBytes() { return maxShardBytes; }
        /** PRISM input transforms by stored tensor name (unmodifiable). */
        public Map<String, HadamardSpec> hadamard() { return hadamard; }

        public static final class Builder {
            private final UploadPolicy policy;
            private Predicate<TensorRef> filter = t -> true;
            private long maxShardBytes = Long.MAX_VALUE;
            private Map<String, HadamardSpec> hadamard = Collections.emptyMap();

            private Builder(UploadPolicy policy) {
                this.policy = Objects.requireNonNull(policy, "policy");
            }

            /** Which layer tensors are streamed; the default streams all of them. */
            public Builder filter(Predicate<TensorRef> v) {
                this.filter = Objects.requireNonNull(v, "filter");
                return this;
            }

            /** Largest bytes of one device buffer; positive. */
            public Builder maxShardBytes(long v) {
                if (v <= 0) {
                    throw new IllegalArgumentException("maximum shard size must be positive, got " + v);
                }
                this.maxShardBytes = v;
                return this;
            }

            /** PRISM forward transforms ({@code PrismHadamardLoader.Result.forward()}). */
            public Builder hadamard(Map<String, HadamardSpec> v) {
                this.hadamard = Objects.requireNonNull(v, "hadamard");
                return this;
            }

            public Options build() {
                return new Options(this);
            }
        }
    }

    /** One streamed tensor: its stored form and its device form. Immutable. */
    public static final class Entry {
        private final TensorRef source;
        private final GGMLType uploadType;
        private final long uploadRowBytes;
        private final long[] shardRows;
        private final HadamardSpec hadamard;
        private final TensorRef scale;

        Entry(TensorRef source, GGMLType uploadType, long uploadRowBytes, long[] shardRows, HadamardSpec hadamard,
              TensorRef scale) {
            this.source = source;
            this.uploadType = uploadType;
            this.uploadRowBytes = uploadRowBytes;
            this.shardRows = shardRows;
            this.hadamard = hadamard;
            this.scale = scale;
        }

        /** Stored name, type and shape. */
        public TensorRef source() { return source; }
        public String name() { return source.name(); }
        /** ggml shape (copy); identical on the host and the device. */
        public long[] ne() { return source.ne(); }
        /** Input width {@code ne[0]}. */
        public long cols() { return source.cols(); }
        /** Rows: the product of {@code ne[1..]}. */
        public long rows() { return source.rows(); }
        /** Type of the device buffer. */
        public GGMLType uploadType() { return uploadType; }
        /** True when the stored rows are decoded to F32 on the host before upload. */
        public boolean dequantize() { return uploadType != source.type(); }
        /** Bytes of one stored row. */
        public long storedRowBytes() { return source.type().getBytesForElements(source.cols()); }
        /** Bytes of one device row. */
        public long uploadRowBytes() { return uploadRowBytes; }
        public int shardCount() { return shardRows.length; }
        /** Rows of shard {@code i}; shards are contiguous from row 0. */
        public long shardRows(int i) { return shardRows[i]; }
        /** Bytes of the device buffer of shard {@code i}. */
        public long shardBytes(int i) { return shardRows[i] * uploadRowBytes; }
        /** Row counts of all shards (copy). */
        public long[] shardRowCounts() { return shardRows.clone(); }
        /** PRISM input transform, or null. */
        public HadamardSpec hadamard() { return hadamard; }
        /**
         * The scalar {@code <base>.scale} tensor holding the output scale {@code w_s} of this {@code <base>.weight}
         * (llama-model.cpp:1724-1790, applied at llama-graph.cpp:1643-1645), or null. Expert
         * tensors never carry one, as on the resident path.
         */
        public TensorRef scale() { return scale; }
        /** Bytes of the whole tensor on the device. */
        public long uploadBytes() { return rows() * uploadRowBytes; }

        @Override
        public String toString() {
            return source + (dequantize() ? " -> " + uploadType : "")
                    + (shardRows.length > 1 ? " x" + shardRows.length : "");
        }
    }

    private final String stack;
    private final int weightLayer;
    private final List<Entry> entries;
    private final long uploadBytes;

    private LayerWeightSet(String stack, int weightLayer, List<Entry> entries) {
        this.stack = stack;
        this.weightLayer = weightLayer;
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
        long total = 0;
        for (Entry e : entries) {
            total += e.uploadBytes();
        }
        this.uploadBytes = total;
    }

    /**
     * One set per physical weight layer of the trunk, in weight-layer order; a layer whose filtered tensor list is
     * empty gets no set.
     *
     * @throws IllegalArgumentException when two trunk layers share a weight layer but not a stack
     * @throws IllegalStateException    when a tensor can neither be consumed nor decoded on the host, or one row (one
     *                                  expert of a 3-D tensor) exceeds the shard limit
     */
    public static List<LayerWeightSet> forTrunk(ModelSpec spec, Options options) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(options, "options");
        TreeMap<Integer, String> stacks = new TreeMap<>();
        for (LayerSpec l : spec.layers()) {
            String prev = stacks.put(l.weightLayer(), l.stack());
            if (prev != null && !prev.equals(l.stack())) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": weight layer " + l.weightLayer()
                        + " is used by stacks " + prev + " and " + l.stack());
            }
        }
        List<LayerWeightSet> out = new ArrayList<>(stacks.size());
        for (Map.Entry<Integer, String> e : stacks.entrySet()) {
            LayerWeightSet set = of(spec.tensors(), e.getValue(), e.getKey(), options);
            if (!set.entries.isEmpty()) {
                out.add(set);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The streamed tensors of one physical layer of a stack.
     *
     * @throws IllegalStateException as {@link #forTrunk}
     */
    public static LayerWeightSet of(TensorCatalog catalog, String stack, int weightLayer, Options options) {
        Objects.requireNonNull(catalog, "catalog");
        Objects.requireNonNull(options, "options");
        String prefix = TensorNames.layerName(stack, weightLayer, "x");
        prefix = prefix.substring(0, prefix.length() - 1);
        String hfPrefix = null;
        if (TensorNames.STACK_BLK.equals(stack)) {
            String hf = TensorNames.layerName(TensorNames.HF_LAYER_PREFIX, weightLayer, "x");
            hfPrefix = hf.substring(0, hf.length() - 1);
        }
        List<Entry> entries = new ArrayList<>();
        for (String name : catalog.names()) {
            if (!name.startsWith(prefix) && (hfPrefix == null || !name.startsWith(hfPrefix))) {
                continue;
            }
            TensorRef ref = catalog.get(name);
            if (options.filter().test(ref)) {
                entries.add(plan(catalog.archName(), ref, options, scaleOf(catalog, ref)));
            }
        }
        return new LayerWeightSet(stack, weightLayer, entries);
    }

    /**
     * The scalar {@code <base>.scale} of a non-expert {@code <base>.weight}, or null: the rule the resident uploader
     * attaches output scales by.
     */
    private static TensorRef scaleOf(TensorCatalog catalog, TensorRef ref) {
        String name = ref.name();
        if (!name.endsWith(WEIGHT_SUFFIX) || isExpert(ref)) {
            return null;
        }
        String scaleName = name.substring(0, name.length() - WEIGHT_SUFFIX.length()) + SCALE_SUFFIX;
        if (!catalog.has(scaleName)) {
            return null;
        }
        TensorRef scale = catalog.get(scaleName);
        return scale.numElements() == 1 ? scale : null;
    }

    /** Expert tensor {@code [cols, rows, n_expert]} ({@link WeightShards#isExpert}). */
    private static boolean isExpert(TensorRef ref) {
        return WeightShards.isExpert(ref.name(), ref.rank());
    }

    private static Entry plan(String arch, TensorRef ref, Options options, TensorRef scale) {
        GGMLType upload = Objects.requireNonNull(options.policy().uploadType(ref), "upload type of " + ref.name());
        if (upload != ref.type() && upload != GGMLType.F32) {
            throw new IllegalStateException(arch + ": upload policy converts " + ref + " to " + upload
                    + "; only the stored type or F32 is supported");
        }
        if (upload != ref.type() && !CpuBlockDecoders.supports(ref.type())) {
            throw new IllegalStateException(arch + ": tensor " + ref + " has a type the device cannot consume and the"
                    + " host cannot decode");
        }
        if (ref.rows() <= 0 || ref.cols() <= 0) {
            throw new IllegalStateException(arch + ": tensor " + ref + " is empty");
        }
        long rowBytes = upload.getBytesForElements(ref.cols());
        long unit = ref.rank() >= 3 ? ref.ne(1) : 1;
        long unitBytes = Math.multiplyExact(unit, rowBytes);
        long limit = options.maxShardBytes();
        if (unitBytes > limit) {
            throw new IllegalStateException(arch + ": tensor " + ref + " needs " + unitBytes + " bytes per "
                    + (unit > 1 ? "expert" : "row") + " as " + upload + ", more than the device buffer limit of "
                    + limit + " bytes");
        }
        long[] shardRows = WeightShards.shardRows(ref.rows(), unit, unitBytes, limit);
        return new Entry(ref, upload, rowBytes, shardRows, options.hadamard().get(ref.name()), scale);
    }

    /** Stack prefix of the layer's tensors. */
    public String stack() { return stack; }
    /** Physical weight layer ({@code LayerSpec.weightLayer()}). */
    public int weightLayer() { return weightLayer; }
    /** Streamed tensors in file order (unmodifiable). */
    public List<Entry> entries() { return entries; }
    /** Bytes of all tensors on the device. */
    public long uploadBytes() { return uploadBytes; }

    /** Stored names of the streamed tensors, in file order. */
    public Set<String> names() {
        Set<String> out = new LinkedHashSet<>();
        for (Entry e : entries) {
            out.add(e.name());
        }
        return Collections.unmodifiableSet(out);
    }

    /**
     * Stored names streamed per weight layer, element {@code L} for weight layer {@code L} (empty when the layer has
     * no set): the per-layer tensor lists of the host tier ({@code WeightDataProvider}).
     *
     * @throws IllegalArgumentException when a set's weight layer is outside {@code [0, nWeightLayers)} or repeats
     */
    public static List<List<String>> namesByLayer(List<LayerWeightSet> sets, int nWeightLayers) {
        Objects.requireNonNull(sets, "sets");
        List<List<String>> out = new ArrayList<>(nWeightLayers);
        for (int i = 0; i < nWeightLayers; i++) {
            out.add(Collections.<String>emptyList());
        }
        for (LayerWeightSet s : sets) {
            if (s.weightLayer < 0 || s.weightLayer >= nWeightLayers) {
                throw new IllegalArgumentException("weight layer " + s.weightLayer + " outside [0, " + nWeightLayers
                        + ")");
            }
            if (!out.get(s.weightLayer).isEmpty()) {
                throw new IllegalArgumentException("two streamed sets for weight layer " + s.weightLayer);
            }
            out.set(s.weightLayer, Collections.unmodifiableList(new ArrayList<>(s.names())));
        }
        return Collections.unmodifiableList(out);
    }

    /** Stored names of every streamed tensor of all sets (the tensors the resident uploader must skip). */
    public static Set<String> allNames(List<LayerWeightSet> sets) {
        Set<String> out = new LinkedHashSet<>();
        for (LayerWeightSet s : Objects.requireNonNull(sets, "sets")) {
            out.addAll(s.names());
        }
        return Collections.unmodifiableSet(out);
    }

    @Override
    public String toString() {
        return "LayerWeightSet{" + stack + "." + weightLayer + ", " + entries.size() + " tensors, " + uploadBytes
                + " B}";
    }
}
