package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.SourceOrigin;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.repack.CpuRepack;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The tensors of one model for the CPU, read through the {@link TensorRegions} of its {@link ModelSource}: in place
 * from memory-mapped file regions, or, in {@link TensorReadMode#READ} (llama.cpp {@code --no-mmap}), from direct
 * buffers filled by positional file reads.
 *
 * <ul>
 *   <li><b>Matrices</b> are {@link CpuTensorRef} views over the tensor bytes in the tensor's own type; nothing is
 *       dequantized and no size cap applies. Each tensor is mapped (or read) on first use in row-aligned regions of
 *       at most {@value #MAX_REGION_BYTES} bytes and cached, so a name always yields the same view.</li>
 *   <li><b>Small vectors</b> (norm weights, biases, relative-position tables, scales) are decoded to F32 once by
 *       the {@code floats} lookups and cached. The returned arrays are shared and must not be modified.</li>
 *   <li><b>Transforms.</b> This class is the {@link CpuMatVec.TransformSource} of its {@link #matVec()}, which
 *       applies a weight's full input transform before the weight kernel, in the order HADAMARD →
 *       ACT_QUANT → PERMUTE ({@link InputTransform}): a weight folded by PRISM or compressed-tensors gets its
 *       input rotation (llama.cpp src/llama-graph.cpp:1598-1626), a checkpoint's activation fake-quantization and
 *       act-order input permutation ride along ({@link InputTransformLoader.Result#forward()}), a weight
 * {@code <base>.weight}
 *       with a sibling {@code <base>.scale} tensor gets that output scale {@code w_s} (one value, or one per expert
 * for a 3-D expert tensor; llama.cpp src/llama-model.cpp:1718-1792, src/llama-graph.cpp:1643-1645, 1722-1729), and a
 *       latent lookup table gets its inverse Hadamard for {@link #embedRow} (llama.cpp src/llama-graph.cpp:2683-2690).
 *       This class loads no transform itself: it only binds the {@link InputTransformLoader.Result} that {@code open}
 *       receives.</li>
 *   <li><b>Repacking.</b> Matmul weights whose layout the Vector API kernels exploit are copied once into an
 *       interleaved layout by {@link CpuRepack} (llama.cpp ggml-cpu/repack.cpp) when their first product asks for
 *       it, and every later product runs over the copy; a tensor used as a lookup table ({@link #embedRow}) is
 *       never copied, so its mapped rows keep serving it. {@value CpuRepack#PROP_REPACK} selects the types
 *       ({@code auto} by default, {@code on}, {@code off}) and {@value CpuRepack#PROP_BUDGET} caps the copies,
 *       which are also bounded by the free heap; a weight that does not fit stays on the plain kernels.</li>
 * </ul>
 *
 * <p>Lookups resolve aliases as {@link TensorCatalog} does. Required lookups throw {@link MissingTensorException}
 * naming the architecture, layer and tensor; {@code OrNull} lookups return null. Single-file and split
 * ({@code -0000N-of-0000M.gguf}) models are both mapped in place. Thread-safe.</p>
 */
public final class CpuWeights implements CpuMatVec.TransformSource, AutoCloseable {

    /** Largest mapped region of one tensor; larger tensors are split on row boundaries. */
    static final long MAX_REGION_BYTES = 1L << 30;

    private static final String WEIGHT_SUFFIX = "." + TensorNames.WEIGHT;

    /** One loaded tensor: its description and bound transforms. */
    private static final class Entry {
        final TensorRef ref;
        final CpuMatVec.Transforms transforms;
        final HadamardSpec inverse;

        Entry(TensorRef ref, CpuMatVec.Transforms transforms, HadamardSpec inverse) {
            this.ref = ref;
            this.transforms = transforms;
            this.inverse = inverse;
        }
    }

    private final TensorCatalog catalog;
    private final TensorRegions regions;
    private final TensorReadMode mode;
    private final SourceOrigin origin;
    private final InputTransformLoader.Result transforms;
    private final CpuRepack repack;
    private final CpuMatVec matVec;
    private final Map<String, CpuTensorRef> views = new ConcurrentHashMap<>();
    private final Map<CpuTensorRef, Entry> entries = new ConcurrentHashMap<>();
    private final Map<String, float[]> vectors = new ConcurrentHashMap<>();
    private final Object loadLock = new Object();
    private volatile boolean closed;

    private CpuWeights(TensorCatalog catalog, TensorRegions regions, TensorReadMode mode, SourceOrigin origin,
                       InputTransformLoader.Result transforms) {
        this.catalog = catalog;
        this.regions = regions;
        this.mode = mode;
        this.origin = origin;
        this.transforms = transforms;
        this.repack = new CpuRepack();
        this.matVec = new CpuMatVec(this, repack::planFor);
    }

    /**
     * Weights of an open model, with the input transforms its metadata and tensors declare
     * ({@link InputTransformLoader}, activation quantization ignored as {@link ActivationQuant#WEIGHT_ONLY}), read
     * in the process's {@link TensorReadMode#current()} mode.
     *
     * @throws IOException         when a tensor file cannot be opened
     * @throws GgufFormatException when the transform metadata or tensors are malformed
     * @throws IllegalArgumentException when {@value TensorReadMode#PROP_DISABLE_MMAP} is not a boolean
     */
    public static CpuWeights open(ModelSource source) throws IOException {
        Objects.requireNonNull(source, "source");
        TensorCatalog catalog = TensorCatalog.of(source);
        GgufKv kv = GgufKv.of(source);
        return open(source, catalog, InputTransformLoader.load(source, kv, catalog, ActivationQuant.WEIGHT_ONLY));
    }

    /**
     * Weights of an open model over an existing catalog of its tensors, read in the process's
     * {@link TensorReadMode#current()} mode. The full input transform of every weight
     * ({@link InputTransformLoader.Result#forward()}) is applied by the weights' {@link #matVec()}.
     *
     * @param transforms the model's input transforms, or null when it has none
     * @throws IOException         when a tensor file cannot be opened
     * @throws GgufFormatException when a transform does not fit the tensor it is bound to
     * @throws IllegalArgumentException when {@value TensorReadMode#PROP_DISABLE_MMAP} is not a boolean
     */
    public static CpuWeights open(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms)
            throws IOException {
        return open(source, catalog, transforms, TensorReadMode.current());
    }

    /**
     * Weights of an open model over an existing catalog of its tensors, read through the source's
     * {@link ModelSource#openRegions(TensorReadMode) regions}, which these weights own and close. The full input
     * transform of every weight ({@link InputTransformLoader.Result#forward()}) is applied by the weights'
     * {@link #matVec()}.
     *
     * @param transforms the model's input transforms, or null when it has none
     * @param mode       whether tensors are memory-mapped or read into direct buffers
     * @throws IOException         when a tensor file cannot be opened
     * @throws GgufFormatException when a transform does not fit the tensor it is bound to
     */
    public static CpuWeights open(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                                  TensorReadMode mode) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(catalog, "catalog");
        Objects.requireNonNull(mode, "mode");
        SourceOrigin origin = new SourceOrigin(Paths.get(source.getFilePath()), source.format());
        TensorRegions regions = source.openRegions(mode);
        try {
            return new CpuWeights(catalog, regions, mode, origin, transforms);
        } catch (RuntimeException | Error e) {
            try {
                regions.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /** Whether the tensors are memory-mapped or read into direct buffers. */
    public TensorReadMode readMode() {
        return mode;
    }

    /** The checkpoint the weights were opened from. */
    public SourceOrigin origin() {
        return origin;
    }

    /** The catalog the lookups resolve through. */
    public TensorCatalog catalog() {
        return catalog;
    }

    /** {@code general.architecture} of the model, used in missing-tensor messages. */
    public String archName() {
        return catalog.archName();
    }

    /** The matvec bound to this model's transforms; it runs repacked weights over their {@link CpuRepack} copies. */
    public CpuMatVec matVec() {
        return matVec;
    }

    /** The repacker this instance's matvec consults, holding one on-heap copy per repacked weight. */
    public CpuRepack repack() {
        return repack;
    }

    // ------------------------------------------------------------------ matrices

    /** The tensor stored under exactly this name, or null. */
    public CpuTensorRef find(String storedName) {
        ensureOpen();
        CpuTensorRef v = views.get(Objects.requireNonNull(storedName, "name"));
        if (v != null) {
            return v;
        }
        GGUFTensorInfo info = catalog.info(storedName);
        return info == null ? null : load(info);
    }

    /** Tensor by name or model alias, or null. */
    public CpuTensorRef getOrNull(String name) {
        return view(catalog.get(name));
    }

    /**
     * Required tensor by name or model alias.
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public CpuTensorRef get(String name) {
        return view(catalog.require(name));
    }

    /** True when {@link #getOrNull(String)} finds the tensor. */
    public boolean has(String name) {
        return catalog.get(name) != null;
    }

    /** True when {@code w} is a view these weights created, so {@link #describe} names it. */
    public boolean owns(CpuTensorRef w) {
        return w != null && entries.containsKey(w);
    }

    /** The tensor a catalog reference names; null for a null reference. */
    public CpuTensorRef get(TensorRef ref) {
        return view(ref);
    }

    /** Main-stack tensor {@code blk.<il>.<suffix>} or its Hugging Face alias, or null. */
    public CpuTensorRef layerOrNull(int il, String suffix) {
        return view(catalog.layer(il, suffix));
    }

    /**
     * Required main-stack tensor.
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public CpuTensorRef layer(int il, String suffix) {
        return view(catalog.require(il, suffix));
    }

    /** Tensor {@code <stack>.<il>.<suffix>} of any stack, or null. */
    public CpuTensorRef stackOrNull(String stack, int il, String suffix) {
        return view(catalog.layer(stack, il, suffix));
    }

    /**
     * Required tensor of any stack.
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public CpuTensorRef stack(String stack, int il, String suffix) {
        return view(catalog.require(stack, il, suffix));
    }

    /** Tensor of a layer in its own stack at its weight layer ({@code LayerSpec.weightLayer()}), or null. */
    public CpuTensorRef layerOrNull(LayerSpec layer, String suffix) {
        return stackOrNull(layer.stack(), layer.weightLayer(), suffix);
    }

    /**
     * Required tensor of a layer in its own stack at its weight layer.
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public CpuTensorRef layer(LayerSpec layer, String suffix) {
        CpuTensorRef w = layerOrNull(layer, suffix);
        if (w == null) {
            throw new MissingTensorException(archName(), layer.il(), layer.tensorName(suffix));
        }
        return w;
    }

    // ------------------------------------------------------------------ descriptions and transforms

    /**
     * The description ({@code ne}, type, stored name) of a view returned by this instance.
     *
     * @throws IllegalArgumentException for a view this instance did not create
     */
    public TensorRef describe(CpuTensorRef w) {
        return entry(w).ref;
    }

    /** The inverse PRISM Hadamard applied to rows looked up from {@code table}, or null. */
    public HadamardSpec inverseHadamard(CpuTensorRef table) {
        return entry(table).inverse;
    }

    @Override
    public CpuMatVec.Transforms of(CpuTensorRef w) {
        Entry e = entries.get(Objects.requireNonNull(w, "weight"));
        return e != null ? e.transforms : CpuMatVec.Transforms.NONE;
    }

    /**
     * Decodes row {@code row} of a lookup table (token, position or type embeddings) into
     * {@code dst[dstOff .. dstOff + cols)}, then applies the table's inverse PRISM Hadamard when it has one. The
     * table stays in place: a repacked copy would not replace any of its mapped rows, which every lookup reads.
     *
     * @throws IndexOutOfBoundsException when the row is outside the table
     */
    public void embedRow(CpuTensorRef table, long row, float[] dst, int dstOff) {
        Entry e = entry(table);
        repack.keepInPlace(table, "lookup table");
        table.decodeRow(row, dst, dstOff);
        if (e.inverse != null) {
            CpuHadamard.applyInPlace(e.inverse, dst, dstOff, 1, (int) table.cols());
        }
    }

    // ------------------------------------------------------------------ small vectors

    /**
     * A tensor decoded to F32, by name or model alias, or null. The array is cached and shared: read it only.
     *
     * @throws IllegalArgumentException when the tensor has more elements than one float array holds
     */
    public float[] floatsOrNull(String name) {
        return floats(catalog.get(name));
    }

    /**
     * A required tensor decoded to F32 (shared, read only).
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public float[] floats(String name) {
        return floats(catalog.require(name));
    }

    /** A main-stack tensor decoded to F32 (shared, read only), or null. */
    public float[] layerFloatsOrNull(int il, String suffix) {
        return floats(catalog.layer(il, suffix));
    }

    /**
     * A required main-stack tensor decoded to F32 (shared, read only).
     *
     * @throws MissingTensorException when the model has no such tensor
     */
    public float[] layerFloats(int il, String suffix) {
        return floats(catalog.require(il, suffix));
    }

    /** A tensor of any stack decoded to F32 (shared, read only), or null. */
    public float[] stackFloatsOrNull(String stack, int il, String suffix) {
        return floats(catalog.layer(stack, il, suffix));
    }

    /** A layer's tensor at its weight layer decoded to F32 (shared, read only), or null. */
    public float[] layerFloatsOrNull(LayerSpec layer, String suffix) {
        return stackFloatsOrNull(layer.stack(), layer.weightLayer(), suffix);
    }

    /** The tensor a catalog reference names, decoded to F32 (shared, read only); null for a null reference. */
    public float[] floats(TensorRef ref) {
        if (ref == null) {
            return null;
        }
        ensureOpen();
        float[] v = vectors.get(ref.name());
        if (v != null) {
            return v;
        }
        CpuTensorRef w = view(ref);
        long n = w.rows() * w.cols();
        if (n > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("tensor " + ref.name() + " has " + n
                    + " elements, too many for one float array; read it through its CpuTensorRef");
        }
        float[] decoded = new float[(int) n];
        w.decodeRows(0L, (int) w.rows(), decoded, 0);
        float[] prev = vectors.putIfAbsent(ref.name(), decoded);
        return prev != null ? prev : decoded;
    }

    // ------------------------------------------------------------------ loading

    private CpuTensorRef view(TensorRef ref) {
        if (ref == null) {
            return null;
        }
        CpuTensorRef v = find(ref.name());
        if (v == null) {
            throw new IllegalStateException("catalog tensor " + ref.name() + " has no parser description");
        }
        return v;
    }

    private Entry entry(CpuTensorRef w) {
        Entry e = entries.get(Objects.requireNonNull(w, "tensor"));
        if (e == null) {
            throw new IllegalArgumentException(w + " was not created by these weights");
        }
        return e;
    }

    /** Maps a tensor and binds its transforms; one view per stored name. */
    private CpuTensorRef load(GGUFTensorInfo info) {
        synchronized (loadLock) {
            ensureOpen();
            CpuTensorRef existing = views.get(info.name());
            if (existing != null) {
                return existing;
            }
            TensorRef ref = TensorRef.of(info);
            CpuTensorRef view;
            try {
                view = mapTensor(info, ref);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot " + (mode.usesMmap() ? "map" : "read")
                        + " tensor " + info.name() + " of " + regions.description(), e);
            }
            int experts = ref.rank() >= 3 ? checkedInt(ref.ne(2), ref) : 0;
            float[] scale = outputScale(ref, experts);
            InputTransform it = transforms != null ? transforms.forward(ref.name()) : InputTransform.IDENTITY;
            it = checkedInputTransform(it, experts, ref);
            HadamardSpec in = checkedSpec(it.hadamard(), ref, "input");
            HadamardSpec inv = checkedSpec(transforms != null ? transforms.inverse().get(ref.name()) : null, ref,
                    "inverse");
            entries.put(view, new Entry(ref, CpuMatVec.Transforms.from(it, scale, experts), inv));
            views.put(ref.name(), view);
            // A matmul weight is repacked when its first product asks for it, not here: only the product path can
            // tell a matmul from a lookup table, and the first product runs once the KV cache is allocated.
            return view;
        }
    }

    // ------------------------------------------------------------------ live updates

    /**
     * Writes updated contents over the loaded view of the tensor {@code name}: a live weight update (vLLM's
     * in-process weight transfer). The new bytes become the view every later lookup returns; a forward pass running
     * while the view swaps keeps the bytes it already read, the next one serves the update. The decoded small-vector
     * cache of the tensor and its repacked copy drop, so a later product sees the new contents either way.
     *
     * @param name  the tensor's stored name, exactly as the model file holds it
     * @param dtype element type the {@code data} bytes hold: the tensor's stored type — the CPU consumes weights as
     *             stored, so F32 data is refused unless the tensor itself is F32
     * @param ne    ggml shape of the update; must equal the model's
     * @param data  the whole tensor, little-endian, row-major over {@code ne[1..]}
     * @throws IllegalArgumentException      when the model has no tensor of that name, or the shape or dtype does
     *                                       not match
     * @throws UnsupportedOperationException when the tensor was never mapped (it serves nothing yet)
     */
    public void update(String name, GGMLType dtype, long[] ne, ByteBuffer data) {
        Objects.requireNonNull(dtype, "dtype");
        Objects.requireNonNull(data, "data");
        synchronized (loadLock) {
            ensureOpen();
            GGUFTensorInfo info = catalog.info(Objects.requireNonNull(name, "name"));
            if (info == null) {
                throw new IllegalArgumentException(catalog.archName() + " has no tensor " + name + " to update");
            }
            CpuTensorRef old = views.get(info.name());
            if (old == null) {
                throw new UnsupportedOperationException("tensor " + name + " was never mapped; the CPU model"
                        + " loads its tensors on first use and holds nothing to update");
            }
            if (!Arrays.equals(info.dims(), ne)) {
                throw new IllegalArgumentException("tensor " + name + " has shape " + Arrays.toString(info.dims())
                        + ", the update carries " + Arrays.toString(ne));
            }
            if (dtype != info.type()) {
                throw new IllegalArgumentException("tensor " + name + " is stored and consumed as " + info.type()
                        + "; " + (dtype == GGMLType.F32 ? "F32 update data would requantize it — send its stored "
                        + info.type() + " bytes instead" : "updates arrive as the stored type, got " + dtype));
            }
            CpuTensorRef fresh = CpuTensorRef.of(info.type(),
                    ByteBuffer.allocateDirect(data.remaining()).order(ByteOrder.LITTLE_ENDIAN).put(data.duplicate())
                            .flip(),
                    old.rows(), old.cols());
            Entry e = entries.remove(old);
            if (e == null) {
                throw new IllegalStateException("tensor " + name + " has no bound entry in these weights");
            }
            entries.put(fresh, new Entry(TensorRef.of(info), e.transforms, e.inverse));
            views.put(info.name(), fresh);
            vectors.remove(info.name());
            repack.discard(old);
        }
    }

    /**
     * Maps (or reads) {@code rows x rowBytes} bytes in row-aligned regions of at most {@link #MAX_REGION_BYTES},
     * none crossing a boundary of the tensor's contiguous runs.
     */
    private CpuTensorRef mapTensor(GGUFTensorInfo info, TensorRef ref) throws IOException {
        GGMLType type = info.type();
        long rows = ref.rows();
        long cols = ref.cols();
        long rowBytes = BlockLayout.of(type).rowBytes(cols);
        long total = Math.multiplyExact(rows, rowBytes);
        if (total != type.getBytesForElements(ref.numElements())) {
            throw new GgufFormatException("tensor " + ref.name() + " (" + type + ", ne=" + Arrays.toString(ref.ne())
                    + ") does not split into whole rows of " + rowBytes + " bytes");
        }
        if (rowBytes > MAX_REGION_BYTES) {
            throw new GgufFormatException("tensor " + ref.name() + " has rows of " + rowBytes
                    + " bytes, larger than one mapped region");
        }
        long rowsPerRun = runRows(info, rows, rowBytes, total);
        long rowsPerRegion = Math.min(rowsPerRun, Math.max(1L, MAX_REGION_BYTES / rowBytes));
        List<ByteBuffer> shards = new ArrayList<>();
        List<Long> offsets = new ArrayList<>();
        long firstRow = 0;
        do {
            long runEnd = (firstRow / rowsPerRun + 1) * rowsPerRun;
            long end = Math.min(rows, Math.min(firstRow + rowsPerRegion, runEnd));
            offsets.add(firstRow * rowBytes);
            shards.add(regions.region(info, firstRow * rowBytes, (end - firstRow) * rowBytes));
            firstRow = end;
        } while (firstRow < rows);
        long[] starts = new long[offsets.size()];
        for (int i = 0; i < starts.length; i++) {
            starts[i] = offsets.get(i);
        }
        return new CpuTensorRef(type, shards.toArray(new ByteBuffer[0]), starts, rows, cols, rowBytes);
    }

    /**
     * Rows of one contiguous run of {@code info} ({@link TensorRegions#contiguousRunBytes}), so no mapped region
     * crosses a run boundary and every region is served in place; the whole tensor when it is one run.
     */
    private long runRows(GGUFTensorInfo info, long rows, long rowBytes, long total) throws GgufFormatException {
        long runBytes = regions.contiguousRunBytes(info);
        if (runBytes >= total || rows == 0) {
            return Math.max(1L, rows);
        }
        if (runBytes <= 0 || runBytes % rowBytes != 0) {
            throw new GgufFormatException("tensor " + info.name() + " is stored in runs of " + runBytes
                    + " bytes, not a positive multiple of its " + rowBytes + "-byte rows");
        }
        return runBytes / rowBytes;
    }

    /**
     * The values of {@code <base>.scale} for a weight {@code <base>.weight}: one value for the whole tensor, or
     * {@code experts} values for an expert tensor; null when the model has no such tensor. The router's
     * {@code ffn_gate_inp.scale} is excluded: it is an {@code {n_embd}} scale of the router INPUT (gemma4.cpp:327),
     * applied by the MoE router block before the projection, not an output scale of the matrix.
     */
    private float[] outputScale(TensorRef weight, int experts) {
        String name = weight.name();
        if (!name.endsWith(WEIGHT_SUFFIX)) {
            return null;
        }
        String base = name.substring(0, name.length() - WEIGHT_SUFFIX.length());
        if (base.equals(TensorNames.FFN_GATE_INP) || base.endsWith("." + TensorNames.FFN_GATE_INP)) {
            return null;
        }
        String scaleName = TensorNames.scale(base);
        TensorRef scale = catalog.get(scaleName);
        if (scale == null) {
            return null;
        }
        long n = scale.numElements();
        if (n != 1 && (experts == 0 || n != experts)) {
            throw new GgufFormatException("scale tensor " + scaleName + " has " + n + " values; " + name
                    + " needs 1" + (experts > 0 ? " or " + experts + " (one per expert)" : ""));
        }
        return floats(scale);
    }

    private static HadamardSpec checkedSpec(HadamardSpec spec, TensorRef ref, String role) {
        if (spec == null) {
            return null;
        }
        try {
            spec.checkWidth(checkedInt(ref.cols(), ref));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("PRISM " + role + " Hadamard does not fit tensor " + ref.name() + ": "
                    + e.getMessage(), e);
        }
        return spec;
    }

    /**
     * Validates the input transform {@code it} that {@code transforms} binds to the weight {@code ref}: the row
     * splits into its transform parts ({@link InputTransform#requireWidth}), a PERMUTE part never serves a 3-D
     * expert tensor (TP-1 folds it), and the ACT_QUANT parameters of a per-expert scheme match the expert count
     * ({@link InputTransform.ActQuant}).
     *
     * @throws GgufFormatException naming the tensor and the part that does not fit
     */
    private static InputTransform checkedInputTransform(InputTransform it, int experts, TensorRef ref) {
        if (it.isIdentity()) {
            return it;
        }
        if (it.hasPerm() && experts > 0) {
            throw new GgufFormatException("input permutation of tensor " + ref.name()
                    + ": a 3-D expert tensor folds its input permutation into the weights (TP-1)");
        }
        try {
            it.requireWidth(checkedInt(ref.cols(), ref));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("input transform does not fit tensor " + ref.name() + ": "
                    + e.getMessage(), e);
        }
        InputTransform.ActQuant actQuant = it.actQuant();
        if (actQuant != null && actQuant.entries() > 1 && actQuant.entries() != experts) {
            throw new GgufFormatException("activation quantization of tensor " + ref.name() + " has "
                    + actQuant.entries() + " parameter set(s)" + (experts == 0 ? ", but the tensor is not an"
                            + " expert tensor" : ", but the tensor holds " + experts + " expert(s)"));
        }
        float[] preScale = it.preScale();
        if (preScale != null) {
            int cols = checkedInt(ref.cols(), ref);
            if (preScale.length != cols && (experts == 0 || preScale.length != experts * cols)) {
                throw new GgufFormatException("activation pre-scale of tensor " + ref.name() + " has "
                        + preScale.length + " values; expected " + cols
                        + (experts > 0 ? " or " + experts * cols + " (one per expert)" : ""));
            }
        }
        return it;
    }

    private static int checkedInt(long v, TensorRef ref) {
        if (v > Integer.MAX_VALUE) {
            throw new GgufFormatException("dimension " + v + " of tensor " + ref.name() + " exceeds the int range");
        }
        return (int) v;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("CPU weights of " + regions.description() + " are closed");
        }
    }

    /**
     * Closes the tensor files. Views handed out earlier stay readable until they are no longer referenced (mapped
     * and direct memory is released by the garbage collector); new lookups fail.
     */
    @Override
    public void close() throws IOException {
        synchronized (loadLock) {
            if (closed) {
                return;
            }
            closed = true;
            views.clear();
            entries.clear();
            vectors.clear();
            repack.clear();
            regions.close();
        }
    }

    @Override
    public String toString() {
        return "CpuWeights{" + archName() + ", " + regions.description() + ", loaded=" + views.size() + "}";
    }
}
