package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The tensors of one GGUF model for the CPU, read in place from memory-mapped file regions, or, in
 * {@link TensorReadMode#READ} (llama.cpp {@code --no-mmap}), from direct buffers filled by positional file reads.
 *
 * <ul>
 *   <li><b>Matrices</b> are {@link CpuTensorRef} views over the tensor bytes in the tensor's own type; nothing is
 *       dequantized and no size cap applies. Each tensor is mapped (or read) on first use in row-aligned regions of
 *       at most {@value #MAX_REGION_BYTES} bytes and cached, so a name always yields the same view.</li>
 *   <li><b>Small vectors</b> (norm weights, biases, relative-position tables, scales) are decoded to F32 once by
 *       the {@code floats} lookups and cached. The returned arrays are shared and must not be modified.</li>
 *   <li><b>Transforms.</b> This class is the {@link CpuMatVec.TransformSource} of its {@link #matVec()}: a weight
 * folded by PRISM gets its input Hadamard (llama.cpp src/llama-graph.cpp:1598-1626), a weight {@code <base>.weight}
 *       with a sibling {@code <base>.scale} tensor gets that output scale {@code w_s} (one value, or one per expert
 * for a 3-D expert tensor; llama.cpp src/llama-model.cpp:1718-1792, src/llama-graph.cpp:1643-1645, 1722-1729), and a
 * latent lookup table gets its inverse Hadamard for {@link #embedRow} (llama.cpp src/llama-graph.cpp:2683-2690).</li>
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
    private final GgufRegions regions;
    private final Map<String, HadamardSpec> forward;
    private final Map<String, HadamardSpec> inverse;
    private final CpuMatVec matVec;
    private final Map<String, CpuTensorRef> views = new ConcurrentHashMap<>();
    private final Map<CpuTensorRef, Entry> entries = new ConcurrentHashMap<>();
    private final Map<String, float[]> vectors = new ConcurrentHashMap<>();
    private final Object loadLock = new Object();
    private volatile boolean closed;

    private CpuWeights(TensorCatalog catalog, GgufRegions regions, PrismHadamardLoader.Result prism) {
        this.catalog = catalog;
        this.regions = regions;
        this.forward = prism != null ? prism.forward() : Collections.<String, HadamardSpec>emptyMap();
        this.inverse = prism != null ? prism.inverse() : Collections.<String, HadamardSpec>emptyMap();
        this.matVec = new CpuMatVec(this);
    }

    /**
     * Weights of a parsed model, with the PRISM transforms its metadata declares, read in the process's
     * {@link TensorReadMode#current()} mode.
     *
     * @throws IOException         when a tensor file cannot be opened
     * @throws GgufFormatException when the PRISM metadata is malformed
     * @throws IllegalArgumentException when {@value TensorReadMode#PROP_DISABLE_MMAP} is not a boolean
     */
    public static CpuWeights open(GGUFFileParser parser) throws IOException {
        Objects.requireNonNull(parser, "parser");
        TensorCatalog catalog = TensorCatalog.of(parser);
        GgufKv kv = GgufKv.of(parser);
        PrismHadamardLoader.Result prism = PrismHadamardLoader.load(kv, name -> {
            TensorRef r = catalog.get(name);
            return r != null ? r.ne() : null;
        }, kv.u32(GgufKeys.SSM_TIME_STEP_RANK, 0), kv.u32(GgufKeys.SSM_GROUP_COUNT, 0));
        return open(parser, catalog, prism);
    }

    /**
     * Weights of a parsed model over an existing catalog of its tensors, read in the process's
     * {@link TensorReadMode#current()} mode.
     *
     * @param prism the model's PRISM transforms, or null when it has none
     * @throws IOException         when a tensor file cannot be opened
     * @throws IllegalArgumentException when {@value TensorReadMode#PROP_DISABLE_MMAP} is not a boolean
     */
    public static CpuWeights open(GGUFFileParser parser, TensorCatalog catalog, PrismHadamardLoader.Result prism)
            throws IOException {
        return open(parser, catalog, prism, TensorReadMode.current());
    }

    /**
     * Weights of a parsed model over an existing catalog of its tensors.
     *
     * @param prism the model's PRISM transforms, or null when it has none
     * @param mode  whether tensors are memory-mapped or read into direct buffers
     * @throws IOException         when a tensor file cannot be opened
     */
    public static CpuWeights open(GGUFFileParser parser, TensorCatalog catalog, PrismHadamardLoader.Result prism,
                                  TensorReadMode mode) throws IOException {
        Objects.requireNonNull(parser, "parser");
        Objects.requireNonNull(catalog, "catalog");
        Objects.requireNonNull(mode, "mode");
        return new CpuWeights(catalog, GgufRegions.open(parser, mode), prism);
    }

    /** Whether the tensors are memory-mapped or read into direct buffers. */
    public TensorReadMode readMode() {
        return regions.mode;
    }

    /** The catalog the lookups resolve through. */
    public TensorCatalog catalog() {
        return catalog;
    }

    /** {@code general.architecture} of the model, used in missing-tensor messages. */
    public String archName() {
        return catalog.archName();
    }

    /** The matvec bound to this model's transforms. */
    public CpuMatVec matVec() {
        return matVec;
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
     * {@code dst[dstOff .. dstOff + cols)}, then applies the table's inverse PRISM Hadamard when it has one.
     *
     * @throws IndexOutOfBoundsException when the row is outside the table
     */
    public void embedRow(CpuTensorRef table, long row, float[] dst, int dstOff) {
        Entry e = entry(table);
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
                throw new UncheckedIOException("cannot " + (regions.mode == TensorReadMode.MMAP ? "map" : "read")
                        + " tensor " + info.name() + " of " + regions, e);
            }
            int experts = ref.rank() >= 3 ? checkedInt(ref.ne(2), ref) : 0;
            float[] scale = outputScale(ref, experts);
            HadamardSpec in = checkedSpec(forward.get(ref.name()), ref, "input");
            HadamardSpec inv = checkedSpec(inverse.get(ref.name()), ref, "inverse");
            entries.put(view, new Entry(ref, CpuMatVec.Transforms.of(in, scale, experts), inv));
            views.put(ref.name(), view);
            return view;
        }
    }

    /** Maps (or reads) {@code rows x rowBytes} bytes in row-aligned regions of at most {@link #MAX_REGION_BYTES}. */
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
        long rowsPerRegion = Math.max(1L, MAX_REGION_BYTES / rowBytes);
        int count = (int) Math.max(1L, (rows + rowsPerRegion - 1) / rowsPerRegion);
        ByteBuffer[] shards = new ByteBuffer[count];
        long[] offsets = new long[count];
        for (int s = 0; s < count; s++) {
            long firstRow = s * rowsPerRegion;
            long n = Math.min(rowsPerRegion, rows - firstRow);
            offsets[s] = firstRow * rowBytes;
            shards[s] = regions.region(info, offsets[s], n * rowBytes);
        }
        return new CpuTensorRef(type, shards, offsets, rows, cols, rowBytes);
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

    private static int checkedInt(long v, TensorRef ref) {
        if (v > Integer.MAX_VALUE) {
            throw new GgufFormatException("dimension " + v + " of tensor " + ref.name() + " exceeds the int range");
        }
        return (int) v;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("CPU weights of " + regions + " are closed");
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
            regions.close();
        }
    }

    @Override
    public String toString() {
        return "CpuWeights{" + archName() + ", " + regions + ", loaded=" + views.size() + "}";
    }

    // ================================================================== file regions

    /**
     * The files holding a model's tensor data, as {@link GGUFFileParser#getShards()} describes them. The parser
     * places every tensor at a virtual offset; a tensor of split {@code s} lives in that split's file at
     * {@code Shard.fileOffset(info.offset())}. A single-file model is the one-split case. Regions are memory-mapped,
     * or in {@link TensorReadMode#READ} copied into direct buffers with positional reads.
     */
    private static final class GgufRegions implements AutoCloseable {

        private final String description;
        private final TensorReadMode mode;
        private final FileChannel[] channels;
        private final GGUFFileParser.Shard[] shards;
        private final Map<String, Integer> splitOf;

        private GgufRegions(String description, TensorReadMode mode, FileChannel[] channels,
                            GGUFFileParser.Shard[] shards, Map<String, Integer> splitOf) {
            this.description = description;
            this.mode = mode;
            this.channels = channels;
            this.shards = shards;
            this.splitOf = splitOf;
        }

        static GgufRegions open(GGUFFileParser parser, TensorReadMode mode) throws IOException {
            GGUFFileParser.Shard[] shards = parser.getShards().toArray(new GGUFFileParser.Shard[0]);
            Map<String, Integer> splitOf = shards.length > 1 ? indexSplits(parser, shards) : null;
            FileChannel[] channels = new FileChannel[shards.length];
            try {
                for (int s = 0; s < shards.length; s++) {
                    channels[s] = FileChannel.open(Paths.get(shards[s].path()), StandardOpenOption.READ);
                }
            } catch (IOException | RuntimeException e) {
                closeAll(channels, e);
                throw e;
            }
            String description = shards.length == 1 ? parser.getFilePath()
                    : parser.getFilePath() + " (" + shards.length + " splits)";
            return new GgufRegions(description, mode, channels, shards, splitOf);
        }

        /** The split of each tensor, by stored name. */
        private static Map<String, Integer> indexSplits(GGUFFileParser parser, GGUFFileParser.Shard[] shards) {
            List<GGUFTensorInfo> tensors = parser.getTensors();
            Map<String, Integer> splitOf = new HashMap<>(tensors.size() * 2);
            for (GGUFFileParser.Shard shard : shards) {
                for (int i = shard.firstTensor(); i < shard.firstTensor() + shard.tensorCount(); i++) {
                    splitOf.put(tensors.get(i).name(), shard.index());
                }
            }
            return Collections.unmodifiableMap(splitOf);
        }

        /** The {@code length} bytes of a tensor starting {@code byteOff} bytes into it, mapped or read. */
        ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
            int s = 0;
            if (splitOf != null) {
                Integer split = splitOf.get(info.name());
                if (split == null) {
                    throw new GgufFormatException("tensor " + info.name() + " is in no split of " + description);
                }
                s = split;
            }
            long start = shards[s].fileOffset(info.offset()) + byteOff;
            FileChannel ch = channels[s];
            if (start < 0 || length < 0 || length > Integer.MAX_VALUE || start > ch.size() - length) {
                throw new GgufFormatException("tensor " + info.name() + " bytes [" + start + ", +" + length
                        + ") lie outside its file of " + ch.size() + " bytes");
            }
            if (mode == TensorReadMode.MMAP) {
                return ch.map(FileChannel.MapMode.READ_ONLY, start, length);
            }
            return read(ch, info, start, (int) length);
        }

        /** Reads {@code length} bytes at {@code start} into a new direct buffer, positioned at 0. */
        private ByteBuffer read(FileChannel ch, GGUFTensorInfo info, long start, int length) throws IOException {
            ByteBuffer buf;
            try {
                buf = ByteBuffer.allocateDirect(length);
            } catch (OutOfMemoryError e) {
                throw new IOException("cannot allocate " + length + " bytes of direct memory for tensor "
                        + info.name() + " with " + TensorReadMode.PROP_DISABLE_MMAP
                        + "=true; raise -XX:MaxDirectMemorySize or use mmap", e);
            }
            while (buf.hasRemaining()) {
                if (ch.read(buf, start + buf.position()) < 0) {
                    throw new EOFException("tensor " + info.name() + " ends at byte " + (start + buf.position())
                            + " of " + description + ", before its " + length + " bytes");
                }
            }
            buf.flip();
            return buf;
        }

        @Override
        public void close() throws IOException {
            IOException first = null;
            for (FileChannel ch : channels) {
                try {
                    if (ch != null) {
                        ch.close();
                    }
                } catch (IOException e) {
                    if (first == null) {
                        first = e;
                    } else {
                        first.addSuppressed(e);
                    }
                }
            }
            if (first != null) {
                throw first;
            }
        }

        private static void closeAll(FileChannel[] channels, Exception cause) {
            for (FileChannel ch : channels) {
                if (ch != null) {
                    try {
                        ch.close();
                    } catch (IOException e) {
                        cause.addSuppressed(e);
                    }
                }
            }
        }

        @Override
        public String toString() {
            return description;
        }
    }
}
