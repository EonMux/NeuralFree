package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ControlVectorSpec;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * llama.cpp control vectors (src/llama-adapter.h:17, the {@code llama_adapter_cvec} of src/llama-adapter.cpp:59-147,
 * {@code llm_graph_context::build_cvec} llama-graph.cpp:1592-1596): the runtime adjustments of one loaded engine,
 * applied to the residual stream between blocks.
 *
 * <p>A vector is a GGUF file whose tensors are named {@code direction.<il>} — 1-based {@code il}, F32, one-D, all one
 * shared {@code n_embd} — as {@code common_control_vector_load_one} reads them (llama.cpp common/common.cpp:1974-2058;
 * the {@code cvector-generator} tool writes them with the {@code controlvector} arch, its model hint and layer count,
 * tools/cvector-generator/cvector-generator.cpp:367-381). llama.cpp's mapping is normative here: the file stores the
 * row of {@code direction.il} at {@code n_embd * (il - 1)}, {@code apply()} mounts it into {@code tensors[il]} without
 * data for layer 0 (src/llama-adapter.cpp:119-140), and {@code apply_to(cur, il)} adds {@code tensors[il]} to the
 * residual stream leaving graph layer {@code il} (0-based, :64-74) — so {@code direction.1} adds after decoder layer 1,
 * and with the default layer range {@code [1, n_layer]} no layer-0 row is ever read. Several active vectors are summed,
 * each with its own scale and 1-based layer range ({@code --control-vector-scaled} and
 * {@code --control-vector-layer-range}, carried per file here where llama.cpp folds the strengths into the file data
 * and holds one global range for the whole set). Every direction row must be exactly {@code n_embd} wide, as llama.cpp's
 * {@code apply} refuses any other width; a file that ends before the model's last layer adds nothing on the layers it
 * has no row for, which is the zero the llama.cpp buffer holds there.
 *
 * <p>The runtime surface follows the one the adapters established ({@code J/gguf/engine/lora}, the
 * {@code LoraRegistry} / {@code LoraHandle} / {@code LoraListener} shape): {@link #add} makes a vector active under a
 * name at a scale over a range; {@link #setScale} changes the scale — always a new epoch, this is an engine-global
 * (static) change, also when the value returns to an earlier one; {@link #unload} removes it. Every change is delivered
 * synchronously on the engine thread the caller owns, through a {@link Listener} that mirrors
 * {@code LoraListener}, before the change returns, and a failing listener rolls the change back. The revisions carry
 * the epochs ({@link Entry#revision()}), exactly as {@code LoraHandle}'s revision does for the adapters. The identity of
 * an entry is its content digest (the {@code controlvector} metadata, the direction table and the exact
 * tensor bytes) plus its scale bits, layer range and epoch.
 *
 * <p>The application path the endpoint uses is the extras ({@link ControlVectorExtra} on the device,
 * {@link CpuControlVectorExtra} on the host): their {@code afterBlock} adds the graph layer's {@linkplain #snapshot()
 * sum} to every row of the residual stream, a token-wise add, with {@code batchRefusal()} null because the add does not
 * depend on the sequence layout. Mutable, owned by the engine thread like {@code LoraRegistry}; one set per
 * loaded model.
 */
public final class ControlVector {

    /** {@code general.architecture} of a control-vector GGUF (cvector-generator export_gguf). */
    public static final String ARCH = "controlvector";

    /** Prefix of the direction tensor names ({@code direction.<il>}, 1-based {@code il}). */
    public static final String DIRECTION_PREFIX = "direction.";

    private final int nLayer;
    private final int nEmbd;
    /** The active entries in add order — the display and event order. */
    private final Map<String, Entry> entries = new java.util.LinkedHashMap<>();
    /** Rises with every set change; the extras re-read their sum only through it. */
    private long revision;
    private final List<Listener> listeners = new ArrayList<>();

    /**
     * The empty control-vector set of a model of {@code nLayer} decoder layers and {@code nEmbd} residual width.
     *
     * @throws IllegalArgumentException when the layer count or the width is not positive
     */
    public ControlVector(int nLayer, int nEmbd) {
        if (nLayer < 1) {
            throw new IllegalArgumentException("a control-vector set needs at least one decoder layer, got " + nLayer);
        }
        if (nEmbd < 1) {
            throw new IllegalArgumentException("a control-vector set needs a positive n_embd, got " + nEmbd);
        }
        this.nLayer = nLayer;
        this.nEmbd = nEmbd;
    }

    /** The decoder layer count the set sums over — the top of llama.cpp's default layer range. */
    public int nLayer() {
        return nLayer;
    }

    /** The residual-stream width every direction must match. */
    public int nEmbd() {
        return nEmbd;
    }

    /**
     * A counter that changes on every set change: every change is a new epoch of the engine-global
     * set, also when a scale returns to an earlier value; the extras re-read their sum only on it.
     */
    public long revision() {
        return revision;
    }

    /** Registers {@code listener}; every following change reaches it before the change returns. */
    public void addListener(Listener listener) {
        if (listener != null) {
            listeners.add(listener);
        }
    }

    /** Removes {@code listener}. */
    public void removeListener(Listener listener) {
        listeners.remove(listener);
    }

    /**
     * Makes the vector at {@code path} active under {@code name}, at {@code scale}, over the 1-based layer range
     * {@code [layerStart, layerEnd]} — llama.cpp's {@code --control-vector-scaled} with
     * {@code --control-vector-layer-range}, their defaults 1 and {@link #nLayer()}.
     *
     * @throws IOException          when the file cannot be read
     * @throws IllegalArgumentException for a blank name, a non-finite scale or a range outside {@code [1, nLayer]}
     * @throws GgufFormatException  when the file is not a control vector in the llama.cpp file form
     * @throws IllegalStateException when {@code name} is already active
     */
    public Entry add(String name, String path, float scale, int layerStart, int layerEnd) throws IOException {
        return add(name, File.load(Objects.requireNonNull(path, "path")), scale, layerStart, layerEnd);
    }

    /** {@link #add(String, String, float, int, int)} over the full default layer range. */
    public Entry add(String name, String path, float scale) throws IOException {
        return add(name, path, scale, 1, nLayer);
    }

    /** {@link #add(String, String, float, int, int)} at scale 1.0 over the full default layer range. */
    public Entry add(String name, String path) throws IOException {
        return add(name, path, 1.0f, 1, nLayer);
    }

    /**
     * Makes an already-loaded vector active; the shape of the llama.cpp runtime call
     * ({@code llama_set_adapter_cvec}, src/llama-context.cpp:1508-1520) after a load.
     *
     * @throws IllegalArgumentException for a null file, a blank name, a non-finite scale or a range outside
     *                                  {@code [1, nLayer]}
     * @throws GgufFormatException      when the file's rows are not exactly {@link #nEmbd()} wide — llama.cpp refuses
     *                                  any other width at apply ("control vector n_embd does not match model")
     * @throws IllegalStateException    when {@code name} is already active
     */
    public Entry add(String name, File file, float scale, int layerStart, int layerEnd) {
        Objects.requireNonNull(file, "file");
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("a control vector needs a set name, got '" + name + "'");
        }
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("control vector " + name + " has a non-finite scale " + scale);
        }
        if (layerStart < 1 || layerEnd < layerStart || layerEnd > nLayer) {
            throw new IllegalArgumentException("control vector " + name + ": layer range [" + layerStart + ", "
                    + layerEnd + "] is outside [1, " + nLayer + "]");
        }
        if (file.nEmbd() != nEmbd) {
            String where = name.equals(file.path()) ? name : name + " (" + file.path() + ")";
            throw new GgufFormatException("control vector " + where + ": n_embd does not match model: the file's"
                    + " direction rows (layers 1.." + file.maxLayer() + ") are " + file.nEmbd() + " wide, the model's"
                    + " residual stream (layers " + layerStart + ".." + layerEnd + ") is " + nEmbd + " wide");
        }
        if (entries.containsKey(name)) {
            throw new IllegalStateException("control vector " + name + " is already active; unload it first");
        }
        Entry entry = new Entry(name, file, scale, layerStart, layerEnd);
        entries.put(name, entry);
        revision++;
        try {
            for (Listener l : new ArrayList<>(listeners)) {
                l.loaded(entry);
            }
        } catch (RuntimeException e) {
            entries.remove(name);
            throw e;
        }
        return entry;
    }

    /**
     * Changes the scale of the active vector {@code name}. An engine-global change every later step runs with, so it
     * creates a new epoch: {@link Entry#revision()} rises, also when the value returns to an earlier one,
     * and {@link #revision()} rises with it.
     *
     * @throws IllegalArgumentException for a non-finite scale or a name active nowhere
     */
    public void setScale(String name, float scale) {
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("control vector " + name + " has a non-finite scale " + scale);
        }
        Entry entry = entries.get(name);
        if (entry == null) {
            throw new IllegalArgumentException("control vector " + name + " is not active (active: "
                    + entries.keySet() + ")");
        }
        float previous = entry.scale();
        entry.setScale(scale);
        revision++;
        try {
            for (Listener l : new ArrayList<>(listeners)) {
                l.scaleChanged(entry, previous);
            }
        } catch (RuntimeException e) {
            entry.setScale(previous);
            revision--;
            throw e;
        }
    }

    /**
     * Removes the active vector {@code name}, as llama.cpp's {@code apply(null)} disables the vector
     * (src/llama-adapter.cpp:104-108): its rows leave the sum at once; nothing of it is cached, a later add of the same
     * file is a fresh in-process identity (two loads never share one).
     *
     * @throws IllegalArgumentException when {@code name} is active nowhere
     */
    public void unload(String name) {
        Entry entry = entries.remove(name);
        if (entry == null) {
            throw new IllegalArgumentException("control vector " + name + " is not active (active: "
                    + entries.keySet() + ")");
        }
        revision++;
        try {
            for (Listener l : new ArrayList<>(listeners)) {
                l.unloaded(entry);
            }
        } catch (RuntimeException e) {
            entries.put(name, entry);
            revision--;
            throw e;
        }
    }

    /** The active entries in add order. */
    public List<Entry> list() {
        return new ArrayList<>(entries.values());
    }

    /** The active entry {@code name}, or null. */
    public Entry entry(String name) {
        return entries.get(name);
    }

    /** True when nothing is active: the extras add nothing. */
    public boolean isEmpty() {
        return entries.isEmpty();
    }

    // ------------------------------------------------------------------ the load-time surface the engines use

    /**
     * The set of the configured vectors over the model {@code spec} (llama.cpp
     * {@code common/common.cpp:1483-1499}): every {@link ControlVectorSpec} file is read and becomes
     * active at its scale over one shared range, llama.cpp's {@code --control-vector-layer-range} — the configured
     * start and end, each {@code <= 0} value replaced by llama.cpp's default ({@code 1}, and the model's layer
     * count), the end clipped to the layers the model has. The entry name is the file's path.
     *
     * @throws IOException          when a file cannot be read
     * @throws GgufFormatException  when a file is not a control vector in the llama.cpp file form or its direction
     *                              rows are not exactly the model's {@code n_embd} wide
     * @throws IllegalArgumentException for a start below 1, a start above the end, or a scale that is not finite
     * @throws IllegalStateException    when one path is listed twice
     */
    public static ControlVector load(ModelSpec spec, List<ControlVectorSpec> vectors,
                                     int layerStart, int layerEnd) throws IOException {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(vectors, "vectors");
        if (vectors.isEmpty()) {
            throw new IllegalArgumentException("a control-vector set needs at least one vector");
        }
        int nLayer = spec.nLayer();
        int start = layerStart > 0 ? layerStart : 1;
        int end = layerEnd > 0 ? Math.min(layerEnd, nLayer) : nLayer;
        ControlVector set = new ControlVector(nLayer, spec.hparams().nEmbd());
        for (ControlVectorSpec v : vectors) {
            set.add(v.path(), v.path(), v.scale(), start, end);
        }
        return set;
    }

    /**
     * The spec whose trunk layers list {@link LayerSpec.Extras#CONTROL_VECTOR}, so the block registry installs the
     * set's extras on them; the NextN and encoder stacks are untouched, as llama.cpp applies the vector in the
     * decoder layers' graph only ({@code build_cvec(cur, il)}, src/models/llama.cpp:223). Layers that list the id
     * already keep it, so the method is idempotent.
     */
    public static ModelSpec applied(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        boolean anyMissing = false;
        for (LayerSpec l : spec.layers()) {
            anyMissing |= !l.hasExtra(LayerSpec.Extras.CONTROL_VECTOR);
        }
        if (!anyMissing) {
            return spec;
        }
        ModelSpec.Builder b = ModelSpec.builder(spec.hparams(), spec.tensors())
                .schedule(spec.schedule())
                .embedding(spec.embedding())
                .head(spec.head())
                .memoryKind(spec.memoryKind());
        for (LayerSpec l : spec.layers()) {
            if (l.hasExtra(LayerSpec.Extras.CONTROL_VECTOR)) {
                b.layer(l);
            } else {
                b.layer(LayerSpec.builder(l).extra(LayerSpec.Extras.CONTROL_VECTOR).build());
            }
        }
        for (LayerSpec l : spec.nextnLayers()) {
            b.nextnLayer(l);
        }
        for (LayerSpec l : spec.encoderLayers()) {
            b.encoderLayer(l);
        }
        return b.build();
    }

    // ------------------------------------------------------------------ the sum the extras add

    /** The scaled sum of the active entries for graph layer {@code il} (0-based), or a zero-length array when none. */
    public float[] sum(int il) {
        return snapshot().layer(il);
    }

    /**
     * The scaled sums of the whole set at {@linkplain #revision() now}: the per-graph-layer arrays the extras add.
     * The set's changes are seen only through a later snapshot. Immutable; its arrays are shared read only until the
     * next snapshot of a new revision.
     *
     * <p>The add touches graph layers {@code il} of an active entry, where {@code il} runs over the entry's 1-based
     * layer range, clipped to @{@code [1, n_layer)}: llama.cpp mounts tensor {@code il} only for {@code il < n_layer}
     * (src/llama-adapter.cpp:135), so the top layer and layer 0 get nothing.
     */
    public Snapshot snapshot() {
        int top = nLayer - 1;
        float[][] sums = new float[nLayer][];
        for (Entry e : entries.values()) {
            int last = Math.min(e.layerEnd(), top);
            for (int il = e.layerStart(); il <= last; il++) {
                float[] row = e.file().row(il);
                if (sums[il] == null) {
                    sums[il] = new float[nEmbd];
                }
                float s = e.scale();
                for (int i = 0; i < nEmbd; i++) {
                    sums[il][i] += s * row[i];
                }
            }
        }
        for (int il = 0; il < nLayer; il++) {
            if (sums[il] == null) {
                sums[il] = new float[0];
            }
        }
        return new Snapshot(revision, nLayer, nEmbd, sums);
    }

    /**
     * The per-graph-layer sums of one set revision ({@link ControlVector#snapshot()}); {@link #layer(int)} is a
     * zero-length array where no active entry adds — a layer outside every range, or one no file row covers.
     */
    public static final class Snapshot {
        private final long revision;
        private final int nLayer;
        private final int nEmbd;
        private final float[][] layers;

        Snapshot(long revision, int nLayer, int nEmbd, float[][] layers) {
            this.revision = revision;
            this.nLayer = nLayer;
            this.nEmbd = nEmbd;
            this.layers = layers;
        }

        /** The {@link ControlVector#revision()} this was built from. */
        public long revision() {
            return revision;
        }

        public int nLayer() {
            return nLayer;
        }

        public int nEmbd() {
            return nEmbd;
        }

        /**
         * The sum graph layer {@code il} gets (a zero-length array when it gets none).
         *
         * @throws IllegalArgumentException when {@code il} is outside {@code [0, nLayer)}
         */
        public float[] layer(int il) {
            if (il < 0 || il >= nLayer) {
                throw new IllegalArgumentException("layer " + il + " is outside [0, " + nLayer + ")");
            }
            return layers[il];
        }

        /** True when no layer of this snapshot gets anything. */
        public boolean isEmpty() {
            for (float[] l : layers) {
                if (l.length != 0) {
                    return false;
                }
            }
            return true;
        }
    }

    // ------------------------------------------------------------------ one active entry and one listener

    /**
     * One active vector: its set name, its loaded {@link File}, the scale the sum folds and the 1-based layer range the
     * add touches. The scale is mutated only by the owning set, on the engine thread; the epoch rises with every
     * scale change.
     */
    public static final class Entry {

        private final String name;
        private final File file;
        private final int layerStart;
        private final int layerEnd;
        private long revision;
        private float scale;

        Entry(String name, File file, float scale, int layerStart, int layerEnd) {
            this.name = name;
            this.file = file;
            this.scale = scale;
            this.layerStart = layerStart;
            this.layerEnd = layerEnd;
        }

        /** The set name of the entry (unique among the active ones). */
        public String name() {
            return name;
        }

        /** The vector's content. */
        public File file() {
            return file;
        }

        /** The path the vector was read from. */
        public String path() {
            return file.path();
        }

        /** The scale the add folds: {@code scaled sum = scale * direction}. */
        public float scale() {
            return scale;
        }

        /** The 1-based {@code direction.il} the entry's add starts at. */
        public int layerStart() {
            return layerStart;
        }

        /** The 1-based {@code direction.il} the entry's add ends at, inclusive. */
        public int layerEnd() {
            return layerEnd;
        }

        /**
         * The content digest of the vector, as 64 lowercase hex characters.
         * In-process identity is (digest, scale bits, layer range, epoch).
         */
        public String digest() {
            return file.digest();
        }

        /** The number of bytes the digest covers ({@code file.contentLength()}). */
        public long contentLength() {
            return file.contentLength();
        }

        /** The entry's epoch: 0 at load, raised by every scale change, also back to an earlier value. */
        public long revision() {
            return revision;
        }

        /** Sets the scale: a new epoch of the entry (an engine-global change every later step runs with). */
        public void setScale(float scale) {
            this.scale = scale;
            this.revision++;
        }

        @Override
        public String toString() {
            return "Entry{" + name + ", " + file.path() + ", scale=" + scale + ", layers " + layerStart + ".."
                    + layerEnd + ", rev=" + revision + "}";
        }
    }

    /**
     * The changes of the {@link ControlVector} set, delivered synchronously on the engine thread the set's caller owns
     * before the change returns — the listener events for the engine-global vector set:
     * <ul>
     *   <li>{@link #loaded}: a vector became active; the extras take the layers it now adds from their next snapshot
     *       and rewrite their buffers between steps;</li>
     *   <li>{@link #unloaded}: a vector left the set ({@link ControlVector#unload}); its rows go and nothing of it
     *       is held after the event;</li>
     *   <li>{@link #scaleChanged}: the scale of an active vector changed — a new epoch of the entry ({@link
     *       Entry#revision()} rose, also back at an earlier value) and of the set's identity with it.</li>
     * </ul>
     * A failure applies no change and fails the registry call that caused it, as {@code LoraRegistry} fails a hot
     * unload that an attachment cannot execute.
     */
    public interface Listener {

        /** A vector became active. */
        void loaded(Entry entry);

        /** A vector left the set; its adds go and nothing of it is held after the event. */
        void unloaded(Entry entry);

        /** The scale of {@code entry} changed from {@code previous} to {@code entry.scale()}; a new epoch. */
        void scaleChanged(Entry entry, float previous);
    }

    // ------------------------------------------------------------------ one loaded vector

    /**
     * One control-vector GGUF (immutable): the file's path, its shared {@code n_embd}, the file's F32 rows stored the
     * way llama.cpp stores them — {@code direction.<il>}'s row at {@code n_embd * (il - 1)}, one span of
     * {@code n_embd * maxLayer} floats, gaps zero — its content digest. The rows are shared, read only.
     */
    public static final class File {

        private final String path;
        private final int nEmbd;
        private final int maxLayer;
        private final int[] rows;
        /** The file's span, {@code nEmbd * maxLayer} floats; read only. */
        private final float[] data;
        private final Map<String, String> metadata;
        private final String digest;
        private final long contentLength;

        private File(String path, int nEmbd, int maxLayer, int[] rows, float[] data, Map<String, String> metadata,
                     String digest, long contentLength) {
            this.path = path;
            this.nEmbd = nEmbd;
            this.maxLayer = maxLayer;
            this.rows = rows;
            this.data = data;
            this.metadata = metadata;
            this.digest = digest;
            this.contentLength = contentLength;
        }

        /**
         * Loads the control-vector GGUF at {@code path} (llama.cpp {@code common_control_vector_load_one}): every
         * tensor must be a {@code direction.<il>}, F32, one-D; the layer index 0 and unparsable ones are errors; all
         * rows share one {@code n_embd}; the {@code controlvector} arch and, when the file carries it, its declared
         * layer count cross-check the tensor table.
         *
         * @throws IOException         when the file cannot be read or names no direction tensor
         * @throws GgufFormatException when the file is not a control vector in the llama.cpp file form
         */
        public static File load(String path) throws IOException {
            Objects.requireNonNull(path, "path");
            GGUFFileParser source = new GGUFFileParser(path);
            try {
                GgufKv kv = GgufKv.of(source);
                String arch = kv.archName();
                if (!ARCH.equals(arch)) {
                    throw new GgufFormatException(path + ": expect general.architecture to be '" + ARCH
                            + "', but got: " + arch);
                }
                TreeMap<Integer, GGUFTensorInfo> byLayer = new TreeMap<>();
                for (GGUFTensorInfo info : source.getTensors()) {
                    int il = directionIndex(path, info.name());
                    if (byLayer.put(il, info) != null) {
                        throw new GgufFormatException(path + ": duplicate direction tensor for layer " + il);
                    }
                }
                if (byLayer.isEmpty()) {
                    throw new IOException(path + ": no direction tensors found");
                }
                if (kv.has(GgufKeys.CONTROLVECTOR_LAYER_COUNT)) {
                    int declared = kv.i32(GgufKeys.CONTROLVECTOR_LAYER_COUNT, -1);
                    if (declared != byLayer.size()) {
                        throw new GgufFormatException(path + ": " + GgufKeys.CONTROLVECTOR_LAYER_COUNT + " is "
                                + declared + " but the tensor table holds " + byLayer.size() + " direction tensors");
                    }
                }
                int nEmbd = positiveWidth(byLayer.values().iterator().next().dim(0));
                CpuWeights weights = CpuWeights.open(source, TensorCatalog.of(source), null);
                try {
                    int maxLayer = byLayer.lastKey();
                    float[] data = new float[nEmbd * maxLayer];
                    int[] layers = new int[byLayer.size()];
                    int i = 0;
                    for (Map.Entry<Integer, GGUFTensorInfo> e : byLayer.entrySet()) {
                        GGUFTensorInfo info = e.getValue();
                        if (info.type() != GGMLType.F32) {
                            throw new GgufFormatException(path + ": invalid (non-F32) direction tensor type of '"
                                    + info.name() + "'");
                        }
                        if (info.rank() != 1) {
                            throw new GgufFormatException(path + ": invalid (non-1D) direction tensor shape of '"
                                    + info.name() + "', " + info.rank() + " dims");
                        }
                        if (info.dim(0) != nEmbd) {
                            throw new GgufFormatException(path + ": direction tensor '" + info.name() + "' does not"
                                    + " match previous dimensions, " + info.dim(0) + " vs " + nEmbd);
                        }
                        layers[i++] = e.getKey();
                        System.arraycopy(decode(weights, info), 0, data, nEmbd * (e.getKey() - 1), nEmbd);
                    }
                    Map<String, String> meta = metadata(source.getMetadata());
                    long[] length = {0L};
                    String digest = digestOf(ARCH, nEmbd, maxLayer, layers, data, meta, length);
                    return new File(path, nEmbd, maxLayer, layers, data, meta, digest, length[0]);
                } finally {
                    weights.close();
                }
            } finally {
                source.close();
            }
        }

        /**
         * An in-memory control vector, the llama.cpp runtime data form ({@code llama_set_adapter_cvec}: host rows plus
         * the layer range the extras apply them with). {@code data} holds the llama.cpp file layout — the row of
         * {@code direction.il} at {@code n_embd * (il - 1)}, gaps zero-part — and is copied.
         *
         * @throws IllegalArgumentException when the span does not cover {@code nEmbd * maxLayer} values
         */
        public static File of(String path, int nEmbd, int maxLayer, float[] data, Map<String, String> metadata) {
            Objects.requireNonNull(path, "path");
            if (nEmbd < 1 || maxLayer < 1) {
                throw new IllegalArgumentException("a control vector needs a positive n_embd and at least one layer,"
                        + " got " + nEmbd + " and " + maxLayer);
            }
            if (data == null || data.length != (long) nEmbd * maxLayer) {
                throw new IllegalArgumentException("a control-vector span of " + maxLayer + " direction rows of "
                        + nEmbd + " wide holds " + ((long) nEmbd * maxLayer) + " values, got "
                        + (data == null ? -1 : data.length));
            }
            long[] length = {0L};
            int[] layers = new int[maxLayer];
            for (int il = 1; il <= maxLayer; il++) {
                layers[il - 1] = il;
            }
            Map<String, String> meta = new TreeMap<>();
            if (metadata != null) {
                meta.putAll(metadata);
            }
            String digest = digestOf(ARCH, nEmbd, maxLayer, layers, data, meta, length);
            return new File(path, nEmbd, maxLayer, layers, data.clone(), meta, digest, length[0]);
        }

        private static int positiveWidth(long v) {
            if (v < 1 || v > Integer.MAX_VALUE) {
                throw new GgufFormatException("direction tensor width " + v + " is not a positive int");
            }
            return (int) v;
        }

        private static int directionIndex(String path, String name) {
            if (!name.startsWith(DIRECTION_PREFIX)) {
                throw new GgufFormatException(path + ": invalid/unparsable direction tensor name '" + name + "'");
            }
            int il;
            try {
                il = Integer.parseInt(name.substring(DIRECTION_PREFIX.length()));
            } catch (NumberFormatException e) {
                throw new GgufFormatException(path + ": invalid/unparsable direction tensor layer index in '" + name
                        + "'", e);
            }
            if (il <= 0) {
                throw new GgufFormatException(path + ": invalid (zero or negative) direction tensor layer index in '"
                        + name + "', got " + il);
            }
            return il;
        }

        private static float[] decode(CpuWeights weights, GGUFTensorInfo info) {
            CpuTensorRef t = weights.find(info.name());
            if (t == null) {
                throw new GgufFormatException("control-vector tensor '" + info.name() + "' has no data");
            }
            long n = t.rows() * t.cols();
            if (n > Integer.MAX_VALUE - 8L) {
                throw new GgufFormatException("control-vector tensor '" + info.name() + "' has " + n
                        + " elements, too many for one float array");
            }
            float[] out = new float[(int) n];
            t.decodeRows(0L, (int) t.rows(), out, 0);
            return out;
        }

        /** Scalar metadata as strings, in key order (arrays left out, as llama.cpp reads the file's metadata). */
        private static Map<String, String> metadata(Map<String, Object> raw) {
            Map<String, String> out = new TreeMap<>();
            for (Map.Entry<String, Object> e : raw.entrySet()) {
                Object v = e.getValue();
                if (!(v instanceof List) && !(v != null && v.getClass().isArray())) {
                    out.put(e.getKey(), String.valueOf(v));
                }
            }
            return out;
        }

        /** The digest of the canonical view metadata, the direction table and the exact tensor bytes. */
        private static String digestOf(String arch, int nEmbd, int maxLayer, int[] layers, float[] data,
                                       Map<String, String> metadata, long[] lengthOut) {
            MessageDigest sha;
            try {
                sha = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException("SHA-256 is not available", e);
            }
            byte[] tag = bytes("control-vector.n1");
            sha.update(tag);
            long length = tag.length;
            length += hash(sha, bytes(arch));
            length += hash(sha, u32(new int[]{nEmbd, maxLayer}));
            for (int il : layers) {
                length += hash(sha, u32(new int[]{il}));
            }
            for (Map.Entry<String, String> e : metadata.entrySet()) {
                length += hash(sha, bytes(e.getKey()));
                length += hash(sha, bytes(e.getValue()));
            }
            length += hash(sha, f32(data));
            lengthOut[0] = length;
            return hex(sha.digest());
        }

        /** The path the file was read from, or the caller's name for an in-memory one. */
        public String path() {
            return path;
        }

        /** The shared residual width the file's rows hold. */
        public int nEmbd() {
            return nEmbd;
        }

        /** The largest 1-based layer index the file holds a row for. */
        public int maxLayer() {
            return maxLayer;
        }

        /** The 1-based layer indexes the file's tensor table stores, ascending. */
        public int[] layers() {
            return rows.clone();
        }

        /**
         * The row of {@code direction.<il>}: a fresh copy of {@code nEmbd} floats, gaps zero (llama.cpp zero-fills the
         * loaded span), or a zero-length array when {@code il} is outside the file's span ({@code il < 1} or above
         * {@link #maxLayer()}).
         */
        public float[] row(int il) {
            if (il < 1 || il > maxLayer) {
                return new float[0];
            }
            int off = nEmbd * (il - 1);
            return java.util.Arrays.copyOfRange(data, off, off + nEmbd);
        }

        /** Scalar metadata of the file as strings (the canonical view metadata). */
        public Map<String, String> metadata() {
            return new TreeMap<>(metadata);
        }

        /** The content digest of the control vector, 64 hex characters. */
        public String digest() {
            return digest;
        }

        /** The number of bytes the digest covers. */
        public long contentLength() {
            return contentLength;
        }

        @Override
        public String toString() {
            return "File{" + path + ", n_embd=" + nEmbd + ", layers 1.." + maxLayer + "}";
        }
    }

    // ------------------------------------------------------------------ the digest and hashing helpers

    private static long hash(MessageDigest sha, byte[] values) {
        sha.update(values);
        return values.length;
    }

    private static byte[] bytes(String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        byte[] out = new byte[4 + b.length];
        out[0] = (byte) b.length;
        out[1] = (byte) (b.length >>> 8);
        out[2] = (byte) (b.length >>> 16);
        out[3] = (byte) (b.length >>> 24);
        System.arraycopy(b, 0, out, 4, b.length);
        return out;
    }

    private static byte[] u32(int[] values) {
        byte[] out = new byte[4 * values.length];
        for (int i = 0; i < values.length; i++) {
            int v = values[i];
            int o = i * 4;
            out[o] = (byte) v;
            out[o + 1] = (byte) (v >>> 8);
            out[o + 2] = (byte) (v >>> 16);
            out[o + 3] = (byte) (v >>> 24);
        }
        return out;
    }

    private static byte[] f32(float[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(values);
        return buf.array();
    }

    private static String hex(byte[] digest) {
        StringBuilder sb = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            sb.append(String.format(Locale.ROOT, "%02x", b));
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "ControlVector{" + entries.values() + "}";
    }
}
