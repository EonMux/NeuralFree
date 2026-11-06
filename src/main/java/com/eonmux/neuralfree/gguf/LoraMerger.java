package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Merges LoRA adapters into a base model's weights and writes one GGUF (llama.cpp
 * {@code tools/export-lora/export-lora.cpp}, the {@code lora_merge_ctx}). The adapters are read through
 * {@link LoraAdapterLoader} — the runtime loader, which validates the adapter metadata ({@code
 * general.type = adapter}, {@code adapter.type = lora}, the base architecture match), resolves every
 * {@code <base>.lora_a} / {@code <base>.lora_b} pair against the base model's tensor catalog and decodes
 * the factors to F32 — and every adapted tensor becomes
 *
 * <pre>   W' = W + sum_adapters userScale * (alpha != 0 ? alpha / rank : 1) * B A</pre>
 *
 * <p>computed in F32 (llama-adapter.h:53-57 for the scale, export-lora.cpp {@code merge_tensor} for the
 * product): a projection or expert delta is {@code B A} over the rank, a token-embedding delta is
 * {@code A^T B} against the GGUF factor layout, and the base tensor is decoded from its storage type —
 * quantized rows dequantize to F32 first, as the reference dequantizes. The output type of a merged
 * tensor is F32 when the base tensor is F32, F16 otherwise ({@code get_out_tensor_type}); unadapted
 * tensors stream through byte-exact. The output metadata is the base's, plus {@code general.file_type =
 * MOSTLY_F16 (1)} when at least one tensor merged, the {@code neuralfree.extension_types} marker when a
 * kept tensor carries an extension type, and {@code general.alignment} when the base carries none.</p>
 *
 * <p>Refusals, all naming their cause: a split base model ({@code split.count}), an adapter list whose
 * members adapt different tensor lists (export-lora.cpp {@code err_no_subset_adapter}), an adapter with
 * no tensor pairs, a base tensor paired in only some adapters, an activated LoRA (its delta applies only
 * from an invocation sequence in a prompt on, which a merged weight cannot express) and the loader's own
 * (shapes, missing base tensors, input transforms). The output goes to the {@code .tmp} twin of
 * {@code out} and renames atomically on success; an existing output fails the merge before the first
 * byte unless {@code overwrite}. One output file; no shard support. Not thread-safe.</p>
 */
public final class LoraMerger {

    private static final Logger LOGGER = LoggingUtil.getLogger(LoraMerger.class);

    /** {@code general.file_type} of a merged output whose tensors are mostly F16 (llama.h {@code LLAMA_FTYPE_MOSTLY_F16}). */
    public static final long FILE_TYPE_MOSTLY_F16 = 1L;

    /** Largest float window of one merge chunk, in elements (16 Mi floats, like the COMPAT export). */
    static final int CHUNK_ELEMENTS = 1 << 24;

    private LoraMerger() {
    }

    /** One adapter input of a merge: its path ({@code --lora} / {@code --lora-scaled}) and its user scale. */
    public static final class Adapter {

        private final String path;
        private final float scale;

        /**
         * An adapter input at the given user scale.
         *
         * @throws IllegalArgumentException when {@code path} is null or empty or {@code scale} is not finite
         */
        public Adapter(String path, float scale) {
            if (path == null || path.isEmpty()) {
                throw new IllegalArgumentException("an adapter needs a file path");
            }
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("adapter '" + path + "' has the non-finite scale " + scale);
            }
            this.path = path;
            this.scale = scale;
        }

        /** The adapter file: a GGUF adapter file or a PEFT adapter directory. */
        public String path() {
            return path;
        }

        /** The user scale of the adapter's deltas; the {@code alpha / rank} factor applies on top. */
        public float scale() {
            return scale;
        }

        @Override
        public String toString() {
            return "Adapter{" + path + ", scale " + scale + "}";
        }
    }

    /**
     * Merges {@code adapters} into the base model {@code base} and writes the result to {@code out}.
     *
     * @param options the source options of the adapter opens ({@code -Dneuralfree.*}); a GGUF base with a
     *                PEFT adapter directory needs {@link ModelSourceOptions#loraBaseConfig()} when the
     *                adapter names no readable base configuration itself
     * @return {@code out}
     * @throws IOException naming the base, an adapter or a tensor for every refusal above, and the file
     *                     system for open and write problems
     */
    public static Path merge(ModelSource base, List<Adapter> adapters, Path out, ModelSourceOptions options,
                             boolean overwrite) throws IOException {
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(adapters, "adapters");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(options, "options");
        if (adapters.isEmpty()) {
            throw new IllegalArgumentException("no adapters to merge: pass at least one adapter");
        }
        refuseSplit(base);
        int alignment = GgufWriter.effectiveAlignment(base, GGUFFileParser.DEFAULT_ALIGNMENT);
        try (WeightStore store = base.createMappedStore(StoreUse.RESIDENT_UPLOAD);
             CpuWeights weights = CpuWeights.open(base)) {
            List<Map<String, LoraAdapter.Delta>> perAdapter = loadAdapters(adapters, weights, options);
            List<GGUFTensorInfo> srcTensors = List.copyOf(base.getTensors());
            Plan plan = plan(base, srcTensors, perAdapter);
            List<GgufWriter.KvPair> kvs = metadata(base, plan, alignment);
            GgufTensorSink sink = GgufTensorSink.singleFile(out, overwrite);
            try {
                checkTarget(out, overwrite);
                try (sink) {
                    write(sink, base, store, plan, kvs, alignment);
                }
                LOGGER.info("LoRA merge of '{}': merged {} of {} tensors with {} adapter(s), wrote '{}'",
                        base.getFilePath(), plan.merged, srcTensors.size(), perAdapter.size(), sink.file(0));
                return out;
            } finally {
                sink.close();
            }
        }
    }

    // ------------------------------------------------------------------ inputs

    /** Refuses a base of a split: llama.cpp's export-lora takes single-file models only. */
    private static void refuseSplit(ModelSource base) throws IOException {
        if (base.getMetadata().containsKey(GgufKeys.SPLIT_COUNT)) {
            throw new IOException("base model '" + base.getFilePath() + "' is a split (" + GgufKeys.SPLIT_COUNT
                    + " is present); a split model is not yet supported; merge into a single-file model");
        }
    }

    /**
     * Loads every adapter through {@link LoraAdapterLoader} and binds each pair to its adapter's user scale
     * ({@link LoraAdapter#delta}); the returned maps are keyed by the base tensor's stored name.
     *
     * @throws IOException                   when an adapter file cannot be read or carries no tensor pairs
     * @throws UnsupportedOperationException for an activated LoRA
     */
    private static List<Map<String, LoraAdapter.Delta>> loadAdapters(List<Adapter> adapters, CpuWeights base,
                                                                     ModelSourceOptions options)
            throws IOException {
        List<Map<String, LoraAdapter.Delta>> loaded = new ArrayList<>(adapters.size());
        for (Adapter a : adapters) {
            LoraAdapter adapter = LoraAdapterLoader.load(a.path(), base, options);
            if (adapter.isActivated()) {
                throw new UnsupportedOperationException("adapter '" + a.path() + "' is an activated LoRA ("
                        + GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS + "): its delta applies only from the invocation"
                        + " sequence in a prompt on, which a merged weight cannot express; serve it with --lora");
            }
            if (adapter.targets().isEmpty()) {
                throw new IOException("adapter '" + a.path() + "' carries no .lora_a / .lora_b tensor pairs;"
                        + " nothing to merge");
            }
            Map<String, LoraAdapter.Delta> pairs = new LinkedHashMap<>();
            for (String target : adapter.targets()) {
                // The loader resolved every target or it would have thrown; info() gives the stored name.
                String stored = base.catalog().info(LoraAdapterLoader.resolve(base.catalog(), target)).name();
                pairs.put(stored, adapter.delta(target, a.scale()));
            }
            loaded.add(pairs);
        }
        if (loaded.size() > 1) {
            Set<String> first = loaded.get(0).keySet();
            for (int i = 1; i < loaded.size(); i++) {
                Set<String> others = loaded.get(i).keySet();
                if (others.size() != first.size() || !first.containsAll(others)) {
                    throw new IOException("input adapters do not have the same list of tensors. This is not yet"
                            + " supported. Please merge the adapter one-by-one instead of merging all at once.");
                }
            }
        }
        return loaded;
    }

    // ------------------------------------------------------------------ the merge plan

    /** One planned tensor: the source info, the output info and its per-adapter deltas (null for a copy). */
    private static final class Planned {
        final GGUFTensorInfo src;
        final GGUFTensorInfo out;
        final List<LoraAdapter.Delta> deltas;

        Planned(GGUFTensorInfo src, GGUFTensorInfo out, List<LoraAdapter.Delta> deltas) {
            this.src = src;
            this.out = out;
            this.deltas = deltas;
        }
    }

    /** The whole merge: one planned tensor per base tensor, in file order, and the merged count. */
    private static final class Plan {
        final List<Planned> tensors;
        final int merged;

        Plan(List<Planned> tensors, int merged) {
            this.tensors = tensors;
            this.merged = merged;
        }
    }

    /**
     * Plans every base tensor: paired in every adapter means merged (F32 out for an F32 base tensor, F16
     * otherwise), paired in none means a byte-exact copy, and anything between is the named error of
     * export-lora.cpp. A multi-adapter subset was refused before the plan; this is the single-adapter guard.
     */
    private static Plan plan(ModelSource base, List<GGUFTensorInfo> srcTensors,
                             List<Map<String, LoraAdapter.Delta>> perAdapter) throws IOException {
        if (base.getTensorCount() != srcTensors.size()) {
            throw new IOException("source '" + base.getFilePath() + "' declares " + base.getTensorCount()
                    + " tensors but its table holds " + srcTensors.size()
                    + "; the count must match for a single-file merge");
        }
        List<Planned> tensors = new ArrayList<>(srcTensors.size());
        int merged = 0;
        for (GGUFTensorInfo t : srcTensors) {
            List<LoraAdapter.Delta> deltas = null;
            boolean any = false;
            boolean all = true;
            for (Map<String, LoraAdapter.Delta> pairs : perAdapter) {
                LoraAdapter.Delta d = pairs.get(t.name());
                any |= d != null;
                all &= d != null;
            }
            if (all) {
                deltas = new ArrayList<>(perAdapter.size());
                for (Map<String, LoraAdapter.Delta> pairs : perAdapter) {
                    deltas.add(pairs.get(t.name()));
                }
                merged++;
            } else if (any) {
                throw new IOException("tensor '" + t.name() + "' missing either lora_a or lora_b");
            }
            GGMLType type = deltas == null ? t.type() : outTypeOf(t.type());
            tensors.add(new Planned(t, new GGUFTensorInfo(t.name(), t.dims(), type, t.offset()), deltas));
        }
        return new Plan(tensors, merged);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The output metadata: the base's keys in order with their GGUF types — {@code general.file_type}
     * replaced (or appended) as UINT32 {@link #FILE_TYPE_MOSTLY_F16} when at least one tensor merged — plus
     * the {@code neuralfree.extension_types} marker when a kept tensor carries an extension type and the
     * base carries no marker, plus {@code general.alignment} when the base carries none.
     */
    private static List<GgufWriter.KvPair> metadata(ModelSource base, Plan plan, int alignment) throws IOException {
        Map<String, Object> md = base.getMetadata();
        boolean extensionKept = false;
        for (Planned p : plan.tensors) {
            extensionKept |= p.out.type().isExtension();
        }
        List<GgufWriter.KvPair> kvs = new ArrayList<>(md.size() + 3);
        boolean fileType = false;
        for (Map.Entry<String, Object> e : md.entrySet()) {
            String key = e.getKey();
            GGUFValueType type = base.getValueType(key);
            Object value = e.getValue();
            if (GgufKeys.GENERAL_FILE_TYPE.equals(key)) {
                // export-lora.cpp: the output is forced to F16 for now, the key stays at its position.
                type = GGUFValueType.UINT32;
                value = plan.merged > 0 ? (int) FILE_TYPE_MOSTLY_F16 : value;
                fileType = true;
            }
            if (type == null) {
                throw new IOException("metadata key '" + key + "' of source '" + base.getFilePath()
                        + "' has no GGUF value type");
            }
            GGUFValueType elem = type == GGUFValueType.ARRAY ? base.getArrayElementType(key) : null;
            kvs.add(new GgufWriter.KvPair(key, type, elem, value));
        }
        if (!fileType && plan.merged > 0) {
            kvs.add(new GgufWriter.KvPair(GgufKeys.GENERAL_FILE_TYPE, GGUFValueType.UINT32, null,
                    (int) FILE_TYPE_MOSTLY_F16));
        }
        if (extensionKept && !md.containsKey(GgufKeys.NF_EXTENSION_TYPES)) {
            kvs.add(new GgufWriter.KvPair(GgufKeys.NF_EXTENSION_TYPES, GGUFValueType.BOOL, null, Boolean.TRUE));
        }
        if (!md.containsKey(GgufKeys.GENERAL_ALIGNMENT)) {
            kvs.add(new GgufWriter.KvPair(GgufKeys.GENERAL_ALIGNMENT, GGUFValueType.UINT32, null, alignment));
        }
        return kvs;
    }

    // ------------------------------------------------------------------ the write

    /** Fails before the first byte when {@code out} exists and may not be replaced. */
    private static void checkTarget(Path out, boolean overwrite) throws IOException {
        if (!overwrite && Files.exists(out)) {
            throw new IOException("file '" + out + "' already exists; write with overwrite enabled to replace it");
        }
    }

    /** Writes the whole file: header, key/values, tensor table, then the copied and merged data in order. */
    private static void write(GgufTensorSink sink, ModelSource base, WeightStore store, Plan plan,
                              List<GgufWriter.KvPair> kvs, int alignment) throws IOException {
        GgufEncoder enc = sink.begin(0);
        String prefix = "LoRA merge of '" + base.getFilePath() + "' to '" + sink.file(0) + "'";
        enc.writeHeader(sink.channel(0), plan.tensors.size(), kvs.size());
        for (GgufWriter.KvPair kv : kvs) {
            enc.writeKv(kv.key, kv.type, kv.elementType, kv.value);
        }
        long offset = 0;
        for (Planned p : plan.tensors) {
            GGUFTensorInfo placed = new GGUFTensorInfo(p.out.name(), p.out.dims(), p.out.type(), offset);
            GGUFTensorInfo.validate(prefix, placed);
            enc.writeTensorInfo(placed, offset);
            offset = GgufWriter.alignUp(offset + placed.sizeInBytes(), alignment);
        }
        enc.align(alignment);
        for (Planned p : plan.tensors) {
            if (p.deltas == null) {
                GgufWriter.copyStreamed(p.src, store, enc);
            } else {
                mergeStreamed(p.src, store, p.deltas, enc);
            }
            enc.align(alignment);
        }
        enc.flush();
        sink.end(0);
    }

    /**
     * Streams one merged tensor: decodes whole rows from the store to F32, adds every adapter's scaled
     * {@code B A} delta in adapter order, and writes the rows in the output type (F32 or F16).
     */
    private static void mergeStreamed(GGUFTensorInfo src, WeightStore store, List<LoraAdapter.Delta> deltas,
                                      GgufEncoder enc) throws IOException {
        long ne0 = src.dims()[0];
        if (ne0 > CHUNK_ELEMENTS) {
            throw new IOException("tensor '" + src.name() + "' has a first dimension of " + ne0
                    + " elements, more than one merge chunk of " + CHUNK_ELEMENTS + " holds");
        }
        long rows = src.dims().length > 1 ? src.dims()[1] : 1;
        long experts = src.dims().length > 2 ? src.dims()[2] : 1;
        GGMLType srcType = src.type();
        GGMLType outType = outTypeOf(srcType);
        int elementBytes = outType == GGMLType.F32 ? 4 : 2;
        BlockLayout layout = BlockLayout.of(srcType);
        int blockElements = layout.blockElements();
        if (ne0 % blockElements != 0) {
            throw new IOException("tensor '" + src.name() + "' of " + srcType + " has rows of " + ne0
                    + " elements, not a multiple of the " + blockElements + "-element block size;"
                    + " the merge cannot decode whole rows");
        }
        long bytesPerRow = layout.bytesForElements(ne0);
        int rowLen = (int) ne0;
        long totalRows = Math.multiplyExact(rows, experts);
        int rowsPerChunk = (int) Math.max(1, Math.min(totalRows, CHUNK_ELEMENTS / ne0));
        byte[] raw = new byte[Math.toIntExact(Math.multiplyExact(rowsPerChunk, bytesPerRow))];
        float[] x = new float[rowsPerChunk * rowLen];
        float[] acc = new float[rowLen];
        byte[] outBytes = new byte[Math.multiplyExact(rowsPerChunk * rowLen, elementBytes)];
        long done = 0;
        while (done < totalRows) {
            int n = (int) Math.min(rowsPerChunk, totalRows - done);
            int bytes = Math.toIntExact(Math.multiplyExact(n, bytesPerRow));
            store.readTensor(src.offset() + Math.multiplyExact(done, bytesPerRow), bytes, raw, 0);
            CpuBlockDecoders.decode(srcType, ByteBuffer.wrap(raw), 0L, x, 0, n * (int) (ne0 / blockElements));
            for (LoraAdapter.Delta d : deltas) {
                applyDelta(x, n, done, d, rowLen, acc);
            }
            int values = n * rowLen;
            if (outType == GGMLType.F32) {
                for (int i = 0; i < values; i++) {
                    putInt(outBytes, 4 * i, Float.floatToRawIntBits(x[i]));
                }
            } else {
                for (int i = 0; i < values; i++) {
                    putShort(outBytes, 2 * i, FpBlockDecoders.fp32ToFp16(x[i]));
                }
            }
            enc.writeData(outBytes, 0, values * elementBytes);
            done += n;
        }
    }

    /** The merged output type of a base tensor type: F32 stays F32, everything else becomes F16. */
    private static GGMLType outTypeOf(GGMLType srcType) {
        return srcType == GGMLType.F32 ? GGMLType.F32 : GGMLType.F16;
    }

    /**
     * Adds the delta of one pair to rows {@code [rowBase, rowBase + rows)} of the decoded F32 window: a
     * projection or expert row accumulates {@code sum_k B[row][k] * A[k][i]} per element in {@code acc} and
     * adds it to the row — the delta's B carries the scale — while an embedding row accumulates
     * {@code sum_k A[t][k] * B[e][k]} per column.
     */
    private static void applyDelta(float[] x, int rows, long rowBase, LoraAdapter.Delta d, int rowLen,
                                   float[] acc) {
        LoraAdapter.Pair p = d.pair();
        float[] a = d.a();
        float[] sb = d.scaledB();
        int r = p.rank();
        switch (p.kind()) {
            case EMBEDDING: {
                int width = p.outDim();
                for (int l = 0; l < rows; l++) {
                    int t = (int) rowBase + l;
                    int aBase = t * r;
                    int xBase = l * rowLen;
                    for (int e = 0; e < width; e++) {
                        float sum = 0f;
                        for (int k = 0; k < r; k++) {
                            sum += a[aBase + k] * sb[e * r + k];
                        }
                        x[xBase + e] += sum;
                    }
                }
                return;
            }
            case EXPERTS: {
                int out = p.outDim();
                int in = p.inDim();
                for (int l = 0; l < rows; l++) {
                    long row = rowBase + l;
                    addProjectionRow(x, l * rowLen, acc, sb, (int) (row * r), a,
                            (int) (row / out) * r * in, r, in);
                }
                return;
            }
            default: {
                int in = p.inDim();
                for (int l = 0; l < rows; l++) {
                    int o = (int) rowBase + l;
                    addProjectionRow(x, l * rowLen, acc, sb, o * r, a, 0, r, in);
                }
            }
        }
    }

    /**
     * One projection or expert row: {@code acc[i] = sum_k B[row][k] * A[row0 + k][i]}, then
     * {@code x[xOff + i] += acc[i]}.
     */
    private static void addProjectionRow(float[] x, int xOff, float[] acc, float[] b, int bOff, float[] a,
                                         int aOff, int r, int in) {
        Arrays.fill(acc, 0, in, 0f);
        for (int k = 0; k < r; k++) {
            float bk = b[bOff + k];
            int aBase = aOff + k * in;
            for (int i = 0; i < in; i++) {
                acc[i] += bk * a[aBase + i];
            }
        }
        for (int i = 0; i < in; i++) {
            x[xOff + i] += acc[i];
        }
    }

    private static void putInt(byte[] out, int off, int v) {
        out[off] = (byte) v;
        out[off + 1] = (byte) (v >>> 8);
        out[off + 2] = (byte) (v >>> 16);
        out[off + 3] = (byte) (v >>> 24);
    }

    private static void putShort(byte[] out, int off, int v) {
        out[off] = (byte) v;
        out[off + 1] = (byte) (v >>> 8);
    }
}
