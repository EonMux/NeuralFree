package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.quant.RowQuantizers;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile.Read;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The quantize driver ({@code llama_model_quantize_impl}, llama.cpp src/llama-quant.cpp:993-1439): it
 * opens the input GGUF, plans every kept tensor's target type through {@link QuantizeTypes}, applies the
 * importance matrix, and writes the output through {@link GgufWriter} over a {@link QuantizedSource} —
 * one streamed pass, no tensor in the heap whole. {@code --keep-split} writes one output file per input
 * shard with the reference's shard keys; {@code --prune-layers} renumbers the surviving blocks and remaps
 * the imatrix names with them; {@code --dry-run} only prints the sizes. The metadata of the output is the
 * input's with {@code general.quantization_version} and {@code general.file_type} set, the {@code
 * split.*} keys removed, the {@code --override-kv} entries and the {@code quantize.imatrix.*} keys
 * applied.
 */
public final class QuantizeTool {

    /** {@code general.quantization_version} of a quantized output ({@code GGML_QNT_VERSION}). */
    public static final int QNT_VERSION = 2;

    private static final String KEY_IMATRIX_FILE = "quantize.imatrix.file";
    private static final String KEY_IMATRIX_DATASET = "quantize.imatrix.dataset";
    private static final String KEY_IMATRIX_ENTRIES = "quantize.imatrix.entries_count";
    private static final String KEY_IMATRIX_CHUNKS = "quantize.imatrix.chunks_count";

    private static final Pattern BLOCK = Pattern.compile("blk\\.(\\d+)\\.");

    private QuantizeTool() {
    }
    /** What one run produced. */
    public static final class Result {
        /** The files written, in order; empty for a {@code --dry-run}. */
        public final List<Path> files;
        /** The stored bytes of the input's kept tensors. */
        public final long totalSizeOrg;
        /** The quantized bytes of the output. */
        public final long totalSizeNew;
        /** Tensors the shape fallback demoted. */
        public final int fallbackCount;
        /** True for a dry run whose real run needs an imatrix. */
        public final boolean imatrixRequired;

        Result(List<Path> files, long totalSizeOrg, long totalSizeNew, int fallbackCount,
               boolean imatrixRequired) {
            this.files = List.copyOf(files);
            this.totalSizeOrg = totalSizeOrg;
            this.totalSizeNew = totalSizeNew;
            this.fallbackCount = fallbackCount;
            this.imatrixRequired = imatrixRequired;
        }
    }

    /** One kept tensor of the plan. */
    private static final class Planned {
        String name;
        GGUFTensorInfo info;
        int shard;
        String imatrixName;
        GGMLType target;
        boolean requiresImatrix;
        float[] weights;
        boolean quantize;

        long newSize() {
            return quantize ? nRows(info) * rowBytes(target, (int) info.dims()[0]) : info.sizeInBytes();
        }
    }

    /** The plan of one run: the kept tensors in table order, the mixture and the prune state. */
    private static final class Plan {
        final List<Planned> tensors = new ArrayList<>();
        QuantizeTypes.Mixture mixture;
        Prune prune;
    }

    /**
     * Runs the quantization of {@code input} to {@code out} (ignored for a dry run).
     *
     * @throws IOException naming the tensor, the flag or the file for every refusal
     */
    public static Result run(Path input, Path out, QuantizeParams params) throws IOException {
        GGMLType defaultType = params.ftype.defaultType();
        if (!params.dryRun && out == null) {
            throw new IOException("quantize needs an output path (a dry run writes none)");
        }
        try (GGUFFileParser src = ModelSources.parseGguf(input.toString())) {
            Imatrix imatrix = loadImatrix(params);
            QuantizeTypes.ModelInfo model = modelInfo(src, !imatrix.data.isEmpty());
            Plan plan = plan(src, params, model, defaultType, imatrix);

            long totalOrg = 0;
            long totalNew = 0;
            long elements = 0;
            for (Planned p : plan.tensors) {
                totalOrg += p.info.sizeInBytes();
                totalNew += p.newSize();
                elements += elements(p.info);
            }
            int fallback = plan.mixture.fallbackCount();

            if (params.dryRun) {
                boolean required = report(input, plan, params, totalOrg, totalNew, elements, fallback);
                return new Result(List.of(), totalOrg, totalNew, fallback, required);
            }
            if (Files.exists(out)) {
                if (Files.isDirectory(out)) {
                    throw new IOException("output " + out + " is a directory");
                }
                if (Files.isSameFile(input.toAbsolutePath(), out.toAbsolutePath())) {
                    throw new IOException("input and output files are the same: '" + input + "'");
                }
            }
            List<Path> written = write(src, out, params, plan, imatrix);
            summary(plan.tensors, totalOrg, totalNew, elements, fallback);
            return new Result(written, totalOrg, totalNew, fallback, false);
        }
    }

    // ------------------------------------------------------------------ planning

    /** The pruned-layer rename state ({@code remap_layer} and {@code remap_imatrix}). */
    private static final class Prune {
        private final List<Integer> layers;
        private final Map<Integer, String> mapped = new TreeMap<>();
        private int nextId;

        Prune(List<Integer> layers) {
            this.layers = layers;
        }

        String remap(String name) {
            if (layers.isEmpty()) {
                return name;
            }
            Matcher m = BLOCK.matcher(name);
            if (!m.find()) {
                return name;
            }
            int blk = Integer.parseInt(m.group(1));
            if (mapped.containsKey(blk)) {
                // already mapped
            } else if (layers.contains(blk)) {
                mapped.put(blk, "");
            } else if (blk < layers.get(0)) {
                mapped.put(blk, String.valueOf(blk));
                nextId = blk + 1;
            } else {
                mapped.put(blk, String.valueOf(nextId));
                ++nextId;
            }
            String replacement = mapped.get(blk);
            if (replacement.isEmpty()) {
                return "";
            }
            return new StringBuilder(name).replace(m.start(1), m.end(1), replacement).toString();
        }

        /** The original block number behind {@code newName}'s block ({@code remap_imatrix}). */
        String original(String newName) {
            if (mapped.isEmpty()) {
                return newName;
            }
            Matcher m = BLOCK.matcher(newName);
            if (!m.find()) {
                return newName;
            }
            String blk = m.group(1);
            for (Map.Entry<Integer, String> e : mapped.entrySet()) {
                if (e.getValue().equals(blk)) {
                    return new StringBuilder(newName).replace(m.start(1), m.end(1),
                            String.valueOf(e.getKey())).toString();
                }
            }
            throw new IllegalStateException("imatrix mapping error for " + newName);
        }

        int blocks() {
            int n = 0;
            for (String v : mapped.values()) {
                if (!v.isEmpty()) {
                    n++;
                }
            }
            return n;
        }
    }

    /** The collected importance matrix, normalized and filtered ({@code prepare_imatrix}). */
    private static final class Imatrix {
        final Map<String, float[]> data = new TreeMap<>();
        List<String> datasets = List.of();
        int chunkCount;
    }

    private static Imatrix loadImatrix(QuantizeParams params) throws IOException {
        Imatrix imatrix = new Imatrix();
        if (params.imatrixFile != null) {
            Read read;
            try {
                read = ImatrixFile.read(params.imatrixFile);
            } catch (IOException e) {
                throw new IOException("failed to load imatrix from '" + params.imatrixFile + "': "
                        + e.getMessage(), e);
            }
            if (!read.isLegacy() && !read.hasMetadata()) {
                throw new IOException("missing imatrix metadata in file " + params.imatrixFile);
            }
            for (ImatrixFile.Entry e : read.entries()) {
                float[] sums = e.sumSq();
                float[] out = new float[sums.length];
                if (!read.isLegacy()) {
                    // GGUF format: normalize by per-expert counts
                    int ncounts = e.nMat();
                    int ne0 = sums.length / ncounts;
                    for (int j = 0; j < ncounts; ++j) {
                        float count = e.count(j);
                        for (int i = 0; i < ne0; ++i) {
                            out[j * ne0 + i] = count > 0.0f ? sums[j * ne0 + i] / count : 1.0f;
                        }
                    }
                } else {
                    // Legacy format: sums contain (raw/count)*ncall, divide by ncall
                    long ncall = e.counts().length > 0 ? e.counts()[0] : 0;
                    for (int i = 0; i < sums.length; ++i) {
                        out[i] = ncall > 0 ? sums[i] / ncall : sums[i];
                    }
                }
                imatrix.data.put(e.tensor(), out);
            }
            imatrix.datasets = read.datasets();
            imatrix.chunkCount = read.chunkCount();
        }
        for (String name : params.excludeWeights) {
            imatrix.data.keySet().removeIf(n -> n.contains(name));
        }
        if (!params.includeWeights.isEmpty()) {
            Map<String, float[]> kept = new TreeMap<>();
            for (String name : params.includeWeights) {
                for (Map.Entry<String, float[]> e : imatrix.data.entrySet()) {
                    if (e.getKey().contains(name)) {
                        kept.put(e.getKey(), e.getValue());
                    }
                }
            }
            imatrix.data.clear();
            imatrix.data.putAll(kept);
        }
        for (Map.Entry<String, float[]> e : imatrix.data.entrySet()) {
            for (float f : e.getValue()) {
                if (!Float.isFinite(f)) {
                    throw new IOException("imatrix contains non-finite value " + f + " in " + e.getKey());
                }
            }
        }
        return imatrix;
    }

    /**
     * The model facts the policy reads from the metadata: {@code <arch>.attention.head_count} and
     * {@code head_count_kv} (layer 0's entry when the file stores one per layer, as the reference's
     * {@code n_gqa()} reads them), the block and expert counts.
     */
    private static QuantizeTypes.ModelInfo modelInfo(GGUFFileParser src, boolean hasImatrix) {
        String arch = src.getString(GgufKeys.GENERAL_ARCHITECTURE);
        String a = arch == null ? "" : arch;
        Integer head = layerZeroInt(src, a, GgufKeys.ATTENTION_HEAD_COUNT);
        Integer headKv = layerZeroInt(src, a, GgufKeys.ATTENTION_HEAD_COUNT_KV);
        Integer layer = intOf(src, key(a, "block_count"));
        if (layer == null) {
            layer = intOf(src, key(a, "n_layer"));
        }
        Integer expert = intOf(src, key(a, "expert_count"));
        return QuantizeTypes.ModelInfo.of(arch, layer != null ? layer : 0, head != null ? head : 0,
                headKv, expert != null ? expert : 0, hasImatrix);
    }

    private static String key(String arch, String suffix) {
        return arch.isEmpty() ? "" : arch + "." + suffix;
    }

    private static Integer intOf(GGUFFileParser src, String key) {
        return key.isEmpty() ? null : src.getInt(key);
    }

    /** The scalar value of the per-architecture {@code format} key, or the first entry of its array. */
    private static Integer layerZeroInt(GGUFFileParser src, String arch, String format) {
        if (arch.isEmpty()) {
            return null;
        }
        String key = String.format(Locale.ROOT, format, arch);
        Integer scalar = src.getInt(key);
        if (scalar != null) {
            return scalar;
        }
        long[] perLayer = src.getValueType(key) == GGUFValueType.ARRAY ? src.getUInt32Array(key) : null;
        return perLayer != null && perLayer.length > 0 ? (int) perLayer[0] : null;
    }

    /** The target types of every kept tensor, decided in table order. */
    private static Plan plan(GGUFFileParser src, QuantizeParams params, QuantizeTypes.ModelInfo model,
                             GGMLType defaultType, Imatrix imatrix) throws IOException {
        Prune prune = new Prune(params.pruneLayers);
        List<String> names = new ArrayList<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            String renamed = prune.remap(t.name());
            if (!renamed.isEmpty()) {
                names.add(renamed);
            }
        }
        QuantizeTypes.Mixture mixture = new QuantizeTypes.Mixture(params, model, names);

        Map<Integer, Integer> shardOf = new TreeMap<>();
        List<GGUFFileParser.Shard> shards = src.getShards();
        for (int s = 0; s < shards.size(); s++) {
            GGUFFileParser.Shard shard = shards.get(s);
            for (int i = 0; i < shard.tensorCount(); i++) {
                shardOf.put(shard.firstTensor() + i, s);
            }
        }

        Plan plan = new Plan();
        plan.mixture = mixture;
        plan.prune = prune;
        int tableIndex = 0;
        for (GGUFTensorInfo t : src.getTensors()) {
            Planned p = new Planned();
            p.name = prune.remap(t.name());
            p.info = t;
            p.shard = shardOf.getOrDefault(tableIndex, 0);
            tableIndex++;
            if (p.name.isEmpty()) {
                continue;
            }
            p.imatrixName = imatrix.data.isEmpty() ? p.name : prune.original(p.name);
            p.target = mixture.target(p.name, t.dims(), t.type(), defaultType);
            p.requiresImatrix = QuantizeTypes.requiresImatrix(p.name, p.target, params.ftype);
            p.quantize = t.type() != p.target;
            if (p.quantize && !RowQuantizers.supports(p.target)) {
                throw new IOException("tensor '" + p.name + "' targets " + typeName(p.target)
                        + ", which the quantize tool has no row encoder for");
            }

            if (params.imatrixFile != null) {
                p.weights = weightsFor(p, t, imatrix);
                if (!params.dryRun && p.quantize && p.weights == null && p.requiresImatrix) {
                    throw new IOException("Missing importance matrix for tensor " + p.name
                            + " in a very low-bit quantization");
                }
            } else if (p.quantize && p.requiresImatrix && !params.dryRun) {
                throw new IOException("this quantization requires an importance matrix!\n"
                        + "        - offending tensor: " + p.name + "\n"
                        + "        - target type: " + typeName(p.target));
            }
            if (!params.dryRun && p.quantize && p.weights == null && RowQuantizers.requiresWeights(p.target)) {
                throw new IOException("tensor '" + p.name + "' targets " + typeName(p.target)
                        + ", whose quantizer needs importance weights: pass --imatrix with an entry for"
                        + " this tensor, or choose another type for it");
            }
            if (!params.dryRun && p.quantize && t.type().isQuantized() && !params.allowRequantize) {
                throw new IOException("requantizing from type " + typeName(t.type()) + " is disabled");
            }
            plan.tensors.add(p);
        }
        return plan;
    }

    /**
     * The importance slice of one planned tensor, or null: the entry of its remapped name, checked
     * against the expert layout ({@code ne0 * ne2 * ne3}) the way the reference checks it.
     */
    private static float[] weightsFor(Planned p, GGUFTensorInfo t, Imatrix imatrix) throws IOException {
        float[] w = imatrix.data.get(p.imatrixName);
        if (w == null) {
            return null;
        }
        long[] ne = t.dims();
        long ne2 = ne.length >= 3 ? ne[2] : 1;
        long ne3 = ne.length >= 4 ? ne[3] : 1;
        long expected = ne[0] * ne2 * ne3;
        if (w.length != expected) {
            if (p.name.equals("token_embd.weight") || p.name.equals("per_layer_token_embd.weight")) {
                return null;
            }
            throw new IOException("imatrix size " + w.length + " is different from tensor size " + expected
                    + " for " + p.name);
        }
        return w;
    }

    // ------------------------------------------------------------------ writing

    private static List<Path> write(GGUFFileParser src, Path out, QuantizeParams params, Plan plan,
                                    Imatrix imatrix) throws IOException {
        try (WeightStore input = src.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            int alignment = (int) src.getAlignment();
            GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE, true, alignment);
            List<Path> written;
            if (params.keepSplit && src.getShards().size() > 1) {
                written = writeSplit(src, out, params, plan, imatrix, input, alignment, options);
            } else {
                QuantizedSource source = singleSource(src, out.toString(), params, plan, imatrix, input,
                        alignment);
                written = GgufWriter.write(source, out, options);
            }
            progress(plan.tensors);
            return written;
        }
    }

    /** The single-file output source over the whole plan. */
    private static QuantizedSource singleSource(GGUFFileParser src, String filePath, QuantizeParams params,
                                                Plan plan, Imatrix imatrix, WeightStore input,
                                                int alignment) {
        Map<String, Object> meta = new LinkedHashMap<>();
        Map<String, GGUFValueType> types = new LinkedHashMap<>();
        Map<String, GGUFValueType> arrays = new LinkedHashMap<>();
        baseMetadata(src, params, meta, types, arrays, plan.prune.blocks(), imatrix);
        return new QuantizedSource(filePath, meta, types, arrays, tensors(plan.tensors, alignment), input,
                params.nthread);
    }

    /**
     * One output file per input shard ({@link GgufWriter#writeSplitFile}): the kept tensors of the shard in file
     * order, every file with its {@code split.no}, {@code split.count} and {@code split.tensors.count}, file 0 with
     * the output metadata.
     */
    private static List<Path> writeSplit(GGUFFileParser src, Path out, QuantizeParams params, Plan plan,
                                         Imatrix imatrix, WeightStore input, int alignment,
                                         GgufWriter.Options options) throws IOException {
        List<GGUFFileParser.Shard> shards = src.getShards();
        int n = shards.size();
        String prefix = stripSplitSuffix(out.toString());
        List<Path> written = new ArrayList<>(n);
        for (int s = 0; s < n; s++) {
            List<Planned> shardPlan = new ArrayList<>();
            for (Planned p : plan.tensors) {
                if (p.shard == s) {
                    shardPlan.add(p);
                }
            }
            Map<String, Object> meta = new LinkedHashMap<>();
            Map<String, GGUFValueType> types = new LinkedHashMap<>();
            Map<String, GGUFValueType> arrays = new LinkedHashMap<>();
            if (s == 0) {
                baseMetadata(src, params, meta, types, arrays, plan.prune.blocks(), imatrix);
            }
            Path shardOut = Path.of(prefix + String.format(Locale.ROOT, "-%05d-of-%05d.gguf", s + 1, n));
            QuantizedSource source = new QuantizedSource(shardOut.toString(), meta, types, arrays,
                    tensors(shardPlan, alignment), input, params.nthread);
            written.add(GgufWriter.writeSplitFile(source, shardOut, options,
                    new GgufWriter.SplitShard(s, n, plan.tensors.size())));
        }
        return written;
    }

    private static String stripSplitSuffix(String name) {
        String s = name;
        if (s.endsWith(".gguf")) {
            s = s.substring(0, s.length() - ".gguf".length());
        }
        return s.replaceAll("-\\d{5}-of-\\d{5}$", "");
    }

    /** The output tensors placed back to back at the alignment, one context per planned tensor. */
    private static List<QuantizedSource.Tensor> tensors(List<Planned> plan, int alignment) {
        List<QuantizedSource.Tensor> out = new ArrayList<>(plan.size());
        long offset = 0;
        for (Planned p : plan) {
            int nPerRow = (int) p.info.dims()[0];
            long nRows = nRows(p.info);
            long rowsPerExpert = p.info.dims().length >= 2 ? p.info.dims()[1] : 1;
            int srcBytes = rowBytes(p.info.type(), nPerRow);
            int dstBytes = rowBytes(p.target, nPerRow);
            offset = alignUp(offset, alignment);
            out.add(new QuantizedSource.Tensor(p.name, p.info.dims(), p.target, p.info.type(),
                    p.info.offset(), offset, nPerRow, nRows, rowsPerExpert, srcBytes, dstBytes, p.weights));
            offset += nRows * dstBytes;
        }
        return out;
    }

    /**
     * The metadata of the output's first file: the input's keys without the {@code split.*}, with
     * {@code general.quantization_version} and {@code general.file_type} set, the prune
     * {@code block_count}, the {@code --override-kv} entries and the {@code quantize.imatrix.*} keys.
     */
    private static void baseMetadata(GGUFFileParser src, QuantizeParams params, Map<String, Object> meta,
                                     Map<String, GGUFValueType> types, Map<String, GGUFValueType> arrays,
                                     int prunedBlocks, Imatrix imatrix) {
        for (Map.Entry<String, Object> e : src.getMetadata().entrySet()) {
            String key = e.getKey();
            if (key.equals(GgufKeys.SPLIT_NO) || key.equals(GgufKeys.SPLIT_COUNT)
                    || key.equals(GgufKeys.SPLIT_TENSORS_COUNT)) {
                continue;
            }
            meta.put(key, e.getValue());
            types.put(key, src.getValueType(key));
            if (src.getValueType(key) == GGUFValueType.ARRAY) {
                arrays.put(key, src.getArrayElementType(key));
            }
        }
        setKv(meta, types, GgufKeys.GENERAL_QUANTIZATION_VERSION, QNT_VERSION, GGUFValueType.UINT32);
        if (!params.ftype.onlyCopy() || !meta.containsKey(GgufKeys.GENERAL_FILE_TYPE)) {
            setKv(meta, types, GgufKeys.GENERAL_FILE_TYPE, params.ftype.id(), GGUFValueType.UINT32);
        }
        if (prunedBlocks >= 0 && !params.pruneLayers.isEmpty()) {
            String arch = src.getString(GgufKeys.GENERAL_ARCHITECTURE);
            if (arch != null) {
                setKv(meta, types, arch + ".block_count", prunedBlocks, GGUFValueType.UINT32);
            }
        }
        for (MetadataOverrides.Override o : params.kvOverrides) {
            if (o.type() == GGUFValueType.INT32) {
                // the reference writes int overrides as UINT32 of the absolute value
                setKv(meta, types, o.key(), Math.abs((Integer) o.value()), GGUFValueType.UINT32);
            } else {
                setKv(meta, types, o.key(), o.value(), o.type());
            }
        }
        if (!imatrix.data.isEmpty()) {
            setKv(meta, types, KEY_IMATRIX_FILE, params.imatrixFile.toString(), GGUFValueType.STRING);
            if (!imatrix.datasets.isEmpty()) {
                setKv(meta, types, KEY_IMATRIX_DATASET, imatrix.datasets.get(0), GGUFValueType.STRING);
            }
            setKv(meta, types, KEY_IMATRIX_ENTRIES, imatrix.data.size(), GGUFValueType.UINT32);
            if (imatrix.chunkCount > 0) {
                setKv(meta, types, KEY_IMATRIX_CHUNKS, imatrix.chunkCount, GGUFValueType.UINT32);
            }
        }
    }

    /** Sets one key, replacing an existing entry in place and appending a missing one. */
    private static void setKv(Map<String, Object> meta, Map<String, GGUFValueType> types, String key,
                              Object value, GGUFValueType type) {
        meta.put(key, value);
        types.put(key, type);
    }

    // ------------------------------------------------------------------ reporting

    /** The per-tensor progress lines, in the reference's format. */
    private static void progress(List<Planned> plan) {
        int idx = 0;
        for (Planned p : plan) {
            idx++;
            long[] dims = p.info.dims();
            StringBuilder shape = new StringBuilder();
            int rank = 1;
            for (int i = 0; i < dims.length; i++) {
                if (dims[i] > 1) {
                    rank = i + 1;
                }
            }
            for (int i = 0; i < rank; i++) {
                if (i > 0) {
                    shape.append(", ");
                }
                shape.append(dims[i]);
            }
            if (!p.quantize) {
                System.out.printf(Locale.ROOT, "[%4d/%4d] %-36s - [%s], type = %6s, size = %8.3f MiB%n",
                        idx, plan.size(), p.name, shape, typeName(p.info.type()),
                        p.info.sizeInBytes() / 1024.0 / 1024.0);
            } else {
                System.out.printf(Locale.ROOT,
                        "[%4d/%4d] %-36s - [%s], type = %6s, converting to %s .. size = %8.2f MiB"
                                + " -> %8.2f MiB%n",
                        idx, plan.size(), p.name, shape, typeName(p.info.type()), typeName(p.target),
                        p.info.sizeInBytes() / 1024.0 / 1024.0, p.newSize() / 1024.0 / 1024.0);
            }
        }
    }

    /** The dry-run report; returns whether a real run would need an imatrix. */
    private static boolean report(Path input, Plan plan, QuantizeParams params, long totalOrg,
                                  long totalNew, long elements, int fallback) {
        System.out.printf(Locale.ROOT, "calculating quantization size for '%s' as %s%n", input,
                params.ftype.recipeName());
        boolean required = false;
        for (Planned p : plan.tensors) {
            if (p.quantize && p.requiresImatrix) {
                required = true;
            }
        }
        progress(plan.tensors);
        System.out.printf(Locale.ROOT, "model size  = %8.2f MiB (%.2f BPW)%n", totalOrg / 1024.0 / 1024.0,
                totalOrg * 8.0 / Math.max(1, elements));
        System.out.printf(Locale.ROOT, "quant size  = %8.2f MiB (%.2f BPW)%n", totalNew / 1024.0 / 1024.0,
                totalNew * 8.0 / Math.max(1, elements));
        if (params.imatrixFile == null && required) {
            System.out.println("WARNING: dry run completed successfully, but actually completing this"
                    + " quantization will require an imatrix!");
        }
        warnFallback(plan.tensors, fallback);
        return required;
    }

    /** The wet-run summary lines. */
    private static void summary(List<Planned> plan, long totalOrg, long totalNew, long elements,
                                int fallback) {
        System.out.printf(Locale.ROOT, "model size  = %8.2f MiB (%.2f BPW)%n", totalOrg / 1024.0 / 1024.0,
                totalOrg * 8.0 / Math.max(1, elements));
        System.out.printf(Locale.ROOT, "quant size  = %8.2f MiB (%.2f BPW)%n", totalNew / 1024.0 / 1024.0,
                totalNew * 8.0 / Math.max(1, elements));
        warnFallback(plan, fallback);
    }

    private static void warnFallback(List<Planned> plan, int fallback) {
        if (fallback > 0) {
            System.out.println("WARNING: " + fallback + " of " + plan.size()
                    + " tensor(s) required fallback quantization");
        }
    }

    private static int rowBytes(GGMLType type, int nPerRow) {
        return type.getTypeSize() * (nPerRow / type.getBlockSize());
    }

    private static long nRows(GGUFTensorInfo t) {
        long n = 1;
        for (int i = 1; i < t.dims().length; i++) {
            n *= t.dims()[i];
        }
        return n;
    }

    private static long elements(GGUFTensorInfo t) {
        long n = 1;
        for (long d : t.dims()) {
            n *= d;
        }
        return n;
    }

    private static long alignUp(long v, int alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    /** The ggml type name as the logs print it. */
    private static String typeName(GGMLType type) {
        return type.name().toLowerCase(Locale.ROOT);
    }
}
