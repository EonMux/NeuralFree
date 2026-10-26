package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * The llama.cpp imatrix file format (PARITY.md WP TL-3, ported from {@code tools/imatrix/imatrix.cpp} and
 * {@code common/imatrix-loader.*}). The collected statistics are per-tensor entries: for every weight tensor
 * whose matmul inputs were collected, the column sums of the squared activations ({@code <tensor>.in_sum2}, one
 * F32 row per matrix — per expert of a 3-D expert tensor) and the activation-row counts ({@code <tensor>.counts},
 * one F32 value per matrix). The GGUF form carries the keys {@code general.type} = {@code "imatrix"}, the
 * {@code imatrix.datasets} string array, and the {@code imatrix.chunk_count} and {@code imatrix.chunk_size}
 * counters; the legacy {@code .dat} form is llama.cpp's pre-GGUF binary layout, written and read back verbatim.
 *
 * <p>The GGUF form is written through HF.md GW-1's {@link GgufWriter}, like every GGUF output of the tools: the
 * statistics are a {@link ModelSource} in the GGUF contract (the F32 stats tensors over one backing byte array)
 * and {@link GgufWriter#write(ModelSource, Path, GgufWriter.Options)} streams the file the way it streams an
 * exported model. Reading accepts both llama.cpp's GGUF form (through {@link GGUFFileParser}) and the legacy
 * {@code .dat} form, so a matrix either tool collected combines into a run.</p>
 */
public final class ImatrixFile {

    /** {@code general.type} of a GGUF imatrix file (llama.cpp {@code save_as_gguf}). */
    public static final String TYPE_VALUE = "imatrix";
    /** {@code imatrix.datasets}: the files the matrix was collected from. */
    public static final String KEY_DATASETS = "imatrix.datasets";
    /** {@code imatrix.chunk_count}: how many chunks the matrix was computed with. */
    public static final String KEY_CHUNK_COUNT = "imatrix.chunk_count";
    /** {@code imatrix.chunk_size}: the tokens per chunk of the collection. */
    public static final String KEY_CHUNK_SIZE = "imatrix.chunk_size";

    static final String IN_SUM2_SUFFIX = ".in_sum2";
    static final String COUNTS_SUFFIX = ".counts";

    private ImatrixFile() {
    }

    // ------------------------------------------------------------------ data model

    /**
     * The collected statistics of one weight tensor: the column sums of the squared activations over each of the
     * {@code nMat} matrices ({@code 1} for a dense tensor, the expert count of a 3-D expert tensor), and the
     * activation-row count of each matrix. Immutable; a defensive copy of every array.
     */
    public static final class Entry {
        private final String tensor;
        private final int nMat;
        private final int cols;
        private final float[] sumSq;
        private final long[] counts;

        /**
         * An entry's statistics.
         *
         * @param tensor the GGUF name of the weight, without the {@code .in_sum2} suffix
         * @param sumSq  the column sums of the squared activations, {@code nMat * cols} values in matrix-major
         *               order ({@code sumSq[expert * cols + col]}); a defensive copy
         * @param counts the activation-row count of each matrix; a defensive copy
         * @throws IllegalArgumentException for a matrix count below 1, a column count below 1, a {@code sumSq} of
         *                                  another length, a count array of another length or a negative count
         */
        public Entry(String tensor, int nMat, int cols, float[] sumSq, long[] counts) {
            Objects.requireNonNull(tensor, "tensor");
            Objects.requireNonNull(sumSq, "sumSq");
            if (nMat < 1) {
                throw new IllegalArgumentException(tensor + " holds " + nMat + " matrices; at least one");
            }
            if (cols < 1) {
                throw new IllegalArgumentException(tensor + " collects over " + cols + " columns; at least one");
            }
            if (sumSq.length != (long) nMat * cols) {
                throw new IllegalArgumentException(tensor + ": " + sumSq.length + " summed values do not fill "
                        + nMat + " matrices of " + cols + " columns");
            }
            long[] c = counts == null ? new long[nMat] : counts;
            if (c.length != nMat) {
                throw new IllegalArgumentException(tensor + ": " + c.length + " counts for " + nMat + " matrices");
            }
            for (long v : c) {
                if (v < 0) {
                    throw new IllegalArgumentException(tensor + ": the activation count " + v + " is negative");
                }
            }
            this.tensor = tensor;
            this.nMat = nMat;
            this.cols = cols;
            this.sumSq = sumSq.clone();
            this.counts = c.clone();
        }

        /** The GGUF name of the weight tensor. */
        public String tensor() {
            return tensor;
        }

        /** The matrices the tensor's matmul saw ({@code 1} dense, the expert count of a 3-D expert tensor). */
        public int nMat() {
            return nMat;
        }

        /** The activation columns collected. */
        public int cols() {
            return cols;
        }

        /** The column sums of the squared activations, {@code nMat * cols} values; a copy. */
        public float[] sumSq() {
            return sumSq.clone();
        }

        /** The activation-row counts, one per matrix; a copy. */
        public long[] counts() {
            return counts.clone();
        }

        /** The activation-row count of {@code matrix}. */
        public long count(int matrix) {
            if (matrix < 0 || matrix >= nMat) {
                throw new IllegalArgumentException("matrix " + matrix + " outside [0, " + nMat + ")");
            }
            return counts[matrix];
        }

        @Override
        public String toString() {
            return tensor + "[" + nMat + "x" + cols + ", counts=" + counts[0] + (nMat > 1 ? ".." : "") + "]";
        }
    }

    // ------------------------------------------------------------------ writing

    /**
     * Writes the GGUF imatrix file {@code out} for {@code entries}, through {@link GgufWriter} as every GGUF
     * output of the tools is written. The tensors are the {@code <tensor>.in_sum2} and {@code <tensor>.counts}
     * F32 pairs in sorted name order in llama.cpp's 2-D shapes {@code [cols, nMat]} and {@code [1, nMat]}
     * ({@code save_as_gguf}), the data section one tensor pair behind the other at the default alignment.
     *
     * @param datasets   the files the matrix was collected from ({@code imatrix.datasets}); never null
     * @param chunkCount the chunk count of the collection ({@code imatrix.chunk_count})
     * @param chunkSize  the tokens per chunk ({@code imatrix.chunk_size}); {@code 0}, as llama.cpp writes for a
     *                   matrix combined from legacy files
     * @return {@code out}, written (and replaced when it exists, like llama.cpp saves)
     * @throws IOException when the file cannot be written: a tensor name beyond GGUF's 63-byte limit, or a
     *                     file-system error
     */
    public static Path writeGguf(Path out, List<Entry> entries, List<String> datasets, int chunkCount, int chunkSize)
            throws IOException {
        Objects.requireNonNull(entries, "entries");
        Objects.requireNonNull(datasets, "datasets");
        List<Entry> stats = sorted(entries);
        ImatrixSource source = new ImatrixSource(out.toString(), stats, datasets, chunkCount, chunkSize);
        try {
            GgufWriter.write(source, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        } finally {
            source.close();
        }
        return out;
    }

    private static List<Entry> sorted(List<Entry> entries) {
        Map<String, Entry> m = new TreeMap<>();
        for (Entry e : entries) {
            if (m.put(e.tensor(), e) != null) {
                throw new IllegalArgumentException("two imatrix entries for tensor '" + e.tensor() + "'");
            }
        }
        return new ArrayList<>(m.values());
    }

    private static List<String> sortedUnique(List<String> values) {
        List<String> out = new ArrayList<>();
        for (String v : new TreeSet<>(Objects.requireNonNull(values, "values"))) {
            if (v == null) {
                throw new IllegalArgumentException("the dataset list holds a null name");
            }
            out.add(v);
        }
        return out;
    }

    /**
     * Writes the legacy {@code .dat} imatrix of {@code entries} ({@code save_imatrix_legacy}), the form older
     * quantize tools read: the entry count, then per entry the length-prefixed name, the call count (the largest
     * matrix count rounded up to whole chunks), the count of summed values per entry and the values scaled back
     * to it ({@code value/count * ncall}, or {@code 1} for a matrix with no counted rows, as llama.cpp stores
     * partial data), then the trailing chunk count and dataset name (explicit prompt file, else the last loaded
     * dataset, else empty). Overwrites {@code out}.
     */
    public static Path writeLegacy(Path out, List<Entry> entries, int chunkCount, int chunkSize,
                                   List<String> datasets, String promptFile) throws IOException {
        if (chunkSize < 1) {
            throw new IllegalArgumentException("a legacy .dat write chains counts to a chunk size, got "
                    + chunkSize);
        }
        List<Entry> stats = sorted(entries);
        ByteBuffer buf = ByteBuffer.allocate(estimatedBytes(stats)).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(stats.size());
        for (Entry e : stats) {
            byte[] name = e.tensor.getBytes(StandardCharsets.UTF_8);
            buf.putInt(name.length);
            buf.put(name);
            long maxCount = 0L;
            for (long c : e.counts) {
                maxCount = Math.max(maxCount, c);
            }
            int ncall = maxCount < 1 ? 0 : (int) Math.min(Integer.MAX_VALUE,
                    maxCount / chunkSize + (maxCount % chunkSize != 0 ? 1 : 0));
            buf.putInt(ncall);
            int nval = (int) ((long) e.nMat * e.cols);
            buf.putInt(nval);
            for (int j = 0; j < nval; j++) {
                long count = e.counts[j / e.cols];
                if (count == 0L) {
                    buf.putFloat(1.0f);
                } else {
                    buf.putFloat(e.sumSq[j] / count * ncall);
                }
            }
        }
        buf.putInt(Math.max(0, chunkCount));
        String dataset = promptFile != null && !promptFile.isEmpty() ? promptFile
                : (stats.isEmpty() || datasets.isEmpty() ? "" : datasets.get(datasets.size() - 1));
        byte[] ds = dataset.getBytes(StandardCharsets.UTF_8);
        buf.putInt(ds.length);
        buf.put(ds);
        Files.write(out, Arrays.copyOf(buf.array(), buf.position()));
        return out;
    }

    private static int estimatedBytes(List<Entry> entries) {
        int n = 4 + 8;
        for (Entry e : entries) {
            n += 4 + e.tensor.getBytes(StandardCharsets.UTF_8).length + 8
                    + 4 + e.nMat * e.cols * 4
                    + 4 + Math.min(64, e.tensor.length()) + 1;
        }
        return n;
    }

    // ------------------------------------------------------------------ reading

    /**
     * Reads an imatrix file: llama.cpp's GGUF form when the first bytes are the magic, else the legacy
     * {@code .dat} form ({@code common/imatrix-loader.cpp}).
     *
     * @throws IllegalArgumentException when the file holds malformed entries (the message names the entry)
     */
    public static Read read(Path file) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException(file + " is not a readable file");
        }
        byte[] head = new byte[4];
        try (var in = Files.newInputStream(file)) {
            int n = in.readNBytes(head, 0, 4);
            if (n != 4) {
                throw new IOException(file + " holds only " + n + " bytes, no GGUF magic");
            }
        }
        return isGguf(head) ? readGguf(file) : readLegacy(file);
    }

    private static boolean isGguf(byte[] head) {
        return head[0] == 'G' && head[1] == 'G' && head[2] == 'U' && head[3] == 'F';
    }

    /**
     * Reads the GGUF form: per weight tensor its {@code .in_sum2} (F32) and its matrix counts whose element sums
     * match; every other tensor ({@code .in_sum}, {@code .stats} or an unrelated name) is skipped, as
     * {@code common_imatrix_load} skips them. The counters come from the two {@code imatrix.*} keys; missing ones
     * leave the read without metadata, and datasets when the key is absent are empty.
     */
    public static Read readGguf(Path file) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            if (!TYPE_VALUE.equals(parser.getString("general.type"))) {
                throw new IOException(file + " is a GGUF file of general.type '" + parser.getString("general.type")
                        + "', not an imatrix");
            }
            List<String> datasets = stringArray(parser);
            Long nextn = parser.getUInt32("imatrix.n_layer_nextn");
            TensorRegions regions = parser.openRegions(TensorReadMode.current());
            try {
                Map<String, float[]> sums = new TreeMap<>();
                Map<String, GGUFTensorInfo> sumInfo = new TreeMap<>();
                Map<String, GGUFTensorInfo> countInfo = new TreeMap<>();
                for (GGUFTensorInfo t : parser.getTensors()) {
                    if (t.type() != GGMLType.F32 && (t.name().endsWith(IN_SUM2_SUFFIX)
                            || t.name().endsWith(COUNTS_SUFFIX))) {
                        throw new IOException(file + ": tensor '" + t.name() + "' is " + t.type()
                                + ", but the imatrix tensors are F32");
                    }
                    if (t.name().endsWith(IN_SUM2_SUFFIX)) {
                        String name = t.name().substring(0, t.name().length() - IN_SUM2_SUFFIX.length());
                        if (sums.containsKey(name)) {
                            throw new IOException(file + ": two " + IN_SUM2_SUFFIX + " tensors for '" + name + "'");
                        }
                        sums.put(name, floats(regions, t));
                        sumInfo.put(name, t);
                    } else if (t.name().endsWith(COUNTS_SUFFIX)) {
                        String name = t.name().substring(0, t.name().length() - COUNTS_SUFFIX.length());
                        if (countInfo.containsKey(name)) {
                            throw new IOException(file + ": two " + COUNTS_SUFFIX + " tensors for '" + name + "'");
                        }
                        countInfo.put(name, t);
                    }
                }
                List<Entry> entries = new ArrayList<>();
                for (String name : sums.keySet()) {
                    GGUFTensorInfo counts = countInfo.get(name);
                    if (counts == null) {
                        throw new IOException(file + ": tensor '" + name + "' has no matching "
                                + COUNTS_SUFFIX + " tensor");
                    }
                    float[] countFloats = floats(regions, counts);
                    long[] countLongs = new long[countFloats.length];
                    for (int i = 0; i < countFloats.length; i++) {
                        countLongs[i] = Math.round(countFloats[i]);
                    }
                    int nMat = countLongs.length;
                    if (nMat < 1) {
                        throw new IOException(file + ": tensor '" + counts.name() + "' counts " + nMat
                                + " matrices");
                    }
                    float[] sumSq = sums.get(name);
                    long sumElements = elements(sumInfo.get(name).dims());
                    if (sumSq.length == 0 || sumElements != sumSq.length || sumSq.length % nMat != 0) {
                        throw new IOException(file + ": the " + sumSq.length + " summed values of '" + name
                                + "' do not fill " + nMat + " matrices");
                    }
                    int cols = sumSq.length / nMat;
                    entries.add(new Entry(name, nMat, cols, sumSq, countLongs));
                }
                Long chunkCount = parser.getUInt32(KEY_CHUNK_COUNT);
                Long chunkSize = parser.getUInt32(KEY_CHUNK_SIZE);
                boolean hasMetadata = parser.getValueType(KEY_DATASETS) == GGUFValueType.ARRAY
                        && parser.getValueType(KEY_CHUNK_COUNT) == GGUFValueType.UINT32
                        && parser.getValueType(KEY_CHUNK_SIZE) == GGUFValueType.UINT32;
                if (nextn != null && nextn != 0) {
                    // an MTP matrix is quantized against the NextN layer set; no TL-3 consumer reads it today
                    throw new IOException(file + " was collected over NextN layers (imatrix.n_layer_nextn "
                            + nextn + "); combining it is not supported");
                }
                return new Read(entries, datasets, chunkCount == null ? 0 : chunkCount.intValue(),
                        chunkSize == null ? 0 : chunkSize.intValue(), false, hasMetadata, file);
            } finally {
                regions.close();
            }
        }
    }

    /** Reads the F32 tensor {@code t} of {@code regions} as a host float array. */
    private static float[] floats(TensorRegions regions, GGUFTensorInfo t) throws IOException {
        ByteBuffer buf = regions.region(t, 0, t.sizeInBytes());
        buf = buf.order(ByteOrder.LITTLE_ENDIAN);
        if (buf.remaining() != t.sizeInBytes()) {
            throw new IOException(t.name() + " holds " + buf.remaining() + " bytes, expected " + t.sizeInBytes());
        }
        float[] out = new float[Math.toIntExact(t.sizeInBytes() / 4L)];
        buf.asFloatBuffer().get(out);
        return out;
    }

    /** The per-matrix activation counts of one read, as longs (each an F32 value, read through the regions). */
    private static float[] countsF32(TensorRegions regions, GGUFTensorInfo t) throws IOException {
        return floats(regions, t);
    }

    private static long elements(long[] dims) {
        long n = dims.length == 0 ? 1 : 1;
        for (long d : dims) {
            n *= d;
        }
        return n;
    }

    private static List<String> stringArray(GGUFFileParser parser) {
        List<Object> arr = parser.getArray(KEY_DATASETS);
        if (arr == null) {
            return List.of();
        }
        List<String> out = new ArrayList<>(arr.size());
        for (Object o : arr) {
            if (o instanceof String) {
                out.add((String) o);
            }
        }
        return out;
    }

    /**
     * Reads the legacy {@code .dat} form ({@code common_imatrix_load_legacy}): the entry count, per entry the
     * length-prefixed name, the call count, the value count and the summed values, then the trailing chunk count
     * and dataset name, both optional (an older tool may not have written them). The sums stay raw; llama.cpp
     * scales a legacy entry by the chunk size at the combination, not the read.
     */
    public static Read readLegacy(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (buf.remaining() < 4) {
            throw new IOException(file + " holds " + bytes.length + " bytes, no legacy imatrix header");
        }
        int nEntries = buf.getInt();
        if (nEntries < 1) {
            throw new IOException(file + ": no data in the legacy imatrix (" + nEntries + " entries)");
        }
        List<Entry> entries = new ArrayList<>(Math.min(nEntries, 1 << 16));
        for (int i = 0; i < nEntries; i++) {
            if (buf.remaining() < 4) {
                throw new IOException(file + ": entry " + (i + 1) + " of " + nEntries + " is missing its name");
            }
            int len = buf.getInt();
            if (len < 1 || len > buf.remaining()) {
                throw new IOException(file + ": entry " + (i + 1) + " names " + len
                        + " bytes, more than the file has left");
            }
            byte[] nameBytes = new byte[len];
            buf.get(nameBytes);
            String name = new String(nameBytes, StandardCharsets.UTF_8);
            if (buf.remaining() < 8) {
                throw new IOException(file + ": entry '" + name + "' is missing its counts");
            }
            int ncall = buf.getInt();
            int nval = buf.getInt();
            if (ncall < 0 || nval < 1 || (long) nval * 4L > buf.remaining()) {
                throw new IOException(file + ": entry '" + name + "' records " + nval
                        + " values after " + ncall + " calls, more than the file holds");
            }
            float[] vals = new float[nval];
            for (int j = 0; j < nval; j++) {
                vals[j] = buf.getFloat();
            }
            entries.add(new Entry(name, 1, nval, vals, new long[]{ncall}));
        }
        int chunkCount = 0;
        String dataset = "";
        if (buf.remaining() >= 4) {
            chunkCount = buf.getInt();
        }
        if (chunkCount < 0) {
            throw new IOException(file + ": the legacy chunk count " + chunkCount + " is negative");
        }
        if (buf.remaining() >= 4) {
            int len = buf.getInt();
            if (len == 0) {
                dataset = "";
            } else if (len > 0 && len <= buf.remaining()) {
                byte[] ds = new byte[len];
                buf.get(ds);
                dataset = new String(ds, StandardCharsets.UTF_8);
            }
        }
        return new Read(entries, dataset.isEmpty() ? List.of() : List.of(dataset), chunkCount, 0, true, false,
                file);
    }

    // ------------------------------------------------------------------ read result

    /** One read imatrix file: its entries, its dataset names and its chunk counters. */
    public static final class Read {
        private final List<Entry> entries;
        private final List<String> datasets;
        private final int chunkCount;
        private final int chunkSize;
        private final boolean legacy;
        private final boolean hasMetadata;
        private final Path file;

        Read(List<Entry> entries, List<String> datasets, int chunkCount, int chunkSize, boolean legacy,
             boolean hasMetadata, Path file) {
            this.entries = List.copyOf(entries);
            this.datasets = List.copyOf(datasets);
            this.chunkCount = chunkCount;
            this.chunkSize = chunkSize;
            this.legacy = legacy;
            this.hasMetadata = hasMetadata;
            this.file = file;
        }

        /** The imatrix entries, one per weight tensor, in name order. */
        public List<Entry> entries() {
            return entries;
        }

        /** The dataset names the file records. */
        public List<String> datasets() {
            return datasets;
        }

        /** {@code imatrix.chunk_count}; {@code 0} when the file records none (a legacy file may omit the trailer). */
        public int chunkCount() {
            return chunkCount;
        }

        /** {@code imatrix.chunk_size}; {@code 0} when the file records none. */
        public int chunkSize() {
            return chunkSize;
        }

        /** True when the file is the legacy {@code .dat} form. */
        public boolean isLegacy() {
            return legacy;
        }

        /** True when the GGUF file carries the complete metadata set. */
        public boolean hasMetadata() {
            return hasMetadata;
        }

        /** The file that was read, for messages. */
        public Path file() {
            return file;
        }

        @Override
        public String toString() {
            return file + ": " + entries.size() + " entries, " + (legacy ? "legacy .dat" : "gguf")
                    + ", chunk_count " + chunkCount + (chunkSize > 0 ? ", chunk_size " + chunkSize : "");
        }
    }

    // ------------------------------------------------------------------ the source behind the GGUF writer

    /**
     * The collected statistics as a {@link ModelSource} in the GGUF contract, written by {@link GgufWriter}: the
     * key set of llama.cpp's {@code save_as_gguf} and one F32 {@code .in_sum2}/{@code .counts} pair per entry
     * over one aligned backing byte array. The virtual offsets of the tensor table are the backing layout — the
     * writer aliases every tensor of the pair sequence in order, so the offsets the writer passes to the store
     * (its own per-shard placement of the same sequence with the same alignment) land on the same bytes.
     */
    static final class ImatrixSource implements ModelSource {
        private static final int ALIGNMENT = GGUFFileParser.DEFAULT_ALIGNMENT;

        private static long alignUp(long v, int alignment) {
            return (v + alignment - 1) & ~((long) alignment - 1);
        }

        private final String filePath;
        private final Map<String, Object> metadata;
        private final Map<String, GGUFValueType> valueTypes;
        private final Map<String, GGUFValueType> elementTypes;
        private final List<GGUFTensorInfo> tensors;
        private final byte[] data;
        private volatile boolean closed;

        ImatrixSource(String filePath, List<Entry> entries, List<String> datasets, int chunkCount, int chunkSize) {
            this.filePath = Objects.requireNonNull(filePath, "filePath");
            this.metadata = new LinkedHashMap<>();
            this.valueTypes = new LinkedHashMap<>();
            this.elementTypes = new LinkedHashMap<>();
            metadata.put("general.type", TYPE_VALUE);
            valueTypes.put("general.type", GGUFValueType.STRING);
            List<String> ds = sortedUnique(Objects.requireNonNull(datasets, "datasets"));
            if (!ds.isEmpty()) {
                metadata.put(KEY_DATASETS, ds);
                valueTypes.put(KEY_DATASETS, GGUFValueType.ARRAY);
                elementTypes.put(KEY_DATASETS, GGUFValueType.STRING);
            }
            metadata.put(KEY_CHUNK_COUNT, chunkCount);
            valueTypes.put(KEY_CHUNK_COUNT, GGUFValueType.UINT32);
            metadata.put(KEY_CHUNK_SIZE, Math.max(0, chunkSize));
            valueTypes.put(KEY_CHUNK_SIZE, GGUFValueType.UINT32);
            this.tensors = layout(entries);
            this.data = fill(entries);
        }

        /** The stats tensor pairs in sorted name order, placed back to back at the default alignment. */
        private static List<GGUFTensorInfo> layout(List<Entry> entries) {
            List<GGUFTensorInfo> out = new ArrayList<>();
            long offset = 0;
            for (Entry e : entries) {
                GGUFTensorInfo sums = new GGUFTensorInfo(e.tensor() + IN_SUM2_SUFFIX,
                        new long[]{e.cols(), e.nMat()}, GGMLType.F32, offset);
                out.add(sums);
                offset = alignUp(offset + sums.sizeInBytes(), ALIGNMENT);
                GGUFTensorInfo counts = new GGUFTensorInfo(e.tensor() + COUNTS_SUFFIX,
                        new long[]{1, e.nMat()}, GGMLType.F32, offset);
                out.add(counts);
                offset = alignUp(offset + counts.sizeInBytes(), ALIGNMENT);
            }
            return Collections.unmodifiableList(out);
        }

        /** The flat little-endian data section behind the tensor table. */
        private static byte[] fill(List<Entry> entries) {
            byte[] data = layoutCapacity(entries);
            ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            TreeMap<String, Entry> byName = new TreeMap<>();
            for (Entry e : entries) {
                byName.put(e.tensor(), e);
            }
            for (Entry e : byName.values()) {
                for (int j = 0; j < e.nMat() * e.cols(); j++) {
                    buf.putFloat(e.sumSq[j]);
                }
                for (int p = 0; p < pad(e.nMat() * (long) e.cols() * 4L); p++) {
                    buf.put((byte) 0);
                }
                for (int j = 0; j < e.nMat(); j++) {
                    buf.putFloat(e.counts[j]);
                }
                for (int p = 0; p < pad(e.nMat() * 4L); p++) {
                    buf.put((byte) 0);
                }
            }
            if (buf.remaining() != 0) {
                throw new IllegalStateException("the imatrix data section is " + buf.remaining()
                        + " bytes larger than its layout");
            }
            return data;
        }

        private static long pad(long bytes) {
            return alignUp(bytes, ALIGNMENT) - bytes;
        }

        private static byte[] layoutCapacity(List<Entry> entries) {
            long bytes = 0;
            for (Entry e : entries) {
                bytes = alignUp(bytes + e.nMat() * (long) e.cols() * 4L, ALIGNMENT);
                bytes = alignUp(bytes + e.nMat() * 4L, ALIGNMENT);
            }
            if (bytes > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException("the collected imatrix takes " + bytes
                        + " bytes, more than one array holds");
            }
            return new byte[(int) bytes];
        }

        // ------------------------------------------------------------------ ModelSource

        @Override
        public ModelFormat format() {
            return ModelFormat.GGUF;
        }

        @Override
        public String getFilePath() {
            return filePath;
        }

        @Override
        public Map<String, Object> getMetadata() {
            return Collections.unmodifiableMap(metadata);
        }

        @Override
        public GGUFValueType getValueType(String key) {
            return valueTypes.get(key);
        }

        @Override
        public GGUFValueType getArrayElementType(String key) {
            return elementTypes.get(key);
        }

        @Override
        public String getString(String key) {
            Object v = metadata.get(key);
            return v instanceof String ? (String) v : null;
        }

        @Override
        public List<GGUFTensorInfo> getTensors() {
            return tensors;
        }

        @Override
        public long getTensorCount() {
            return tensors.size();
        }

        @Override
        public WeightStore createMappedStore(StoreUse use) {
            rejectClosed();
            return new StatsStore();
        }

        @Override
        public TensorRegions openRegions(TensorReadMode mode) {
            rejectClosed();
            return new StatsRegions();
        }

        private void rejectClosed() {
            if (closed) {
                throw new IllegalStateException(filePath + " is closed");
            }
        }

        @Override
        public void close() {
            closed = true;
        }

        @Override
        public String toString() {
            return "imatrix[" + tensors.size() + " stats tensors, " + data.length + " data bytes]";
        }

        /** Reads tensor data at the virtual offsets of the table, validating every requested range. */
        private final class StatsStore implements WeightStore {

            private void check(long tensorOffset, int size) {
                if (tensorOffset < 0 || size < 0
                        || (tensorOffset + size) > data.length) {
                    throw new IllegalArgumentException("tensor data [" + tensorOffset + ", "
                            + (tensorOffset + size) + ") is outside the " + data.length + "-byte imatrix source");
                }
            }

            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst) {
                readTensor(tensorOffset, size, dst, 0);
            }

            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
                check(tensorOffset, size);
                System.arraycopy(data, (int) tensorOffset, dst, dstOffset, size);
            }

            @Override
            public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
                check(tensorOffset, size);
                dst.put(ByteBuffer.wrap(data, (int) tensorOffset, size).order(ByteOrder.LITTLE_ENDIAN));
            }

            @Override
            public void close() {
            }
        }

        /** Host-side regions over the same backing array. */
        private final class StatsRegions implements TensorRegions {

            @Override
            public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) {
                long start = info.offset() + byteOff;
                if (byteOff < 0 || length < 0 || start + length > data.length) {
                    throw new IllegalArgumentException(info.name() + " data at [" + byteOff + ", "
                            + (byteOff + length) + ") is outside the imatrix source");
                }
                return ByteBuffer.wrap(data, (int) start, (int) length)
                        .order(ByteOrder.LITTLE_ENDIAN).slice();
            }

            @Override
            public String description() {
                return "the imatrix statistics of " + filePath;
            }

            @Override
            public void close() {
            }
        }
    }

    /** A percent the way llama.cpp logs partial entries. */
    static String percent(long partial, long total) {
        return total == 0 ? "0.00" : String.format(Locale.ROOT, "%.2f", 100.0 * (total - partial) / total);
    }
}
