package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Exports any {@link ModelSource} to one or more GGUF v3 files through {@link GgufEncoder}, the single
 * GGUF encoder of the repository, streamed into the {@link GgufTensorSink} of the export — one file, or
 * a rolling set of llama.cpp split shards. Every file holds the header, the key/value pairs in the
 * source's {@code getMetadata()} order with their GGUF types (writing {@code general.alignment} only
 * when the source lacks it), the tensor infos, each validated with {@link GGUFTensorInfo#validate}, and
 * the data streamed per tensor in one sequential pass from {@code src.createMappedStore(
 * StoreUse.RESIDENT_UPLOAD)} in chunks of at most {@value #CHUNK_BYTES} bytes, zero-padded to the
 * alignment — so no tensor is ever read into the heap whole, and the whole-tensor window of the mapped
 * store applies.
 *
 * <p>{@link Mode#NATIVE} keeps every extension tensor, marking the file with
 * {@link GgufKeys#NF_EXTENSION_TYPES} = BOOL true when the source does not carry the key itself.
 * {@link Mode#COMPAT} converts every extension tensor to a llama.cpp type through
 * {@link GgufCompatConverter} and drops the marker; {@link Mode#COMPAT_EXACT} exports every lossy or
 * approximate row as F32 instead. Conversion notes are logged per tensor. The {@code neuralfree.*} keys
 * stay in COMPAT too, because llama.cpp ignores keys it does not read; the {@code .input_perm} and
 * {@code .input_zero_point} tensors stay only in NATIVE, for which COMPAT fails with a named error.</p>
 *
 * <p>Every output file is written to a {@code .tmp} twin and renamed atomically on success, and the
 * targets of the whole export (the shard files of a split too) are checked before the first byte is
 * written, so a failing export leaves the previous files untouched and removes the temporary ones.
 * {@link Options#vocabOnly} writes metadata and no tensor table at all (llama.cpp {@code --vocab-only}),
 * never opening a store, and {@link Options#smallFirstShard} writes a split whose first shard carries the
 * metadata and no tensors (llama.cpp {@code --no-tensor-first-split}). {@link #plan} reports the files an
 * export writes without writing them (llama.cpp {@code --dry-run}). The split limits of {@link Options}
 * compute a llama.cpp-style tensor-to-shard
 * plan: the files are named {@code <prefix>-<shard, %05d>-of-<count, %05d>.gguf}
 * ({@code llama_split_path}, src/llama.cpp:570-596) and every shard carries the {@code split.*} keys
 * {@link GGUFFileParser#loadShards()} reads — shard 0 the full metadata plus {@code split.no},
 * {@code split.count} and {@code split.tensors.count}, every later shard the alignment plus those three
 * keys. {@link #writeSplitFile} writes one file of a split whose tensor-to-shard plan the caller owns
 * (llama-quantize {@code --keep-split}, which keeps the input's shard boundaries) with the same keys.
 * The declared tensor count is checked against the tensor table only for a single-file export of a
 * source that is not itself a split: a split source's declared count is its first file's shard-local
 * count, and its tensor total was validated against {@code split.tensors.count} when it was opened, so
 * exporting one (a merge) or several files (a re-split) skips the check. Not thread-safe.</p>
 */
public final class GgufWriter {

    private static final Logger LOGGER = LoggingUtil.getLogger(GgufWriter.class);

    /** Largest chunk of tensor data read or written at once (64 MiB). */
    static final int CHUNK_BYTES = 1 << 26;

    /** How the writer treats the NeuralFree extension tensors. */
    public enum Mode {
        /** Keeps the extension types, marking the file with {@link GgufKeys#NF_EXTENSION_TYPES}. */
        NATIVE,
        /** Converts every extension tensor to a llama.cpp type, some rows loose. */
        COMPAT,
        /** Like COMPAT, but every lossy or approximate row becomes F32. */
        COMPAT_EXACT
    }

    /** Immutable options of one export. */
    public static final class Options {
        /** Export mode; never null. */
        public final Mode mode;
        /** Replace an existing {@code out}; otherwise the export fails before anything is written. */
        public final boolean overwrite;
        /** Data alignment, used when the source carries no {@code general.alignment}. */
        public final int alignment;
        /** Metadata-only export, no tensor table (llama.cpp {@code --vocab-only}); the split limits have no effect. */
        public final boolean vocabOnly;
        /** Largest tensor count of one shard: {@code 0} for no limit. */
        public final long splitMaxTensors;
        /**
         * Largest data-section length of one shard in bytes, the length {@link GGUFFileParser} reads back
         * and llama.cpp's split tool compares: {@code 0} for no limit.
         */
        public final long splitMaxBytes;
        /**
         * The first shard of a split carries metadata and no tensors (llama.cpp {@code --no-tensor-first-split},
         * {@code small_first_shard}); the export always writes at least two files when it is set.
         */
        public final boolean smallFirstShard;

        /** Options with the default alignment {@link GGUFFileParser#DEFAULT_ALIGNMENT} and no overwrite. */
        public Options(Mode mode) {
            this(mode, false, GGUFFileParser.DEFAULT_ALIGNMENT);
        }

        public Options(Mode mode, boolean overwrite, int alignment) {
            this(mode, overwrite, alignment, false, 0L, 0L);
        }

        /** The full option set without {@link #smallFirstShard}; validated as the full constructor validates it. */
        public Options(Mode mode, boolean overwrite, int alignment, boolean vocabOnly, long splitMaxTensors,
                       long splitMaxBytes) {
            this(mode, overwrite, alignment, vocabOnly, splitMaxTensors, splitMaxBytes, false);
        }

        /** The full option set; every part validated as the narrower constructors validate it. */
        public Options(Mode mode, boolean overwrite, int alignment, boolean vocabOnly, long splitMaxTensors,
                       long splitMaxBytes, boolean smallFirstShard) {
            if (mode == null) {
                throw new IllegalArgumentException("export mode is null");
            }
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
                throw new IllegalArgumentException("alignment " + alignment + " is not a positive power of two");
            }
            if (splitMaxTensors < 0 || splitMaxTensors > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("splitMaxTensors " + splitMaxTensors
                        + " is not 0 (no limit) or an int tensor count");
            }
            if (splitMaxBytes < 0) {
                throw new IllegalArgumentException("splitMaxBytes " + splitMaxBytes + " is negative");
            }
            if (vocabOnly && (splitMaxTensors != 0 || splitMaxBytes != 0)) {
                throw new IllegalArgumentException("--vocab-only exports no tensors; the split limits "
                        + "(--split-max-tensors, --split-max-size) have nothing to limit");
            }
            if (vocabOnly && smallFirstShard) {
                throw new IllegalArgumentException("--vocab-only exports no tensors; --no-tensor-first-split"
                        + " has no first shard to shrink");
            }
            this.mode = mode;
            this.overwrite = overwrite;
            this.alignment = alignment;
            this.vocabOnly = vocabOnly;
            this.splitMaxTensors = splitMaxTensors;
            this.splitMaxBytes = splitMaxBytes;
            this.smallFirstShard = smallFirstShard;
        }

        /** This options with {@code mode} set to the given value; the other values are kept. */
        public Options mode(Mode mode) {
            return mode == this.mode ? this : copy(mode, overwrite);
        }

        /** This options with {@code overwrite} set to the given value; the other values are kept. */
        public Options overwrite(boolean overwrite) {
            return overwrite == this.overwrite ? this : copy(mode, overwrite);
        }

        /** This options with {@code vocabOnly} set to the given value; the other values are kept. */
        public Options vocabOnly(boolean vocabOnly) {
            return vocabOnly == this.vocabOnly ? this : copy(vocabOnly, alignment, splitMaxTensors, splitMaxBytes,
                    smallFirstShard);
        }

        /** This options with {@code splitMaxTensors} set to the given value; the other values are kept. */
        public Options splitMaxTensors(long splitMaxTensors) {
            return splitMaxTensors == this.splitMaxTensors ? this
                    : copy(vocabOnly, alignment, splitMaxTensors, splitMaxBytes, smallFirstShard);
        }

        /** This options with {@code splitMaxBytes} set to the given value; the other values are kept. */
        public Options splitMaxBytes(long splitMaxBytes) {
            return splitMaxBytes == this.splitMaxBytes ? this
                    : copy(vocabOnly, alignment, splitMaxTensors, splitMaxBytes, smallFirstShard);
        }

        /** This options with {@code smallFirstShard} set to the given value; the other values are kept. */
        public Options smallFirstShard(boolean smallFirstShard) {
            return smallFirstShard == this.smallFirstShard ? this
                    : copy(vocabOnly, alignment, splitMaxTensors, splitMaxBytes, smallFirstShard);
        }

        private Options copy(Mode mode, boolean overwrite) {
            return new Options(mode, overwrite, alignment, vocabOnly, splitMaxTensors, splitMaxBytes,
                    smallFirstShard);
        }

        private Options copy(boolean vocabOnly, int alignment, long splitMaxTensors, long splitMaxBytes,
                             boolean smallFirstShard) {
            return new Options(mode, overwrite, alignment, vocabOnly, splitMaxTensors, splitMaxBytes,
                    smallFirstShard);
        }
    }

    /** One file of a planned export: its path, its tensor count and its tensor data bytes. */
    public static final class PlannedShard {

        private final Path file;
        private final int tensorCount;
        private final long tensorBytes;

        PlannedShard(Path file, int tensorCount, long tensorBytes) {
            this.file = file;
            this.tensorCount = tensorCount;
            this.tensorBytes = tensorBytes;
        }

        /** The final path the export writes this shard to. */
        public Path file() {
            return file;
        }

        /** The tensors the shard's table holds. */
        public int tensorCount() {
            return tensorCount;
        }

        /** The total tensor data bytes of the shard, without header, key/value and padding bytes. */
        public long tensorBytes() {
            return tensorBytes;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof PlannedShard)) {
                return false;
            }
            PlannedShard other = (PlannedShard) o;
            return file.equals(other.file) && tensorCount == other.tensorCount && tensorBytes == other.tensorBytes;
        }

        @Override
        public int hashCode() {
            return Objects.hash(file, tensorCount, tensorBytes);
        }

        @Override
        public String toString() {
            return file + ": n_tensors = " + tensorCount + ", total_size = " + tensorBytes;
        }
    }

    /**
     * The position of one file in a split whose tensor-to-shard plan the caller owns: the file's
     * {@code split.no}, the split's {@code split.count} and the {@code split.tensors.count} of all its files,
     * the values llama-quantize {@code --keep-split} writes on every shard (src/llama-quant.cpp). Immutable.
     */
    public static final class SplitShard {

        private final int shardNo;
        private final int shardCount;
        private final int totalTensors;

        /**
         * @throws IllegalArgumentException when {@code shardCount} is not in {@code [2, 65535]} (a UINT16
         *                                  {@code split.count} of a real split), {@code shardNo} not in
         *                                  {@code [0, shardCount)}, or {@code totalTensors} not an INT32 count
         */
        public SplitShard(int shardNo, int shardCount, long totalTensors) {
            if (shardCount < 2 || shardCount > 0xFFFF) {
                throw new IllegalArgumentException("split.count " + shardCount + " is not in [2, 65535]");
            }
            if (shardNo < 0 || shardNo >= shardCount) {
                throw new IllegalArgumentException("split.no " + shardNo + " is not in [0, " + shardCount + ")");
            }
            if (totalTensors < 0 || totalTensors > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("split.tensors.count " + totalTensors
                        + " is not a non-negative INT32 count");
            }
            this.shardNo = shardNo;
            this.shardCount = shardCount;
            this.totalTensors = (int) totalTensors;
        }

        /** The file's {@code split.no}, 0-based. */
        public int shardNo() {
            return shardNo;
        }

        /** The split's {@code split.count}. */
        public int shardCount() {
            return shardCount;
        }

        /** The {@code split.tensors.count}: the tensors of all files of the split. */
        public int totalTensors() {
            return totalTensors;
        }
    }

    private GgufWriter() {
    }

    /**
     * Exports {@code src} to {@code out} — or to the llama.cpp shard names derived from it, per the
     * options' split limits.
     *
     * @return the output files written, in order: {@code out} itself, or the split's shard paths
     * @throws IOException naming the source or the tensor for a malformed key/value value, an
     *                     export-blocking tensor, a tensor larger than the byte split limit or a
     *                     file-system problem; an output file is only replaced when the whole export
     *                     succeeded
     */
    public static List<Path> write(ModelSource src, Path out, Options o) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(o, "options");
        int alignment = effectiveAlignment(src, o);
        if (o.vocabOnly) {
            return exportVocabOnly(src, out, o, alignment);
        }
        try (WeightStore store = src.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            return export(src, out, o, alignment, store);
        }
    }

    /**
     * The files an export of {@code src} would {@linkplain #write write} with the options, read from the
     * source alone (the converter's {@code --dry-run}): the shard paths in order, each with its tensor count
     * and total tensor bytes. A vocab-only export plans its single file with no tensors. Nothing is written
     * and no output target is checked.
     *
     * @throws IOException naming the source or the tensor for a malformed key/value value or an
     *                     export-blocking tensor, the same checks {@link #write} makes before opening a file
     */
    public static List<PlannedShard> plan(ModelSource src, Path out, Options o) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(o, "options");
        if (o.vocabOnly) {
            return List.of(new PlannedShard(out, 0, 0));
        }
        int alignment = effectiveAlignment(src, o);
        try (WeightStore store = src.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            List<GGUFTensorInfo> targetTypes = targetTypes(src, store, o);
            List<List<GGUFTensorInfo>> shards = shardPlanOf(targetTypes, o, alignment);
            GgufTensorSink sink = sinkOf(out, shards.size(), o);
            List<PlannedShard> plan = new ArrayList<>(shards.size());
            for (int i = 0; i < shards.size(); i++) {
                plan.add(new PlannedShard(sink.file(i), shards.get(i).size(), dataBytes(shards.get(i))));
            }
            return List.copyOf(plan);
        }
    }

    /**
     * Exports {@code src}'s whole tensor table to the single file {@code out} as file {@code shard.shardNo()} of a
     * split the caller plans. The {@code split.*} keys are {@code shard}'s, never the source's own; file 0 carries
     * the source's metadata, every later file the alignment and the extension marker its tensors need, as the files
     * of a split {@link #write} plans itself.
     *
     * @return {@code out}
     * @throws IllegalArgumentException when the options carry a split limit, {@link Options#smallFirstShard} or
     *                                  {@link Options#vocabOnly}, which plan files of their own
     * @throws IOException              as {@link #write} throws it, and when the source's declared tensor count
     *                                  does not match its table
     */
    public static Path writeSplitFile(ModelSource src, Path out, Options o, SplitShard shard) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(o, "options");
        Objects.requireNonNull(shard, "shard");
        if (o.vocabOnly || o.smallFirstShard || o.splitMaxTensors != 0 || o.splitMaxBytes != 0) {
            throw new IllegalArgumentException("a split file of a caller-planned split takes no options that plan"
                    + " files (--split-max-tensors, --split-max-size, --no-tensor-first-split, --vocab-only)");
        }
        int alignment = effectiveAlignment(src, o);
        try (WeightStore store = src.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            List<GGUFTensorInfo> srcTensors = List.copyOf(src.getTensors());
            List<GGUFTensorInfo> targetTypes = targetTypes(src, store, o);
            checkDeclaredCount(src, srcTensors.size());
            GgufTensorSink sink = GgufTensorSink.singleFile(out, o.overwrite);
            try {
                checkTargets(sink, o);
                try (sink) {
                    writeShard(sink, 0, src, srcTensors, targetTypes, shardKvs(src, o.mode, alignment,
                            shard.shardNo(), shard.shardCount(), shard.totalTensors(), targetTypes), alignment, store);
                }
                return out;
            } finally {
                sink.close();
            }
        }
    }

    /** Fails when the source's declared tensor count differs from its table, which one file must match. */
    private static void checkDeclaredCount(ModelSource src, int tableCount) throws IOException {
        if (src.getTensorCount() != tableCount) {
            throw new IOException("source '" + src.getFilePath() + "' declares " + src.getTensorCount()
                    + " tensors but its table holds " + tableCount
                    + "; the count must match for a single-file export");
        }
    }

    /**
     * Whether the source presents itself as the first file of a split: its metadata carries
     * {@code split.count} above 1, the form {@link GGUFFileParser#loadShards()} opens as a multi-shard
     * model. Such a source's declared tensor count is its first file's shard-local count, while its
     * tensor table holds every shard and was validated against {@code split.tensors.count} when it was
     * opened, so the declared-count check of a single-file export does not apply to it.
     *
     * @throws IOException naming {@code split.count} when the source carries the key with a value no
     *                     integer metadata key can hold
     */
    static boolean isSplitSource(ModelSource src) throws IOException {
        String key = GgufKeys.SPLIT_COUNT;
        Object value = src.getMetadata().get(key);
        if (value == null) {
            return false;
        }
        long count;
        try {
            count = GGUFValueType.integerValue(key, value, src.getValueType(key));
        } catch (IllegalArgumentException e) {
            throw new IOException("source '" + src.getFilePath() + "': " + e.getMessage(), e);
        }
        return count > 1;
    }

    /**
     * One metadata-and-tokenizer file ({@code --vocab-only}): the source's metadata order, no tensors, padded
     * to the alignment as llama.cpp's {@code gguf_write_out} pads its meta-only output.
     */
    private static List<Path> exportVocabOnly(ModelSource src, Path out, Options o, int alignment)
            throws IOException {
        GgufTensorSink sink = GgufTensorSink.singleFile(out, o.overwrite);
        try {
            checkTargets(sink, o);
            try (sink) {
                List<KvPair> kvs = vocabKvs(src, o.mode, alignment);
                GgufEncoder enc = sink.begin(0);
                enc.writeHeader(sink.channel(0), 0, kvs.size());
                writeKvs(enc, kvs);
                enc.align(alignment);
                enc.flush();
                sink.end(0);
            }
            return List.of(out);
        } finally {
            sink.close();
        }
    }

    /** The whole export: the per-tensor target types, the shard plan, then every file written in order. */
    private static List<Path> export(ModelSource src, Path out, Options o, int alignment, WeightStore store)
            throws IOException {
        List<GGUFTensorInfo> srcTensors = List.copyOf(src.getTensors());
        int tensorCount = srcTensors.size();
        // The per-tensor target types are decided for the whole table before the first file is opened, so
        // a COMPAT-mode export of a native-only tensor fails before anything is replaced.
        List<GGUFTensorInfo> targetTypes = targetTypes(src, store, o);
        List<List<GGUFTensorInfo>> shards = shardPlanOf(targetTypes, o, alignment);
        int shardCount = shards.size();
        if (shardCount == 1 && !isSplitSource(src)) {
            checkDeclaredCount(src, tensorCount);
        }
        GgufTensorSink sink = sinkOf(out, shardCount, o);
        try {
            checkTargets(sink, o);
            List<Path> written = new ArrayList<>(shardCount);
            for (int i = 0; i < sink.files(); i++) {
                written.add(sink.file(i));
            }
            try (sink) {
                int first = 0;
                for (int s = 0; s < shardCount; s++) {
                    List<GGUFTensorInfo> shardTypes = shards.get(s);
                    writeShard(sink, s, src,
                            srcTensors.subList(first, first + shardTypes.size()), shardTypes,
                            shardKvs(src, o.mode, alignment, s, shardCount, tensorCount, shardTypes), alignment,
                            store);
                    first += shardTypes.size();
                }
            }
            return List.copyOf(written);
        } finally {
            sink.close();
        }
    }

    /**
     * The per-tensor target types of the whole table: every source tensor through
     * {@link GgufCompatConverter#targetType}, with a conversion note per changed type.
     */
    private static List<GGUFTensorInfo> targetTypes(ModelSource src, WeightStore store, Options o)
            throws IOException {
        List<GGUFTensorInfo> srcTensors = List.copyOf(src.getTensors());
        List<GGUFTensorInfo> targetTypes = new ArrayList<>(srcTensors.size());
        for (GGUFTensorInfo t : srcTensors) {
            GGMLType target = GgufCompatConverter.targetType(t, store, o.mode);
            if (target != t.type()) {
                LOGGER.info("export converts tensor '{}' from {} to {} ({})", t.name(), t.type(), target,
                        GgufCompatConverter.isLossy(t.type(), target) ? "lossy" : "exact");
            }
            // The offsets of this type-side view are re-placed with the shard-local offsets below.
            targetTypes.add(new GGUFTensorInfo(t.name(), t.dims(), target, t.offset()));
        }
        return targetTypes;
    }

    /**
     * The shard plan of the export: {@link #shardPlan} of the target types, plus the leading metadata-only
     * shard of {@link Options#smallFirstShard}.
     */
    private static List<List<GGUFTensorInfo>> shardPlanOf(List<GGUFTensorInfo> targetTypes, Options o, int alignment)
            throws IOException {
        List<List<GGUFTensorInfo>> shards = shardPlan(targetTypes, o, alignment);
        if (!o.smallFirstShard) {
            return shards;
        }
        List<List<GGUFTensorInfo>> withEmptyFirst = new ArrayList<>(shards.size() + 2);
        withEmptyFirst.add(new ArrayList<>(0));
        withEmptyFirst.addAll(shards);
        if (withEmptyFirst.size() == 1) {
            // An empty tensor table still splits: llama.cpp's small_first_shard always writes two files.
            withEmptyFirst.add(new ArrayList<>(0));
        }
        return withEmptyFirst;
    }

    /** The sink of {@code shardCount} files: {@code out} itself, or the llama.cpp shard names from it. */
    private static GgufTensorSink sinkOf(Path out, int shardCount, Options o) {
        return shardCount == 1
                ? GgufTensorSink.singleFile(out, o.overwrite)
                : GgufTensorSink.shards(out, shardCount, o.overwrite);
    }

    /** The total tensor data bytes of one shard, the length the split plan reports. */
    private static long dataBytes(List<GGUFTensorInfo> tensors) {
        long bytes = 0;
        for (GGUFTensorInfo t : tensors) {
            bytes += t.sizeInBytes();
        }
        return bytes;
    }

    /** Writes one file of the export: header, key/values, tensor infos and the streamed tensor data. */
    private static void writeShard(GgufTensorSink sink, int shardNo, ModelSource src,
                                   List<GGUFTensorInfo> srcTensors, List<GGUFTensorInfo> targetTypes,
                                   List<KvPair> kvs, int alignment, WeightStore store)
            throws IOException {
        GgufEncoder enc = sink.begin(shardNo);
        String prefix = "export of '" + src.getFilePath() + "' to '" + sink.file(shardNo) + "'";
        enc.writeHeader(sink.channel(shardNo), targetTypes.size(), kvs.size());
        writeKvs(enc, kvs);
        long offset = 0;
        List<GGUFTensorInfo> placed = new ArrayList<>(targetTypes.size());
        for (GGUFTensorInfo t : targetTypes) {
            GGUFTensorInfo p = new GGUFTensorInfo(t.name(), t.dims(), t.type(), offset);
            GGUFTensorInfo.validate(prefix, p);
            enc.writeTensorInfo(p, offset);
            placed.add(p);
            offset = alignUp(offset + t.sizeInBytes(), alignment);
        }
        enc.align(alignment);
        for (int i = 0; i < placed.size(); i++) {
            GGUFTensorInfo source = srcTensors.get(i);
            GGUFTensorInfo target = placed.get(i);
            if (source.type() == target.type()) {
                copyStreamed(source, store, enc);
            } else {
                GgufCompatConverter.convert(source, target, store, enc);
            }
            enc.align(alignment);
        }
        enc.flush();
        sink.end(shardNo);
    }

    private static void writeKvs(GgufEncoder enc, List<KvPair> kvs) throws IOException {
        for (KvPair kv : kvs) {
            enc.writeKv(kv.key, kv.type, kv.elementType, kv.value);
        }
    }

    /**
     * The key/value pairs of one output file: the source's metadata in {@code getMetadata()} order minus
     * {@link GgufKeys#NF_EXTENSION_TYPES} in the COMPAT modes and the source's own {@code split.*} keys, plus
     * the {@code split.*} keys of {@link GGUFFileParser#loadShards()} for a split ({@code split.no},
     * {@code split.count}, {@code split.tensors.count} on every shard; shard 0 in its full metadata set, the
     * later shards with only the alignment), plus {@code general.alignment} when the source carries none — so a later
     * shard's data alignment is never silently the 32 default either.
     */
    private static List<KvPair> shardKvs(ModelSource src, Mode mode, int alignment, int shardNo, int shardCount,
                                         long totalTensors, List<GGUFTensorInfo> shardTensors) throws IOException {
        List<KvPair> kvs = new ArrayList<>();
        if (shardNo > 0) {
            kvs.add(new KvPair(GgufKeys.GENERAL_ALIGNMENT, GGUFValueType.UINT32, null, alignment));
        }
        // The header reader accepts an extension tensor only in a file whose own key/value section marks
        // the extension types (see GGUFHeaderReader), so every shard holding one carries the key itself.
        boolean extensionTensors = false;
        for (GGUFTensorInfo t : shardTensors) {
            if (t.type().isDeviceOnly()) {
                throw new IllegalArgumentException("tensor '" + t.name() + "' has type " + t.type()
                        + ", a device-resident form that GGUF files never store");
            }
            extensionTensors |= t.type().isExtension();
        }
        if (extensionTensors && mode == Mode.NATIVE && shardNo > 0 && !src.getMetadata()
                .containsKey(GgufKeys.NF_EXTENSION_TYPES)) {
            kvs.add(new KvPair(GgufKeys.NF_EXTENSION_TYPES, GGUFValueType.BOOL, null, Boolean.TRUE));
        }
        if (shardCount > 1) {
            kvs.add(new KvPair(GgufKeys.SPLIT_NO, GGUFValueType.UINT16, null, shardNo));
            kvs.add(new KvPair(GgufKeys.SPLIT_COUNT, GGUFValueType.UINT16, null, shardCount));
            kvs.add(new KvPair(GgufKeys.SPLIT_TENSORS_COUNT, GGUFValueType.INT32, null, (int) totalTensors));
        }
        if (shardNo == 0) {
            kvs.addAll(compatKvs(src, mode, alignment));
        }
        return kvs;
    }

    /**
     * The key/value pairs of a {@code --vocab-only} export: the source's metadata in
     * {@code getMetadata()} order, minus {@link GgufKeys#NF_EXTENSION_TYPES} in the COMPAT modes and
     * the source's own {@code split.*} keys, plus
     * {@code general.alignment} when the source carries none. No tensor-based marker rule applies,
     * because no tensor is exported.
     */
    private static List<KvPair> vocabKvs(ModelSource src, Mode mode, int alignment) throws IOException {
        Map<String, Object> metadata = src.getMetadata();
        List<KvPair> kvs = new ArrayList<>(metadata.size() + 1);
        for (Map.Entry<String, Object> e : metadata.entrySet()) {
            String key = e.getKey();
            if ((mode != Mode.NATIVE && GgufKeys.NF_EXTENSION_TYPES.equals(key)) || isSplitKey(key)) {
                continue;
            }
            GGUFValueType type = src.getValueType(key);
            if (type == null) {
                throw new IOException("metadata key '" + key + "' of source '" + src.getFilePath()
                        + "' has no GGUF value type");
            }
            GGUFValueType elem = type == GGUFValueType.ARRAY ? src.getArrayElementType(key) : null;
            kvs.add(new KvPair(key, type, elem, e.getValue()));
        }
        if (!metadata.containsKey(GgufKeys.GENERAL_ALIGNMENT)) {
            kvs.add(new KvPair(GgufKeys.GENERAL_ALIGNMENT, GGUFValueType.UINT32, null, alignment));
        }
        return kvs;
    }

    /**
     * Whether {@code key} is one of the {@code split.*} keys the writer derives from its own shard
     * plan, so a split source's keys never reach the output.
     */
    private static boolean isSplitKey(String key) {
        return GgufKeys.SPLIT_NO.equals(key) || GgufKeys.SPLIT_COUNT.equals(key)
                || GgufKeys.SPLIT_TENSORS_COUNT.equals(key);
    }

    /**
     * Key/value pairs of shard 0's full metadata set: the source's metadata in {@code getMetadata()}
     * order, minus {@link GgufKeys#NF_EXTENSION_TYPES} in the COMPAT modes and the source's own
     * {@code split.*} keys, plus the marker in NATIVE for a source whose tensors are of extension types
     * but which does not carry the key, plus
     * {@code general.alignment} when the source carries none.
     */
    private static List<KvPair> compatKvs(ModelSource src, Mode mode, int alignment) throws IOException {
        Map<String, Object> metadata = src.getMetadata();
        List<KvPair> kvs = new ArrayList<>(metadata.size() + 2);
        boolean extensionTensors = false;
        for (GGUFTensorInfo t : src.getTensors()) {
            extensionTensors |= t.type().isExtension();
        }
        for (Map.Entry<String, Object> e : metadata.entrySet()) {
            String key = e.getKey();
            if ((mode != Mode.NATIVE && GgufKeys.NF_EXTENSION_TYPES.equals(key)) || isSplitKey(key)) {
                continue;
            }
            GGUFValueType type = src.getValueType(key);
            if (type == null) {
                throw new IOException("metadata key '" + key + "' of source '" + src.getFilePath()
                        + "' has no GGUF value type");
            }
            GGUFValueType elem = type == GGUFValueType.ARRAY ? src.getArrayElementType(key) : null;
            kvs.add(new KvPair(key, type, elem, e.getValue()));
        }
        if (extensionTensors && mode == Mode.NATIVE
                && !metadata.containsKey(GgufKeys.NF_EXTENSION_TYPES)) {
            kvs.add(new KvPair(GgufKeys.NF_EXTENSION_TYPES, GGUFValueType.BOOL, null, Boolean.TRUE));
        }
        if (!metadata.containsKey(GgufKeys.GENERAL_ALIGNMENT)) {
            kvs.add(new KvPair(GgufKeys.GENERAL_ALIGNMENT, GGUFValueType.UINT32, null, alignment));
        }
        return kvs;
    }

    /**
     * The tensor-to-shard plan of the split: the target table in order, cut after a tensor that fills a
     * limit the way llama.cpp's split tool cuts at tensor boundaries. A shard holds at most
     * {@code splitMaxTensors} tensors and its data section at most {@code splitMaxBytes} bytes counting
     * the padding the parser reads back, except that a single tensor always gets its own shard when it
     * alone exceeds the byte limit — that is a named error instead. The table may be arbitrarily large;
     * {@code 0} on either limit means no limit for it.
     *
     * @throws IOException naming the first tensor no split can make smaller
     */
    static List<List<GGUFTensorInfo>> shardPlan(List<GGUFTensorInfo> tensors, Options o, int alignment)
            throws IOException {
        if (o.splitMaxTensors == 0 && o.splitMaxBytes == 0) {
            List<List<GGUFTensorInfo>> one = new ArrayList<>(1);
            one.add(tensors);
            return one;
        }
        List<List<GGUFTensorInfo>> shards = new ArrayList<>();
        List<GGUFTensorInfo> shard = new ArrayList<>();
        long end = 0;
        for (GGUFTensorInfo t : tensors) {
            long size = t.sizeInBytes();
            long next = alignUp(end, alignment) + size;
            if (o.splitMaxBytes != 0 && size > o.splitMaxBytes) {
                throw new IOException("tensor '" + t.name() + "' alone occupies " + size
                        + " bytes, more than --split-max-size " + o.splitMaxBytes
                        + "; no split can produce a smaller shard");
            }
            if (!shard.isEmpty()) {
                if (o.splitMaxTensors != 0 && shard.size() >= o.splitMaxTensors) {
                    shards.add(shard);
                    shard = new ArrayList<>();
                    next = size;
                } else if (o.splitMaxBytes != 0 && next > o.splitMaxBytes) {
                    shards.add(shard);
                    shard = new ArrayList<>();
                    next = size;
                }
            }
            shard.add(t);
            end = next;
        }
        if (!shard.isEmpty()) {
            shards.add(shard);
        }
        return shards;
    }

    // ------------------------------------------------------------------ alignment and file targets

    private static int effectiveAlignment(ModelSource src, Options o) throws IOException {
        return effectiveAlignment(src, o.alignment);
    }

    /**
     * The alignment the data section uses: the source's {@code general.alignment} when it carries the key,
     * else {@code fallback}. As llama.cpp does, the value must be UINT32 and a positive power of two,
     * and the export reports a malformed one with a named error.
     */
    static int effectiveAlignment(ModelSource src, int fallback) throws IOException {
        String key = GgufKeys.GENERAL_ALIGNMENT;
        Object value = src.getMetadata().get(key);
        if (value == null) {
            return fallback;
        }
        GGUFValueType type = src.getValueType(key);
        long a;
        try {
            a = GGUFValueType.uint32Value(key, value, type);
        } catch (IllegalArgumentException e) {
            throw new IOException("source '" + src.getFilePath() + "': " + e.getMessage(), e);
        }
        if (a <= 0 || (a & (a - 1)) != 0) {
            throw new IOException("source '" + src.getFilePath() + "': " + key + " " + a
                    + " is not a power of two");
        }
        return (int) a;
    }

    /** {@code v} rounded up to the next multiple of {@code alignment}. */
    static long alignUp(long v, int alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    /** Fails before the first byte when any target of the export exists and {@code overwrite} is off. */
    private static void checkTargets(GgufTensorSink sink, Options o) throws IOException {
        if (o.overwrite) {
            return;
        }
        for (int i = 0; i < sink.files(); i++) {
            Path out = sink.file(i);
            if (Files.exists(out)) {
                throw new IOException("file '" + out + "' already exists; write with overwrite enabled to "
                        + "replace it");
            }
        }
    }

    // ------------------------------------------------------------------ streaming and held key/values

    /** Copies a tensor of unchanged type from the store in chunks; the alignment padding follows. */
    static void copyStreamed(GGUFTensorInfo src, WeightStore store, GgufEncoder enc) throws IOException {
        long bytes = src.sizeInBytes();
        byte[] buf = new byte[(int) Math.min(bytes, CHUNK_BYTES)];
        long done = 0;
        while (done < bytes) {
            int n = (int) Math.min(buf.length, bytes - done);
            store.readTensor(src.offset() + done, n, buf, 0);
            enc.writeData(buf, 0, n);
            done += n;
        }
    }

    /** One key/value pair of the export, held until every header count is known. */
    static final class KvPair {
        final String key;
        final GGUFValueType type;
        final GGUFValueType elementType;
        final Object value;

        KvPair(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            this.key = key;
            this.type = type;
            this.elementType = elementType;
            this.value = value;
        }
    }
}
