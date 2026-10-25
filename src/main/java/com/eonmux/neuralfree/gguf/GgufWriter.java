package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Exports any {@link ModelSource} to a GGUF v3 file through {@link GgufEncoder}, the single GGUF encoder of
 * the repository. The export writes the header, the key/value pairs in the source's {@code getMetadata()}
 * order with their GGUF types (writing {@code general.alignment} only when the source lacks it), the tensor
 * infos, each validated with {@link GGUFTensorInfo#validate}, and the data streamed per tensor in one
 * sequential pass from {@code src.createMappedStore(StoreUse.RESIDENT_UPLOAD)} in chunks of at most
 * {@value #CHUNK_BYTES} bytes, zero-padded to the alignment — so no tensor is ever read into the heap
 * whole, and the whole-tensor window of the mapped store applies.
 *
 * <p>{@link Mode#NATIVE} keeps every extension tensor, marking the file with
 * {@link GgufKeys#NF_EXTENSION_TYPES} = BOOL true when the source does not carry the key itself.
 * {@link Mode#COMPAT} converts every extension tensor to a llama.cpp type through
 * {@link GgufCompatConverter} and drops the marker; {@link Mode#COMPAT_EXACT} exports every lossy or
 * approximate row as F32 instead. Conversion notes are logged per tensor. The {@code neuralfree.*} keys
 * stay in COMPAT too, because llama.cpp ignores keys it does not read; the {@code .input_perm} and
 * {@code .input_zero_point} tensors stay only in NATIVE, for which COMPAT fails with a named error.
 *
 * <p>Output goes to {@code out + ".tmp"} and is renamed atomically on success; an existing {@code out} is
 * replaced only with {@code overwrite}. A failing export leaves the previous file untouched and removes the
 * temporary one. Split output, {@code --vocab-only} and the {@code runNfTool} task extend this class as
 * PAR TL-1 (§0.5). Not thread-safe.</p>
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

        /** Options with the default alignment {@link GGUFFileParser#DEFAULT_ALIGNMENT} and no overwrite. */
        public Options(Mode mode) {
            this(mode, false, GGUFFileParser.DEFAULT_ALIGNMENT);
        }

        public Options(Mode mode, boolean overwrite, int alignment) {
            if (mode == null) {
                throw new IllegalArgumentException("export mode is null");
            }
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
                throw new IllegalArgumentException("alignment " + alignment + " is not a positive power of two");
            }
            this.mode = mode;
            this.overwrite = overwrite;
            this.alignment = alignment;
        }

        /** This options with {@code overwrite} set to the given value; the other values are kept. */
        public Options overwrite(boolean overwrite) {
            return overwrite == this.overwrite
                    ? this
                    : new Options(mode, overwrite, alignment);
        }
    }

    private GgufWriter() {
    }

    /**
     * Exports {@code src} to {@code out}.
     *
     * @throws IOException naming the source or the tensor for a malformed key/value value, an
     *                     export-blocking tensor or a file-system problem; {@code out} is only replaced
     *                     when the whole export succeeded
     */
    public static void write(ModelSource src, Path out, Options o) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(o, "options");
        if (!o.overwrite && Files.exists(out)) {
            throw new IOException("file '" + out + "' already exists; write with overwrite enabled to "
                    + "replace it");
        }
        Path tmp = out.resolveSibling(out.getFileName().toString() + ".tmp");
        boolean complete = false;
        try (WeightStore store = src.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            writeFile(src, out, tmp, store, o);
            complete = true;
        } finally {
            if (!complete) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException e) {
                    LOGGER.warn("cannot remove the incomplete export {}", tmp, e);
                }
            }
        }
    }

    /** Writes the whole export to {@code tmp} and renames it to {@code out}. */
    private static void writeFile(ModelSource src, Path out, Path tmp, WeightStore store, Options o)
            throws IOException {
        List<GGUFTensorInfo> srcTensors = src.getTensors();
        long declared = src.getTensorCount();
        if (declared != srcTensors.size()) {
            throw new IOException("source '" + src.getFilePath() + "' declares " + declared
                    + " tensors but its table holds " + srcTensors.size()
                    + "; the count must match for a single-file export");
        }
        int alignment = effectiveAlignment(src, o);
        List<KvPair> kvs = compatKvs(src, o.mode, alignment);
        List<GGUFTensorInfo> outTensors = new ArrayList<>(srcTensors.size());
        for (GGUFTensorInfo t : srcTensors) {
            GGMLType target = GgufCompatConverter.targetType(t, store, o.mode);
            if (target != t.type()) {
                LOGGER.info("export converts tensor '{}' from {} to {} ({})", t.name(), t.type(), target,
                        GgufCompatConverter.isLossy(t.type(), target) ? "lossy" : "exact");
            }
            outTensors.add(new GGUFTensorInfo(t.name(), t.dims(), target, t.offset()));
        }
        try (FileChannel channel = FileChannel.open(tmp, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            GgufEncoder enc = new GgufEncoder(channel, tmp.toString());
            enc.writeHeader(channel, outTensors.size(), kvs.size());
            for (KvPair kv : kvs) {
                enc.writeKv(kv.key, kv.type, kv.elementType, kv.value);
            }
            long offset = 0;
            String validatePrefix = "export of '" + src.getFilePath() + "' to '" + out + "'";
            for (int i = 0; i < outTensors.size(); i++) {
                GGUFTensorInfo info = outTensors.get(i);
                GGUFTensorInfo.validate(validatePrefix, info);
                GGUFTensorInfo placed = new GGUFTensorInfo(info.name(), info.dims(), info.type(), offset);
                enc.writeTensorInfo(placed, offset);
                outTensors.set(i, placed);
                offset = alignUp(offset + info.sizeInBytes(), alignment);
            }
            enc.align(alignment);
            for (int i = 0; i < outTensors.size(); i++) {
                GGUFTensorInfo source = srcTensors.get(i);
                if (source.type() == outTensors.get(i).type()) {
                    copyStreamed(source, store, enc);
                } else {
                    GgufCompatConverter.convert(source, outTensors.get(i), store, enc);
                }
                enc.align(alignment);
            }
            enc.flush();
            channel.force(true);
        }
        rename(tmp, out, o);
    }

    /** Copies a tensor of unchanged type from the store in chunks; the alignment padding follows. */
    private static void copyStreamed(GGUFTensorInfo src, WeightStore store, GgufEncoder enc) throws IOException {
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

    /**
     * Key/value pairs of the export: the source's metadata in {@code getMetadata()} order, minus
     * {@link GgufKeys#NF_EXTENSION_TYPES} in the COMPAT modes, plus the marker in NATIVE for a source whose
     * tensors are of extension types but which does not carry the key, plus {@code general.alignment} when
     * the source carries none.
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
            if (mode != Mode.NATIVE && GgufKeys.NF_EXTENSION_TYPES.equals(key)) {
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
     * The alignment the data section uses: the source's {@code general.alignment} when it carries the key,
     * else the options'. As llama.cpp does, the value must be UINT32 and a positive power of two,
     * and the export reports a malformed one with a named error.
     */
    private static int effectiveAlignment(ModelSource src, Options o) throws IOException {
        String key = GgufKeys.GENERAL_ALIGNMENT;
        Object value = src.getMetadata().get(key);
        if (value == null) {
            return o.alignment;
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
    private static long alignUp(long v, int alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    /** Renames {@code tmp} over {@code out}, replacing the previous file only when the options allow it. */
    private static void rename(Path tmp, Path out, Options o) throws IOException {
        if (o.overwrite) {
            try {
                Files.move(tmp, out, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
                return;
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING);
                return;
            }
        }
        Files.move(tmp, out, StandardCopyOption.ATOMIC_MOVE);
    }

    /** One key/value pair of the export, held until every header count is known. */
    private static final class KvPair {
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
