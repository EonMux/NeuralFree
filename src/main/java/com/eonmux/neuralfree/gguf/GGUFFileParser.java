package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Parses a GGUF model: its metadata, its tensor infos and, for a split model ({@code <prefix>-00001-of-0000N.gguf}),
 * the headers of every split. Header and split checks follow llama.cpp ggml/src/gguf.cpp and
 * src/llama-model-loader.cpp, so a file llama.cpp rejects is rejected here with the reason.
 *
 * <p>The tensors of all splits share one virtual offset space: the tensors of split {@code s} are at
 * {@link Shard#virtualBase()} plus their offset inside that split's data section. {@link #getShards()} describes
 * each split, and tensor data reads resolve virtual offsets through it.</p>
 */
public final class GGUFFileParser {
    public static final String MAGIC = "GGUF";
    public static final int DEFAULT_ALIGNMENT = 32;

    /** Largest byte count a single {@code byte[]} tensor read can return. */
    public static final long MAX_ARRAY_TENSOR_BYTES = Integer.MAX_VALUE - 8L;

    /** Largest element count one decoded {@code float[]} can hold. */
    private static final long MAX_FLOAT_ARRAY_ELEMENTS = Integer.MAX_VALUE - 8L;

    private static final String KEY_ALIGNMENT = "general.alignment";

    /**
     * One file of the model: its path, where its data section starts, its place in the virtual offset space, and
     * the range of {@link #getTensors()} it holds. Immutable.
     */
    public static final class Shard {
        private final int index;
        private final String path;
        private final long fileDataOffset;
        private final long virtualBase;
        private final long dataLength;
        private final int firstTensor;
        private final int tensorCount;

        Shard(int index, String path, long fileDataOffset, long virtualBase, long dataLength, int firstTensor,
              int tensorCount) {
            this.index = index;
            this.path = path;
            this.fileDataOffset = fileDataOffset;
            this.virtualBase = virtualBase;
            this.dataLength = dataLength;
            this.firstTensor = firstTensor;
            this.tensorCount = tensorCount;
        }

        /** Position of the split, 0 for the file the parser was opened with. */
        public int index() { return index; }

        /** Path of the split file. */
        public String path() { return path; }

        /** File offset of the split's data section. */
        public long fileDataOffset() { return fileDataOffset; }

        /** Virtual offset of the first byte of the split's data section. */
        public long virtualBase() { return virtualBase; }

        /** Bytes from the start of the data section to the end of the split's last tensor. */
        public long dataLength() { return dataLength; }

        /** Index in {@link GGUFFileParser#getTensors()} of the split's first tensor. */
        public int firstTensor() { return firstTensor; }

        /** Number of tensors the split holds. */
        public int tensorCount() { return tensorCount; }

        /** File offset of the byte at {@code virtualOffset}, which must lie in this split. */
        public long fileOffset(long virtualOffset) {
            return fileDataOffset + (virtualOffset - virtualBase);
        }

        @Override
        public String toString() {
            return "Shard{" + index + ", " + path + ", data@" + fileDataOffset + ", base=" + virtualBase
                    + ", tensors=[" + firstTensor + ", +" + tensorCount + ")}";
        }
    }

    private final String filePath;
    private final RandomAccessFile file;
    private final int version;
    private final long tensorCount;
    private final long kvCount;
    private final long alignment;
    private final long dataOffset;

    private final Map<String, Object> metadata = new HashMap<>();
    private final Map<String, GGUFValueType> valueTypes = new HashMap<>();
    private final Map<String, GGUFValueType> arrayElementTypes = new HashMap<>();
    private final List<GGUFTensorInfo> tensors = new ArrayList<>();
    private final Map<String, Integer> tensorNameToIndex = new HashMap<>();

    /** Every file of the model in split order; one entry for a single-file model. */
    private final List<Shard> shards;

    /**
     * Lazily-initialized store used only when tensor data is read from a multi-shard GGUF.
     * Null for single-file GGUFs (positional reads on {@link #file} are used instead).
     * Closed in {@link #close()}.
     */
    private WeightStore internalStore;

    public GGUFFileParser(String path) throws IOException {
        this.filePath = path;
        RandomAccessFile raf = new RandomAccessFile(path, "r");
        try {
            GGUFHeaderReader reader = new GGUFHeaderReader(path, raf.getChannel());
            GGUFHeaderReader.Header header = reader.readHeader();
            this.file = raf;
            this.version = header.version;
            this.tensorCount = header.tensorCount;
            this.kvCount = header.kvCount;

            for (long i = 0; i < kvCount; i++) {
                reader.readKVPair(i, metadata, valueTypes, arrayElementTypes);
            }

            this.alignment = parseAlignment(path, metadata.get(KEY_ALIGNMENT), valueTypes.get(KEY_ALIGNMENT));

            for (long i = 0; i < tensorCount; i++) {
                GGUFTensorInfo info = reader.readTensorInfo(i);
                validateTensorInfo(path, info);
                tensors.add(info);
            }

            this.dataOffset = alignUp(reader.position(), alignment);
            validateDataSection(path, tensors, alignment, dataOffset, raf.length());

            for (int i = 0; i < tensors.size(); i++) {
                GGUFTensorInfo t = tensors.get(i);
                tensorNameToIndex.put(t.name(), i);
            }

            this.shards = Collections.unmodifiableList(loadShards());
        } catch (Throwable t) {
            try {
                raf.close();
            } catch (IOException e) {
                t.addSuppressed(e);
            }
            throw t;
        }
    }

    // -----------------------------------------------------------------------
    // Split loading (llama.cpp src/llama-model-loader.cpp:596-672)
    // -----------------------------------------------------------------------

    /**
     * The model's files. With {@code split.count > 1} this file must be split 0, every other split is parsed and
     * must carry its own {@code split.no}, its tensors are appended to {@link #tensors} at virtual offsets, and the
     * tensor total must equal {@code split.tensors.count}.
     */
    private List<Shard> loadShards() throws IOException {
        Long splitCount = splitKey(filePath, metadata, valueTypes, GgufKeys.SPLIT_COUNT, GGUFValueType.UINT16, false);
        Shard first = new Shard(0, filePath, dataOffset, 0L, computeShardDataLength(tensors), 0, tensors.size());
        if (splitCount == null || splitCount <= 1) {
            return Collections.singletonList(first);
        }
        int n = splitCount.intValue();
        long splitNo = splitKey(filePath, metadata, valueTypes, GgufKeys.SPLIT_NO, GGUFValueType.UINT16, true);
        if (splitNo != 0) {
            throw new IOException("illegal split file idx: " + splitNo + " (file: " + filePath
                    + "), model must be loaded with the first split");
        }
        String prefix = splitPrefix(filePath, n);

        List<Shard> result = new ArrayList<>(n);
        result.add(first);
        long base = first.dataLength();
        for (int s = 1; s < n; s++) {
            String shardPath = splitPath(prefix, s, n);
            ShardHeader header;
            try {
                header = readShardHeader(shardPath, s);
            } catch (IOException e) {
                throw new IOException("failed to load GGUF split " + (s + 1) + " of " + n + " from '" + shardPath
                        + "': " + e.getMessage(), e);
            }
            int firstTensor = tensors.size();
            for (GGUFTensorInfo t : header.tensors) {
                int globalIndex = tensors.size();
                tensors.add(new GGUFTensorInfo(t.name(), t.dims(), t.type(), base + t.offset()));
                if (tensorNameToIndex.putIfAbsent(t.name(), globalIndex) != null) {
                    throw new IOException("invalid model: tensor '" + t.name() + "' is duplicated (split '"
                            + shardPath + "')");
                }
            }
            long length = computeShardDataLength(header.tensors);
            result.add(new Shard(s, shardPath, header.dataOffset, base, length, firstTensor, header.tensors.size()));
            base = Math.addExact(base, length);
        }

        long expected = splitKey(filePath, metadata, valueTypes, GgufKeys.SPLIT_TENSORS_COUNT, GGUFValueType.INT32,
                true);
        if (expected != tensors.size()) {
            throw new IOException("corrupted model: " + expected + " tensors expected but " + tensors.size()
                    + " found in the " + n + " splits of '" + filePath + "'");
        }
        return result;
    }

    /** Shard-local tensor infos and the aligned data-section offset of one shard file. */
    private static final class ShardHeader {
        final List<GGUFTensorInfo> tensors;
        final long dataOffset;

        ShardHeader(List<GGUFTensorInfo> tensors, long dataOffset) {
            this.tensors = tensors;
            this.dataOffset = dataOffset;
        }
    }

    /**
     * Parses the header of split {@code expectedShardNo}, which must carry {@code split.no} equal to it, and
     * returns its raw (shard-local) tensor infos and data offset without modifying any state in this parser.
     */
    private static ShardHeader readShardHeader(String shardPath, int expectedShardNo) throws IOException {
        try (RandomAccessFile shardFile = new RandomAccessFile(shardPath, "r")) {
            GGUFHeaderReader reader = new GGUFHeaderReader(shardPath, shardFile.getChannel());
            GGUFHeaderReader.Header header = reader.readHeader();

            Map<String, Object> shardMeta = new HashMap<>();
            Map<String, GGUFValueType> shardTypes = new HashMap<>();
            for (long i = 0; i < header.kvCount; i++) {
                reader.readKVPair(i, shardMeta, shardTypes, null);
            }

            long splitNo = splitKey(shardPath, shardMeta, shardTypes, GgufKeys.SPLIT_NO, GGUFValueType.UINT16, true);
            if (splitNo != expectedShardNo) {
                throw new IOException("invalid split file idx: " + splitNo + " (file: " + shardPath + "), expected "
                        + expectedShardNo);
            }
            long align = parseAlignment(shardPath, shardMeta.get(KEY_ALIGNMENT), shardTypes.get(KEY_ALIGNMENT));

            List<GGUFTensorInfo> result = new ArrayList<>((int) header.tensorCount);
            for (long i = 0; i < header.tensorCount; i++) {
                GGUFTensorInfo info = reader.readTensorInfo(i);
                validateTensorInfo(shardPath, info);
                result.add(info);
            }
            long shardDataOffset = alignUp(reader.position(), align);
            validateDataSection(shardPath, result, align, shardDataOffset, shardFile.length());
            return new ShardHeader(result, shardDataOffset);
        }
    }

    /**
     * A split key read as llama.cpp {@code get_key} reads it (src/llama-model-loader.cpp:160-173, 276-290): the
     * stored type must be {@code expected}; null when the key is absent and not {@code required}.
     */
    private static Long splitKey(String path, Map<String, Object> meta, Map<String, GGUFValueType> types, String key,
                                 GGUFValueType expected, boolean required) throws IOException {
        Object value = meta.get(key);
        if (value == null) {
            if (required) {
                throw new IOException("Invalid GGUF file '" + path + "': key not found in model: " + key);
            }
            return null;
        }
        GGUFValueType type = types.get(key);
        if (type != expected) {
            throw new IOException("Invalid GGUF file '" + path + "': key " + key + " has wrong type " + type
                    + " but expected type " + expected);
        }
        return GGUFValueType.integerValue(key, value, type);
    }

    /**
     * The split prefix of the first split's path, as llama.cpp {@code llama_split_prefix} derives it
     * (src/llama.cpp:570-596): the path must end with {@code -00001-of-<count>.gguf}. Split numbers are formatted
     * with ASCII digits whatever the default locale, as C {@code snprintf} formats them.
     */
    private static String splitPrefix(String path, int splitCount) throws IOException {
        String suffix = String.format(Locale.ROOT, "-%05d-of-%05d.gguf", 1, splitCount);
        if (path.length() <= suffix.length() || !path.endsWith(suffix)) {
            throw new IOException("invalid split file name: '" + path + "'; the first file of a " + splitCount
                    + "-split model must be named <prefix>" + suffix);
        }
        return path.substring(0, path.length() - suffix.length());
    }

    /**
     * Validates {@code general.alignment} as llama.cpp does (ggml/src/gguf.cpp:622-635): the key must be UINT32
     * and hold a non-zero power of two; absent means {@link #DEFAULT_ALIGNMENT}.
     */
    private static long parseAlignment(String path, Object value, GGUFValueType type) throws IOException {
        if (value == null) {
            return DEFAULT_ALIGNMENT;
        }
        if (type != GGUFValueType.UINT32) {
            throw new IOException("Invalid GGUF file '" + path + "': " + KEY_ALIGNMENT + " must be of type UINT32 but is "
                    + type);
        }
        long a;
        try {
            a = GGUFValueType.integerValue(KEY_ALIGNMENT, value, type);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid GGUF file '" + path + "': " + e.getMessage(), e);
        }
        if (a <= 0 || (a & (a - 1)) != 0) {
            throw new IOException("Invalid GGUF file '" + path + "': general.alignment " + a + " is not a power of two");
        }
        return a;
    }

    /** Byte length of a data section: the end of its furthest tensor. */
    private static long computeShardDataLength(List<GGUFTensorInfo> shardTensors) {
        long maxEnd = 0;
        for (GGUFTensorInfo t : shardTensors) {
            long end = t.offset() + t.sizeInBytes();
            if (end > maxEnd) maxEnd = end;
        }
        return maxEnd;
    }

    /** Path of split {@code shardIdx} (0-based), as llama.cpp {@code llama_split_path} builds it. */
    private static String splitPath(String prefix, int shardIdx, int splitCount) {
        return String.format(Locale.ROOT, "%s-%05d-of-%05d.gguf", prefix, shardIdx + 1, splitCount);
    }

    public int getVersion() { return version; }
    /** Tensor count in the header of the opened file; {@link #getTensors()} holds the tensors of every split. */
    public long getTensorCount() { return tensorCount; }
    /** Key/value pair count in the header of the opened file. */
    public long getKVCount() { return kvCount; }
    /** Tensor data alignment in bytes ({@code general.alignment}, default 32). */
    public long getAlignment() { return alignment; }
    public long getDataOffset() { return dataOffset; }

    public Map<String, Object> getMetadata() { return metadata; }
    public List<GGUFTensorInfo> getTensors() { return tensors; }

    public int getTensorIndex(String name) {
        Integer idx = tensorNameToIndex.get(name);
        return idx != null ? idx : -1;
    }

    public GGUFTensorInfo getTensor(String name) {
        int idx = getTensorIndex(name);
        return idx >= 0 ? tensors.get(idx) : null;
    }

    public GGUFTensorInfo getTensor(int index) {
        return index >= 0 && index < tensors.size() ? tensors.get(index) : null;
    }

    public String getString(String key) {
        Object val = metadata.get(key);
        return val instanceof String ? (String) val : null;
    }

    /**
     * Scalar integer value as a signed int; UINT32 values of 2^31 and above come back negative (use
     * {@link #getUInt32(String)}). Returns null when the key is absent or not an integer scalar.
     */
    public Integer getInt(String key) {
        Object val = metadata.get(key);
        if (val instanceof Integer) return (Integer) val;
        if (val instanceof Long) return ((Long) val).intValue();
        return null;
    }

    public Long getLong(String key) {
        Object val = metadata.get(key);
        if (val instanceof Long) return (Long) val;
        if (val instanceof Integer) return ((Integer) val).longValue();
        return null;
    }

    public Float getFloat(String key) {
        Object val = metadata.get(key);
        if (val instanceof Float) return (Float) val;
        if (val instanceof Double) return ((Double) val).floatValue();
        return null;
    }

    public Boolean getBool(String key) {
        Object val = metadata.get(key);
        return val instanceof Boolean ? (Boolean) val : null;
    }

    @SuppressWarnings("unchecked")
    public List<Object> getArray(String key) {
        Object val = metadata.get(key);
        return val instanceof List ? (List<Object>) val : null;
    }

    /** GGUF type of a metadata value, or null when the key is absent. */
    public GGUFValueType getValueType(String key) {
        return valueTypes.get(key);
    }

    /** Element type of an array metadata value, or null when the key is absent or not an array. */
    public GGUFValueType getArrayElementType(String key) {
        return arrayElementTypes.get(key);
    }

    // Typed accessors: each returns null when the key is absent and throws IllegalArgumentException
    // naming the key when the value has the wrong type or an out-of-range element (GGUFValueType).

    /** Scalar unsigned 32-bit value; UINT32 is read unsigned, other integer types must be in range. */
    public Long getUInt32(String key) {
        Object val = metadata.get(key);
        return val == null ? null : GGUFValueType.uint32Value(key, val, valueTypes.get(key));
    }

    /** Unsigned 32-bit array; UINT32 elements are read unsigned, BOOL elements as 0/1. */
    public long[] getUInt32Array(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.uint32Array(key, arr, arrayElementTypes.get(key));
    }

    /** Signed 32-bit array; BOOL elements as 0/1. */
    public int[] getInt32Array(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.int32Array(key, arr, arrayElementTypes.get(key));
    }

    /** Unsigned 64-bit array; UINT64 elements keep their bit pattern. */
    public long[] getUInt64Array(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.uint64Array(key, arr, arrayElementTypes.get(key));
    }

    /** Numeric array as floats. */
    public float[] getFloatArray(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.floatArray(key, arr, arrayElementTypes.get(key));
    }

    /** BOOL or integer array as booleans (non-zero is true). */
    public boolean[] getBoolArray(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.boolArray(key, arr, arrayElementTypes.get(key));
    }

    /** STRING array. */
    public String[] getStringArray(String key) {
        List<Object> arr = requireArray(key);
        return arr == null ? null : GGUFValueType.stringArray(key, arr, arrayElementTypes.get(key));
    }

    private List<Object> requireArray(String key) {
        Object val = metadata.get(key);
        if (val == null) return null;
        List<Object> arr = getArray(key);
        if (arr == null) {
            throw new IllegalArgumentException("metadata key '" + key + "' must be an array but has GGUF type "
                    + valueTypes.get(key));
        }
        return arr;
    }

    /**
     * Reads a whole tensor into a new array. Thread-safe.
     *
     * @throws IOException when the tensor exceeds {@link #MAX_ARRAY_TENSOR_BYTES}; such tensors are read
     *                     in ranges with {@link #readTensorData(GGUFTensorInfo, long, byte[], int, int)} or
     *                     {@link #readTensorData(GGUFTensorInfo, long, ByteBuffer)}
     */
    public byte[] readTensorData(GGUFTensorInfo tensor) throws IOException {
        long size = tensor.sizeInBytes();
        if (size > MAX_ARRAY_TENSOR_BYTES) {
            throw new IOException("Tensor '" + tensor.name() + "' is " + size
                    + " bytes, larger than a Java array; read it in ranges");
        }
        byte[] data = new byte[(int) size];
        readTensorData(tensor, 0L, data, 0, data.length);
        return data;
    }

    /**
     * Reads {@code length} bytes of a tensor starting {@code byteOffset} bytes into it. Works for tensors
     * of any size. Thread-safe.
     *
     * @throws IllegalArgumentException when the range lies outside the tensor or the destination
     */
    public void readTensorData(GGUFTensorInfo tensor, long byteOffset, byte[] dst, int dstOffset, int length)
            throws IOException {
        checkTensorRange(tensor, byteOffset, length);
        if (dstOffset < 0 || length > dst.length - dstOffset) {
            throw new IllegalArgumentException("destination range [" + dstOffset + ", +" + length
                    + ") exceeds array length " + dst.length);
        }
        if (!isMultiShard()) {
            readFully(dataOffset + tensor.offset() + byteOffset, ByteBuffer.wrap(dst, dstOffset, length));
        } else {
            mappedStore().readTensor(tensor.offset() + byteOffset, length, dst, dstOffset);
        }
    }

    /**
     * Fills the remaining bytes of {@code dst} with tensor data starting {@code byteOffset} bytes into the
     * tensor, advancing the buffer position. Works for tensors of any size. Thread-safe.
     *
     * @throws IllegalArgumentException when the range lies outside the tensor
     */
    public void readTensorData(GGUFTensorInfo tensor, long byteOffset, ByteBuffer dst) throws IOException {
        int length = dst.remaining();
        checkTensorRange(tensor, byteOffset, length);
        if (!isMultiShard()) {
            readFully(dataOffset + tensor.offset() + byteOffset, dst);
        } else {
            mappedStore().readTensorDirect(tensor.offset() + byteOffset, length, dst);
        }
    }

    private static void checkTensorRange(GGUFTensorInfo tensor, long byteOffset, int length) {
        long size = tensor.sizeInBytes();
        if (byteOffset < 0 || length < 0 || byteOffset > size - length) {
            throw new IllegalArgumentException("range [" + byteOffset + ", +" + length + ") lies outside tensor '"
                    + tensor.name() + "' of " + size + " bytes");
        }
    }

    /** Positional read that leaves the shared file pointer untouched. */
    private void readFully(long position, ByteBuffer dst) throws IOException {
        FileChannel channel = file.getChannel();
        long pos = position;
        while (dst.hasRemaining()) {
            int n = channel.read(dst, pos);
            if (n < 0) {
                throw new EOFException("Unexpected end of GGUF file '" + filePath + "' at byte " + pos);
            }
            pos += n;
        }
    }

    private synchronized WeightStore mappedStore() throws IOException {
        if (internalStore == null) {
            internalStore = createMappedStore();
        }
        return internalStore;
    }

    public float[] readTensorDataAsFloat(GGUFTensorInfo tensor) throws IOException {
        byte[] rawData = readTensorData(tensor);
        return dequantize(rawData, tensor.type(), tensor.numElements());
    }

    public String getFilePath() { return filePath; }

    /**
     * Total byte length of the tensor data: the sum of every split's {@link Shard#dataLength()}, which for a
     * single-file model is the end of its furthest tensor.
     */
    public long getDataLength() {
        long total = 0;
        for (Shard shard : shards) {
            total += shard.dataLength();
        }
        return total;
    }

    /**
     * Creates a {@link WeightStore} backed by this file's tensor data region.
     * Returns a {@link MultiShardMappedWeightStore} for split GGUFs,
     * or a plain {@link MappedWeightStore} for single-file GGUFs.
     */
    public WeightStore createMappedStore() throws IOException {
        if (!isMultiShard()) {
            return new MappedWeightStore(filePath, dataOffset, getDataLength());
        }
        int n = shards.size();
        String[] paths = new String[n];
        long[] fileDataOffsets = new long[n];
        long[] virtualBases = new long[n];
        long[] dataLengths = new long[n];
        for (int s = 0; s < n; s++) {
            Shard shard = shards.get(s);
            paths[s] = shard.path();
            fileDataOffsets[s] = shard.fileDataOffset();
            virtualBases[s] = shard.virtualBase();
            dataLengths[s] = shard.dataLength();
        }
        return new MultiShardMappedWeightStore(paths, fileDataOffsets, virtualBases, dataLengths);
    }

    /** Returns true if this GGUF was loaded as a multi-shard split file. */
    public boolean isMultiShard() {
        return shards.size() > 1;
    }

    /** Returns the number of shards (1 for single-file GGUFs). */
    public int getShardCount() {
        return shards.size();
    }

    /** Every file of the model in split order (one entry for a single-file model); unmodifiable. */
    public List<Shard> getShards() {
        return shards;
    }

    /** The split holding the tensor at {@code tensorIndex} in {@link #getTensors()}. */
    public Shard getShardOf(int tensorIndex) {
        if (tensorIndex < 0 || tensorIndex >= tensors.size()) {
            throw new IndexOutOfBoundsException("tensor index " + tensorIndex + " is outside [0, " + tensors.size()
                    + ")");
        }
        int lo = 0;
        int hi = shards.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (shards.get(mid).firstTensor() <= tensorIndex) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return shards.get(lo);
    }

    /**
     * Closes the file and, when one was created, the store that reads split tensor data. Every close is attempted;
     * the first failure is thrown with later ones suppressed.
     */
    public void close() throws IOException {
        IOException failure = null;
        synchronized (this) {
            if (internalStore != null) {
                try {
                    internalStore.close();
                } catch (IOException e) {
                    failure = e;
                }
                internalStore = null;
            }
        }
        try {
            file.close();
        } catch (IOException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private static long alignUp(long value, long alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    /**
     * Rejects the data-section layouts llama.cpp rejects: a tensor name given twice
     * (llama.cpp ggml/src/gguf.cpp:664-669); offsets that are not the running sum of the preceding tensors' sizes,
     * each padded to the alignment (gguf.cpp:790-820); and tensor data beyond the end of the file
     * (llama.cpp src/llama-model-loader.h:44-47).
     */
    private static void validateDataSection(String path, List<GGUFTensorInfo> infos, long alignment,
                                            long dataOffset, long fileLength) throws IOException {
        String prefix = "Invalid GGUF file '" + path + "': ";
        Set<String> names = new HashSet<>();
        long expected = 0;
        for (GGUFTensorInfo t : infos) {
            if (!names.add(t.name())) {
                throw new IOException(prefix + "duplicate tensor name '" + t.name() + "'");
            }
            if (t.offset() != expected) {
                throw new IOException(prefix + "tensor '" + t.name() + "' has data offset "
                        + Long.toUnsignedString(t.offset()) + ", expected " + expected
                        + " (tensor data must be contiguous and padded to " + alignment + " bytes)");
            }
            long bytes = t.sizeInBytes();
            if (bytes > Long.MAX_VALUE - dataOffset - expected) {
                throw new IOException(prefix + "tensor '" + t.name() + "' of " + bytes
                        + " bytes overflows the data section");
            }
            long end = dataOffset + expected + bytes;
            if (end > fileLength) {
                throw new IOException(prefix + "data of tensor '" + t.name() + "' ends at byte " + end
                        + ", beyond the end of the file (" + fileLength + " bytes)");
            }
            expected = alignUp(expected + bytes, alignment);
        }
    }

    /**
     * Rejects the tensor infos that llama.cpp rejects (llama.cpp ggml/src/gguf.cpp:686-753): a negative dimension
     * (a uint64 extent of 2^63 or more), an element count or byte size that overflows a signed 64-bit
     * value, a type without a storage layout (an id unknown to {@link GGMLType}, or the removed
     * Q4_2 / Q4_3) and a row length {@code ne[0]} that is not a whole number of blocks.
     */
    private static void validateTensorInfo(String path, GGUFTensorInfo tensor) throws IOException {
        String prefix = "Invalid GGUF file '" + path + "': tensor '" + tensor.name() + "' ";
        long[] dims = tensor.dims();
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] < 0) {
                throw new IOException(prefix + "dimension " + i + " has " + Long.toUnsignedString(dims[i])
                        + " elements, more than a signed 64-bit count");
            }
        }
        long elements = dims.length == 0 ? 1 : dims[0];
        for (int i = 1; i < dims.length; i++) {
            if (dims[i] > 1 && Long.MAX_VALUE / dims[i] < elements) {
                throw new IOException(prefix + "has shape " + Arrays.toString(dims)
                        + ", whose element count overflows a signed 64-bit value");
            }
            elements *= dims[i];
        }
        GGMLType type = tensor.type();
        if (!BlockLayout.isDefined(type)) {
            throw new IOException(prefix + "has ggml type " + type + ", which is unknown or was removed from ggml");
        }
        BlockLayout layout = BlockLayout.of(type);
        long rowElements = tensor.dim(0);
        if (rowElements % layout.blockElements() != 0) {
            throw new IOException(prefix + "of type " + type + " has " + rowElements
                    + " elements per row, not a multiple of the block size " + layout.blockElements());
        }
        if (elements / layout.blockElements() > Long.MAX_VALUE / layout.blockBytes()) {
            throw new IOException(prefix + "of type " + type + " with shape " + Arrays.toString(dims)
                    + " has a byte size that overflows a signed 64-bit value");
        }
    }

    // -----------------------------------------------------------------------
    // F32 decoding: every storage type is decoded by gguf.quant.CpuBlockDecoders
    // -----------------------------------------------------------------------

    /**
     * Decodes a whole tensor held in a heap array to F32.
     *
     * @param rawData     tensor bytes starting at index 0; may be longer than the tensor
     * @param numElements element count; a whole number of blocks of {@code type} and at most
     *                    {@code Integer.MAX_VALUE - 8}
     * @throws UnsupportedOperationException when {@code type} has no storage layout
     * @throws IllegalArgumentException      when {@code numElements} is not a whole number of blocks or does
     *                                       not fit one float array
     * @throws IndexOutOfBoundsException     when {@code rawData} is shorter than the tensor
     */
    public static float[] dequantize(byte[] rawData, GGMLType type, long numElements) {
        return CpuBlockDecoders.dequantize(rawData, type, numElements);
    }

    /**
     * Decodes {@code numElements} elements starting {@code byteOffset} bytes into {@code rawData} (for example
     * one expert's slice of a packed 3-D tensor) into a new float array. {@code rawData} starts at the
     * tensor's first block, so {@code byteOffset} must be a multiple of the block byte size; failures are
     * reported as by {@link #dequantize(byte[], GGMLType, long)}.
     *
     * @throws IllegalArgumentException when {@code byteOffset} is not on a block boundary
     */
    public static float[] dequantizeSlice(byte[] rawData, long byteOffset, GGMLType type, long numElements) {
        if (rawData == null) {
            throw new IllegalArgumentException("raw tensor data is null");
        }
        CpuBlockDecoders.Decoder decoder = CpuBlockDecoders.forType(type);
        BlockLayout layout = BlockLayout.of(type);
        if (byteOffset % layout.blockBytes() != 0) {
            throw new IllegalArgumentException("slice byte offset " + byteOffset + " is not on a " + type
                    + " block boundary (" + layout.blockBytes() + " bytes per block)");
        }
        long blocks = layout.blocksForElements(numElements);
        if (numElements > MAX_FLOAT_ARRAY_ELEMENTS) {
            throw new IllegalArgumentException(numElements + " " + type
                    + " elements do not fit one float array; decode the slice in row ranges");
        }
        float[] out = new float[(int) numElements];
        decoder.decode(ByteBuffer.wrap(rawData), byteOffset, out, 0, (int) blocks);
        return out;
    }

    /**
     * Decodes {@code result.length} Q6_K elements from the start of {@code rawData}; {@code result.length}
     * must be a multiple of {@link BlockLayout#QK_K}.
     */
    static void dequantizeQ6_K(byte[] rawData, float[] result) {
        if (rawData == null) {
            throw new IllegalArgumentException("raw tensor data is null");
        }
        CpuBlockDecoders.decodeElements(GGMLType.Q6_K, ByteBuffer.wrap(rawData), 0L, result, 0, result.length);
    }

    /** IEEE half bits (low 16 bits of {@code h}) to float, bit-exact to {@code ggml_compute_fp16_to_fp32}. */
    public static float halfToFloat(int h) {
        return FpBlockDecoders.fp16ToFp32(h);
    }

    /** MXFP4 codebook ({@code kvalues_mxfp4}, doubled E2M1 values); pairs with {@link #e8m0ToFloat(int)}. */
    static final int[] MXFP4_KVALUES = mxfp4Codebook();

    private static int[] mxfp4Codebook() {
        int[] values = new int[16];
        for (int i = 0; i < values.length; i++) {
            values[i] = QuantTables.kvalueFp4(i);
        }
        return values;
    }

    /**
     * E8M0 exponent byte (low 8 bits of {@code e}) to half of its power-of-two scale, bit-exact to
     * {@code ggml_e8m0_to_fp32_half}; multiplies the doubled {@link #MXFP4_KVALUES} codebook.
     */
    public static float e8m0ToFloat(int e) {
        return FpBlockDecoders.e8m0ToFp32Half(e);
    }

    /** MXFP4 block scale from its E8M0 byte; identical to {@link #e8m0ToFloat(int)}. */
    public static float mxfp4ScaleFromE8M0(int e) {
        return FpBlockDecoders.e8m0ToFp32Half(e);
    }
}
