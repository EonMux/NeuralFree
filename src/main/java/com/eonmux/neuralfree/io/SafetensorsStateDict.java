package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.gguf.hf.st.SafetensorsCheckpoint;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import com.eonmux.neuralfree.tensor.Shape;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * {@link Module} state dicts stored as safetensors, the container PyTorch's {@code safetensors.torch.save_file} and
 * Hugging Face's {@code save_pretrained} write.
 *
 * <p>{@link #save} writes every {@link Module#stateDict()} entry under its dotted path with its
 * {@link Module#stateInfo()} shape and an element type that keeps its dtype: FLOAT32, FLOAT16, BFLOAT16 and FLOAT64 as
 * F32, F16, BF16 and F64; INT8 to INT64 as I8 to I64; BOOL as BOOL. {@code __metadata__} always carries
 * {@code format = pt}, which the Hugging Face loaders require. The file is written by {@link SafetensorsWriter} in
 * the reference tensor order and renamed into place atomically.</p>
 *
 * <p>{@link #load} reads a single file or a checkpoint directory (sharded with an index, or plain files) through
 * {@link SafetensorsCheckpoint}, which validates every header and the index. Each stored tensor whose name is a state
 * path must have that entry's shape and a float, integer or BOOL element type; it is decoded to {@code float[]}
 * (integers rounded to the nearest float, booleans as 0 or 1) and handed to {@link Module#loadStateDict(Map, boolean)},
 * which converts the values to each tensor's dtype and loads all of them or none. {@link #read} decodes a whole
 * checkpoint for tools.</p>
 */
public final class SafetensorsStateDict {

    /** The {@code __metadata__} key naming the framework the tensors come from. */
    public static final String FORMAT_KEY = "format";
    /** The {@link #FORMAT_KEY} value of PyTorch state dicts, which this class writes. */
    public static final String FORMAT_PT = "pt";

    /** Bytes read per step when decoding a stored tensor. */
    private static final int READ_CHUNK_BYTES = 1 << 20;
    /** Largest element count of one decoded tensor: the largest Java array. */
    private static final long MAX_VALUES = Integer.MAX_VALUE - 8;
    /** Largest byte length of one tensor written: {@link SafetensorsWriter} takes each tensor as one heap buffer. */
    private static final long MAX_TENSOR_BYTES = Integer.MAX_VALUE - 8;

    /** Element types decoded to {@code float[]}; the others (FP8, FP6, FP4, C64) hold no state-dict values. */
    private static final Set<StDType> DECODABLE = Collections.unmodifiableSet(EnumSet.of(
            StDType.F32, StDType.F16, StDType.BF16, StDType.F64, StDType.BOOL,
            StDType.U8, StDType.I8, StDType.I16, StDType.U16, StDType.I32, StDType.U32, StDType.I64, StDType.U64));

    /** The element type each module dtype is stored as. */
    private static final Map<DType, StDType> STORED_TYPE;

    static {
        Map<DType, StDType> m = new EnumMap<>(DType.class);
        m.put(DType.FLOAT32, StDType.F32);
        m.put(DType.FLOAT16, StDType.F16);
        m.put(DType.BFLOAT16, StDType.BF16);
        m.put(DType.FLOAT64, StDType.F64);
        m.put(DType.INT8, StDType.I8);
        m.put(DType.INT16, StDType.I16);
        m.put(DType.INT32, StDType.I32);
        m.put(DType.INT64, StDType.I64);
        m.put(DType.BOOL, StDType.BOOL);
        STORED_TYPE = Collections.unmodifiableMap(m);
    }

    private SafetensorsStateDict() {
    }

    // ------------------------------------------------------------------ save

    /** {@link #save(Module, Path, SaveOptions)} with {@link SaveOptions#defaults()}. */
    public static void save(Module module, Path out) throws IOException {
        save(module, out, SaveOptions.defaults());
    }

    /**
     * Writes {@code module}'s state dict to the safetensors file {@code out}, creating missing parent directories. A
     * {@link SaveOptions#dtype()} stores the floating-point entries as that dtype, as {@link Module#to(DType)} would
     * convert them (buffers that keep their dtype stay as they are); integer and BOOL entries always keep theirs.
     * Values come from the {@code float[]} view of {@link Module#stateDict()}: FLOAT16 and BFLOAT16 values are
     * re-encoded exactly, FLOAT64 values are those of the float view, and integer values are rounded to the nearest
     * integer and saturated to the dtype's range.
     *
     * @throws IllegalArgumentException when an entry's encoding exceeds the largest heap buffer
     *                                  ({@code Integer.MAX_VALUE - 8} bytes) or a state path cannot be a safetensors
     *                                  tensor name
     * @throws IOException              when the file cannot be written; an existing file at {@code out} is then
     *                                  unchanged
     */
    public static void save(Module module, Path out, SaveOptions options) throws IOException {
        Objects.requireNonNull(module, "Module cannot be null");
        Objects.requireNonNull(out, "Path cannot be null");
        Objects.requireNonNull(options, "Save options cannot be null");

        Map<String, Module.StateInfo> info = module.stateInfo();
        Map<String, Object> state = module.stateDict();
        if (!info.keySet().equals(state.keySet())) {
            throw new IllegalStateException("The state of " + module.moduleName() + " changed while it was saved: "
                + info.keySet() + " vs " + state.keySet());
        }

        SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        for (Map.Entry<String, Module.StateInfo> entry : info.entrySet()) {
            String key = entry.getKey();
            Module.StateInfo stateInfo = entry.getValue();
            float[] values = (float[]) state.get(key);
            if (values.length != stateInfo.shape().numElements()) {
                throw new IllegalStateException("State entry '" + key + "' of " + module.moduleName() + " holds "
                    + values.length + " values for " + stateInfo.shape());
            }
            DType dtype = storedDtype(stateInfo, options.dtype());
            long bytes = (long) values.length * dtype.byteSize();
            if (bytes > MAX_TENSOR_BYTES) {
                throw new IllegalArgumentException("State entry '" + key + "' of " + module.moduleName() + " needs "
                    + bytes + " bytes as " + dtype + ", more than the " + MAX_TENSOR_BYTES
                    + " bytes one tensor buffer of the safetensors writer holds");
            }
            builder.add(key, STORED_TYPE.get(dtype), dims(stateInfo.shape()), () -> encode(values, dtype));
        }
        builder.metadata(options.metadata());
        builder.metadata(FORMAT_KEY, FORMAT_PT);

        Path parent = out.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        builder.build().write(out);
    }

    /** The dtype an entry is stored as: {@code override} for a convertible floating-point entry, else its own. */
    private static DType storedDtype(Module.StateInfo info, DType override) {
        DType own = info.dtype();
        if (override == null || !own.isFloatingPoint() || info.keepsDtype()) {
            return own;
        }
        return override;
    }

    /** {@code values} encoded as little-endian elements of {@code dtype}, positioned at 0. */
    private static ByteBuffer encode(float[] values, DType dtype) {
        int size = dtype.byteSize();
        ByteBuffer out = ByteBuffer.allocate(values.length * size).order(ByteOrder.LITTLE_ENDIAN);
        switch (dtype) {
            case FLOAT32:
                out.asFloatBuffer().put(values);
                break;
            case FLOAT16:
                for (int i = 0; i < values.length; i++) {
                    out.putShort(i * size, (short) HalfFloats.floatToHalf(values[i]));
                }
                break;
            case BFLOAT16:
                for (int i = 0; i < values.length; i++) {
                    out.putShort(i * size, (short) HalfFloats.floatToBFloat16(values[i]));
                }
                break;
            case FLOAT64:
                for (int i = 0; i < values.length; i++) {
                    out.putDouble(i * size, values[i]);
                }
                break;
            case BOOL:
                for (int i = 0; i < values.length; i++) {
                    out.put(i, (byte) (values[i] != 0f ? 1 : 0));
                }
                break;
            case INT8:
                for (int i = 0; i < values.length; i++) {
                    out.put(i, (byte) saturatedInteger(values[i], Byte.MIN_VALUE, Byte.MAX_VALUE));
                }
                break;
            case INT16:
                for (int i = 0; i < values.length; i++) {
                    out.putShort(i * size, (short) saturatedInteger(values[i], Short.MIN_VALUE, Short.MAX_VALUE));
                }
                break;
            case INT32:
                for (int i = 0; i < values.length; i++) {
                    out.putInt(i * size, (int) saturatedInteger(values[i], Integer.MIN_VALUE, Integer.MAX_VALUE));
                }
                break;
            case INT64:
                for (int i = 0; i < values.length; i++) {
                    out.putLong(i * size, saturatedInteger(values[i], Long.MIN_VALUE, Long.MAX_VALUE));
                }
                break;
            default:
                throw new IllegalStateException("No safetensors encoding for dtype " + dtype);
        }
        return out;
    }

    /** {@code value} rounded to the nearest integer (ties to even) and clamped to {@code [min, max]}; NaN is 0. */
    private static long saturatedInteger(float value, long min, long max) {
        long rounded = (long) Math.rint(value);
        return Math.max(min, Math.min(max, rounded));
    }

    // ------------------------------------------------------------------ load and read

    /**
     * Loads the safetensors file or checkpoint directory {@code fileOrDir} into {@code module}. Stored tensors named
     * like a state entry are checked (shape and element type) before any is decoded, and loaded together by
     * {@link Module#loadStateDict(Map, boolean)}, so a refused load changes nothing. In strict mode the entries the
     * checkpoint lacks and the stored tensors that name no state are reported by {@link Module#loadStateDict}; the
     * latter are passed by name only, without decoding them. Otherwise both are ignored.
     *
     * @throws java.io.FileNotFoundException when {@code fileOrDir} does not exist
     * @throws IOException                   when the checkpoint cannot be read or is invalid
     * @throws IllegalArgumentException      when a stored state tensor has another shape than the entry, an element
     *                                       type without {@code float[]} values, or, in strict mode, when the keys
     *                                       do not match
     * @throws UnsupportedOperationException when a stored tensor names an INT8, INT16 or INT64 parameter, which
     *                                       {@link Module#loadStateDict(Map, boolean)} cannot load
     */
    public static void load(Module module, Path fileOrDir, boolean strict) throws IOException {
        Objects.requireNonNull(module, "Module cannot be null");
        Objects.requireNonNull(fileOrDir, "Path cannot be null");

        Map<String, Module.StateInfo> info = module.stateInfo();
        try (SafetensorsCheckpoint checkpoint = openCheckpoint(fileOrDir)) {
            Map<String, SafetensorsFile> stored = owners(checkpoint);
            List<String> matched = new ArrayList<>();
            for (Map.Entry<String, Module.StateInfo> entry : info.entrySet()) {
                SafetensorsFile file = stored.get(entry.getKey());
                if (file == null) {
                    continue;
                }
                StTensor tensor = file.tensor(entry.getKey());
                requireDecodable(file, tensor);
                requireShape(module, file, tensor, entry.getValue().shape());
                matched.add(entry.getKey());
            }

            Map<String, Object> state = new LinkedHashMap<>();
            for (String key : matched) {
                SafetensorsFile file = stored.get(key);
                state.put(key, decode(file, file.tensor(key)));
            }
            if (strict) {
                for (Map.Entry<String, SafetensorsFile> entry : stored.entrySet()) {
                    if (!info.containsKey(entry.getKey())) {
                        state.put(entry.getKey(), entry.getValue().tensor(entry.getKey()));
                    }
                }
            }
            module.loadStateDict(state, strict);
        }
    }

    /**
     * Decodes every tensor of the safetensors file or checkpoint directory {@code fileOrDir} to {@code float[]}, as
     * {@link #load} decodes state tensors.
     *
     * @throws java.io.FileNotFoundException when {@code fileOrDir} does not exist
     * @throws IOException                   when the checkpoint cannot be read or is invalid
     * @throws IllegalArgumentException      when a tensor has an element type without {@code float[]} values or more
     *                                       elements than a Java array holds
     */
    public static Contents read(Path fileOrDir) throws IOException {
        Objects.requireNonNull(fileOrDir, "Path cannot be null");
        try (SafetensorsCheckpoint checkpoint = openCheckpoint(fileOrDir)) {
            Map<String, SafetensorsFile> stored = owners(checkpoint);
            for (Map.Entry<String, SafetensorsFile> entry : stored.entrySet()) {
                requireDecodable(entry.getValue(), entry.getValue().tensor(entry.getKey()));
            }
            Map<String, float[]> values = new LinkedHashMap<>();
            Map<String, StTensor> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, SafetensorsFile> entry : stored.entrySet()) {
                StTensor tensor = entry.getValue().tensor(entry.getKey());
                values.put(entry.getKey(), decode(entry.getValue(), tensor));
                tensors.put(entry.getKey(), tensor);
            }
            Map<String, String> metadata = new LinkedHashMap<>();
            for (SafetensorsFile file : checkpoint.files()) {
                for (Map.Entry<String, String> m : file.metadata().entrySet()) {
                    metadata.putIfAbsent(m.getKey(), m.getValue());
                }
            }
            return new Contents(values, tensors, metadata);
        }
    }

    /** Opens a file, with its directory as the model directory, or a checkpoint directory. */
    private static SafetensorsCheckpoint openCheckpoint(Path fileOrDir) throws IOException {
        Path absolute = fileOrDir.toAbsolutePath();
        Path modelDir = Files.isDirectory(absolute) || absolute.getParent() == null ? absolute : absolute.getParent();
        return SafetensorsCheckpoint.open(fileOrDir, modelDir, false);
    }

    /** The file holding each tensor of {@code checkpoint}, in checkpoint order. */
    private static Map<String, SafetensorsFile> owners(SafetensorsCheckpoint checkpoint) {
        Map<String, SafetensorsFile> owners = new LinkedHashMap<>();
        for (SafetensorsFile file : checkpoint.files()) {
            for (StTensor tensor : file.tensors()) {
                owners.put(tensor.name(), file);
            }
        }
        return owners;
    }

    private static void requireDecodable(SafetensorsFile file, StTensor tensor) {
        if (!DECODABLE.contains(tensor.dtype())) {
            throw new IllegalArgumentException("Tensor '" + tensor.name() + "' in '" + file.path() + "' has dtype "
                + tensor.dtype().headerName() + ", which holds no state-dict values; supported dtypes are F32, F16,"
                + " BF16, F64, BOOL and the integer dtypes");
        }
        if (tensor.numel() > MAX_VALUES) {
            throw new IllegalArgumentException("Tensor '" + tensor.name() + "' in '" + file.path() + "' has "
                + tensor.numel() + " elements, more than the " + MAX_VALUES + " a float[] holds");
        }
    }

    private static void requireShape(Module module, SafetensorsFile file, StTensor tensor, Shape expected) {
        long[] want = dims(expected);
        if (!Arrays.equals(want, tensor.shape())) {
            throw new IllegalArgumentException("Tensor '" + tensor.name() + "' in '" + file.path() + "' has shape "
                + Arrays.toString(tensor.shape()) + ", but the state of " + module.moduleName() + " has shape "
                + Arrays.toString(want));
        }
    }

    private static long[] dims(Shape shape) {
        int[] dims = shape.dims();
        long[] out = new long[dims.length];
        for (int i = 0; i < dims.length; i++) {
            out[i] = dims[i];
        }
        return out;
    }

    /** The values of {@code tensor}, a decodable tensor of {@code file}, read in chunks. */
    private static float[] decode(SafetensorsFile file, StTensor tensor) throws IOException {
        int n = (int) tensor.numel();
        int size = tensor.dtype().bits() / Byte.SIZE;
        int chunk = READ_CHUNK_BYTES / size;
        float[] values = new float[n];
        ByteBuffer buffer = ByteBuffer.allocate(Math.min(chunk, n) * size).order(ByteOrder.LITTLE_ENDIAN);
        for (int first = 0; first < n; first += chunk) {
            int count = Math.min(chunk, n - first);
            buffer.clear();
            buffer.limit(count * size);
            file.read(tensor, (long) first * size, buffer);
            buffer.flip();
            decodeChunk(tensor.dtype(), buffer, values, first, count);
        }
        return values;
    }

    /** Decodes {@code count} little-endian elements of {@code dtype} from {@code in} into {@code out[first..]}. */
    private static void decodeChunk(StDType dtype, ByteBuffer in, float[] out, int first, int count) {
        for (int i = 0; i < count; i++) {
            out[first + i] = element(dtype, in, i);
        }
    }

    /** Element {@code i} of the little-endian {@code in} as a float. */
    private static float element(StDType dtype, ByteBuffer in, int i) {
        switch (dtype) {
            case F32: return in.getFloat(i * Float.BYTES);
            case F16: return HalfFloats.halfToFloat(in.getShort(i * Short.BYTES) & 0xffff);
            case BF16: return HalfFloats.bfloat16ToFloat(in.getShort(i * Short.BYTES) & 0xffff);
            case F64: return (float) in.getDouble(i * Double.BYTES);
            case BOOL: return in.get(i) != 0 ? 1f : 0f;
            case U8: return in.get(i) & 0xff;
            case I8: return in.get(i);
            case I16: return in.getShort(i * Short.BYTES);
            case U16: return in.getShort(i * Short.BYTES) & 0xffff;
            case I32: return in.getInt(i * Integer.BYTES);
            case U32: return in.getInt(i * Integer.BYTES) & 0xffffffffL;
            case I64: return in.getLong(i * Long.BYTES);
            case U64: return unsignedToFloat(in.getLong(i * Long.BYTES));
            default: throw new IllegalStateException("No float decoding for safetensors dtype " + dtype);
        }
    }

    /** The unsigned 64-bit {@code v} rounded to the nearest float. */
    private static float unsignedToFloat(long v) {
        if (v >= 0) {
            return v;
        }
        return ((v >>> 1) | (v & 1)) * 2f;
    }

    // ------------------------------------------------------------------ value types

    /**
     * How {@link #save(Module, Path, SaveOptions)} stores a state dict: an optional floating-point dtype for the
     * float entries (null keeps each entry's own) and string metadata for {@code __metadata__}, to which
     * {@code format = pt} is always added. Immutable.
     */
    public static final class SaveOptions {
        private static final SaveOptions DEFAULTS = new SaveOptions(null, Collections.emptyMap());

        private final DType dtype;
        private final Map<String, String> metadata;

        private SaveOptions(DType dtype, Map<String, String> metadata) {
            this.dtype = dtype;
            this.metadata = metadata;
        }

        /** Each entry at its own dtype, no metadata beyond {@code format = pt}. */
        public static SaveOptions defaults() {
            return DEFAULTS;
        }

        /**
         * Options storing float entries as {@code dtype} (null keeps their own) with {@code metadata} (copied).
         *
         * @throws IllegalArgumentException when {@code dtype} is not FLOAT32, FLOAT16 or BFLOAT16, a metadata key or
         *                                  value is null, or {@value SafetensorsStateDict#FORMAT_KEY} is set to
         *                                  another value than {@value SafetensorsStateDict#FORMAT_PT}
         */
        public static SaveOptions of(DType dtype, Map<String, String> metadata) {
            Objects.requireNonNull(metadata, "Metadata cannot be null");
            if (dtype != null && dtype != DType.FLOAT32 && dtype != DType.FLOAT16 && dtype != DType.BFLOAT16) {
                throw new IllegalArgumentException("A safetensors state dict stores float entries as FLOAT32, FLOAT16"
                    + " or BFLOAT16, not " + dtype);
            }
            Map<String, String> copy = new LinkedHashMap<>();
            for (Map.Entry<String, String> e : metadata.entrySet()) {
                if (e.getKey() == null || e.getValue() == null) {
                    throw new IllegalArgumentException("Metadata entries need a key and a value, got " + e.getKey()
                        + " = " + e.getValue());
                }
                copy.put(e.getKey(), e.getValue());
            }
            String format = copy.get(FORMAT_KEY);
            if (format != null && !format.equals(FORMAT_PT)) {
                throw new IllegalArgumentException("Metadata '" + FORMAT_KEY + "' of a state dict is '" + FORMAT_PT
                    + "', not '" + format + "'");
            }
            return new SaveOptions(dtype, Collections.unmodifiableMap(copy));
        }

        /** These options with float entries stored as {@code dtype} (null keeps their own). */
        public SaveOptions withDtype(DType dtype) {
            return of(dtype, metadata);
        }

        /** These options with {@code metadata} in place of the current metadata. */
        public SaveOptions withMetadata(Map<String, String> metadata) {
            return of(dtype, metadata);
        }

        /** The dtype float entries are stored as, or null for each entry's own. */
        public DType dtype() {
            return dtype;
        }

        /** The metadata written besides {@code format = pt} (unmodifiable). */
        public Map<String, String> metadata() {
            return metadata;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof SaveOptions)) {
                return false;
            }
            SaveOptions other = (SaveOptions) o;
            return dtype == other.dtype && metadata.equals(other.metadata);
        }

        @Override
        public int hashCode() {
            return Objects.hash(dtype, metadata);
        }

        @Override
        public String toString() {
            return "SaveOptions{dtype=" + (dtype == null ? "own" : dtype) + ", metadata=" + metadata + "}";
        }
    }

    /**
     * The decoded tensors of a checkpoint ({@link #read}), in checkpoint order: files in natural order, tensors in
     * name order within each. The value arrays were decoded for this read and belong to the caller; the rest is
     * immutable.
     */
    public static final class Contents {
        private final Map<String, float[]> values;
        private final Map<String, StTensor> tensors;
        private final Map<String, String> metadata;

        private Contents(Map<String, float[]> values, Map<String, StTensor> tensors, Map<String, String> metadata) {
            this.values = Collections.unmodifiableMap(values);
            this.tensors = Collections.unmodifiableMap(tensors);
            this.metadata = Collections.unmodifiableMap(metadata);
        }

        /** Every tensor's row-major values, keyed by name (unmodifiable). */
        public Map<String, float[]> values() {
            return values;
        }

        /** Every tensor's shape, outermost dimension first, keyed by name (a new map of copies). */
        public Map<String, long[]> shapes() {
            Map<String, long[]> shapes = new LinkedHashMap<>();
            for (Map.Entry<String, StTensor> e : tensors.entrySet()) {
                shapes.put(e.getKey(), e.getValue().shape());
            }
            return Collections.unmodifiableMap(shapes);
        }

        /**
         * The shape of {@code name} (a copy).
         *
         * @throws IllegalArgumentException when the checkpoint has no such tensor
         */
        public long[] shape(String name) {
            return entry(name).shape();
        }

        /**
         * The element type {@code name} is stored as.
         *
         * @throws IllegalArgumentException when the checkpoint has no such tensor
         */
        public StDType dtype(String name) {
            return entry(name).dtype();
        }

        /** The {@code __metadata__} of every file; a key several files set keeps the first file's value. */
        public Map<String, String> metadata() {
            return metadata;
        }

        /** A new, modifiable state dict of the values, for {@link Module#loadStateDict(Map, boolean)}. */
        public Map<String, Object> toStateDict() {
            return new LinkedHashMap<>(values);
        }

        /** The number of tensors. */
        public int size() {
            return values.size();
        }

        private StTensor entry(String name) {
            StTensor tensor = tensors.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("No tensor named '" + name + "'");
            }
            return tensor;
        }

        @Override
        public String toString() {
            return "SafetensorsStateDict.Contents{" + values.size() + " tensors}";
        }
    }
}
