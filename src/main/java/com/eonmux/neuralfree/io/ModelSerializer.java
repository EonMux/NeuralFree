package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Saves a model, and optionally its optimizer, to a zip checkpoint and loads it back. {@code model_state.bin} holds
 * the model's {@link Module#stateDict()} (every parameter and persistent buffer, such as BatchNorm running statistics)
 * as little-endian FLOAT32 records with their shapes; {@code optimizer_state.bin} holds the optimizer's
 * {@link Optimizer#stateDict()} (format 2: every record starts with a kind, so state tensors keep their shapes and
 * the step count and hyper-parameters keep their integer, floating-point or boolean values; format-1 files, whose
 * records are all FLOAT32 tensors, still load); {@code model_meta.json} describes the parameters for
 * {@link #inspect(Path)}. {@link #load(Path, Module, Optimizer, ComputeBackend)} refuses a file that is not such an
 * archive, a corrupt record, and model state that lacks a parameter of the model, names state the model does not
 * have, or holds another shape; buffers may be absent, for checkpoints written before buffers were saved. The
 * optimizer state is restored with {@link Optimizer#loadStateDict(Map)}, which validates it.
 */
public final class ModelSerializer {
    private static final String MODEL_STATE_ENTRY = "model_state.bin";
    private static final String MODEL_META_ENTRY = "model_meta.json";
    private static final String OPTIMIZER_STATE_ENTRY = "optimizer_state.bin";
    private static final String TRAINING_STATE_ENTRY = "training_state.json";
    private static final int FORMAT_VERSION = 1;
    /** Format of {@code optimizer_state.bin}: 1 holds FLOAT32 tensor records only, 2 kind-tagged records. */
    private static final int OPTIMIZER_FORMAT_VERSION = 2;
    private static final int RECORD_TENSOR = 0;
    private static final int RECORD_INTEGER = 1;
    private static final int RECORD_REAL = 2;
    private static final int RECORD_BOOLEAN = 3;
    /** Upper bound on the UTF-8 name length of a record, refusing corrupt lengths early. */
    private static final int MAX_NAME_BYTES = 1 << 16;
    /** Upper bound on the rank of a tensor record. */
    private static final int MAX_RANK = 64;
    /** Upper bound on the size of a metadata entry ({@code model_meta.json}, {@code training_state.json}). */
    private static final int MAX_TEXT_ENTRY_BYTES = 64 << 20;
    /** Bytes read or written per step when streaming tensor values. */
    private static final int STREAM_CHUNK_BYTES = 1 << 16;
    /** Initial capacity, in floats, of the values of a tensor record being read. */
    private static final int INITIAL_FLOAT_CAPACITY = 1 << 20;

    private ModelSerializer() {
    }

    public static void save(Module model, String path) throws IOException {
        save(model, null, path);
    }

    public static void save(Module model, Path path) throws IOException {
        save(model, null, path);
    }

    public static void save(Module model, Optimizer optimizer, String path) throws IOException {
        save(model, optimizer, Paths.get(path));
    }

    public static void save(Module model, Optimizer optimizer, Path path) throws IOException {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(path, "Path cannot be null");

        Path parentDir = path.getParent();
        if (parentDir != null && !Files.exists(parentDir)) {
            Files.createDirectories(parentDir);
        }

        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(path.toFile()))) {
            saveModelState(model, zos);
            saveModelMeta(model, zos);
            
            if (optimizer != null) {
                saveOptimizerState(optimizer, zos);
            }
            saveTrainingState(optimizer, zos);
        }
    }

    /**
     * Writes the model's {@link Module#stateDict()}: every parameter and every persistent buffer, keyed by dotted
     * path, each with its shape. Values cross through the host.
     */
    private static void saveModelState(Module model, ZipOutputStream zos) throws IOException {
        zos.putNextEntry(new ZipEntry(MODEL_STATE_ENTRY));

        Map<String, Object> state = model.stateDict();
        Map<String, int[]> shapes = stateShapes(model);

        ByteBuffer headerBuffer = ByteBuffer.allocate(8);
        headerBuffer.order(ByteOrder.LITTLE_ENDIAN);
        headerBuffer.putInt(FORMAT_VERSION);
        headerBuffer.putInt(state.size());
        zos.write(headerBuffer.array());

        for (Map.Entry<String, Object> entry : state.entrySet()) {
            int[] shape = shapes.get(entry.getKey());
            if (shape == null) {
                throw new IllegalStateException("State entry '" + entry.getKey() + "' of " + model.moduleName()
                    + " is neither a parameter nor a buffer");
            }
            writeEntry(zos, entry.getKey(), shape, (float[]) entry.getValue());
        }

        zos.closeEntry();
    }

    /** The shape of every parameter and buffer of {@code model}, keyed by the dotted path {@code stateDict} uses. */
    private static Map<String, int[]> stateShapes(Module model) {
        Map<String, int[]> shapes = new HashMap<>();
        for (Map.Entry<String, Parameter> entry : model.namedParameters().entrySet()) {
            shapes.put(entry.getKey(), entry.getValue().shape().dims());
        }
        for (Map.Entry<String, Tensor> entry : model.namedBuffers().entrySet()) {
            shapes.put(entry.getKey(), entry.getValue().shape().dims());
        }
        return shapes;
    }

    /**
     * Writes one tensor record: name length, rank, element count, UTF-8 name, dims, then little-endian floats, the
     * values in chunks.
     *
     * @throws IllegalStateException when the name or the rank exceeds what {@link #readEntry} accepts
     */
    private static void writeEntry(OutputStream out, String name, int[] shape, float[] data) throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        if (nameBytes.length > MAX_NAME_BYTES || shape.length > MAX_RANK) {
            throw new IllegalStateException("State entry '" + name + "' has a " + nameBytes.length
                + "-byte name and rank " + shape.length + "; a checkpoint holds names up to " + MAX_NAME_BYTES
                + " bytes and ranks up to " + MAX_RANK);
        }

        ByteBuffer entryHeader = ByteBuffer.allocate(12 + nameBytes.length + shape.length * Integer.BYTES);
        entryHeader.order(ByteOrder.LITTLE_ENDIAN);
        entryHeader.putInt(nameBytes.length);
        entryHeader.putInt(shape.length);
        entryHeader.putInt(data.length);
        entryHeader.put(nameBytes);
        for (int dim : shape) {
            entryHeader.putInt(dim);
        }
        out.write(entryHeader.array());

        ByteBuffer chunk = ByteBuffer.allocate(STREAM_CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        int perChunk = STREAM_CHUNK_BYTES / Float.BYTES;
        for (int offset = 0; offset < data.length; offset += perChunk) {
            int n = Math.min(perChunk, data.length - offset);
            chunk.clear();
            chunk.asFloatBuffer().put(data, offset, n);
            out.write(chunk.array(), 0, n * Float.BYTES);
        }
    }

    private static void saveModelMeta(Module model, ZipOutputStream zos) throws IOException {
        zos.putNextEntry(new ZipEntry(MODEL_META_ENTRY));

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"format_version\": ").append(FORMAT_VERSION).append(",\n");
        json.append("  \"module_type\": \"").append(escapeJson(model.moduleName())).append("\",\n");
        json.append("  \"parameter_count\": ").append(model.parameterCount()).append(",\n");
        json.append("  \"num_parameters\": ").append(model.parameters().size()).append(",\n");
        json.append("  \"timestamp\": \"").append(new Date().toInstant().toString()).append("\",\n");
        json.append("  \"parameters\": {\n");
        
        Map<String, Parameter> namedParams = model.namedParameters();
        int i = 0;
        for (Map.Entry<String, Parameter> entry : namedParams.entrySet()) {
            String name = entry.getKey();
            Parameter param = entry.getValue();
            
            json.append("    \"").append(escapeJson(name)).append("\": {\n");
            json.append("      \"shape\": [");
            int[] shape = param.shape().dims();
            for (int j = 0; j < shape.length; j++) {
                json.append(shape[j]);
                if (j < shape.length - 1) json.append(", ");
            }
            json.append("],\n");
            json.append("      \"dtype\": \"").append(param.dtype()).append("\",\n");
            json.append("      \"num_elements\": ").append(param.numElements()).append("\n");
            json.append("    }");
            if (i < namedParams.size() - 1) json.append(",");
            json.append("\n");
            i++;
        }
        
        json.append("  }\n");
        json.append("}\n");
        
        zos.write(json.toString().getBytes(StandardCharsets.UTF_8));
        zos.closeEntry();
    }

    /**
     * Writes the optimizer's {@link Optimizer#stateDict()} in format {@value #OPTIMIZER_FORMAT_VERSION}: a header
     * (version, entry count), then per entry a kind ({@code RECORD_*}) followed by a tensor record
     * ({@link #writeEntry}) or a scalar record ({@link #writeScalarRecord}). The state dict's tensors are read to the
     * host and closed before this returns.
     */
    private static void saveOptimizerState(Optimizer optimizer, ZipOutputStream zos) throws IOException {
        Map<String, Object> stateDict = optimizer.stateDict();
        try {
            zos.putNextEntry(new ZipEntry(OPTIMIZER_STATE_ENTRY));
            ByteBuffer headerBuffer = ByteBuffer.allocate(8);
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN);
            headerBuffer.putInt(OPTIMIZER_FORMAT_VERSION);
            headerBuffer.putInt(stateDict.size());
            zos.write(headerBuffer.array());

            for (Map.Entry<String, Object> entry : stateDict.entrySet()) {
                writeOptimizerRecord(zos, optimizer, entry.getKey(), entry.getValue());
            }
            zos.closeEntry();
        } catch (IOException | RuntimeException e) {
            closeStateDictAfter(stateDict, e);
            throw e;
        }
        Optimizer.closeStateDict(stateDict);
    }

    /**
     * Writes one optimizer state entry with its kind: a FLOAT32 tensor as a tensor record, a {@link Float} or
     * {@link Double} as a real, a {@link Byte}, {@link Short}, {@link Integer} or {@link Long} as an integer, and a
     * {@link Boolean}.
     *
     * @throws IllegalStateException when the optimizer returned a value of another type
     */
    private static void writeOptimizerRecord(OutputStream out, Optimizer optimizer, String name, Object value)
            throws IOException {
        if (value instanceof Tensor) {
            Tensor tensor = (Tensor) value;
            writeInt(out, RECORD_TENSOR);
            writeEntry(out, name, tensor.shape().dims(), tensor.toFloatArray());
        } else if (value instanceof Float || value instanceof Double) {
            writeScalarRecord(out, RECORD_REAL, name, Double.doubleToRawLongBits(((Number) value).doubleValue()));
        } else if (value instanceof Byte || value instanceof Short || value instanceof Integer
                || value instanceof Long) {
            writeScalarRecord(out, RECORD_INTEGER, name, ((Number) value).longValue());
        } else if (value instanceof Boolean) {
            writeScalarRecord(out, RECORD_BOOLEAN, name, ((Boolean) value) ? 1L : 0L);
        } else {
            throw new IllegalStateException(optimizer.getClass().getSimpleName() + " state entry '" + name
                + "' has type " + (value == null ? "null" : value.getClass().getName())
                + ", which an optimizer checkpoint cannot hold");
        }
    }

    /** Writes a scalar record: kind, UTF-8 name length, name, then the 8-byte little-endian payload. */
    private static void writeScalarRecord(OutputStream out, int kind, String name, long payload)
            throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        if (nameBytes.length > MAX_NAME_BYTES) {
            throw new IllegalStateException("Optimizer state name of " + nameBytes.length + " bytes exceeds "
                + MAX_NAME_BYTES);
        }
        ByteBuffer record = ByteBuffer.allocate(16 + nameBytes.length).order(ByteOrder.LITTLE_ENDIAN);
        record.putInt(kind);
        record.putInt(nameBytes.length);
        record.put(nameBytes);
        record.putLong(payload);
        out.write(record.array());
    }

    private static void writeInt(OutputStream out, int value) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    /** Closes the tensors of {@code state} after {@code failure}, recording a close failure on it. */
    private static void closeStateDictAfter(Map<String, ?> state, Exception failure) {
        try {
            Optimizer.closeStateDict(state);
        } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
        }
    }

    private static void saveTrainingState(Optimizer optimizer, ZipOutputStream zos) throws IOException {
        zos.putNextEntry(new ZipEntry(TRAINING_STATE_ENTRY));

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"format_version\": ").append(FORMAT_VERSION).append(",\n");
        json.append("  \"timestamp\": \"").append(new Date().toInstant().toString()).append("\",\n");
        
        if (optimizer != null) {
            json.append("  \"optimizer_type\": \"").append(escapeJson(optimizer.getClass().getSimpleName())).append("\",\n");
            json.append("  \"step_count\": ").append(optimizer.stepCount()).append(",\n");
            json.append("  \"learning_rate\": ").append(optimizer.getLearningRate()).append("\n");
        } else {
            json.append("  \"optimizer_type\": null\n");
        }
        
        json.append("}\n");
        
        zos.write(json.toString().getBytes(StandardCharsets.UTF_8));
        zos.closeEntry();
    }

    public static void load(String path, Module model, ComputeBackend backend) throws IOException {
        load(Paths.get(path), model, null, backend);
    }

    public static void load(Path path, Module model, ComputeBackend backend) throws IOException {
        load(path, model, null, backend);
    }

    public static void load(String path, Module model, Optimizer optimizer, ComputeBackend backend) throws IOException {
        load(Paths.get(path), model, optimizer, backend);
    }

    /**
     * Loads a checkpoint written by {@link #save}. The model state must be present and must hold every parameter of
     * {@code model} with the model's shapes and no entry that names none of its parameters or buffers; buffers may be
     * absent (checkpoints written before buffers were saved keep their buffers' current values). When
     * {@code optimizer} is given the checkpoint must hold optimizer state, which is restored together with the step
     * count of the training metadata. Every entry is read and checked first and the optimizer refuses state that does
     * not fit it before anything is written, so a refused checkpoint leaves the model and the optimizer unchanged.
     *
     * @throws FileNotFoundException when {@code path} does not exist
     * @throws IOException           when the file is not a checkpoint archive, lacks the model state (or the optimizer
     *                               state when {@code optimizer} is given), holds a corrupt record, or does not fit
     *                               the model or the optimizer
     */
    public static void load(Path path, Module model, Optimizer optimizer, ComputeBackend backend) throws IOException {
        Objects.requireNonNull(path, "Path cannot be null");
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(backend, "Backend cannot be null");

        try (ZipFile zip = openCheckpoint(path)) {
            Map<String, Object> modelState;
            try (EntryStream in = openEntry(zip, path, MODEL_STATE_ENTRY)) {
                modelState = readModelState(in, model);
            }
            if (optimizer != null) {
                int trainingStep = readTrainingStep(zip);
                try (EntryStream in = openEntry(zip, path, OPTIMIZER_STATE_ENTRY)) {
                    loadOptimizerState(in, optimizer, backend, trainingStep);
                }
            }
            model.loadStateDict(modelState);
        }
    }

    /**
     * Opens {@code path} as a checkpoint archive.
     *
     * @throws FileNotFoundException when {@code path} does not exist
     * @throws IOException           when it is not a readable zip archive
     */
    private static ZipFile openCheckpoint(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Checkpoint file not found: " + path);
        }
        try {
            return new ZipFile(path.toFile());
        } catch (ZipException e) {
            throw new IOException("Not a NeuralFree checkpoint (not a zip archive): " + path + ": " + e.getMessage(),
                e);
        }
    }

    /**
     * The data of the entry {@code name} of {@code zip}, ending at the size the archive records for it.
     *
     * @throws IOException when the archive has no such entry
     */
    private static EntryStream openEntry(ZipFile zip, Path path, String name) throws IOException {
        ZipEntry entry = zip.getEntry(name);
        if (entry == null) {
            throw new IOException("Not a NeuralFree checkpoint, or one without this state: " + path + " has no "
                + name);
        }
        return new EntryStream(zip.getInputStream(entry), entry.getSize());
    }

    /**
     * Reads the model state written by {@link #saveModelState} for {@link Module#loadStateDict(Map)}. Refuses an
     * entry whose shape differs from that of the model's parameter or buffer of that name, an entry naming none of
     * them, an entry listed twice, and a state missing any of the model's parameters.
     */
    private static Map<String, Object> readModelState(EntryStream in, Module model) throws IOException {
        int numEntries = readHeader(in, "checkpoint");
        Map<String, int[]> shapes = stateShapes(model);
        Map<String, Object> stateDict = new LinkedHashMap<>();
        List<String> unexpected = new ArrayList<>();
        for (int i = 0; i < numEntries; i++) {
            SerializedTensor entry = readEntry(in);
            int[] expected = shapes.get(entry.name);
            if (expected == null) {
                unexpected.add(entry.name);
                continue;
            }
            if (!Arrays.equals(expected, entry.shape)) {
                throw new IOException("Checkpoint entry '" + entry.name + "' has shape " + Arrays.toString(entry.shape)
                    + ", but the model's is " + Arrays.toString(expected));
            }
            if (stateDict.put(entry.name, entry.data) != null) {
                throw new IOException("Corrupt checkpoint: entry '" + entry.name + "' appears twice");
            }
        }
        List<String> missing = new ArrayList<>();
        for (String parameter : model.namedParameters().keySet()) {
            if (!stateDict.containsKey(parameter)) {
                missing.add(parameter);
            }
        }
        if (!missing.isEmpty() || !unexpected.isEmpty()) {
            throw new IOException("Checkpoint does not fit " + model.moduleName() + ": missing parameters " + missing
                + ", entries the model does not have " + unexpected);
        }
        return stateDict;
    }

    /**
     * Reads the optimizer state written by {@link #saveOptimizerState} (format 1 or 2): tensor records become
     * tensors on {@code backend}, scalar records {@link Long}, {@link Double} or {@link Boolean} values. A positive
     * {@code trainingStep} (the step count of the training metadata) replaces the state's own step count. Loads them
     * with {@link Optimizer#loadStateDict(Map)}, which validates them before copying anything, and closes the
     * tensors.
     *
     * @throws IOException when the entry is corrupt or does not fit the optimizer (a state buffer of another shape,
     *                     or one the optimizer does not keep)
     */
    private static void loadOptimizerState(EntryStream in, Optimizer optimizer, ComputeBackend backend,
                                           int trainingStep) throws IOException {
        StateHeader header = readStateHeader(in, "optimizer state", OPTIMIZER_FORMAT_VERSION);
        Map<String, Object> stateDict = new LinkedHashMap<>();
        try {
            for (int i = 0; i < header.count; i++) {
                int kind = header.version < 2 ? RECORD_TENSOR : readInt(in);
                if (kind < RECORD_TENSOR || kind > RECORD_BOOLEAN) {
                    throw new IOException("Corrupt optimizer state: record " + i + " has unknown kind " + kind);
                }
                String name;
                Object value;
                if (kind == RECORD_TENSOR) {
                    SerializedTensor entry = readEntry(in);
                    name = entry.name;
                    value = optimizerTensor(entry, backend);
                } else {
                    name = readScalarName(in);
                    value = scalarValue(kind, name, readLong(in));
                }
                if (stateDict.containsKey(name)) {
                    if (value instanceof Tensor) {
                        ((Tensor) value).close();
                    }
                    throw new IOException("Corrupt optimizer state: entry '" + name + "' appears twice");
                }
                stateDict.put(name, value);
            }
            if (trainingStep > 0) {
                Object replaced = stateDict.put(Optimizer.STEP_COUNT_KEY, trainingStep);
                if (replaced instanceof Tensor) {
                    ((Tensor) replaced).close();
                }
            }
            try {
                optimizer.loadStateDict(stateDict);
            } catch (IllegalArgumentException e) {
                throw new IOException("Optimizer state does not fit " + optimizer.getClass().getSimpleName() + ": "
                    + e.getMessage(), e);
            }
        } catch (IOException | RuntimeException e) {
            closeStateDictAfter(stateDict, e);
            throw e;
        }
        Optimizer.closeStateDict(stateDict);
    }

    /** A FLOAT32 tensor on {@code backend} holding a tensor record ({@link #readEntry} matched shape and data). */
    private static Tensor optimizerTensor(SerializedTensor entry, ComputeBackend backend) {
        Shape shape = entry.shape.length == 0 ? Shape.scalar() : Shape.of(entry.shape);
        TensorStorage storage = TensorStorage.fromHostFloat(backend, entry.data);
        try {
            return Tensor.fromStorage(backend, shape, DType.FLOAT32, storage);
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
    }

    /** The value of a scalar record of {@code kind} with {@code payload}. */
    private static Object scalarValue(int kind, String name, long payload) throws IOException {
        switch (kind) {
            case RECORD_INTEGER:
                return payload;
            case RECORD_REAL:
                return Double.longBitsToDouble(payload);
            case RECORD_BOOLEAN:
                if (payload != 0L && payload != 1L) {
                    throw new IOException("Corrupt optimizer state: boolean entry '" + name + "' holds " + payload);
                }
                return payload == 1L;
            default:
                throw new IllegalArgumentException("not a scalar record kind: " + kind);
        }
    }

    /** Reads the name of a scalar record (UTF-8 length, then the bytes), refusing corrupt lengths. */
    private static String readScalarName(InputStream in) throws IOException {
        int length = readInt(in);
        if (length < 0 || length > MAX_NAME_BYTES) {
            throw new IOException("Corrupt optimizer state: scalar name length " + length);
        }
        byte[] nameBytes = new byte[length];
        readFully(in, nameBytes);
        return new String(nameBytes, StandardCharsets.UTF_8);
    }

    private static int readInt(InputStream in) throws IOException {
        byte[] bytes = new byte[4];
        readFully(in, bytes);
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    private static long readLong(InputStream in) throws IOException {
        byte[] bytes = new byte[8];
        readFully(in, bytes);
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
    }

    /** Reads a state header (format version, entry count) and refuses versions newer than {@link #FORMAT_VERSION}. */
    private static int readHeader(InputStream in, String what) throws IOException {
        return readStateHeader(in, what, FORMAT_VERSION).count;
    }

    /** Reads a state header (format version, entry count) and refuses versions newer than {@code maxVersion}. */
    private static StateHeader readStateHeader(InputStream in, String what, int maxVersion) throws IOException {
        byte[] headerBytes = new byte[8];
        readFully(in, headerBytes);
        ByteBuffer headerBuffer = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN);
        int version = headerBuffer.getInt();
        int count = headerBuffer.getInt();
        if (version > maxVersion) {
            throw new IOException("Unsupported " + what + " format version: " + version
                + " (maximum supported: " + maxVersion + ")");
        }
        if (count < 0) {
            throw new IOException("Corrupt " + what + ": negative entry count " + count);
        }
        return new StateHeader(version, count);
    }

    /** The format version and entry count of a state entry. */
    private static final class StateHeader {
        private final int version;
        private final int count;

        StateHeader(int version, int count) {
            this.version = version;
            this.count = count;
        }
    }

    /**
     * Reads one record written by {@link #writeEntry}. Refuses, before allocating anything of the sizes it names, a
     * negative or oversized name length or rank, a negative dimension, an element count that differs from the
     * product of the dimensions, and a record longer than what is left of the entry; the values are then read in
     * chunks, so a truncated entry fails at its end without a buffer of the full declared size.
     */
    private static SerializedTensor readEntry(EntryStream in) throws IOException {
        int nameLength = readInt(in);
        int rank = readInt(in);
        int dataLength = readInt(in);
        if (nameLength < 0 || nameLength > MAX_NAME_BYTES || rank < 0 || rank > MAX_RANK || dataLength < 0) {
            throw new IOException("Corrupt tensor record: name length " + nameLength + ", rank " + rank
                + ", element count " + dataLength);
        }
        long recordBytes = nameLength + 4L * rank + 4L * dataLength;
        if (recordBytes > in.remaining()) {
            throw new IOException("Corrupt tensor record: it needs " + recordBytes + " bytes, but its entry has "
                + in.remaining() + " left");
        }

        byte[] nameBytes = new byte[nameLength];
        readFully(in, nameBytes, nameLength);
        String name = new String(nameBytes, StandardCharsets.UTF_8);

        int[] shape = new int[rank];
        for (int j = 0; j < rank; j++) {
            shape[j] = readInt(in);
            if (shape[j] < 0) {
                throw new IOException("Corrupt tensor record '" + name + "': negative dimension " + shape[j]);
            }
        }
        if (elementCount(shape, dataLength) != dataLength) {
            throw new IOException("Corrupt tensor record '" + name + "': shape " + Arrays.toString(shape)
                + " does not hold its " + dataLength + " values");
        }
        return new SerializedTensor(name, shape, readFloats(in, dataLength));
    }

    /**
     * The product of the non-negative {@code dims} (1 for none), or a value above {@code limit} when the product
     * exceeds it, so the result never overflows.
     */
    private static long elementCount(int[] dims, int limit) {
        for (int dim : dims) {
            if (dim == 0) {
                return 0;
            }
        }
        long product = 1;
        for (int dim : dims) {
            product *= dim;
            if (product > limit) {
                return product;
            }
        }
        return product;
    }

    /**
     * Reads {@code count} little-endian floats in chunks, growing the result as values arrive, so the memory taken
     * follows the data actually present rather than a count read from the file.
     */
    private static float[] readFloats(InputStream in, int count) throws IOException {
        float[] values = new float[Math.min(count, INITIAL_FLOAT_CAPACITY)];
        byte[] chunk = new byte[STREAM_CHUNK_BYTES];
        int filled = 0;
        while (filled < count) {
            int n = Math.min(count - filled, STREAM_CHUNK_BYTES / Float.BYTES);
            readFully(in, chunk, n * Float.BYTES);
            if (filled + n > values.length) {
                values = Arrays.copyOf(values, (int) Math.min(count, Math.max(2L * values.length, filled + n)));
            }
            ByteBuffer.wrap(chunk, 0, n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                .get(values, filled, n);
            filled += n;
        }
        return values;
    }

    /** One tensor record of a state entry. */
    private static final class SerializedTensor {
        private final String name;
        private final int[] shape;
        private final float[] data;

        SerializedTensor(String name, int[] shape, float[] data) {
            this.name = name;
            this.shape = shape;
            this.data = data;
        }
    }

    /**
     * The step count the training metadata records (exact for every int, also for format-1 optimizer state that
     * stored it as a float), or 0 when the checkpoint has no training metadata or it records none.
     */
    private static int readTrainingStep(ZipFile zip) throws IOException {
        ZipEntry entry = zip.getEntry(TRAINING_STATE_ENTRY);
        if (entry == null) {
            return 0;
        }
        try (EntryStream in = new EntryStream(zip.getInputStream(entry), entry.getSize())) {
            return extractIntFromJson(readText(in, TRAINING_STATE_ENTRY), "step_count");
        }
    }

    /**
     * The UTF-8 text of a metadata entry.
     *
     * @throws IOException when it is longer than {@link #MAX_TEXT_ENTRY_BYTES}
     */
    private static String readText(InputStream in, String what) throws IOException {
        ByteArrayOutputStream text = new ByteArrayOutputStream();
        byte[] buffer = new byte[STREAM_CHUNK_BYTES];
        int len;
        while ((len = in.read(buffer)) >= 0) {
            if (text.size() + (long) len > MAX_TEXT_ENTRY_BYTES) {
                throw new IOException("Corrupt checkpoint: " + what + " is longer than " + MAX_TEXT_ENTRY_BYTES
                    + " bytes");
            }
            text.write(buffer, 0, len);
        }
        return new String(text.toByteArray(), StandardCharsets.UTF_8);
    }

    private static void readFully(InputStream in, byte[] buffer) throws IOException {
        readFully(in, buffer, buffer.length);
    }

    /** Reads exactly {@code length} bytes into the start of {@code buffer}. */
    private static void readFully(InputStream in, byte[] buffer, int length) throws IOException {
        int offset = 0;
        while (offset < length) {
            int read = in.read(buffer, offset, length - offset);
            if (read < 0) {
                throw new IOException("Corrupt checkpoint: unexpected end of entry");
            }
            offset += read;
        }
    }

    /**
     * The data of one archive entry, ending at the size the archive records for it ({@link #remaining()} tells how
     * much is left), so lengths read from the entry can be checked against it before anything is allocated. An entry
     * of unrecorded size is unbounded.
     */
    private static final class EntryStream extends FilterInputStream {
        private long remaining;

        EntryStream(InputStream in, long size) {
            super(in);
            this.remaining = size < 0 ? Long.MAX_VALUE : size;
        }

        long remaining() {
            return remaining;
        }

        @Override
        public int read() throws IOException {
            if (remaining <= 0) {
                return -1;
            }
            int value = super.read();
            if (value >= 0) {
                remaining--;
            }
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            if (remaining <= 0) {
                return -1;
            }
            int read = super.read(buffer, offset, (int) Math.min(length, remaining));
            if (read > 0) {
                remaining -= read;
            }
            return read;
        }

        @Override
        public long skip(long n) throws IOException {
            long skipped = super.skip(Math.min(n, remaining));
            remaining -= skipped;
            return skipped;
        }

        @Override
        public int available() throws IOException {
            return (int) Math.min(super.available(), remaining);
        }

        @Override
        public boolean markSupported() {
            return false;
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < ' ') {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }

    /**
     * The int value of {@code "key": <integer>} in the metadata, or 0 when the key is absent.
     *
     * @throws IOException when the value is not an integer in the int range
     */
    private static int extractIntFromJson(String json, String key) throws IOException {
        long value = extractLongFromJson(json, key);
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IOException("Malformed checkpoint metadata: \"" + key + "\" is out of the int range: " + value);
        }
        return (int) value;
    }

    /**
     * The long value of {@code "key": <integer>} in the metadata, or 0 when the key is absent.
     *
     * @throws IOException when the value is not an integer in the long range
     */
    private static long extractLongFromJson(String json, String key) throws IOException {
        String searchKey = "\"" + key + "\":";
        int idx = json.indexOf(searchKey);
        if (idx < 0) return 0;
        
        idx += searchKey.length();
        while (idx < json.length() && Character.isWhitespace(json.charAt(idx))) {
            idx++;
        }
        
        int endIdx = idx;
        while (endIdx < json.length() && 
               (Character.isDigit(json.charAt(endIdx)) || json.charAt(endIdx) == '-')) {
            endIdx++;
        }
        
        String text = json.substring(idx, endIdx);
        try {
            return Long.parseLong(text);
        } catch (NumberFormatException e) {
            throw new IOException(
                "Malformed checkpoint metadata: \"" + key + "\" is not an integer: \"" + text + "\"", e);
        }
    }

    public static class CheckpointInfo {
        private final int formatVersion;
        private final String moduleType;
        private final long parameterCount;
        private final int numParameters;
        private final String timestamp;
        private final Map<String, ParameterInfo> parameters;

        CheckpointInfo(int formatVersion, String moduleType, long parameterCount,
                      int numParameters, String timestamp, Map<String, ParameterInfo> parameters) {
            this.formatVersion = formatVersion;
            this.moduleType = moduleType;
            this.parameterCount = parameterCount;
            this.numParameters = numParameters;
            this.timestamp = timestamp;
            this.parameters = parameters;
        }

        public int formatVersion() { return formatVersion; }
        public String moduleType() { return moduleType; }
        public long parameterCount() { return parameterCount; }
        public int numParameters() { return numParameters; }
        public String timestamp() { return timestamp; }
        public Map<String, ParameterInfo> parameters() { return parameters; }
    }

    public static class ParameterInfo {
        private final int[] shape;
        private final String dtype;
        private final long numElements;

        ParameterInfo(int[] shape, String dtype, long numElements) {
            this.shape = shape;
            this.dtype = dtype;
            this.numElements = numElements;
        }

        public int[] shape() { return shape; }
        public String dtype() { return dtype; }
        public long numElements() { return numElements; }
    }

    public static CheckpointInfo inspect(String path) throws IOException {
        return inspect(Paths.get(path));
    }

    /**
     * The metadata of the checkpoint at {@code path}.
     *
     * @throws FileNotFoundException when {@code path} does not exist
     * @throws IOException           when the file is not a checkpoint archive, has no metadata, or the metadata is
     *                               malformed
     */
    public static CheckpointInfo inspect(Path path) throws IOException {
        Objects.requireNonNull(path, "Path cannot be null");

        try (ZipFile zip = openCheckpoint(path); EntryStream in = openEntry(zip, path, MODEL_META_ENTRY)) {
            return parseMetaJson(readText(in, MODEL_META_ENTRY));
        }
    }

    private static CheckpointInfo parseMetaJson(String json) throws IOException {
        int formatVersion = extractIntFromJson(json, "format_version");
        long parameterCount = extractLongFromJson(json, "parameter_count");
        int numParameters = extractIntFromJson(json, "num_parameters");
        
        String moduleType = extractStringFromJson(json, "module_type");
        String timestamp = extractStringFromJson(json, "timestamp");
        
        Map<String, ParameterInfo> parameters = new LinkedHashMap<>();
        String paramsStart = "\"parameters\": {";
        int paramsIdx = json.indexOf(paramsStart);
        if (paramsIdx >= 0) {
            paramsIdx += paramsStart.length();
            int paramsEnd = json.indexOf("\n  }", paramsIdx);
            if (paramsEnd > paramsIdx) {
                parseParameterEntries(json.substring(paramsIdx, paramsEnd), parameters);
            }
        }
        
        return new CheckpointInfo(formatVersion, moduleType, parameterCount, 
                                  numParameters, timestamp, parameters);
    }

    /**
     * Parses the entries {@code "name": {"shape": [...], "dtype": "...", "num_elements": n}} of the metadata's
     * {@code parameters} object into {@code into}, in file order; names are unescaped.
     *
     * @throws IOException when a name or an entry is unterminated, or an entry holds a malformed number
     */
    private static void parseParameterEntries(String section, Map<String, ParameterInfo> into) throws IOException {
        int currentIdx = 0;
        while (true) {
            int nameStart = section.indexOf('"', currentIdx);
            if (nameStart < 0) {
                return;
            }
            int nameEnd = closingQuote(section, nameStart + 1);
            if (nameEnd < 0) {
                throw new IOException("Malformed checkpoint metadata: unterminated parameter name");
            }
            String name = unescapeJson(section.substring(nameStart + 1, nameEnd));
            int entryEnd = section.indexOf('}', nameEnd);
            if (entryEnd < 0) {
                throw new IOException(
                    "Malformed checkpoint metadata: unterminated entry of parameter \"" + name + "\"");
            }
            String entry = section.substring(nameEnd + 1, entryEnd);
            into.put(name, new ParameterInfo(extractShapeFromJson(entry, 0), extractStringFromJson(entry, "dtype"),
                extractLongFromJson(entry, "num_elements")));
            currentIdx = entryEnd + 1;
        }
    }

    /** The index of the unescaped {@code '"'} that closes a JSON string whose content starts at {@code from}, or -1. */
    private static int closingQuote(String json, int from) {
        int idx = from;
        while (idx < json.length()) {
            char c = json.charAt(idx);
            if (c == '"') {
                return idx;
            }
            idx += c == '\\' ? 2 : 1;
        }
        return -1;
    }

    /**
     * The inverse of {@link #escapeJson(String)}.
     *
     * @throws IOException when an escape sequence is incomplete or unknown
     */
    private static String unescapeJson(String s) throws IOException {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            if (i + 1 >= s.length()) {
                throw new IOException("Malformed checkpoint metadata: dangling escape in \"" + s + "\"");
            }
            char e = s.charAt(++i);
            switch (e) {
                case '"': sb.append('"'); break;
                case '\\': sb.append('\\'); break;
                case '/': sb.append('/'); break;
                case 'b': sb.append('\b'); break;
                case 'f': sb.append('\f'); break;
                case 'n': sb.append('\n'); break;
                case 'r': sb.append('\r'); break;
                case 't': sb.append('\t'); break;
                case 'u':
                    if (i + 4 >= s.length()) {
                        throw new IOException("Malformed checkpoint metadata: incomplete \\u escape in \"" + s + "\"");
                    }
                    try {
                        sb.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
                    } catch (NumberFormatException ex) {
                        throw new IOException("Malformed checkpoint metadata: bad \\u escape in \"" + s + "\"", ex);
                    }
                    i += 4;
                    break;
                default:
                    throw new IOException("Malformed checkpoint metadata: unknown escape \\" + e + " in \"" + s + "\"");
            }
        }
        return sb.toString();
    }

    /**
     * The unescaped string value of the first {@code "key": "..."} in {@code json}; an absent key or a non-string
     * value (such as {@code null}) gives the empty string.
     *
     * @throws IOException when the value is unterminated or holds a malformed escape sequence
     */
    private static String extractStringFromJson(String json, String key) throws IOException {
        String searchKey = "\"" + key + "\":";
        int idx = json.indexOf(searchKey);
        if (idx < 0) {
            return "";
        }

        idx += searchKey.length();
        while (idx < json.length() && Character.isWhitespace(json.charAt(idx))) {
            idx++;
        }
        if (idx >= json.length() || json.charAt(idx) != '"') {
            return "";
        }

        int start = idx + 1;
        int end = closingQuote(json, start);
        if (end < 0) {
            throw new IOException("Malformed checkpoint metadata: unterminated value of \"" + key + "\"");
        }
        return unescapeJson(json.substring(start, end));
    }

    /**
     * The dimensions of the first {@code "shape": [d0, d1, ...]} at or after {@code startIdx}; an empty list (a scalar)
     * and an absent key give no dimensions.
     *
     * @throws IOException when the list is unterminated or a dimension is not an int
     */
    private static int[] extractShapeFromJson(String json, int startIdx) throws IOException {
        String shapeKey = "\"shape\": [";
        int idx = json.indexOf(shapeKey, startIdx);
        if (idx < 0) return new int[0];
        
        idx += shapeKey.length();
        int endIdx = json.indexOf("]", idx);
        if (endIdx < 0) {
            throw new IOException("Malformed checkpoint metadata: unterminated \"shape\" list");
        }
        
        String shapeStr = json.substring(idx, endIdx).trim();
        if (shapeStr.isEmpty()) {
            return new int[0];
        }
        String[] parts = shapeStr.split(",");
        int[] shape = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            try {
                shape[i] = Integer.parseInt(part);
            } catch (NumberFormatException e) {
                throw new IOException(
                    "Malformed checkpoint metadata: shape dimension \"" + part + "\" is not an int", e);
            }
        }
        return shape;
    }
}
