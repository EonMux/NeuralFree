package com.eonmux.neuralfree.gguf.hf.torch;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.st.SafetensorsCheckpoint;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoCheckpoints;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoPickle;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoWeight;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A Hugging Face checkpoint stored as PyTorch pickle files ({@code pytorch_model*.bin}, {@code .pt}, {@code
 * .pth}), read by the restricted unpickler. Its files are those Hugging Face finds ({@link
 * SafetensorsCheckpoint} in kind): the shards an index maps ({@code pytorch_model.bin.index.json}), else every
 * {@code pytorch_model*.bin} / {@code .pt} / {@code .pth} file but an adapter's. A file is torch's zip container:
 * {@code <name>/data.pkl} holds an ordered map of ({@code torch._utils._rebuild_tensor_v2}) views over {@code
 * archive/data/<k>} storages. Only tensor views are served; the pickle's other entries are skipped with a log
 * (the container is the checkpoint's weights). A value that is a torchao tensor subclass ({@code
 * _rebuild_from_type_v2} over {@code _rebuild_wrapper_subclass}, with its inner {@code _rebuild_tensor_v3} views)
 * is served as a {@link TorchaoWeight} for the shared decoder; its flattened inner tensors are not served.
 * Views are checked for offset, stride and contiguity; the caps of
 * safetensors (file and tensor counts, index size) hold, and the pickle runs under {@link RestrictedUnpickler}'s
 * own limits, nothing executed.
 *
 * <p>The files occupy the virtual address space as safetensors files do ({@code V_f = Σ align64(dataLength_g)}),
 * one segment per stored (uncompressed) storage entry plus one for the pickle itself, so the weights digest
 * covers the pickle and the storages, and a tensor of a stored entry keeps its direct virtual offset and is
 * mapped in place. {@code head.pt} and the registry's rule 6 keep the decision hooks' named refusals.</p>
 */
public final class TorchCheckpoint implements CheckpointContainer, TorchaoCheckpoints.Source {

    private static final Logger LOGGER = LogManager.getLogger(TorchCheckpoint.class);

    /** Index of a sharded PyTorch checkpoint. */
    public static final String INDEX = "pytorch_model.bin.index.json";
    /** Name prefix of a model weights pickle ({@code pytorch_model*bin}). */
    public static final String PREFIX = "pytorch_model";
    /** An adapter's weights, never part of a model checkpoint found by file name. */
    public static final String ADAPTER_MODEL = "adapter_model";
    /** The entry name of a container's pickle ({@code archive/data.pkl} in older torch versions). */
    public static final String PICKLE_ENTRY = "archive/data.pkl";
    /** Most files of one checkpoint. */
    public static final int MAX_FILES = 10_000;
    /** Most tensors of one checkpoint. */
    public static final int MAX_TENSORS = 10_000_000;
    /** Largest index file read. */
    public static final long MAX_INDEX_BYTES = 512L << 20;
    /** Largest side file opened. */
    public static final long MAX_SIDE_FILE_BYTES = 4L << 30;
    /** Alignment of each file's virtual base. */
    public static final int SEGMENT_ALIGNMENT = 64;
    /** The weight map of a sharded checkpoint's index. */
    private static final String WEIGHT_MAP = "weight_map";

    /** Where a tensor lives: its file, storage and virtual offset. */
    private static final class Located {
        final int file;
        final TorchStorage storage;
        final StTensor tensor;
        final long virtualOffset;

        Located(int file, TorchStorage storage, StTensor tensor, long virtualOffset) {
            this.file = file;
            this.storage = storage;
            this.tensor = tensor;
            this.virtualOffset = virtualOffset;
        }
    }

    /** One storage's use inside a parse (the pickle references it through its persistent ids). */
    private static final class Parse {
        final Map<String, TorchStorage> storages = new LinkedHashMap<>();
        final List<NamedView> views = new ArrayList<>();
        final List<TorchaoWeight> torchao = new ArrayList<>();
        TorchZipReader.Entry pickle;
        String prefix = TorchStorage.ARCHIVE;
    }

    /** A tensor name and its pickled view. */
    private static final class NamedView {
        final String name;
        final RestrictedUnpickler.TensorView view;

        NamedView(String name, RestrictedUnpickler.TensorView view) {
            this.name = name;
            this.view = view;
        }
    }

    private final String description;
    private final Path modelDir;
    private final List<TorchZipReader> files;
    private final List<DataSegment> segments;
    private final List<StTensor> tensors;
    private final Map<String, Located> byName;
    private final List<TorchaoWeight> torchaoWeights;
    private final List<SafetensorsFile> sideFiles = new ArrayList<>();
    private volatile boolean closed;

    private TorchCheckpoint(String description, Path modelDir, List<TorchZipReader> files,
                            List<DataSegment> segments, List<StTensor> tensors, Map<String, Located> byName,
                            List<TorchaoWeight> torchaoWeights) {
        this.description = description;
        this.modelDir = modelDir;
        this.files = files;
        this.segments = segments;
        this.tensors = tensors;
        this.byName = byName;
        this.torchaoWeights = Collections.unmodifiableList(new ArrayList<>(torchaoWeights));
    }

    /**
     * Opens a single {@code pytorch_model*.bin} / {@code .pt} / {@code .pth} file or a checkpoint directory.
     *
     * @throws FileNotFoundException when {@code path} does not exist
     * @throws IOException           when a file cannot be read or is invalid (prefix {@code "Invalid PyTorch
     *                               checkpoint file '<file>': "}), or the files do not form one checkpoint (prefix
     *                               {@code "Invalid Hugging Face checkpoint '<dir>': "})
     */
    public static TorchCheckpoint open(Path path, Path configDir, boolean mistralNative) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(configDir, "configDir");
        if (Files.isRegularFile(path)) {
            return openFiles(Collections.singletonList(path), configDir);
        }
        if (!Files.isDirectory(path)) {
            throw new FileNotFoundException("PyTorch checkpoint '" + path + "' does not exist");
        }
        List<Path> paths = checkpointFiles(path);
        if (paths.isEmpty()) {
            throw new IOException("Invalid Hugging Face checkpoint '" + path + "': it holds no PyTorch checkpoint"
                    + " files (looked for a " + PREFIX + "*.bin index shard and *.pt, *.pth files)");
        }
        return openFiles(paths, configDir);
    }

    // ------------------------------------------------------------------ discovery

    /** The regular files of a checkpoint directory (symbolic links followed), sorted by name. */
    private static List<Path> regularFiles(Path dir) throws IOException {
        List<Path> out = new ArrayList<>();
        try (java.nio.file.DirectoryStream<Path> entries = Files.newDirectoryStream(dir)) {
            for (Path entry : entries) {
                if (Files.isRegularFile(entry)) {
                    out.add(entry);
                }
            }
        }
        out.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        return out;
    }

    /**
     * The checkpoint files of a directory: the ones the {@code pytorch_model.bin.index.json} weight map names
     * when the index exists, else every {@code pytorch_model*.bin} / {@code .pt} / {@code .pth} file but an
     * adapter's, in file-name order.
     */
    private static List<Path> checkpointFiles(Path dir) throws IOException {
        List<Path> all = regularFiles(dir);
        Path indexFile = null;
        for (Path f : all) {
            if (f.getFileName().toString().equals(INDEX)) {
                indexFile = f;
                break;
            }
        }
        if (indexFile == null) {
            List<Path> out = new ArrayList<>();
            for (Path f : all) {
                if (isModelPickle(f.getFileName().toString())) {
                    out.add(f);
                    if (out.size() > MAX_FILES) {
                        throw new IOException("Invalid Hugging Face checkpoint '" + dir + "': it holds more than "
                                + MAX_FILES + " PyTorch checkpoint files");
                    }
                }
            }
            return out;
        }
        Map<String, Path> named = new HashMap<>();
        for (Path f : all) {
            named.put(f.getFileName().toString(), f);
        }
        Map<String, String> weightMap = weightMap(indexFile);
        Map<String, Path> resolved = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : weightMap.entrySet()) {
            String file = e.getValue();
            if (resolved.containsKey(file)) {
                if (!named.containsKey(file)) {
                    throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map"
                            + " maps tensor '" + e.getKey() + "' to '" + file + "', which does not exist");
                }
                continue;
            }
            Path target = named.get(file);
            if (target == null) {
                throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map maps"
                        + " tensor '" + e.getKey() + "' to '" + file + "', which does not exist");
            }
            if (!isModelPickle(file)) {
                throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map maps"
                        + " tensor '" + e.getKey() + "' to '" + file + "', which is not a PyTorch checkpoint file");
            }
            if (resolved.size() >= MAX_FILES) {
                throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map names"
                        + " more than " + MAX_FILES + " files");
            }
            resolved.put(file, target);
        }
        List<Path> out = new ArrayList<>(resolved.values());
        out.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        return out;
    }

    /**
     * Whether {@code name} is one torch checkpoint file the container reads: {@code pytorch_model*.bin}, a
     * {@code .pt} or a {@code .pth} file, but never an adapter's weights.
     */
    public static boolean isModelPickle(String name) {
        if (name.endsWith(".bin") && name.startsWith(PREFIX)) {
            return true;
        }
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return !name.startsWith(ADAPTER_MODEL);
        }
        return false;
    }

    /** The weight map of the sharded checkpoint's index, read with the strict JSON limits. */
    private static Map<String, String> weightMap(Path indexFile) throws IOException {
        Object root;
        try {
            root = Json.parse(indexFile, MAX_INDEX_BYTES, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': not valid JSON: "
                    + e.getMessage(), e);
        }
        Object weightMap = root instanceof Map ? ((Map<?, ?>) root).get(WEIGHT_MAP) : null;
        if (!(weightMap instanceof Map)) {
            throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': '" + WEIGHT_MAP + "' is"
                    + " missing or not an object");
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) weightMap).entrySet()) {
            if (!(e.getKey() instanceof String) || !(e.getValue() instanceof String)) {
                throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map maps '"
                        + e.getKey() + "' to a non-string value");
            }
            if (out.size() >= MAX_TENSORS) {
                throw new IOException("Invalid PyTorch checkpoint index '" + indexFile + "': the weight map lists"
                        + " more than " + MAX_TENSORS + " tensors");
            }
            out.put((String) e.getKey(), (String) e.getValue());
        }
        return out;
    }

    // ------------------------------------------------------------------ build

    /** The checkpoint over {@code paths}, one pickle parse per file. */
    private static TorchCheckpoint openFiles(List<Path> paths, Path configDir) throws IOException {
        List<TorchZipReader> zipFiles = new ArrayList<>(paths.size());
        try {
            for (Path p : paths) {
                zipFiles.add(TorchZipReader.open(p));
            }
            return assemble(configDir, zipFiles);
        } catch (IOException | RuntimeException e) {
            close(zipFiles, e);
            throw e;
        }
    }

    private static void close(List<TorchZipReader> files, Exception cause) {
        for (TorchZipReader f : files) {
            try {
                f.close();
            } catch (IOException e) {
                cause.addSuppressed(e);
            }
        }
    }

    /** The tensors, storages and virtual address space over the opened files. */
    private static TorchCheckpoint assemble(Path configDir, List<TorchZipReader> files) throws IOException {
        if (files.size() > MAX_FILES) {
            throw new IOException("Invalid Hugging Face checkpoint '" + configDir + "': it holds more than "
                    + MAX_FILES + " PyTorch checkpoint files");
        }
        List<DataSegment> segments = new ArrayList<>();
        List<StTensor> tensors = new ArrayList<>();
        Map<String, Located> byName = new HashMap<>();
        List<TorchaoWeight> torchao = new ArrayList<>();
        long base = 0;
        int count = 0;
        for (int f = 0; f < files.size(); f++) {
            TorchZipReader zip = files.get(f);
            Parse parse = parsePickle(configDir, zip);
            TorchZipReader.Entry pickle = parse.pickle;
            Map<TorchStorage, Long> storageBases = new LinkedHashMap<>();
            if (pickle.size() > 0) {
                segments.add(new DataSegment(zip.path().toString(), pickle.dataOffset(), base, pickle.size()));
                base = align(base, pickle.size());
            }
            for (TorchStorage storage : parse.storages.values()) {
                if (storage.absoluteOffset() < 0) {
                    continue; // a deflated storage keeps its bytes out of the virtual address space
                }
                segments.add(new DataSegment(zip.path().toString(), storage.absoluteOffset(), base,
                        storage.length()));
                storageBases.put(storage, base);
                base = align(base, storage.length());
            }
            for (NamedView named : parse.views) {
                StTensor tensor = tensor(zip, named);
                TorchStorage storage = (TorchStorage) named.view.storage();
                Long storageBase = storageBases.get(storage);
                long direct = storageBase == null ? -1 : Math.addExact(storageBase, tensor.begin());
                Located prev = byName.put(tensor.name(), new Located(f, storage, tensor, direct));
                if (prev != null) {
                    throw new IOException("Invalid Hugging Face checkpoint '" + configDir + "': tensor '"
                            + tensor.name() + "' is stored in both '" + files.get(prev.file).path() + "' and '"
                            + zip.path() + "'");
                }
                count++;
                if (count > MAX_TENSORS) {
                    throw new IOException("Invalid Hugging Face checkpoint '" + configDir + "': it holds more"
                            + " than " + MAX_TENSORS + " tensors");
                }
                tensors.add(tensor);
            }
            for (TorchaoWeight w : parse.torchao) {
                for (TorchaoWeight existing : torchao) {
                    if (existing.name().equals(w.name())) {
                        throw new IOException("Invalid Hugging Face checkpoint '" + configDir + "': tensor '"
                                + w.name() + "' is a torchao tensor in both '" + files.get(f).path()
                                + "' and another file");
                    }
                }
                torchao.add(w);
            }
        }
        String description = "PyTorch checkpoint '" + configDir + "' (" + files.size() + " file"
                + (files.size() == 1 ? "" : "s") + ", " + tensors.size() + " tensors)";
        return new TorchCheckpoint(description, configDir, files, Collections.unmodifiableList(segments),
                Collections.unmodifiableList(tensors), byName, Collections.unmodifiableList(torchao));
    }

    /** The pickle of {@code zip}, its persistent ids resolved into the parse's storages and tensor views. */
    private static Parse parsePickle(Path configDir, TorchZipReader zip) throws IOException {
        Parse out = new Parse();
        out.pickle = pickleEntry(zip);
        out.prefix = prefix(out.pickle.name());
        ByteBuffer data = zip.readAll(out.pickle);
        Object root;
        try {
            root = RestrictedUnpickler.unpickle(data.array(), id -> persistent(zip, id, out.storages, out.prefix));
        } catch (IOException e) {
            close(Collections.singletonList(zip), e);
            throw e;
        }
        if (!(root instanceof Map)) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': its pickle's top level is "
                    + name(root) + " instead of a map of tensors");
        }
        collect(zip, out, root, 0);
        if (out.views.isEmpty() && out.torchao.isEmpty()) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': its " + out.pickle.name()
                    + " holds no tensors");
        }
        return out;
    }

    /** The container's pickle entry: {@code archive/data.pkl}, or the one {@code <name>/data.pkl} torch writes. */
    private static TorchZipReader.Entry pickleEntry(TorchZipReader zip) throws IOException {
        TorchZipReader.Entry archive = zip.entry(PICKLE_ENTRY);
        if (archive != null && !archive.directory()) {
            return archive;
        }
        TorchZipReader.Entry found = null;
        for (TorchZipReader.Entry e : zip.entries()) {
            if (!e.directory() && e.name().endsWith("/data.pkl")) {
                if (found != null) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': it holds both '"
                            + found.name() + "' and '" + e.name() + "' as its pickle");
                }
                found = e;
            }
        }
        if (found == null) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': it holds no '"
                    + PICKLE_ENTRY + "' entry (not a torch.save container, or a legacy torch pickle)");
        }
        return found;
    }

    /** The archive directory of a pickle entry: {@code archive/} for {@code archive/data.pkl}. */
    private static String prefix(String pickleName) throws IOException {
        String prefix = pickleName.substring(0, pickleName.length() - "data.pkl".length());
        if (!prefix.matches("[A-Za-z0-9_.-]+/")) {
            throw new IOException("Invalid PyTorch checkpoint pickle path '" + pickleName + "': its archive"
                    + " directory is not a single safe path component");
        }
        return prefix;
    }

    /** A persistent id ({@code ('storage', <type>, <key>, <location>[, <numel>])}): the file's storage. */
    private static Object persistent(TorchZipReader zip, List<Object> id, Map<String, TorchStorage> storages,
                                     String prefix) throws IOException {
        if (id.size() != 4 && id.size() != 5) {
            throw new IOException("Invalid PyTorch checkpoint pickle: the persistent id carries " + id.size()
                    + " values instead of the 4 of ('storage', <type>, <key>, <location>) or the 5 of torch's"
                    + " ('storage', <type>, <key>, <location>, <numel>)");
        }
        if (!"storage".equals(id.get(0))) {
            throw new IOException("Invalid PyTorch checkpoint pickle: the persistent id is '" + name(id.get(0))
                    + "', not storage");
        }
        if (!(id.get(1) instanceof String) || !(id.get(2) instanceof String)
                || !(id.get(3) instanceof String)) {
            throw new IOException("Invalid PyTorch checkpoint pickle: the persistent id carries " + name(id.get(1))
                    + ", " + name(id.get(2)) + " and " + name(id.get(3)) + " instead of a storage type, key and"
                    + " location");
        }
        String typeName = (String) id.get(1);
        String key = (String) id.get(2);
        TorchStorage cached = storages.get(key);
        if (cached != null) {
            if (!typeName.equals(cached.typeName())) {
                throw new IOException("Invalid PyTorch checkpoint pickle: storage '" + key + "' is declared "
                        + cached.typeName() + " and " + typeName + " in the same pickle");
            }
            return cached;
        }
        TorchStorage storage = TorchStorage.of(zip, typeName, key, (String) id.get(3), prefix);
        if (id.size() == 5) {
            if (!(id.get(4) instanceof Long) || (Long) id.get(4) < 0) {
                throw new IOException("Invalid PyTorch checkpoint pickle: storage '" + key + "' carries " + name(
                        id.get(4)) + " instead of a non-negative element count");
            }
            long numel = (Long) id.get(4);
            long bytes = storage.dtype() == null ? numel
                    : Math.multiplyExact(numel, (long) storage.elementBytes());
            if (bytes != storage.length()) {
                throw new IOException("Invalid PyTorch checkpoint pickle: storage '" + key + "' declares " + numel
                        + " elements (" + bytes + " bytes) but holds " + storage.length() + " bytes");
            }
        }
        storages.put(key, storage);
        return storage;
    }

    /** Collects the pickle's tensor views, one level of nested maps allowed ({@code {"model": <state>}}) and the
     *  non-tensor entries (scalars and markers) skipped with a log. */
    private static void collect(TorchZipReader zip, Parse out, Object root, int depth) throws IOException {
        if (depth > 1) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': its pickle nests tensors"
                    + " more than 1 level deep");
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) root).entrySet()) {
            Object value = e.getValue();
            if (value instanceof RestrictedUnpickler.TensorView) {
                if (!(e.getKey() instanceof String)) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor name "
                            + name(e.getKey()) + " is not a string");
                }
                String tensorName = (String) e.getKey();
                for (NamedView named : out.views) {
                    if (named.name.equals(tensorName)) {
                        throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor '"
                                + tensorName + "' appears twice in its pickle");
                    }
                }
                if (out.views.size() >= MAX_TENSORS) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': it holds more than"
                            + " " + MAX_TENSORS + " tensors");
                }
                out.views.add(new NamedView(tensorName, (RestrictedUnpickler.TensorView) value));
                continue;
            }
            if (value instanceof TorchaoPickle.TensorObject) {
                if (!(e.getKey() instanceof String)) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor name "
                            + name(e.getKey()) + " is not a string");
                }
                String tensorName = (String) e.getKey();
                for (TorchaoWeight existing : out.torchao) {
                    if (existing.name().equals(tensorName)) {
                        throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor '"
                                + tensorName + "' appears twice in its pickle");
                    }
                }
                out.torchao.add(torchao(zip, tensorName, (TorchaoPickle.TensorObject) value));
                continue;
            }
            if (value instanceof Map) {
                collect(zip, out, value, depth + 1);
                continue;
            }
            LOGGER.debug("{}: skipping the pickle entry '{}' ({}), which is not a saved tensor", zip.path(),
                    e.getKey(), name(value));
        }
    }

    /**
     * The torchao weight a wrapper subclass rebuilds: its inner tensor views become row sources, its other state
     * fields (block size, shape, swizzle flag, activation kwargs) become attributes. Every field is checked to be a
     * value the restricted unpickler models; nothing else is carried into the decode.
     */
    private static TorchaoWeight torchao(TorchZipReader zip, String name, TorchaoPickle.TensorObject obj)
            throws IOException {
        int dot = obj.type().lastIndexOf('.');
        String simple = dot < 0 ? obj.type() : obj.type().substring(dot + 1);
        if (!TorchaoPickle.TENSOR_SIMPLE_NAMES.contains(simple)) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor '" + name
                    + "' is the torchao tensor subclass '" + obj.type() + "', which NeuralFree does not decode");
        }
        Map<String, Object> attributes = new LinkedHashMap<>();
        Map<String, TorchaoWeight.Inner> inner = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : obj.state().entrySet()) {
            Object value = e.getValue();
            if (value instanceof RestrictedUnpickler.TensorView) {
                RestrictedUnpickler.TensorView view = (RestrictedUnpickler.TensorView) value;
                StTensor tensor = tensor(zip, new NamedView(name + "." + e.getKey(), view));
                if (!(view.storage() instanceof TorchStorage)) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': tensor '" + name
                            + "' inner tensor '" + e.getKey() + "' resolves to " + name(view.storage())
                            + " instead of a storage of its archive");
                }
                inner.put(e.getKey(), new TorchaoWeight.Inner(tensor.dtype(), tensor.shape(),
                        new Row((TorchStorage) view.storage(), tensor, -1), null));
            } else {
                field(name, e.getKey(), value, zip);
                attributes.put(e.getKey(), value);
            }
        }
        return new TorchaoWeight(name, simple, obj.view().size(), attributes, inner);
    }

    /** Checks that a torchao state field is a value the restricted unpickler models. */
    private static void field(String tensor, String field, Object value, TorchZipReader zip) throws IOException {
        if (value == null || value instanceof String || value instanceof Long || value instanceof Boolean
                || value instanceof Double || value instanceof TorchaoPickle.EnumValue) {
            return;
        }
        if (value instanceof TorchaoPickle.AttributeObject) {
            TorchaoPickle.AttributeObject o = (TorchaoPickle.AttributeObject) value;
            for (Map.Entry<String, Object> e : o.state().entrySet()) {
                field(tensor, field + "." + e.getKey(), e.getValue(), zip);
            }
            return;
        }
        if (value instanceof TorchaoPickle.WrapperView || value instanceof TorchaoPickle.TensorObject) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': torchao tensor '" + tensor
                    + "' field '" + field + "' holds a nested tensor subclass, which NeuralFree does not decode");
        }
        if (value instanceof Object[]) {
            for (Object item : (Object[]) value) {
                field(tensor, field, item, zip);
            }
            return;
        }
        if (value instanceof List) {
            for (Object item : (List<?>) value) {
                field(tensor, field, item, zip);
            }
            return;
        }
        if (value instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) value).entrySet()) {
                if (!(e.getKey() instanceof String)) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': torchao tensor '"
                            + tensor + "' field '" + field + "' has a non-string key");
                }
                field(tensor, field + "." + e.getKey(), e.getValue(), zip);
            }
            return;
        }
        throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': torchao tensor '" + tensor
                + "' field '" + field + "' is " + name(value) + ", which is not a serialized tensor or plain"
                + " attribute");
    }

    private static String name(Object value) {
        if (value == null) {
            return "None";
        }
        if (value instanceof String) {
            return "string '" + value + "'";
        }
        if (value instanceof Long) {
            return "integer " + value;
        }
        if (value instanceof Boolean) {
            return "flag " + value;
        }
        if (value instanceof Object[]) {
            return "tuple of " + ((Object[]) value).length;
        }
        if (value instanceof List) {
            return "list of " + ((List<?>) value).size();
        }
        if (value instanceof Map) {
            return "map of " + ((Map<?, ?>) value).size();
        }
        return "a " + value.getClass().getSimpleName();
    }

    /** The stored tensor entry of a view: element count, offset, contiguity and storage bounds checked. */
    private static StTensor tensor(TorchZipReader zip, NamedView named) throws IOException {
        RestrictedUnpickler.TensorView view = named.view;
        if (!(view.storage() instanceof TorchStorage)) {
            throw new IOException("Invalid PyTorch checkpoint file '" + zip.path() + "': the persistent id of tensor '"
                    + named.name + "' resolves to " + name(view.storage()) + " instead of a storage of 'archive/data'");
        }
        TorchStorage storage = (TorchStorage) view.storage();
        String file = zip.path().toString();
        if (!contiguous(view.size(), view.stride())) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' is stored as a non-contiguous view of storage '" + storage.key() + "' (shape "
                    + Arrays.toString(view.size()) + ", strides " + Arrays.toString(view.stride()) + ")");
        }
        long numel;
        try {
            numel = 1;
            for (long d : view.size()) {
                numel = Math.multiplyExact(numel, d);
            }
        } catch (ArithmeticException e) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' has a shape " + Arrays.toString(view.size()) + " whose element count overflows a 64-bit"
                    + " integer", e);
        }
        StDType dtype = storage.dtype();
        if (view.dtype() != null) {
            dtype = TorchaoPickle.dtype(view.dtype());
            if (dtype == null) {
                throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                        + "' names the element dtype '" + view.dtype() + "', which NeuralFree does not store");
            }
        }
        if (dtype == null) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' is stored in an untyped storage without a dtype");
        }
        long elemBytes = dtype.bits() / 8;
        if (numel > Long.MAX_VALUE / elemBytes) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name + "' of "
                    + numel + " " + dtype.headerName() + " elements overflows a 64-bit byte count");
        }
        long byteLength = numel * elemBytes;
        long begin;
        try {
            begin = Math.multiplyExact(view.storageOffset(), elemBytes);
        } catch (ArithmeticException e) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' starts at element " + view.storageOffset() + " of storage '" + storage.key() + "', past"
                    + " its byte length", e);
        }
        if (byteLength > Long.MAX_VALUE - begin) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' spans past the end of its storage");
        }
        if (begin + byteLength > storage.length()) {
            throw new IOException("Invalid PyTorch checkpoint file '" + file + "': tensor '" + named.name
                    + "' spans bytes [" + begin + ", " + (begin + byteLength) + ") of storage '" + storage.key()
                    + "' of " + storage.length() + " bytes");
        }
        return new StTensor(named.name, dtype, view.size(), begin, byteLength);
    }

    /** The torch contiguity test (the view's own strides match its shape as PyTorch lays them out). */
    private static boolean contiguous(long[] size, long[] stride) {
        long expected = 1;
        for (int i = size.length - 1; i >= 0; i--) {
            if (size[i] != 1) {
                if (stride[i] != expected) {
                    return false;
                }
                expected = Math.multiplyExact(expected, size[i]);
            }
        }
        return true;
    }

    private static long align(long base, long length) {
        long end = Math.addExact(base, length);
        long rem = end % SEGMENT_ALIGNMENT;
        return rem == 0 ? end : Math.addExact(end, SEGMENT_ALIGNMENT - rem);
    }

    // ------------------------------------------------------------------ CheckpointContainer

    @Override
    public String description() {
        return description;
    }

    @Override
    public List<StTensor> tensors() {
        return tensors;
    }

    @Override
    public StTensor tensor(String name) {
        Located loc = byName.get(name);
        return loc == null ? null : loc.tensor;
    }

    @Override
    public RowSource source(String name) {
        Located loc = locate(name);
        return new Row(loc.storage, loc.tensor, loc.virtualOffset);
    }

    @Override
    public List<TorchaoWeight> torchaoWeights() {
        return torchaoWeights;
    }

    @Override
    public List<DataSegment> segments() {
        return segments;
    }

    @Override
    public long directVirtualOffset(String name) {
        return locate(name).virtualOffset;
    }

    @Override
    public SafetensorsFile openSideFile(String relPath) throws IOException {
        Objects.requireNonNull(relPath, "relPath");
        Path file;
        try {
            file = SafetensorsCheckpoint.resolveContentName(modelDir, relPath);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid Hugging Face checkpoint side file '" + relPath + "': " + e.getMessage());
        }
        if (!Files.isRegularFile(file)) {
            throw new FileNotFoundException("side file '" + file + "' of " + description + " does not exist");
        }
        long size = Files.size(file);
        if (size > MAX_SIDE_FILE_BYTES) {
            throw new IOException("Invalid Hugging Face checkpoint side file '" + file + "': " + size
                    + " bytes, more than the limit of " + MAX_SIDE_FILE_BYTES);
        }
        synchronized (sideFiles) {
            if (closed) {
                throw new IllegalStateException(description + " is closed");
            }
            SafetensorsFile side = SafetensorsFile.open(file);
            sideFiles.add(side);
            return side;
        }
    }

    @Override
    public void close() throws IOException {
        List<SafetensorsFile> side;
        synchronized (sideFiles) {
            if (closed) {
                return;
            }
            closed = true;
            side = new ArrayList<>(sideFiles);
            sideFiles.clear();
        }
        IOException first = null;
        for (TorchZipReader f : files) {
            try {
                f.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        for (SafetensorsFile f : side) {
            try {
                f.close();
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

    @Override
    public String toString() {
        return description;
    }

    private Located locate(String name) {
        Located loc = byName.get(name);
        if (loc == null) {
            throw new IllegalArgumentException(description + " has no tensor '" + name + "'");
        }
        return loc;
    }

    // ------------------------------------------------------------------ rows

    /** A {@link RowSource} over one stored tensor: its rows are read from the storage's bytes (positional reads),
     *  and a tensor of a stored (uncompressed) entry has its direct virtual offset. Immutable and thread-safe
     *  while the checkpoint is open. */
    private static final class Row implements RowSource {

        private final TorchStorage storage;
        private final StTensor tensor;
        private final long directVirtualOffset;
        private final ElementFormat format;
        private final long rows;
        private final long innermost;

        Row(TorchStorage storage, StTensor tensor, long directVirtualOffset) {
            this.storage = storage;
            this.tensor = tensor;
            this.directVirtualOffset = directVirtualOffset;
            this.format = ElementFormat.of(tensor.dtype());
            long[] shape = tensor.shape();
            this.innermost = shape.length == 0 ? 1 : shape[shape.length - 1];
            this.rows = outerProduct(tensor.name(), shape);
        }

        @Override
        public String describe() {
            return "tensor '" + tensor.name() + "' (" + tensor.dtype().headerName() + " "
                    + Arrays.toString(tensor.shape()) + ") of '" + storage.key() + "' in '" + storage.entry().name()
                    + "'";
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return tensor.shape();
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            try {
                return format.rowBytes(innermost);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(describe() + " has no byte-addressable rows: " + e.getMessage(),
                        e);
            }
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            Objects.requireNonNull(dst, "dst");
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                        + " rows of " + describe());
            }
            long rb = rowBytes();
            long bytes = Math.multiplyExact(rb, (long) count);
            if (bytes > dst.remaining()) {
                throw new IllegalArgumentException(count + " rows of " + describe() + " need " + bytes + " bytes;"
                        + " the buffer has " + dst.remaining());
            }
            ByteBuffer window = dst.slice().order(dst.order());
            window.limit((int) bytes);
            storage.read(tensor.begin() + Math.multiplyExact(first, rb), window);
            dst.position(dst.position() + (int) bytes);
        }

        @Override
        public long directVirtualOffset() {
            return directVirtualOffset;
        }

        private static long outerProduct(String tensorName, long[] shape) {
            long r = 1;
            try {
                for (int i = 0; i < shape.length - 1; i++) {
                    r = Math.multiplyExact(r, shape[i]);
                }
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("tensor '" + tensorName + "': the row count of shape "
                        + Arrays.toString(shape) + " overflows a 64-bit integer", e);
            }
            return r;
        }

        @Override
        public String toString() {
            return "TorchRowSource{" + describe() + "}";
        }
    }
}
