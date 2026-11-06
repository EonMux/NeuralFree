package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoCheckpoints;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoSafetensors;
import com.eonmux.neuralfree.gguf.hf.torchao.TorchaoWeight;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A Hugging Face checkpoint stored as safetensors files: a single file, or the files of a directory.
 *
 * <p>Files of a directory are found as vLLM and llama.cpp find them (vLLM {@code default_loader.py} and
 * {@code weight_utils.py}, llama.cpp {@code ModelBase.index_tensors}):</p>
 * <ul>
 *   <li>With an index ({@value #INDEX}, or {@value #MISTRAL_INDEX} for a Mistral native checkpoint): the files its
 *       {@code weight_map} names. Each name must be a relative path inside the directory, each file must exist and
 *       hold every tensor mapped to it, and every tensor of those files is used. A {@value #MTP_FILE} the index does
 *       not name is added (SGLang {@code maybe_add_mtp_safetensors}); other unnamed {@code *.safetensors} files are
 *       ignored.</li>
 *   <li>Without one: {@code model*.safetensors} ({@code consolidated*.safetensors} for a Mistral native checkpoint),
 *       else every {@code *.safetensors} file except a PEFT {@value #ADAPTER_MODEL}.</li>
 * </ul>
 * <p>Files are taken in natural order (digit runs compared as numbers) and tensors in name order within each file; a
 * tensor stored in two files is an error. Directory entries may be symbolic links (the Hugging Face cache snapshot
 * layout) and are followed. File {@code f} occupies the virtual range starting at
 * {@code V_f = sum over earlier files g of align64(dataLength_g)}, so a tensor's direct virtual offset is
 * {@code V_f + begin}. At most {@value #MAX_FILES} files and {@value #MAX_TENSORS} tensors are accepted.</p>
 *
 * <p>A file's {@code __metadata__} may carry torchao's flattened subclass descriptors ({@code tensor_names} and a
 * per-tensor {@code _type}/{@code _data}/{@code _tensor_data_names} JSON object). They are read into {@link
 * TorchaoWeight}s for the shared decoder ({@link TorchaoCheckpoints}); the flattened inner tensors they name are
 * hidden when the decoder exposes its canonical tensors.</p>
 */
public final class SafetensorsCheckpoint implements CheckpointContainer, TorchaoCheckpoints.Source {

    private static final Logger LOGGER = LogManager.getLogger(SafetensorsCheckpoint.class);

    /** Index of a sharded Hugging Face checkpoint. */
    public static final String INDEX = "model.safetensors.index.json";
    /** Index of a sharded Mistral native checkpoint. */
    public static final String MISTRAL_INDEX = "consolidated.safetensors.index.json";
    /** NextN / MTP weights that some checkpoints ship without listing them in the index. */
    public static final String MTP_FILE = "mtp.safetensors";
    /** Weights of a PEFT adapter, never part of a model checkpoint found by file name. */
    public static final String ADAPTER_MODEL = "adapter_model.safetensors";
    /** File-name suffix of safetensors files. */
    public static final String SUFFIX = ".safetensors";
    /** Most safetensors files of one checkpoint. */
    public static final int MAX_FILES = 10_000;
    /** Most tensors of one checkpoint. */
    public static final int MAX_TENSORS = 10_000_000;
    /** Largest index file read. */
    public static final long MAX_INDEX_BYTES = 512L << 20;
    /** Largest side file opened. */
    public static final long MAX_SIDE_FILE_BYTES = 4L << 30;
    /** Alignment of each file's virtual base. */
    public static final int SEGMENT_ALIGNMENT = 64;

    private static final String WEIGHT_MAP = "weight_map";

    /** Natural file order: digit runs compared as numbers, other runs as text, then the plain name. */
    static final Comparator<String> NATURAL_ORDER = SafetensorsCheckpoint::compareNatural;

    /** The built-in factory of {@link SafetensorsCheckpoint}s; it wraps the container with {@link
     *  TorchaoCheckpoints} so torchao subclasses become ordinary checkpoint tensors. */
    public static final CheckpointContainers.Factory FACTORY = new CheckpointContainers.Factory() {
        @Override
        public boolean accepts(Path dir, List<Path> files) {
            boolean safetensors = false;
            for (Path f : files) {
                String name = f.getFileName().toString();
                if (VllmShardFiles.isPreSharded(name)) {
                    return false;
                }
                if (name.endsWith(SUFFIX)) {
                    safetensors = true;
                }
            }
            return safetensors;
        }

        @Override
        public CheckpointContainer open(Path path, Path configDir, boolean mistralNative) throws IOException {
            return TorchaoCheckpoints.wrap(SafetensorsCheckpoint.open(path, configDir, mistralNative));
        }

        @Override
        public String toString() {
            return "safetensors";
        }
    };

    /** Where a tensor lives: its file and its virtual offset. */
    private static final class Located {
        final int file;
        final StTensor tensor;
        final long virtualOffset;

        Located(int file, StTensor tensor, long virtualOffset) {
            this.file = file;
            this.tensor = tensor;
            this.virtualOffset = virtualOffset;
        }
    }

    private final String description;
    private final Path modelDir;
    private final List<SafetensorsFile> files;
    private final List<DataSegment> segments;
    private final List<StTensor> tensors;
    private final Map<String, Located> byName;
    private final List<TorchaoWeight> torchaoWeights;
    private final List<SafetensorsFile> sideFiles = new ArrayList<>();
    private boolean closed;

    private SafetensorsCheckpoint(String description, Path modelDir, List<SafetensorsFile> files,
                                  List<DataSegment> segments, List<StTensor> tensors, Map<String, Located> byName,
                                  List<TorchaoWeight> torchaoWeights) {
        this.description = description;
        this.modelDir = modelDir;
        this.files = Collections.unmodifiableList(files);
        this.segments = Collections.unmodifiableList(segments);
        this.tensors = Collections.unmodifiableList(tensors);
        this.byName = byName;
        this.torchaoWeights = Collections.unmodifiableList(new ArrayList<>(torchaoWeights));
    }

    /**
     * Opens a single {@code .safetensors} file or a checkpoint directory.
     *
     * @param path          the file or directory
     * @param configDir     the model directory, which holds side files
     * @param mistralNative true for a Mistral native checkpoint
     * @throws FileNotFoundException when {@code path} does not exist
     * @throws IOException           when a file cannot be read, the index or a header is invalid (prefix
     *                               {@code "Invalid safetensors file '<file>': "}), or the files do not form one
     *                               checkpoint (prefix {@code "Invalid Hugging Face checkpoint '<dir>': "})
     */
    public static SafetensorsCheckpoint open(Path path, Path configDir, boolean mistralNative) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(configDir, "configDir");
        Map<String, Path> index = new LinkedHashMap<>();
        Path indexFile = null;
        List<Path> paths;
        if (Files.isDirectory(path)) {
            Path idx = path.resolve(mistralNative ? MISTRAL_INDEX : INDEX);
            if (Files.isRegularFile(idx)) {
                indexFile = idx;
                paths = fromIndex(path, idx, index);
            } else {
                paths = byName(path, mistralNative);
            }
        } else if (Files.isRegularFile(path)) {
            paths = Collections.singletonList(path);
        } else {
            throw new FileNotFoundException("safetensors checkpoint '" + path + "' does not exist");
        }
        List<SafetensorsFile> opened = new ArrayList<>(paths.size());
        try {
            for (Path p : paths) {
                opened.add(SafetensorsFile.open(p));
            }
            return assemble(path, configDir, opened, indexFile, index);
        } catch (IOException | RuntimeException e) {
            closeAll(opened, e);
            throw e;
        }
    }

    // ------------------------------------------------------------------ discovery

    /**
     * The files an index names, in natural order, plus an unlisted {@value #MTP_FILE}; fills {@code mapped} with the
     * file each listed tensor is mapped to.
     */
    private static List<Path> fromIndex(Path dir, Path indexFile, Map<String, Path> mapped) throws IOException {
        Object root;
        try {
            root = Json.parse(indexFile, MAX_INDEX_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw invalidFile(indexFile, "not a valid index: " + e.getMessage(), e);
        }
        Object weightMap = root instanceof Map ? ((Map<?, ?>) root).get(WEIGHT_MAP) : null;
        if (!(weightMap instanceof Map)) {
            throw invalidFile(indexFile, "'" + WEIGHT_MAP + "' is missing or not an object");
        }
        Map<?, ?> wm = (Map<?, ?>) weightMap;
        if (wm.size() > MAX_TENSORS) {
            throw invalidFile(indexFile, WEIGHT_MAP + " lists " + wm.size() + " tensors, more than the limit of "
                    + MAX_TENSORS);
        }
        Path base = dir.toAbsolutePath().normalize();
        Map<String, Path> resolved = new HashMap<>();
        Set<Path> listed = new LinkedHashSet<>();
        for (Map.Entry<?, ?> e : wm.entrySet()) {
            String tensor = (String) e.getKey();
            if (!(e.getValue() instanceof String)) {
                throw invalidFile(indexFile, WEIGHT_MAP + " maps tensor '" + tensor + "' to a non-string value");
            }
            String fileName = (String) e.getValue();
            Path file = resolved.get(fileName);
            if (file == null) {
                try {
                    file = resolveContentName(dir, fileName);
                } catch (IllegalArgumentException bad) {
                    throw invalidFile(indexFile, WEIGHT_MAP + " maps tensor '" + tensor + "' to '" + fileName
                            + "', which is not a file inside the model directory: " + bad.getMessage());
                }
                if (!Files.isRegularFile(file)) {
                    throw invalidFile(indexFile, WEIGHT_MAP + " maps tensor '" + tensor + "' to '" + fileName
                            + "', which does not exist");
                }
                if (!listed.contains(file) && listed.size() == MAX_FILES) {
                    throw invalidFile(indexFile, WEIGHT_MAP + " names more than " + MAX_FILES + " files");
                }
                listed.add(file);
                resolved.put(fileName, file);
            }
            mapped.put(tensor, file);
        }
        List<Path> out = new ArrayList<>(listed);
        out.sort((a, b) -> NATURAL_ORDER.compare(base.relativize(a).toString(), base.relativize(b).toString()));
        for (Path f : CheckpointContainers.regularFiles(dir)) {
            String name = f.getFileName().toString();
            if (!name.endsWith(SUFFIX) || listed.contains(f.toAbsolutePath().normalize())) {
                continue;
            }
            if (name.equals(MTP_FILE)) {
                LOGGER.info("{} is not listed in {}; adding its tensors (an MTP/NextN file the index omits)", f,
                        indexFile);
                out.add(f);
            } else {
                LOGGER.debug("ignoring {}: {} does not list it", f, indexFile);
            }
        }
        if (out.size() > MAX_FILES) {
            throw invalidCheckpoint(dir, "it holds " + out.size() + " *" + SUFFIX + " files, more than the limit"
                    + " of " + MAX_FILES);
        }
        return out;
    }

    /** The files of a directory without an index, in natural order. */
    private static List<Path> byName(Path dir, boolean mistralNative) throws IOException {
        String prefix = mistralNative ? "consolidated" : "model";
        List<Path> preferred = new ArrayList<>();
        List<Path> any = new ArrayList<>();
        for (Path f : CheckpointContainers.regularFiles(dir)) {
            String name = f.getFileName().toString();
            if (!name.endsWith(SUFFIX)) {
                continue;
            }
            if (name.startsWith(prefix)) {
                preferred.add(f);
            }
            if (!name.equals(ADAPTER_MODEL)) {
                any.add(f);
            }
        }
        List<Path> out = preferred.isEmpty() ? any : preferred;
        if (out.isEmpty()) {
            throw invalidCheckpoint(dir, "it holds no model *" + SUFFIX + " files");
        }
        if (out.size() > MAX_FILES) {
            throw invalidCheckpoint(dir, "it holds " + out.size() + " *" + SUFFIX + " files, more than the limit"
                    + " of " + MAX_FILES);
        }
        out.sort((a, b) -> NATURAL_ORDER.compare(a.getFileName().toString(), b.getFileName().toString()));
        return out;
    }

    /** Places the files in the virtual address space, lists their tensors and checks them against the index. */
    private static SafetensorsCheckpoint assemble(Path path, Path configDir, List<SafetensorsFile> files,
                                                  Path indexFile, Map<String, Path> index) throws IOException {
        List<DataSegment> segments = new ArrayList<>(files.size());
        List<StTensor> tensors = new ArrayList<>();
        Map<String, Located> byName = new HashMap<>();
        long base = 0;
        for (int f = 0; f < files.size(); f++) {
            SafetensorsFile file = files.get(f);
            segments.add(file.segment(base));
            for (StTensor t : file.tensors()) {
                Located prev = byName.put(t.name(), new Located(f, t, Math.addExact(base, t.begin())));
                if (prev != null) {
                    throw invalidCheckpoint(path, "tensor '" + t.name() + "' is stored in both '"
                            + files.get(prev.file).path() + "' and '" + file.path() + "'");
                }
                if (byName.size() > MAX_TENSORS) {
                    throw invalidCheckpoint(path, "it holds more than " + MAX_TENSORS + " tensors");
                }
                tensors.add(t);
            }
            base = Math.addExact(base, align(file.dataLength()));
        }
        for (Map.Entry<String, Path> e : index.entrySet()) {
            Located loc = byName.get(e.getKey());
            if (loc == null || !files.get(loc.file).path().toAbsolutePath().normalize()
                    .equals(e.getValue().toAbsolutePath().normalize())) {
                throw invalidFile(indexFile, WEIGHT_MAP + " maps tensor '" + e.getKey() + "' to '" + e.getValue()
                        + "', which does not hold it" + (loc == null ? "" : " (it is in '"
                        + files.get(loc.file).path() + "')"));
            }
        }
        List<TorchaoWeight> torchao = TorchaoSafetensors.read(metadata(files), new TorchaoSafetensors.Tensors() {
            @Override
            public StTensor tensor(String name) {
                Located loc = byName.get(name);
                return loc == null ? null : loc.tensor;
            }

            @Override
            public RowSource source(String name) {
                Located loc = locate(byName, name);
                return new StRowSource(files.get(loc.file), loc.tensor, loc.virtualOffset);
            }
        }, path.toString());
        String description = "safetensors checkpoint '" + path + "' (" + files.size() + " file"
                + (files.size() == 1 ? "" : "s") + ", " + tensors.size() + " tensors)";
        return new SafetensorsCheckpoint(description, configDir, files, segments, tensors, byName, torchao);
    }

    /** The {@code __metadata__} entries of every file, in file order. */
    private static List<Map<String, String>> metadata(List<SafetensorsFile> files) {
        List<Map<String, String>> out = new ArrayList<>(files.size());
        for (SafetensorsFile f : files) {
            out.add(f.metadata());
        }
        return out;
    }

    private static Located locate(Map<String, Located> byName, String name) {
        Located loc = byName.get(name);
        if (loc == null) {
            throw new IllegalArgumentException("the checkpoint has no tensor '" + name + "'");
        }
        return loc;
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
        return new StRowSource(files.get(loc.file), loc.tensor, loc.virtualOffset);
    }

    @Override
    public List<DataSegment> segments() {
        return segments;
    }

    @Override
    public long directVirtualOffset(String name) {
        return locate(name).virtualOffset;
    }

    /** The checkpoint files in virtual order (unmodifiable). */
    public List<SafetensorsFile> files() {
        return files;
    }

    @Override
    public List<TorchaoWeight> torchaoWeights() {
        return torchaoWeights;
    }

    @Override
    public SafetensorsFile openSideFile(String relPath) throws IOException {
        Objects.requireNonNull(relPath, "relPath");
        Path file;
        try {
            file = resolveContentName(modelDir, relPath);
        } catch (IllegalArgumentException e) {
            throw invalidCheckpoint(modelDir, "side file '" + relPath + "' is not a file inside the model directory: "
                    + e.getMessage());
        }
        if (!Files.isRegularFile(file)) {
            throw new FileNotFoundException("side file '" + file + "' of " + description + " does not exist");
        }
        long size = Files.size(file);
        if (size > MAX_SIDE_FILE_BYTES) {
            throw invalidFile(file, "the side file has " + size + " bytes, more than the limit of "
                    + MAX_SIDE_FILE_BYTES);
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
        List<SafetensorsFile> all;
        synchronized (sideFiles) {
            if (closed) {
                return;
            }
            closed = true;
            all = new ArrayList<>(files);
            all.addAll(sideFiles);
        }
        IOException first = null;
        for (SafetensorsFile f : all) {
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

    // ------------------------------------------------------------------ helpers

    /**
     * The path inside {@code modelDir} that the content-supplied {@code name} denotes. The name must be relative,
     * hold no {@code ..} segment, no NUL, no root or drive, and resolve lexically under {@code modelDir} after
     * normalization. Symbolic links are not resolved here; the caller follows them when it opens the file.
     *
     * @throws IllegalArgumentException with the reason when the name breaks a rule
     */
    public static Path resolveContentName(Path modelDir, String name) {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(name, "name");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("the name is empty");
        }
        if (name.indexOf('\0') >= 0) {
            throw new IllegalArgumentException("the name holds a NUL character");
        }
        if (name.startsWith("/") || name.startsWith("\\")
                || (name.length() >= 2 && name.charAt(1) == ':' && Character.isLetter(name.charAt(0)))) {
            throw new IllegalArgumentException("the name is absolute");
        }
        for (String segment : name.split("[/\\\\]", -1)) {
            if (segment.equals("..")) {
                throw new IllegalArgumentException("the name holds a '..' segment");
            }
        }
        Path rel;
        try {
            rel = Paths.get(name);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("the name is not a valid path (" + e.getReason() + ")", e);
        }
        if (rel.isAbsolute() || rel.getRoot() != null) {
            throw new IllegalArgumentException("the name is absolute");
        }
        Path dir = modelDir.toAbsolutePath().normalize();
        Path resolved = dir.resolve(rel).normalize();
        if (!resolved.startsWith(dir) || resolved.equals(dir)) {
            throw new IllegalArgumentException("the name does not resolve to a file inside '" + dir + "'");
        }
        return resolved;
    }

    private Located locate(String name) {
        Located loc = byName.get(Objects.requireNonNull(name, "name"));
        if (loc == null) {
            throw new IllegalArgumentException(description + " has no tensor '" + name + "'");
        }
        return loc;
    }

    private static long align(long n) {
        long rem = n % SEGMENT_ALIGNMENT;
        return rem == 0 ? n : Math.addExact(n, SEGMENT_ALIGNMENT - rem);
    }

    private static int compareNatural(String a, String b) {
        int i = 0;
        int j = 0;
        while (i < a.length() && j < b.length()) {
            char ca = a.charAt(i);
            char cb = b.charAt(j);
            if (isDigit(ca) && isDigit(cb)) {
                int ei = runEnd(a, i);
                int ej = runEnd(b, j);
                int c = compareDigits(a.substring(i, ei), b.substring(j, ej));
                if (c != 0) {
                    return c;
                }
                i = ei;
                j = ej;
            } else {
                if (ca != cb) {
                    return Character.compare(ca, cb);
                }
                i++;
                j++;
            }
        }
        int c = Integer.compare(a.length() - i, b.length() - j);
        return c != 0 ? c : a.compareTo(b);
    }

    /** Compares two runs of ASCII digits by value. */
    private static int compareDigits(String x, String y) {
        String sx = stripZeros(x);
        String sy = stripZeros(y);
        int c = Integer.compare(sx.length(), sy.length());
        return c != 0 ? c : sx.compareTo(sy);
    }

    private static String stripZeros(String digits) {
        int k = 0;
        while (k < digits.length() - 1 && digits.charAt(k) == '0') {
            k++;
        }
        return digits.substring(k);
    }

    private static int runEnd(String s, int from) {
        int k = from;
        while (k < s.length() && isDigit(s.charAt(k))) {
            k++;
        }
        return k;
    }

    private static boolean isDigit(char c) {
        return c >= '0' && c <= '9';
    }

    private static IOException invalidFile(Path file, String reason) {
        return new IOException("Invalid safetensors file '" + file + "': " + reason);
    }

    private static IOException invalidFile(Path file, String reason, Throwable cause) {
        return new IOException("Invalid safetensors file '" + file + "': " + reason, cause);
    }

    private static IOException invalidCheckpoint(Path dir, String reason) {
        return new IOException("Invalid Hugging Face checkpoint '" + dir + "': " + reason);
    }

    private static void closeAll(List<SafetensorsFile> files, Exception cause) {
        for (SafetensorsFile f : files) {
            try {
                f.close();
            } catch (IOException e) {
                cause.addSuppressed(e);
            }
        }
    }
}
