package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hf.lora.PeftAdapterSource;
import com.eonmux.neuralfree.gguf.hf.config.HfCheckpointDetectors;
import com.eonmux.neuralfree.gguf.load.DummyWeights;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.load.OverriddenSource;
import com.eonmux.neuralfree.gguf.load.RopeOverrides;
import com.eonmux.neuralfree.gguf.load.TensorChecker;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Opens model checkpoints as {@link ModelSource}s, detecting the format from the path:
 * <ol>
 *   <li>a regular file starting with the GGUF magic, or named {@code *.gguf}: GGUF (split models open from their
 *       first split);</li>
 *   <li>a regular file named {@code *.safetensors}: a Hugging Face checkpoint of that file alone, configured by its
 *       directory; a file listed by a {@code *.safetensors.index.json} beside it is one shard of a sharded checkpoint
 *       and is refused with the directory to pass instead;</li>
 *   <li>a directory with {@code *.safetensors} files and {@code config.json}, {@code params.json} (Mistral
 *       native) or the marker files of a {@link HfCheckpointDetectors checkpoint detector}: a Hugging Face
 *       checkpoint;</li>
 *   <li>a directory without {@code *.safetensors} files holding exactly one {@code .gguf} file, or one split set
 *       {@code <prefix>-0000N-of-0000M.gguf}: that GGUF model (the vLLM / SGLang "gguf" load format);</li>
 *   <li>a directory with {@code adapter_config.json} and {@code adapter_model.safetensors}: a PEFT adapter, opened only
 *       through {@link #openAdapter} (when the directory also holds a checkpoint of rule 3, that checkpoint is the
 *       model and the adapter is still opened by {@link #openAdapter}; so is a directory without
 *       {@code config.json} that a checkpoint detector recognizes, such as a decision-LoRA release);</li>
 *   <li>PyTorch pickle checkpoints ({@code pytorch_model*.bin}, {@code *.pt}, {@code *.pth}) are refused: unpickling
 *       executes code;</li>
 *   <li>anything else is a {@link FileNotFoundException} naming what was looked for.</li>
 * </ol>
 * Directory entries and the given path may be symbolic links (the Hugging Face cache snapshot layout) and are
 * followed.
 */
public final class ModelSources {

    /** Legacy message of the S0 open of a PEFT adapter, kept for its metadata-check expectations. */
    static final String PEFT_UNSUPPORTED = "PEFT adapters need the safetensors loader";

    private static final String GGUF_SUFFIX = ".gguf";
    private static final String SAFETENSORS_SUFFIX = ".safetensors";
    private static final String INDEX_SUFFIX = ".safetensors.index.json";
    private static final String CONFIG_JSON = "config.json";
    private static final String PARAMS_JSON = "params.json";
    private static final String ADAPTER_CONFIG = "adapter_config.json";
    private static final String ADAPTER_MODEL = "adapter_model.safetensors";
    private static final String PICKLE_PREFIX = "pytorch_model";
    private static final String LOOKED_FOR = "looked for a GGUF file, *.safetensors with config.json or params.json,"
            + " and adapter_config.json with adapter_model.safetensors";

    /** A split GGUF file name ({@code llama_split_path}): prefix, split number, split count. */
    private static final Pattern SPLIT_NAME = Pattern.compile("(.+)-(\\d{5})-of-(\\d{5})\\.gguf");

    /** What a path holds. */
    private enum Kind { GGUF, SAFETENSORS_FILE, SAFETENSORS_DIR, PEFT_ADAPTER }

    /** A detected path: its kind and the path to open (for a directory holding one GGUF model, that model). */
    private static final class Detection {
        final Kind kind;
        final String path;

        Detection(Kind kind, String path) {
            this.kind = kind;
            this.path = path;
        }
    }

    private ModelSources() {
    }

    /**
     * The GGUF parser of the file at {@code path}, the one internal reader of a bare GGUF file: the loader seam
     * owns constructing parsers so callers never do (the final gate's {@code new GGUFFileParser} rule).
     *
     * @throws IOException when {@code path} is not a readable GGUF v3 file
     */
    public static GGUFFileParser parseGguf(String path) throws IOException {
        return new GGUFFileParser(path);
    }

    /** {@link #open(String, ModelSourceOptions)} with {@link ModelSourceOptions#fromSystemProperties()}. */
    public static ModelSource open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /**
     * Opens the model at {@code path}.
     *
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no model
     * @throws IOException           when the model cannot be read or is invalid, or the path is one shard of a
     *                               sharded checkpoint, a PEFT adapter or a PyTorch pickle checkpoint
     */
    public static ModelSource open(String path, ModelSourceOptions options) throws IOException {
        return open(path, options, null);
    }

    /**
     * Opens the model at {@code path} as a drafter of the model at {@code target}, whose configuration a draft
     * checkpoint may need; GGUF drafts carry their own. A Hugging Face checkpoint loads weight-only.
     *
     * @param target the model the draft serves, or null when the model is not a draft
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no model
     * @throws IOException           when the model cannot be read or is invalid, or the path is one shard of a
     *                               sharded checkpoint, a PEFT adapter or a PyTorch pickle checkpoint
     */
    public static ModelSource open(String path, ModelSourceOptions options, Path target) throws IOException {
        return open(path, options, target, ActivationQuant.WEIGHT_ONLY);
    }

    /**
     * Opens the model at {@code path} as {@link #open(String, ModelSourceOptions, Path)} does, with the engine's
     * treatment of a Hugging Face checkpoint's activation quantization: {@link ActivationQuant#CHECKPOINT} refuses
     * activation schemes NeuralFree does not emulate. GGUF files carry their activation schemes in their metadata.
     *
     * @param target the model the draft serves, or null when the model is not a draft
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no model
     * @throws IOException           when the model cannot be read or is invalid, or the path is one shard of a
     *                               sharded checkpoint, a PEFT adapter or a PyTorch pickle checkpoint
     */
    public static ModelSource open(String path, ModelSourceOptions options, Path target, ActivationQuant actQuant)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(actQuant, "actQuant");
        Detection d = detectPath(path);
        ModelSource source;
        switch (d.kind) {
            case GGUF:
                source = new GGUFFileParser(d.path);
                break;
            case SAFETENSORS_FILE:
                rejectIndexedShard(path);
                source = HfModelSource.open(Paths.get(path), options, target, actQuant);
                break;
            case SAFETENSORS_DIR:
                source = HfModelSource.open(Paths.get(path), options, target, actQuant);
                break;
            default:
                throw new IOException("'" + path + "' is a PEFT adapter; load it with --lora on its base model");
        }
        return withLoadExtras(source, options);
    }

    /**
     * The opened source with the loading extras of {@code options} applied to its GGUF view, in order:
     * <ol>
     *   <li>the rope overrides ({@code ModelSourceOptions#ropeOverrides()}, when set) fail with an
     *       {@link IllegalArgumentException} when the source names no {@code general.architecture} to write their
     *       {@code %s.rope.*} keys for, and become metadata overrides of its keys;</li>
     *   <li>the {@code metadataOverrides} ({@code --override-kv}), the same view rewrite for GGUF and
     *       Hugging Face sources, into {@link OverriddenSource};</li>
     *   <li>{@code loadFormat = DUMMY}, weights regenerated as {@link DummyWeights} when the option asks;</li>
     *   <li>{@code checkTensors}, the value validation of {@link TensorChecker} over the weights the view holds.
     *       </li>
     * </ol>
     * A source with none of the extras set comes back unchanged, so every opener of a plain model sees today's
     * behavior.
     *
     * @throws IOException  when a Hugging Face checkpoint cannot be read, the tensor check fails, or the check
     *                      cannot read a tensor
     * @throws IllegalArgumentException when a rope override or an {@code --override-kv} entry is malformed, or the
     *                                  rope overrides have no architecture to write their keys for, or the
     *                                  tensor table of a dummy load is not whole-block and non-overlapping
     */
    private static ModelSource withLoadExtras(ModelSource source, ModelSourceOptions options) throws IOException {
        List<MetadataOverrides.Override> overrides = new ArrayList<>();
        if (!options.ropeOverrides().isEmpty()) {
            String arch = source.getString("general.architecture");
            if (arch == null) {
                throw new IllegalArgumentException("the model " + source.getFilePath() + " has no"
                        + " general.architecture; the rope overrides cannot name their rope.* keys");
            }
            overrides.addAll(options.ropeOverrides().toOverrides(arch));
        }
        overrides.addAll(options.metadataOverrides());
        ModelSource view = source;
        if (!overrides.isEmpty()) {
            view = new OverriddenSource(view, overrides);
        }
        if (options.loadFormat() == ModelSourceOptions.LoadFormat.DUMMY) {
            view = DummyWeights.over(view);
        }
        if (options.checkTensors()) {
            try {
                TensorChecker.check(view);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                IOException io = new IOException("the tensor check of " + source.getFilePath() + " was"
                        + " interrupted", e);
                throw io;
            }
        }
        return view;
    }

    /**
     * Opens the LoRA adapter at {@code path} for the model loaded from {@code base}: a GGUF adapter file (S0's
     * parser), a PEFT adapter directory (LA-1's {@link PeftAdapterSource}), or a model directory that also holds a
     * PEFT adapter's two files (the adapter of the model beside it).
     *
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no adapter
     * @throws IOException           when the adapter cannot be read or is invalid, or its base configuration does
     *                               not resolve
     */
    public static ModelSource openAdapter(String path, SourceOrigin base, ModelSourceOptions options)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(options, "options");
        Detection d = detectPath(path);
        if (d.kind == Kind.GGUF) {
            return new GGUFFileParser(d.path);
        }
        if (d.kind == Kind.PEFT_ADAPTER
                || (d.kind == Kind.SAFETENSORS_DIR && isPeftAdapter(regularFileNames(Paths.get(path))))) {
            return PeftAdapterSource.open(Paths.get(d.path), base, options);
        }
        throw new FileNotFoundException("'" + path + "' is not a LoRA adapter: looked for a GGUF adapter file"
                + " and a PEFT directory (adapter_config.json with adapter_model.safetensors)");
    }

    /**
     * Opens the vision / audio projector view of the Hugging Face checkpoint directory {@code dir}.
     *
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no model
     * @throws IOException           when {@code dir} is not a Hugging Face checkpoint directory, or the checkpoint
     *                               cannot be read, is invalid or has no projector conversion
     */
    public static ModelSource openProjector(Path dir, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(options, "options");
        Detection d = detectPath(dir.toString());
        if (d.kind == Kind.SAFETENSORS_DIR) {
            return HfModelSource.openProjector(dir, options);
        }
        throw new IOException("'" + dir + "' is not a Hugging Face checkpoint directory; a projector view is built"
                + " from one");
    }

    /**
     * The container format at {@code path}: {@link ModelFormat#GGUF} for a GGUF file or a directory holding one GGUF
     * model, {@link ModelFormat#SAFETENSORS} for a Hugging Face checkpoint file or directory and for a PEFT adapter
     * directory.
     *
     * @throws FileNotFoundException when the path does not exist, cannot be read or holds no model
     * @throws IOException           for a PyTorch pickle checkpoint, or when the path cannot be read
     */
    public static ModelFormat detect(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        return detectPath(path.toString()).kind == Kind.GGUF ? ModelFormat.GGUF : ModelFormat.SAFETENSORS;
    }

    /**
     * Checks that {@code path} names a readable regular file or directory, before anything is loaded.
     *
     * @throws FileNotFoundException when it does not
     */
    public static void requireOpenable(String path) throws FileNotFoundException {
        Objects.requireNonNull(path, "path");
        Path p;
        try {
            p = Paths.get(path);
        } catch (InvalidPathException e) {
            throw new FileNotFoundException("model not found: '" + path + "' is not a valid path (" + e.getReason()
                    + ")");
        }
        if (!(Files.isRegularFile(p) || Files.isDirectory(p)) || !Files.isReadable(p)) {
            throw new FileNotFoundException("model not found or not readable: " + path
                    + " (expected a GGUF file, a Hugging Face model directory or a .safetensors file)");
        }
    }

    // ------------------------------------------------------------------ detection

    private static Detection detectPath(String path) throws IOException {
        Path p;
        try {
            p = Paths.get(path);
        } catch (InvalidPathException e) {
            throw new FileNotFoundException("model not found: '" + path + "' is not a valid path (" + e.getReason()
                    + ")");
        }
        if (!Files.exists(p)) {
            throw new FileNotFoundException("model not found: '" + path + "' does not exist");
        }
        if (!Files.isReadable(p)) {
            throw new FileNotFoundException("model not readable: '" + path + "'");
        }
        if (Files.isRegularFile(p)) {
            return detectFile(p, path);
        }
        if (Files.isDirectory(p)) {
            return detectDirectory(p, path);
        }
        throw new FileNotFoundException("'" + path + "' is neither a regular file nor a directory; " + LOOKED_FOR);
    }

    private static Detection detectFile(Path file, String path) throws IOException {
        String name = file.getFileName().toString();
        if (hasGgufMagic(file) || name.endsWith(GGUF_SUFFIX)) {
            return new Detection(Kind.GGUF, path);
        }
        if (name.endsWith(SAFETENSORS_SUFFIX)) {
            return new Detection(Kind.SAFETENSORS_FILE, path);
        }
        if (isPickle(name)) {
            throw pickle(path);
        }
        throw new FileNotFoundException("'" + path + "' is not a model file: looked for the GGUF magic, a .gguf name"
                + " and a .safetensors name");
    }

    /**
     * A directory's kind. A PEFT adapter is recognized by its two files unless the directory also holds a model
     * checkpoint (configuration plus other {@code *.safetensors} files, or a layout a checkpoint detector recognizes),
     * which then takes precedence as rule 3 does. The detectors' marker files are read only for a directory with
     * safetensors files and neither {@code config.json} nor {@code params.json}.
     */
    private static Detection detectDirectory(Path dir, String path) throws IOException {
        List<String> names = regularFileNames(dir);
        boolean adapter = isPeftAdapter(names);
        boolean config = names.contains(CONFIG_JSON) || names.contains(PARAMS_JSON);
        boolean anySafetensors = names.stream().anyMatch(n -> n.endsWith(SAFETENSORS_SUFFIX));
        if (anySafetensors && !config && detectorMatches(dir, path)) {
            return new Detection(Kind.SAFETENSORS_DIR, path);
        }
        int safetensors = 0;
        int modelSafetensors = 0;
        List<String> ggufs = new ArrayList<>();
        boolean pickles = false;
        for (String name : names) {
            if (name.endsWith(SAFETENSORS_SUFFIX)) {
                safetensors++;
                if (!name.equals(ADAPTER_MODEL)) {
                    modelSafetensors++;
                }
            } else if (name.endsWith(GGUF_SUFFIX)) {
                ggufs.add(name);
            } else if (isPickle(name)) {
                pickles = true;
            }
        }
        if (adapter && !(config && modelSafetensors > 0)) {
            return new Detection(Kind.PEFT_ADAPTER, path);
        }
        if (safetensors > 0) {
            if (config) {
                return new Detection(Kind.SAFETENSORS_DIR, path);
            }
            throw new FileNotFoundException("'" + path + "' holds no model: " + LOOKED_FOR + "; it holds "
                    + safetensors + " .safetensors file(s) but neither " + CONFIG_JSON + " nor " + PARAMS_JSON
                    + ", and no checkpoint detector (" + String.join(", ", HfCheckpointDetectors.names())
                    + ") recognizes it");
        }
        if (!ggufs.isEmpty()) {
            String model = singleGgufModel(ggufs);
            if (model == null) {
                throw new FileNotFoundException("'" + path + "' holds " + ggufs.size() + " GGUF files that are not"
                        + " one split set (" + String.join(", ", ggufs) + "); pass one of them");
            }
            return new Detection(Kind.GGUF, dir.resolve(model).toString());
        }
        if (pickles) {
            throw pickle(path);
        }
        throw new FileNotFoundException("'" + path + "' holds no model: " + LOOKED_FOR);
    }

    /**
     * Whether a checkpoint detector recognizes {@code dir}.
     *
     * @throws FileNotFoundException when a marker file or safetensors header a detector reads is invalid, with that
     *                               failure as the cause
     */
    private static boolean detectorMatches(Path dir, String path) throws FileNotFoundException {
        try {
            return HfCheckpointDetectors.match(dir).isPresent();
        } catch (IOException e) {
            FileNotFoundException f = new FileNotFoundException("'" + path + "' holds no model: " + LOOKED_FOR
                    + "; it has neither " + CONFIG_JSON + " nor " + PARAMS_JSON + ", and checking its files against"
                    + " the known checkpoint layouts failed: " + e.getMessage());
            f.initCause(e);
            throw f;
        }
    }

    /** Names of the regular files in {@code dir} (links followed), sorted. */
    private static List<String> regularFileNames(Path dir) throws IOException {
        List<String> names = new ArrayList<>();
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(dir)) {
            for (Path entry : entries) {
                if (Files.isRegularFile(entry)) {
                    names.add(entry.getFileName().toString());
                }
            }
        }
        Collections.sort(names);
        return names;
    }

    /**
     * The GGUF file to open among {@code ggufs}: the only one, or the first split of the one split set they form;
     * null otherwise.
     */
    private static String singleGgufModel(List<String> ggufs) {
        if (ggufs.size() == 1) {
            return ggufs.get(0);
        }
        String prefix = null;
        String count = null;
        String first = null;
        for (String name : ggufs) {
            Matcher m = SPLIT_NAME.matcher(name);
            if (!m.matches()) {
                return null;
            }
            if (prefix == null) {
                prefix = m.group(1);
                count = m.group(3);
            } else if (!prefix.equals(m.group(1)) || !count.equals(m.group(3))) {
                return null;
            }
            if (Integer.parseInt(m.group(2)) == 1) {
                first = name;
            }
        }
        return first;
    }

    private static boolean hasGgufMagic(Path file) throws IOException {
        byte[] magic = GGUFFileParser.MAGIC.getBytes(StandardCharsets.US_ASCII);
        byte[] head = new byte[magic.length];
        int n = 0;
        try (InputStream in = Files.newInputStream(file)) {
            while (n < head.length) {
                int r = in.read(head, n, head.length - n);
                if (r < 0) {
                    return false;
                }
                n += r;
            }
        }
        for (int i = 0; i < magic.length; i++) {
            if (head[i] != magic[i]) {
                return false;
            }
        }
        return true;
    }

    /** Whether a directory listing holds a PEFT adapter's configuration and weights. */
    private static boolean isPeftAdapter(List<String> names) {
        return names.contains(ADAPTER_CONFIG) && names.contains(ADAPTER_MODEL);
    }

    private static boolean isPickle(String name) {
        return (name.startsWith(PICKLE_PREFIX) && name.endsWith(".bin")) || name.endsWith(".pt")
                || name.endsWith(".pth");
    }

    private static IOException pickle(String path) {
        return new IOException("Invalid checkpoint '" + path + "': PyTorch pickle checkpoints are not supported; use"
                + " the model's safetensors files");
    }

    // ------------------------------------------------------------------ sharded checkpoints

    /**
     * Refuses a {@code .safetensors} file that an index beside it ({@code *.safetensors.index.json}) lists: it holds
     * only part of the checkpoint's tensors.
     */
    private static void rejectIndexedShard(String path) throws IOException {
        Path file = Paths.get(path);
        Path dir = file.toAbsolutePath().getParent();
        String name = file.getFileName().toString();
        for (String entry : regularFileNames(dir)) {
            if (entry.endsWith(INDEX_SUFFIX) && indexLists(dir.resolve(entry), name)) {
                throw new IOException("'" + path + "' is one shard of a sharded checkpoint; pass the directory '"
                        + dir + "'");
            }
        }
    }

    /**
     * Whether a JSON string of the index file equals {@code fileName}. The index maps tensor names to the file
     * names holding them, so a string equal to a file name is one of its file references.
     *
     * @throws IOException when the index is not well-formed JSON text at the string level
     */
    private static boolean indexLists(Path index, String fileName) throws IOException {
        try (Reader in = new BufferedReader(new InputStreamReader(Files.newInputStream(index),
                StandardCharsets.UTF_8))) {
            StringBuilder s = new StringBuilder(fileName.length());
            int c;
            while ((c = in.read()) >= 0) {
                if (c != '"') {
                    continue;
                }
                s.setLength(0);
                boolean tooLong = false;
                while ((c = in.read()) != '"') {
                    if (c < 0) {
                        throw invalidIndex(index, "a string is not terminated");
                    }
                    if (c == '\\') {
                        c = unescape(index, in);
                    } else if (c < 0x20) {
                        throw invalidIndex(index, "a string holds the control character U+"
                                + String.format(Locale.ROOT, "%04X", c));
                    }
                    if (s.length() < fileName.length()) {
                        s.append((char) c);
                    } else {
                        tooLong = true;
                    }
                }
                if (!tooLong && fileName.contentEquals(s)) {
                    return true;
                }
            }
            return false;
        }
    }

    /** The character of the escape sequence after a backslash inside a string. */
    private static int unescape(Path index, Reader in) throws IOException {
        int e = in.read();
        switch (e) {
            case '"':
            case '\\':
            case '/':
                return e;
            case 'b':
                return '\b';
            case 'f':
                return '\f';
            case 'n':
                return '\n';
            case 'r':
                return '\r';
            case 't':
                return '\t';
            case 'u': {
                int v = 0;
                for (int i = 0; i < 4; i++) {
                    int h = hexDigit(in.read());
                    if (h < 0) {
                        throw invalidIndex(index, "a string holds a malformed \\u escape");
                    }
                    v = v << 4 | h;
                }
                return v;
            }
            case -1:
                throw invalidIndex(index, "a string is not terminated");
            default:
                throw invalidIndex(index, "a string holds the invalid escape \\" + (char) e);
        }
    }

    /** The value of an ASCII hexadecimal digit, or -1 for any other character (JSON allows no others). */
    private static int hexDigit(int c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return -1;
    }

    private static IOException invalidIndex(Path index, String detail) {
        return new IOException("Invalid safetensors file '" + index + "': " + detail);
    }
}
