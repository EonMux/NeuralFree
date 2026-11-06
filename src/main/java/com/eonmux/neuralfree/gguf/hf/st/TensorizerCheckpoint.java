package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * A CoreWeave tensorizer checkpoint, vLLM's {@code tensorizer} load format: {@code model.tensors} (or one file per
 * tensor-parallel rank) beside the model's configuration and tokenizer files.
 *
 * <p>{@link TensorizerFile} reads the serialized format; a file per rank is reassembled with
 * {@link VllmShardAssembly} (a {@code --path-to-tensors} template like {@code model-%04d.tensors} is the rank in the
 * file name), and vLLM's packed module names are mapped back to Hugging Face names with {@link VllmStateDict}. A file
 * holding a Hugging Face state dict reads through unchanged.</p>
 */
public final class TensorizerCheckpoint implements CheckpointContainer {

    /** The built-in factory: accepts a file set with a {@code *.tensors} file. */
    public static final CheckpointContainers.Factory FACTORY = new CheckpointContainers.Factory() {
        @Override
        public boolean accepts(Path dir, List<Path> files) {
            boolean tensorizer = false;
            for (Path f : files) {
                String name = f.getFileName().toString();
                if (name.endsWith(".safetensors") && !VllmShardFiles.isPreSharded(name)) {
                    // a safetensors checkpoint beside a tensorizer file is read as the safetensors checkpoint
                    return false;
                }
                if (name.endsWith(SUFFIX)) {
                    tensorizer = true;
                }
            }
            return tensorizer;
        }

        @Override
        public CheckpointContainer open(Path path, Path configDir, boolean mistralNative) throws IOException {
            return openCheckpoint(path, configDir, null);
        }

        @Override
        public CheckpointContainer open(Path path, Path configDir, boolean mistralNative, HfConfig config)
                throws IOException {
            return openCheckpoint(path, configDir, config);
        }

        @Override
        public String toString() {
            return "tensorizer";
        }
    };

    /** The suffix of tensorizer files. */
    static final String SUFFIX = ".tensors";

    private final VllmStoredCheckpoint delegate;

    private TensorizerCheckpoint(VllmStoredCheckpoint delegate) {
        this.delegate = delegate;
    }

    @Override
    public String description() {
        return delegate.description();
    }

    @Override
    public List<StTensor> tensors() {
        return delegate.tensors();
    }

    @Override
    public StTensor tensor(String name) {
        return delegate.tensor(name);
    }

    @Override
    public com.eonmux.neuralfree.gguf.hf.tensor.RowSource source(String name) {
        return delegate.source(name);
    }

    @Override
    public List<com.eonmux.neuralfree.gguf.DataSegment> segments() {
        return delegate.segments();
    }

    @Override
    public long directVirtualOffset(String name) {
        return delegate.directVirtualOffset(name);
    }

    @Override
    public SafetensorsFile openSideFile(String relPath) throws IOException {
        return delegate.openSideFile(relPath);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return delegate.toString();
    }

    static CheckpointContainer openCheckpoint(Path path, Path configDir, HfConfig config) throws IOException {
        return openCheckpoint(path, configDir, config, null);
    }

    /**
     * Opens a tensorizer checkpoint, decrypting its files with {@code key} when given (null takes the key from the
     * environment, as {@link TensorizerFile} documents).
     */
    static CheckpointContainer openCheckpoint(Path path, Path configDir, HfConfig config, byte[] key)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(configDir, "configDir");
        List<Path> files = tensorizerFiles(path);
        HfConfig cfg = config != null ? config : HfConfigLoader.load(configDir, ModelSourceOptions.defaults());
        List<TensorizerFile> opened = new ArrayList<>();
        try {
            List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
            if (files.size() == 1) {
                TensorizerFile file = TensorizerFile.open(files.get(0), key);
                opened.add(file);
                Map<String, StoredTensor> byName = new LinkedHashMap<>();
                for (StoredTensor t : file.tensors()) {
                    byName.put(t.name(), t);
                }
                ranks.add(new VllmShardAssembly.Rank(0, 0, byName));
            } else {
                TreeMap<Integer, Path> byRank = ranks(files, path);
                for (Map.Entry<Integer, Path> e : byRank.entrySet()) {
                    TensorizerFile file = TensorizerFile.open(e.getValue(), key);
                    opened.add(file);
                    Map<String, StoredTensor> byName = new LinkedHashMap<>();
                    for (StoredTensor t : file.tensors()) {
                        byName.put(t.name(), t);
                    }
                    ranks.add(new VllmShardAssembly.Rank(0, e.getKey(), byName));
                }
            }
            String what = "tensorizer checkpoint '" + path + "'";
            List<StoredTensor> reassembled = VllmShardAssembly.assemble(ranks, cfg, what);
            List<StoredTensor> mapped = VllmStateDict.toHf(reassembled, cfg);
            String description = what + " (" + files.size() + " file" + (files.size() == 1 ? "" : "s") + ", "
                    + mapped.size() + " tensors)";
            return new TensorizerCheckpoint(new VllmStoredCheckpoint(description, cfg.dir(), mapped, opened));
        } catch (IOException | RuntimeException e) {
            for (TensorizerFile f : opened) {
                try {
                    f.close();
                } catch (IOException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        }
    }

    /** The tensorizer files of the checkpoint: {@code path} itself, or every {@code *.tensors} file of a directory. */
    private static List<Path> tensorizerFiles(Path path) throws IOException {
        if (Files.isRegularFile(path)) {
            if (!path.getFileName().toString().endsWith(SUFFIX)) {
                throw new FileNotFoundException("'" + path + "' is not a " + SUFFIX
                        + " tensorizer file");
            }
            return Collections.singletonList(path);
        }
        if (!Files.isDirectory(path)) {
            throw new FileNotFoundException("tensorizer checkpoint '" + path + "' does not exist");
        }
        List<Path> out = new ArrayList<>();
        for (Path f : CheckpointContainers.regularFiles(path)) {
            if (f.getFileName().toString().endsWith(SUFFIX)) {
                out.add(f);
            }
        }
        if (out.isEmpty()) {
            throw new IOException("Invalid tensorizer checkpoint '" + path + "': it holds no "
                    + SUFFIX + " file");
        }
        return out;
    }

    /**
     * The rank each file of a sharded tensorizer checkpoint belongs to: the trailing digit run of its name, which
     * must be distinct and contiguous from 0.
     */
    private static TreeMap<Integer, Path> ranks(List<Path> files, Path path) throws IOException {
        TreeMap<Integer, Path> out = new TreeMap<>();
        for (Path f : files) {
            String name = f.getFileName().toString();
            String stem = name.substring(0, name.length() - SUFFIX.length());
            int at = stem.length();
            while (at > 0 && Character.isDigit(stem.charAt(at - 1))) {
                at--;
            }
            if (at == stem.length()) {
                throw new IOException("Invalid tensorizer checkpoint '" + path + "': '" + name + "' carries no"
                        + " tensor-parallel rank in its name, but the checkpoint holds " + files.size() + " files");
            }
            int rank;
            try {
                rank = Integer.parseInt(stem.substring(at));
            } catch (NumberFormatException e) {
                throw new IOException("Invalid tensorizer checkpoint '" + path + "': '" + name + "' has an invalid"
                        + " rank", e);
            }
            Path prev = out.put(rank, f);
            if (prev != null) {
                throw new IOException("Invalid tensorizer checkpoint '" + path + "': rank " + rank + " appears in"
                        + " both '" + prev + "' and '" + f + "'");
            }
        }
        for (int i = 0; i < out.size(); i++) {
            if (!out.containsKey(i)) {
                throw new IOException("Invalid tensorizer checkpoint '" + path + "': rank " + i + " is missing");
            }
        }
        return out;
    }
}
