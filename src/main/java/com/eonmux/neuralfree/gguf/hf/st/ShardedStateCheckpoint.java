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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * A vLLM checkpoint saved by {@code ShardedStateLoader.save_model}: {@code model-rank-<rank>-part-<part>.safetensors}
 * or {@code model-pp-<pp>-tp-<tp>-part-<part>.safetensors} files holding each worker's own state dict.
 *
 * <p>The container reads every worker's safetensors parts, reassembles the whole model with
 * {@link VllmShardAssembly} (column-, row-, vocabulary-parallel and replicated parameters, including the vocabulary
 * padding) and maps vLLM's packed module names back to Hugging Face names with {@link VllmStateDict}. Single-file
 * checkpoints and directories holding both layouts read the same way as long as every file names a worker.</p>
 */
public final class ShardedStateCheckpoint implements CheckpointContainer {

    /** The built-in factory: accepts a file set with a pre-sharded vLLM name. */
    public static final CheckpointContainers.Factory FACTORY = new CheckpointContainers.Factory() {
        @Override
        public boolean accepts(Path dir, List<Path> files) {
            for (Path f : files) {
                if (VllmShardFiles.isPreSharded(f.getFileName().toString())) {
                    return true;
                }
            }
            return false;
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
            return "vLLM sharded_state";
        }
    };

    private final VllmStoredCheckpoint delegate;

    private ShardedStateCheckpoint(VllmStoredCheckpoint delegate) {
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

    private static CheckpointContainer openCheckpoint(Path path, Path configDir, HfConfig config)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(configDir, "configDir");
        Path dir;
        if (Files.isDirectory(path)) {
            dir = path.toAbsolutePath().normalize();
        } else if (Files.isRegularFile(path)) {
            // a single file names one worker: the whole set is its directory
            dir = path.toAbsolutePath().normalize().getParent();
        } else {
            throw new FileNotFoundException("pre-sharded checkpoint '" + path + "' does not exist");
        }
        HfConfig cfg = config != null ? config : HfConfigLoader.load(configDir, ModelSourceOptions.defaults());
        Map<Integer, Map<Integer, TreeMap<Integer, Path>>> grid = new TreeMap<>();
        for (Path file : CheckpointContainers.regularFiles(dir)) {
            VllmShardFiles.ShardFile shard = VllmShardFiles.parse(file.getFileName().toString()).orElse(null);
            if (shard == null) {
                continue;
            }
            TreeMap<Integer, Path> parts = grid
                    .computeIfAbsent(shard.pipeline(), k -> new TreeMap<>())
                    .computeIfAbsent(shard.tensor(), k -> new TreeMap<>());
            Path prev = parts.put(shard.part(), file);
            if (prev != null) {
                throw new IOException("Invalid vLLM checkpoint '" + dir + "': part " + shard.part() + " of rank pp="
                        + shard.pipeline() + ", tp=" + shard.tensor() + " appears twice ('" + prev + "' and '"
                        + file + "')");
            }
        }
        if (grid.isEmpty()) {
            throw new IOException("Invalid vLLM checkpoint '" + dir + "': it holds no file named"
                    + " model-rank-<rank>-part-<part>.safetensors or model-pp-<pp>-tp-<tp>-part-<part>.safetensors");
        }
        List<SafetensorsFile> opened = new ArrayList<>();
        try {
            List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
            for (Map.Entry<Integer, Map<Integer, TreeMap<Integer, Path>>> stage : grid.entrySet()) {
                for (Map.Entry<Integer, TreeMap<Integer, Path>> worker : stage.getValue().entrySet()) {
                    Map<String, StoredTensor> tensors = new LinkedHashMap<>();
                    for (Path part : worker.getValue().values()) {
                        SafetensorsFile file = SafetensorsFile.open(part);
                        opened.add(file);
                        for (StTensor t : file.tensors()) {
                            StoredTensor prev = tensors.put(t.name(), StoredTensor.of(t.name(),
                                    StRowSource.of(file, t)));
                            if (prev != null) {
                                throw new IOException("Invalid vLLM checkpoint '" + dir + "': tensor '" + t.name()
                                        + "' appears in two parts of rank pp=" + stage.getKey() + ", tp="
                                        + worker.getKey());
                            }
                        }
                    }
                    ranks.add(new VllmShardAssembly.Rank(stage.getKey(), worker.getKey(), tensors));
                }
            }
            String what = "vLLM pre-sharded checkpoint '" + dir + "'";
            List<StoredTensor> reassembled = VllmShardAssembly.assemble(ranks, cfg, what);
            List<StoredTensor> mapped = VllmStateDict.toHf(reassembled, cfg);
            String description = what + " (" + grid.size() + " pipeline stage" + (grid.size() == 1 ? "" : "s")
                    + ", " + ranks.size() / grid.size() + " tensor-parallel rank"
                    + (ranks.size() / grid.size() == 1 ? "" : "s") + ", " + mapped.size() + " tensors)";
            return new ShardedStateCheckpoint(new VllmStoredCheckpoint(description, cfg.dir(), mapped, opened));
        } catch (IOException | RuntimeException e) {
            for (SafetensorsFile f : opened) {
                try {
                    f.close();
                } catch (IOException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        }
    }
}
