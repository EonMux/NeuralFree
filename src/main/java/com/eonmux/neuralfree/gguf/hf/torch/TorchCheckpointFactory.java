package com.eonmux.neuralfree.gguf.hf.torch;

import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * The built-in factory of {@link TorchCheckpoint}s: the container the checkpoint-container registry's rule 6
 * (H-K, row A6) asks for the pickle files ({@code pytorch_model*.bin}, {@code .pt}, {@code .pth}), as safetensors
 * is asked for its files first (HF.md's rule order: safetensors win). {@code accepts} holds when the directory
 * holds one of those files (or a shard index naming them) and no safetensors file; {@code open} dispatches into
 * {@link TorchCheckpoint#open}. HF.md's planning, quantization rewriting and hooks then apply unchanged; no HF
 * class beside this package and the registry's built-in list is touched, and {@code ModelSources}' detection only
 * forwards the rule-6 files.
 */
public final class TorchCheckpointFactory {

    /** The built-in factory of {@link TorchCheckpoint}s. */
    public static final CheckpointContainers.Factory FACTORY = new CheckpointContainers.Factory() {
        @Override
        public boolean accepts(Path dir, List<Path> files) {
            Objects.requireNonNull(dir, "dir");
            Objects.requireNonNull(files, "files");
            boolean pickles = false;
            for (Path f : files) {
                String name = f.getFileName().toString();
                if (name.endsWith(".safetensors")) {
                    return false; // safetensors win (this factory is asked after that one)
                }
                if (TorchCheckpoint.isModelPickle(name) || TorchCheckpoint.INDEX.equals(name)) {
                    pickles = true;
                }
            }
            return pickles;
        }

        @Override
        public CheckpointContainer open(Path path, Path configDir, boolean mistralNative) throws IOException {
            Objects.requireNonNull(path, "path");
            Objects.requireNonNull(configDir, "configDir");
            return TorchCheckpoint.open(path, configDir, mistralNative);
        }

        @Override
        public String toString() {
            return "torch pickle";
        }
    };

    private TorchCheckpointFactory() {
    }
}
