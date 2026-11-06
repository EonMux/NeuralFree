package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Registry of checkpoint container formats: the built-in factories ({@link #builtIns()}: safetensors, vLLM's
 * pre-sharded state dicts and tensorizer files, and the restricted pickle reader) merged with factories registered at
 * run time, asked in that order. The first factory that accepts a checkpoint opens it.
 * Registering the same factory twice fails, as a duplicate row fails the architecture tables.
 */
public final class CheckpointContainers {

    /** Recognizes and opens one container format. Implementations are stateless and thread-safe. */
    public interface Factory {

        /**
         * True when this format reads the checkpoint whose files are {@code files}.
         *
         * @param dir   the checkpoint directory, or the directory of a single checkpoint file
         * @param files the directory's regular files (symbolic links followed), or the single file
         */
        boolean accepts(Path dir, List<Path> files);

        /**
         * Opens the checkpoint.
         *
         * @param path          the checkpoint directory or a single checkpoint file
         * @param configDir     the directory of the model configuration, which holds side files
         * @param mistralNative true for a Mistral native checkpoint ({@code params.json}, {@code consolidated*})
         * @throws IOException when the checkpoint cannot be read or is invalid
         */
        CheckpointContainer open(Path path, Path configDir, boolean mistralNative) throws IOException;

        /**
         * Opens the checkpoint with the configuration the load already read, for formats whose tensor names depend
         * on it (vLLM's packed module state dicts). The default ignores the configuration and calls
         * {@link #open(Path, Path, boolean)}.
         *
         * @param config the checkpoint's configuration, or null when the load has none to pass
         */
        default CheckpointContainer open(Path path, Path configDir, boolean mistralNative, HfConfig config)
                throws IOException {
            return open(path, configDir, mistralNative);
        }
    }

    private static final List<Factory> REGISTERED = new CopyOnWriteArrayList<>();

    private CheckpointContainers() {
    }

    private static final class Holder {
        static final List<Factory> BUILT_INS = merge(builtIns(), Collections.emptyList());
    }

    /**
     * The container formats NeuralFree reads, in the order they are asked: safetensors, vLLM's pre-sharded and
     * tensorizer state dicts (their own file names and extensions, which safetensors never accepts) and the restricted
     * PyTorch pickle container.
     */
    static List<Factory> builtIns() {
        return Arrays.asList(SafetensorsCheckpoint.FACTORY,
                ShardedStateCheckpoint.FACTORY,
                TensorizerCheckpoint.FACTORY,
                com.eonmux.neuralfree.gguf.hf.torch.TorchCheckpointFactory.FACTORY);
    }

    /**
     * Adds a container format, asked after the built-in ones and those registered before it.
     *
     * @throws IllegalStateException when {@code factory} is already registered
     */
    public static void register(Factory factory) {
        Objects.requireNonNull(factory, "factory");
        synchronized (REGISTERED) {
            if (Holder.BUILT_INS.contains(factory) || REGISTERED.contains(factory)) {
                throw new IllegalStateException("checkpoint container factory " + factory + " is registered twice");
            }
            REGISTERED.add(factory);
        }
    }

    /**
     * Removes a factory added by {@link #register}; built-in factories stay.
     *
     * @return whether {@code factory} was registered
     */
    public static boolean unregister(Factory factory) {
        Objects.requireNonNull(factory, "factory");
        synchronized (REGISTERED) {
            return REGISTERED.remove(factory);
        }
    }

    /** Every factory in the order they are asked (unmodifiable snapshot). */
    public static List<Factory> factories() {
        return merge(Holder.BUILT_INS, new ArrayList<>(REGISTERED));
    }

    /**
     * The factory that reads the checkpoint at {@code path} (a directory or a single file; symbolic links are
     * followed), or null when none does.
     *
     * @throws FileNotFoundException when {@code path} is neither a regular file nor a directory
     * @throws IOException           when the directory cannot be listed
     */
    public static Factory forFile(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        Path dir;
        List<Path> files;
        if (Files.isDirectory(path)) {
            dir = path;
            files = regularFiles(path);
        } else if (Files.isRegularFile(path)) {
            Path parent = path.toAbsolutePath().getParent();
            dir = parent;
            files = Collections.singletonList(path);
        } else {
            throw new FileNotFoundException("checkpoint '" + path + "' is neither a regular file nor a directory");
        }
        for (Factory f : factories()) {
            if (f.accepts(dir, files)) {
                return f;
            }
        }
        return null;
    }

    /**
     * Opens the checkpoint at {@code path} with the first factory that accepts it.
     *
     * @param path          the checkpoint directory or a single checkpoint file
     * @param configDir     the directory of the model configuration, which holds side files
     * @param mistralNative true for a Mistral native checkpoint
     * @throws IOException when no factory accepts the checkpoint, or it cannot be read or is invalid
     */
    public static CheckpointContainer open(Path path, Path configDir, boolean mistralNative) throws IOException {
        return open(path, configDir, mistralNative, null);
    }

    /**
     * Opens the checkpoint at {@code path} with the first factory that accepts it, passing the load's configuration
     * to formats whose tensor names depend on it.
     *
     * @param config the checkpoint's configuration, or null when the caller has none
     * @throws IOException when no factory accepts the checkpoint, or it cannot be read or is invalid
     */
    public static CheckpointContainer open(Path path, Path configDir, boolean mistralNative, HfConfig config)
            throws IOException {
        Objects.requireNonNull(configDir, "configDir");
        Factory f = forFile(path);
        if (f == null) {
            throw new IOException("Invalid checkpoint '" + path + "': no checkpoint container reads it (looked for"
                    + " *.safetensors, *.tensors and vLLM's pre-sharded *.safetensors files)");
        }
        return f.open(path, configDir, mistralNative, config);
    }

    /** The regular files of {@code dir} (symbolic links followed), sorted by name. */
    static List<Path> regularFiles(Path dir) throws IOException {
        List<Path> out = new ArrayList<>();
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(dir)) {
            for (Path entry : entries) {
                if (Files.isRegularFile(entry)) {
                    out.add(entry);
                }
            }
        }
        out.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        return out;
    }

    private static List<Factory> merge(List<Factory> first, List<Factory> second) {
        List<Factory> out = new ArrayList<>(first.size() + second.size());
        for (List<Factory> part : Arrays.asList(first, second)) {
            for (Factory f : part) {
                if (out.contains(f)) {
                    throw new IllegalStateException("checkpoint container factory " + f + " is registered twice");
                }
                out.add(f);
            }
        }
        return Collections.unmodifiableList(out);
    }
}
