package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
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

/**
 * A checkpoint container whose tensors are already mapped and served by {@link RowSource}s: the common body of the
 * vLLM state-dict containers (pre-sharded safetensors and tensorizer). Every tensor is produced, so the checkpoint
 * has no data segments and no tensor is mapped in place; the sources read their files positionally. Side files of the
 * model directory are opened as safetensors, as {@link SafetensorsCheckpoint} does. Immutable except for the side
 * files it opens; thread-safe.
 */
public final class VllmStoredCheckpoint implements CheckpointContainer {

    /** Largest side file opened. */
    public static final long MAX_SIDE_FILE_BYTES = 4L << 30;

    private final String description;
    private final Path modelDir;
    private final List<StoredTensor> stored;
    private final List<StTensor> tensors;
    private final Map<String, StoredTensor> byName;
    private final Map<String, StTensor> infosByName;
    private final List<AutoCloseable> resources;
    private final List<SafetensorsFile> sideFiles = new ArrayList<>();
    private boolean closed;

    /**
     * @param description what was opened, for messages
     * @param modelDir    the model directory that holds side files
     * @param stored      the mapped tensors, in checkpoint order
     * @param resources   the files and channels to close with the container
     * @throws IllegalArgumentException when two tensors share a name
     */
    public VllmStoredCheckpoint(String description, Path modelDir, List<StoredTensor> stored,
                                List<? extends AutoCloseable> resources) {
        this.description = Objects.requireNonNull(description, "description");
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
        this.stored = Collections.unmodifiableList(new ArrayList<>(stored));
        this.resources = Collections.unmodifiableList(new ArrayList<>(resources));
        Map<String, StoredTensor> m = new LinkedHashMap<>();
        Map<String, StTensor> infoMap = new LinkedHashMap<>();
        List<StTensor> infos = new ArrayList<>(stored.size());
        for (StoredTensor t : stored) {
            StoredTensor prev = m.put(t.name(), t);
            if (prev != null) {
                throw new IllegalArgumentException("tensor '" + t.name() + "' appears twice in " + description);
            }
            StTensor info = info(t);
            infos.add(info);
            infoMap.put(t.name(), info);
        }
        this.byName = m;
        this.infosByName = infoMap;
        this.tensors = Collections.unmodifiableList(infos);
    }

    private static StTensor info(StoredTensor t) {
        long numel = 1;
        for (long d : t.shape()) {
            numel = Math.multiplyExact(numel, d);
        }
        long bits = Math.multiplyExact(numel, (long) t.dtype().bits());
        if (bits % 8 != 0) {
            throw new IllegalArgumentException("tensor '" + t.name() + "' is not a whole number of bytes");
        }
        return new StTensor(t.name(), t.dtype(), t.shape(), 0, bits / 8);
    }

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
        Objects.requireNonNull(name, "name");
        return byName.containsKey(name) ? infosByName.get(name) : null;
    }

    @Override
    public RowSource source(String name) {
        StoredTensor t = byName.get(Objects.requireNonNull(name, "name"));
        if (t == null) {
            throw new IllegalArgumentException(description + " has no tensor '" + name + "'");
        }
        return t.source();
    }

    @Override
    public List<DataSegment> segments() {
        return Collections.emptyList();
    }

    @Override
    public long directVirtualOffset(String name) {
        return source(name).directVirtualOffset();
    }

    @Override
    public SafetensorsFile openSideFile(String relPath) throws IOException {
        Objects.requireNonNull(relPath, "relPath");
        Path file;
        try {
            file = SafetensorsCheckpoint.resolveContentName(modelDir, relPath);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid Hugging Face checkpoint '" + modelDir + "': side file '" + relPath
                    + "' is not a file inside the model directory: " + e.getMessage());
        }
        if (!Files.isRegularFile(file)) {
            throw new FileNotFoundException("side file '" + file + "' of " + description + " does not exist");
        }
        long size = Files.size(file);
        if (size > MAX_SIDE_FILE_BYTES) {
            throw new IOException("Invalid safetensors file '" + file + "': the side file has " + size
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
        List<AutoCloseable> all;
        synchronized (sideFiles) {
            if (closed) {
                return;
            }
            closed = true;
            all = new ArrayList<>(sideFiles);
            all.addAll(resources);
        }
        IOException first = null;
        for (AutoCloseable c : all) {
            try {
                c.close();
            } catch (Exception e) {
                IOException io = e instanceof IOException ? (IOException) e
                        : new IOException("closing " + c + " of " + description + " failed", e);
                if (first == null) {
                    first = io;
                } else {
                    first.addSuppressed(io);
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
}
