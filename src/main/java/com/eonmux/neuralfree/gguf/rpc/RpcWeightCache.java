package com.eonmux.neuralfree.gguf.rpc;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * The server-side weight cache of a remote compute server ({@code rpc-server --cache}, llama.cpp's rpc-server
 * {@code --cache}): payload bytes stored under their {@link RpcWire#fnv1a} content hash, one file per hash, so a
 * reconnecting client re-probing {@code WRITE_HASH} skips re-uploading the weights it already sent. A file holds
 * exactly the bytes of the upload it was stored from; two concurrent stores of one hash write distinct temporary
 * files and move them into place atomically, so a reader never sees a partial payload.
 *
 * <p>The directory is not evicted: the entries are the weights the clients served here use, and the operator wipes
 * the directory to reclaim the space. The cache is keyed by content only, so entries are shared by every connection
 * and client that upload the same bytes.</p>
 */
public final class RpcWeightCache {

    private final Path directory;

    /**
     * Opens or creates {@code directory} as the cache.
     *
     * @throws IllegalArgumentException naming the directory when it cannot be created or is not a directory
     */
    public RpcWeightCache(Path directory) {
        this.directory = Objects.requireNonNull(directory, "directory");
        try {
            Files.createDirectories(directory);
        } catch (IOException e) {
            throw new IllegalArgumentException("rpc-server weight cache directory '" + directory
                    + "' could not be created: " + e.getMessage(), e);
        }
        if (!Files.isDirectory(directory)) {
            throw new IllegalArgumentException("rpc-server weight cache directory '" + directory
                    + "' is not a directory");
        }
    }

    /** The cache directory. */
    public Path directory() {
        return directory;
    }

    /** The cached bytes of {@code hash}, or null on a miss. */
    public byte[] load(long hash) {
        Path file = directory.resolve(RpcWire.hashName(hash));
        try {
            return Files.readAllBytes(file);
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Stores {@code data} under {@code hash}; an existing entry is replaced. A collision (a different payload whose
     * hash equals the stored entry's) is overwritten by the newest store, as any content-addressed store does.
     *
     * @throws IllegalStateException when the write or the atomic move fails
     */
    public void store(long hash, byte[] data) {
        Objects.requireNonNull(data, "data");
        Path target = directory.resolve(RpcWire.hashName(hash));
        Path temp = null;
        try {
            temp = Files.createTempFile(directory, "upload-", ".tmp");
            Files.write(temp, data);
            try {
                Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
                temp = null;
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
                temp = null;
            }
        } catch (IOException e) {
            throw new IllegalStateException("rpc-server weight cache could not store '" + RpcWire.hashName(hash)
                    + "' in " + directory + ": " + e.getMessage(), e);
        } finally {
            if (temp != null) {
                try {
                    Files.deleteIfExists(temp);
                } catch (IOException ignored) {
                    // the failed store left a temporary file; the next store makes another
                }
            }
        }
    }

    /** The bytes of a cached payload whose stored size differs from {@code size}; removes it and returns true. */
    public boolean dropIfStale(long hash, long size) {
        Path file = directory.resolve(RpcWire.hashName(hash));
        try {
            if (Files.size(file) == size) {
                return false;
            }
            Files.deleteIfExists(file);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /** The entry count of the cache directory. */
    public long entries() {
        try (Stream<Path> files = Files.list(directory)) {
            return files.filter(Files::isRegularFile)
                    .filter(p -> !p.getFileName().toString().endsWith(".tmp"))
                    .count();
        } catch (IOException e) {
            return -1L;
        }
    }

    /** The bytes the cache entries occupy, or -1 when the directory cannot be read. */
    public long bytes() {
        try (Stream<Path> files = Files.list(directory)) {
            return files.filter(Files::isRegularFile)
                    .filter(p -> !p.getFileName().toString().endsWith(".tmp"))
                    .mapToLong(p -> {
                        try {
                            return Files.size(p);
                        } catch (IOException e) {
                            return 0L;
                        }
                    }).sum();
        } catch (IOException e) {
            return -1L;
        }
    }

    /** Deletes every entry, keeping the directory. Returns the number of entries removed. */
    public long clear() {
        try (Stream<Path> files = Files.list(directory)) {
            return files.sorted(Comparator.reverseOrder()).filter(Files::isRegularFile).mapToInt(p -> {
                try {
                    Files.deleteIfExists(p);
                    return 1;
                } catch (IOException e) {
                    return 0;
                }
            }).sum();
        } catch (IOException e) {
            return 0L;
        }
    }

    /**
     * The default cache directory: {@code $XDG_CACHE_HOME/neuralfree/rpc}, else {@code ~/.cache/neuralfree/rpc}.
     * The path is not created here; the constructor creates it.
     */
    public static Path defaultCacheDir() {
        String xdg = System.getenv("XDG_CACHE_HOME");
        Path base = xdg != null && !xdg.trim().isEmpty()
                ? Paths.get(xdg.trim())
                : Paths.get(System.getProperty("user.home"), ".cache");
        return base.resolve("neuralfree").resolve("rpc");
    }

    @Override
    public String toString() {
        return "RpcWeightCache{" + directory + ", " + entries() + " entries, "
                + (bytes() < 0 ? "?" : String.valueOf(bytes())) + " bytes}";
    }

    /** Service-safe equality over the directory path. */
    @Override
    public boolean equals(Object o) {
        return this == o || o instanceof RpcWeightCache && directory.equals(((RpcWeightCache) o).directory);
    }

    @Override
    public int hashCode() {
        return directory.hashCode();
    }
}
