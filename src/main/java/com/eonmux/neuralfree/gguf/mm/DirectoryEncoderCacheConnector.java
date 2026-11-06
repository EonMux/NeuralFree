package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaKey;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Objects;

/**
 * An {@link EncoderCacheConnector} over a directory every server of the deployment can reach (a shared filesystem,
 * the counterpart of vLLM's {@code ECExampleConnector} with its {@code shared_storage_path}): one entry file per
 * item, named by the SHA-256 of the item's {@link MediaKey} handle, holding the item's
 * {@link EncoderCacheEntry} in the {@link EncoderCacheFormat} form.
 *
 * <p>A producer writes an entry after it encodes the item (to a temporary file in the same directory, then moved
 * into place, so a consumer never reads a half-written entry); a consumer looks the file up before it decodes or
 * encodes the item. A file another format or a truncated copy left behind is refused by name rather than fed to the
 * prompt. Files are held as given: nothing evicts them, so the deployment decides their lifetime (vLLM's example
 * connector leaves that to its storage too). Thread-safe.
 */
public final class DirectoryEncoderCacheConnector implements EncoderCacheConnector {

    /** The extension of an entry file. */
    public static final String EXTENSION = ".nfec";

    private final Path directory;
    private final EncoderCacheConnector.Role role;

    /**
     * A connector over {@code directory}, created when it does not exist.
     *
     * @throws IOException              when the directory cannot be created or is not a directory
     * @throws IllegalArgumentException when {@code role} is null
     */
    public DirectoryEncoderCacheConnector(Path directory, EncoderCacheConnector.Role role) throws IOException {
        this.directory = Objects.requireNonNull(directory, "directory").toAbsolutePath();
        this.role = Objects.requireNonNull(role, "role");
        Files.createDirectories(this.directory);
        if (!Files.isDirectory(this.directory)) {
            throw new IOException("the encoder cache directory " + this.directory + " is not a directory");
        }
    }

    /** The directory the entries live in. */
    public Path directory() {
        return directory;
    }

    @Override
    public EncoderCacheConnector.Role role() {
        return role;
    }

    @Override
    public boolean has(MediaKey key) {
        return Files.isRegularFile(entryFile(Objects.requireNonNull(key, "key")));
    }

    @Override
    public EncoderCacheEntry get(MediaKey key) throws IOException {
        Objects.requireNonNull(key, "key");
        Path file = entryFile(key);
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(file);
        } catch (NoSuchFileException e) {
            return null;
        }
        try {
            return EncoderCacheFormat.decode(key.toString(), bytes);
        } catch (IOException e) {
            throw new IOException("the encoder cache entry " + file + " of item " + key + " is unusable: "
                    + e.getMessage(), e);
        }
    }

    @Override
    public void put(MediaKey key, EncoderCacheEntry entry) throws IOException {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(entry, "entry");
        if (!isProducer()) {
            throw new IllegalArgumentException("this encoder cache connector is a " + role.label()
                    + " and saves nothing");
        }
        Path target = entryFile(key);
        byte[] bytes = EncoderCacheFormat.encode(key.toString(), entry);
        Path temp = Files.createTempFile(directory, ".entry-", ".tmp");
        try {
            Files.write(temp, bytes);
            move(temp, target);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    /** Moves {@code temp} over {@code target}, atomically where the filesystem supports it. */
    private static void move(Path temp, Path target) throws IOException {
        try {
            Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** The file of one item's entry: the SHA-256 of its key handle, so any key names a portable file name. */
    private Path entryFile(MediaKey key) {
        String handle = key.toString();
        String digest = MediaKey.hex(MediaKey.sha256().digest(handle.getBytes(StandardCharsets.UTF_8)));
        return directory.resolve(digest + EXTENSION);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "DirectoryEncoderCacheConnector{%s, role=%s}", directory, role.label());
    }
}
