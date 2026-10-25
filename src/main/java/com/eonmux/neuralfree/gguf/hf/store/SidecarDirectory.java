package com.eonmux.neuralfree.gguf.hf.store;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * The private, per-process directory that holds the DISK copies of derived tensors. Nothing in it is reused across
 * runs:
 * <ul>
 *   <li>it is {@code <cacheDir>/run-<pid>-<16 random hex digits>/}, made with {@link Files#createDirectory} (which
 *       fails when the name exists) and POSIX permissions {@code rwx------} where the file system supports them;</li>
 *   <li>it holds the lock file {@value #LOCK_FILE}, locked with {@link FileChannel#lock()} until the directory is
 *       closed, which marks the directory as in use;</li>
 *   <li>each sidecar file is created with {@code CREATE_NEW, READ, WRITE, DELETE_ON_CLOSE} and {@code rw-------};</li>
 *   <li>when the directory is opened, sibling {@code run-*} directories whose lock file can be locked belong to a
 *       process that is gone and are deleted;</li>
 *   <li>before a sidecar is created, the usable space of the directory's file store must cover it.</li>
 * </ul>
 * The directory opens on the first {@link #create} and is deleted by {@link #close()}. Thread-safe.
 */
public final class SidecarDirectory implements AutoCloseable {

    private static final Logger LOG = LogManager.getLogger(SidecarDirectory.class);

    /** Name prefix of every run directory. */
    public static final String RUN_PREFIX = "run-";
    /** The lock file of a run directory. */
    public static final String LOCK_FILE = ".lock";
    /** Largest mapped piece of a sidecar file. */
    static final long PIECE_BYTES = 1L << 30;
    /**
     * Largest slice of a heap buffer handed to one channel write, which bounds the temporary direct buffer the JDK
     * copies it through.
     */
    private static final int WRITE_SLICE = 8 << 20;

    private static final Set<PosixFilePermission> DIR_PERMISSIONS = PosixFilePermissions.fromString("rwx------");
    private static final Set<PosixFilePermission> FILE_PERMISSIONS = PosixFilePermissions.fromString("rw-------");
    private static final SecureRandom RANDOM = new SecureRandom();

    /** The file store of a path, for the free-space check. */
    @FunctionalInterface
    public interface FileStores {
        /** The file store {@code path} lives on. */
        FileStore of(Path path) throws IOException;
    }

    private final Path cacheDir;
    private final FileStores fileStores;
    private final Object lock = new Object();
    private final Set<Sidecar> open = new LinkedHashSet<>();
    private Path dir;
    private FileChannel lockChannel;
    private FileLock fileLock;
    private long created;
    private boolean closed;

    /** A directory under {@code cacheDir} whose free space is read from the default file stores. */
    public SidecarDirectory(Path cacheDir) {
        this(cacheDir, Files::getFileStore);
    }

    /** A directory under {@code cacheDir} whose free space is read from {@code fileStores}. */
    public SidecarDirectory(Path cacheDir, FileStores fileStores) {
        this.cacheDir = Objects.requireNonNull(cacheDir, "cacheDir").toAbsolutePath().normalize();
        this.fileStores = Objects.requireNonNull(fileStores, "fileStores");
    }

    /** The parent of the run directory. */
    public Path cacheDir() {
        return cacheDir;
    }

    /** The run directory, or null before the first sidecar was created or after {@link #close()}. */
    public Path directory() {
        synchronized (lock) {
            return dir;
        }
    }

    /**
     * Creates the sidecar file of {@code bytes} bytes for the tensor {@code tensor}, opening the run directory first
     * when this is the first sidecar.
     *
     * @throws IOException when the directory is closed or cannot be created, its file store has less usable space
     *                     than {@code bytes}, or the file cannot be created
     */
    public Sidecar create(String tensor, long bytes) throws IOException {
        Objects.requireNonNull(tensor, "tensor");
        if (bytes < 0) {
            throw new IllegalArgumentException("sidecar of " + tensor + " needs " + bytes + " bytes");
        }
        synchronized (lock) {
            if (closed) {
                throw new IOException("materializing " + tensor + ": the sidecar directory under " + cacheDir
                        + " is closed");
            }
            ensureOpen();
            long usable = fileStores.of(dir).getUsableSpace();
            if (usable < bytes) {
                throw new IOException("materializing " + tensor + " needs " + bytes + " bytes in " + dir
                        + "; free space or pass --hf-cache-dir");
            }
            Path file = dir.resolve("t" + created++ + ".bin");
            Set<StandardOpenOption> options = EnumSet.of(StandardOpenOption.CREATE_NEW, StandardOpenOption.READ,
                    StandardOpenOption.WRITE, StandardOpenOption.DELETE_ON_CLOSE);
            FileChannel ch = FileChannel.open(file, options, attributes(FILE_PERMISSIONS));
            Sidecar s = new Sidecar(this, file, ch, tensor, bytes);
            open.add(s);
            return s;
        }
    }

    private void ensureOpen() throws IOException {
        if (dir != null) {
            return;
        }
        Files.createDirectories(cacheDir);
        Path run = cacheDir.resolve(RUN_PREFIX + ProcessHandle.current().pid() + "-" + randomHex());
        Files.createDirectory(run, attributes(DIR_PERMISSIONS));
        FileChannel ch = null;
        try {
            ch = FileChannel.open(run.resolve(LOCK_FILE), EnumSet.of(StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE), attributes(FILE_PERMISSIONS));
            fileLock = ch.lock();
        } catch (IOException | RuntimeException e) {
            if (ch != null) {
                closeQuietly(ch, e);
            }
            deleteQuietly(run.resolve(LOCK_FILE), e);
            deleteQuietly(run, e);
            throw e;
        }
        lockChannel = ch;
        dir = run;
        LOG.info("Derived tensors materialize on disk under {}", run);
        deleteAbandoned(run);
    }

    /** The POSIX permission attribute when the cache directory's file system supports POSIX attributes. */
    private FileAttribute<?>[] attributes(Set<PosixFilePermission> permissions) {
        if (cacheDir.getFileSystem().supportedFileAttributeViews().contains("posix")) {
            return new FileAttribute<?>[] {PosixFilePermissions.asFileAttribute(permissions)};
        }
        return new FileAttribute<?>[0];
    }

    private static String randomHex() {
        byte[] b = new byte[8];
        RANDOM.nextBytes(b);
        StringBuilder sb = new StringBuilder(16);
        for (byte x : b) {
            sb.append(String.format(Locale.ROOT, "%02x", x & 0xFF));
        }
        return sb.toString();
    }

    /** Deletes the sibling run directories whose lock is free: their process is gone. */
    private void deleteAbandoned(Path own) {
        List<Path> runs = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(cacheDir, RUN_PREFIX + "*")) {
            for (Path p : ds) {
                if (!p.equals(own) && Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS)) {
                    runs.add(p);
                }
            }
        } catch (IOException e) {
            LOG.warn("Cannot list {} for abandoned sidecar directories: {}", cacheDir, e.toString());
            return;
        }
        for (Path run : runs) {
            try {
                if (deleteIfAbandoned(run)) {
                    LOG.info("Deleted the abandoned sidecar directory {}", run);
                }
            } catch (IOException e) {
                LOG.warn("Cannot delete the abandoned sidecar directory {}: {}", run, e.toString());
            }
        }
    }

    /** Deletes {@code run} when its lock file exists and can be locked; returns whether it was deleted. */
    static boolean deleteIfAbandoned(Path run) throws IOException {
        Path lockFile = run.resolve(LOCK_FILE);
        if (!Files.isRegularFile(lockFile, LinkOption.NOFOLLOW_LINKS)) {
            return false;
        }
        try (FileChannel ch = FileChannel.open(lockFile, StandardOpenOption.WRITE)) {
            FileLock held;
            try {
                held = ch.tryLock();
            } catch (OverlappingFileLockException inUseByThisProcess) {
                return false;
            }
            if (held == null) {
                return false;
            }
            try {
                deleteEntries(run, lockFile);
            } finally {
                held.release();
            }
        }
        Files.deleteIfExists(lockFile);
        Files.deleteIfExists(run);
        return true;
    }

    /** Deletes every regular file and link directly inside {@code run} except {@code keep}. */
    private static void deleteEntries(Path run, Path keep) throws IOException {
        List<Path> entries = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(run)) {
            for (Path p : ds) {
                if (!p.equals(keep)) {
                    entries.add(p);
                }
            }
        }
        for (Path p : entries) {
            if (Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS)) {
                throw new IOException("unexpected directory " + p + " in a sidecar directory");
            }
            Files.deleteIfExists(p);
        }
    }

    private void closed(Sidecar s) {
        synchronized (lock) {
            open.remove(s);
        }
    }

    /**
     * Closes every open sidecar (deleting its file), releases the lock and deletes the run directory. Later
     * {@link #create} calls fail.
     *
     * @throws IOException the first failure, with later ones suppressed
     */
    @Override
    public void close() throws IOException {
        List<Sidecar> files;
        Path run;
        synchronized (lock) {
            if (closed) {
                return;
            }
            closed = true;
            files = new ArrayList<>(open);
            run = dir;
            dir = null;
        }
        IOException first = null;
        for (Sidecar s : files) {
            try {
                s.close();
            } catch (IOException e) {
                first = add(first, e);
            }
        }
        if (run == null) {
            return;
        }
        try {
            fileLock.release();
        } catch (IOException e) {
            first = add(first, e);
        }
        try {
            lockChannel.close();
        } catch (IOException e) {
            first = add(first, e);
        }
        try {
            deleteEntries(run, run.resolve(LOCK_FILE));
            Files.deleteIfExists(run.resolve(LOCK_FILE));
            Files.deleteIfExists(run);
        } catch (IOException e) {
            first = add(first, e);
        }
        if (first != null) {
            throw first;
        }
    }

    private static IOException add(IOException first, IOException e) {
        if (first == null) {
            return e;
        }
        first.addSuppressed(e);
        return first;
    }

    private static void closeQuietly(Closeable c, Exception cause) {
        try {
            c.close();
        } catch (IOException e) {
            cause.addSuppressed(e);
        }
    }

    private static void deleteQuietly(Path p, Exception cause) {
        try {
            Files.deleteIfExists(p);
        } catch (IOException e) {
            cause.addSuppressed(e);
        }
    }

    @Override
    public String toString() {
        Path d = directory();
        return "SidecarDirectory{" + (d != null ? d : cacheDir + " (not opened)") + "}";
    }

    /**
     * One sidecar file: written once with {@link #write}, then {@link #seal sealed}, which maps it read-only in pieces
     * of at most 1 GiB. Reads copy from the mapped pieces, which stay readable after the file is closed. Closing deletes
     * the file. Thread-safe once sealed.
     */
    public static final class Sidecar implements Closeable {
        private final SidecarDirectory owner;
        private final Path path;
        private final FileChannel channel;
        private final String tensor;
        private final long size;
        private volatile MappedByteBuffer[] pieces;
        private boolean closed;

        private Sidecar(SidecarDirectory owner, Path path, FileChannel channel, String tensor, long size) {
            this.owner = owner;
            this.path = path;
            this.channel = channel;
            this.tensor = tensor;
            this.size = size;
        }

        /** The file. */
        public Path path() {
            return path;
        }

        /** The byte size. */
        public long size() {
            return size;
        }

        /**
         * Writes the remaining bytes of {@code src} at {@code position}.
         *
         * @throws IllegalStateException when the sidecar is sealed
         * @throws IOException           when the range exceeds the size or the write fails
         */
        public void write(long position, ByteBuffer src) throws IOException {
            if (pieces != null) {
                throw new IllegalStateException("sidecar of " + tensor + " is sealed");
            }
            if (position < 0 || src.remaining() > size - position) {
                throw new IOException("sidecar of " + tensor + ": " + src.remaining() + " bytes at " + position
                        + " exceed its " + size + " bytes");
            }
            long pos = position;
            int end = src.limit();
            try {
                while (src.position() < end) {
                    src.limit(src.isDirect() ? end : Math.min(end, src.position() + WRITE_SLICE));
                    while (src.hasRemaining()) {
                        pos += channel.write(src, pos);
                    }
                }
            } finally {
                src.limit(end);
            }
        }

        /** Maps the written file read-only; later writes fail. */
        public synchronized void seal() throws IOException {
            if (pieces != null) {
                return;
            }
            int n = (int) ((size + PIECE_BYTES - 1) / PIECE_BYTES);
            MappedByteBuffer[] maps = new MappedByteBuffer[n];
            for (int i = 0; i < n; i++) {
                long off = i * PIECE_BYTES;
                maps[i] = channel.map(FileChannel.MapMode.READ_ONLY, off, Math.min(PIECE_BYTES, size - off));
            }
            pieces = maps;
        }

        /**
         * Copies {@code length} bytes at {@code offset} into {@code dst} at its position, advancing it.
         *
         * @throws IllegalStateException when the sidecar is not sealed
         */
        public void copy(long offset, int length, ByteBuffer dst) {
            MappedByteBuffer[] maps = requireSealed();
            checkRange(offset, length);
            long pos = offset;
            int left = length;
            while (left > 0) {
                int piece = (int) (pos / PIECE_BYTES);
                int at = (int) (pos - piece * PIECE_BYTES);
                ByteBuffer src = maps[piece].duplicate();
                int n = Math.min(left, src.limit() - at);
                src.position(at).limit(at + n);
                dst.put(src);
                pos += n;
                left -= n;
            }
        }

        /**
         * The {@code length} bytes at {@code offset} as a little-endian buffer positioned at 0: a view of the mapped
         * piece when the range lies inside one, else a mapping of exactly the range.
         *
         * @throws IOException when a range spanning two pieces cannot be mapped (the file is closed)
         */
        public ByteBuffer slice(long offset, int length) throws IOException {
            MappedByteBuffer[] maps = requireSealed();
            checkRange(offset, length);
            if (length == 0) {
                return ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN);
            }
            int piece = (int) (offset / PIECE_BYTES);
            int at = (int) (offset - piece * PIECE_BYTES);
            if ((long) at + length <= maps[piece].limit()) {
                ByteBuffer b = maps[piece].duplicate();
                b.position(at).limit(at + length);
                return b.slice().order(ByteOrder.LITTLE_ENDIAN);
            }
            return channel.map(FileChannel.MapMode.READ_ONLY, offset, length).order(ByteOrder.LITTLE_ENDIAN);
        }

        private MappedByteBuffer[] requireSealed() {
            MappedByteBuffer[] maps = pieces;
            if (maps == null) {
                throw new IllegalStateException("sidecar of " + tensor + " is read before it is sealed");
            }
            return maps;
        }

        private void checkRange(long offset, int length) {
            if (offset < 0 || length < 0 || offset > size - length) {
                throw new IndexOutOfBoundsException("sidecar of " + tensor + ": bytes [" + offset + ", +" + length
                        + ") lie outside its " + size + " bytes");
            }
        }

        /** Closes and deletes the file; sealed pieces stay readable until they are no longer referenced. */
        @Override
        public void close() throws IOException {
            synchronized (this) {
                if (closed) {
                    return;
                }
                closed = true;
            }
            try {
                channel.close();
            } finally {
                owner.closed(this);
            }
        }

        @Override
        public String toString() {
            return "Sidecar{" + tensor + ", " + size + " bytes, " + path + "}";
        }
    }
}
