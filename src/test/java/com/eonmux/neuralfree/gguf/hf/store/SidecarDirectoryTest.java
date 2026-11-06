package com.eonmux.neuralfree.gguf.hf.store;

import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.FileAttributeView;
import java.nio.file.attribute.FileStoreAttributeView;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.regex.Pattern;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SidecarDirectory}: a private run directory per process, sidecar files created new and deleted on close,
 * abandoned run directories removed when their lock is free, and the free-space check.
 */
public class SidecarDirectoryTest {

    private static final Pattern RUN_NAME = Pattern.compile("run-\\d+-[0-9a-f]{16}");

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path cache() {
        return tmp.getRoot().toPath().resolve("cache");
    }

    private static boolean posix(Path p) {
        return p.getFileSystem().supportedFileAttributeViews().contains("posix");
    }

    /**
     * A path whose attributes are those of the open sidecar file {@code file}: the file itself while its name exists,
     * or, where {@code DELETE_ON_CLOSE} unlinks the name at open (the JDK on Unix), this process's descriptor link
     * of the unlinked file under {@code /proc/self/fd}; null when the name is gone and the platform has no
     * {@code /proc} to reach the open file through.
     */
    private static Path openedFile(Path file) throws IOException {
        if (Files.exists(file)) {
            return file;
        }
        Path fds = Paths.get("/proc/self/fd");
        if (!Files.isDirectory(fds)) {
            return null;
        }
        String deleted = file + " (deleted)";
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(fds)) {
            for (Path fd : entries) {
                String target;
                try {
                    target = Files.readSymbolicLink(fd).toString();
                } catch (IOException e) {
                    continue; // the descriptor of the directory stream itself closes while it is read
                }
                if (target.equals(deleted)) {
                    return fd;
                }
            }
        }
        throw new AssertionError("the open sidecar " + file + " has neither its name nor a descriptor under " + fds);
    }

    private static byte[] ramp(int n, int first) {
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) (first + i);
        }
        return b;
    }

    /** A file store that reports {@code usable} free bytes. */
    private static final class FakeStore extends FileStore {
        private final long usable;

        FakeStore(long usable) {
            this.usable = usable;
        }

        @Override
        public String name() {
            return "fake";
        }

        @Override
        public String type() {
            return "fake";
        }

        @Override
        public boolean isReadOnly() {
            return false;
        }

        @Override
        public long getTotalSpace() {
            return usable;
        }

        @Override
        public long getUsableSpace() {
            return usable;
        }

        @Override
        public long getUnallocatedSpace() {
            return usable;
        }

        @Override
        public boolean supportsFileAttributeView(Class<? extends FileAttributeView> type) {
            return false;
        }

        @Override
        public boolean supportsFileAttributeView(String name) {
            return false;
        }

        @Override
        public <V extends FileStoreAttributeView> V getFileStoreAttributeView(Class<V> type) {
            return null;
        }

        @Override
        public Object getAttribute(String attribute) {
            throw new UnsupportedOperationException(attribute);
        }
    }

    @Test
    public void theRunDirectoryOpensOnTheFirstSidecarAndIsPrivate() throws IOException {
        try (SidecarDirectory d = new SidecarDirectory(cache())) {
            assertNull("nothing is created before a sidecar is needed", d.directory());
            assertFalse(Files.exists(cache()));
            try (SidecarDirectory.Sidecar s = d.create("blk.0.ffn_down_exps.weight", 16)) {
                Path run = d.directory();
                assertEquals(cache().toAbsolutePath().normalize(), run.getParent());
                assertTrue(run.getFileName().toString(), RUN_NAME.matcher(run.getFileName().toString()).matches());
                assertTrue(run.getFileName().toString().startsWith(
                        SidecarDirectory.RUN_PREFIX + ProcessHandle.current().pid() + "-"));
                assertTrue(Files.isRegularFile(run.resolve(SidecarDirectory.LOCK_FILE)));
                assertEquals(run, s.path().getParent());
                if (posix(run)) {
                    assertEquals("rwx------", PosixFilePermissions.toString(Files.getPosixFilePermissions(run)));
                    Path file = openedFile(s.path());
                    if (file != null) {
                        assertEquals("rw-------", PosixFilePermissions.toString(Files.getPosixFilePermissions(file)));
                    }
                }
            }
        }
    }

    @Test
    public void twoDirectoriesNeverShareARunDirectory() throws IOException {
        try (SidecarDirectory a = new SidecarDirectory(cache());
             SidecarDirectory b = new SidecarDirectory(cache())) {
            a.create("a", 1).close();
            b.create("b", 1).close();
            assertNotEquals(a.directory(), b.directory());
            assertTrue("a live directory's lock is held, so it is not removed", Files.isDirectory(a.directory()));
        }
    }

    @Test
    public void sidecarsAreWrittenSealedAndRead() throws IOException {
        try (SidecarDirectory d = new SidecarDirectory(cache())) {
            SidecarDirectory.Sidecar s = d.create("t", 100);
            byte[] data = ramp(100, 7);
            s.write(0, ByteBuffer.wrap(data, 0, 60));
            ByteBuffer direct = ByteBuffer.allocateDirect(40);
            direct.put(data, 60, 40).flip();
            s.write(60, direct);
            assertThrows("reads need a sealed file", IllegalStateException.class,
                    () -> s.copy(0, 1, ByteBuffer.allocate(1)));
            assertThrows("writes past the size fail", IOException.class, () -> s.write(90, ByteBuffer.allocate(11)));
            s.seal();
            assertThrows(IllegalStateException.class, () -> s.write(0, ByteBuffer.allocate(1)));

            ByteBuffer out = ByteBuffer.allocate(30);
            s.copy(35, 30, out);
            byte[] expect = new byte[30];
            System.arraycopy(data, 35, expect, 0, 30);
            assertArrayEquals(expect, out.array());

            ByteBuffer slice = s.slice(10, 8);
            assertEquals(ByteOrder.LITTLE_ENDIAN, slice.order());
            assertEquals(0, slice.position());
            assertEquals(8, slice.remaining());
            assertEquals(data[10], slice.get(0));
            assertEquals(0, s.slice(100, 0).remaining());
            assertThrows(IndexOutOfBoundsException.class, () -> s.slice(95, 6));

            Path file = s.path();
            s.close();
            assertFalse("a closed sidecar is deleted", Files.exists(file));
            assertEquals("the mapping stays readable", data[10], slice.get(0));
        }
    }

    @Test
    public void sidecarFilesAreCreatedNew() throws IOException {
        try (SidecarDirectory d = new SidecarDirectory(cache())) {
            d.create("first", 4).close();
            Path squatter = d.directory().resolve("t1.bin");
            Files.write(squatter, new byte[]{1});
            assertThrows(FileAlreadyExistsException.class, () -> d.create("second", 4));
            assertArrayEquals("an existing file is never reused or truncated", new byte[]{1},
                    Files.readAllBytes(squatter));
            Files.delete(squatter);
        }
    }

    @Test
    public void closeDeletesTheRunDirectoryAndRefusesNewSidecars() throws IOException {
        SidecarDirectory d = new SidecarDirectory(cache());
        SidecarDirectory.Sidecar open = d.create("x", 8);
        open.write(0, ByteBuffer.allocate(8));
        Path run = d.directory();
        d.close();
        assertFalse(Files.exists(open.path()));
        assertFalse(Files.exists(run));
        assertNull(d.directory());
        IOException e = assertThrows(IOException.class, () -> d.create("y", 1));
        assertTrue(e.getMessage(), e.getMessage().contains("closed"));
        d.close();
    }

    @Test
    public void abandonedRunDirectoriesAreDeletedWhenTheirLockIsFree() throws IOException {
        Path stale = cache().resolve("run-1-0123456789abcdef");
        Files.createDirectories(stale);
        Files.write(stale.resolve(SidecarDirectory.LOCK_FILE), new byte[0]);
        Files.write(stale.resolve("t0.bin"), new byte[]{1, 2, 3});
        Path foreign = cache().resolve("run-2-fedcba9876543210");
        Files.createDirectories(foreign);
        Files.write(foreign.resolve("data.bin"), new byte[]{1});
        Path other = cache().resolve("models");
        Files.createDirectories(other);

        try (SidecarDirectory d = new SidecarDirectory(cache())) {
            d.create("t", 1).close();
            assertFalse("a run directory whose lock is free is gone", Files.exists(stale));
            assertTrue("a run directory without a lock file is not ours to delete", Files.exists(foreign));
            assertTrue(Files.exists(other));
        }
    }

    @Test
    public void aRunDirectoryWhoseLockIsHeldSurvives() throws IOException {
        Path held = cache().resolve("run-3-0000000000000003");
        Files.createDirectories(held);
        Path lockFile = held.resolve(SidecarDirectory.LOCK_FILE);
        Files.write(lockFile, new byte[0]);
        try (FileChannel ch = FileChannel.open(lockFile, StandardOpenOption.WRITE);
             java.nio.channels.FileLock lock = ch.lock()) {
            assertTrue(lock.isValid());
            assertFalse(SidecarDirectory.deleteIfAbandoned(held));
            try (SidecarDirectory d = new SidecarDirectory(cache())) {
                d.create("t", 1).close();
                assertTrue(Files.exists(held));
            }
        }
        assertTrue("released, it is abandoned", SidecarDirectory.deleteIfAbandoned(held));
        assertFalse(Files.exists(held));
    }

    @Test
    public void tooLittleFreeSpaceNamesTheTensorTheSizeAndTheOption() throws IOException {
        try (SidecarDirectory d = new SidecarDirectory(cache(), p -> new FakeStore(1000))) {
            d.create("fits", 1000).close();
            IOException e = assertThrows(IOException.class, () -> d.create("blk.7.ffn_up_exps.weight", 1001));
            assertEquals("materializing blk.7.ffn_up_exps.weight needs 1001 bytes in " + d.directory()
                    + "; free space or pass --hf-cache-dir", e.getMessage());
        }
    }

    @Test
    public void aCacheDirectoryThatIsAFileFails() throws IOException {
        Path file = tmp.newFile("not-a-dir").toPath();
        Assume.assumeTrue(Files.isRegularFile(file));
        try (SidecarDirectory d = new SidecarDirectory(file)) {
            assertThrows(IOException.class, () -> d.create("t", 1));
        }
    }
}
