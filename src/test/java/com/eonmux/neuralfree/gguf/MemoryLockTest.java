package com.eonmux.neuralfree.gguf;

import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MemoryLock}: direct and mapped buffers lock and unlock through libc, the locked pages show as locked in
 * the kernel's accounting, heap buffers leave it alone, and the readahead advice is accepted.
 */
public class MemoryLockTest {

    private static ByteBuffer direct(int bytes) {
        return ByteBuffer.allocateDirect(bytes);
    }

    @Test
    public void directAndMappedBuffersLockAndUnlock() throws IOException {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue("the host exposes mlock", MemoryLock.supported());
        ByteBuffer buffer = direct(4096);
        Assume.assumeTrue("the locked-memory limit allows 4 KB", MemoryLock.lock(buffer));
        assertTrue("a locked buffer unlocks", MemoryLock.unlock(buffer));

        Path file = Files.createTempFile("memory-lock", ".bin");
        try {
            Files.write(file, new byte[4096]);
            try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r");
                 FileChannel channel = raf.getChannel()) {
                MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, 4096);
                assertTrue("a mapped file region locks", MemoryLock.lock(mapped));
                assertTrue(MemoryLock.unlock(mapped));
            }
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theKernelReportsTheLockedPagesAsLocked() throws IOException {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue("the host exposes mlock", MemoryLock.supported());
        Path smaps = Path.of("/proc/self/smaps");
        Assume.assumeTrue("/proc/self/smaps is readable", Files.isReadable(smaps));
        ByteBuffer buffer = direct(64 * 1024);
        Assume.assumeTrue("the locked-memory limit allows 64 KB", MemoryLock.lock(buffer));
        try {
            Pointer address = Native.getDirectBufferPointer(buffer);
            Assume.assumeTrue("the JVM exposes the buffer's address", address != null);
            assertTrue("the pages are locked in the process's page tables",
                    hasLockedPages(smaps, Pointer.nativeValue(address)));
        } finally {
            MemoryLock.unlock(buffer);
        }
    }

    /** Whether the {@code smaps} block holding {@code address} reports locked pages. */
    private static boolean hasLockedPages(Path smaps, long address) throws IOException {
        boolean inBlock = false;
        boolean locked = false;
        for (String line : Files.readAllLines(smaps)) {
            if (isHeader(line)) {
                if (inBlock && locked) {
                    return true;
                }
                locked = false;
                int dash = line.indexOf('-');
                int space = line.indexOf(' ');
                long start = Long.parseUnsignedLong(line.substring(0, dash), 16);
                long end = Long.parseUnsignedLong(line.substring(dash + 1, space), 16);
                inBlock = address >= start && address < end;
            } else if (inBlock && line.startsWith("Locked:")) {
                String[] parts = line.trim().split("\\s+");
                locked = parts.length > 1 && Long.parseLong(parts[1]) > 0;
            }
        }
        return inBlock && locked;
    }

    /** Whether {@code line} is an {@code smaps} mapping header, {@code start-end perms ...}. */
    private static boolean isHeader(String line) {
        int dash = line.indexOf('-');
        int space = line.indexOf(' ');
        if (line.isEmpty() || Character.isWhitespace(line.charAt(0)) || dash <= 0 || space <= dash) {
            return false;
        }
        for (int i = 0; i < dash; i++) {
            if (Character.digit(line.charAt(i), 16) < 0) {
                return false;
            }
        }
        return true;
    }

    @Test
    public void heapBuffersAreLeftAlone() {
        Assume.assumeTrue(MemoryLock.supported());
        assertFalse("heap buffers cannot be addressed by mlock", MemoryLock.lock(ByteBuffer.allocate(4096)));
        assertFalse(MemoryLock.unlock(ByteBuffer.allocate(4096)));
    }

    @Test
    public void theReadaheadAdviceIsAccepted() {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue(MemoryLock.supported());
        assertTrue(MemoryLock.adviseRandom(direct(4096)));
        assertFalse("an empty buffer has nothing to advise", MemoryLock.adviseRandom(ByteBuffer.allocateDirect(0)));
    }

    @Test
    public void refusesNullBuffers() {
        assertThrows(NullPointerException.class, () -> MemoryLock.lock(null));
        assertThrows(NullPointerException.class, () -> MemoryLock.unlock(null));
        assertThrows(NullPointerException.class, () -> MemoryLock.adviseRandom(null));
    }
}
