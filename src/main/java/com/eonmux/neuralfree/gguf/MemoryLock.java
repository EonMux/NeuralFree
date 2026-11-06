package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.utils.LoggingUtil;
import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pins mapped or direct memory in RAM with the C library's {@code mlock}, and sets the readahead advice with
 * {@code posix_madvise} (llama.cpp's {@code llama_mlock} and the {@code POSIX_MADV_RANDOM} it applies to a NUMA
 * model, {@code src/llama-mmap.cpp}).
 *
 * <p>A locked region stays in physical RAM, so the kernel neither swaps nor compresses it. The calls are reached
 * through JNA because the JVM exposes neither; a process that does not run on Linux, or whose C library lacks them,
 * reports {@link #supported()} false and {@link #lock(ByteBuffer)} returns false. A failed lock is logged once, with
 * the Linux remedy (raise {@code RLIMIT_MEMLOCK}, {@code ulimit -l}); the caller keeps running with pageable memory,
 * as llama.cpp does.
 */
public final class MemoryLock {

    /** {@code posix_madvise} advice: no readahead, the pattern a NUMA model's scattered reads want. */
    private static final int POSIX_MADV_RANDOM = 1;

    private static final AtomicBoolean LOCK_FAILED = new AtomicBoolean();
    private static final AtomicBoolean UNSUPPORTED = new AtomicBoolean();
    private static final AtomicBoolean ADVICE_FAILED = new AtomicBoolean();
    private static final AtomicBoolean NOT_DIRECT = new AtomicBoolean();

    private static final class Libc {
        static final Function MLOCK;
        static final Function MUNLOCK;
        static final Function POSIX_MADVISE;
        static final int PAGE_SIZE;

        static {
            Function lock = null;
            Function unlock = null;
            Function advise = null;
            int page = 4096;
            if (Platform.isLinux()) {
                NativeLibrary libc = null;
                try {
                    libc = NativeLibrary.getInstance(Platform.C_LIBRARY_NAME);
                    lock = libc.getFunction("mlock");
                    unlock = libc.getFunction("munlock");
                } catch (Throwable t) {
                    LoggingUtil.getLogger(MemoryLock.class).debug("mlock is unavailable: {}", t.toString());
                    lock = null;
                    unlock = null;
                }
                if (lock != null) {
                    try {
                        advise = libc.getFunction("posix_madvise");
                    } catch (Throwable t) {
                        advise = null;
                    }
                    try {
                        page = libc.getFunction("getpagesize").invokeInt(new Object[0]);
                    } catch (Throwable t) {
                        page = 4096;
                    }
                }
            }
            MLOCK = lock;
            MUNLOCK = unlock;
            POSIX_MADVISE = advise;
            PAGE_SIZE = page > 0 ? page : 4096;
        }
    }

    private MemoryLock() {
    }

    /** Whether {@code mlock} and {@code munlock} can run on this host. */
    public static boolean supported() {
        return Libc.MLOCK != null && Libc.MUNLOCK != null;
    }

    /**
     * Locks the whole buffer ({@link ByteBuffer#capacity()} bytes) in RAM. The pages stay locked until the buffer is
     * unmapped or {@link #unlock(ByteBuffer) unlocked}; a direct buffer's memory is unmapped by the JVM when the
     * buffer is collected.
     *
     * @return true when the kernel locked the pages, false for unsupported hosts, non-direct buffers and refused
     *         requests (logged once)
     */
    public static boolean lock(ByteBuffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        if (!supported()) {
            if (UNSUPPORTED.compareAndSet(false, true)) {
                LoggingUtil.getLogger(MemoryLock.class).warn(
                        "mlock is not supported on this host; the tensor pages stay pageable");
            }
            return false;
        }
        if (buffer.capacity() == 0) {
            return false;
        }
        Pointer address = address(buffer);
        if (address == null) {
            return false;
        }
        int rc = Libc.MLOCK.invokeInt(new Object[] {address, sizeT(buffer.capacity())});
        if (rc != 0) {
            if (LOCK_FAILED.compareAndSet(false, true)) {
                LoggingUtil.getLogger(MemoryLock.class).warn(
                        "mlock of {} MB failed (errno {}); the memory stays pageable. Raise the locked-memory limit"
                                + " ('ulimit -l', RLIMIT_MEMLOCK) to pin the model in RAM",
                        buffer.capacity() / (1024 * 1024), Native.getLastError());
            }
            return false;
        }
        LoggingUtil.getLogger(MemoryLock.class).debug("mlock: locked {} MB", buffer.capacity() / (1024 * 1024));
        return true;
    }

    /**
     * Releases the lock of a buffer locked by {@link #lock(ByteBuffer)}.
     *
     * @return true when the kernel released the pages
     */
    public static boolean unlock(ByteBuffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        if (!supported() || buffer.capacity() == 0) {
            return false;
        }
        Pointer address = address(buffer);
        if (address == null) {
            return false;
        }
        return Libc.MUNLOCK.invokeInt(new Object[] {address, sizeT(buffer.capacity())}) == 0;
    }

    /**
     * Tells the kernel the pages are read at random rather than sequentially, skipping readahead
     * ({@code POSIX_MADV_RANDOM}, llama.cpp applies it to a memory-mapped model on a NUMA host).
     *
     * @return true when the kernel took the advice
     */
    public static boolean adviseRandom(ByteBuffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        if (Libc.POSIX_MADVISE == null || buffer.capacity() == 0) {
            return false;
        }
        Pointer address = address(buffer);
        if (address == null) {
            return false;
        }
        // posix_madvise needs a page-aligned range; the buffer's whole pages are the pages it covers
        long page = Libc.PAGE_SIZE;
        long start = Pointer.nativeValue(address) & ~(page - 1);
        long end = (Pointer.nativeValue(address) + buffer.capacity()) & ~(page - 1);
        if (end <= start) {
            end = start + page;
        }
        int rc = Libc.POSIX_MADVISE.invokeInt(new Object[] {new Pointer(start), sizeT(end - start),
                Integer.valueOf(POSIX_MADV_RANDOM)});
        if (rc != 0) {
            if (ADVICE_FAILED.compareAndSet(false, true)) {
                LoggingUtil.getLogger(MemoryLock.class).warn(
                        "posix_madvise(POSIX_MADV_RANDOM) failed (errno {}); readahead stays as it is",
                        Native.getLastError());
            }
            return false;
        }
        return true;
    }

    /** The buffer's native address, or null (logged once) when the JVM does not expose it. */
    private static Pointer address(ByteBuffer buffer) {
        try {
            Pointer address = Native.getDirectBufferPointer(buffer);
            if (address == null && NOT_DIRECT.compareAndSet(false, true)) {
                LoggingUtil.getLogger(MemoryLock.class).warn(
                        "mlock needs direct or mapped memory; heap buffers stay pageable");
            }
            return address;
        } catch (RuntimeException | LinkageError e) {
            if (NOT_DIRECT.compareAndSet(false, true)) {
                LoggingUtil.getLogger(MemoryLock.class).warn("mlock cannot address direct buffers on this JVM: {}",
                        e.toString());
            }
            return null;
        }
    }

    /** {@code size_t} as a JNA argument: a 64-bit long on LP64, a 32-bit int elsewhere. */
    private static Object sizeT(long value) {
        return Native.SIZE_T_SIZE == Long.BYTES
                ? (Object) Long.valueOf(value)
                : (Object) Integer.valueOf((int) value);
    }
}
