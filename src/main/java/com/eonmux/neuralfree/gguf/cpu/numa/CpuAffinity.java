package com.eonmux.neuralfree.gguf.cpu.numa;

import com.eonmux.neuralfree.utils.LoggingUtil;
import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Platform;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The Linux CPU affinity calls the JVM does not expose, reached through {@code libc} with JNA:
 * {@code sched_getaffinity}, {@code sched_setaffinity} and {@code sched_getcpu}.
 *
 * <p>The affinity mask is a {@code cpu_set_t}: a bitset of the possible CPUs, sized from
 * {@code /sys/devices/system/cpu/possible} (at least {@value #MIN_MASK_BYTES} bytes). A process that does not run on
 * Linux, or whose C library lacks the calls, reports {@link #available()} false and every operation is a no-op.
 */
final class CpuAffinity {

    /** {@code cpu_set_t}'s size on a kernel built for at most 1024 CPUs. */
    private static final int MIN_MASK_BYTES = 128;

    private static final AtomicBoolean GET_FAILED = new AtomicBoolean();
    private static final AtomicBoolean SET_FAILED = new AtomicBoolean();

    private static final class Libc {
        static final Function GET_AFFINITY;
        static final Function SET_AFFINITY;
        static final Function GET_CPU;
        static final int MASK_BYTES;

        static {
            Function get = null;
            Function set = null;
            Function cpu = null;
            int bytes = MIN_MASK_BYTES;
            if (Platform.isLinux()) {
                NativeLibrary libc = null;
                try {
                    libc = NativeLibrary.getInstance(Platform.C_LIBRARY_NAME);
                    get = libc.getFunction("sched_getaffinity");
                    set = libc.getFunction("sched_setaffinity");
                    bytes = maskBytes();
                } catch (Throwable t) {
                    LoggingUtil.getLogger(CpuAffinity.class).debug(
                            "CPU affinity is unavailable: {}", t.toString());
                    get = null;
                    set = null;
                }
                if (get != null) {
                    try {
                        cpu = libc.getFunction("sched_getcpu");
                    } catch (Throwable t) {
                        cpu = null;
                    }
                }
            }
            GET_AFFINITY = get;
            SET_AFFINITY = set;
            GET_CPU = cpu;
            MASK_BYTES = bytes;
        }
    }

    private CpuAffinity() {
    }

    /** Whether the current-thread affinity calls can run on this host. */
    static boolean available() {
        return Libc.GET_AFFINITY != null && Libc.SET_AFFINITY != null;
    }

    /**
     * The CPUs the calling thread may run on, in increasing order; empty when the platform has no affinity calls or
     * the call fails.
     */
    static int[] currentThreadCpus() {
        if (!available()) {
            return new int[0];
        }
        Memory mask = new Memory(Libc.MASK_BYTES);
        mask.clear();
        int rc = Libc.GET_AFFINITY.invokeInt(new Object[] {Integer.valueOf(0), sizeT(Libc.MASK_BYTES), mask});
        if (rc != 0) {
            if (GET_FAILED.compareAndSet(false, true)) {
                LoggingUtil.getLogger(CpuAffinity.class).warn(
                        "sched_getaffinity failed (errno {}); CPU affinity is left alone", Native.getLastError());
            }
            return new int[0];
        }
        return cpusOf(mask);
    }

    /**
     * Restricts the calling thread to {@code cpus}.
     *
     * @return true when the kernel accepted the mask
     * @throws IllegalArgumentException when {@code cpus} is empty or holds a negative or oversized CPU number
     */
    static boolean setCurrentThreadCpus(int[] cpus) {
        if (cpus.length == 0) {
            throw new IllegalArgumentException("an affinity mask needs at least one CPU");
        }
        int maxCpu = Libc.MASK_BYTES * 8;
        for (int cpu : cpus) {
            if (cpu < 0 || cpu >= maxCpu) {
                throw new IllegalArgumentException("CPU " + cpu + " is outside the affinity mask of "
                        + maxCpu + " CPUs");
            }
        }
        if (!available()) {
            return false;
        }
        Memory mask = new Memory(Libc.MASK_BYTES);
        mask.clear();
        for (int cpu : cpus) {
            int off = cpu / 8;
            mask.setByte(off, (byte) (mask.getByte(off) | (1 << (cpu % 8))));
        }
        int rc = Libc.SET_AFFINITY.invokeInt(new Object[] {Integer.valueOf(0), sizeT(Libc.MASK_BYTES), mask});
        if (rc != 0) {
            if (SET_FAILED.compareAndSet(false, true)) {
                LoggingUtil.getLogger(CpuAffinity.class).warn(
                        "sched_setaffinity to {} CPU(s) failed (errno {}); threads stay where the OS schedules them",
                        cpus.length, Native.getLastError());
            }
            return false;
        }
        return true;
    }

    /** The CPU the calling thread last ran on, or -1 when unknown. */
    static int currentCpu() {
        Function getCpu = Libc.GET_CPU;
        if (getCpu != null) {
            try {
                int cpu = getCpu.invokeInt(new Object[0]);
                if (cpu >= 0) {
                    return cpu;
                }
            } catch (RuntimeException e) {
                // fall through to the procfs reader
            }
        }
        return currentCpuFromProc();
    }

    /** The mask bytes as CPU numbers. */
    private static int[] cpusOf(Memory mask) {
        List<Integer> cpus = new ArrayList<>();
        for (int cpu = 0; cpu < Libc.MASK_BYTES * 8; cpu++) {
            if ((mask.getByte(cpu / 8) & (1 << (cpu % 8))) != 0) {
                cpus.add(Integer.valueOf(cpu));
            }
        }
        int[] out = new int[cpus.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = cpus.get(i).intValue();
        }
        return out;
    }

    /** {@code processor} (field 39) of {@code /proc/self/stat}, the fallback when libc lacks {@code sched_getcpu}. */
    private static int currentCpuFromProc() {
        try {
            String stat = new String(Files.readAllBytes(Paths.get("/proc/self/stat")), StandardCharsets.US_ASCII);
            int close = stat.lastIndexOf(')');
            if (close < 0) {
                return -1;
            }
            String[] fields = stat.substring(close + 1).trim().split("\\s+");
            int index = 39 - 3;
            return index < fields.length ? Integer.parseInt(fields[index]) : -1;
        } catch (IOException | RuntimeException e) {
            return -1;
        }
    }

    /** The kernel's {@code cpu_set_t} size: the possible CPUs' bitset, at least {@value #MIN_MASK_BYTES} bytes. */
    private static int maskBytes() {
        int maxCpu = highestCpu("/sys/devices/system/cpu/possible");
        if (maxCpu < 0) {
            maxCpu = highestCpu("/sys/devices/system/cpu/online");
        }
        if (maxCpu < 0) {
            return MIN_MASK_BYTES;
        }
        int bytes = ((maxCpu + 1) + 63) / 64 * 8;
        return Math.max(MIN_MASK_BYTES, bytes);
    }

    /** The highest CPU number of a cpulist file such as {@code /sys/devices/system/cpu/possible}, or -1. */
    private static int highestCpu(String path) {
        int[] cpus = CpuLists.parse(read(path));
        int max = -1;
        for (int cpu : cpus) {
            max = Math.max(max, cpu);
        }
        return max;
    }

    /** The contents of a sysfs/procfs file, or null when it does not read. */
    static String read(String path) {
        try {
            return new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.US_ASCII).trim();
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    /** Whether a path exists, treating an unreadable parent as absent. */
    static boolean exists(Path path) {
        try {
            return Files.exists(path);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** {@code size_t} as a JNA argument: a 64-bit long on LP64, a 32-bit int elsewhere. */
    private static Object sizeT(long value) {
        return Native.SIZE_T_SIZE == Long.BYTES
                ? (Object) Long.valueOf(value)
                : (Object) Integer.valueOf((int) value);
    }
}
