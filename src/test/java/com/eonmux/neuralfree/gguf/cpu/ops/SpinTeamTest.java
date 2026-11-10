package com.eonmux.neuralfree.gguf.cpu.ops;

import com.eonmux.neuralfree.gguf.cpu.numa.NumaPlacement;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaTopology;
import com.sun.jna.Platform;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.lang.ref.WeakReference;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SpinTeam}: every index runs once in the fork/join form's chunk ranges with the caller taking part, a chunk's
 * exception reaches the caller and leaves the team usable, a loop started from inside a loop is refused, the slots'
 * callers run at once while a further caller waits, concurrent callers share the team's threads, a finished loop is
 * not kept reachable, a caller interrupted before or while it waits parks and keeps its interrupt status, under NUMA
 * placement every chunk runs on a pinned worker and the waiting caller parks without spinning, parked workers wake
 * for a later loop, and the team and the {@value CpuParallel#SPIN_PROPERTY} property reject invalid values by name.
 * Every test has a timeout, and a test that holds a team's slots does so on a team of its own, so one failure cannot
 * stall the others.
 */
public class SpinTeamTest {

    /** How long a chunk keeps a waiting caller waiting in the tests that measure the caller's CPU time. */
    private static final long HOLD_MS = 400;

    /** The CPU time a caller may use while it waits {@link #HOLD_MS}: a parked caller uses almost none. */
    private static final long PARKED_CPU_NANOS = TimeUnit.MILLISECONDS.toNanos(HOLD_MS / 4);

    @Rule
    public final Timeout timeout = Timeout.seconds(60);

    private static SpinTeam team;
    private String savedSpin;

    @BeforeClass
    public static void startTeam() {
        team = new SpinTeam(4, TimeUnit.MICROSECONDS.toNanos(200), NumaPlacement.disabled());
    }

    @AfterClass
    public static void dropTeam() {
        team = null;
    }

    @After
    public void restoreProperty() {
        if (savedSpin == null) {
            System.clearProperty(CpuParallel.SPIN_PROPERTY);
        } else {
            System.setProperty(CpuParallel.SPIN_PROPERTY, savedSpin);
        }
    }

    @Test
    public void everyIndexRunsOnceInTheForkJoinChunkRangesWithTheCallerTakingPart() {
        for (int[] shape : new int[][]{{1, 1}, {7, 3}, {1000, 16}, {4096, 1}, {33, 32}}) {
            int n = shape[0];
            int chunk = shape[1];
            AtomicIntegerArray hits = new AtomicIntegerArray(n);
            Set<String> ranges = ConcurrentHashMap.newKeySet();
            Set<Thread> threads = ConcurrentHashMap.newKeySet();
            assertTrue(team.run(n, chunk, (from, to) -> {
                ranges.add(from + ":" + to);
                threads.add(Thread.currentThread());
                for (int i = from; i < to; i++) {
                    hits.incrementAndGet(i);
                }
            }));
            for (int i = 0; i < n; i++) {
                assertEquals(n + "/" + chunk + " index " + i, 1, hits.get(i));
            }
            int chunks = (n + chunk - 1) / chunk;
            assertEquals(chunks, ranges.size());
            for (int c = 0; c < chunks; c++) {
                assertTrue(ranges.toString(), ranges.contains(c * chunk + ":" + Math.min(n, (c + 1) * chunk)));
            }
            assertTrue(threads.size() <= team.members());
        }
    }

    @Test
    public void aChunkExceptionReachesTheCallerAndLeavesTheTeamUsable() {
        IllegalStateException boom = new IllegalStateException("chunk 5");
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> team.run(64, 1, (from, to) -> {
            if (from == 5) {
                throw boom;
            }
        }));
        assertSame(boom, e);
        AssertionError err = assertThrows(AssertionError.class, () -> team.run(8, 1, (from, to) -> {
            throw new AssertionError("error " + from);
        }));
        assertTrue(err.getMessage(), err.getMessage().startsWith("error "));
        AtomicIntegerArray hits = new AtomicIntegerArray(100);
        assertTrue(team.run(100, 3, (from, to) -> {
            for (int i = from; i < to; i++) {
                hits.incrementAndGet(i);
            }
        }));
        for (int i = 0; i < 100; i++) {
            assertEquals(1, hits.get(i));
        }
    }

    @Test
    public void aLoopFromInsideALoopIsRefusedAndCallersBeyondTheSlotsWait() throws Exception {
        SpinTeam own = new SpinTeam(4, TimeUnit.MICROSECONDS.toNanos(200), NumaPlacement.disabled());
        assertFalse("outside a loop", own.isMember());
        AtomicBoolean nestedTaken = new AtomicBoolean(true);
        AtomicBoolean allMembers = new AtomicBoolean(true);
        own.run(16, 1, (from, to) -> {
            allMembers.compareAndSet(true, own.isMember());
            if (from == 0) {
                nestedTaken.set(own.run(4, 1, (a, b) -> { }));
            }
        });
        assertTrue("every chunk runs on a member", allMembers.get());
        assertFalse("a nested loop is refused", nestedTaken.get());

        CountDownLatch inside = new CountDownLatch(2);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean released = new AtomicBoolean();
        AtomicReference<Throwable> ownerFailure = new AtomicReference<>();
        Thread[] owners = new Thread[2];
        Thread extra = null;
        try {
            for (int o = 0; o < owners.length; o++) {
                owners[o] = new Thread(() -> {
                    try {
                        own.run(1, 1, (from, to) -> {
                            inside.countDown();
                            try {
                                release.await();
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                            released.set(true);
                        });
                    } catch (Throwable t) {
                        ownerFailure.set(t);
                    }
                });
                owners[o].start();
            }
            assertTrue("two callers' loops run at once", inside.await(10, TimeUnit.SECONDS));
            AtomicBoolean ranAfterRelease = new AtomicBoolean(true);
            extra = new Thread(() -> own.run(4, 1, (a, b) -> ranAfterRelease.compareAndSet(true, released.get())));
            extra.start();
            Thread.sleep(20);
            assertTrue("a third caller waits for a slot", extra.isAlive());
            release.countDown();
            for (Thread o : owners) {
                o.join(10_000);
            }
            extra.join(10_000);
            assertFalse(extra.isAlive());
            assertTrue("the waiting caller's loop ran once a slot freed", ranAfterRelease.get());
            assertEquals(null, ownerFailure.get());
        } finally {
            release.countDown();
            for (Thread o : owners) {
                if (o != null) {
                    o.join(10_000);
                }
            }
            if (extra != null) {
                extra.join(10_000);
            }
        }
    }

    @Test
    public void anInterruptedCallerParksAndKeepsItsInterruptStatus() throws Exception {
        ThreadMXBean mx = cpuClock();
        SpinTeam own = new SpinTeam(4, TimeUnit.MICROSECONDS.toNanos(200), NumaPlacement.disabled());
        Thread caller = Thread.currentThread();
        CountDownLatch workersHolding = new CountDownLatch(own.members() - 1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicLong cpuAtWait = new AtomicLong();
        AtomicBoolean callerSawWorkers = new AtomicBoolean();
        Thread releaser = new Thread(() -> {
            try {
                if (workersHolding.await(10, TimeUnit.SECONDS)) {
                    Thread.sleep(HOLD_MS / 2);
                    caller.interrupt();
                    Thread.sleep(HOLD_MS / 2);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                release.countDown();
            }
        });
        releaser.start();
        try {
            own.run(own.members(), 1, (from, to) -> {
                try {
                    if (Thread.currentThread() != caller) {
                        workersHolding.countDown();
                        release.await();
                        return;
                    }
                    callerSawWorkers.set(workersHolding.await(10, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                }
                caller.interrupt();
                cpuAtWait.set(mx.getCurrentThreadCpuTime());
            });
            long waitCpu = mx.getCurrentThreadCpuTime() - cpuAtWait.get();
            assertTrue("every worker held a chunk while the caller waited", callerSawWorkers.get());
            assertTrue("the caller keeps its interrupt status", Thread.interrupted());
            assertTrue("the interrupted caller parked instead of spinning: " + waitCpu + " ns of CPU",
                    waitCpu < PARKED_CPU_NANOS);
        } finally {
            Thread.interrupted();
            release.countDown();
            releaser.join(10_000);
        }
    }

    @Test
    public void concurrentCallersShareTheTeamsThreads() throws Exception {
        int loops = 300;
        AtomicIntegerArray hits = new AtomicIntegerArray(2 * 16);
        Set<Thread> threads = ConcurrentHashMap.newKeySet();
        Thread[] callers = new Thread[2];
        AtomicReference<Throwable> failure = new AtomicReference<>();
        for (int c = 0; c < 2; c++) {
            final int me = c;
            callers[c] = new Thread(() -> {
                try {
                    for (int l = 0; l < loops; l++) {
                        team.run(16, 1, (from, to) -> {
                            threads.add(Thread.currentThread());
                            for (int i = from; i < to; i++) {
                                hits.incrementAndGet(me * 16 + i);
                            }
                        });
                    }
                } catch (Throwable t) {
                    failure.set(t);
                }
            });
        }
        for (Thread t : callers) {
            t.start();
        }
        for (Thread t : callers) {
            t.join(60_000);
        }
        assertEquals(null, failure.get());
        for (int i = 0; i < 2 * 16; i++) {
            assertEquals("caller " + i / 16 + " index " + i % 16, loops, hits.get(i));
        }
        assertTrue(threads + " ran on the team and its two callers only", threads.size() <= team.members() + 1);
    }

    @Test
    public void theTeamKeepsNoReferenceToAFinishedLoop() throws InterruptedException {
        WeakReference<Object> ref = runCapturing();
        for (int i = 0; i < 100 && ref.get() != null; i++) {
            System.gc();
            Thread.sleep(10);
        }
        assertEquals("the finished loop's body is unreachable", null, ref.get());
    }

    /** Runs a loop whose body captures a fresh object, and returns a weak reference to that object. */
    private static WeakReference<Object> runCapturing() {
        Object captured = new Object();
        AtomicInteger sink = new AtomicInteger();
        Set<Thread> ran = ConcurrentHashMap.newKeySet();
        team.run(8, 1, (from, to) -> {
            ran.add(Thread.currentThread());
            sink.addAndGet(System.identityHashCode(captured) & 1);
            LockSupport.parkNanos(2_000_000L);
        });
        assertTrue("workers ran chunks of the loop: " + ran, ran.size() > 1);
        return new WeakReference<>(captured);
    }

    @Test
    public void underNumaPlacementEveryChunkRunsOnAPinnedWorker() throws Exception {
        int[] before = allowedCpus();
        SpinTeam numa = new SpinTeam(2, TimeUnit.MICROSECONDS.toNanos(50), twoNodePlacement(before));
        int first = before[0];
        int second = before[before.length - 1];
        assertFalse(numa.callerWorks());
        assertEquals(2, numa.members());
        Thread caller = Thread.currentThread();
        Set<String> affinities = ConcurrentHashMap.newKeySet();
        AtomicBoolean callerRan = new AtomicBoolean();
        for (int l = 0; l < 50; l++) {
            numa.run(64, 1, (from, to) -> {
                if (Thread.currentThread() == caller) {
                    callerRan.set(true);
                }
                affinities.add(Arrays.toString(allowedCpus()));
            });
        }
        assertFalse("the caller only waits", callerRan.get());
        Set<String> nodes = Set.of(Arrays.toString(new int[] {first}), Arrays.toString(new int[] {second}));
        assertTrue(affinities + " are node CPU sets", nodes.containsAll(affinities));
        assertArrayEquals("the caller keeps its affinity", before, allowedCpus());
    }

    @Test
    public void underNumaPlacementTheWaitingCallerParksWithoutSpinning() {
        ThreadMXBean mx = cpuClock();
        SpinTeam numa = new SpinTeam(2, TimeUnit.MILLISECONDS.toNanos(2 * HOLD_MS), twoNodePlacement(allowedCpus()));
        AtomicInteger ran = new AtomicInteger();
        long cpuBefore = mx.getCurrentThreadCpuTime();
        numa.run(2, 1, (from, to) -> {
            long end = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(HOLD_MS);
            for (long left = end - System.nanoTime(); left > 0; left = end - System.nanoTime()) {
                LockSupport.parkNanos(left);
            }
            ran.incrementAndGet();
        });
        long waitCpu = mx.getCurrentThreadCpuTime() - cpuBefore;
        assertEquals(2, ran.get());
        assertTrue("the waiting caller parked instead of spinning: " + waitCpu + " ns of CPU",
                waitCpu < PARKED_CPU_NANOS);
    }

    /**
     * A placement of a fake two-node topology over the first and the last of {@code allowed}, the test thread's CPUs;
     * skips the test off Linux, with fewer than two CPUs, or without {@code sched_setaffinity}.
     */
    private static NumaPlacement twoNodePlacement(int[] allowed) {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue("the test thread may run on at least two CPUs", allowed.length >= 2);
        int first = allowed[0];
        int second = allowed[allowed.length - 1];
        NumaTopology topology = NumaTopology.of(new int[][] {{first}, {second}}, allowed, first);
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.DISTRIBUTE, topology);
        Assume.assumeTrue("sched_setaffinity is available", placement.active());
        return placement;
    }

    /** The thread CPU clock; skips the test when the JVM does not measure the current thread's CPU time. */
    private static ThreadMXBean cpuClock() {
        ThreadMXBean mx = ManagementFactory.getThreadMXBean();
        Assume.assumeTrue("the JVM measures thread CPU time", mx.isCurrentThreadCpuTimeSupported());
        if (!mx.isThreadCpuTimeEnabled()) {
            mx.setThreadCpuTimeEnabled(true);
        }
        return mx;
    }

    @Test
    public void concurrentParallelForCallersRunOnTheTeamOnly() throws Exception {
        Assume.assumeTrue("a multi-thread pool with the default spin",
                CpuParallel.parallelism() > 1 && System.getProperty(CpuParallel.SPIN_PROPERTY) == null);
        Set<String> names = ConcurrentHashMap.newKeySet();
        AtomicIntegerArray hits = new AtomicIntegerArray(2);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread[] callers = new Thread[2];
        for (int c = 0; c < 2; c++) {
            final int me = c;
            callers[c] = new Thread(() -> {
                try {
                    for (int l = 0; l < 200; l++) {
                        CpuParallel.parallelFor(64, 1, (from, to) -> {
                            names.add(Thread.currentThread().getName());
                            hits.addAndGet(me, to - from);
                        });
                    }
                } catch (Throwable t) {
                    failure.set(t);
                }
            }, "caller-" + c);
        }
        for (Thread t : callers) {
            t.start();
        }
        for (Thread t : callers) {
            t.join(60_000);
        }
        assertEquals(null, failure.get());
        assertEquals(200 * 64, hits.get(0));
        assertEquals(200 * 64, hits.get(1));
        for (String name : names) {
            assertFalse(names + ": no fork/join worker runs beside the team", name.matches("nf-cpu-\\d+"));
        }
    }

    @Test
    public void aLoopNestedInASaturatedTeamLoopRunsInline() {
        Assume.assumeTrue("a multi-thread pool with the default spin",
                CpuParallel.parallelism() > 1 && System.getProperty(CpuParallel.SPIN_PROPERTY) == null);
        AtomicBoolean elsewhere = new AtomicBoolean();
        AtomicInteger nested = new AtomicInteger();
        CpuParallel.parallelFor(CpuParallel.parallelism() * 4, 1, (from, to) -> {
            Thread outer = Thread.currentThread();
            CpuParallel.parallelFor(1000, 1, (a, b) -> {
                if (Thread.currentThread() != outer) {
                    elsewhere.set(true);
                }
                nested.addAndGet(b - a);
            });
        });
        assertFalse("every nested range runs on the member that started it", elsewhere.get());
        assertEquals(CpuParallel.parallelism() * 4 * 1000, nested.get());
    }

    /** The calling thread's {@code Cpus_allowed_list} from {@code /proc/thread-self/status}. */
    private static int[] allowedCpus() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/thread-self/status"))) {
                if (line.startsWith("Cpus_allowed_list:")) {
                    return parseCpuList(line.substring(line.indexOf(':') + 1).trim());
                }
            }
        } catch (IOException e) {
            return new int[0];
        }
        return new int[0];
    }

    private static int[] parseCpuList(String list) {
        List<Integer> cpus = new ArrayList<>();
        for (String part : list.split(",")) {
            int dash = part.indexOf('-');
            int lo = Integer.parseInt(dash < 0 ? part : part.substring(0, dash));
            int hi = dash < 0 ? lo : Integer.parseInt(part.substring(dash + 1));
            for (int c = lo; c <= hi; c++) {
                cpus.add(c);
            }
        }
        return cpus.stream().mapToInt(Integer::intValue).toArray();
    }

    @Test
    public void parkedWorkersWakeForALaterLoop() throws InterruptedException {
        Thread.sleep(50);
        Set<Thread> threads = ConcurrentHashMap.newKeySet();
        CountDownLatch all = new CountDownLatch(team.members());
        assertTrue(team.run(team.members(), 1, (from, to) -> {
            threads.add(Thread.currentThread());
            all.countDown();
            try {
                all.await(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }));
        assertEquals("every member, the parked workers included, took a chunk", team.members(), threads.size());
    }

    @Test
    public void invalidTeamsAndSpinTimesAreRejectedByName() {
        IllegalArgumentException one = assertThrows(IllegalArgumentException.class,
                () -> new SpinTeam(1, 1000, NumaPlacement.disabled()));
        assertTrue(one.getMessage(), one.getMessage().contains("two members"));
        assertThrows(IllegalArgumentException.class, () -> new SpinTeam(2, 0, NumaPlacement.disabled()));

        savedSpin = System.getProperty(CpuParallel.SPIN_PROPERTY);
        System.clearProperty(CpuParallel.SPIN_PROPERTY);
        assertEquals(CpuParallel.DEFAULT_SPIN_US, CpuParallel.spinMicros(8, 8));
        assertEquals("a pool larger than the processors does not spin", 0L, CpuParallel.spinMicros(16, 8));
        System.setProperty(CpuParallel.SPIN_PROPERTY, "250");
        assertEquals("an explicit spin time holds whatever the pool", 250L, CpuParallel.spinMicros(16, 8));
        System.setProperty(CpuParallel.SPIN_PROPERTY, " 0 ");
        assertEquals(0L, CpuParallel.spinMicros(8, 8));
        for (String bad : new String[]{"-1", "fast", "1000001"}) {
            System.setProperty(CpuParallel.SPIN_PROPERTY, bad);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> CpuParallel.spinMicros(8, 8));
            assertTrue(e.getMessage(), e.getMessage().contains(CpuParallel.SPIN_PROPERTY));
        }
    }
}
