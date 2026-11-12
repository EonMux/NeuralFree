package com.eonmux.neuralfree.gguf.gpu.pipeline;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The split stages' concurrent work: every stage at once, every failure after all ended, interrupts deferred. */
public class ConcurrentStageWorkTest {

    @Test
    public void everyStageRunsAtTheSameTime() {
        int n = 3;
        CyclicBarrier together = new CyclicBarrier(n);
        List<String> threads = Collections.synchronizedList(new ArrayList<>());
        List<Runnable> tasks = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            tasks.add(() -> {
                try {
                    together.await(10, TimeUnit.SECONDS);
                } catch (Exception e) {
                    throw new IllegalStateException("the stages did not run together", e);
                }
                threads.add(Thread.currentThread().getName());
            });
        }
        ConcurrentStageWork.runAll(Arrays.asList("cuda0", "vulkan1", "opencl2"), tasks);
        assertEquals(new HashSet<>(Arrays.asList(Thread.currentThread().getName(), "nf-stage-vulkan1",
                "nf-stage-opencl2")), new HashSet<>(threads));
    }

    @Test
    public void failuresAreThrownInStageOrderAfterEveryStageEnded() {
        IllegalStateException first = new IllegalStateException("stage 1");
        IllegalArgumentException second = new IllegalArgumentException("stage 2");
        AtomicBoolean slowEnded = new AtomicBoolean();
        CountDownLatch failed = new CountDownLatch(2);
        List<Runnable> tasks = Arrays.asList(
                () -> {
                    awaitQuietly(failed);
                    sleepQuietly(100);
                    slowEnded.set(true);
                },
                () -> {
                    failed.countDown();
                    throw first;
                },
                () -> {
                    failed.countDown();
                    throw second;
                });
        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> ConcurrentStageWork.runAll(Arrays.asList("a", "b", "c"), tasks));
        assertSame(first, thrown);
        assertEquals(1, thrown.getSuppressed().length);
        assertSame(second, thrown.getSuppressed()[0]);
        assertTrue("the failure waits for the stage still running", slowEnded.get());
    }

    @Test
    public void anInterruptDoesNotCutTheWaitShortAndIsKept() throws InterruptedException {
        CountDownLatch started = new CountDownLatch(1);
        AtomicBoolean otherEnded = new AtomicBoolean();
        Thread caller = Thread.currentThread();
        List<Runnable> tasks = Arrays.asList(
                () -> {
                    awaitQuietly(started);
                    caller.interrupt();
                },
                () -> {
                    started.countDown();
                    sleepQuietly(200);
                    otherEnded.set(true);
                });
        try {
            ConcurrentStageWork.runAll(Arrays.asList("a", "b"), tasks);
            assertTrue(otherEnded.get());
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    public void namesAndTasksMustPair() {
        assertThrows(IllegalArgumentException.class, () -> ConcurrentStageWork.runAll(
                Collections.singletonList("a"), Collections.emptyList()));
    }

    private static void awaitQuietly(CountDownLatch latch) {
        try {
            assertTrue(latch.await(10, TimeUnit.SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    private static void sleepQuietly(long ms) {
        long end = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(ms);
        while (System.nanoTime() < end) {
            try {
                Thread.sleep(Math.max(1, TimeUnit.NANOSECONDS.toMillis(end - System.nanoTime())));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
