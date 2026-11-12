package com.eonmux.neuralfree.acceleration.backend.ocl;

import org.jocl.cl_program;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The OpenCL backend's program cache: one build per key shared by concurrent callers, builds of different keys running
 * at the same time, failed builds not kept, and the release at close waiting for running builds and refusing new ones.
 */
public class OclProgramCacheTest {

    private final ExecutorService pool = Executors.newFixedThreadPool(4);

    @After
    public void stop() throws InterruptedException {
        pool.shutdownNow();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));
    }

    @Test
    public void concurrentCallersOfOneKeyShareOneBuild() throws Exception {
        OclProgramCache cache = new OclProgramCache();
        CountDownLatch building = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        AtomicInteger builds = new AtomicInteger();
        cl_program built = new cl_program();
        Future<cl_program> first = pool.submit(() -> cache.getOrBuild("k", () -> {
            builds.incrementAndGet();
            building.countDown();
            await(finish);
            return built;
        }));
        assertTrue(building.await(10, TimeUnit.SECONDS));
        Future<cl_program> second = pool.submit(() -> cache.getOrBuild("k", () -> {
            builds.incrementAndGet();
            return new cl_program();
        }));
        finish.countDown();
        assertSame(built, first.get(10, TimeUnit.SECONDS));
        assertSame(built, second.get(10, TimeUnit.SECONDS));
        assertEquals(1, builds.get());
        assertSame(built, cache.getOrBuild("k", () -> {
            throw new AssertionError("a cached program is not built again");
        }));
    }

    @Test
    public void aRunningBuildDoesNotHoldBackBuildsOfOtherKeys() throws Exception {
        OclProgramCache cache = new OclProgramCache();
        CountDownLatch otherDone = new CountDownLatch(1);
        Future<cl_program> slow = pool.submit(() -> cache.getOrBuild("slow", () -> {
            await(otherDone);
            return new cl_program();
        }));
        Future<cl_program> fast = pool.submit(() -> cache.getOrBuild("fast", cl_program::new));
        fast.get(10, TimeUnit.SECONDS);
        otherDone.countDown();
        slow.get(10, TimeUnit.SECONDS);
        assertEquals(2, cache.size());
    }

    @Test
    public void aFailedBuildIsNotKept() {
        OclProgramCache cache = new OclProgramCache();
        IllegalArgumentException failure = new IllegalArgumentException("build log");
        assertSame(failure, assertThrows(IllegalArgumentException.class, () -> cache.getOrBuild("k", () -> {
            throw failure;
        })));
        assertEquals(0, cache.size());
        cl_program program = new cl_program();
        assertSame(program, cache.getOrBuild("k", () -> program));
    }

    @Test
    public void releaseWaitsForRunningBuildsReleasesEachProgramOnceAndRefusesLaterBuilds() throws Exception {
        OclProgramCache cache = new OclProgramCache();
        cl_program ready = cache.getOrBuild("ready", cl_program::new);
        CountDownLatch building = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        cl_program late = new cl_program();
        Future<cl_program> running = pool.submit(() -> cache.getOrBuild("running", () -> {
            building.countDown();
            await(finish);
            return late;
        }));
        assertTrue(building.await(10, TimeUnit.SECONDS));
        List<cl_program> released = new ArrayList<>();
        Future<?> close = pool.submit(() -> cache.releaseAll(p -> {
            synchronized (released) {
                released.add(p);
            }
        }));
        Thread.sleep(50);
        assertTrue("the release waits for the running build", !close.isDone());
        finish.countDown();
        close.get(10, TimeUnit.SECONDS);
        assertSame(late, running.get(10, TimeUnit.SECONDS));
        assertEquals(2, released.size());
        assertTrue(released.contains(ready) && released.contains(late));
        assertEquals(0, cache.size());
        assertThrows(IllegalStateException.class, () -> cache.getOrBuild("after", cl_program::new));
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("latch timed out");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }
}
