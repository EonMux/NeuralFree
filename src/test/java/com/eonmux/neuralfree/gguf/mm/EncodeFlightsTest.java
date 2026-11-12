package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import org.junit.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Single-flight encodes: concurrent callers with one key share one encode and its result or failure, other keys
 * encode beside it, and a finished key encodes afresh.
 */
public class EncodeFlightsTest {

    private static final long WAIT_SECONDS = 30;

    @Test
    public void concurrentCallersOfOneKeyShareOneEncode() throws Exception {
        EncodeFlights<String, Object> flights = new EncodeFlights<>();
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger encodes = new AtomicInteger();
        Object result = new Object();
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            Future<Object> leader = pool.submit(() -> flights.run("img", () -> {
                encodes.incrementAndGet();
                started.countDown();
                awaitQuietly(release);
                return result;
            }));
            assertTrue(started.await(WAIT_SECONDS, TimeUnit.SECONDS));
            Future<Object> waiter1 = pool.submit(() -> flights.run("img", () -> {
                encodes.incrementAndGet();
                return new Object();
            }));
            Future<Object> waiter2 = pool.submit(() -> flights.run("img", () -> {
                encodes.incrementAndGet();
                return new Object();
            }));
            // the waiters are parked on the running encode: the key is still in flight
            Thread.sleep(50);
            assertEquals(1, flights.inFlight());
            release.countDown();
            assertSame(result, leader.get(WAIT_SECONDS, TimeUnit.SECONDS));
            assertSame(result, waiter1.get(WAIT_SECONDS, TimeUnit.SECONDS));
            assertSame(result, waiter2.get(WAIT_SECONDS, TimeUnit.SECONDS));
            assertEquals("one encode served the three callers", 1, encodes.get());
            assertEquals(0, flights.inFlight());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void aWaiterTakesTheEncodesFailure() throws Exception {
        EncodeFlights<String, Object> flights = new EncodeFlights<>();
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        MediaInputException broken = new MediaInputException("the image cannot be decoded");
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<Object> leader = pool.submit(() -> flights.run("img", () -> {
                started.countDown();
                awaitQuietly(release);
                throw broken;
            }));
            assertTrue(started.await(WAIT_SECONDS, TimeUnit.SECONDS));
            Future<Object> waiter = pool.submit(() -> flights.run("img", () -> {
                throw new AssertionError("the waiter encoded the item again");
            }));
            Thread.sleep(50);
            release.countDown();
            for (Future<Object> f : java.util.List.of(leader, waiter)) {
                try {
                    f.get(WAIT_SECONDS, TimeUnit.SECONDS);
                    fail("the failed encode answered");
                } catch (java.util.concurrent.ExecutionException e) {
                    assertSame(broken, e.getCause());
                }
            }
            assertEquals(0, flights.inFlight());
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * A leader whose decode is interrupted fails alone: the waiter on its flight encodes the item itself instead of
     * taking the leader's interruption.
     */
    @Test
    public void aWaiterEncodesAgainWhenTheLeaderIsInterrupted() throws Exception {
        EncodeFlights<String, String> flights = new EncodeFlights<>();
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger waiterEncodes = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<String> leader = pool.submit(() -> flights.run("video", () -> {
                started.countDown();
                awaitQuietly(release);
                Thread.currentThread().interrupt();
                throw new MediaInputException("video: the ffmpeg decode was interrupted",
                        new InterruptedException());
            }));
            assertTrue(started.await(WAIT_SECONDS, TimeUnit.SECONDS));
            Future<String> waiter = pool.submit(() -> flights.run("video", () -> {
                waiterEncodes.incrementAndGet();
                return "frames";
            }));
            Thread.sleep(50);
            release.countDown();
            try {
                leader.get(WAIT_SECONDS, TimeUnit.SECONDS);
                fail("the interrupted encode answered");
            } catch (java.util.concurrent.ExecutionException e) {
                assertTrue(e.getCause() instanceof MediaInputException);
            }
            assertEquals("frames", waiter.get(WAIT_SECONDS, TimeUnit.SECONDS));
            assertEquals("the waiter encoded once itself", 1, waiterEncodes.get());
            assertEquals(0, flights.inFlight());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void interruptionsAndCancellationsAreTheCallersOwnFailures() {
        assertTrue(EncodeFlights.callersOwn(new MediaInputException("x: the ffprobe run was interrupted",
                new InterruptedException())));
        assertTrue(EncodeFlights.callersOwn(new IllegalStateException(new java.io.InterruptedIOException())));
        assertTrue(EncodeFlights.callersOwn(new java.util.concurrent.CancellationException("request cancelled")));
        assertFalse("a broken item fails every caller",
                EncodeFlights.callersOwn(new MediaInputException("the image cannot be decoded")));
    }

    @Test
    public void otherKeysEncodeBesideARunningOneAndAFinishedKeyEncodesAgain() throws Exception {
        EncodeFlights<String, String> flights = new EncodeFlights<>();
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<String> slow = pool.submit(() -> flights.run("a", () -> {
                started.countDown();
                awaitQuietly(release);
                return "a1";
            }));
            assertTrue(started.await(WAIT_SECONDS, TimeUnit.SECONDS));
            assertEquals("another key does not wait", "b1", flights.run("b", () -> "b1"));
            release.countDown();
            assertEquals("a1", slow.get(WAIT_SECONDS, TimeUnit.SECONDS));
            assertEquals("a finished key encodes afresh", "a2", flights.run("a", () -> "a2"));
        } finally {
            pool.shutdownNow();
        }
    }

    private static void awaitQuietly(CountDownLatch latch) {
        try {
            if (!latch.await(WAIT_SECONDS, TimeUnit.SECONDS)) {
                throw new AssertionError("the test never released the encode");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }
}
