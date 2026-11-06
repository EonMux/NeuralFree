package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class GpuGraphCacheTest {

    /** A graph that counts its close calls and records nothing else. */
    private static final class FakeGraph implements GpuGraph {

        private int closes;

        @Override
        public int dispatchCount() {
            return 0;
        }

        @Override
        public GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents) {
            return null;
        }

        @Override
        public void replay(Runnable body) {
        }

        @Override
        public boolean isClosed() {
            return closes > 0;
        }

        @Override
        public void close() {
            closes++;
        }
    }

    private static GpuGraphCache.CaptureStep capturing(FakeGraph graph) {
        return () -> graph;
    }

    private static GpuGraphCache.CaptureStep failing(RuntimeException failure) {
        return () -> {
            throw failure;
        };
    }

    @Test
    public void storesOneGraphPerShapeKey() {
        GpuGraphCache cache = new GpuGraphCache(4);
        FakeGraph first = new FakeGraph();
        FakeGraph second = new FakeGraph();
        assertSame(first, cache.getOrCapture("t1", capturing(first)));
        assertSame(second, cache.getOrCapture("t2", capturing(second)));
        assertEquals(2, cache.size());
        assertTrue(cache.contains("t1"));
        assertTrue(cache.contains("t2"));
        assertFalse(cache.contains("t3"));
        cache.close();
    }

    @Test
    public void returnsTheStoredGraphWithoutCapturingAgain() {
        GpuGraphCache cache = new GpuGraphCache(4);
        FakeGraph graph = new FakeGraph();
        GpuGraph captured = cache.getOrCapture("t1", capturing(graph));
        AtomicInteger bodies = new AtomicInteger();
        GpuGraph replayed = cache.getOrCapture("t1", () -> {
            bodies.incrementAndGet();
            throw new AssertionError("a cached shape must not capture again");
        });
        assertSame(captured, replayed);
        assertEquals(0, bodies.get());
        cache.close();
    }

    @Test
    public void evictsTheLeastRecentlyUsedGraphBeyondTheCapacity() {
        GpuGraphCache cache = new GpuGraphCache(2);
        FakeGraph oldest = new FakeGraph();
        FakeGraph middle = new FakeGraph();
        FakeGraph newest = new FakeGraph();
        cache.getOrCapture("a", capturing(oldest));
        cache.getOrCapture("b", capturing(middle));
        cache.getOrCapture("a", capturing(oldest));
        cache.getOrCapture("c", capturing(newest));
        assertEquals(2, cache.size());
        assertFalse(cache.contains("b"));
        assertTrue(cache.contains("a"));
        assertTrue(cache.contains("c"));
        assertEquals(1, middle.closes);
        assertEquals(0, oldest.closes);
        assertEquals(0, newest.closes);
        cache.close();
    }

    @Test
    public void closeClosesEveryStoredGraphOnce() {
        GpuGraphCache cache = new GpuGraphCache(4);
        FakeGraph first = new FakeGraph();
        FakeGraph second = new FakeGraph();
        cache.getOrCapture("a", capturing(first));
        cache.getOrCapture("b", capturing(second));
        cache.close();
        cache.close();
        assertEquals(1, first.closes);
        assertEquals(1, second.closes);
        assertEquals(0, cache.size());
    }

    @Test
    public void refusesUseAfterClose() {
        GpuGraphCache cache = new GpuGraphCache(2);
        cache.close();
        try {
            cache.getOrCapture("a", capturing(new FakeGraph()));
            fail("getOrCapture after close must fail");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    @Test
    public void aFailedCaptureStoresNoGraph() {
        GpuGraphCache cache = new GpuGraphCache(2);
        RuntimeException failure = new RuntimeException("capture failed");
        try {
            cache.getOrCapture("a", failing(failure));
            fail("a failed capture must propagate");
        } catch (RuntimeException e) {
            assertSame(failure, e);
        }
        assertEquals(0, cache.size());
        FakeGraph later = new FakeGraph();
        assertSame(later, cache.getOrCapture("a", capturing(later)));
        cache.close();
    }

    @Test
    public void rejectsNullKeysAndSteps() {
        GpuGraphCache cache = new GpuGraphCache(2);
        try {
            cache.getOrCapture(null, capturing(new FakeGraph()));
            fail("null key must fail");
        } catch (NullPointerException | IllegalArgumentException e) {
            // either message is fine: the argument is named
        }
        try {
            cache.getOrCapture("a", null);
            fail("null step must fail");
        } catch (NullPointerException | IllegalArgumentException e) {
            // either message is fine: the argument is named
        }
        cache.close();
    }

    @Test
    public void rejectsACaptureStepWithoutResult() {
        GpuGraphCache cache = new GpuGraphCache(2);
        try {
            cache.getOrCapture("a", () -> null);
            fail("a step without a graph must fail");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("a"));
        }
        assertEquals(0, cache.size());
        cache.close();
    }

    @Test
    public void capturedStepAbortsTheQueueCaptureWhenTheBodyFails() {
        AtomicReference<ComputeQueue> queue = new AtomicReference<>();
        AtomicReference<Integer> state = new AtomicReference<>(0);
        ComputeQueue fake = new ComputeQueue() {
            @Override
            public QueueType getQueueType() {
                return QueueType.COMPUTE;
            }

            @Override
            public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                               byte[] srcData, GpuEvent... waitOnEvents) {
                throw new UnsupportedOperationException();
            }

            @Override
            public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                               float[] srcData, GpuEvent... waitOnEvents) {
                throw new UnsupportedOperationException();
            }

            @Override
            public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                              byte[] dstData, GpuEvent... waitOnEvents) {
                throw new UnsupportedOperationException();
            }

            @Override
            public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                              float[] dstData, GpuEvent... waitOnEvents) {
                throw new UnsupportedOperationException();
            }

            @Override
            public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                          Object[] args, GpuEvent... waitOnEvents) {
                throw new UnsupportedOperationException();
            }

            @Override
            public void flush() {
            }

            @Override
            public void finish() {
            }

            @Override
            public void close() {
            }

            @Override
            public void beginCapture() {
                state.set(1);
            }

            @Override
            public GpuGraph endCapture() {
                state.set(2);
                return new FakeGraph();
            }

            @Override
            public void abortCapture() {
                state.set(0);
            }
        };
        queue.set(fake);
        RuntimeException failure = new RuntimeException("body failed");
        try {
            GpuGraphCache.captured(fake, () -> {
                throw failure;
            }).capture();
            fail("the body failure must propagate");
        } catch (RuntimeException e) {
            assertSame(failure, e);
        }
        assertEquals(Integer.valueOf(0), state.get());
        FakeGraph captured = (FakeGraph) GpuGraphCache.captured(fake, () -> {
        }).capture();
        assertEquals(Integer.valueOf(2), state.get());
        assertEquals(0, captured.closes);
    }
}
