package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import java.lang.reflect.Proxy;

import static org.junit.Assert.*;

/** Routed-id readbacks release their event even when waiting fails or is interrupted. */
public class GpuRouterHeatTest {

    @Test
    public void completedReadsReleaseTheirEvent() {
        try (FakeComputeBackend backend = FakeComputeBackend.builder().build();
             GpuBuffer buffer = backend.allocateBuffer(Integer.BYTES)) {
            MoeExpertHeat heat = new MoeExpertHeat();
            GpuRouterHeat reader = new GpuRouterHeat(backend.computeQueue(), heat, null, 0, 0);
            reader.note(ScratchSlot.of("ids", buffer, 0, 1), 1);
            assertEquals(1, backend.openEventCount());
            assertNull(reader.drain());
            assertEquals(0, backend.openEventCount());
            assertFalse(heat.isEmpty());
        }
    }

    @Test
    public void failedWaitReleasesItsEvent() {
        assertFailedWaitCloses(false);
    }

    @Test
    public void interruptedWaitReleasesItsEventAndPreservesInterrupt() {
        try {
            assertFailedWaitCloses(true);
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
    }

    private static void assertFailedWaitCloses(boolean interrupted) {
        FailingEvent event = new FailingEvent(interrupted);
        ComputeQueue queue = (ComputeQueue) Proxy.newProxyInstance(ComputeQueue.class.getClassLoader(),
                new Class<?>[]{ComputeQueue.class}, (proxy, method, args) -> {
                    if (method.getName().equals("enqueueReadBuffer")) {
                        return event;
                    }
                    throw new AssertionError("unexpected queue call: " + method.getName());
                });
        try (FakeComputeBackend backend = FakeComputeBackend.builder().build();
             GpuBuffer buffer = backend.allocateBuffer(Integer.BYTES)) {
            MoeExpertHeat heat = new MoeExpertHeat();
            GpuRouterHeat reader = new GpuRouterHeat(queue, heat, null, 0, 7);
            reader.note(ScratchSlot.of("ids", buffer, 0, 1), 1);
            assertNotNull(reader.drain());
            assertTrue("the failed read's event must be closed", event.closed);
            assertTrue("a failed read must not contribute heat", heat.isEmpty());
            assertNull(reader.drain());
        }
    }

    private static final class FailingEvent implements GpuEvent {
        private final boolean interrupted;
        private boolean closed;

        private FailingEvent(boolean interrupted) {
            this.interrupted = interrupted;
        }

        @Override
        public void awaitCompletion() throws InterruptedException {
            if (interrupted) {
                throw new InterruptedException("read interrupted");
            }
            throw new IllegalStateException("read failed");
        }

        @Override
        public boolean isComplete() {
            return false;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
