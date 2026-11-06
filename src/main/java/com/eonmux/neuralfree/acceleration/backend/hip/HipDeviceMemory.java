package com.eonmux.neuralfree.acceleration.backend.hip;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * {@link HipMemoryDriver} on the HIP runtime for one {@link HipComputeBackend}. A fence is one
 * pooled event recorded on the stream of every open queue of the backend (its compute and
 * transfer queues and the caller-owned queues of {@code createComputeQueue()}); a queue that has
 * been released drained its stream and needs none.
 */
final class HipDeviceMemory implements HipMemoryDriver {

    private final HipComputeBackend backend;

    /**
     * @throws IllegalArgumentException when {@code backend} is null
     */
    HipDeviceMemory(HipComputeBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("HIP backend must not be null");
        }
        this.backend = backend;
    }

    @Override
    public long allocate(long bytes) {
        return backend.driver().memAlloc(bytes);
    }

    @Override
    public void free(long pointer) {
        backend.driver().memFree(pointer);
    }

    @Override
    public Fence recordFence() {
        List<HipGpuEvent> events = new ArrayList<>();
        try {
            for (HipComputeQueue queue : backend.openQueues()) {
                HipGpuEvent event = queue.recordFence();
                if (event != null) {
                    events.add(event);
                }
            }
        } catch (RuntimeException e) {
            for (HipGpuEvent event : events) {
                event.close();
            }
            throw e;
        }
        return new EventFence(events);
    }

    @Override
    public void synchronize() {
        backend.driver().synchronize();
    }

    /** The events of one fence; complete when every event is. */
    private static final class EventFence implements Fence {
        private final List<HipGpuEvent> events;

        EventFence(List<HipGpuEvent> events) {
            this.events = Collections.unmodifiableList(events);
        }

        @Override
        public boolean isComplete() {
            for (HipGpuEvent event : events) {
                if (!event.isComplete()) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public void release() {
            for (HipGpuEvent event : events) {
                event.close();
            }
        }
    }
}
