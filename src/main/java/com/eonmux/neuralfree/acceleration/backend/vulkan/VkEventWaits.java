package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * Checks and ordering of the {@link GpuEvent} waits a queue is handed: every wait must be an open event of this
 * backend, and another queue's event is flushed on its own queue before this queue submits the wait.
 */
final class VkEventWaits {

    private VkEventWaits() {
    }

    /**
     * Checks every wait is an open event of {@code backend}.
     *
     * @throws IllegalArgumentException for a null, foreign or closed event
     */
    static void validate(GpuEvent[] waits, VkComputeBackend backend) {
        if (waits == null) {
            return;
        }
        for (int i = 0; i < waits.length; i++) {
            GpuEvent e = waits[i];
            if (!(e instanceof VkGpuEvent)) {
                throw new IllegalArgumentException("wait event " + i + " is not a Vulkan event of this backend: "
                        + (e == null ? "null" : e.getClass().getName()));
            }
            VkGpuEvent event = (VkGpuEvent) e;
            if (event.owner().backend() != backend) {
                throw new IllegalArgumentException("wait event " + i + " belongs to another Vulkan backend");
            }
            if (event.isClosed()) {
                throw new IllegalArgumentException("wait event " + i + " was already closed");
            }
        }
    }

    /** Submits the producer batches of other queues' events; runs without this queue's monitor. */
    static void flushProducers(GpuEvent[] waits, VkComputeQueue self) {
        if (waits == null) {
            return;
        }
        for (GpuEvent e : waits) {
            VkGpuEvent event = (VkGpuEvent) e;
            if (event.owner() != self) {
                event.owner().flushIfPending(event.value());
            }
        }
    }
}
