package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;

/** State-row routing writes the row without retaining one transfer event per decoded segment. */
public class GpuMixerEventTest {

    @Test
    public void routingReleasesEachWriteEventWithoutFinishingTheQueue() {
        try (FakeComputeBackend backend = FakeComputeBackend.builder().build();
             GpuBuffer rows = backend.allocateBuffer(Integer.BYTES)) {
            for (int row = 0; row < 100; row++) {
                GpuMixer.routeStateRow(row, rows, backend.computeQueue());
                assertEquals("routing must close its write event", 0, backend.openEventCount());
            }
            byte[] bytes = new byte[Integer.BYTES];
            try (GpuEvent read = backend.computeQueue().enqueueReadBuffer(rows, 0, bytes.length, bytes)) {
                backend.computeQueue().finish();
                assertEquals(99, ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt());
            }
            assertEquals(0, backend.openEventCount());
        }
    }
}
