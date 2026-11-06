package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.training.LogCapture;
import org.apache.logging.log4j.Level;
import org.junit.Test;

import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.*;

/** CUDA's existing event pool releases outstanding driver events and diagnoses leaked leases at backend close. */
public class CudaEventOwnershipTest {

    @Test
    public void backendCloseDestroysLeakedEventsAndWarnsOnce() {
        GpuTestUtil.assumeCudaAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        try (CudaComputeBackend backend = new CudaComputeBackend(hw.getCUDevices().get(0));
             LogCapture logs = new LogCapture(CudaEventPool.class)) {
            ComputeQueue caller = backend.createComputeQueue();
            CudaGpuEvent first = (CudaGpuEvent) backend.computeQueue().enqueueMarker();
            CudaGpuEvent shared = first.share();
            CudaGpuEvent second = (CudaGpuEvent) caller.enqueueMarker();
            caller.close();
            assertEquals(2, backend.eventPool().createdCount());

            backend.close();
            backend.close();

            assertEquals(0, backend.eventPool().createdCount());
            assertThrows(IllegalStateException.class, first::getCUEvent);
            assertThrows(IllegalStateException.class, shared::getCUEvent);
            assertThrows(IllegalStateException.class, second::getCUEvent);
            List<String> warnings = logs.messages(Level.WARN);
            assertEquals(warnings.toString(), 1, warnings.size());
            assertTrue(warnings.get(0), warnings.get(0).contains("2"));
            assertTrue(warnings.get(0), warnings.get(0).contains("CUDA backend"));
            first.close();
            shared.close();
            second.close();
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void closedLeasesDoNotProduceALeakWarning() {
        GpuTestUtil.assumeCudaAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        try (CudaComputeBackend backend = new CudaComputeBackend(hw.getCUDevices().get(0));
             LogCapture logs = new LogCapture(CudaEventPool.class)) {
            try (GpuEvent event = backend.transferQueue().enqueueMarker()) {
                assertNotNull(event);
            }
            backend.close();
            assertTrue(logs.messages(Level.WARN).toString(), logs.messages(Level.WARN).isEmpty());
        } finally {
            hw.shutdown();
        }
    }
}
