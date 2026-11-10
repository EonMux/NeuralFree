package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch timing on a Vulkan queue ({@link VkDispatchTimer}): under {@code -Dneuralfree.gpu.profile=true} every
 * dispatch, across several submissions, lands in the queue's {@link KernelTiming} under its OpenCL-style key
 * {@code name#global x local} with a positive device time, the timed dispatches still compute their results in
 * order, and a queue opened without the property keeps no timing at all. Runs on every Vulkan device, labelled by
 * index and name so a run can select one (e.g. {@code --tests '*VkDispatchTimerTest*Intel*'}).
 */
@RunWith(Parameterized.class)
public class VkDispatchTimerTest {

    private static final int SIZE = 256;
    /** The local size shaders/test/increment.comp fixes. */
    private static final int LOCAL = 256;

    private final int deviceIndex;
    private HWAcceleration hw;
    private VkComputeBackend backend;
    private GpuBuffer values;
    private KernelHandle increment;
    private String previous;

    public VkDispatchTimerTest(String label, int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> devices() {
        List<Object[]> devices = new ArrayList<>();
        try {
            HWAcceleration probe = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.VULKAN));
            try {
                List<VkDevice> found = probe.getVkDevices();
                for (int i = 0; i < found.size(); i++) {
                    devices.add(new Object[]{"vulkan:" + i + " " + found.get(i).getInfo().name(), i});
                }
            } finally {
                probe.shutdown();
            }
        } catch (RuntimeException | LinkageError e) {
            devices.clear();
        }
        if (devices.isEmpty()) {
            devices.add(new Object[]{"vulkan:none", -1});
        }
        return devices;
    }

    private void open(boolean profile) {
        Assume.assumeTrue("no Vulkan device", deviceIndex >= 0);
        previous = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, Boolean.toString(profile));
        hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.VULKAN));
        backend = (VkComputeBackend) hw.createBackend(hw.getVkDevices().get(deviceIndex));
        values = backend.allocateBuffer((long) SIZE * Float.BYTES);
        increment = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        backend.computeQueue().enqueueWriteBuffer(values, 0, (long) SIZE * Float.BYTES, new float[SIZE]);
        backend.computeQueue().finish();
    }

    @After
    public void tearDown() {
        if (increment != null) {
            increment.close();
        }
        if (values != null) {
            values.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
        if (deviceIndex < 0) {
            return;
        }
        if (previous == null) {
            System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        } else {
            System.setProperty(KernelTiming.PROFILE_PROPERTY, previous);
        }
    }

    private void dispatch(ComputeQueue queue, int n) {
        queue.enqueueKernelNoEvent(increment, new long[]{SIZE}, new long[]{LOCAL}, new Object[]{values, n});
    }

    @Test
    public void everyDispatchIsTimedUnderItsKey() {
        open(true);
        ComputeQueue queue = backend.computeQueue();
        KernelTiming timing = queue.kernelTiming();
        Assume.assumeTrue("the compute queue family writes no timestamps", timing != null);
        timing.reset();
        int dispatches = 3 * backend.dispatchesPerSubmit() + 5;
        for (int i = 0; i < dispatches; i++) {
            dispatch(queue, SIZE);
            if (i % 7 == 0) {
                queue.flush();
            }
        }
        float[] out = new float[SIZE];
        queue.enqueueReadBuffer(values, 0, (long) SIZE * Float.BYTES, out);
        queue.finish();
        float[] want = new float[SIZE];
        Arrays.fill(want, dispatches);
        assertArrayEquals("the timed dispatches ran in order", want, out, 0.0f);

        List<KernelTiming.Entry> entries = timing.snapshot();
        assertEquals(entries.toString(), 1, entries.size());
        KernelTiming.Entry e = entries.get(0);
        assertEquals("increment#" + SIZE + "x" + LOCAL, e.name());
        assertEquals(dispatches, e.count());
        assertTrue("device time " + e.totalNanos(), e.totalNanos() > 0);
        assertNotNull(timing.format());
    }

    @Test
    public void aQueueOpenedWithoutTheProfileKeepsNoTiming() {
        open(false);
        ComputeQueue queue = backend.computeQueue();
        dispatch(queue, SIZE);
        queue.finish();
        assertNull(queue.kernelTiming());
    }
}
