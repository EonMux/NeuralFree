package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Command timing on a Vulkan queue ({@link VkDispatchTimer}): under {@code -Dneuralfree.gpu.profile=true} every
 * dispatch, across several submissions, lands in the queue's {@link KernelTiming} under its OpenCL-style key
 * {@code name#global x local} with a positive device time, the timed dispatches still compute their results in
 * order, and a queue opened without the property keeps no timing at all. Under {@code step} no dispatch is timed
 * on its own: each submitted batch reports as one {@link KernelTiming#SUBMISSION_KEY} entry, and in both modes every
 * submission of a captured graph reports once under its {@link KernelTiming#graphKey(int)}: back-to-back submissions
 * of one captured command buffer and re-recorded ones alike, and a retire that covers only the first of two pending
 * submissions reads that one without waiting for the second. Runs on every Vulkan
 * device, labelled by index and name so a run can select one (e.g. {@code --tests '*VkDispatchTimerTest*Intel*'}).
 */
@RunWith(Parameterized.class)
public class VkDispatchTimerTest {

    private static final int SIZE = 256;
    /** The local size shaders/test/increment.comp fixes. */
    private static final int LOCAL = 256;
    /** The square output of the matmul that makes a submission long, and its reduction length. */
    private static final int MM = 256;
    private static final int MM_K = 1024;
    /** The device time a long graph submission is sized to. */
    private static final long LONG_SUBMISSION_NANOS = 40_000_000L;

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
        open(Boolean.toString(profile));
    }

    private void open(String profile) {
        Assume.assumeTrue("no Vulkan device", deviceIndex >= 0);
        previous = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, profile);
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
    public void theStepModeTimesEachSubmissionWhole() {
        open("step");
        ComputeQueue queue = backend.computeQueue();
        KernelTiming timing = queue.kernelTiming();
        Assume.assumeTrue("the compute queue family writes no timestamps", timing != null);
        timing.reset();
        int rounds = 3;
        int perRound = 4;
        for (int r = 0; r < rounds; r++) {
            for (int i = 0; i < perRound; i++) {
                dispatch(queue, SIZE);
            }
            queue.finish();
        }
        float[] out = new float[SIZE];
        queue.enqueueReadBuffer(values, 0, (long) SIZE * Float.BYTES, out);
        queue.finish();
        float[] want = new float[SIZE];
        Arrays.fill(want, rounds * perRound);
        assertArrayEquals("the dispatches ran in order", want, out, 0.0f);

        List<KernelTiming.Entry> entries = timing.snapshot();
        assertEquals("no dispatch has an entry of its own: " + entries, 1, entries.size());
        KernelTiming.Entry e = entries.get(0);
        assertEquals(KernelTiming.SUBMISSION_KEY, e.name());
        assertTrue("one entry per submitted batch, at least one per round: " + e.count(), e.count() >= rounds);
        assertTrue("device time " + e.totalNanos(), e.totalNanos() > 0);
    }

    @Test
    public void everySubmissionOfACapturedGraphIsTimedOnce() {
        for (String mode : new String[]{"true", "step"}) {
            open(mode);
            ComputeQueue queue = backend.computeQueue();
            KernelTiming timing = queue.kernelTiming();
            Assume.assumeTrue("the compute queue family writes no timestamps", timing != null);
            assertTrue(mode + ": the queue times its graph launches", queue.timesGraphLaunches());
            queue.beginCapture();
            dispatch(queue, SIZE);
            dispatch(queue, SIZE);
            GpuGraph graph = queue.endCapture();
            assertNotNull(graph);
            try {
                timing.reset();
                graph.launch(queue).close();
                queue.finish();
                for (int i = 0; i < 2; i++) {
                    graph.replay(() -> {
                        dispatch(queue, SIZE);
                        dispatch(queue, SIZE);
                    });
                    queue.finish();
                }
                float[] out = new float[SIZE];
                queue.enqueueReadBuffer(values, 0, (long) SIZE * Float.BYTES, out);
                queue.finish();
                float[] want = new float[SIZE];
                Arrays.fill(want, 3 * 2);
                assertArrayEquals(mode + ": the launch and both replays ran", want, out, 0.0f);
                KernelTiming.Entry graphEntry = null;
                for (KernelTiming.Entry e : timing.snapshot()) {
                    if (e.name().equals(KernelTiming.graphKey(2))) {
                        graphEntry = e;
                    }
                }
                assertNotNull(mode + ": the graph submissions are timed: " + timing.format(), graphEntry);
                assertEquals(mode + ": one entry per submission", 3, graphEntry.count());
                assertTrue(mode + ": device time " + graphEntry.totalNanos(), graphEntry.totalNanos() > 0);
            } finally {
                graph.close();
            }
            tearDown();
            values = null;
            increment = null;
            backend = null;
            hw = null;
        }
    }

    @Test
    public void backToBackSubmissionsOfAGraphAreEachTimedOnce() {
        for (String mode : new String[]{"true", "step"}) {
            open(mode);
            ComputeQueue queue = backend.computeQueue();
            KernelTiming timing = queue.kernelTiming();
            Assume.assumeTrue("the compute queue family writes no timestamps", timing != null);
            queue.beginCapture();
            dispatch(queue, SIZE);
            dispatch(queue, SIZE);
            GpuGraph graph = queue.endCapture();
            assertNotNull(graph);
            try {
                timing.reset();
                // the launch and the unchanged replay submit the captured command buffer twice before any retire;
                // the changed replay submits a re-recorded ring slot
                graph.launch(queue).close();
                graph.replay(() -> {
                    dispatch(queue, SIZE);
                    dispatch(queue, SIZE);
                });
                graph.replay(() -> {
                    dispatch(queue, SIZE / 2);
                    dispatch(queue, SIZE / 2);
                });
                float[] out = new float[SIZE];
                queue.enqueueReadBuffer(values, 0, (long) SIZE * Float.BYTES, out);
                queue.finish();
                float[] want = new float[SIZE];
                Arrays.fill(want, 0, SIZE / 2, 3 * 2);
                Arrays.fill(want, SIZE / 2, SIZE, 2 * 2);
                assertArrayEquals(mode + ": the three submissions ran in order", want, out, 0.0f);
                KernelTiming.Entry graphEntry = graphEntry(timing, 2);
                assertNotNull(mode + ": the graph submissions are timed: " + timing.format(), graphEntry);
                assertEquals(mode + ": one entry per submission", 3, graphEntry.count());
                assertTrue(mode + ": device time " + graphEntry.totalNanos(), graphEntry.totalNanos() > 0);
            } finally {
                graph.close();
            }
            tearDown();
            values = null;
            increment = null;
            backend = null;
            hw = null;
        }
    }

    @Test
    public void aRetireOfTheFirstOfTwoPendingSubmissionsReadsItAloneWithoutWaiting() throws InterruptedException {
        open("step");
        VkComputeQueue queue = (VkComputeQueue) backend.computeQueue();
        KernelTiming timing = queue.kernelTiming();
        Assume.assumeTrue("the compute queue family writes no timestamps", timing != null);
        KernelHandle matmul = backend.compileKernel("matmul", "shaders/test/matmul.spv", null);
        GpuBuffer a = backend.allocateBuffer((long) MM * MM_K * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer((long) MM_K * MM * Float.BYTES);
        GpuBuffer c = backend.allocateBuffer((long) MM * MM * Float.BYTES);
        GpuGraph graph = null;
        try {
            queue.enqueueWriteBuffer(a, 0, (long) MM * MM_K * Float.BYTES, new float[MM * MM_K]);
            queue.enqueueWriteBuffer(b, 0, (long) MM_K * MM * Float.BYTES, new float[MM_K * MM]);
            Runnable step = () -> queue.enqueueKernelNoEvent(matmul, new long[]{MM, MM}, new long[]{16, 16},
                    new Object[]{a, b, c, MM, MM_K, MM});
            queue.finish();
            timing.reset();
            step.run();
            queue.finish();
            int repeats = (int) Math.min(1024, Math.max(1, LONG_SUBMISSION_NANOS / Math.max(1, timing.totalNanos())));
            queue.beginCapture();
            for (int i = 0; i < repeats; i++) {
                step.run();
            }
            graph = queue.endCapture();
            assertNotNull(graph);
            timing.reset();
            GpuEvent launched = graph.launch(queue);
            long first = ((VkGpuEvent) launched).value();
            launched.close();
            int n = repeats;
            graph.replay(() -> {
                for (int i = 0; i < n; i++) {
                    step.run();
                }
            });
            VkTimelineWait.wait(backend.device(), queue.timelineSemaphore(), first);
            // the second submission of the captured command buffer starts once the first completes
            Thread.sleep(2);
            long start = System.nanoTime();
            queue.waitForValue(first);
            long retireNanos = System.nanoTime() - start;
            KernelTiming.Entry e = graphEntry(timing, repeats);
            assertNotNull("the first submission is timed: " + timing.format(), e);
            assertEquals("the retire reads the first submission alone", 1, e.count());
            long firstNanos = e.totalNanos();
            assertTrue("the retire of the first submission took " + retireNanos + " ns against its "
                    + firstNanos + " ns on the device: it waited for the second", retireNanos < firstNanos / 4);
            queue.finish();
            e = graphEntry(timing, repeats);
            assertEquals("the finish reads the second submission", 2, e.count());
            assertTrue("each submission adds its own time", e.totalNanos() > firstNanos);
        } finally {
            if (graph != null) {
                graph.close();
            }
            c.close();
            b.close();
            a.close();
            matmul.close();
        }
    }

    private static KernelTiming.Entry graphEntry(KernelTiming timing, int dispatches) {
        for (KernelTiming.Entry e : timing.snapshot()) {
            if (e.name().equals(KernelTiming.graphKey(dispatches))) {
                return e;
            }
        }
        return null;
    }

    @Test
    public void aQueueOpenedWithoutTheProfileKeepsNoTiming() {
        open(false);
        ComputeQueue queue = backend.computeQueue();
        dispatch(queue, SIZE);
        queue.finish();
        assertNull(queue.kernelTiming());
        assertFalse(queue.timesGraphLaunches());
    }
}
