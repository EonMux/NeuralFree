package com.eonmux.neuralfree.gguf.gpu.diag;

import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver.Stage;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The per-device reports of the diagnostics and the profiler over two devices, as a layer split observes them:
 * {@link GpuDiagnostics} names the device of every stage it reads and reads it on that device's queue, and the
 * {@link GpuProfiler} device views attribute each stage and layer to its device, count a step every view repeats
 * once, and synchronize every device's queue. {@link ObserverSlot} swaps an observer in and restores the base.
 */
public class DeviceObserversTest {

    private static final int WIDTH = 4;

    private final List<AutoCloseable> closeables = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
    }

    private FakeComputeBackend backend() {
        FakeComputeBackend b = FakeComputeBackend.create();
        closeables.add(b);
        return b;
    }

    private ScratchSlot slot(FakeComputeBackend backend, float... values) {
        GpuScratch scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, WIDTH).build();
        closeables.add(scratch);
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ((FakeGpuBuffer) x.buffer()).putFloats(x.offset(), values);
        return x;
    }

    @Test
    public void diagnosticsNameTheDeviceOfEveryStage() {
        FakeComputeBackend one = backend();
        FakeComputeBackend two = backend();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(bytes, true);
        GpuDiagnostics.Config cfg = new GpuDiagnostics.Config(true, false, Paths.get("unused"), 1);
        GpuDiagnostics first = new GpuDiagnostics(one.computeQueue(), cfg, out, "opencl0 'GeForce'");
        GpuDiagnostics second = new GpuDiagnostics(two.computeQueue(), cfg, out, "opencl1 'Intel'");
        first.onStepStart(0, 1);
        second.onStepStart(0, 1);
        first.onStageEnd(0, Stage.LAYER, slot(one, 1, 2, 3, 4), 1, WIDTH);
        second.onStageEnd(1, Stage.LAYER, slot(two, 5, 6, 7, 8), 1, WIDTH);
        String[] lines = new String(bytes.toByteArray(), StandardCharsets.UTF_8).split("\n");
        assertEquals(2, lines.length);
        assertTrue(lines[0], lines[0].contains("device=opencl0 'GeForce' layer-0"));
        assertTrue(lines[0], lines[0].contains("head=[1.00000, 2.00000"));
        assertTrue(lines[1], lines[1].contains("device=opencl1 'Intel' layer-1"));
        assertTrue(lines[1], lines[1].contains("head=[5.00000, 6.00000"));
        assertTrue("each device's stage is read on its own queue",
                two.ops().stream().anyMatch(op -> op.kind() == FakeComputeBackend.OpKind.READ));

        GpuDiagnostics unnamed = new GpuDiagnostics(one.computeQueue(), cfg, out);
        assertEquals(null, unnamed.device());
    }

    @Test
    public void profilerViewsAttributeStagesAndLayersToTheirDevice() {
        FakeComputeBackend one = backend();
        FakeComputeBackend two = backend();
        GpuProfiler profiler = new GpuProfiler(one.computeQueue());
        ExecutionObserver a = profiler.device("gpu-a", one.computeQueue());
        ExecutionObserver b = profiler.device("gpu-b", two.computeQueue());
        two.clearLog();

        a.onStepStart(0, 1);
        b.onStepStart(0, 1);
        stage(a, -1, Stage.EMBEDDING);
        stage(a, 0, Stage.LAYER);
        stage(a, 1, Stage.LAYER);
        stage(b, 2, Stage.LAYER);
        stage(a, -1, Stage.LOGITS);
        a.onStepEnd(0, 1);
        b.onStepEnd(0, 1);

        GpuProfiler.Report r = profiler.report();
        assertEquals("a step every view repeats counts once", 1, r.steps());
        assertEquals(1L, r.phases().get(GpuProfiler.STEP).count());
        assertEquals(3L, r.phases().get(Stage.LAYER.name()).count());
        assertEquals(Arrays.asList("gpu-a", "gpu-b"), new ArrayList<>(r.devicePhases().keySet()));
        assertEquals(Arrays.asList(Stage.EMBEDDING.name(), Stage.LAYER.name(), Stage.LOGITS.name()),
                new ArrayList<>(r.devicePhases().get("gpu-a").keySet()));
        assertEquals(Arrays.asList(Stage.LAYER.name()), new ArrayList<>(r.devicePhases().get("gpu-b").keySet()));
        assertArrayEquals(new String[]{"gpu-a", "gpu-a", "gpu-b"}, r.layerDevices());
        assertEquals("0-1", r.layersOf("gpu-a"));
        assertEquals("2", r.layersOf("gpu-b"));
        String text = r.format("decode", r.nanos(GpuProfiler.STEP));
        assertTrue(text, text.contains("on gpu-a (layers 0-1):"));
        assertTrue(text, text.contains("on gpu-b (layers 2):"));
        assertTrue("every boundary synchronizes every device's queue",
                two.ops().stream().anyMatch(op -> op.kind() == FakeComputeBackend.OpKind.FINISH));

        profiler.reset();
        assertTrue(profiler.report().devicePhases().isEmpty());
        assertEquals(0, profiler.report().layerDevices().length);
    }

    @Test
    public void aProfilerWithoutViewsReportsNoDevice() {
        FakeComputeBackend one = backend();
        GpuProfiler profiler = new GpuProfiler(one.computeQueue());
        profiler.onStepStart(0, 1);
        stage(profiler, 0, Stage.LAYER);
        profiler.onStepEnd(0, 1);
        GpuProfiler.Report r = profiler.report();
        assertEquals(1, r.steps());
        assertTrue(r.devicePhases().isEmpty());
        assertFalse(r.format("decode", 0).contains(" on "));
    }

    @Test
    public void theSlotSwapsAnObserverInAndRestoresTheBase() {
        ObserverSlot slot = new ObserverSlot();
        assertFalse(slot.enabled());
        List<String> seen = new ArrayList<>();
        ExecutionObserver recording = new ExecutionObserver() {
            @Override
            public void onStageStart(int layer, Stage stage) {
                seen.add(stage + "-" + layer);
            }
        };
        slot.use(recording);
        assertTrue(slot.enabled());
        slot.onStageStart(3, Stage.LAYER);
        slot.restore();
        assertFalse(slot.enabled());
        slot.onStageStart(4, Stage.LAYER);
        assertEquals(Arrays.asList("LAYER-3"), seen);
    }

    private static void stage(ExecutionObserver o, int layer, Stage stage) {
        o.onStageStart(layer, stage);
        o.onStageEnd(layer, stage, null, 1, WIDTH);
    }
}
