package com.eonmux.neuralfree.gguf.gpu.diag;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-op tensor trace: it prints the values of the stages its filters select, truncates a wide row to its first
 * and last entries as llama.cpp's {@code common_debug_print_tensor} does, counts steps and aborts on a NaN stage.
 */
public class TensorTraceObserverTest {

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

    private ScratchSlot slot(FakeComputeBackend backend, int width, float... values) {
        GpuScratch scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, width).build();
        closeables.add(scratch);
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ((FakeGpuBuffer) x.buffer()).putFloats(x.offset(), values);
        return x;
    }

    private static String decode(ByteArrayOutputStream bytes) {
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    @Test
    public void aFilterSelectsOnlyTheStagesWhoseLabelStartsWithIt() {
        FakeComputeBackend backend = backend();
        ScratchSlot x = slot(backend, WIDTH, 1, 2, 3, 4);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TensorTraceObserver trace = new TensorTraceObserver(backend.computeQueue(),
                new PrintStream(bytes, true), Collections.singletonList("mixer"), false, 0, "gpu-a");
        trace.onStepStart(0, 1);
        trace.onStageEnd(-1, Stage.EMBEDDING, x, 1, WIDTH);
        trace.onStageEnd(1, Stage.MIXER, x, 1, WIDTH);
        trace.onStageEnd(1, Stage.LAYER, x, 1, WIDTH);
        String text = decode(bytes);
        assertTrue(text, text.contains("mixer-1"));
        assertFalse(text, text.contains("embedding"));
        assertFalse(text, text.contains("layer-1"));
        assertTrue(text, text.contains("device=gpu-a"));
        assertTrue(text, text.contains("1.0000"));
        assertTrue(text, text.contains("4.0000"));
        assertTrue(text, text.contains("sum = 10.000000"));
    }

    @Test
    public void aWideRowPrintsItsFirstAndLastEntriesWithAnEllipsis() {
        FakeComputeBackend backend = backend();
        float[] values = new float[10];
        for (int i = 0; i < values.length; i++) {
            values[i] = 100 + i;
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TensorTraceObserver trace = new TensorTraceObserver(backend.computeQueue(),
                new PrintStream(bytes, true), Collections.emptyList(), false, 0, null);
        trace.onStepStart(0, 1);
        trace.onStageEnd(0, Stage.MIXER, slot(backend, 10, values), 1, 10);
        String text = decode(bytes);
        assertTrue(text, text.contains("100.0000"));
        assertTrue(text, text.contains("102.0000"));
        assertTrue(text, text.contains("107.0000"));
        assertTrue(text, text.contains("109.0000"));
        assertFalse("the middle is elided", text.contains("104.0000"));
        assertTrue(text, text.contains("...,"));
    }

    @Test
    public void stepsLimitTheStagesReported() {
        FakeComputeBackend backend = backend();
        ScratchSlot x = slot(backend, WIDTH, 1, 2, 3, 4);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TensorTraceObserver trace = new TensorTraceObserver(backend.computeQueue(),
                new PrintStream(bytes, true), Collections.emptyList(), false, 1, null);
        trace.onStepStart(0, 1);
        trace.onStageEnd(0, Stage.LAYER, x, 1, WIDTH);
        trace.onStepStart(1, 1);
        trace.onStageEnd(0, Stage.LAYER, x, 1, WIDTH);
        String text = decode(bytes);
        assertEquals(text, 1, occurrences(text, "[TRACE]"));
        assertTrue(text, text.contains("step=1"));
    }

    @Test
    public void abortOnNanThrowsWhenAStageSumsToNan() {
        FakeComputeBackend backend = backend();
        ScratchSlot x = slot(backend, WIDTH, 1, Float.NaN, 3, 4);
        TensorTraceObserver trace = new TensorTraceObserver(backend.computeQueue(),
                new PrintStream(new ByteArrayOutputStream(), true), Collections.emptyList(), true, 0, null);
        trace.onStepStart(0, 1);
        TensorTraceObserver.NonFiniteException e = assertThrows(TensorTraceObserver.NonFiniteException.class,
                () -> trace.onStageEnd(0, Stage.LAYER, x, 1, WIDTH));
        assertTrue(e.getMessage(), e.getMessage().contains("NaN"));
        assertTrue(e.getMessage(), e.getMessage().contains("layer-0"));
    }

    @Test
    public void anInvalidFilterIsRefusedByName() {
        FakeComputeBackend backend = backend();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new TensorTraceObserver(backend.computeQueue(), new PrintStream(new ByteArrayOutputStream()),
                        Collections.singletonList("[unclosed"), false, 0, null));
        assertTrue(e.getMessage(), e.getMessage().contains("[unclosed"));
    }

    private static int occurrences(String text, String needle) {
        int count = 0;
        int at = text.indexOf(needle);
        while (at >= 0) {
            count++;
            at = text.indexOf(needle, at + needle.length());
        }
        return count;
    }
}
