package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import jcuda.driver.CUfunction_attribute;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static jcuda.driver.JCudaDriver.cuFuncGetAttribute;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dynamic shared memory on the device: a kernel whose buffer is {@code extern .shared} fills and reads back more
 * than the 48 KiB a block gets without the opt-in, once its function reserved the size - through the backend, or
 * at load from the size its module declares - in a direct launch, in a captured graph's launch and in the replays
 * that rewrite its node's parameters; clones launch with the function's reservation, a second size, a negative one
 * or one past the device's opt-in limit is refused, and of two threads reserving two sizes at once one wins and the
 * function's opt-in attribute holds its size.
 */
public class CudaDynamicSharedMemoryTest {

    /** Each thread fills words {@code tid, tid + ntid, ...} of the dynamic buffer with {@code 3 * index}, then writes
     *  word {@code words - 1 - tid} to {@code out[tid]}. */
    private static final String FILL_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".extern .shared .align 16 .b8 dyn[];\n" +
            "\n" +
            ".visible .entry dyn_fill(\n" +
            "    .param .u64 out,\n" +
            "    .param .u32 words\n" +
            ") {\n" +
            "    .reg .b32   %r<9>;\n" +
            "    .reg .f32   %f<2>;\n" +
            "    .reg .b64   %rd<4>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [out];\n" +
            "    ld.param.u32    %r1, [words];\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, dyn;\n" +
            "    mov.u32         %r5, %r2;\n" +
            "L_fill:\n" +
            "    setp.ge.u32     p0, %r5, %r1;\n" +
            "    @p0 bra         L_read;\n" +
            "    mul.lo.u32      %r6, %r5, 3;\n" +
            "    shl.b32         %r7, %r5, 2;\n" +
            "    add.u32         %r7, %r7, %r4;\n" +
            "    st.shared.u32   [%r7], %r6;\n" +
            "    add.u32         %r5, %r5, %r3;\n" +
            "    bra             L_fill;\n" +
            "L_read:\n" +
            "    bar.sync        0;\n" +
            "    sub.u32         %r5, %r1, 1;\n" +
            "    sub.u32         %r5, %r5, %r2;\n" +
            "    shl.b32         %r7, %r5, 2;\n" +
            "    add.u32         %r7, %r7, %r4;\n" +
            "    ld.shared.u32   %r8, [%r7];\n" +
            "    cvt.rn.f32.u32  %f1, %r8;\n" +
            "    mul.wide.u32    %rd2, %r2, 4;\n" +
            "    add.s64         %rd3, %rd1, %rd2;\n" +
            "    st.global.f32   [%rd3], %f1;\n" +
            "    ret;\n" +
            "}\n";

    /** {@link #FILL_PTX} as {@code dyn_auto}, its module declaring the kernel's dynamic shared memory through
     *  {@code dyn_auto_dynamic_shared} ({@link CudaComputeBackend#DYNAMIC_SHARED_SUFFIX}). */
    private static final String DECLARED_PTX = FILL_PTX
            .replace("dyn_fill(", "dyn_auto(")
            .replace(".extern .shared", ".visible .const .align 4 .u32 dyn_auto"
                    + CudaComputeBackend.DYNAMIC_SHARED_SUFFIX + " = 65536;\n.extern .shared");

    private static final int THREADS = 64;
    /** 64 KiB: past the 48 KiB a block launches with before the function opts in. */
    private static final int BYTES = 64 * 1024;
    private static final int WORDS = BYTES / 4;
    /** Fresh functions the concurrent-reservation test races two sizes on. */
    private static final int RACE_ROUNDS = 200;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private GpuBuffer out;
    private KernelHandle fill;

    @Before
    public void setUp() {
        HWAcceleration accel = null;
        try {
            accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeTrue("no CUDA GPU", !accel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(accel.getCUDevices().get(0));
            hwAccel = accel;
            accel = null;
        } finally {
            if (accel != null) {
                accel.shutdown();
            }
        }
        Assume.assumeTrue("the device offers no " + BYTES + "-byte block",
                backend.capabilities().getMaxLocalMemoryBytes() >= BYTES);
        queue = backend.computeQueue();
        out = backend.allocateBuffer((long) THREADS * Float.BYTES);
        fill = backend.compileKernel("dyn_fill", FILL_PTX, null);
    }

    @After
    public void tearDown() {
        if (out != null) {
            out.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    private void dispatch(KernelHandle kernel) {
        dispatch(kernel, out);
    }

    private void dispatch(KernelHandle kernel, GpuBuffer target) {
        queue.enqueueKernelNoEvent(kernel, new long[]{THREADS}, new long[]{THREADS}, new Object[]{target, WORDS});
    }

    private float[] readOut() {
        return read(out);
    }

    private float[] read(GpuBuffer buffer) {
        float[] values = new float[THREADS];
        queue.enqueueReadBuffer(buffer, 0, (long) THREADS * Float.BYTES, values);
        queue.finish();
        return values;
    }

    private void clearOut() {
        clear(out);
    }

    private void clear(GpuBuffer buffer) {
        queue.enqueueFillBuffer(buffer, 0, (long) THREADS * Float.BYTES, 0);
        queue.finish();
    }

    private static float[] expected() {
        float[] values = new float[THREADS];
        for (int t = 0; t < THREADS; t++) {
            values[t] = 3f * (WORDS - 1 - t);
        }
        return values;
    }

    @Test
    public void aDirectLaunchUsesTheReservedDynamicSharedMemory() {
        backend.reserveDynamicSharedMemory(fill, BYTES);
        assertEquals(BYTES, ((CudaKernelHandle) fill).dynamicSharedMemoryBytes());
        dispatch(fill);
        assertArrayEquals(expected(), readOut(), 0f);
    }

    @Test
    public void aModuleDeclaringTheKernelsDynamicSharedMemoryReservesItAtLoad() {
        KernelHandle declared = backend.compileKernel("dyn_auto", DECLARED_PTX, null);
        assertEquals(BYTES, ((CudaKernelHandle) declared).dynamicSharedMemoryBytes());
        assertEquals("a module without the symbol reserves nothing", 0,
                ((CudaKernelHandle) fill).dynamicSharedMemoryBytes());
        dispatch(declared);
        assertArrayEquals(expected(), readOut(), 0f);
    }

    /**
     * The captured node launches with the reservation, and so do the node parameters a replay rewrites
     * ({@code cuGraphExecKernelNodeSetParams}): a body replay that moves the output to another buffer, and a patched
     * replay that moves it again.
     */
    @Test
    public void aCapturedGraphLaunchesAndReplaysWithTheReservedDynamicSharedMemory() {
        backend.reserveDynamicSharedMemory(fill, BYTES);
        float[] zeros = new float[THREADS];
        GpuBuffer second = backend.allocateBuffer((long) THREADS * Float.BYTES);
        GpuBuffer third = backend.allocateBuffer((long) THREADS * Float.BYTES);
        try {
            queue.beginCapture();
            dispatch(fill);
            GpuGraph graph = queue.endCapture();
            assertNotNull(graph);
            try {
                clearOut();
                graph.launch(queue);
                assertArrayEquals("the captured node", expected(), readOut(), 0f);
                clearOut();
                clear(second);
                graph.replay(() -> dispatch(fill, second));
                assertArrayEquals("the replay rewrote the node's parameters", new int[]{0},
                        graph.takeReplayChanges());
                assertArrayEquals("the rewritten node", expected(), read(second), 0f);
                assertArrayEquals("the rewritten node no longer writes the captured output", zeros, readOut(), 0f);
                Assume.assumeTrue("the graph has no patched replay", graph.supportsPatchedReplay());
                clear(second);
                clear(third);
                GpuGraphPatchPlan plan = (pos, sink) -> {
                    sink.patchKernel(0, null, new int[]{0}, new Object[]{third});
                    return true;
                };
                assertTrue(graph.replayPatched(plan, 0));
                assertArrayEquals("the patched node", expected(), read(third), 0f);
                assertArrayEquals("the patched node moved its output", zeros, read(second), 0f);
            } finally {
                graph.close();
            }
        } finally {
            third.close();
            second.close();
        }
    }

    /**
     * Two threads reserve two sizes past the default 48 KiB for a fresh function at once, round after round: one
     * reservation wins and the other is refused, and the function's opt-in attribute holds the winner's size, so its
     * launches run.
     */
    @Test
    public void concurrentReservationsOfTwoSizesLeaveTheAttributeAtTheRecordedSize() throws Exception {
        int[] sizes = {BYTES - 8 * 1024, BYTES};
        ExecutorService pool = Executors.newFixedThreadPool(sizes.length);
        try {
            for (int round = 0; round < RACE_ROUNDS; round++) {
                KernelHandle kernel = backend.compileKernel("dyn_fill", FILL_PTX + "// round " + round + "\n", null);
                CyclicBarrier start = new CyclicBarrier(sizes.length);
                List<Future<Boolean>> reservations = new ArrayList<>();
                for (int size : sizes) {
                    Callable<Boolean> reserve = () -> {
                        start.await(10, TimeUnit.SECONDS);
                        try {
                            backend.reserveDynamicSharedMemory(kernel, size);
                            return true;
                        } catch (IllegalStateException refused) {
                            assertTrue(refused.getMessage(), refused.getMessage().contains("already reserved"));
                            return false;
                        }
                    };
                    reservations.add(pool.submit(reserve));
                }
                int won = 0;
                for (Future<Boolean> r : reservations) {
                    won += result(r) ? 1 : 0;
                }
                CudaKernelHandle handle = (CudaKernelHandle) kernel;
                int recorded = handle.dynamicSharedMemoryBytes();
                assertEquals("round " + round + ": reservations that succeeded", 1, won);
                assertEquals("round " + round + ": the function's opt-in attribute", recorded,
                        maxDynamicSharedBytes(handle));
                if (round == RACE_ROUNDS - 1) {
                    dispatch(kernel);
                    assertArrayEquals(expected(), readOut(), 0f);
                }
            }
        } finally {
            pool.shutdownNow();
            assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    private static boolean result(Future<Boolean> f) throws Exception {
        try {
            return f.get(30, TimeUnit.SECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof Exception) {
                throw (Exception) cause;
            }
            throw (Error) cause;
        }
    }

    private int maxDynamicSharedBytes(CudaKernelHandle handle) {
        backend.enter();
        int[] value = new int[1];
        CudaErrors.check("cuFuncGetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)", () -> cuFuncGetAttribute(value,
                CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, handle.function()));
        return value[0];
    }

    @Test
    public void clonesLaunchWithTheFunctionsReservation() {
        backend.reserveDynamicSharedMemory(fill, BYTES);
        KernelHandle clone = backend.cloneKernel(fill);
        try {
            assertEquals(BYTES, ((CudaKernelHandle) clone).dynamicSharedMemoryBytes());
            dispatch(clone);
            assertArrayEquals(expected(), readOut(), 0f);
        } finally {
            clone.close();
        }
    }

    @Test
    public void aSecondSizeANegativeSizeAndASizePastTheLimitAreRefused() {
        int limit = (int) Math.min(Integer.MAX_VALUE, backend.capabilities().getMaxLocalMemoryBytes());
        IllegalArgumentException past = assertThrows(IllegalArgumentException.class,
                () -> backend.reserveDynamicSharedMemory(fill, limit + 1));
        assertTrue(past.getMessage(), past.getMessage().contains("the device allows"));
        assertThrows(IllegalArgumentException.class, () -> backend.reserveDynamicSharedMemory(fill, -1));
        backend.reserveDynamicSharedMemory(fill, BYTES);
        backend.reserveDynamicSharedMemory(fill, BYTES);
        IllegalStateException second = assertThrows(IllegalStateException.class,
                () -> backend.reserveDynamicSharedMemory(fill, BYTES / 2));
        assertTrue(second.getMessage(), second.getMessage().contains("already reserved"));
        assertEquals(BYTES, ((CudaKernelHandle) fill).dynamicSharedMemoryBytes());
    }
}
