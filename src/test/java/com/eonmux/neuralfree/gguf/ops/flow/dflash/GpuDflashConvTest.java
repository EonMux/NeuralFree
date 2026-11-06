package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static org.junit.Assert.assertArrayEquals;

/** Dynamic depthwise convolution on each available GPU API against the host's causal block convolution. */
@RunWith(Parameterized.class)
public class GpuDflashConvTest {
    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() { return TrainParity.apis(); }

    private final TrainParity.Api api;

    public GpuDflashConvTest(TrainParity.Api api) { this.api = api; }

    @Test
    public void randomBlocksMatchBothSidesIncludingLongKernels() {
        try (TrainParity.Session session = TrainParity.open(api)) {
            ComputeBackend backend = session.backend();
            LlmKernels kernels = LlmKernels.open(backend);
            try (ComputeQueue queue = backend.createComputeQueue()) {
                for (int e : new int[]{12, 32, 60}) {
                    for (int group : new int[]{1, 2, e / 2, e}) {
                        for (int n : new int[]{1, 3, 7}) {
                            for (int k : new int[]{1, 2, 9}) {
                                check(backend, queue, kernels, e, group, n, k);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void check(ComputeBackend backend, ComputeQueue queue, LlmKernels kernels,
                              int e, int group, int n, int k) {
        DflashConv conv = new DflashConv(e, k, group);
        float[] x = TrainParity.uniform(11L + e + n, n * e, -2, 2);
        float[] dynamic = TrainParity.uniform(12L + group + k, n * conv.dynamicWidth(), -1, 1);
        float[] base = TrainParity.uniform(13L + k, conv.baseLength(), -1, 1);
        try (GpuBuffer bx = backend.allocateBuffer(x.length * 4L);
             GpuBuffer bd = backend.allocateBuffer(dynamic.length * 4L);
             GpuBuffer bb = backend.allocateBuffer(base.length * 4L);
             GpuBuffer out = backend.allocateBuffer(x.length * 4L)) {
            close(queue.enqueueWriteBuffer(bx, 0, x.length * 4L, x));
            close(queue.enqueueWriteBuffer(bd, 0, dynamic.length * 4L, dynamic));
            close(queue.enqueueWriteBuffer(bb, 0, base.length * 4L, base));
            KernelHandle kernel = kernels.get("LLM_DflashConv");
            for (int side = 0; side < 2; side++) {
                Object[] args = {e, k, group, n, side, bx, bd, bb, out};
                close(queue.enqueueKernel(kernel, new long[]{((n * e + 31L) / 32) * 32}, new long[]{32}, args));
                float[] actual = new float[x.length];
                GpuEvent read = queue.enqueueReadBuffer(out, 0, actual.length * 4L, actual);
                queue.finish();
                close(read);
                float[] expected = new float[x.length];
                conv.apply(x, 0, dynamic, 0, base, side, n, expected, 0);
                assertArrayEquals(apiLabel(backend, e, group, n, k, side), expected, actual, 3e-6f);
            }
        }
    }

    private static String apiLabel(ComputeBackend b, int e, int g, int n, int k, int side) {
        return b.backendType() + " E=" + e + " group=" + g + " block=" + n + " K=" + k + " side=" + side;
    }

    private static void close(GpuEvent event) { if (event != null) event.close(); }
}
