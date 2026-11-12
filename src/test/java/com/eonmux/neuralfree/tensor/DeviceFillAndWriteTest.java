package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TransferCountingBackend;
import org.junit.Assume;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * On each GPU API: {@link Tensor#full} fills on the device with the exact bits of its value ({@code -0}, NaN,
 * infinities), {@link Tensor#ones} and {@link Tensor#scalar} follow it, and a host write that returns without waiting
 * for its upload (the compute queue is ordered after it instead) is seen by the next kernel, never the values of the
 * write before it, and copies the host array before returning. Each queue is flushed before the other queue waits
 * on it, as OpenCL requires of a cross-queue wait: the compute queue before the write waits for pending train work,
 * and the writing queue before the compute queue waits for the write. Skipped where an API or the train kernels are
 * unavailable.
 */
@RunWith(Parameterized.class)
public class DeviceFillAndWriteTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;

    public DeviceFillAndWriteTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
    }

    @After
    public void tearDown() {
        if (session != null) {
            session.close();
        }
    }

    @Test
    public void fullWritesTheExactBitsOfItsValue() {
        for (float value : new float[]{-0.0f, Float.NaN, Float.NEGATIVE_INFINITY, 1.5e-42f, 3.25f}) {
            try (Tensor t = Tensor.full(gpu, Shape.of(3, 1000), value)) {
                float[] got = t.toFloatArray();
                for (int i = 0; i < got.length; i++) {
                    assertEquals("element " + i + " of full(" + value + ")", Float.floatToRawIntBits(value),
                            Float.floatToRawIntBits(got[i]));
                }
            }
        }
        try (Tensor ones = Tensor.ones(gpu, Shape.of(7));
             Tensor scalar = Tensor.scalar(gpu, -2.5f)) {
            float[] expected = new float[7];
            Arrays.fill(expected, 1f);
            assertArrayEquals(expected, ones.toFloatArray(), 0f);
            assertEquals(Shape.scalar(), scalar.shape());
            assertArrayEquals(new float[]{-2.5f}, scalar.toFloatArray(), 0f);
        }
    }

    @Test
    public void theNextKernelSeesEachWriteOfAReusedBuffer() {
        int n = 1 << 16;
        try (Tensor t = Tensor.zeros(gpu, Shape.of(n))) {
            float[] host = new float[n];
            for (int round = 1; round <= 40; round++) {
                Arrays.fill(host, round);
                GpuEvent done = t.storage().copyFromHostFloat(host, gpu.transferQueue());
                if (done != null) {
                    done.close();
                }
                Arrays.fill(host, -1f);
                try (Tensor doubled = TensorOps.mul(t, 2f)) {
                    float[] got = doubled.toFloatArray();
                    float expected = 2f * round;
                    assertEquals("round " + round + " first element", expected, got[0], 0f);
                    assertEquals("round " + round + " last element", expected, got[n - 1], 0f);
                }
            }
        }
    }

    @Test
    public void eachQueueIsFlushedBeforeTheOtherWaitsOnIt() {
        try (TransferCountingBackend counting = new TransferCountingBackend(gpu)) {
            Assume.assumeTrue("train kernels on the counting view", TrainGpu.forBackend(counting) != null);
            Assume.assumeTrue("a transfer queue of its own", counting.transferQueue() != counting.computeQueue());
            try (Tensor pending = Tensor.full(counting, Shape.of(16), 1f)) {
                counting.clearLog();
                try (Tensor t = Tensor.fromArray(counting, new float[]{1f, 2f, 3f})) {
                    List<String> log = counting.operations();
                    int write = log.indexOf("write transfer");
                    assertTrue("the write in " + log, write > 0);
                    assertEquals("the write waits on a flushed compute marker: " + log,
                            Arrays.asList("marker compute", "flush compute"), log.subList(write - 2, write));
                    assertEquals("the compute queue waits on the flushed write: " + log,
                            Arrays.asList("write transfer", "flush transfer", "marker compute"),
                            log.subList(write, Math.min(log.size(), write + 3)));
                    assertArrayEquals(new float[]{1f, 2f, 3f}, t.toFloatArray(), 0f);
                }
            }
        }
    }
}
