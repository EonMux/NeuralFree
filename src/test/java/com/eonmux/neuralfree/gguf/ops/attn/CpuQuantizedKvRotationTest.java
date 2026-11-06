package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import org.junit.Test;

import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

/**
 * {@link CpuQuantizedKvRotation} over the five-plane query buffer of dual-chunk attention: every plane rotates like
 * the keys it is scored against, the slack between planes and the rows past the step stay untouched, and the
 * single-view form keeps rotating the first plane alone.
 */
public class CpuQuantizedKvRotationTest {

    private static final int WIDTH = 64;
    private static final int ROWS = 2;
    private static final int PLANES = 5;
    private static final int STRIDE = 200;

    private static RotationPlan rotateK() {
        RotationPlan plan = RotationPlan.of(KvDType.Q8_0, KvDType.F32, WIDTH, 0, WIDTH, 0, true, false);
        assertEquals(WIDTH, plan.nRotK());
        return plan;
    }

    private static float[] noise(int n) {
        SplittableRandom r = new SplittableRandom(7);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (r.nextDouble() * 2.0 - 1.0);
        }
        return v;
    }

    private static float[] rotated(float[] v, int rows, int width, int stride, int planes) {
        float[] expected = v.clone();
        HadamardSpec spec = HadamardSpec.rotation(WIDTH);
        for (int plane = 0; plane < planes; plane++) {
            CpuHadamard.applyInPlace(spec, expected, plane * stride, rows, width);
        }
        return expected;
    }

    @Test
    public void everyQueryPlaneRotatesAndTheSlackDoesNot() {
        CpuQuantizedKvRotation rotation = CpuQuantizedKvRotation.of(rotateK(), WIDTH, WIDTH, 0, 0);
        assertNotNull(rotation);
        float[] q = noise(PLANES * STRIDE + STRIDE);
        float[] k = noise(ROWS * WIDTH);
        float[] expectedQ = rotated(q, ROWS, WIDTH, STRIDE, PLANES);
        float[] expectedK = k.clone();
        CpuHadamard.applyInPlace(HadamardSpec.rotation(WIDTH), expectedK, 0, ROWS, WIDTH);

        rotation.beforeStorePlanes(ROWS, q, PLANES, STRIDE, k, null);

        assertArrayEquals(expectedQ, q, 0f);
        assertArrayEquals(expectedK, k, 0f);
        // the tail past the last plane's rows is the same as before
        for (int i = PLANES * STRIDE; i < q.length; i++) {
            assertEquals("value " + i, expectedQ[i], q[i], 0f);
        }
    }

    @Test
    public void onePlaneIsThePlainQueryRotation() {
        CpuQuantizedKvRotation rotation = CpuQuantizedKvRotation.of(rotateK(), WIDTH, WIDTH, 0, 0);
        float[] q = noise(ROWS * WIDTH);
        float[] k = noise(ROWS * WIDTH);
        float[] expected = q.clone();
        CpuHadamard.applyInPlace(HadamardSpec.rotation(WIDTH), expected, 0, ROWS, WIDTH);

        rotation.beforeStore(ROWS, q, k, null);

        assertArrayEquals(expected, q, 0f);
    }

    @Test
    public void aRotationPlanThatRotatesNothingYieldsNoRotation() {
        RotationPlan none = RotationPlan.of(KvDType.F32, KvDType.F32, WIDTH, 0, WIDTH, 0, true, false);
        assertNull(CpuQuantizedKvRotation.of(none, WIDTH, WIDTH, 0, 0));
    }
}
