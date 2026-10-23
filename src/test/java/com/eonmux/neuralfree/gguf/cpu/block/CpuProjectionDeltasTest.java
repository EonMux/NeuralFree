package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Serving, registration, application and removal of projection deltas. */
public class CpuProjectionDeltasTest {

    private static final String Q = TensorNames.weight(TensorNames.ATTN_Q);

    private static LayerSpec layer(int il) {
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, 4).build())
                .build();
    }

    /** {@code y[t] += k * sum(x[t])} over two-wide rows. */
    private static CpuProjectionDeltas.Delta sumTimes(float k) {
        return (n, x, xOff, xStride, y, yOff, yStride) -> {
            for (int t = 0; t < n; t++) {
                float s = x[xOff + t * xStride] + x[xOff + t * xStride + 1];
                y[yOff + t * yStride] += k * s;
                y[yOff + t * yStride + 1] += k * s;
            }
        };
    }

    @Test
    public void registeredDeltasAreAppliedThroughTheLiveSiteAndRemovedByIdentity() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        LayerSpec l = layer(3);
        CpuProjectionDeltas.Site site = deltas.serve(l, Q, null);
        assertTrue(site.isEmpty());

        CpuProjectionDeltas.Delta a = sumTimes(1.0f);
        CpuProjectionDeltas.Delta b = sumTimes(10.0f);
        deltas.register(l, Q, a);
        deltas.register(layer(3), Q, b);
        assertFalse(site.isEmpty());

        float[] x = {1, 2, 3, 4};
        float[] y = new float[6];
        site.apply(2, x, 0, 2, y, 1, 3);
        assertArrayEquals(new float[]{0, 33, 33, 0, 77, 77}, y, 0f);

        deltas.unregister(l, Q, a);
        float[] y2 = new float[2];
        site.apply(1, x, 0, 2, y2, 0, 2);
        assertArrayEquals(new float[]{30, 30}, y2, 0f);

        deltas.unregister(l, Q, b);
        assertTrue(site.isEmpty());
    }

    @Test
    public void sitesAreKeyedByLayerAndWeight() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        CpuProjectionDeltas.Site s0 = deltas.serve(layer(0), Q, null);
        CpuProjectionDeltas.Site s1 = deltas.serve(layer(1), Q, null);
        CpuProjectionDeltas.Site k0 = deltas.serve(layer(0), TensorNames.weight(TensorNames.ATTN_K), null);
        deltas.register(layer(1), Q, sumTimes(1.0f));
        assertTrue(s0.isEmpty());
        assertFalse(s1.isEmpty());
        assertTrue(k0.isEmpty());
    }

    @Test
    public void registrationOnAnUnservedProjectionFails() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        try {
            deltas.register(layer(0), Q, sumTimes(1.0f));
            fail("an unserved projection must refuse a delta");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains(Q));
        }
    }

    @Test
    public void registrationOnARefusingProjectionFailsWithTheReason() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        deltas.serve(layer(0), Q, "rotated input");
        try {
            deltas.register(layer(0), Q, sumTimes(1.0f));
            fail("a refusing projection must reject a delta");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage().contains("rotated input"));
        }
    }
}
