package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The softplus output gates: laguna's per-head softplus broadcast over each head (models/laguna.cpp:247-258) and
 * k2-horizon's base-2 softplus {@code softplus(g ln 2) / ln 2} per element (models/k2-horizon.cpp:283-288), their
 * kernel kind codes and widths.
 */
public class AttnExtMathGateTest {

    private static final int N_HEAD = 3;
    private static final int HEAD_DIM = 4;

    private static double softplus(double x) {
        return x > 20 ? x : Math.log1p(Math.exp(x));
    }

    @Test
    public void kindCodesFollowTheEnumOrder() {
        assertEquals(4, AttnExtMath.gateKernelKind(AttnGateKind.PER_HEAD_SOFTPLUS));
        assertEquals(5, AttnExtMath.gateKernelKind(AttnGateKind.SOFTPLUS_BASE2));
        assertEquals(AttnGateKind.PER_HEAD_SOFTPLUS.ordinal() - 2,
                AttnExtMath.gateKernelKind(AttnGateKind.PER_HEAD_SOFTPLUS));
        assertEquals(AttnGateKind.SOFTPLUS_BASE2.ordinal() - 2,
                AttnExtMath.gateKernelKind(AttnGateKind.SOFTPLUS_BASE2));
    }

    @Test
    public void widthsAndBiasRules() {
        assertTrue(AttnExtMath.gatePerHead(AttnGateKind.PER_HEAD_SOFTPLUS));
        assertFalse(AttnExtMath.gatePerHead(AttnGateKind.SOFTPLUS_BASE2));
        assertEquals(N_HEAD, AttnExtMath.gateWidth(AttnGateKind.PER_HEAD_SOFTPLUS, N_HEAD, HEAD_DIM));
        assertEquals(N_HEAD * HEAD_DIM, AttnExtMath.gateWidth(AttnGateKind.SOFTPLUS_BASE2, N_HEAD, HEAD_DIM));
        assertFalse(AttnExtMath.gateRequiresBias(AttnGateKind.PER_HEAD_SOFTPLUS));
        assertFalse(AttnExtMath.gateRequiresBias(AttnGateKind.SOFTPLUS_BASE2));
    }

    @Test
    public void base2SoftplusScalesAroundTheSoftplus() {
        for (float g : new float[]{-30f, -2.5f, 0f, 0.75f, 12f, 28.9f, 40f}) {
            double expected = softplus(g * 0.6931471805599453) * 1.4426950408889634;
            assertEquals("g = " + g, expected,
                    AttnExtMath.gateActivation(AttnExtMath.GATE_SOFTPLUS_BASE2, g), 1e-5 * Math.max(1, expected));
        }
        assertEquals(1.0, AttnExtMath.gateActivation(AttnExtMath.GATE_SOFTPLUS_BASE2, 0f), 1e-6);
    }

    @Test
    public void perHeadSoftplusBroadcastsOneGatePerHead() {
        int n = 2;
        int stride = N_HEAD * HEAD_DIM + 1;
        float[] attn = new float[n * stride];
        for (int i = 0; i < attn.length; i++) {
            attn[i] = 0.1f * (i + 1);
        }
        float[] before = attn.clone();
        int gateStride = N_HEAD + 2;
        float[] gate = new float[n * gateStride];
        for (int i = 0; i < gate.length; i++) {
            gate[i] = 0.4f * i - 1.5f;
        }
        AttnExtMath.applyOutGate(AttnGateKind.PER_HEAD_SOFTPLUS, n, N_HEAD, HEAD_DIM, attn, 0, stride, gate, 0,
                gateStride, null, 0);
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                double s = softplus(gate[t * gateStride + h]);
                for (int d = 0; d < HEAD_DIM; d++) {
                    int i = t * stride + h * HEAD_DIM + d;
                    assertEquals(before[i] * s, attn[i], 1e-5);
                }
            }
            assertEquals(before[t * stride + N_HEAD * HEAD_DIM], attn[t * stride + N_HEAD * HEAD_DIM], 0f);
        }
    }

    @Test
    public void base2GateIsElementwise() {
        int width = N_HEAD * HEAD_DIM;
        float[] attn = new float[width];
        float[] gate = new float[width];
        for (int i = 0; i < width; i++) {
            attn[i] = 1f + i;
            gate[i] = 0.5f * i - 3f;
        }
        float[] before = attn.clone();
        AttnExtMath.applyOutGate(AttnGateKind.SOFTPLUS_BASE2, 1, N_HEAD, HEAD_DIM, attn, 0, width, gate, 0, width,
                null, 0);
        for (int i = 0; i < width; i++) {
            double s = softplus(gate[i] * 0.6931471805599453) * 1.4426950408889634;
            assertEquals(before[i] * s, attn[i], 1e-4);
        }
    }
}
