package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Reference values of the CPU activation, GLU and attention-extra math (the CPU twins of
 * {@code LLM_Unary}, {@code LLM_GLU}, {@code LLM_AttnTempScale} and {@code LLM_AttnOutGate}).
 */
public class ActMathTest {

    private static final float EPS = 1e-6f;

    @Test
    public void erfMatchesReferenceValues() {
        assertEquals(0.0, ActMath.erf(0.0), 0.0);
        assertEquals(0.1124629160182849, ActMath.erf(0.1), 1e-15);
        assertEquals(0.5204998778130465, ActMath.erf(0.5), 1e-15);
        assertEquals(0.8427007929497149, ActMath.erf(1.0), 1e-15);
        assertEquals(0.9953222650189527, ActMath.erf(2.0), 1e-15);
        assertEquals(0.9999779095030014, ActMath.erf(3.0), 1e-15);
        assertEquals(0.9999999999984626, ActMath.erf(5.0), 1e-15);
        assertEquals(1.0, ActMath.erf(30.0), 0.0);
        assertEquals(-0.8427007929497149, ActMath.erf(-1.0), 1e-15);
    }

    @Test
    public void scalarActivationsFollowGgml() {
        assertEquals(0f, ActMath.relu(-2f), 0f);
        assertEquals(9f, ActMath.reluSqr(3f), 0f);
        assertEquals(0.8411920f, ActMath.geluTanh(1f), EPS);
        assertEquals(0.8413447f, ActMath.geluErf(1f), EPS);
        assertEquals(0.8457958f, ActMath.geluQuick(1f), EPS);
        assertEquals(0.7310586f, ActMath.silu(1f), EPS);
        assertEquals(0.7310586f, ActMath.sigmoid(1f), EPS);
        assertEquals(1.3132616f, ActMath.softplus(1f), EPS);
        assertEquals(25f, ActMath.softplus(25f), 0f);
        assertEquals((float) Math.sqrt(1.3132616), ActMath.sqrtSoftplus(1f), EPS);
        assertEquals(2f, ActMath.clamp(5f, -2f, 2f), 0f);
        assertEquals(-2f, ActMath.clamp(-5f, -2f, 2f), 0f);
        assertEquals(30f * (float) Math.tanh(0.5), ActMath.softcap(15f, 30f, 1f / 30f), 1e-5f);
    }

    @Test
    public void xieluUsesPreparedParameters() {
        float an = 0.8f;
        float ap = 0.3f;
        float beta = 0.5f;
        float eps = -1e-6f;
        assertEquals(ap * 4f + beta * 2f, ActMath.xielu(2f, an, ap, beta, eps), EPS);
        float x = -1f;
        float expected = ((float) Math.expm1(Math.min(x, eps)) - x) * an + beta * x;
        assertEquals(expected, ActMath.xielu(x, an, ap, beta, eps), EPS);
    }

    @Test
    public void everyActOpCodeDispatches() {
        ActParams p = ActParams.builder().xieluRaw(0.8f, 0.3f, 0.5f, -1e-6f).scaleBias(2f, 1f)
                .clamp(-1f, 1f).softcap(10f, 1f).build();
        for (ActOp op : ActOp.values()) {
            float[] a = p.unaryArgs(op);
            float byEnum = ActMath.apply(op, 0.75f, a[0], a[1], a[2], a[3]);
            float byCode = ActMath.apply(op.code(), 0.75f, a[0], a[1], a[2], a[3]);
            assertEquals(op.name(), byEnum, byCode, 0f);
            float[] in = {0.75f};
            float[] out = new float[1];
            ActMath.apply(op, p, in, 0, out, 0, 1);
            assertEquals(op.name(), byEnum, out[0], 0f);
        }
        assertEquals(2.5f, ActMath.apply(ActOp.SCALE_BIAS, p, 0.75f), EPS);
        assertEquals(0.75f, ActMath.apply(ActOp.IDENTITY, p, 0.75f), 0f);
    }

    @Test
    public void arrayFormHonoursOffsetsAndInPlace() {
        float[] x = {9f, -1f, 2f, 9f};
        ActMath.applyInPlace(ActOp.RELU_SQR, ActParams.NONE, x, 1, 2);
        assertArrayEquals(new float[]{9f, 0f, 4f, 9f}, x, 0f);
    }

    @Test
    public void gluOpsFollowLlamaCpp() {
        float g = 2f;
        float u = 3f;
        assertEquals(ActMath.silu(g) * u, GluMath.apply(GluOp.SWIGLU, g, u, Float.POSITIVE_INFINITY, 1f, 1f, 0f), 0f);
        assertEquals(g * u, GluMath.apply(GluOp.MUL, g, u, 0f, 0f, 1f, 0f), 0f);
        assertEquals(ActMath.sigmoid(g) * u, GluMath.apply(GluOp.SIGMUL, g, u, 0f, 0f, 1f, 0f), 0f);
        // SWIGLU_CLAMP clamps the gate input; SWIGLU_LIMITED clamps the activated gate.
        float limit = 1f;
        assertEquals(ActMath.silu(1f) * 1f, GluMath.apply(GluOp.SWIGLU_CLAMP, g, u, limit, 1f, 1f, 0f), EPS);
        assertEquals(Math.min(ActMath.silu(g), limit) * 1f, GluMath.apply(GluOp.SWIGLU_LIMITED, g, u, limit, 1f, 1f, 0f), EPS);
        // gpt-oss: x = min(g, 7), y = clamp(u, -7, 7), x * sigmoid(1.702 x) * (y + 1).
        float oai = 2f / (1f + (float) Math.exp(-1.702f * 2f)) * 4f;
        assertEquals(oai, GluMath.apply(GluOp.SWIGLU_OAI, g, u, 7f, 1.702f, 1f, 0f), EPS);
    }

    @Test
    public void situMatchesKimiK3() {
        float beta = 3f;
        float lambda = 2f;
        float g = 1.5f;
        float u = -0.7f;
        float expected = beta * (float) Math.tanh(g / beta) * ActMath.sigmoid(g)
                * lambda * (float) Math.tanh(u / lambda);
        ActParams p = ActParams.builder().beta(beta).lambda(lambda).build();
        assertEquals(expected, GluMath.apply(GluOp.SITU, p, g, u), EPS);
        ActParams linear = ActParams.builder().beta(beta).build();
        float expectedLinear = beta * (float) Math.tanh(g / beta) * ActMath.sigmoid(g) * u;
        assertEquals(expectedLinear, GluMath.apply(GluOp.SITU, linear, g, u), EPS);
    }

    @Test
    public void gluKernelScalarsMapSituParameters() {
        ActParams p = ActParams.builder().limit(5f).alpha(1.702f).beta(3f).lambda(2f).build();
        assertArrayEquals(new float[]{5f, 3f, 2f}, ActKernels.gluScalars(GluOp.SITU, p), 0f);
        assertArrayEquals(new float[]{5f, 1.702f, 2f}, ActKernels.gluScalars(GluOp.SWIGLU_OAI, p), 0f);
    }

    @Test
    public void gluRowsUseStridesAndFusedLayout() {
        // Fused gate+up rows of width 2 * nFf: [g0 g1 | u0 u1] per row.
        int nFf = 2;
        float[] fused = {1f, 2f, 3f, 4f, -1f, 0.5f, 2f, -2f};
        float[] out = new float[4];
        GluMath.applyRows(GluOp.MUL, ActParams.NONE, 2, nFf, fused, 0, 2 * nFf, fused, nFf, 2 * nFf, out, 0, nFf);
        assertArrayEquals(new float[]{3f, 8f, -2f, -1f}, out, 0f);
    }

    @Test
    public void codesMatchKernelDefines() {
        assertEquals(15, ActOp.SOFTCAP.code());
        assertEquals(13, ActOp.SQRT_SOFTPLUS.code());
        assertEquals(8, GluOp.SITU.code());
        assertEquals(10, GluOp.SIGMUL.code());
        assertEquals(10, ActKernels.UNARY_ARG_COUNT);
        assertEquals(15, ActKernels.GLU_ARG_COUNT);
    }

    @Test
    public void temperatureLogAndPowForms() {
        // llama4: scale 0.1, floor 8192, offset 1, log form.
        assertEquals(1f, AttnExtMath.temperatureFactor(false, 8192, 0.1f, 1f, 0), 0f);
        assertEquals(1f, AttnExtMath.temperatureFactor(false, 8192, 0.1f, 1f, 8190), 0f);
        assertEquals((float) (Math.log(2.0) * 0.1f + 1.0), AttnExtMath.temperatureFactor(false, 8192, 0.1f, 1f, 8191), 0f);
        // beam: pow form, scale 0.1, length 2048, offset 1.
        float pow = (float) Math.exp(0.1f * Math.log1p(101.0 / 2048.0));
        assertEquals(pow, AttnExtMath.temperatureFactor(true, 2048, 0.1f, 1f, 100), 0f);
        AttentionSpec.Temperature t = AttentionSpec.Temperature.of(0.1f, 8192, 1f, false);
        assertEquals(t.at(20000), AttnExtMath.temperatureFactor(t, 20000), 0f);
        assertEquals(1f, AttnExtMath.temperatureFactor(AttentionSpec.Temperature.NONE, 20000), 0f);

        float[] q = {1f, 1f, 7f, 1f, 1f};
        AttnExtMath.applyTemperature(t, 2, 1, 2, new int[]{0, 8191}, 0, q, 0, 3);
        float f = AttnExtMath.temperatureFactor(t, 8191);
        assertArrayEquals(new float[]{1f, 1f, 7f, f, f}, q, 0f);
    }

    @Test
    public void outputGateKinds() {
        assertEquals(0, AttnExtMath.gateKernelKind(AttnGateKind.Q_INTERLEAVED));
        assertEquals(AttnGateKind.ELEMENTWISE_SIGMOID.ordinal() - 2, AttnExtMath.gateKernelKind(AttnGateKind.ELEMENTWISE_SIGMOID));
        assertEquals(AttnGateKind.ELEMENTWISE_SOFTPLUS.ordinal() - 2, AttnExtMath.gateKernelKind(AttnGateKind.ELEMENTWISE_SOFTPLUS));
        assertEquals(AttnGateKind.PER_HEAD_SIGMOID.ordinal() - 2, AttnExtMath.gateKernelKind(AttnGateKind.PER_HEAD_SIGMOID));
        assertEquals(AttnGateKind.PER_HEAD_SIGMOID_BIAS.ordinal() - 2, AttnExtMath.gateKernelKind(AttnGateKind.PER_HEAD_SIGMOID_BIAS));
        assertTrue(AttnExtMath.gatePerHead(AttnGateKind.PER_HEAD_SIGMOID));
        assertFalse(AttnExtMath.gatePerHead(AttnGateKind.ELEMENTWISE_SOFTPLUS));
        assertEquals(4, AttnExtMath.gateWidth(AttnGateKind.PER_HEAD_SIGMOID, 4, 8));
        assertEquals(32, AttnExtMath.gateWidth(AttnGateKind.ELEMENTWISE_SIGMOID, 4, 8));

        // Two heads of dim 2, one token; per-head sigmoid with bias.
        float[] attn = {1f, 2f, 3f, 4f};
        float[] gate = {0f, 1f};
        float[] bias = {1f, -1f};
        AttnExtMath.applyOutGate(AttnGateKind.PER_HEAD_SIGMOID_BIAS, 1, 2, 2, attn, 0, 4, gate, 0, 2, bias, 0);
        float s0 = ActMath.sigmoid(1f);
        float s1 = ActMath.sigmoid(0f);
        assertArrayEquals(new float[]{s0, 2f * s0, 3f * s1, 4f * s1}, attn, EPS);

        // Element-wise softplus without bias.
        float[] attn2 = {1f, 1f};
        float[] gate2 = {0f, 30f};
        AttnExtMath.applyOutGate(AttnGateKind.ELEMENTWISE_SOFTPLUS, 1, 1, 2, attn2, 0, 2, gate2, 0, 2, null, 0);
        assertArrayEquals(new float[]{ActMath.softplus(0f), 30f}, attn2, EPS);
    }

    @Test(expected = IllegalArgumentException.class)
    public void noneGateHasNoKernel() {
        AttnExtMath.gateKernelKind(AttnGateKind.NONE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rangeOutsideArrayIsRejected() {
        ActMath.apply(ActOp.SILU, ActParams.NONE, new float[2], 1, new float[2], 0, 2);
    }
}
