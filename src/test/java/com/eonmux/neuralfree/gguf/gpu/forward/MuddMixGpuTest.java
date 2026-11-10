package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.cpu.forward.MuddMath;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.testutil.LlmApiBackends;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@value BuiltInGpuExtras#MUDD_MIX} on every device of each API (CUDA; OpenCL, the NVIDIA and the Intel driver;
 * Vulkan) against the host reference {@link MuddMath#mixRow} over the biased weights: fractional, negative and
 * large banked values mixed over repeated and out-of-order taps. Every view (bank, weights, taps, both outputs)
 * starts at its own non-zero offset, and the elements around the outputs stay untouched. The banked values must
 * enter the products as floats (an integer conversion truncates every activation).
 */
@RunWith(Parameterized.class)
public class MuddMixGpuTest {

    private static final int N_LAYERS = 4;
    private static final int UBATCH = 5;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> apis() {
        return Arrays.asList(new Object[]{BackendType.CUDA}, new Object[]{BackendType.OPENCL},
                new Object[]{BackendType.VULKAN});
    }

    private final BackendType api;
    private LlmApiBackends devices;

    public MuddMixGpuTest(BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        devices = LlmApiBackends.open(api);
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void fractionalBankValuesMixAsFloats() {
        check(3, 37, new int[]{0, 2, 1}, 1f, 11L);
    }

    @Test
    public void largeAndNegativeValuesOverRepeatedTaps() {
        check(UBATCH, 64, new int[]{3, 3, 0, 1}, 1000f, 12L);
    }

    @Test
    public void aSingleTokenOfOneTap() {
        check(1, 129, new int[]{2}, 0.01f, 13L);
    }

    /** Mixes {@code nTokens} rows of width {@code nEmbd} on every device and compares with the host reference. */
    private void check(int nTokens, int nEmbd, int[] taps, float scale, long seed) {
        Random rnd = new Random(seed);
        int bankOff = 7;
        int wOff = 5;
        int tapsOff = 3;
        int curOff = 11;
        int residOff = 13;
        int pad = 9;
        int bankStride = UBATCH * nEmbd;
        float[] bank = new float[bankOff + N_LAYERS * bankStride];
        boolean fractional = false;
        for (int i = 0; i < bank.length; i++) {
            bank[i] = (rnd.nextFloat() * 2f - 1f) * scale;
            fractional |= bank[i] != (float) (long) bank[i];
        }
        assertTrue("the bank must hold non-integer values", fractional);
        int nTaps = taps.length;
        float[] w = new float[wOff + nTokens * 2 * nTaps + pad];
        Arrays.fill(w, Float.NaN);
        float[] bias = new float[2 * nTaps];
        float[] biased = new float[nTokens * 2 * nTaps];
        for (int j = 0; j < bias.length; j++) {
            bias[j] = (rnd.nextFloat() - 0.5f) * 0.5f;
        }
        for (int i = 0; i < biased.length; i++) {
            w[wOff + i] = (rnd.nextFloat() * 2f - 1f) * 0.75f;
            biased[i] = w[wOff + i] + bias[i % (2 * nTaps)];
        }
        int[] tapBuffer = new int[tapsOff + nTaps + pad];
        Arrays.fill(tapBuffer, -1);
        System.arraycopy(taps, 0, tapBuffer, tapsOff, nTaps);
        float[] wantCur = new float[nTokens * nEmbd];
        float[] wantResid = new float[nTokens * nEmbd];
        for (int t = 0; t < nTokens; t++) {
            MuddMath.mixRow(bank, bankOff, bankStride, taps, t, nEmbd, biased, t * 2 * nTaps, wantCur, wantResid,
                    t * nEmbd);
        }
        float[] curInit = sentinels(curOff + nTokens * nEmbd + pad, -3f);
        float[] residInit = sentinels(residOff + nTokens * nEmbd + pad, -5f);
        for (ComputeBackend backend : devices.backends()) {
            String at = api + " " + backend.capabilities().getDeviceName();
            LlmKernels kernels = LlmKernels.open(backend);
            Assume.assumeTrue("no " + BuiltInGpuExtras.MUDD_MIX + " on " + at, kernels.has(BuiltInGpuExtras.MUDD_MIX));
            GpuBuffer dBank = devices.upload(backend, bank);
            GpuBuffer dW = devices.upload(backend, w);
            GpuBuffer dBias = devices.upload(backend, bias);
            GpuBuffer dTaps = devices.uploadInts(backend, tapBuffer);
            GpuBuffer dCur = devices.upload(backend, curInit);
            GpuBuffer dResid = devices.upload(backend, residInit);
            int local = kernels.rowReduceLocal();
            long global = ((long) nTokens * nEmbd + local - 1) / local * local;
            GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(BuiltInGpuExtras.MUDD_MIX),
                    new long[]{global}, new long[]{local},
                    new Object[]{nTokens, nEmbd, nTaps, bankOff, bankStride, wOff, tapsOff, curOff, residOff, dBank,
                            dW, dBias, dTaps, dCur, dResid});
            if (e != null) {
                e.close();
            }
            float[] gotCur = LlmApiBackends.read(backend, dCur, curInit.length);
            float[] gotResid = LlmApiBackends.read(backend, dResid, residInit.length);
            for (int i = 0; i < wantCur.length; i++) {
                assertEquals(at + " cur " + i, wantCur[i], gotCur[curOff + i], tolerance(scale, nTaps));
                assertEquals(at + " resid " + i, wantResid[i], gotResid[residOff + i], tolerance(scale, nTaps));
            }
            assertUntouched(at + " cur", curInit, gotCur, curOff, wantCur.length);
            assertUntouched(at + " resid", residInit, gotResid, residOff, wantResid.length);
        }
    }

    private static float[] sentinels(int n, float base) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = base - i;
        }
        return out;
    }

    /** The elements outside {@code [off, off + n)} keep their initial values. */
    private static void assertUntouched(String at, float[] init, float[] got, int off, int n) {
        for (int i = 0; i < init.length; i++) {
            if (i < off || i >= off + n) {
                assertEquals(at + " element " + i + " outside the output view", init[i], got[i], 0f);
            }
        }
    }

    /** The bound on a sum of {@code nTaps} products of magnitude at most {@code scale}, contracted or not. */
    private static double tolerance(float scale, int nTaps) {
        return 4.0 * nTaps * scale * Math.ulp(1f);
    }
}
