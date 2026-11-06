package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Assume;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Shared fixtures of the layer-split combination tests on the OpenCL devices of the machine: the one-device and
 * two-device configurations, prefill and decode logits, and greedy generations to compare a split with the one-device
 * load of the same settings.
 */
final class SplitFixtures {

    /** Six tokens: two ubatches of four. */
    static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    static final int[] DECODE = {77, 3};
    static final float TOL = 1e-3f;
    static final int GREEDY_TOKENS = 8;

    private SplitFixtures() {
    }

    /** Skips the test unless OpenCL exposes at least two devices. */
    static void assumeTwoDevices() {
        GpuTestUtil.assumeOpenCLAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        int devices;
        try {
            devices = hw.getCLDevices().size();
        } finally {
            hw.shutdown();
        }
        Assume.assumeTrue("the split needs two OpenCL devices, found " + devices, devices >= 2);
    }

    /** The one-device load on OpenCL device {@code device}, 64 positions in ubatches of four. */
    static EngineConfig.Builder oneDevice(String device) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice(device)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build());
    }

    /** The one-device load on device 0. */
    static EngineConfig.Builder oneDevice() {
        return oneDevice("0");
    }

    /** The layer split over devices 0 (main) and 1. */
    static EngineConfig.Builder split() {
        return oneDevice("0").splitMode(EngineConfig.SplitMode.LAYER).addGpuDevice("1");
    }

    /** Prefill logits of {@link #PROMPT} and the decode logits of {@link #DECODE}. */
    static float[][] logits(Path model, EngineConfig cfg) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            return logits(m);
        }
    }

    /** {@link #logits(Path, EngineConfig)} of an open model, which is reset afterwards. */
    static float[][] logits(GGUFModelGPU m) {
        ForwardBackend f = m.forwardBackend();
        float[][] out = new float[1 + DECODE.length][];
        out[0] = f.prefill(PROMPT, 0).clone();
        for (int i = 0; i < DECODE.length; i++) {
            out[1 + i] = f.decode(DECODE[i], PROMPT.length + i).clone();
        }
        f.reset();
        return out;
    }

    /** The tokens of a greedy generation of {@link #GREEDY_TOKENS} after {@link #PROMPT}. */
    static int[] greedy(GGUFModelGPU m) {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(m.engineConfig().legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
        int[] tokens = m.generate(PROMPT, request, null).tokens();
        m.forwardBackend().reset();
        return tokens;
    }

    /** The logits and the greedy tokens of {@code cfg}. */
    static Run run(Path model, EngineConfig cfg) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            float[][] l = logits(m);
            return new Run(l, greedy(m), m.forwardBackend().toString());
        }
    }

    /** Every step of {@code got} matches {@code reference} within {@link #TOL}, and the greedy tokens agree. */
    static void assertParity(String what, Run reference, Run got) {
        assertEquals(what + ": steps", reference.logits.length, got.logits.length);
        for (int i = 0; i < reference.logits.length; i++) {
            assertArrayEquals(what + ": step " + i + " matches the one-device load", reference.logits[i],
                    got.logits[i], TOL);
        }
        assertArrayEquals(what + ": greedy tokens", reference.greedy, got.greedy);
    }

    static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    /** The logits, greedy tokens and forward-backend description of one load. */
    static final class Run {
        final float[][] logits;
        final int[] greedy;
        final String backend;

        Run(float[][] logits, int[] greedy, String backend) {
            this.logits = logits;
            this.greedy = greedy;
            this.backend = backend;
        }
    }
}
