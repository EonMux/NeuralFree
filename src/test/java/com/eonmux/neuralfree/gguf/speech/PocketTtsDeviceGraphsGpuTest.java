package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The pockettts generator graphs over the engine's device: the flow network's velocity field, latent projection and
 * end-of-speech probe, and the mimi decoder's windows over their persisted attention prefix and convolution state,
 * each compared with the host graphs on the synthetic generator fixture. Gaussian draws come from the same seed on
 * both paths, so latents are compared within {@link #TOLERANCE}; waveforms likewise.
 */
public class PocketTtsDeviceGraphsGpuTest {

    /** Absolute tolerance of the device graphs' float outputs against the host graphs'. */
    static final float TOLERANCE = 1e-4f;

    private GGUFModelGPU model;
    private PocketTtsClip clip;
    private SpeechDeviceWeights device;

    private void openDevice() throws IOException {
        openDevice(EngineConfig.BackendMode.OPENCL);
    }

    private void openDevice(EngineConfig.BackendMode backends) throws IOException {
        if (backends == EngineConfig.BackendMode.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
        } else if (backends == EngineConfig.BackendMode.VULKAN) {
            GpuTestUtil.assumeVulkanAvailable();
        } else {
            GpuTestUtil.assumeOpenCLAvailable();
        }
        EngineConfig gpu = EngineConfig.builder()
                .backend(backends)
                .addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(128).ubatch(128).typeK(KvDType.F32).typeV(KvDType.F32)
                        .build())
                .build();
        model = new GGUFModelGPU(PocketTtsBackboneFixture.write().toString(), null, gpu);
        clip = PocketTtsClip.open(new PocketTtsFixtures(31L).write().toString());
        device = clip.bindDevice(model.forwardContext());
    }

    @After
    public void closeDevice() throws IOException {
        if (clip != null) {
            clip.close();
        }
        if (model != null) {
            model.close();
        }
    }

    private static float[][] frames(int n, long seed) {
        Random rng = new Random(seed);
        float[][] out = new float[n][PocketTtsFixtures.LATENT];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < out[t].length; i++) {
                out[t][i] = (rng.nextFloat() * 2f - 1f) * 0.8f;
            }
        }
        return out;
    }

    @Test
    public void flowMatchesTheHostGraph() throws IOException {
        openDevice();
        assertFlowMatchesHost();
    }

    @Test
    public void decoderWindowsMatchTheHostSession() throws IOException {
        openDevice();
        assertDecoderMatchesHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnCuda() throws IOException {
        openDevice(EngineConfig.BackendMode.CUDA);
        assertFlowMatchesHost();
        assertDecoderMatchesHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnVulkan() throws IOException {
        openDevice(EngineConfig.BackendMode.VULKAN);
        assertFlowMatchesHost();
        assertDecoderMatchesHost();
    }

    private void assertFlowMatchesHost() {
        PocketTtsFlow host = clip.flow();
        try (PocketTtsFlow.Engine deviceFlow = host.newEngine(device)) {
            Random hostDraws = new Random(17L);
            Random deviceDraws = new Random(17L);
            Random condDraw = new Random(43L);
            for (int f = 0; f < 4; f++) {
                float[] cond = new float[PocketTtsFixtures.EMB];
                for (int i = 0; i < cond.length; i++) {
                    cond[i] = condDraw.nextFloat() * 2f - 1f;
                }
                float[] want = host.decode(cond, 0.7f, hostDraws);
                float[] got = deviceFlow.decode(cond, 0.7f, deviceDraws);
                assertClose("frame " + f + " latent", want, got);
                assertClose("frame " + f + " projection", host.project(want), deviceFlow.project(want));
                assertEquals("frame " + f + " end-of-speech", host.isEndOfSpeech(cond),
                        deviceFlow.isEndOfSpeech(cond));
            }
        }
    }

    private void assertDecoderMatchesHost() {
        float[][] all = frames(8, 53L);
        PocketTtsMimiDecoder hostGeo = clip.decoder();
        PocketTtsMimiDecoder.Decoder host = hostGeo.newSession();
        try (PocketTtsMimiDecoder.Decoder dev = hostGeo.newDecoder(device)) {
            for (int at = 0; at < all.length; at += 4) {
                float[][] window = new float[Math.min(4, all.length - at)][];
                System.arraycopy(all, at, window, 0, window.length);
                float[] want = host.decode(window);
                float[] got = dev.decode(window);
                assertArrayEquals("window at " + at + " sample count", want, got, TOLERANCE);
                assertEquals("window at " + at + " position", host.position(), dev.position());
            }
            assertTrue("the sequence spans more than one window", all.length > 4);
        }
    }

    @Test
    public void decoderRefusesBadBatches() throws IOException {
        openDevice();
        try (PocketTtsMimiDecoder.Decoder dev = clip.decoder().newDecoder(device)) {
            assertThrows(IllegalArgumentException.class, () -> dev.decode(new float[0][]));
            assertThrows(IllegalArgumentException.class,
                    () -> dev.decode(frames(PocketTtsMimiDecoder.WINDOW_FRAMES + 1, 1L)));
            assertThrows(IllegalArgumentException.class,
                    () -> dev.decode(new float[][]{new float[PocketTtsFixtures.LATENT - 1]}));
            float[][] nonFinite = frames(1, 2L);
            nonFinite[0][0] = Float.NaN;
            assertThrows(IllegalArgumentException.class, () -> dev.decode(nonFinite));
            dev.close();
            assertThrows(IllegalStateException.class, () -> dev.decode(frames(1, 3L)));
        }
    }

    private static void assertClose(String what, float[] want, float[] got) {
        assertEquals(what + " length", want.length, got.length);
        float max = 0f;
        for (int i = 0; i < want.length; i++) {
            max = Math.max(max, Math.abs(want[i] - got[i]));
        }
        assertTrue(what + " differs by " + max + " from the host graph, above " + TOLERANCE, max <= TOLERANCE);
    }
}
