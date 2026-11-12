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
 * The qwen3tts generator graphs over the OpenCL engine: the code predictor's per-frame steps and feedback rows and
 * the code2wav decoder's windows over their persisted key/value prefixes and convolution state, each compared with
 * the host graphs on the synthetic generator fixture. Codes are discrete and must match exactly; feedback rows and
 * waveforms are compared within {@link #TOLERANCE}, the accumulated rounding difference of the F32 graphs.
 */
public class Qwen3TtsDeviceGraphsGpuTest {

    /** Absolute tolerance of the device graphs' float outputs against the host graphs'. */
    static final float TOLERANCE = 1e-4f;

    private GGUFModelGPU model;
    private Qwen3TtsClip clip;
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
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(64).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        EngineConfig gpu = EngineConfig.builder()
                .backend(backends)
                .addGpuDevice(selector(backends))
                .memory(memory)
                .build();
        model = new GGUFModelGPU(Qwen3TtsBackboneFixture.write().toString(), null, gpu);
        clip = Qwen3TtsClip.open(new Qwen3TtsClipFixture(71L, Qwen3TtsBackboneFixture.EMB).write().toString());
        device = clip.bindDevice(model.forwardContext());
    }

    /**
     * Device 0 of OpenCL and CUDA; on Vulkan the fastest device by name ({@link GpuTestUtil#getVulkanDeviceName}),
     * since the Vulkan enumeration order puts an integrated GPU first on some machines.
     */
    private static String selector(EngineConfig.BackendMode backends) {
        return backends == EngineConfig.BackendMode.VULKAN ? GpuTestUtil.getVulkanDeviceName() : "0";
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

    private static int[][] codes(int frames) {
        int[][] out = new int[frames][];
        for (int f = 0; f < frames; f++) {
            int[] row = new int[Qwen3TtsClipFixture.PAGES + 1];
            row[0] = (f * 3 + 1) % Qwen3TtsClipFixture.VOCAB0;
            for (int g = 1; g < row.length; g++) {
                row[g] = (f * 5 + g * 7 + 2) % Qwen3TtsClipFixture.VOCAB;
            }
            out[f] = row;
        }
        return out;
    }

    @Test
    public void codePredictorMatchesTheHostSession() throws IOException {
        openDevice();
        assertCodePredictorMatchesHost();
    }

    @Test
    public void decoderWindowsMatchTheHostSession() throws IOException {
        openDevice();
        assertDecoderWindowsMatchHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnCuda() throws IOException {
        openDevice(EngineConfig.BackendMode.CUDA);
        assertCodePredictorMatchesHost();
        assertDecoderWindowsMatchHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnVulkan() throws IOException {
        openDevice(EngineConfig.BackendMode.VULKAN);
        assertCodePredictorMatchesHost();
        assertDecoderWindowsMatchHost();
    }

    private void assertCodePredictorMatchesHost() {
        int frames = 4;
        Qwen3TtsCodePredictor.Engine host = clip.predictor().newSession(1, 1f);
        try (Qwen3TtsCodePredictor.Engine devicePredictor = new GpuQwen3TtsCodePredictor(clip.predictor(),
                device, 1, 1f)) {
            Random hostDraws = new Random(11L);
            Random deviceDraws = new Random(11L);
            Random hiddenDraw = new Random(29L);
            int talkerWidth = clip.predictor().talkerWidth();
            for (int f = 0; f < frames; f++) {
                float[] hidden = new float[talkerWidth];
                for (int i = 0; i < hidden.length; i++) {
                    hidden[i] = hiddenDraw.nextFloat() * 2f - 1f;
                }
                int code0 = (f * 3 + 1) % Qwen3TtsClipFixture.VOCAB0;
                Qwen3TtsCodePredictor.Frame want = host.process(hidden, code0, hostDraws);
                Qwen3TtsCodePredictor.Frame got = devicePredictor.process(hidden, code0, deviceDraws);
                assertArrayEquals("frame " + f + " codes", want.codes(), got.codes());
                assertClose("frame " + f + " feedback", want.feedback(), got.feedback());
            }
        }
    }

    private void assertDecoderWindowsMatchHost() {
        int[][] all = codes(7);
        int window = clip.code2Wav().window();
        Qwen3TtsCode2WavSession host = clip.code2Wav().newSession();
        try (GpuQwen3TtsCode2Wav deviceDecoder = new GpuQwen3TtsCode2Wav(clip.code2Wav(), device)) {
            int at = 0;
            int call = 0;
            while (at < all.length) {
                int n = Math.min(window, all.length - at);
                int[][] batch = new int[n][];
                System.arraycopy(all, at, batch, 0, n);
                float[] want = host.decode(batch);
                float[] got = deviceDecoder.decode(batch);
                assertArrayEquals("call " + call + " sample count", want, got, TOLERANCE);
                assertEquals("call " + call + " position", host.position(), deviceDecoder.position(), 0f);
                at += n;
                call++;
            }
            assertTrue("the sequence spans more than one window", call > 1);
        }
    }

    @Test
    public void decoderRefusesBadBatchesAndCodes() throws IOException {
        openDevice();
        try (GpuQwen3TtsCode2Wav deviceDecoder = new GpuQwen3TtsCode2Wav(clip.code2Wav(), device)) {
            int window = clip.code2Wav().window();
            assertThrows(IllegalArgumentException.class, () -> deviceDecoder.decode(new int[0][]));
            assertThrows(IllegalArgumentException.class, () -> deviceDecoder.decode(codes(window + 1)));
            assertThrows(IllegalArgumentException.class, () -> deviceDecoder.decode(new int[][]{{0, 0}}));
            assertThrows(IllegalArgumentException.class, () -> deviceDecoder.decode(new int[][]{{0, 0, 0, 0}}));
            assertThrows(IllegalArgumentException.class, () -> deviceDecoder.decode(new int[][]{{-1, 0, 0}}));
            assertThrows(IllegalArgumentException.class,
                    () -> deviceDecoder.decode(new int[][]{{0, Qwen3TtsClipFixture.VOCAB, 0}}));
            deviceDecoder.close();
            assertThrows(IllegalStateException.class, () -> deviceDecoder.decode(codes(1)));
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
