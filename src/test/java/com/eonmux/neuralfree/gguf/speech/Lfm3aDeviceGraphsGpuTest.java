package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lfm3a generator graphs over the engine's device: the MaskGIT depthformer's per-pass logits and feedback row,
 * and the liquid-codec decoder's spectra and PCM over their persisted position, convolution taps and attention
 * windows, each compared with the host graphs on the synthetic generator fixture. Codes are sampled on the host,
 * so the depthformer's logits are compared within {@link #TOLERANCE}, as are the spectra and waveforms.
 */
public class Lfm3aDeviceGraphsGpuTest {

    /** Absolute tolerance of the device graphs' float outputs against the host graphs'. */
    static final float TOLERANCE = 1e-4f;

    private static final int CB = Lfm3aClipFixture.CODEBOOKS;
    private static final int VB = Lfm3aClipFixture.VOCAB;

    private static final int[] BATCH_A = {0, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 0, 2, 1, 0, 2, 1, 1, 1, 0};
    private static final int[] BATCH_B = {1, 1, 0, 2, 2, 1, 1, 0, 0, 0, 1, 1};

    private GGUFModelGPU model;
    private Lfm3aGenClip clip;
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
                .addGpuDevice(selector(backends))
                .memory(MemoryConfig.builder().nCtx(256).ubatch(256).typeK(KvDType.F32).typeV(KvDType.F32)
                        .build())
                .build();
        model = new GGUFModelGPU(Lfm3aBackboneFixture.ttsBackbone().toString(), null, gpu);
        clip = Lfm3aGenClip.open(Lfm3aClipFixture.randomWeights(41L).writeTempFile().toString());
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

    @Test
    public void depthformerPassMatchesTheHostGraph() throws IOException {
        openDevice();
        assertDepthformerMatchesHost();
    }

    @Test
    public void codecWindowsMatchTheHostSession() throws IOException {
        openDevice();
        assertCodecMatchesHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnCuda() throws IOException {
        openDevice(EngineConfig.BackendMode.CUDA);
        assertDepthformerMatchesHost();
        assertCodecMatchesHost();
    }

    @Test
    public void deviceGraphsMatchTheHostOnVulkan() throws IOException {
        openDevice(EngineConfig.BackendMode.VULKAN);
        assertDepthformerMatchesHost();
        assertCodecMatchesHost();
    }

    private void assertDepthformerMatchesHost() {
        Lfm3aDepthformer host = clip.depthformer();
        try (Lfm3aDepthformer.Engine dev = host.newEngine(device)) {
            Random hiddenDraw = new Random(23L);
            Random codeDraw = new Random(29L);
            for (int f = 0; f < 3; f++) {
                float[] hidden = new float[Lfm3aClipFixture.TALKER];
                for (int i = 0; i < hidden.length; i++) {
                    hidden[i] = hiddenDraw.nextFloat() * 2f - 1f;
                }
                int[] codes = new int[CB];
                for (int c = 0; c < CB; c++) {
                    codes[c] = codeDraw.nextInt(4) == 0 ? -1 : codeDraw.nextInt(VB);
                }
                Lfm3aDepthformer.Frame want = host.pass(hidden, codes);
                Lfm3aDepthformer.Frame got = dev.pass(hidden, codes);
                assertClose("frame " + f + " feedback", want.feedback(), got.feedback());
                assertEquals("frame " + f + " logits presence", want.logits() == null, got.logits() == null);
                if (want.logits() != null) {
                    assertClose("frame " + f + " logits", want.logits(), got.logits());
                }
            }
            int[] known = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
            Lfm3aDepthformer.Frame want = host.pass(new float[Lfm3aClipFixture.TALKER], known);
            Lfm3aDepthformer.Frame got = dev.pass(new float[Lfm3aClipFixture.TALKER], known);
            assertNull("an all-known pass runs no transformer", got.logits());
            assertClose("all-known feedback", want.feedback(), got.feedback());
        }
    }

    private void assertCodecMatchesHost() {
        Lfm3aCodec hostGeo = clip.codec();
        Lfm3aCodecSession host = hostGeo.newSession();
        try (Lfm3aCodec.Decoder dev = hostGeo.newDecoder(device)) {
            float[] wantA = host.decodePolar(BATCH_A);
            float[] gotA = dev.decodePolar(BATCH_A);
            assertArrayEquals("first batch spectra", wantA, gotA, TOLERANCE);
            assertEquals("first batch position", host.position(), dev.position());
            float[] wantB = host.decodePolar(BATCH_B);
            float[] gotB = dev.decodePolar(BATCH_B);
            assertArrayEquals("second batch spectra", wantB, gotB, TOLERANCE);
            assertEquals("second batch position", host.position(), dev.position());
        }
        try (Lfm3aCodecSession hostPcm = hostGeo.newSession();
             Lfm3aCodec.Decoder devPcm = hostGeo.newDecoder(device)) {
            AudioClip want = hostPcm.decodeClip(BATCH_A);
            AudioClip got = devPcm.decodeClip(BATCH_A);
            assertEquals("decoded sample rate", want.sampleRate(), got.sampleRate());
            assertArrayEquals("decoded waveform", want.samples(), got.samples(), TOLERANCE);
            assertArrayEquals("flushed tail", hostPcm.flushClip().samples(), devPcm.flushClip().samples(),
                    TOLERANCE);
        }
    }

    @Test
    public void codecRefusesBadBatches() throws IOException {
        openDevice();
        try (Lfm3aCodec.Decoder dev = clip.codec().newDecoder(device)) {
            assertThrows(IllegalArgumentException.class, () -> dev.decodePolar(new int[0]));
            assertThrows(IllegalArgumentException.class, () -> dev.decodePolar(new int[CB + 1]));
            int[] negative = new int[CB];
            negative[0] = -1;
            assertThrows(IllegalArgumentException.class, () -> dev.decodePolar(negative));
            int[] beyond = new int[CB];
            beyond[0] = Lfm3aClipFixture.VOCAB;
            assertThrows(IllegalArgumentException.class, () -> dev.decodePolar(beyond));
            dev.close();
            assertThrows(IllegalStateException.class, () -> dev.decodePolar(BATCH_B));
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
