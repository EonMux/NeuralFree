package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The lfm3a pipeline over the GPU engine: the MaskGIT depthformer and the liquid-codec decoder run on the device
 * while the backbone's hidden states come from the engine, so the GPU pipeline samples the same codes and produces
 * the same waveform as the CPU pipeline over the same synthetic model. Codes are discrete and must match exactly;
 * PCM is compared within {@link #PCM_TOLERANCE}.
 */
public class Lfm3aSpeechGeneratorGpuTest {

    /** Absolute tolerance of the device waveform against the host pipeline's. */
    static final float PCM_TOLERANCE = 1e-4f;

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(256).ubatch(256).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    @Test
    public void gpuGeneratorMatchesTheCpuPipeline() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path generatorFile = Lfm3aClipFixture.deterministicEos().writeTempFile();
        EngineConfig cpu = EngineConfig.builder().memory(memory()).build();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        int[][] wantCodes;
        AudioClip wantAudio;
        try (GGUFModel model = new GGUFModel(backbone.toString(), cpu);
             Lfm3aGenClip clip = Lfm3aGenClip.open(generatorFile.toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            wantCodes = generator.generateCodes("Hello world", SpeechOptions.defaults());
            wantAudio = generator.synthesize("Hello world", SpeechOptions.defaults());
        }
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             Lfm3aGenClip clip = Lfm3aGenClip.open(generatorFile.toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            int[][] codes = generator.generateCodes("Hello world", SpeechOptions.defaults());
            assertEquals("the GPU runs every frame the CPU ran", wantCodes.length, codes.length);
            for (int i = 0; i < wantCodes.length; i++) {
                assertArrayEquals("frame " + i + " of the GPU pipeline", wantCodes[i], codes[i]);
            }
            AudioClip audio = generator.synthesize("Hello world", SpeechOptions.defaults());
            assertEquals(wantAudio.length(), audio.length());
            assertEquals(wantAudio.sampleRate(), audio.sampleRate());
            assertArrayEquals("the GPU pipeline's waveform", wantAudio.samples(), audio.samples(), PCM_TOLERANCE);
        }
    }
}
