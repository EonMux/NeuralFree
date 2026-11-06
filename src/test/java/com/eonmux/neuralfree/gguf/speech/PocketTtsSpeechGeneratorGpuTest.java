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
import static org.junit.Assert.assertTrue;

/**
 * The pockettts pipeline over the GPU engine: the flow network and the mimi decoder run on the device while the
 * backbone serves its embedding rows and output norm through the engine, so the GPU pipeline produces the same
 * latents and waveform as the CPU pipeline over the same synthetic model. Latents and PCM are compared within
 * {@link #PCM_TOLERANCE}, the accumulated rounding difference of the F32 graphs.
 */
public class PocketTtsSpeechGeneratorGpuTest {

    /** Absolute tolerance of the device latents and waveform against the host pipeline's. */
    static final float PCM_TOLERANCE = 1e-4f;

    private static SpeechOptions twoFrames() {
        return SpeechOptions.defaults().withMaxFrames(2);
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(128).ubatch(128).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static Path fixture() throws IOException {
        return new PocketTtsFixtures(61L).writeWithEosBias(-100f);
    }

    @Test
    public void gpuGeneratorMatchesTheCpuPipeline() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = PocketTtsBackboneFixture.write();
        Path generatorFile = fixture();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        float[][] wantLatents;
        AudioClip wantAudio;
        try (GGUFModel model = new GGUFModel(backbone.toString(), EngineConfig.builder().memory(memory()).build());
             PocketTtsClip clip = PocketTtsClip.open(generatorFile.toString());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            wantLatents = generator.generateLatents("Hi", twoFrames());
            wantAudio = generator.synthesize("Hi", twoFrames());
        }
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             PocketTtsClip clip = PocketTtsClip.open(generatorFile.toString());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            float[][] latents = generator.generateLatents("Hi", twoFrames());
            assertEquals("the GPU runs every frame the CPU ran", wantLatents.length, latents.length);
            for (int f = 0; f < wantLatents.length; f++) {
                assertArrayEquals("frame " + f + " latent", wantLatents[f], latents[f], PCM_TOLERANCE);
            }
            AudioClip audio = generator.synthesize("Hi", twoFrames());
            assertEquals(wantAudio.length(), audio.length());
            assertEquals(wantAudio.sampleRate(), audio.sampleRate());
            assertArrayEquals("the GPU pipeline's waveform", wantAudio.samples(), audio.samples(), PCM_TOLERANCE);
        }
    }

    @Test
    public void streamedChunksConcatenateToTheWholeDecode() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = PocketTtsBackboneFixture.write();
        Path generatorFile = fixture();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             PocketTtsClip clip = PocketTtsClip.open(generatorFile.toString());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            AudioClip whole = generator.synthesize("Hi", twoFrames());
            java.util.List<Float> streamed = new java.util.ArrayList<>();
            generator.synthesize("Hi", twoFrames().withChunkFrames(1), chunkAudio -> {
                for (float sample : chunkAudio.samples()) {
                    streamed.add(sample);
                }
            });
            assertEquals("the sink receives every decoded sample once", whole.length(), streamed.size());
            for (int i = 0; i < whole.length(); i++) {
                assertEquals("streamed sample " + i, whole.samples()[i], streamed.get(i), 0f);
            }
            assertTrue(whole.length() > 0);
        }
    }
}
