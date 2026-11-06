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
 * The qwen3tts pipeline over the GPU engine: the generator reads the backbone's token embedding rows and output norm
 * through the engine's own accessors, so a backbone whose forward runs on the device produces the same codes and PCM
 * as the CPU engine over the same synthetic model. Codes are discrete and must match exactly; the waveform is
 * compared within {@link #PCM_TOLERANCE}, the accumulated rounding difference of the graph's F32 arithmetic.
 */
public class Qwen3TtsSpeechGeneratorGpuTest {

    /** Absolute waveform tolerance of the device decode against the host decode. */
    static final float PCM_TOLERANCE = 1e-4f;

    private static SpeechOptions twoFrames() {
        return SpeechOptions.defaults().withTopK(1).withTopP(1f).withMaxFrames(2);
    }

    /** A short reference waveform at the speaker encoder's rate. */
    private static AudioClip reference() {
        float[] samples = new float[4096];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
        }
        return new AudioClip(samples, 24000);
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(64).ubatch(64).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    @Test
    public void streamedChunksConcatenateToTheWholeDecode() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = Qwen3TtsBackboneFixture.write();
        Path generatorFile = new Qwen3TtsClipFixture(71L, Qwen3TtsBackboneFixture.EMB).write();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             Qwen3TtsClip clip = Qwen3TtsClip.open(generatorFile.toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            AudioClip whole = generator.synthesize("Hello world", twoFrames());
            java.util.List<Float> streamed = new java.util.ArrayList<>();
            generator.synthesize("Hello world", twoFrames().withChunkFrames(1), chunkAudio -> {
                for (float sample : chunkAudio.samples()) {
                    streamed.add(sample);
                }
            });
            assertEquals("the sink receives every decoded sample once", whole.length(), streamed.size());
            for (int i = 0; i < whole.length(); i++) {
                assertEquals("streamed sample " + i, whole.samples()[i], streamed.get(i), 0f);
            }
        }
    }

    @Test
    public void gpuGeneratorMatchesTheCpuPipeline() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(71L, Qwen3TtsBackboneFixture.EMB);
        Path generatorFile = fixture.write();

        EngineConfig cpu = EngineConfig.builder().memory(memory()).build();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        int[][] wantCodes;
        AudioClip wantAudio;
        try (GGUFModel model = new GGUFModel(backbone.toString(), cpu);
             Qwen3TtsClip clip = Qwen3TtsClip.open(generatorFile.toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            wantCodes = generator.generateCodes("Hello world", twoFrames());
            wantAudio = generator.synthesize("Hello world", twoFrames());
        }
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             Qwen3TtsClip clip = Qwen3TtsClip.open(generatorFile.toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            int[][] codes = generator.generateCodes("Hello world", twoFrames());
            assertEquals("the GPU runs every frame the CPU ran", wantCodes.length, codes.length);
            for (int i = 0; i < wantCodes.length; i++) {
                assertArrayEquals("frame " + i + " of the GPU pipeline", wantCodes[i], codes[i]);
            }
            AudioClip audio = generator.synthesize("Hello world", twoFrames());
            assertEquals(wantAudio.length(), audio.length());
            assertEquals(wantAudio.sampleRate(), audio.sampleRate());
            float[] want = wantAudio.samples();
            float[] got = audio.samples();
            float max = 0f;
            for (int i = 0; i < want.length; i++) {
                max = Math.max(max, Math.abs(want[i] - got[i]));
            }
            assertTrue("the decoded waveform of the GPU pipeline differs by " + max + " from the CPU decode, above "
                    + PCM_TOLERANCE, max <= PCM_TOLERANCE);
            assertArrayEquals("the decoded waveform of the GPU pipeline", want, got, PCM_TOLERANCE);
        }
    }

    @Test
    public void gpuSpeakerReferenceMatchesTheCpuPipeline() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(72L, Qwen3TtsBackboneFixture.EMB, true);
        Path generatorFile = fixture.write();
        SpeechOptions cloning = twoFrames().withSpeakerReference(reference());

        EngineConfig cpu = EngineConfig.builder().memory(memory()).build();
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(memory())
                .build();
        int[][] wantCodes;
        AudioClip wantAudio;
        try (GGUFModel model = new GGUFModel(backbone.toString(), cpu);
             Qwen3TtsClip clip = Qwen3TtsClip.open(generatorFile.toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            wantCodes = generator.generateCodes("Hello world", cloning);
            wantAudio = generator.synthesize("Hello world", cloning);
        }
        try (GGUFModelGPU model = new GGUFModelGPU(backbone.toString(), null, gpu);
             Qwen3TtsClip clip = Qwen3TtsClip.open(generatorFile.toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            int[][] codes = generator.generateCodes("Hello world", cloning);
            assertEquals(wantCodes.length, codes.length);
            for (int i = 0; i < wantCodes.length; i++) {
                assertArrayEquals("cloned frame " + i + " of the GPU pipeline", wantCodes[i], codes[i]);
            }
            AudioClip audio = generator.synthesize("Hello world", cloning);
            assertEquals(wantAudio.length(), audio.length());
            assertArrayEquals("the cloned waveform of the GPU pipeline", wantAudio.samples(), audio.samples(),
                    PCM_TOLERANCE);
        }
    }
}
