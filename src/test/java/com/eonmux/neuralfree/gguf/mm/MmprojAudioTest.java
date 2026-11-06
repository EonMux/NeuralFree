package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioDecoder;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioFixture.N_MEL;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.TEXT;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Audio encoders opened from real mmproj GGUF files through {@link MmprojModel} (llama.cpp
 * {@code mtmd_init_from_file} with an audio tower): raw audio bytes to embedding rows, one file serving a vision
 * and an audio encoder, quantized weights, and the loader's refusals.
 */
public class MmprojAudioTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int FRAMES = 10;

    private Path write(GgufTestWriter w) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    private Path writeAudio(String projector, int stack, MmprojFixture f, GGMLType matrices) throws IOException {
        return write(AudioFixture.write(new GgufTestWriter(), AudioFixture.whisperMeta(projector, stack), f,
                matrices));
    }

    /** A 16 kHz mono PCM16 sine WAV. */
    private static byte[] wavSine(int seconds) {
        int rate = 16000;
        int frames = seconds * rate;
        int dataBytes = frames * 2;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + dataBytes);
        bb.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        bb.put("fmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16);
        bb.putShort((short) 1).putShort((short) 1).putInt(rate).putInt(rate * 2);
        bb.putShort((short) 2).putShort((short) 16);
        bb.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(dataBytes);
        for (int i = 0; i < frames; i++) {
            bb.putShort((short) (Math.sin(2.0 * Math.PI * 440 * i / rate) * 12000));
        }
        return bb.array();
    }

    @Test
    public void embedsWavAudioEndToEnd() throws IOException {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 3, 1500);
        try (MmprojModel mm = MmprojModel.open(writeAudio("qwen2a", 0, f, GGMLType.F16).toString());
             CpuAudioEncoder enc = mm.openAudioEncoder()) {
            assertTrue(mm.caps().inpAudio());
            assertFalse(mm.supportsVision());
            ClipAudioHparams audio = mm.audioHparams().audio();
            assertEquals(16000, audio.sampleRate());
            assertEquals(ProjectorType.QWEN2A, enc.model().hparams().projectorType());
            AudioClip clip = AudioDecoder.decode(wavSine(2), "speech.wav");
            List<float[][]> embeddings = enc.embed(clip);
            // 2 s of audio gives one 3000-frame whisper chunk; the conv halves it to 1500 positions and the
            // average pool to 750 rows
            assertEquals(1, embeddings.size());
            assertEquals(750, embeddings.get(0).length);
            assertEquals(TEXT, embeddings.get(0)[0].length);
            for (float[] row : embeddings.get(0)) {
                for (float v : row) {
                    assertTrue(Float.isFinite(v));
                }
            }
            assertEquals(750, enc.nTokens(clip));
        }
    }

    @Test
    public void longAudioSplitsIntoChunksEncodedInOrder() throws IOException {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 3, 1500);
        try (MmprojModel mm = MmprojModel.open(writeAudio("qwen2a", 0, f, GGMLType.F32).toString());
             CpuAudioEncoder enc = mm.openAudioEncoder()) {
            AudioClip clip = AudioDecoder.decode(wavSine(30), "long.wav");
            List<MelChunk> chunks = enc.chunks(clip);
            List<float[][]> all = enc.embed(clip);
            assertEquals(2, chunks.size());
            assertEquals(2, all.size());
            float[][] first = enc.encode(chunks.get(0));
            for (int r = 0; r < first.length; r++) {
                assertArrayEquals(first[r], all.get(0)[r], 0f);
            }
        }
    }

    @Test
    public void fileEncoderMatchesTheReference() throws IOException {
        MmprojFixture f = AudioFixture.whisper("ultravox", 3, 21);
        Path file = writeAudio("ultravox", 3, f, GGMLType.F32);
        float[] mel = AudioFixture.mel(N_MEL, FRAMES, 99);
        try (MmprojModel mm = MmprojModel.open(file.toString()); CpuAudioEncoder enc = mm.openAudioEncoder()) {
            CpuAudioEncoderTest.assertRows(AudioFixture.whisperReference(f, "ultravox", 3, mel, FRAMES),
                    enc.encode(MelChunk.of(N_MEL, FRAMES, mel)), 2e-4);
        }
    }

    @Test
    public void quantizedWeightsRunThroughTheEngineMatmul() throws IOException {
        MmprojFixture f = AudioFixture.whisper("musicflamingo", 0, 8);
        Path f32 = writeAudio("musicflamingo", 0, f, GGMLType.F32);
        Path q8 = writeAudio("musicflamingo", 0, f, GGMLType.Q8_0);
        float[] mel = AudioFixture.mel(N_MEL, FRAMES, 5);
        try (MmprojModel full = MmprojModel.open(f32.toString());
             MmprojModel quant = MmprojModel.open(q8.toString());
             CpuAudioEncoder a = full.openAudioEncoder();
             CpuAudioEncoder b = quant.openAudioEncoder()) {
            assertTrue(quant.memoryUsageBytes() < full.memoryUsageBytes());
            float[][] want = a.encode(MelChunk.of(N_MEL, FRAMES, mel));
            float[][] got = b.encode(MelChunk.of(N_MEL, FRAMES, mel));
            double max = 0;
            double diff = 0;
            for (int r = 0; r < want.length; r++) {
                for (int c = 0; c < TEXT; c++) {
                    max = Math.max(max, Math.abs(want[r][c]));
                    diff = Math.max(diff, Math.abs(want[r][c] - got[r][c]));
                }
            }
            assertTrue("Q8_0 drift " + diff + " of " + max, diff < 0.05 * max);
            assertTrue("the Q8_0 weights are used", diff > 0);
        }
    }

    @Test
    public void oneFileServesVisionAndAudioFromOneOpen() throws IOException {
        MmprojFixture vision = CpuVisionEncoderTest.llavaFixture(4, true, false);
        MmprojFixture audio = AudioFixture.whisper("qwen2a", 0, 6);
        Map<String, Object> meta = new HashMap<>(AudioFixture.whisperMeta("qwen2a", 0));
        meta.remove("clip.projector_type");
        meta.put("clip.audio.projector_type", "qwen2a");
        meta.putAll(visionMeta());
        GgufTestWriter w = AudioFixture.write(new GgufTestWriter(), meta, audio, GGMLType.F32);
        for (Map.Entry<String, long[]> e : vision.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), vision.get(e.getKey()));
        }
        Path file = write(w);
        float[] pixels = MmprojFixture.pixels(8, 8, 42);
        float[] mel = AudioFixture.mel(N_MEL, FRAMES, 17);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertTrue(mm.caps().inpVision());
            assertTrue(mm.caps().inpAudio());
            assertEquals(ProjectorType.MLP, mm.visionHparams().projectorType());
            assertEquals(ProjectorType.QWEN2A, mm.audioHparams().projectorType());
            try (CpuVisionEncoder venc = mm.openCpuVisionEncoder(); CpuAudioEncoder aenc = mm.openAudioEncoder()) {
                float[][] image = venc.encode(ClipImage.of(8, 8, pixels));
                float[][] imageWant = MmprojFixture.Reference.clip(vision, mm.visionHparams(), pixels, 8, 8);
                assertEquals(imageWant.length, image.length);
                for (int r = 0; r < image.length; r++) {
                    assertArrayEquals(imageWant[r], image[r], 2e-4f);
                }
                CpuAudioEncoderTest.assertRows(AudioFixture.whisperReference(audio, "qwen2a", 1, mel, FRAMES),
                        aenc.encode(MelChunk.of(N_MEL, FRAMES, mel)), 2e-4);
            }
        }
    }

    private static Map<String, Object> visionMeta() {
        Map<String, Object> m = new HashMap<>();
        m.put("clip.has_vision_encoder", true);
        m.put("clip.vision.projector_type", "mlp");
        m.put("clip.vision.image_size", 8);
        m.put("clip.vision.patch_size", 4);
        m.put("clip.vision.embedding_length", 8);
        m.put("clip.vision.feed_forward_length", 12);
        m.put("clip.vision.block_count", 2);
        m.put("clip.vision.projection_dim", 6);
        m.put("clip.vision.attention.head_count", 2);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        m.put("clip.vision.image_mean", new float[]{0.48f, 0.45f, 0.4f});
        m.put("clip.vision.image_std", new float[]{0.26f, 0.26f, 0.27f});
        m.put("clip.use_gelu", true);
        return m;
    }

    @Test
    public void unknownProjectorNameIsRefusedByName() throws IOException {
        Map<String, Object> meta = AudioFixture.whisperMeta("nonsense", 0);
        Path file = write(AudioFixture.write(new GgufTestWriter(), meta, AudioFixture.whisper("qwen2a", 0, 1),
                GGMLType.F32));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("nonsense"));
    }

    @Test
    public void missingProjectorTypeIsRefusedByKeyName() throws IOException {
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen2a", 0);
        meta.remove("clip.projector_type");
        Path file = write(AudioFixture.write(new GgufTestWriter(), meta, AudioFixture.whisper("qwen2a", 0, 1),
                GGMLType.F32));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.projector_type"));
    }

    @Test
    public void audioProjectorWithoutAGraphIsRefusedByName() throws IOException {
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen3tts_spkenc", 0);
        Path file = write(AudioFixture.write(new GgufTestWriter(), meta, AudioFixture.whisper("qwen2a", 0, 1),
                GGMLType.F32));
        // the Qwen3-TTS speaker encoder is a known audio name, so the refusal names the family the registry
        // cannot run
        UnsupportedProjectorException e = assertThrows(UnsupportedProjectorException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("qwen3tts_spkenc"));
        assertTrue(e.getMessage(), e.getMessage().contains("not runnable"));
    }

    @Test
    public void conformerProjectorNamesItsMissingTensors() throws IOException {
        Map<String, Object> meta = AudioFixture.whisperMeta("lfm2a", 0);
        Path file = write(AudioFixture.write(new GgufTestWriter(), meta, AudioFixture.whisper("qwen2a", 0, 1),
                GGMLType.F32));
        // lfm2a runs, so a file holding whisper tensors is refused for the first conformer tensor it lacks
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.conv1d.0.weight"));
    }

    @Test
    public void filesWithoutAnAudioEncoderRefuseTheAudioExecutor() throws IOException {
        MmprojFixture vision = CpuVisionEncoderTest.llavaFixture(4, true, false);
        Map<String, Object> meta = visionMeta();
        meta.put("general.architecture", "clip");
        GgufTestWriter w = AudioFixture.write(new GgufTestWriter(), meta, new MmprojFixture(0), GGMLType.F32);
        for (Map.Entry<String, long[]> e : vision.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), vision.get(e.getKey()));
        }
        try (MmprojModel mm = MmprojModel.open(write(w).toString())) {
            assertFalse(mm.supportsAudio());
            assertThrows(IllegalArgumentException.class, mm::openAudioEncoder);
        }
    }
}
