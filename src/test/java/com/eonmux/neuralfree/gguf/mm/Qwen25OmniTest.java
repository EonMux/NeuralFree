package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Qwen2.5-Omni files ({@code clip.projector_type qwen2.5o}): one open serves the vision encoder as
 * {@code qwen2.5vl_merger} and the audio encoder as {@code qwen2a}, as clip.cpp {@code load_hparams} resolves the
 * legacy name, and both encoders give the rows of the stored references ({@code qwen25o-reference.txt}): an exact
 * evaluation of llama.cpp's graphs within {@code 1e-4} of the output scale, and llama.cpp's own CPU rows within the
 * drift of its F16 convolution inputs and GELU table.
 */
public class Qwen25OmniTest {

    private static final String REFERENCE = "com/eonmux/neuralfree/gguf/mm/qwen25o-reference.txt";
    private static final int WIDTH = 16;
    private static final int IMAGE_W = 16;
    private static final int IMAGE_H = 8;
    private static final int FRAMES = 3000;
    /** Audio rows the reference file stores: every 25th. */
    private static final int AUDIO_STRIDE = 25;
    /** Largest difference to the exact rows, as a fraction of their largest magnitude. */
    private static final double EXACT_TOLERANCE = 1e-4;
    /** Largest difference to llama.cpp's rows, as a fraction of their largest magnitude (its F16 rounding). */
    private static final double LLAMA_TOLERANCE = 2e-3;

    @Test
    public void bothEncodersRunUnderTheirResolvedTypesAndMatchTheReferences() throws IOException {
        Path file = MmprojTestFiles.qwen25Omni(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder vision = mm.openCpuVisionEncoder();
             CpuAudioEncoder audio = mm.openAudioEncoder()) {
            assertTrue(mm.supportsVision());
            assertTrue(mm.supportsAudio());
            assertEquals(ProjectorType.QWEN25VL, mm.visionHparams().projectorType());
            assertEquals(ProjectorType.QWEN2A, mm.audioHparams().projectorType());

            float[][] image = vision.encode(ClipImage.of(IMAGE_W, IMAGE_H,
                    MmprojFixture.pixels(IMAGE_W, IMAGE_H, 7)));
            assertEquals(8, image.length);
            assertMatches("vision", image, 1);

            float[][] sound = audio.encode(MelChunk.of(AudioFixture.N_MEL, FRAMES,
                    AudioFixture.mel(AudioFixture.N_MEL, FRAMES, 17)));
            assertEquals(FRAMES / 4, sound.length);
            assertMatches("audio", sound, AUDIO_STRIDE);
        }
    }

    private static void assertMatches(String encoder, float[][] got, int stride) throws IOException {
        ReferenceRows.assertClose(encoder + " exact", ReferenceRows.read(REFERENCE, encoder, "exact"), got, stride,
                EXACT_TOLERANCE);
        ReferenceRows.assertClose(encoder + " llama.cpp", ReferenceRows.read(REFERENCE, encoder, "llama.cpp"), got,
                stride, LLAMA_TOLERANCE);
    }

    @Test
    public void theRegistryResolvesTheLegacyNamePerModality() {
        assertEquals(ProjectorEntry.ENCODER_QWEN2VL_VIT,
                ProjectorRegistry.get("qwen2.5o", ProjectorModality.VISION).encoderGraph());
        assertEquals(ProjectorEntry.ENCODER_WHISPER,
                ProjectorRegistry.get("qwen2.5o", ProjectorModality.AUDIO).encoderGraph());
    }
}
