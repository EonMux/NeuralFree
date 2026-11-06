package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Gemma 3n audio encoder ({@code gemma3na}): the log-mel frontend and the conformer give the frames and rows of
 * Hugging Face transformers over the same file and clip ({@code gemma3na-reference.txt}, within {@code 1e-4} of
 * their largest magnitude), a clip past 30 s continues in a second window of 188 rows, short windows pad with the
 * last hard token's row, and a Gemma 3n file serves its audio while its gemma3nv vision tower is refused by name.
 */
public class Gemma3nAudioTest {

    private static final String REFERENCE = "com/eonmux/neuralfree/gguf/mm/gemma3na-reference.txt";
    private static final double SECONDS = 31.23;
    private static final double TOLERANCE = 1e-4;

    private static Path file() throws IOException {
        return Gemma3nAudioFixture.write(Gemma3nAudioFixture.meta(), Gemma3nAudioFixture.tensors(31));
    }

    @Test
    public void theFrontendAndEncoderMatchHuggingFace() throws IOException {
        AudioClip clip = new AudioClip(Gemma3nAudioFixture.waveform(SECONDS, 5), Gemma3nAudioFixture.SAMPLE_RATE);
        try (MmprojModel mm = MmprojModel.open(file().toString()); CpuAudioEncoder enc = mm.openAudioEncoder()) {
            List<MelChunk> windows = enc.chunks(clip);
            assertEquals(2, windows.size());
            assertEquals(2997, windows.get(0).nLen());
            assertEquals(120, windows.get(1).nLen());
            List<float[][]> embedded = enc.embed(clip);
            assertEquals(2 * Gemma3nAudioModel.TOKENS_PER_WINDOW, enc.nTokens(clip));
            for (int w = 0; w < windows.size(); w++) {
                String name = "window" + w;
                ReferenceRows.assertClose(name + " mel", ReferenceRows.read(REFERENCE, name, "mel"),
                        frameMajor(windows.get(w)), w == 0 ? 100 : 8, TOLERANCE);
                float[][] rows = embedded.get(w);
                assertEquals(Gemma3nAudioModel.TOKENS_PER_WINDOW, rows.length);
                assertEquals(Gemma3nAudioFixture.TEXT, rows[0].length);
                ReferenceRows.assertClose(name + " rows", ReferenceRows.read(REFERENCE, name, "rows"), rows,
                        w == 0 ? 4 : 1, TOLERANCE);
            }
        }
    }

    private static float[][] frameMajor(MelChunk mel) {
        float[][] out = new float[mel.nLen()][mel.nMel()];
        for (int t = 0; t < mel.nLen(); t++) {
            for (int m = 0; m < mel.nMel(); m++) {
                out[t][m] = mel.value(m, t);
            }
        }
        return out;
    }

    @Test
    public void aShortWindowPadsWithTheLastHardTokenRow() throws IOException {
        AudioClip clip = new AudioClip(Gemma3nAudioFixture.waveform(0.5, 9), Gemma3nAudioFixture.SAMPLE_RATE);
        try (MmprojModel mm = MmprojModel.open(file().toString()); CpuAudioEncoder enc = mm.openAudioEncoder()) {
            List<float[][]> windows = enc.embed(clip);
            assertEquals(1, windows.size());
            float[][] rows = windows.get(0);
            // 8000 samples pad to 8064: 48 frames, 24 and 12 after the convolutions, 3 rows after the reduction
            int real = Gemma3nAudioModel.encoderRows(enc.chunks(clip).get(0).nLen());
            assertEquals(3, real);
            float[] pad = rows[real];
            assertFalse(java.util.Arrays.equals(rows[real - 1], pad));
            for (int r = real; r < rows.length; r++) {
                assertArrayEquals(pad, rows[r], 0f);
            }
        }
    }

    @Test
    public void theGeometryReadsTheConverterLayout() throws IOException {
        ClipHparams hp = Gemma3nAudioFixture.hparams(Gemma3nAudioFixture.meta());
        assertEquals(ProjectorType.GEMMA3NA, hp.projectorType());
        assertEquals("every RMS norm of the tower uses rms_norm_eps, not the converter's 1e-5", 1e-6f, hp.eps(), 0f);
        assertEquals(16000, hp.audio().sampleRate());
        assertEquals(1024, hp.audio().nFft());
        assertEquals(512, hp.audio().windowLen());
        assertEquals(160, hp.audio().hopLen());
        assertEquals(30, hp.audio().chunkSeconds());
        ClipAudioModel model = ClipAudioModel.from(hp, Gemma3nAudioFixture.tensors(31).catalog());
        assertEquals(ProjectorEntry.ENCODER_GEMMA3N_USM, model.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_GEMMA3NA, model.projectorGraph());
        assertEquals(MelFrontend.Flavor.GEMMA3N, model.melFlavor());
        assertEquals(Gemma3nAudioModel.TOKENS_PER_WINDOW, model.nTokens(120));
        assertTrue(model.layers().isEmpty());
        Gemma3nAudioModel g = model.gemma3n();
        assertEquals(Gemma3nAudioFixture.C0, g.c0());
        assertEquals(Gemma3nAudioFixture.C1, g.c1());
        assertEquals(32, g.f1());
        assertEquals(Gemma3nAudioFixture.FF, g.ff());
        assertEquals(Gemma3nAudioFixture.KERNEL, g.convKernel());
        assertEquals(Gemma3nAudioFixture.VOCAB, g.vocab());
    }

    @Test
    public void aWindowBeyondThirtySecondsIsRefused() throws IOException {
        try (MmprojModel mm = MmprojModel.open(file().toString()); CpuAudioEncoder enc = mm.openAudioEncoder()) {
            int frames = 3100;
            MelChunk mel = MelChunk.of(Gemma3nAudioFixture.N_MEL, frames,
                    AudioFixture.mel(Gemma3nAudioFixture.N_MEL, frames, 3));
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> enc.encode(mel));
            assertTrue(e.getMessage(), e.getMessage().contains("194 gemma3na rows, above the 188"));
        }
    }

    @Test
    public void aMisshapenDepthwiseKernelIsRefusedByName() {
        MmprojFixture f = Gemma3nAudioFixture.tensors(31);
        f.remove("a.blk.1.conv_dw.weight");
        f.weight("a.blk.1.conv_dw.weight", 3, 1, Gemma3nAudioFixture.HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(Gemma3nAudioFixture.hparams(Gemma3nAudioFixture.meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.1.conv_dw.weight"));
    }

    @Test
    public void aMissingGroupNormScaleIsRefusedAtOpen() throws IOException {
        MmprojFixture f = Gemma3nAudioFixture.tensors(31);
        f.remove("a.conv1d.1.norm.weight");
        Path file = Gemma3nAudioFixture.write(Gemma3nAudioFixture.meta(), f);
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.conv1d.1.norm.weight"));
    }

    @Test
    public void aGemma3nFileServesAudioAndRefusesItsVisionTowerByName() throws IOException {
        Map<String, Object> meta = Gemma3nAudioFixture.withVisionMeta(Gemma3nAudioFixture.meta());
        Path file = Gemma3nAudioFixture.write(meta, Gemma3nAudioFixture.tensors(31));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertTrue(mm.caps().inpVision());
            assertTrue(mm.caps().inpAudio());
            assertFalse(mm.supportsVision());
            assertTrue(mm.supportsAudio());
            assertNull(mm.unrunnableReason(ProjectorModality.AUDIO));
            assertTrue(mm.unrunnableReason(ProjectorModality.VISION).contains("gemma3nv"));
            UnsupportedProjectorException e = assertThrows(UnsupportedProjectorException.class,
                    mm::openCpuVisionEncoder);
            assertEquals("gemma3nv", e.projectorType());
            try (CpuAudioEncoder enc = mm.openAudioEncoder()) {
                assertEquals(Gemma3nAudioModel.TOKENS_PER_WINDOW, enc.encode(MelChunk.of(Gemma3nAudioFixture.N_MEL,
                        40, AudioFixture.mel(Gemma3nAudioFixture.N_MEL, 40, 4))).length);
            }
        }
    }

    @Test
    public void aFileWhoseEveryEncoderIsUnrunnableStillFailsAtOpen() throws IOException {
        Map<String, Object> meta = Gemma3nAudioFixture.withVisionMeta(Gemma3nAudioFixture.meta());
        meta.put("clip.has_audio_encoder", false);
        Path file = Gemma3nAudioFixture.write(meta, Gemma3nAudioFixture.tensors(31));
        // the declared gemma3nv tower is the file's only encoder and the file carries none of its tensors:
        // nothing serves, so open fails naming the missing tensor
        com.eonmux.neuralfree.gguf.spec.MissingTensorException e =
                assertThrows(com.eonmux.neuralfree.gguf.spec.MissingTensorException.class,
                        () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("v.conv_stem.conv.weight"));
    }
}
