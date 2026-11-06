package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.mm.AudioFixture.HIDDEN;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.MAX_POS;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.N_MEL;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.Q3_MEL;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.TEXT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU audio encoder against an independent double-precision transcription of llama.cpp
 * tools/mtmd/models/whisper-enc.cpp (both conv1d blocks, learned positions, the shared build_vit tower with its
 * average pool and post-norm, the frame stack and every whisper-family projector) and qwen3a.cpp (the three
 * stride-2 conv2d blocks, conv_out, the per-batch position repetition, the tower and the MLP projector).
 */
public class CpuAudioEncoderTest {

    private static final double TOL = 2e-4;
    private static final int FRAMES = 10;

    private static CpuAudioEncoder encoder(MmprojFixture f, ClipHparams hp) {
        return CpuAudioEncoder.bind(ClipAudioModel.from(hp, f.catalog()), CpuMatVec.PLAIN, f::get, f::ref);
    }

    private static CpuAudioEncoder whisper(MmprojFixture f, String projector, int stack) {
        return encoder(f, AudioFixture.hparams(AudioFixture.whisperMeta(projector, stack)));
    }

    static void assertRows(double[][] want, float[][] got, double tol) {
        assertEquals("rows", want.length, got.length);
        for (int r = 0; r < want.length; r++) {
            assertEquals("width of row " + r, want[r].length, got[r].length);
            for (int c = 0; c < want[r].length; c++) {
                double scale = Math.max(1.0, Math.abs(want[r][c]));
                assertEquals("row " + r + " col " + c, want[r][c], got[r][c], tol * scale);
            }
        }
    }

    /** Encodes the fixture's mel chunk and checks it against the reference and the model's token count. */
    private static float[][] checkWhisper(String projector, int stack, int wantRows) {
        MmprojFixture f = AudioFixture.whisper(projector, stack, 11);
        float[] mel = AudioFixture.mel(N_MEL, FRAMES, 1234);
        try (CpuAudioEncoder enc = whisper(f, projector, stack)) {
            float[][] out = enc.encode(MelChunk.of(N_MEL, FRAMES, mel));
            assertRows(AudioFixture.whisperReference(f, projector, Math.max(1, stack), mel, FRAMES), out, TOL);
            assertEquals(wantRows, out.length);
            assertEquals(TEXT, out[0].length);
            assertEquals(wantRows, enc.model().nTokens(FRAMES));
            return out;
        }
    }

    @Test
    public void qwen2aPoolAndFcMatchReference() {
        // 10 frames -> 5 positions after the stride-2 conv -> 2 rows after the average pool
        checkWhisper("qwen2a", 0, 2);
    }

    @Test
    public void ultravoxStackAndSwigluMatchReference() {
        checkWhisper("ultravox", 3, (FRAMES / 2 * HIDDEN + HIDDEN * 3 - 1) / (HIDDEN * 3));
    }

    @Test
    public void voxtralPoolStackAndMlpMatchReference() {
        checkWhisper("voxtral", 3, 1);
    }

    @Test
    public void musicflamingoPoolsAndRunsTheBiasedMlp() {
        checkWhisper("musicflamingo", 0, 2);
    }

    @Test
    public void meralionGluAdaptorMatchesReference() {
        checkWhisper("meralion", 3, 2);
    }

    @Test
    public void glmaStacksAndAppendsSeparatorRows() {
        MmprojFixture f = AudioFixture.whisper("glma", 2, 11);
        float[][] out = checkWhisper("glma", 2, (FRAMES / 2 + 1) / 2 + 2);
        float[] boi = f.get("v.boi");
        float[] eoi = f.get("v.eoi");
        for (int c = 0; c < TEXT; c++) {
            assertEquals(boi[c], out[0][c], 0f);
            assertEquals(eoi[c], out[out.length - 1][c], 0f);
        }
    }

    @Test
    public void positionRowsAreReadInGgmlOrder() {
        // row t of a.position_embd.weight is the n_embd values at t * n_embd: value 128 belongs to position 8,
        // which a 10-frame chunk (5 positions) never reads; read transposed it would be position 0 instead
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 11);
        float[] mel = AudioFixture.mel(N_MEL, FRAMES, 1234);
        float[][] base;
        try (CpuAudioEncoder enc = whisper(f, "qwen2a", 0)) {
            base = enc.encode(MelChunk.of(N_MEL, FRAMES, mel));
        }
        f.get("a.position_embd.weight")[8 * HIDDEN] += 5f;
        try (CpuAudioEncoder enc = whisper(f, "qwen2a", 0)) {
            float[][] moved = enc.encode(MelChunk.of(N_MEL, FRAMES, mel));
            for (int r = 0; r < base.length; r++) {
                for (int c = 0; c < TEXT; c++) {
                    assertEquals(base[r][c], moved[r][c], 0f);
                }
            }
        }
    }

    @Test
    public void qwen3aMatchesReference() {
        MmprojFixture f = AudioFixture.qwen3a(5);
        int frames = 2 * ClipAudioModel.QWEN3A_CHUNK_FRAMES;
        float[] mel = AudioFixture.mel(Q3_MEL, frames, 777);
        try (CpuAudioEncoder enc = encoder(f, AudioFixture.hparams(AudioFixture.qwen3aMeta()))) {
            float[][] out = enc.encode(MelChunk.of(Q3_MEL, frames, mel));
            assertRows(AudioFixture.qwen3aReference(f, mel, frames), out, TOL);
            assertEquals(2 * ClipAudioModel.QWEN3A_TOKENS_PER_CHUNK, out.length);
            assertEquals(out.length, enc.model().nTokens(frames));
        }
    }

    @Test
    public void qwen3aRefusesFramesOutsideTheConvBatchAndOtherMelWidths() {
        MmprojFixture f = AudioFixture.qwen3a(5);
        int frames = 2 * ClipAudioModel.QWEN3A_CHUNK_FRAMES;
        try (CpuAudioEncoder enc = encoder(f, AudioFixture.hparams(AudioFixture.qwen3aMeta()))) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(MelChunk.of(Q3_MEL, frames + 1, new float[Q3_MEL * (frames + 1)])));
            assertTrue(e.getMessage().contains("100"));
            assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(MelChunk.of(Q3_MEL + 1, frames, new float[(Q3_MEL + 1) * frames])));
        }
    }

    @Test
    public void wrongMelWidthShortAndOverlongChunksAreRefused() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 11);
        try (CpuAudioEncoder enc = whisper(f, "qwen2a", 0)) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(MelChunk.of(N_MEL + 1, FRAMES, new float[(N_MEL + 1) * FRAMES])))
                    .getMessage().contains("bins"));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(MelChunk.of(N_MEL, 2, new float[N_MEL * 2])))
                    .getMessage().contains("pool"));
            int tooLong = (MAX_POS + 8) * 2 + 4;
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(MelChunk.of(N_MEL, tooLong, new float[N_MEL * tooLong])))
                    .getMessage().contains("position table"));
        }
    }

    @Test
    public void closedEncoderRefusesWork() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 11);
        CpuAudioEncoder enc = whisper(f, "qwen2a", 0);
        enc.close();
        assertThrows(IllegalStateException.class,
                () -> enc.encode(MelChunk.of(N_MEL, FRAMES, AudioFixture.mel(N_MEL, FRAMES, 1))));
    }
}
