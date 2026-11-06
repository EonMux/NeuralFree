package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The MaskGIT depthformer against its synthetic graph, whose math reduces to the reference's identity blocks
 * (lfm3a-gen.cpp build_gen_code): zero projection, zero blocks and unit norms leave {@code cur} as the revealed
 * codes' embedding rows and the mask row, and the tied head reads them through the two norms. Every logits row is
 * {@code normed[0] * A[code]} with {@code A} strictly decreasing but for the end code, so an argmax and every
 * value are hand-computable; the feedback embedding sums one {@code out_embd} axis per codebook.
 */
public class Lfm3aGenDepthformerTest {

    private static final float EPS = 1e-5f;
    private static final int CB = Lfm3aDepthformer.N_CODEBOOKS;
    private static final int VOCAB = Lfm3aGenClipFixture.VOCAB;

    /**
     * The head scale of one nonzero axis {@code x}: the head reads the row through the output norm and then the
     * per-codebook head norm (lfm3a-gen.cpp's two chained RMS norms), so the scale composes
     * {@code hs(hs(x)) / sqrt(hs(x)^2 / n_embd + eps)} and the magnitude collapses onto {@code sqrt(n_embd)}.
     */
    private static float headScale(float x) {
        float once = (float) (x / Math.sqrt(x * x / (double) Lfm3aGenClipFixture.N_EMBD + EPS));
        return (float) (once / Math.sqrt(once * once / (double) Lfm3aGenClipFixture.N_EMBD + EPS));
    }

    @Test
    public void maskedPassReadsTheMaskRowThroughTheTiedHead() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, -1);
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            h[3] = 2f;
            Lfm3aDepthformer.Frame pass = df.pass(h, codes);
            assertTrue(pass.logits() != null);
            // the zero mask row norms to zero: every logits row is zero and the argmax tie falls to code 0
            float[] logits = pass.logits();
            assertEquals(CB * VOCAB, logits.length);
            for (int i = 0; i < logits.length; i++) {
                assertEquals(0f, logits[i], 1e-6f);
            }
            assertEquals("the hidden width the projection takes", Lfm3aBackboneFixture.EMB, df.talkerWidth());
        }
    }

    @Test
    public void revealedCodebooksReadTheirOwnEmbeddingRow() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, -1);
            codes[0] = 2;
            codes[5] = 4;
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            h[7] = 1f;
            float[] logits = df.pass(h, codes).logits();
            float c0 = headScale(Lfm3aGenClipFixture.A[2]);
            float c5 = headScale(Lfm3aGenClipFixture.A[4]);
            for (int cb = 0; cb < CB; cb++) {
                float c = cb == 0 ? c0 : cb == 5 ? c5 : 0f;
                for (int v = 0; v < VOCAB; v++) {
                    assertEquals("codebook " + cb + " code " + v, c * Lfm3aGenClipFixture.A[v],
                            logits[cb * VOCAB + v], 1e-4f);
                }
            }
        }
    }

    @Test
    public void aNegativeMaskFlipsEveryCodebookOntoTheEndCode() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, true).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, -1);
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            float c = headScale(-3f);
            float[] logits = df.pass(h, codes).logits();
            for (int cb = 0; cb < CB; cb++) {
                int best = 0;
                for (int v = 1; v < VOCAB; v++) {
                    if (logits[cb * VOCAB + v] > logits[cb * VOCAB + best]) {
                        best = v;
                    }
                }
                assertEquals("the end code wins codebook " + cb, VOCAB - 1, best);
                for (int v = 0; v < VOCAB; v++) {
                    assertEquals(c * Lfm3aGenClipFixture.A[v], logits[cb * VOCAB + v], 1e-4f);
                }
            }
        }
    }

    @Test
    public void anAllKnownPassSkipsTheBlocksAndOnlyFeedsBack() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, 0);
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            Lfm3aDepthformer.Frame pass = df.pass(h, codes);
            assertNull("all known: the reference's all_known skips the blocks", pass.logits());
            
            // codebook cb feeds axis cb with code + 1: twelve ones over the first twelve axes
            float[] fb = pass.feedback();
            assertEquals(Lfm3aBackboneFixture.EMB, fb.length);
            for (int i = 0; i < Lfm3aBackboneFixture.EMB; i++) {
                assertEquals(i < CB ? 1f : 0f, fb[i], 1e-6f);
            }

            Arrays.fill(codes, VOCAB - 1);
            float[] endFb = df.feedback(codes);
            for (int i = 0; i < CB; i++) {
                assertEquals("the end code carries code + 1 = 6", 6f, endFb[i], 1e-6f);
            }
        }
    }

    @Test
    public void maskedCodebooksFeedBackNothing() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, -1);
            codes[3] = 2;
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            float[] fb = df.pass(h, codes).feedback();
            // only the revealed codebook feeds back: a masked code contributes no row (its logits are unused)
            for (int i = 0; i < Lfm3aBackboneFixture.EMB; i++) {
                float expected = i == 3 ? 3f : 0f;
                assertEquals("axis " + i, expected, fb[i], 1e-6f);
            }
        }
    }

    @Test
    public void feedbackIsACopy() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            int[] codes = new int[CB];
            Arrays.fill(codes, 0);
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            float[] fb = df.pass(h, codes).feedback();
            fb[0] = 99f;
            assertEquals(1f, df.pass(h, codes).feedback()[0], 1e-6f);
        }
    }

    @Test
    public void refusesWrongWidthsAndCodes() throws IOException {
        Path file = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            float[] h = new float[Lfm3aBackboneFixture.EMB];
            assertThrows(IllegalArgumentException.class, () -> df.pass(h, new int[CB - 1]));
            assertThrows(IllegalArgumentException.class,
                    () -> df.pass(new float[Lfm3aBackboneFixture.EMB + 1], new int[CB]));
            int[] tooBig = new int[CB];
            Arrays.fill(tooBig, VOCAB);
            assertThrows(IllegalArgumentException.class, () -> df.pass(h, tooBig));
        }
    }
}
