package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Opening an lfm3a generator file: the metadata it reads and the files it refuses, by name. */
public class Lfm3aGenClipTest {

    @Test
    public void readsTheGeometryAndTheSamplingKeys() throws IOException {
        Lfm3aGenClipFixture fixture = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(fixture.write().toString())) {
            assertEquals(Lfm3aBackboneFixture.EMB, clip.depthformer().talkerWidth());
            assertEquals(Lfm3aGenClipFixture.VOCAB, clip.vocab());
            assertEquals(Lfm3aGenClipFixture.FSQ_VOCAB, clip.fsqVocab());
            assertEquals("the fixture's temp of 0 makes the code sampler deterministic", 0f, clip.temp(), 0f);
            assertEquals(0, clip.topK());
            assertEquals(Lfm3aGenClip.DEFAULT_PENALTY_REPEAT, clip.penaltyRepeat(), 0f);
            assertEquals(Lfm3aGenClip.DEFAULT_PENALTY_LAST_N, clip.penaltyLastN());
            assertEquals("clip.gen.audio.sampling.* defaults", 1.2f, clip.penaltyRepeat(), 0f);
        }
    }

    @Test
    public void readsOverriddenSamplingKeys() throws IOException {
        Lfm3aGenClipFixture fixture = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false);
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = fixture.writer();
        w.f32("clip.gen.audio.sampling.temp", 0.7f);
        w.u32("clip.gen.audio.sampling.top_k", 40);
        w.f32("clip.gen.audio.sampling.penalty_repeat", 1.1f);
        w.u32("clip.gen.audio.sampling.penalty_last_n", 8);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(w.writeTempFile().toString())) {
            assertEquals(0.7f, clip.temp(), 0f);
            assertEquals(40, clip.topK());
            assertEquals(1.1f, clip.penaltyRepeat(), 0f);
            assertEquals(8, clip.penaltyLastN());
        }
    }

    @Test
    public void refusesFilesThatAreNotLfm3aGenerators() throws IOException {
        String backbone = new Lfm3aBackboneFixture(false, true).write().toString();
        IllegalArgumentException wrongArch = assertThrows(IllegalArgumentException.class,
                () -> Lfm3aGenClip.open(backbone));
        assertTrue(wrongArch.getMessage(), wrongArch.getMessage().contains("lfm3a generator"));

        Lfm3aGenClipFixture fixture = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false);
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = fixture.writer();
        w.str("clip.gen.audio.projector_type", "lfm3a");
        IllegalArgumentException wrongType = assertThrows(IllegalArgumentException.class,
                () -> Lfm3aGenClip.open(w.writeTempFile().toString()));
        assertTrue(wrongType.getMessage(), wrongType.getMessage().contains("lfm3a_gen"));
    }

    @Test
    public void refusesAMismatchedFsqTable() throws IOException {
        Lfm3aGenClipFixture fixture = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false);
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = fixture.writer();
        w.tensorFromFloats("a.gen.wav.quant.first.codebook.weight",
                com.eonmux.neuralfree.gguf.GGMLType.F32, new long[]{Lfm3aGenClipFixture.FSQ_LEVELS, 6},
                new float[Lfm3aGenClipFixture.FSQ_LEVELS * 6]);
        assertThrows(GgufFormatException.class, () -> Lfm3aGenClip.open(w.writeTempFile().toString()));
    }

    @Test
    public void closeIsIdempotent() throws IOException {
        Lfm3aGenClipFixture fixture = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false);
        Lfm3aGenClip clip = Lfm3aGenClip.open(fixture.write().toString());
        clip.close();
        clip.close();
    }
}
