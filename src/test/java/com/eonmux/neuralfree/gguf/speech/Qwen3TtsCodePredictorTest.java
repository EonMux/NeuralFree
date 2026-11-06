package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link Qwen3TtsCodePredictor} over the synthetic generator file: the projection, the codec-0 embedding, the
 * per-step codebook pages, the truncated draw and the feedback row's embedding sum.
 */
public class Qwen3TtsCodePredictorTest {

    @Test
    public void oneFrameSamplesThePagesAndSumsTheFeedback() throws IOException {
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(Qwen3TtsFixtures.writer().writeTempFile().toString())) {
            Qwen3TtsCodePredictor predictor = clip.predictor();
            assertEquals(Qwen3TtsFixtures.CODES, predictor.codesPerFrame());
            assertEquals(Qwen3TtsFixtures.W, predictor.talkerWidth());
            assertEquals(Qwen3TtsFixtures.CODEBOOKS, predictor.nCodebooks());
            Qwen3TtsCodePredictor.Session session = predictor.newSession(1, 1f);
            Qwen3TtsCodePredictor.Frame frame = session.process(new float[]{3f, 7f}, 1, new Random(0));
            // page 0 samples code 2 (its head row is the only non-zero one), page 1 code 3
            assertArrayEquals(new int[]{1, 2, 3}, frame.codes());
            // out_embd[1] + embd page 0 row 2 + embd page 1 row 3 = [0,1] + [1,0] + [0,1]
            assertArrayEquals(new float[]{1f, 2f}, frame.feedback(), 1e-6f);
        }
    }

    @Test
    public void sessionsAreIndependentAndRepeatable() throws IOException {
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(Qwen3TtsFixtures.writer().writeTempFile().toString())) {
            Qwen3TtsCodePredictor predictor = clip.predictor();
            Qwen3TtsCodePredictor.Session a = predictor.newSession(1, 1f);
            Qwen3TtsCodePredictor.Session b = predictor.newSession(1, 1f);
            Qwen3TtsCodePredictor.Frame first = a.process(new float[]{3f, 7f}, 1, new Random(1));
            Qwen3TtsCodePredictor.Frame second = b.process(new float[]{3f, 7f}, 1, new Random(1));
            assertArrayEquals(first.codes(), second.codes());
            assertArrayEquals(first.feedback(), second.feedback(), 1e-6f);
            // a second frame on the same session reuses its 16-row cache without growing it
            Qwen3TtsCodePredictor.Frame next = a.process(new float[]{0f, 1f}, 0, new Random(2));
            assertEquals(Qwen3TtsFixtures.CODES, next.codes().length);
        }
    }

    @Test
    public void refusesBadInputs() throws IOException {
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(Qwen3TtsFixtures.writer().writeTempFile().toString())) {
            Qwen3TtsCodePredictor.Session session = clip.predictor().newSession();
            assertThrows(IllegalArgumentException.class,
                    () -> session.process(new float[]{1f}, 0, new Random(0)));
            assertThrows(IllegalArgumentException.class,
                    () -> session.process(new float[]{1f, 2f}, -1, new Random(0)));
            assertThrows(IllegalArgumentException.class,
                    () -> session.process(new float[]{1f, 2f}, Qwen3TtsFixtures.VOCAB, new Random(0)));
        }
    }
}
