package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link PocketTtsFlow}: the resolved geometry, the flow-matching integration and the end-of-speech probe against an
 * independent double-precision transcription of llama.cpp's GEN_CODE graph, the latent projection, and the refusals.
 */
public class PocketTtsFlowTest {

    private static PocketTtsClip open(long seed) throws IOException {
        return PocketTtsClip.open(new PocketTtsFixtures(seed).write().toString());
    }

    private static double[] condition() {
        double[] cond = new double[PocketTtsFixtures.EMB];
        for (int i = 0; i < cond.length; i++) {
            cond[i] = 0.15 * (i + 1) - 0.4;
        }
        return cond;
    }

    private static float[] asFloats(double[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (float) values[i];
        }
        return out;
    }

    @Test
    public void geometryFollowsTheGeneratorFile() throws IOException {
        try (PocketTtsClip clip = open(1L)) {
            PocketTtsFlow flow = clip.flow();
            assertEquals(PocketTtsFixtures.LATENT, flow.latent());
            assertEquals(PocketTtsFixtures.EMB, flow.condWidth());
            assertEquals(PocketTtsFixtures.FLOW, flow.hidden());
            assertEquals(PocketTtsFixtures.FLOW_BLOCKS, flow.blocks());
        }
    }

    @Test
    public void eosProbeMatchesTheReferenceAndItsThreshold() throws IOException {
        try (PocketTtsClip clip = open(2L)) {
            double[] cond = condition();
            PocketTtsReference reference = new PocketTtsReference(new PocketTtsFixtures(2L));
            double expected = reference.eos(cond);
            float actual = clip.flow().eosScore(asFloats(cond));
            assertEquals(expected, actual, 1e-6);
            assertEquals(expected > PocketTtsFlow.EOS_THRESHOLD, clip.flow().isEndOfSpeech(asFloats(cond)));
        }
    }

    @Test
    public void deterministicDecodeMatchesTheReferenceGraph() throws IOException {
        try (PocketTtsClip clip = open(3L)) {
            PocketTtsFixtures fixture = new PocketTtsFixtures(3L);
            PocketTtsReference reference = new PocketTtsReference(fixture);
            double[] cond = condition();
            double[] expected = reference.flow(cond, new double[PocketTtsFixtures.LATENT]);
            float[] actual = clip.flow().decode(asFloats(cond), 0f, new Random(0L));
            assertEquals(expected.length, actual.length);
            for (int i = 0; i < expected.length; i++) {
                assertEquals("latent " + i, expected[i], actual[i], 1e-5);
            }
        }
    }

    @Test
    public void noiseIsScaledByTheSquareRootOfTheTemperature() throws IOException {
        try (PocketTtsClip clip = open(4L)) {
            PocketTtsFixtures fixture = new PocketTtsFixtures(4L);
            PocketTtsReference reference = new PocketTtsReference(fixture);
            double[] cond = condition();
            float temperature = 0.7f;
            Random noiseRng = new Random(11L);
            double[] noise = new double[PocketTtsFixtures.LATENT];
            double std = Math.sqrt(temperature);
            for (int i = 0; i < noise.length; i++) {
                noise[i] = noiseRng.nextGaussian() * std;
            }
            double[] expected = reference.flow(cond, noise);
            float[] actual = clip.flow().decode(asFloats(cond), temperature, new Random(11L));
            for (int i = 0; i < expected.length; i++) {
                assertEquals("latent " + i, expected[i], actual[i], 1e-5);
            }
        }
    }

    @Test
    public void sameSeedRepeatsAndTemperatureZeroIsDeterministic() throws IOException {
        try (PocketTtsClip clip = open(5L)) {
            float[] cond = asFloats(condition());
            float[] a = clip.flow().decode(cond, 0.7f, new Random(99L));
            float[] b = clip.flow().decode(cond, 0.7f, new Random(99L));
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                assertEquals(a[i], b[i], 0f);
            }
            float[] c = clip.flow().decode(cond, 0f, new Random(99L));
            float[] d = clip.flow().decode(cond, 0f, new Random(1234L));
            for (int i = 0; i < c.length; i++) {
                assertEquals("temperature 0 ignores the draw", c[i], d[i], 0f);
            }
        }
    }

    @Test
    public void latentProjectionMatchesTheReference() throws IOException {
        try (PocketTtsClip clip = open(6L)) {
            PocketTtsReference reference = new PocketTtsReference(new PocketTtsFixtures(6L));
            double[] latent = {0.5, -0.25, 0.125};
            double[] expected = reference.project(latent);
            float[] actual = clip.flow().project(asFloats(latent));
            assertEquals(expected.length, actual.length);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], actual[i], 1e-6);
            }
        }
    }

    @Test
    public void refusesWrongRowsAndTemperatures() throws IOException {
        try (PocketTtsClip clip = open(7L)) {
            PocketTtsFlow flow = clip.flow();
            float[] cond = asFloats(condition());
            assertThrows(IllegalArgumentException.class, () -> flow.eosScore(new float[1]));
            assertThrows(IllegalArgumentException.class, () -> flow.isEndOfSpeech(new float[1]));
            assertThrows(IllegalArgumentException.class, () -> flow.decode(cond, -0.1f, new Random(0L)));
            assertThrows(IllegalArgumentException.class, () -> flow.decode(cond, Float.NaN, new Random(0L)));
            assertThrows(IllegalArgumentException.class,
                    () -> flow.decode(cond, Float.POSITIVE_INFINITY, new Random(0L)));
            assertThrows(IllegalArgumentException.class, () -> flow.project(new float[1]));
            assertThrows(NullPointerException.class, () -> flow.decode(cond, 0f, null));
        }
    }

    @Test
    public void refusesAnotherGeneratorFamilyAndMissingTensors() throws IOException {
        GgufTestWriter wrongType = new GgufTestWriter().architecture("clip")
                .str("clip.gen.audio.projector_type", "qwen3tts_gen");
        IllegalArgumentException type = assertThrows(IllegalArgumentException.class,
                () -> PocketTtsClip.open(wrongType.writeTempFile().toString()));
        assertTrue(type.getMessage().contains("qwen3tts_gen"));

        GgufTestWriter missing = new PocketTtsFixtures(8L).writer().removeTensor("a.gen.flow.time.0.freqs");
        assertThrows(MissingTensorException.class, () -> PocketTtsClip.open(missing.writeTempFile().toString()));
    }

    @Test
    public void eosThresholdIsMinusFour() throws IOException {
        try (PocketTtsClip hit = PocketTtsClip.open(new PocketTtsFixtures(9L).writeWithEosBias(100f).toString())) {
            assertTrue(hit.flow().isEndOfSpeech(new float[PocketTtsFixtures.EMB]));
        }
        try (PocketTtsClip miss = PocketTtsClip.open(new PocketTtsFixtures(9L).writeWithEosBias(-100f).toString())) {
            assertFalse(miss.flow().isEndOfSpeech(new float[PocketTtsFixtures.EMB]));
        }
    }
}
