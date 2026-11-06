package com.eonmux.neuralfree.gguf.engine.watermark;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleSupplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Parity with vLLM's watermarking goldens ({@code tests/watermarking/watermarking_goldens.json}): the generation
 * traces and detection scores reproduced here are the ones vLLM's own {@code PhiloxPRF}, Gumbel selection, repeated
 * context mask and detectors produce for the deterministic fixture. The generation fixture is the golden one:
 * {@code logits[t] = ((t * 37 + position * 17) % 131 - 56) / 16}, with a dominant token's bias added; a repeated
 * context is sampled by the ordinary argmax, as vLLM's deterministic test sampler falls back to it.
 */
public class WatermarkGoldenTest {

    private static final double[] ROUTING_UNIFORMS = {
            0.618, 0.2361, 0.8541, 0.4721, 0.0902, 0.7082, 0.3262, 0.9443,
            0.5623, 0.1803, 0.7984, 0.4164, 0.0344, 0.6525, 0.2705, 0.8885,
            0.5066, 0.1246, 0.7426, 0.3607, 0.9787, 0.5967, 0.2148, 0.8328,
            0.4508, 0.0689, 0.6869, 0.305, 0.923, 0.541, 0.1591, 0.7771,
            0.3951, 0.0132, 0.6312, 0.2492, 0.8673, 0.4853, 0.1033, 0.7214,
            0.3394, 0.9574, 0.5755, 0.1935, 0.8115, 0.4296, 0.0476, 0.6656,
            0.2837, 0.9017, 0.5197, 0.1378, 0.7558, 0.3738, 0.9919, 0.6099,
            0.2279, 0.846, 0.464, 0.082, 0.7001, 0.3181, 0.9361, 0.5542
    };

    /** The golden generation fixture. */
    private static final class Fixture {
        final int vocabulary;
        final int tokens;
        final int denominator;
        final int modulus;
        final int dominantToken;
        final int dominantBias;

        Fixture(int vocabulary, int tokens, int denominator, int modulus, int dominantToken, int dominantBias) {
            this.vocabulary = vocabulary;
            this.tokens = tokens;
            this.denominator = denominator;
            this.modulus = modulus;
            this.dominantToken = dominantToken;
            this.dominantBias = dominantBias;
        }

        static Fixture balanced() {
            return new Fixture(128, 64, 16, 131, -1, 0);
        }

        float[] logits(int position) {
            float[] logits = new float[vocabulary];
            for (int token = 0; token < vocabulary; token++) {
                int value = (token * 37 + position * 17) % modulus - 56;
                if (token == dominantToken) {
                    value += dominantBias;
                }
                logits[token] = value / (float) denominator;
            }
            return logits;
        }
    }

    /**
     * Regenerates a golden trace. A repeated context falls back to the argmax of the row, as vLLM's deterministic
     * test sampler does; the dual-key watermarker still takes its routing draw, so the golden routing uniforms are
     * consumed once per position.
     */
    private static int[] generate(WatermarkConfig config, Fixture fixture) {
        Watermarker watermarker = Watermarker.create(config);
        WatermarkState state = new WatermarkState(config);
        AtomicInteger routingAt = new AtomicInteger();
        int[] generated = new int[fixture.tokens];
        for (int position = 0; position < fixture.tokens; position++) {
            float[] logits = fixture.logits(position);
            boolean repeated = state.repeatedContext();
            DoubleSupplier routing = () -> ROUTING_UNIFORMS[routingAt.getAndIncrement() % ROUTING_UNIFORMS.length];
            int sampled = watermarker.sample(logits, state.context(), routing);
            int token = repeated ? argmax(logits) : sampled;
            generated[position] = token;
            state.accept(token);
        }
        return generated;
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    private static void assertGeneration(int[] expected, int[] actual) {
        assertEquals("generated token count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("generated token " + i, expected[i], actual[i]);
        }
    }

    private static void assertDetection(double expectedScore, double expectedPValue, int expectedScored,
                                        boolean expectedWatermarked, WatermarkDetection detection) {
        assertEquals(expectedScored, detection.numScoredTokens());
        assertEquals(expectedWatermarked, detection.isWatermarked());
        double scoreTolerance = Math.abs(expectedScore) * 1e-9 + 1e-12;
        assertEquals("score", expectedScore, detection.score(), scoreTolerance);
        double pTolerance = Math.abs(expectedPValue) * 1e-9 + 1e-300;
        assertEquals("p-value", expectedPValue, detection.pValue(), pTolerance);
    }

    @Test
    public void singleKeyGenerationAndDetectionMatchTheGoldens() {
        WatermarkConfig config = WatermarkConfig.builder(42).contextWidth(4).build();
        int[] expected = {
                46, 24, 45, 80, 14, 26, 22, 124, 21, 45, 16, 30, 5, 89, 0, 10,
                27, 70, 80, 62, 1, 11, 117, 53, 10, 45, 94, 37, 43, 22, 78, 49,
                119, 9, 22, 118, 82, 75, 32, 24, 84, 27, 122, 66, 78, 64, 124, 17,
                91, 12, 51, 15, 75, 64, 127, 45, 112, 15, 83, 114, 121, 109, 24, 77
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        assertDetection(296.27295787856264, 7.128196280874065e-61, 64, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void singleKeyRepeatedContextsAreSkippedAndScoredOnce() {
        WatermarkConfig config = WatermarkConfig.builder(0).contextWidth(1).build();
        int[] expected = {
                24, 42, 66, 26, 51, 107, 36, 25, 63, 24, 6, 48, 65, 124, 117, 85,
                45, 112, 48, 101, 19, 22, 89, 88, 63, 77, 115, 104, 65, 61, 32, 20,
                95, 98, 30, 58, 86, 50, 39, 113, 122, 62, 69, 40, 29, 92, 113, 120,
                52, 112, 108, 1, 25, 96, 3, 95, 84, 65, 76, 3, 68, 71, 81, 56
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        // 10 repeated contexts among the 64 tokens are scored once
        assertDetection(236.2768211235414, 4.532248480235615e-47, 54, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void noDeduplicationScoresEveryContextAndDetectsWithoutIt() {
        WatermarkConfig config = WatermarkConfig.builder(4294967296L).contextWidth(5)
                .deduplicateContexts(WatermarkContextScope.NONE).build();
        int[] expected = {
                21, 10, 73, 94, 19, 61, 85, 42, 95, 116, 105, 47, 58, 36, 74, 42,
                95, 70, 34, 37, 125, 29, 64, 85, 70, 73, 44, 90, 47, 107, 25, 70,
                98, 41, 1, 51, 8, 57, 99, 88, 80, 119, 55, 15, 121, 43, 124, 56,
                34, 87, 115, 65, 4, 110, 53, 77, 58, 126, 19, 72, 64, 113, 109, 123
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        WatermarkDetector detector = WatermarkDetector.create(config, 0.01f, false);
        assertDetection(263.04926919645095, 1.1027575381260964e-49, 64, true, detector.detect(generated));
    }

    @Test
    public void allScopeSkipsThePartialContextAndSearchesTheWholeHistory() {
        WatermarkConfig config = WatermarkConfig.builder(-1L).contextWidth(16)
                .deduplicateContexts(WatermarkContextScope.ALL).build();
        int[] expected = {
                46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2, 44, 125, 121, 117,
                88, 6, 97, 12, 86, 121, 0, 106, 98, 119, 101, 36, 47, 110, 25, 24,
                102, 6, 101, 37, 114, 50, 53, 127, 109, 80, 69, 5, 96, 60, 14, 42,
                38, 58, 118, 79, 4, 110, 42, 102, 66, 80, 72, 33, 78, 106, 67, 52
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        assertDetection(220.2767573421712, 6.166345567165398e-36, 64, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void theWrongDetectionKeyIsNotWatermarked() {
        WatermarkConfig generated = WatermarkConfig.builder(42).contextWidth(4).build();
        int[] generation = generate(generated, Fixture.balanced());
        WatermarkConfig detection = WatermarkConfig.builder(43).contextWidth(4).build();
        assertDetection(63.25096716585621, 0.5208447936891313, 64, false,
                WatermarkDetector.create(detection).detect(generation));
    }

    @Test
    public void theLargeVocabularyFixtureMatchesItsThresholdGolden() {
        WatermarkConfig config = WatermarkConfig.builder(42).contextWidth(4).build();
        Fixture fixture = new Fixture(16, 8, 16, 131, -1, 0);
        int[] expected = {7, 2, 13, 2, 5, 1, 11, 14};
        int[] generated = generate(config, fixture);
        assertGeneration(expected, generated);
        assertDetection(19.672172609864035, 0.0009697880157637389, 8, true,
                WatermarkDetector.create(config, 0.0015f, true).detect(generated));
    }

    @Test
    public void dualKeyGenerationAndFusedDetectionMatchTheGoldens() {
        WatermarkConfig config = WatermarkConfig.builder(0).contextWidth(4)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.1).build();
        int[] expected = {
                70, 56, 91, 90, 72, 68, 17, 78, 21, 116, 104, 62, 15, 100, 82, 13,
                88, 70, 112, 62, 79, 57, 11, 113, 88, 80, 23, 90, 79, 42, 18, 60,
                10, 73, 94, 64, 96, 71, 35, 56, 38, 93, 37, 54, 82, 127, 0, 45,
                37, 75, 19, 92, 68, 127, 28, 56, 45, 69, 8, 1, 110, 39, 127, 56
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        assertDetection(241.81126265648692, 1.234132682420224e-51, 64, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void dualKeyRoutingAndRepeatedContextsMatchTheRepetitiveGoldens() {
        WatermarkConfig config = WatermarkConfig.builder(42).contextWidth(1)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.1).build();
        Fixture fixture = new Fixture(16, 48, 16, 131, 3, 52);
        int[] expected = {
                3, 3, 6, 9, 12, 3, 3, 3, 3, 10, 13, 2, 1, 1, 3, 3,
                3, 6, 2, 5, 12, 8, 4, 14, 3, 6, 2, 5, 1, 3, 3, 3,
                10, 6, 2, 5, 1, 3, 3, 3, 6, 13, 9, 5, 1, 3, 3, 3
        };
        int[] generated = generate(config, fixture);
        assertGeneration(expected, generated);
        assertDetection(23.896456918277963, 0.002750266035191538, 13, true,
                WatermarkDetector.create(config).detect(generated));

        WatermarkConfig all = WatermarkConfig.builder(42).contextWidth(4)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.1)
                .deduplicateContexts(WatermarkContextScope.ALL).build();
        int[] expectedAll = {
                3, 3, 6, 2, 3, 1, 11, 0, 14, 10, 5, 15, 5, 3, 7, 7,
                3, 15, 9, 12, 11, 3, 14, 3, 3, 6, 2, 5, 8, 11, 3, 3,
                10, 13, 2, 12, 1, 3, 3, 3, 13, 13, 9, 12, 3, 3, 3, 3
        };
        int[] generatedAll = generate(all, fixture);
        assertGeneration(expectedAll, generatedAll);
        assertDetection(82.86886159792212, 8.656644652657153e-07, 47, true,
                WatermarkDetector.create(all).detect(generatedAll));
    }

    @Test
    public void dualKeyEqualWeightsUseTheRealShapedSurvival() {
        WatermarkConfig config = WatermarkConfig.builder(4294967296L).contextWidth(5)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.5).build();
        int[] expected = {
                53, 116, 48, 94, 47, 118, 64, 123, 106, 65, 59, 48, 4, 86, 64, 32,
                84, 116, 23, 23, 65, 15, 50, 67, 49, 69, 34, 76, 1, 95, 103, 120,
                102, 69, 125, 121, 57, 39, 127, 113, 98, 87, 9, 51, 100, 53, 46, 42,
                123, 55, 108, 100, 50, 103, 67, 80, 94, 41, 97, 117, 18, 66, 74, 38
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        assertDetection(162.0665949787407, 6.733621085489817e-36, 64, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void dualKeyUnequalWeightsUseTheContinuousSurvival() {
        WatermarkConfig config = WatermarkConfig.builder(-1L).contextWidth(4)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.4).build();
        int[] expected = {
                7, 42, 13, 55, 103, 125, 96, 85, 119, 38, 101, 108, 72, 8, 96, 120,
                116, 109, 119, 62, 26, 107, 63, 113, 102, 34, 83, 90, 121, 53, 18, 67,
                20, 12, 101, 40, 86, 89, 59, 81, 24, 20, 72, 51, 22, 117, 14, 88,
                116, 44, 100, 118, 75, 7, 10, 70, 91, 5, 33, 22, 4, 71, 31, 121
        };
        int[] generated = generate(config, Fixture.balanced());
        assertGeneration(expected, generated);
        assertDetection(177.98266424643373, 5.9713149418801004e-43, 64, true,
                WatermarkDetector.create(config).detect(generated));
    }

    @Test
    public void derivedDualKeysMatchTheGoldens() {
        assertEquals("15691511769010696906", Long.toUnsignedString(WatermarkConfig.keyA(0L)));
        assertEquals("9736202488081271922", Long.toUnsignedString(WatermarkConfig.keyB(0L)));
    }

    @Test
    public void aShortSequenceIsNotEvidenceOfAWatermark() {
        WatermarkConfig config = WatermarkConfig.builder(42).contextWidth(4).build();
        WatermarkDetection empty = WatermarkDetector.create(config).detect(new int[0]);
        assertEquals(0, empty.numScoredTokens());
        assertEquals(1.0, empty.pValue(), 0.0);
        assertFalse(empty.isWatermarked());
        WatermarkDetection shortRun = WatermarkDetector.create(config).detect(new int[]{1, 2, 3});
        assertEquals(3, shortRun.numScoredTokens());
        assertFalse(shortRun.isWatermarked());
    }

    @Test
    public void watermarkedTextIsDetectedAndAnIndependentTextIsNot() {
        WatermarkConfig config = WatermarkConfig.builder(123456789L).contextWidth(4).build();
        Watermarker watermarker = Watermarker.create(config);
        WatermarkState state = new WatermarkState(config);
        int[] watermarked = new int[256];
        int[] plain = new int[256];
        java.util.SplittableRandom random = new java.util.SplittableRandom(99);
        float[] logits = new float[64];
        for (int position = 0; position < watermarked.length; position++) {
            for (int t = 0; t < logits.length; t++) {
                logits[t] = (float) (random.nextDouble() * 2.0);
            }
            watermarked[position] = watermarker.sample(logits, state.context(), () -> 0.5);
            state.accept(watermarked[position]);
            plain[position] = random.nextInt(logits.length);
        }
        assertTrue(WatermarkDetector.create(config).detect(watermarked).isWatermarked());
        assertFalse(WatermarkDetector.create(config).detect(plain).isWatermarked());
    }
}
