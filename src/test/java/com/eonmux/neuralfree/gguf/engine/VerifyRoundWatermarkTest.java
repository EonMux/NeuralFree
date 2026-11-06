package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.RejectionVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.watermark.GumbelWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkContextScope;
import com.eonmux.neuralfree.gguf.engine.watermark.Watermarker;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;

/**
 * The targeting leg of speculative watermarking: a rejected position's replacement and a fully confirmed round's
 * bonus are the target key's draws (vLLM's watermarked resample kernel), while confirmed draft tokens stay the
 * drafts'.
 */
public class VerifyRoundWatermarkTest {

    private static final WatermarkConfig WATERMARK = WatermarkConfig.builder(77L)
            .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL)
            .contextWidth(2)
            .deduplicateContexts(WatermarkContextScope.NONE)
            .build();

    private static Sampler target() {
        SamplerConfig config = SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).typicalP(1f)
                .seed(11L).watermark(WATERMARK).build();
        Sampler sampler = Sampler.createForSpeculativeTarget(config, new int[0], (Map<Integer, Float>) null,
                (SamplerVocab) null);
        sampler.acceptPrompt(new int[]{0, 1, 2, 3}, 0, 4);
        sampler.accept(2);
        return sampler;
    }

    private static float[] logits(double... probs) {
        float[] row = new float[probs.length];
        for (int i = 0; i < probs.length; i++) {
            row[i] = probs[i] <= 0.0 ? Float.NEGATIVE_INFINITY : (float) Math.log(probs[i]);
        }
        return row;
    }

    private static final class Recording implements VerifyRound.Emitter {
        final List<Integer> tokens = new ArrayList<>();

        @Override public boolean cancelled() {
            return false;
        }

        @Override public StopReason offer(int token, float[] row, Verifier.Distribution processed) {
            tokens.add(token);
            return null;
        }
    }

    private static VerifyRound round(Sampler sampler, long seed, int[] draft, float[][] rows,
                                     Recording emitter) {
        return new VerifyRound(sampler, emitter, draft, rows, 0, new SplittableRandom(seed), null);
    }

    @Test
    public void aRejectedPositionsReplacementIsTheTargetKeysResidualDraw() {
        Sampler sampler = target();
        int[] context = sampler.watermarkContext();
        assertEquals(2, context.length);
        // the draft token (2) has no target mass, so the round rejects at position 0
        float[][] rows = {logits(0.5, 0.5, 0.0, 0.0), logits(0.25, 0.25, 0.25, 0.25)};
        Recording emitter = new Recording();
        VerifyRound round = round(sampler, 11L, new int[]{2}, rows, emitter);
        RejectionVerifier verifier = new RejectionVerifier();
        verifier.verify(round);
        round.finish(verifier);

        assertEquals(1, emitter.tokens.size());
        Watermarker keyB = new GumbelWatermarker(WatermarkConfig.keyB(77L), 2);
        float[] residual = logits(0.5, 0.5);
        int expected = keyB.sample(new int[]{0, 1}, residual, 2, context, () -> 0.5);
        assertEquals("the replacement carries the target key", expected, emitter.tokens.get(0).intValue());
        assertEquals(0, round.accepted());
        assertEquals(1, round.emitted());
    }

    @Test
    public void aConfirmedRoundsBonusIsTheTargetKeysDraw() {
        long seed = 0;
        while (seed < 10_000 && new SplittableRandom(seed).nextDouble() > 0.25) {
            seed++;
        }
        Sampler sampler = target();
        int[] context = sampler.watermarkContext();
        float[][] rows = {logits(0.25, 0.25, 0.25, 0.25), logits(0.1, 0.4, 0.3, 0.2)};
        Recording emitter = new Recording();
        VerifyRound round = round(sampler, seed, new int[]{0}, rows, emitter);
        RejectionVerifier verifier = new RejectionVerifier();
        verifier.verify(round);
        round.finish(verifier);

        assertEquals(2, emitter.tokens.size());
        assertEquals("the draft token is confirmed as drawn", 0, emitter.tokens.get(0).intValue());
        // the bonus row's context includes the confirmed draft token
        int[] bonusContext = {context[1], 0};
        Watermarker keyB = new GumbelWatermarker(WatermarkConfig.keyB(77L), 2);
        int expected = keyB.sample(rows[1], bonusContext, () -> 0.5);
        assertEquals("the bonus carries the target key", expected, emitter.tokens.get(1).intValue());
        assertEquals(1, round.accepted());
        assertEquals(2, round.emitted());
    }

    @Test
    public void aTargetRoleSamplerNeverRoutesBetweenTheKeys() {
        // the round's own draws are key B's: an accepted draft with an ordinary q is confirmed, and no dual-key
        // routing uniform is consumed by the target sampler
        Sampler sampler = target();
        int[] reference = sampler.watermarkContext();
        Watermarker keyB = new GumbelWatermarker(WatermarkConfig.keyB(77L), 2);
        Sampler.Candidates candidates = sampler.apply(logits(0.2, 0.5, 0.2, 0.1));
        assertEquals(keyB.sample(logits(0.2, 0.5, 0.2, 0.1), reference, () -> 0.5),
                candidates.selectedId());
    }
}
