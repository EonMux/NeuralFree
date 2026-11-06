package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Mirostat under the verifiers that read verify rows ahead of their confirmation (block, FLy): {@code mu} adapts on
 * the emitted tokens only, as llama.cpp's speculative loop adapts it once per sampled-and-accepted token, so a row
 * scored after a draft token that is then rejected leaves no trace in {@code mu}.
 *
 * <p>Mirostat v2 with {@code tau = 0.45}, {@code eta = 1} starts at {@code mu = 0.9}. A near-certain row (0.97)
 * collapses to its top token, whose renormalized surprise 0 raises {@code mu} by {@code tau}. The prefill row
 * selects {@code a} ({@code mu} 1.35). The round's draft {@code c d} meets a row 0 certain of {@code b}: the block
 * verifier reads rows 0 and 1, rejects {@code c} and emits {@code b} ({@code mu} 1.8). Row 1, read ahead under
 * {@code c}, is near-certain too; had it adapted {@code mu}, {@code mu} would be 2.25. The next plain row is
 * uniform, every surprise 2: under {@code mu} 1.8 it collapses to its first candidate {@code a} whatever the seed,
 * under 2.25 it would keep all four and draw among them.
 *
 * <p>Every verifier mode adapts {@code mu} on the tokens it emits, not on mirostat's own draws: a plain mirostat
 * sampler that accepts the emitted tokens over the same rows offers each emitted token with the probability the
 * generation captured for it (processed logprobs: the chain's distribution of each row).
 */
public class SpeculativeMirostatTest {

    private static final int A = 0;
    private static final int B = 1;
    private static final int C = 2;
    private static final int D = 3;

    private static SamplerConfig mirostat(long seed) {
        return SamplerConfig.builder().temperature(1f).mirostat(2).mirostatTau(0.45f).mirostatEta(1f).seed(seed)
                .build();
    }

    private static FakeTarget target() {
        return new FakeTarget(false)
                .add(logits(0.97, 0.01, 0.01, 0.01))
                .add(logits(0.01, 0.97, 0.01, 0.01), logits(0.01, 0.01, 0.01, 0.97),
                        logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0.25, 0.25, 0.25, 0.25));
    }

    @Test
    public void aRowReadAheadOfARejectedDraftTokenDoesNotAdaptMu() {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.BLOCK, VerifyMode.FLY}) {
            for (long seed = 0; seed < 16; seed++) {
                FakeTarget target = target();
                SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{C, D}),
                        tokenizer()).draftMax(3).verifyMode(mode).build();

                GenerationResult r = dec.generate(new int[]{A}, request(3, mirostat(seed)), (TokenCallback) null);

                int[] tokens = r.tokens();
                assertEquals(mode + " seed " + seed, 3, tokens.length);
                assertEquals(mode + " seed " + seed + ": the prefill row", A, tokens[0]);
                assertEquals(mode + " seed " + seed + ": the rejection's replacement", B, tokens[1]);
                assertEquals(mode + " seed " + seed + ": mu after the round is the emitted tokens' alone", A,
                        tokens[2]);
                assertEquals(mode + " seed " + seed + ": both drafts left memory", 2,
                        (int) target.rollbacks.get(0));
            }
        }
    }

    /**
     * Every row is {@code a 0.74, b 0.24, c 0.01, d 0.01}: under {@code tau = 1.5}, {@code eta = 0.5} mirostat v2
     * keeps {@code {a}} or {@code {a, b}} depending on {@code mu}, and {@code a} and {@code b} move {@code mu} in
     * opposite directions. The drafter always proposes {@code b b}, so the verifiers often emit a token mirostat's
     * own draw at that row did not pick; the replay accepts the emitted tokens alone.
     */
    @Test
    public void everyVerifierAdaptsMuOnTheTokensItEmits() {
        float[] row = logits(0.74, 0.24, 0.01, 0.01);
        int n = 12;
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            for (long seed = 0; seed < 16; seed++) {
                SamplerConfig config = SamplerConfig.builder().temperature(1f).mirostat(2).mirostatTau(1.5f)
                        .mirostatEta(0.5f).seed(seed).build();
                SpeculativeDecoder dec = SpeculativeDecoder.builder(new FakeTarget(false).otherwise(row),
                        new ConstantDrafter(B, B), tokenizer()).draftMax(2).verifyMode(mode).build();

                GenerationResult r = dec.generate(new int[]{A},
                        GenerationRequest.builder().maxNewTokens(n).sampler(config).logprobs(4)
                                .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build(),
                        (TokenCallback) null);

                String what = mode + " seed " + seed;
                int[] tokens = r.tokens();
                List<TokenScore[]> captures = r.logprobs();
                assertEquals(what, n, tokens.length);
                assertEquals(what, n, captures.size());
                Sampler plain = Sampler.create(config);
                for (int i = 0; i < n; i++) {
                    double p = probOf(plain.apply(row), tokens[i]);
                    String at = what + " token " + i + " of " + Arrays.toString(tokens);
                    assertTrue(at + ": the plain sampler's set holds the emitted token", p > 0);
                    assertEquals(at, tokens[i], captures.get(i)[0].tokenId());
                    assertEquals(at + ": the captured probability is the plain sampler's", Math.log(p),
                            captures.get(i)[0].logprob(), 1e-5);
                    plain.accept(tokens[i]);
                }
            }
        }
    }

    private static double probOf(Sampler.Candidates candidates, int token) {
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.id(i) == token) {
                return candidates.prob(i);
            }
        }
        return 0;
    }

    /** A drafter proposing the same tokens at every round, cut to the round's draft size. */
    private static final class ConstantDrafter implements DraftModel {
        private final int[] tokens;

        ConstantDrafter(int... tokens) {
            this.tokens = tokens;
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            return Arrays.copyOf(tokens, Math.min(n, tokens.length));
        }

        @Override public void accept(int nAccepted) {
        }

        @Override public void reset() {
        }
    }
}
