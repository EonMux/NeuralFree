package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * vLLM {@code standard} rejection sampling ({@code v1/sample/rejection_sampler.py:50}, the sequential form of
 * Leviathan et al.): per draft position {@code i} the draft token {@code d} is confirmed iff its uniform
 * {@code u <= min(1, p(d)/q(d))}, {@code p} the processed target distribution of the position's verify row and
 * {@code q} the drafter's proposal distribution. A rejection ends the round with the replacement drawn from
 * the residual {@code norm(max(0, p − q))}; a fully confirmed block adds the bonus token drawn from the last row's
 * {@code p} (the chain's own draw of that row). Lossless: every emitted token is a draw from the processed target
 * distribution at its position, provided {@code q} is the true proposal distribution; one-hot {@code q} (drafters
 * without probabilities) stays lossless with the confirmation probability {@code p(d)}. A greedy target chain
 * degenerates: one-hot {@code p} accepts only the argmax, exactly {@link VerifyMode#MATCH}'s rule.
 *
 * <p>Uniforms are drawn in position order, then one for the replacement or bonus. A drafter that proposed a token
 * with a zero {@code q} entry is a drafter fault ({@link IllegalStateException}), as vLLM's {@code draft_prob > 0}
 * guard.
 */
public final class RejectionVerifier implements Verifier {

    @Override
    public void verify(Round round) {
        int k = round.draftCount();
        for (int i = 0; i < k; i++) {
            if (round.stopped()) {
                return;
            }
            Distribution p = round.target(i);
            Distribution q = round.draft(i);
            int d = round.draftToken(i);
            double pd = p.probOf(d);
            double qd = q.probOf(d);
            if (qd <= 0.0) {
                throw new IllegalStateException("the drafter drafted token " + d + " at position " + i
                        + " with proposal probability 0");
            }
            double u = round.uniform();
            if (pd > 0.0 && u * qd <= pd) {
                // min(1, p/q): accept the draft token as the target's own.
                if (round.confirm(i) != null) {
                    return;
                }
                continue;
            }
            if (round.emit(Verifier.residual(round, p, q, 1.0, 1.0)) != null) {
                return;
            }
            return;
        }
        if (round.emit(round.target(k).selectedToken()) != null) {
            return;
        }
    }

    @Override
    public String toString() {
        return "RejectionVerifier";
    }
}
