package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * Block verification of the draft as one unit (Sun, Ro, Beirami, Suresh: Optimal Block-Level Draft Verification,
 * arXiv:2403.10444, Algorithm 3; vLLM {@code rejection_sample_method} {@code block}). With {@code b_i = p_i(d_i)}
 * the target's probability of draft token {@code i} in the position's verify row and {@code qd_i = q_i(d_i)} its
 * draft probability, the joint masses of a draft prefix of {@code m} tokens are
 * {@code Mb(x^m) = prod_{i<m} b_i} and {@code Ms(x^m) = prod_{i<m} qd_i}, and the coupling verifies
 * <ol>
 *   <li>the whole block once: a uniform accepts it with probability
 *       {@code min(1, Mb(x^k)/Ms(x^k))};</li>
 *   <li>otherwise backward over the prefix lengths {@code m = k-1 .. 0}, each uniform accepting the prefix of
 *       {@code m} tokens with probability {@code min(1, p_remain(x^m)/p_rej(x^m))} of the continuation masses
 *       {@code p_remain(x^m) = sum_x (Mb(x^m)·p_m(x) − Ms(x^m)·q_m(x))_+} and
 *       {@code p_rej(x^m) = sum_x (Ms(x^m)·q_m(x) − Mb(x^m)·p_m(x))_+}: both continuations share the verify row's
 *       {@code p_m} and the drafter's {@code q_m}; the process always ends, since {@code p_remain(x^0)}
 *       equals {@code p_rej(x^0)};</li>
 *   <li>after the first acceptance of a prefix of {@code m} tokens the round emits the replacement from the
 *       residual {@code norm(max(0, Mb(x^m)·p_m(x) − Ms(x^m)·q_m(x))_+)}; a fully confirmed block takes the bonus
 *       token from the last row's {@code p} as usual.</li>
 * </ol>
 * The joint target mass of a draft prefix is the product of its rows' entries, so every quantity follows from the
 * k + 1 verify rows and the {@code q} vectors the verification already has. Lossless (the emissions of
 * {@code (x^m, replacement)} follow the target's joint distribution) and never worse in expected accepted length
 * than the token-level {@link RejectionVerifier}. One uniform is drawn per backward step visited in prefix-length
 * order (the closing {@code m = 0} step accepts without one, being a tie by construction), plus the opening draw
 * and, after an acceptance of a partial prefix, one for the replacement.
 *
 * <p>A drafter that proposed a token with a zero or negative {@code q} entry makes the block impossible under its
 * own distribution and fails the round ({@link IllegalStateException}), as vLLM's {@code draft_prob > 0} guard. A
 * greedy target chain degenerates to {@link VerifyMode#MATCH}'s rule: the one-hot {@code p} rows make every ratio
 * and the residual deterministic.
 */
public final class BlockVerifier implements Verifier {

    @Override
    public void verify(Round round) {
        int k = round.draftCount();
        Distribution[] ps = new Distribution[k + 1];
        Distribution[] qs = new Distribution[k + 1];
        // bprod[m] = Mb(x^m), qprod[m] = Ms(x^m): the joint masses over a prefix of m draft tokens.
        double[] bprod = new double[k + 1];
        double[] qprod = new double[k + 1];
        bprod[0] = 1.0;
        qprod[0] = 1.0;
        for (int i = 0; i < k; i++) {
            ps[i] = round.target(i);
            qs[i] = round.draft(i);
            int d = round.draftToken(i);
            double qdi = qs[i].probOf(d);
            if (qdi <= 0.0) {
                throw new IllegalStateException("the drafter drafted token " + d + " at position " + i
                        + " with proposal probability " + qdi + "; the block is impossible under its own"
                        + " distribution");
            }
            bprod[i + 1] = bprod[i] * ps[i].probOf(d);
            qprod[i + 1] = qprod[i] * qdi;
        }

        double eta = round.uniform();
        if (accepts(eta, bprod[k], qprod[k])) {
            emit(round, k, ps, qs, bprod, qprod);
            return;
        }
        for (int m = k - 1; m >= 0; m--) {
            // The backward pass always ends at m = 0: p_remain(x^0) equals p_rej(x^0), so the paper's last step
            // accepts unconditionally; its float sums may differ in the last ulp, which the m == 0 short-circuit
            // absorbs.
            if (m == 0 || accepts(round.uniform(), backwardRemain(ps[m], qs[m], bprod[m], qprod[m]),
                    backwardRej(ps[m], qs[m], bprod[m], qprod[m]))) {
                emit(round, m, ps, qs, bprod, qprod);
                return;
            }
        }
        throw new IllegalStateException("block verification failed to end its backward pass");
    }

    private static void emit(Round round, int accepted, Distribution[] ps, Distribution[] qs, double[] bprod,
                             double[] qprod) {
        for (int i = 0; i < accepted; i++) {
            if (round.confirm(i) != null) {
                return;
            }
        }
        if (accepted == ps.length - 1) {
            // the whole block was confirmed: the bonus token is a draw from the last row's processed target
            // distribution.
            round.emit(round.target(accepted).selectedToken());
            return;
        }
        round.emit(Verifier.residual(round, ps[accepted], qs[accepted], bprod[accepted], qprod[accepted]));
    }

    /** {@code eta <= min(1, remain/rej)}: a zero rejected mass accepts (nothing can be resampled). */
    private static boolean accepts(double eta, double remain, double rej) {
        if (rej <= 0.0) {
            return true;
        }
        return remain > 0.0 && eta * rej <= remain;
    }

    private static double backwardRemain(Distribution p, Distribution q, double mb, double ms) {
        double total = 0.0;
        for (int i = 0; i < p.size(); i++) {
            double w = mb * p.prob(i) - ms * q.probOf(p.token(i));
            if (w > 0.0) {
                total += w;
            }
        }
        return total;
    }

    private static double backwardRej(Distribution p, Distribution q, double mb, double ms) {
        // the rejected mass also sits on the drafter's support where the target has none.
        double total = 0.0;
        for (int i = 0; i < p.size(); i++) {
            double w = ms * q.probOf(p.token(i)) - mb * p.prob(i);
            if (w > 0.0) {
                total += w;
            }
        }
        for (int i = 0; i < q.size(); i++) {
            int token = q.token(i);
            if (p.indexOf(token) >= 0) {
                continue;
            }
            double w = ms * q.prob(i) - mb * p.probOf(token);
            if (w > 0.0) {
                total += w;
            }
        }
        return total;
    }

    @Override
    public String toString() {
        return "BlockVerifier";
    }
}
