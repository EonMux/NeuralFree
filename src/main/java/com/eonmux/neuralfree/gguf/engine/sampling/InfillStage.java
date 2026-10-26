package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's infill sampler (llama_sampler_infill_apply, llama.cpp src/llama-sampler.cpp:4087, the FIM-fill bias of
 * the code-completion runs): a softmax, then an end-of-generation-aware filter that stops the fill from running away.
 * Uses the vocabulary only: the end-of-generation ids ({@link SamplerVocab#isEog} - the FIM_PAD/REP/SEP tokens included)
 * and the token pieces (tokens whose pieces share a prefix merge into the higher-probability one), so token
 * alternatives like {@code ab} against {@code abc} do not both eat mass.
 *
 * <p>The ports' thresholds are llama.cpp's constants: when {@code 3 * sum(eog probabilities) * size > sum(text
 * probabilities)} the surviving set is the end-of-generation candidates alone; otherwise candidates under the mass
 * threshold are dropped, once at a probability of {@code 0.2} and again at {@code 1/(kept non-EOG + 1)}, with a
 * re-normalization between; when no non-end-of-generation token survives, the set collapses to the single
 * token_eot (or token_eos) candidate at logit 1.0, exactly as llama.cpp leaves it for the dist stage.
 */
public final class InfillStage extends SamplerStage {

    private final SamplerVocab vocab;

    public InfillStage(SamplerVocab vocab) {
        if (vocab == null) {
            throw new IllegalArgumentException(
                    "the infill stage needs the model's vocabulary (SamplerVocab) to resolve its end-of-generation"
                            + " and FIM tokens");
        }
        this.vocab = vocab;
    }

    @Override
    public StageKind kind() {
        return StageKind.INFILL;
    }

    @Override
    public void apply(Candidates candidates) {
        candidates.softmaxSorted();

        float pTxtSum = 0.0f;
        float pEogSum = 0.0f;
        for (int i = 0; i < candidates.size(); i++) {
            if (vocab.isEog(candidates.id(i))) {
                pEogSum += candidates.prob(i);
            } else {
                pTxtSum += candidates.prob(i);
            }
        }

        if (3 * pEogSum * candidates.size() > pTxtSum) {
            // keep just the EOG tokens
            int kept = 0;
            float pSum = 0.0f;
            for (int i = 0; i < candidates.size(); i++) {
                if (vocab.isEog(candidates.id(i))) {
                    pSum += candidates.prob(i);
                    candidates.move(i, kept++);
                }
            }
            candidates.truncate(kept);
            for (int i = 0; i < candidates.size(); i++) {
                candidates.setProb(i, candidates.prob(i) / pSum);
            }
            return;
        }

        // combine tokens with common prefix
        for (int i0 = 0; i0 < candidates.size(); i0++) {
            for (int i1 = 0; i1 < candidates.size(); i1++) {
                if (candidates.logit(i0) == Float.NEGATIVE_INFINITY) {
                    break;
                }
                if (i0 == i1 || candidates.logit(i1) == Float.NEGATIVE_INFINITY) {
                    continue;
                }
                byte[] piece0 = vocab.piece(candidates.id(i0));
                byte[] piece1 = vocab.piece(candidates.id(i1));

                // token i0 is a prefix of token i1
                if (piece0.length > 0 && piece0.length <= piece1.length && startsWith(piece1, piece0)) {
                    int dst = i0;
                    int src = i1;

                    // merge into the token with higher probability
                    if (candidates.prob(i1) > candidates.prob(i0)) {
                        dst = i1;
                        src = i0;
                    }
                    candidates.setProb(dst, candidates.prob(dst) + candidates.prob(src));
                    candidates.setLogit(src, Float.NEGATIVE_INFINITY);
                    candidates.setProb(src, 0.0f);
                }
            }
        }

        int nNonEog = 0;
        int sizeOrg = candidates.size();
        float pSum = 0.0f;
        float thold = 0.2f;
        int kept = 0;
        for (int i = 0; i < sizeOrg; i++) {
            boolean isEog = vocab.isEog(candidates.id(i));
            if (candidates.prob(i) < thold && !isEog) {
                continue;
            }
            if (!isEog) {
                nNonEog++;
            }
            pSum += candidates.prob(i);
            candidates.move(i, kept++);
        }
        candidates.truncate(kept);

        // if no non-EOG tokens are left -> reduce the set to the single EOT token
        if (nNonEog == 0) {
            int eot = vocab.eotToken();
            if (eot < 0) {
                eot = vocab.eosToken();
            }
            if (eot < 0) {
                throw new IllegalStateException("no non-end-of-generation candidate survived and the vocabulary has no"
                        + " end-of-text token");
            }
            candidates.replaceCandidate(0, eot, 1.0f);
            return;
        }

        // normalize probs
        for (int i = 0; i < candidates.size(); i++) {
            candidates.setProb(i, candidates.prob(i) / pSum);
        }

        sizeOrg = candidates.size();
        pSum = 0.0f;
        thold = 1.0f / (nNonEog + 1);
        kept = 0;
        for (int i = 0; i < sizeOrg; i++) {
            boolean isEog = vocab.isEog(candidates.id(i));
            if (candidates.prob(i) < thold && !isEog) {
                continue;
            }
            pSum += candidates.prob(i);
            candidates.move(i, kept++);
        }
        candidates.truncate(kept);

        // normalize probs
        for (int i = 0; i < candidates.size(); i++) {
            candidates.setProb(i, candidates.prob(i) / pSum);
        }
    }

    private static boolean startsWith(byte[] piece, byte[] prefix) {
        for (int i = 0; i < prefix.length; i++) {
            if (piece[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }
}
