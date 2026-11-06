package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The token-level intersection (TLI, vLLM's {@code VocabMapping}) between a draft model's vocabulary and the
 * vocabulary of the target it drafts for, so the two need not share token ids: the token texts both tokenizers
 * carry, matched after the leading space marker of a word-initial piece ({@code Ġ} in byte-level BPE, {@code ▁} in
 * SentencePiece) is normalized to an ASCII space.
 *
 * <p>{@link #toDraft} feeds a target token to the draft model (a token outside the intersection becomes the draft's
 * unknown token), {@link #toTarget} translates a draft token back to its target id ({@code -1} when the draft token
 * is not in the intersection, so the drafter stops rather than proposing a token the target never emits), and
 * {@link #constrain} masks every draft logit outside the intersection to {@code -inf}, which is what keeps a greedy
 * draft — and with it rejection sampling — lossless: the draft can only propose tokens the target can compare.
 *
 * <p>{@link #identity} is the mapping of two vocabularies that share token ids (llama.cpp
 * {@code common_speculative_are_compatible}): it passes ids through unchanged and masks nothing. Immutable.
 */
public final class DraftVocabMapping {

    /** Word-initial space marker of byte-level BPE vocabularies (GPT-2 {@code Ġ}, U+0120). */
    static final String BPE_SPACE_PREFIX = "Ġ";
    /** Word-initial space marker of SentencePiece vocabularies ({@code ▁}, U+2581). */
    static final String SPM_SPACE_PREFIX = "▁";

    /** The markers tried for a tokenizer whose own space marker the probe cannot observe. */
    private static final String[] BOTH_SPACE_PREFIXES = {BPE_SPACE_PREFIX, SPM_SPACE_PREFIX};

    private final int targetVocabSize;
    private final int draftVocabSize;
    /** By draft id, the target id of an intersected token, or -1; null for the identity mapping. */
    private final int[] draftToTarget;
    /** By target id, the draft id of an intersected token, or -1; null for the identity mapping. */
    private final int[] targetToDraft;
    /** By draft id, whether the token is in the intersection; null for the identity mapping. */
    private final boolean[] draftable;
    private final int draftUnk;
    private final int intersectionSize;

    private DraftVocabMapping(int targetVocabSize, int draftVocabSize, int[] draftToTarget, int[] targetToDraft,
                              boolean[] draftable, int draftUnk, int intersectionSize) {
        this.targetVocabSize = targetVocabSize;
        this.draftVocabSize = draftVocabSize;
        this.draftToTarget = draftToTarget;
        this.targetToDraft = targetToDraft;
        this.draftable = draftable;
        this.draftUnk = draftUnk;
        this.intersectionSize = intersectionSize;
    }

    /**
     * The identity mapping of vocabularies that share token ids.
     *
     * @param targetVocabSize the target's vocabulary size (its logits width)
     * @param draftVocabSize  the draft model's vocabulary size
     * @throws IllegalArgumentException when either size is not positive
     */
    public static DraftVocabMapping identity(int targetVocabSize, int draftVocabSize) {
        requireSizes(targetVocabSize, draftVocabSize);
        return new DraftVocabMapping(targetVocabSize, draftVocabSize, null, null, null, -1,
                Math.min(targetVocabSize, draftVocabSize));
    }

    /**
     * Builds the token-level intersection of two vocabularies. Target tokens outside it map to the draft's unknown
     * token, or to its end-of-sequence token when the draft vocabulary has no unknown one.
     *
     * @param target          the target model's tokenizer
     * @param targetVocabSize the target's vocabulary size (its logits width)
     * @param draft           the draft model's tokenizer
     * @param draftVocabSize  the draft model's vocabulary size (its logits width)
     * @throws IllegalArgumentException when a size is not positive, the draft tokenizer has neither an unknown nor
     *                                  an end-of-sequence token, or the vocabularies share no token text
     */
    public static DraftVocabMapping of(GGUFTokenizer target, int targetVocabSize, GGUFTokenizer draft,
                                       int draftVocabSize) {
        Objects.requireNonNull(target, "target tokenizer");
        Objects.requireNonNull(draft, "draft tokenizer");
        requireSizes(targetVocabSize, draftVocabSize);
        int draftUnk = unknownOrEos(draft, "the draft tokenizer");
        Map<String, Integer> targetTokens = index(target, targetVocabSize, spacePrefixes(target));
        Map<String, Integer> draftTokens = index(draft, draftVocabSize, spacePrefixes(draft));

        int[] draftToTarget = new int[draftVocabSize];
        int[] targetToDraft = new int[targetVocabSize];
        boolean[] draftable = new boolean[draftVocabSize];
        Arrays.fill(draftToTarget, -1);
        Arrays.fill(targetToDraft, -1);
        int intersection = 0;
        for (Map.Entry<String, Integer> entry : draftTokens.entrySet()) {
            Integer targetId = targetTokens.get(entry.getKey());
            if (targetId == null) {
                continue;
            }
            int draftId = entry.getValue();
            draftToTarget[draftId] = targetId;
            targetToDraft[targetId] = draftId;
            draftable[draftId] = true;
            intersection++;
        }
        if (intersection == 0) {
            throw new IllegalArgumentException("the draft tokenizer and the target tokenizer share no token text:"
                    + " the token-level intersection cannot draft");
        }
        return new DraftVocabMapping(targetVocabSize, draftVocabSize, draftToTarget, targetToDraft, draftable,
                draftUnk, intersection);
    }

    private static void requireSizes(int targetVocabSize, int draftVocabSize) {
        if (targetVocabSize <= 0) {
            throw new IllegalArgumentException("target vocabulary size must be positive, got " + targetVocabSize);
        }
        if (draftVocabSize <= 0) {
            throw new IllegalArgumentException("draft vocabulary size must be positive, got " + draftVocabSize);
        }
    }

    /** The unknown token id of {@code tokenizer}, or its end-of-sequence id when it has no unknown token. */
    private static int unknownOrEos(GGUFTokenizer tokenizer, String role) {
        int unk = tokenizer.unkId();
        if (unk >= 0 && unk < tokenizer.vocabSize()) {
            return unk;
        }
        int eos = tokenizer.eosId();
        if (eos >= 0 && eos < tokenizer.vocabSize()) {
            return eos;
        }
        throw new IllegalArgumentException(role + " has neither an unknown nor an end-of-sequence token to map"
                + " tokens outside the intersection to");
    }

    /**
     * The token text to id index of a vocabulary, normalized and with the first id of each text kept (vLLM's
     * {@code target_normalized}).
     */
    private static Map<String, Integer> index(GGUFTokenizer tokenizer, int vocabSize, String[] prefixes) {
        Map<String, Integer> out = new HashMap<>();
        int n = Math.min(vocabSize, tokenizer.vocabSize());
        for (int id = 0; id < n; id++) {
            String text = tokenizer.getToken(id);
            if (text.isEmpty()) {
                continue;
            }
            out.putIfAbsent(normalize(text, prefixes), id);
        }
        return out;
    }

    /**
     * The markers a tokenizer uses for a word-initial space, probed by tokenizing a literal space before a letter
     * (vLLM's detection), with both families' markers when the probe cannot tell.
     */
    private static String[] spacePrefixes(GGUFTokenizer tokenizer) {
        try {
            int[] ids = tokenizer.encode(" a", false);
            if (ids.length > 0) {
                String first = tokenizer.getToken(ids[0]);
                if (first.length() > 1 && first.endsWith("a") && first.charAt(0) != ' '
                        && first.charAt(0) != '\t') {
                    return new String[]{first.substring(0, first.length() - 1)};
                }
            }
        } catch (RuntimeException e) {
            // a vocabulary that cannot encode the probe keeps the both-family fallback
        }
        return BOTH_SPACE_PREFIXES;
    }

    /** {@code text} with a leading space marker of {@code prefixes} turned into an ASCII space. */
    private static String normalize(String text, String[] prefixes) {
        for (String prefix : prefixes) {
            if (text.startsWith(prefix)) {
                return " " + text.substring(prefix.length());
            }
        }
        return text;
    }

    /** The target's vocabulary size (its logits width). */
    public int targetVocabSize() {
        return targetVocabSize;
    }

    /** The draft model's vocabulary size (its logits width). */
    public int draftVocabSize() {
        return draftVocabSize;
    }

    /** How many token texts the two vocabularies share. */
    public int intersectionSize() {
        return intersectionSize;
    }

    /** Whether token ids pass through unchanged (the vocabularies already share them). */
    public boolean isIdentity() {
        return draftToTarget == null;
    }

    /**
     * The draft-model token for target token {@code targetToken}: its intersected draft id, or the draft's unknown
     * token when the text is not in the intersection.
     */
    public int toDraft(int targetToken) {
        if (draftToTarget == null) {
            return targetToken;
        }
        if (targetToken < 0 || targetToken >= targetVocabSize || targetToDraft[targetToken] < 0) {
            return draftUnk;
        }
        return targetToDraft[targetToken];
    }

    /**
     * The target token for draft token {@code draftToken}, or -1 when the draft token is not in the intersection
     * ({@link #constrain} masks such logits, so a drafter stops rather than proposing a token the target never
     * emits).
     */
    public int toTarget(int draftToken) {
        if (draftToken < 0) {
            return -1;
        }
        if (draftToTarget == null) {
            return draftToken < targetVocabSize ? draftToken : -1;
        }
        return draftToken >= draftVocabSize || !draftable[draftToken] ? -1 : draftToTarget[draftToken];
    }

    /**
     * Whether {@code targetToken} can start a draft: every target token of an intersection maps to a draft token,
     * while an identity mapping needs the id to exist in the draft vocabulary too.
     */
    public boolean canStartDraft(int targetToken) {
        if (targetToken < 0) {
            return false;
        }
        if (draftToTarget == null) {
            return targetToken < targetVocabSize && targetToken < draftVocabSize;
        }
        return targetToken < targetVocabSize;
    }

    /**
     * Masks every draft logit outside the intersection to {@code -inf}, so a selection can only pick an intersected
     * token. A no-op for the identity mapping.
     */
    public void constrain(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (draftToTarget == null) {
            return;
        }
        for (int i = 0; i < logits.length; i++) {
            if (i >= draftVocabSize || !draftable[i]) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    @Override
    public String toString() {
        if (draftToTarget == null) {
            return "DraftVocabMapping{identity, target=" + targetVocabSize + ", draft=" + draftVocabSize + '}';
        }
        return "DraftVocabMapping{target=" + targetVocabSize + ", draft=" + draftVocabSize + ", intersection="
                + intersectionSize + ", draftUnk=" + draftUnk + '}';
    }
}
