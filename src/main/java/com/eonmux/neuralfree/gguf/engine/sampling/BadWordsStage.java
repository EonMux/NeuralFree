package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The bad-words stage (vLLM's {@code bad_words}): every banned phrase is tokenized once, and while the accepted
 * history ends with a phrase's proper prefix, the token that would continue the phrase takes a {@code -inf}
 * logit — so no continuation of the generated text can spell a banned phrase. The stage keeps the longest
 * phrase's tail of the accepted history in a ring. A masking stage ({@link #masks()}): the chain applies it from
 * the constraint head, before the configured chain, so a truncating stage cannot cut the phrase-continuing
 * tokens out of a set that holds nothing else. A Java-API custom stage ({@code Sampler.create(config,
 * bannedTokens, extra, vocab)}) that always applies.
 */
public final class BadWordsStage extends SamplerStage implements TokenGate {

    private final int[][] phrases;
    private final SamplerVocab vocab;
    private final int keep;
    private final int[] tail;
    private int tailLen;

    /**
     * A stage over the tokenized banned phrases; a phrase that tokenizes to nothing is refused (the request
     * would ban nothing silently).
     */
    public BadWordsStage(List<int[]> tokenizedPhrases, SamplerVocab vocab) {
        Objects.requireNonNull(tokenizedPhrases, "tokenizedPhrases");
        if (tokenizedPhrases.isEmpty()) {
            throw new IllegalArgumentException("bad words needs at least one phrase");
        }
        this.vocab = Objects.requireNonNull(vocab, "vocab");
        this.phrases = new int[tokenizedPhrases.size()][];
        int longest = 0;
        for (int i = 0; i < tokenizedPhrases.size(); i++) {
            int[] phrase = tokenizedPhrases.get(i);
            if (phrase == null || phrase.length == 0) {
                throw new IllegalArgumentException("bad word " + i + " tokenizes to no tokens");
            }
            this.phrases[i] = phrase.clone();
            longest = Math.max(longest, phrase.length);
        }
        this.keep = longest - 1; // the longest history tail a next token can continue into a phrase
        this.tail = new int[keep];
    }

    /** Tokenizes {@code phrases} over the vocab (the request's strings; an empty phrase is refused). */
    public static List<int[]> tokenize(List<String> phrases, SamplerVocab vocab) {
        Objects.requireNonNull(phrases, "phrases");
        Objects.requireNonNull(vocab, "vocab");
        List<int[]> out = new ArrayList<>(phrases.size());
        for (String phrase : phrases) {
            int[] ids = vocab.tokenize(Objects.requireNonNull(phrase, "phrase"));
            if (ids.length == 0) {
                throw new IllegalArgumentException("bad word '" + phrase + "' tokenizes to no tokens");
            }
            out.add(ids);
        }
        return out;
    }

    /** The stage over the request's phrases, or null when the request bans none. */
    public static BadWordsStage of(List<String> phrases, SamplerVocab vocab) {
        if (phrases == null || phrases.isEmpty()) {
            return null;
        }
        return new BadWordsStage(tokenize(phrases, vocab), vocab);
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        for (int i = 0; i < candidates.size(); i++) {
            int token = candidates.id(i);
            for (int[] phrase : phrases) {
                if (continuesPhrase(token, phrase)) {
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                    break;
                }
            }
        }
    }

    /** Whether {@code token} completes none of the phrases after the history tail. */
    @Override
    public boolean keeps(int token) {
        for (int[] phrase : phrases) {
            if (continuesPhrase(token, phrase)) {
                return false;
            }
        }
        return true;
    }

    /** Whether {@code token} extends the history tail into the whole phrase. */
    private boolean continuesPhrase(int token, int[] phrase) {
        if (tailLen < phrase.length - 1) {
            return false;
        }
        // the phrase's first tailLen tokens must match the tail's last tailLen entries
        int offset = tailLen - (phrase.length - 1);
        for (int k = 0; k < phrase.length - 1; k++) {
            if (tail[offset + k] != phrase[k]) {
                return false;
            }
        }
        return phrase[phrase.length - 1] == token;
    }

    @Override
    public void accept(int token) {
        if (keep == 0) {
            return;
        }
        if (tailLen < keep) {
            tail[tailLen++] = token;
            return;
        }
        System.arraycopy(tail, 1, tail, 0, keep - 1);
        tail[keep - 1] = token;
    }

    @Override
    public boolean modifies() {
        return true;
    }

    @Override
    public boolean masks() {
        return true;
    }

    @Override
    public void reset() {
        tailLen = 0;
    }

    /** The history tail a next token can continue into a phrase. */
    @Override
    public Object saveState() {
        return Arrays.copyOf(tail, tailLen);
    }

    @Override
    public void restoreState(Object state) {
        if (!(state instanceof int[]) || ((int[]) state).length > keep) {
            throw new IllegalArgumentException("not a bad-words stage state: " + state);
        }
        int[] saved = (int[]) state;
        System.arraycopy(saved, 0, tail, 0, saved.length);
        tailLen = saved.length;
    }
}
