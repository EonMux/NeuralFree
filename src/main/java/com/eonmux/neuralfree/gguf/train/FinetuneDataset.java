package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The training windows of one corpus, split into a training and a validation part: what {@code llama-finetune}
 * builds from {@code --file} before the first epoch.
 *
 * <p>The corpus is tokenized once, with the vocabulary's special tokens parsed and the BOS added as llama.cpp's
 * {@code common_tokenize(..., true, true)} does. Every window starts {@code nCtx / 2} tokens after the previous one,
 * so consecutive windows overlap by half a context, and a window predicts its own next token at every position but
 * the last: window tokens are {@code t[start .. start + nCtx - 2]} and their labels {@code t[start + 1 .. start +
 * nCtx - 1]}. The model therefore trains on {@code nCtx - 1} positions per window, the same positions llama.cpp
 * scores.</p>
 *
 * <p>The split is llama.cpp's: the last {@code (int) (windows * valSplit)} windows validate and the rest train,
 * truncated toward zero, so a small corpus can have no validation part at all. Immutable; the windows it hands out
 * are copies.</p>
 */
public final class FinetuneDataset {

    /** One window: the tokens the model reads and the next token of every position. */
    public static final class Window {
        private final int[] tokens;
        private final int[] labels;

        Window(int[] tokens, int[] labels) {
            this.tokens = tokens;
            this.labels = labels;
        }

        /** The input tokens, {@code nCtx - 1} of them. */
        public int[] tokens() {
            return tokens.clone();
        }

        /** The target token of every input position; the same length as {@link #tokens()}. */
        public int[] labels() {
            return labels.clone();
        }

        /** The predictions the window trains, its input length. */
        public int positions() {
            return tokens.length;
        }

        /** The input tokens, without a copy, for the trainer's own loops. */
        int[] rawTokens() {
            return tokens;
        }

        /** The target tokens, without a copy, for the trainer's own loops. */
        int[] rawLabels() {
            return labels;
        }
    }

    private final List<Window> training;
    private final List<Window> validation;
    private final int nCtx;
    private final int corpusTokens;

    private FinetuneDataset(List<Window> training, List<Window> validation, int nCtx, int corpusTokens) {
        this.training = training;
        this.validation = validation;
        this.nCtx = nCtx;
        this.corpusTokens = corpusTokens;
    }

    /**
     * The dataset of a token stream: its windows, the last {@code valSplit} of them validating.
     *
     * @param nCtx     the context a window fills, at least 3
     * @param valSplit the fraction of windows that validate, in {@code [0, 1)}
     * @throws IllegalArgumentException when the stream is shorter than one window, when no window would train, or
     *                                  when {@code valSplit} is outside {@code [0, 1)}
     */
    public static FinetuneDataset of(int[] tokens, int nCtx, double valSplit) {
        Objects.requireNonNull(tokens, "tokens");
        List<Window> windows = windows(tokens, nCtx);
        if (!(valSplit >= 0.0 && valSplit < 1.0)) {
            throw new IllegalArgumentException("--val-split " + valSplit + " is not in [0, 1)");
        }
        int nValidation = (int) (windows.size() * valSplit);
        int nTraining = windows.size() - nValidation;
        if (nTraining < 1) {
            throw new IllegalArgumentException("the corpus holds " + windows.size() + " window(s) of " + nCtx
                    + " tokens and --val-split " + valSplit + " leaves none of them for training");
        }
        List<Window> train = new ArrayList<>(windows.subList(0, nTraining));
        List<Window> validation = new ArrayList<>(windows.subList(nTraining, windows.size()));
        return new FinetuneDataset(Collections.unmodifiableList(train), Collections.unmodifiableList(validation),
                nCtx, tokens.length);
    }

    /**
     * The dataset of a corpus text: the text tokenized with the vocabulary of {@code tokenizer}, then
     * {@link #of(int[], int, double)}.
     *
     * @throws IllegalArgumentException when the text tokenizes to fewer than one window
     */
    public static FinetuneDataset ofText(String text, GGUFTokenizer tokenizer, int nCtx, double valSplit) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(tokenizer, "tokenizer");
        int[] tokens = tokenizer.encode(text, true, true);
        if (tokens.length < nCtx) {
            throw new IllegalArgumentException("the corpus tokenizes to " + tokens.length + " tokens, fewer than"
                    + " one window of " + nCtx + "; use a longer --file or a smaller -c");
        }
        return of(tokens, nCtx, valSplit);
    }

    /**
     * The windows of a token stream: every start {@code i} with {@code i + nCtx <= tokens.length}, stepping
     * {@code nCtx / 2}. A context below 3 has no next-token position and is refused.
     */
    static List<Window> windows(int[] tokens, int nCtx) {
        Objects.requireNonNull(tokens, "tokens");
        if (nCtx < 3) {
            throw new IllegalArgumentException("the context of a training window must be at least 3 tokens, not "
                    + nCtx);
        }
        List<Window> windows = new ArrayList<>();
        int stride = Math.max(1, nCtx / 2);
        for (int start = 0; start + nCtx <= tokens.length; start += stride) {
            int[] input = Arrays.copyOfRange(tokens, start, start + nCtx - 1);
            int[] label = Arrays.copyOfRange(tokens, start + 1, start + nCtx);
            windows.add(new Window(input, label));
        }
        if (windows.isEmpty()) {
            throw new IllegalArgumentException("the corpus tokenizes to " + tokens.length + " tokens, fewer than"
                    + " one window of " + nCtx);
        }
        return windows;
    }

    /** The windows one epoch trains on, in corpus order. */
    public List<Window> training() {
        return training;
    }

    /** The windows the validation pass scores, in corpus order; empty when the split left none. */
    public List<Window> validation() {
        return validation;
    }

    /** The context a window fills, as requested; a window's input is one token shorter. */
    public int nCtx() {
        return nCtx;
    }

    /** The tokens the whole corpus tokenized to. */
    public int corpusTokens() {
        return corpusTokens;
    }

    @Override
    public String toString() {
        return corpusTokens + " tokens in " + (training.size() + validation.size()) + " windows of " + nCtx
                + " (" + training.size() + " training, " + validation.size() + " validation)";
    }
}
