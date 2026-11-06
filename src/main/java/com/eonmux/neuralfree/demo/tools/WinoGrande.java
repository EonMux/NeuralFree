package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool perplexity --winogrande}: the WinoGrande score of perplexity.cpp's {@code winogrande_score}
 * over a CSV of {@code index, sentence with _, choice 1, choice 2, answer} rows. Every task is scored by the
 * partial-evaluation rule of the reference: the average token log-probability of the sentence tail after the
 * choice, over {@code first + choice + second}, compared between the two choices. Tasks are drawn with
 * llama.cpp's seeded {@code (int)(scale * rng() * size)} pick ({@link Mt19937#nextFloatScaled}).
 */
final class WinoGrande {

    /** perplexity.cpp {@code k_min_trailing_ctx}: tails this long score the choice through its own tokens. */
    private static final int MIN_TRAILING_CTX = 3;

    private WinoGrande() {
    }

    /** One CSV task and, once tokenized, the sequences and base lengths its scoring needs. */
    private static final class Task {
        final String first;
        final String second;
        final String[] choices;
        final int answer;
        int[][] seqTokens;
        int commonPrefix;
        int nBase1;
        int nBase2;

        Task(String first, String second, String choice1, String choice2, int answer) {
            this.first = first;
            this.second = second;
            this.choices = new String[] {choice1, choice2};
            this.answer = answer;
        }
    }

    /**
     * Scores the tasks of the CSV text and prints the per-task lines; {@code taskLimit} is
     * {@code --winogrande-tasks}, 0 for all.
     *
     * @throws IOException for a task that does not fit in the context
     */
    static void score(GGUFModel model, int taskLimit, String text, Appendable out, Appendable err)
            throws IOException {
        List<Task> data = loadFromCsv(text, err);
        if (data.isEmpty()) {
            throw new IOException("the WinoGrande datafile holds no usable tasks");
        }
        err.append("nftool perplexity: winogrande: loaded " + data.size() + " tasks from the datafile.\n");
        if (taskLimit > 0 && taskLimit < data.size()) {
            err.append("nftool perplexity: winogrande: selecting " + taskLimit + " random tasks.\n");
            Mt19937 rng = new Mt19937(1);
            List<Task> selected = new ArrayList<>(taskLimit);
            int[] aux = new int[data.size()];
            for (int i = 0; i < aux.length; i++) {
                aux[i] = i;
            }
            for (int i = 0; i < taskLimit; i++) {
                int j = rng.nextFloatScaled(aux.length);
                selected.add(data.get(aux[j]));
                aux[j] = aux[aux.length - 1];
                aux = java.util.Arrays.copyOf(aux, aux.length - 1);
            }
            data = selected;
        }

        GGUFTokenizer tok = model.tokenizer();
        err.append("nftool perplexity: winogrande: tokenizing the selected tasks.\n");
        for (Task task : data) {
            task.seqTokens = new int[][] {
                    tok.encode(task.first + task.choices[0] + task.second),
                    tok.encode(task.first + task.choices[1] + task.second)};
            task.commonPrefix = 0;
            int shared = Math.min(task.seqTokens[0].length, task.seqTokens[1].length);
            while (task.commonPrefix < shared
                    && task.seqTokens[0][task.commonPrefix] == task.seqTokens[1][task.commonPrefix]) {
                task.commonPrefix++;
            }
            task.nBase1 = tok.encode(task.first + task.choices[0]).length;
            task.nBase2 = tok.encode(task.first + task.choices[1]).length;
        }

        int nCtx = model.forwardBackend().memory().plan().nCtx();
        ChoiceScorer scorer = new ChoiceScorer(model.forwardBackend());
        err.append("nftool perplexity: winogrande: calculating the score over the selected tasks.\n");

        int nCorrect = 0;
        int nDone = 0;
        for (int i = 0; i < data.size(); i++) {
            Task task = data.get(i);
            int required = task.commonPrefix;
            for (int[] seq : task.seqTokens) {
                required += seq.length - task.commonPrefix;
            }
            if (required > nCtx) {
                throw new IOException("task " + i + " does not fit in the context window (requires "
                        + required + " tokens of the " + nCtx + " the context holds)");
            }
            boolean skipChoice = task.seqTokens[0].length - task.commonPrefix > MIN_TRAILING_CTX
                    && task.seqTokens[1].length - task.commonPrefix > MIN_TRAILING_CTX;
            double score1 = sideScore(scorer, task, i, 0, skipChoice ? task.nBase1 : task.commonPrefix);
            double score2 = sideScore(scorer, task, i, 1, skipChoice ? task.nBase2 : task.commonPrefix);
            int result = score1 > score2 ? 1 : 2;
            if (result == task.answer) {
                nCorrect++;
            }
            nDone++;
            out.append(String.format(Locale.ROOT, "%d\t%.4f\t%10.6f  %10.6f  %d  %d\n",
                    i + 1, 100.0 * nCorrect / nDone, score1, score2, result, task.answer));
        }
        out.append("\n");

        if (nDone < 100) {
            return;
        }
        double p = 1.0 * nCorrect / nDone;
        double sigma = 100.0 * Math.sqrt(p * (1 - p) / (nDone - 1));
        err.append(String.format(Locale.ROOT, "nftool perplexity: winogrande: final score (%d tasks): "
                + "%.4f +/- %.4f\n", nDone, 100.0 * p, sigma));
    }

    /**
     * The average log-probability of one choice's tail (perplexity.cpp's per-side loop): rows from the one
     * before the base to the tail's end minus the unevaluated final token, over the tail length; the
     * reference's empty loop averages its zero sum.
     */
    private static double sideScore(ChoiceScorer scorer, Task task, int taskIdx, int side, int nBase)
            throws IOException {
        int[] seq = task.seqTokens[side];
        if (nBase < 1) {
            throw new IOException("task " + taskIdx + " tokenizes to an empty base for choice "
                    + (side + 1) + "; the CSV row cannot be scored");
        }
        int last = seq.length - nBase > 1 ? 1 : 0;
        int lastRow = seq.length - 2 - last;
        if (lastRow >= nBase - 1) {
            return scorer.sumLogSoftmax(seq, nBase - 1, lastRow) / (seq.length - nBase - last);
        }
        return 0.0 / (seq.length - nBase - last);
    }

    /** The CSV loader of perplexity.cpp's {@code load_winogrande_from_csv}, malformed rows logged and skipped. */
    private static List<Task> loadFromCsv(String text, Appendable err) throws IOException {
        List<Task> result = new ArrayList<>();
        List<String> lines = CliText.getlineComplete(text);
        for (String line : lines) {
            int[] commaPos = new int[4];
            int ipos = 0;
            boolean quoteOpen = false;
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (!quoteOpen) {
                    if (c == ',') {
                        commaPos[ipos++] = i;
                        if (ipos == 4) {
                            break;
                        }
                    } else if (c == '"') {
                        quoteOpen = true;
                    }
                } else {
                    if (c == '"') {
                        quoteOpen = false;
                    }
                }
            }
            if (ipos != 4) {
                err.append("nftool perplexity: winogrande: failed to find comma separators in <" + line + ">\n");
                continue;
            }
            String sentence = line.charAt(commaPos[0] + 1) == '"'
                    ? line.substring(commaPos[0] + 2, commaPos[1] - 1)
                    : line.substring(commaPos[0] + 1, commaPos[1]);
            String choice1 = line.substring(commaPos[1] + 1, commaPos[2]);
            String choice2 = line.substring(commaPos[2] + 1, commaPos[3]);
            String answer = line.substring(commaPos[3] + 1);
            int where = sentence.indexOf('_');
            if (where < 0) {
                err.append("nftool perplexity: winogrande: no _ in <" + sentence + ">\n");
                continue;
            }
            int iAnswer;
            try {
                iAnswer = Integer.parseInt(answer.trim());
            } catch (NumberFormatException e) {
                err.append("nftool perplexity: winogrande: failed to parse answer <" + answer + ">\n");
                continue;
            }
            if (iAnswer < 1 || iAnswer > 2) {
                err.append("nftool perplexity: winogrande: failed to parse answer <" + answer + ">\n");
                continue;
            }
            result.add(new Task(sentence.substring(0, where), sentence.substring(where + 1),
                    choice1, choice2, iAnswer));
        }
        return result;
    }
}
