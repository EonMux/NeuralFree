package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool perplexity --hellaswag}: the HellaSwag {@code acc_norm} of perplexity.cpp's
 * {@code hellaswag_score} over a datafile of 6 lines per task (the context line, the gold ending index and
 * the four endings). Tasks are drawn with llama.cpp's seeded draw over the shrinking pool
 * ({@link Mt19937#nextBounded}), every ending is scored by the mean token log-probability of
 * {@code context + " " + ending}, and the per-task line prints the running accuracy with its Wilson score
 * interval.
 */
final class HellaSwag {

    /** The 95% two-sided normal quantile of perplexity.cpp's {@code za}. */
    private static final double ZA = 1.95996398454;

    private HellaSwag() {
    }

    /** One task: the tokenized sequences of its four endings and the shared prefix they all begin with. */
    private static final class Task {
        final int goldEndingIdx;
        final int[][] seqTokens;
        final int commonPrefix;

        Task(int goldEndingIdx, int[][] seqTokens, int commonPrefix) {
            this.goldEndingIdx = goldEndingIdx;
            this.seqTokens = seqTokens;
            this.commonPrefix = commonPrefix;
        }
    }

    /**
     * Scores the tasks of the datafile text and prints the per-task lines; {@code taskLimit} is
     * {@code --hellaswag-tasks}.
     *
     * @throws IOException for a malformed datafile, an unparsable gold ending or a task longer than the context
     */
    static void score(GGUFModel model, int taskLimit, String text, Appendable out, Appendable err)
            throws IOException {
        List<String> lines = CliText.getlineAll(text);
        if (lines.size() % 6 != 0) {
            throw new IOException("the number of lines in the datafile, " + lines.size()
                    + ", is not a multiple of 6");
        }
        int totalTasks = lines.size() / 6;
        err.append("nftool perplexity: hellaswag: loaded " + totalTasks + " tasks from the datafile.\n");
        int taskCount = Math.min(taskLimit, totalTasks);

        GGUFTokenizer tok = model.tokenizer();
        Mt19937 rng = new Mt19937(1);
        List<Task> tasks = new ArrayList<>(taskCount);
        for (int i = 0; i < taskCount; i++) {
            int idx = (int) rng.nextBounded(lines.size() / 6);
            tasks.add(readTask(lines, idx * 6, tok));
            lines.subList(idx * 6, idx * 6 + 6).clear();
        }
        err.append("nftool perplexity: hellaswag: selected " + taskCount + " randomized tasks.\n");

        int nCtx = model.forwardBackend().memory().plan().nCtx();
        ChoiceScorer scorer = new ChoiceScorer(model.forwardBackend());
        err.append("nftool perplexity: hellaswag: calculating the score over the selected tasks.\n");
        out.append("\ntask\tacc_norm\t95% confidence interval\n");

        double acc = 0.0;
        for (int i = 0; i < tasks.size(); i++) {
            Task task = tasks.get(i);
            int required = task.commonPrefix;
            for (int[] seq : task.seqTokens) {
                required += seq.length - task.commonPrefix;
            }
            if (required > nCtx) {
                throw new IOException("task " + i + " does not fit in the context window (requires "
                        + required + " tokens of the " + nCtx + " the context holds)");
            }
            double bestLogprob = 0;
            int bestIdx = 0;
            for (int s = 0; s < 4; s++) {
                int[] seq = task.seqTokens[s];
                // the shared prefix's last row predicts the first differing token, the ending rows the rest
                double logprob = scorer.sumLogSoftmax(seq, task.commonPrefix - 1, seq.length - 2)
                        / (seq.length - task.commonPrefix);
                if (s == 0 || logprob > bestLogprob) {
                    bestLogprob = logprob;
                    bestIdx = s;
                }
            }
            if (bestIdx == task.goldEndingIdx) {
                acc += 1.0;
            }
            double freq = acc / (i + 1);
            double z = ZA * ZA / (i + 1);
            double cnf = z * Math.sqrt((i + 1) * (4.0 * freq * (1 - freq) + z)) / (ZA + ZA);
            double a = (freq + z * 0.5 - cnf) / (1.0 + z);
            double b = (freq + z * 0.5 + cnf) / (1.0 + z);
            out.append(String.format(Locale.ROOT, "%d\t%3.8f%%\t[%3.4f%%, %3.4f%%]\n",
                    i + 1, freq * 100.0, a * 100.0, b * 100.0));
        }
    }

    /** Reads the 6 lines of one task, tokenizing every ending after the context. */
    private static Task readTask(List<String> lines, int base, GGUFTokenizer tok) throws IOException {
        String context = lines.get(base);
        int gold;
        try {
            gold = Integer.parseInt(lines.get(base + 1).trim());
        } catch (NumberFormatException e) {
            throw new IOException("the datafile line " + (base + 2) + " is not a gold ending index: '"
                    + lines.get(base + 1) + "'");
        }
        if (gold < 0 || gold > 3) {
            throw new IOException("the datafile line " + (base + 2) + " names the gold ending " + gold
                    + ", not one of 0 to 3");
        }
        int[][] seqTokens = new int[4][];
        int commonPrefix = 0;
        for (int s = 0; s < 4; s++) {
            seqTokens[s] = tok.encode(context + " " + lines.get(base + 2 + s));
        }
        int shared = Math.min(Math.min(seqTokens[0].length, seqTokens[1].length),
                Math.min(seqTokens[2].length, seqTokens[3].length));
        while (commonPrefix < shared && seqTokens[0][commonPrefix] == seqTokens[1][commonPrefix]
                && seqTokens[0][commonPrefix] == seqTokens[2][commonPrefix]
                && seqTokens[0][commonPrefix] == seqTokens[3][commonPrefix]) {
            commonPrefix++;
        }
        return new Task(gold, seqTokens, commonPrefix);
    }
}
