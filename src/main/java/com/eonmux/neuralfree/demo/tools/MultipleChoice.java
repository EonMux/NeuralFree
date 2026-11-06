package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool perplexity --multiple-choice}: the multiple-choice score of perplexity.cpp's
 * {@code multiple_choice_score} over a datafile in its binary task format — the task count, the task
 * offsets, then per task the question and two answer sets ({@code mc1} with single labels,
 * {@code mc2} parsed and ignored) as length-prefixed little-endian strings. Every answer is scored by the
 * mean token log-probability of {@code question + " " + answer} and the answer whose label is 1 must win.
 * Tasks are drawn with llama.cpp's seeded {@code (int)(scale * rng() * size)} pick
 * ({@link Mt19937#nextFloatScaled}).
 */
final class MultipleChoice {

    /** The largest answer count of one answer set ({@code multiple_choice_score} reads at most 100). */
    private static final int MAX_ANSWERS = 100;

    private MultipleChoice() {
    }

    /** One task of the datafile and, once prepared, the tokenized sequences of its answers. */
    private static final class Task {
        String question;
        String[] answers;
        int[] labels;
        int[][] seqTokens;
        int commonPrefix;

        int requiredTokens() {
            int required = commonPrefix;
            for (int[] seq : seqTokens) {
                required += seq.length - commonPrefix;
            }
            return required;
        }
    }

    /**
     * Scores the tasks of the datafile bytes and prints the per-task lines; {@code taskLimit} is
     * {@code --multiple-choice-tasks}, 0 for all.
     *
     * @throws IOException for a malformed or truncated datafile, a malformed task or a task that does not
     *                     fit in the context
     */
    static void score(GGUFModel model, int taskLimit, byte[] data, Appendable out, Appendable err)
            throws IOException {
        Cursor in = new Cursor(data);
        int nTask = in.u32();
        if (nTask == 0) {
            throw new IOException("the multiple-choice datafile holds no tasks");
        }
        long[] taskPos = new long[nTask];
        for (int i = 0; i < nTask; i++) {
            taskPos[i] = in.u32() & 0xFFFFFFFFL;
        }
        err.append("nftool perplexity: multiple-choice: the datafile holds " + nTask + " tasks.\n");

        List<Task> tasks;
        boolean usedAll = taskLimit == 0 || taskLimit >= nTask;
        if (usedAll) {
            tasks = new ArrayList<>(nTask);
            for (int i = 0; i < nTask; i++) {
                tasks.add(readTask(in, i));
            }
        } else {
            err.append("nftool perplexity: multiple-choice: selecting " + taskLimit + " random tasks of "
                    + nTask + " available.\n");
            Mt19937 rng = new Mt19937(1);
            int[] aux = new int[nTask];
            for (int i = 0; i < nTask; i++) {
                aux[i] = i;
            }
            tasks = new ArrayList<>(taskLimit);
            for (int i = 0; i < taskLimit; i++) {
                int j = rng.nextFloatScaled(aux.length);
                int idx = aux[j];
                aux[j] = aux[aux.length - 1];
                aux = Arrays.copyOf(aux, aux.length - 1);
                in.seek(taskPos[idx]);
                tasks.add(readTask(in, idx));
            }
        }

        prepare(model.tokenizer(), tasks);
        int nCtx = model.forwardBackend().memory().plan().nCtx();
        ChoiceScorer scorer = new ChoiceScorer(model.forwardBackend());
        err.append("nftool perplexity: multiple-choice: calculating the score over " + tasks.size()
                + " tasks.\n");
        out.append("\ntask\tacc_norm\n");

        int nDone = 0;
        int nCorrect = 0;
        int nTotAnswers = 0;
        for (int i = 0; i < tasks.size(); i++) {
            Task task = tasks.get(i);
            if (task.requiredTokens() > nCtx) {
                throw new IOException("task " + i + " does not fit in the context window (requires "
                        + task.requiredTokens() + " tokens of the " + nCtx + " the context holds)");
            }
            double bestLogprob = 0;
            int bestIdx = 0;
            for (int s = 0; s < task.seqTokens.length; s++) {
                int[] seq = task.seqTokens[s];
                double logprob = scorer.sumLogSoftmax(seq, task.commonPrefix - 1, seq.length - 2)
                        / (seq.length - task.commonPrefix);
                if (s == 0 || logprob > bestLogprob) {
                    bestLogprob = logprob;
                    bestIdx = s;
                }
            }
            nTotAnswers += task.seqTokens.length;
            if (task.labels[bestIdx] == 1) {
                nCorrect++;
            }
            nDone++;
            out.append(String.format(Locale.ROOT, "%d\t%.8f\n", nDone, 100.0 * nCorrect / nDone));
        }

        if (nDone < 100 && usedAll && taskLimit != 0 && taskLimit < nTask) {
            return;
        }
        double p = 1.0 * nCorrect / nDone;
        double sigma = Math.sqrt(p * (1 - p) / (nDone - 1));
        err.append(String.format(Locale.ROOT, "nftool perplexity: multiple-choice: final result: %.4f +/- %.4f\n",
                100.0 * p, 100.0 * sigma));
        p = 1.0 * nDone / nTotAnswers;
        sigma = Math.sqrt(p * (1 - p) / (nDone - 1));
        err.append(String.format(Locale.ROOT, "nftool perplexity: multiple-choice: random chance: %.4f +/- %.4f\n",
                100.0 * p, 100.0 * sigma));
    }

    /** Tokenizes every answer after the question and finds the answers' common prefix. */
    private static void prepare(GGUFTokenizer tok, List<Task> tasks) throws IOException {
        for (Task task : tasks) {
            if (task.question.isEmpty() || task.answers.length == 0) {
                throw new IOException("found a bad task with an empty question and/or answers");
            }
            task.seqTokens = new int[task.answers.length][];
            for (int s = 0; s < task.answers.length; s++) {
                if (task.answers[s].isEmpty()) {
                    throw new IOException("found an empty answer");
                }
                task.seqTokens[s] = tok.encode(task.question + " " + task.answers[s]);
            }
            int shared = Integer.MAX_VALUE;
            for (int[] seq : task.seqTokens) {
                shared = Math.min(shared, seq.length);
            }
            task.commonPrefix = 0;
            outer:
            while (task.commonPrefix < shared) {
                int token = task.seqTokens[0][task.commonPrefix];
                for (int s = 1; s < task.seqTokens.length; s++) {
                    if (task.seqTokens[s][task.commonPrefix] != token) {
                        break outer;
                    }
                }
                task.commonPrefix++;
            }
        }
    }

    /** Reads one task at the cursor ({@code multiple_choice_task::deserialize}). */
    private static Task readTask(Cursor in, int idx) throws IOException {
        Task task = new Task();
        task.question = in.string();
        int n = in.u32();
        if (n > MAX_ANSWERS) {
            throw new IOException("task " + idx + " holds " + n + " answers, more than the "
                    + MAX_ANSWERS + " the format holds");
        }
        task.answers = new String[n];
        for (int i = 0; i < n; i++) {
            task.answers[i] = in.string();
        }
        task.labels = new int[n];
        for (int i = 0; i < n; i++) {
            task.labels[i] = in.i32();
        }
        // mc2 parses after mc1 and is ignored, exactly as the reference reads it
        int n2 = in.u32();
        if (n2 > MAX_ANSWERS) {
            throw new IOException("task " + idx + " holds " + n2 + " mc2 answers, more than the "
                    + MAX_ANSWERS + " the format holds");
        }
        for (int i = 0; i < n2; i++) {
            in.string();
        }
        in.skip(n2 * 4);
        return task;
    }

    /** The little-endian reader of the datafile bytes. */
    private static final class Cursor {
        private final byte[] data;
        private int pos;

        Cursor(byte[] data) {
            this.data = data;
        }

        void seek(long offset) throws IOException {
            if (offset < 0 || offset > data.length) {
                throw new IOException("the datafile task position " + offset + " lies outside the file");
            }
            pos = (int) offset;
        }

        int u32() throws IOException {
            need(4);
            int v = (data[pos] & 0xFF) | ((data[pos + 1] & 0xFF) << 8)
                    | ((data[pos + 2] & 0xFF) << 16) | ((data[pos + 3] & 0xFF) << 24);
            pos += 4;
            return v;
        }

        int i32() throws IOException {
            return u32();
        }

        String string() throws IOException {
            int len = u32();
            if (len < 0) {
                throw new IOException("the datafile names a string of " + (len & 0xFFFFFFFFL) + " bytes");
            }
            need(len);
            String s = new String(data, pos, len, StandardCharsets.UTF_8);
            pos += len;
            return s;
        }

        void skip(int n) throws IOException {
            need(n);
            pos += n;
        }

        private void need(int n) throws IOException {
            if (n < 0 || pos > data.length - n) {
                throw new IOException("the multiple-choice datafile ends inside a task");
            }
        }
    }
}
