package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool perplexity --multiple-choice} over a running engine: the binary task format of perplexity.cpp
 * (header, task positions, length-prefixed strings and label arrays), the sequential and seeded selection
 * paths, and the refusals for truncated files, oversized answer sets, malformed tasks and positions beyond
 * the file.
 */
public class MultipleChoiceTest {

    private static EngineConfig config(int nCtx) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** One question with its answers and the 0/1 labels, the fields the binary format serializes. */
    private static final class TaskSpec {
        final String question;
        final List<String> answers;
        final List<Integer> labels;

        TaskSpec(String question, List<String> answers, List<Integer> labels) {
            this.question = question;
            this.answers = answers;
            this.labels = labels;
        }
    }

    private static byte[] serialize(TaskSpec... tasks) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ByteBuffer header = ByteBuffer.allocate(4 + 4 * tasks.length).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(tasks.length);
        int pos = header.capacity();
        for (TaskSpec task : tasks) {
            header.putInt(pos);
            pos += sizeOf(task);
        }
        bytes.writeBytes(header.array());
        for (TaskSpec task : tasks) {
            writeString(bytes, task.question);
            writeAnswers(bytes, task.answers, task.labels);
            writeAnswers(bytes, List.of(), List.of()); // mc2, empty
        }
        return bytes.toByteArray();
    }

    private static int sizeOf(TaskSpec task) {
        int size = 4 + task.question.getBytes(StandardCharsets.UTF_8).length;
        size += 4;
        for (String answer : task.answers) {
            size += 4 + answer.getBytes(StandardCharsets.UTF_8).length;
        }
        size += 4 * task.labels.size();
        size += 4; // empty mc2: the count only
        return size;
    }

    private static void writeString(ByteArrayOutputStream bytes, String s) {
        byte[] raw = s.getBytes(StandardCharsets.UTF_8);
        ByteBuffer len = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        len.putInt(raw.length);
        bytes.writeBytes(len.array());
        bytes.writeBytes(raw);
    }

    private static void writeAnswers(ByteArrayOutputStream bytes, List<String> answers, List<Integer> labels) {
        ByteBuffer head = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        head.putInt(answers.size());
        bytes.writeBytes(head.array());
        for (String answer : answers) {
            writeString(bytes, answer);
        }
        ByteBuffer tagBuf = ByteBuffer.allocate(4 * labels.size()).order(ByteOrder.LITTLE_ENDIAN);
        for (int label : labels) {
            tagBuf.putInt(label);
        }
        bytes.writeBytes(tagBuf.array());
    }

    private static final TaskSpec CAPITAL =
            new TaskSpec("The capital of France is", List.of("Paris", "London"), List.of(1, 0));
    private static final TaskSpec COLOR =
            new TaskSpec("The sky is", List.of("blue", "red", "green"), List.of(1, 0, 0));

    @Test
    public void theScoreRunsOverEverySerializedTask() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            MultipleChoice.score(m, 0, serialize(CAPITAL, COLOR), out, err);
            String report = out.toString();
            assertTrue(report, report.startsWith("\ntask\tacc_norm\n"));
            String[] lines = report.trim().split("\n");
            assertEquals("the header plus one line per task", 3, lines.length);
            assertTrue("the task number and the running accuracy: " + lines[1],
                    lines[1].trim().matches("1\\t\\d+\\.\\d{8}"));
            assertTrue("the task number and the running accuracy: " + lines[2],
                    lines[2].trim().matches("2\\t\\d+\\.\\d{8}"));
            assertTrue(err.toString(), err.toString().contains("holds 2 tasks"));
            assertTrue(err.toString(), err.toString().contains("final result:"));
            assertTrue(err.toString(), err.toString().contains("random chance:"));
        }
    }

    @Test
    public void theTaskLimitDrawsTheReferenceTasks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            TaskSpec third = new TaskSpec("Two plus two is", List.of("four", "five"), List.of(1, 0));
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            MultipleChoice.score(m, 2, serialize(CAPITAL, COLOR, third), out, err);
            assertEquals("the header plus the two drawn tasks", 3,
                    out.toString().trim().split("\n").length);
            assertTrue(err.toString(), err.toString().contains("selecting 2 random tasks of 3 available"));
        }
    }

    @Test
    public void aTruncatedDatafileIsRefusedByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        byte[] all = serialize(CAPITAL, COLOR);
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            IOException e = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 0, Arrays.copyOf(all, all.length - 3),
                            new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("ends inside a task"));

            IOException noTasks = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 0, new byte[] {0, 0, 0, 0}, new StringBuilder(),
                            new StringBuilder()));
            assertTrue(noTasks.getMessage(), noTasks.getMessage().contains("holds no tasks"));
        }
    }

    @Test
    public void oversizedAnswerSetsAndBadTasksAreRefusedByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            List<String> answers = new java.util.ArrayList<>();
            List<Integer> labels = new java.util.ArrayList<>();
            for (int i = 0; i < 101; i++) {
                answers.add("answer" + i);
                labels.add(i == 5 ? 1 : 0);
            }
            IOException tooMany = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 0,
                            serialize(new TaskSpec("q", answers, labels)), new StringBuilder(),
                            new StringBuilder()));
            assertTrue(tooMany.getMessage(), tooMany.getMessage().contains("more than the 100"));

            IOException noAnswers = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 0,
                            serialize(new TaskSpec("q", List.of(), List.of())), new StringBuilder(),
                            new StringBuilder()));
            assertTrue(noAnswers.getMessage(), noAnswers.getMessage().contains("empty"));
        }
    }

    @Test
    public void aTaskPositionBeyondTheFileIsRefusedByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // the seeded selection draws its first task; that task's position is the one corrupted
        int drawn = new Mt19937(1).nextFloatScaled(2);
        byte[] first = serialize(CAPITAL, COLOR);
        ByteBuffer corrupt = ByteBuffer.allocate(first.length).order(ByteOrder.LITTLE_ENDIAN);
        corrupt.put(first);
        corrupt.putInt(4 + 4 * drawn, Integer.MAX_VALUE);
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            IOException far = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 1, corrupt.array(), new StringBuilder(),
                            new StringBuilder()));
            assertTrue(far.getMessage(), far.getMessage().contains("lies outside the file"));
        }
    }

    @Test
    public void aTaskBeyondTheContextIsRefusedWithItsCounts() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        String[] words = "this question keeps going and going and going and going and going".split(" ");
        StringBuilder question = new StringBuilder();
        for (int i = 0; i < 4; i++) {
            for (String word : words) {
                question.append(word).append(' ');
            }
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(32))) {
            TaskSpec longOne =
                    new TaskSpec(question.toString(), List.of("one", "two"), List.of(1, 0));
            IOException e = assertThrows(IOException.class,
                    () -> MultipleChoice.score(m, 0, serialize(longOne), new StringBuilder(),
                            new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("does not fit in the context window"));
        }
    }
}
