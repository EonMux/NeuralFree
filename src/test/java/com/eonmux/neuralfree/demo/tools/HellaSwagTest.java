package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool perplexity --hellaswag} over a running engine: the 6-line task file, the seeded selection
 * over the shrinking pool, the per-task accuracy line with its Wilson interval, and the refusals for a file
 * that is not a multiple of 6 lines, an unparsable gold ending and a task beyond the context.
 */
public class HellaSwagTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config(int nCtx) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static String tasks(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append("He said\n");
            sb.append(i % 4).append('\n');
            sb.append("yes today\n");
            sb.append("no today\n");
            sb.append("yes tomorrow\n");
            sb.append("no tomorrow\n");
        }
        return sb.toString();
    }

    @Test
    public void theScoreRunsOverEveryTaskAndPrintsTheRunningAccuracy() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            HellaSwag.score(m, 400, tasks(3), out, err);
            String report = out.toString();
            assertTrue(report, report.startsWith("\ntask\tacc_norm\t95% confidence interval\n"));
            assertEquals("one line per selected task", 3, report.split("\n").length - 2);
            String first = report.split("\n")[2];
            String[] fields = first.split("\t");
            assertEquals("the task number leads", "1", fields[0]);
            assertTrue("the accuracy prints as a percent: " + first, fields[1].matches("\\d+\\.\\d+%"));
            assertTrue("the Wilson interval follows in brackets: " + first,
                    fields[2].matches("\\[\\d+\\.\\d+%, \\d+\\.\\d+%\\]"));
            assertTrue(err.toString(), err.toString().contains("loaded 3 tasks"));
            assertTrue(err.toString(), err.toString().contains("selected 3 randomized tasks"));
        }
    }

    @Test
    public void theTaskLimitSelectsAFixedNumberOfTasks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            StringBuilder out = new StringBuilder();
            HellaSwag.score(m, 2, tasks(5), out, new StringBuilder());
            assertEquals(2, out.toString().split("\n").length - 2);
        }
    }

    @Test
    public void aFileThatIsNotSixLineTasksIsRefused() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            IOException e = assertThrows(IOException.class,
                    () -> HellaSwag.score(m, 400, "He said\n0\na\nb\nc\n", new StringBuilder(),
                            new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("not a multiple of 6"));
        }
    }

    @Test
    public void anUnparsableGoldEndingIsRefusedWithItsLine() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            String bad = "He said\nnot a number\na\nb\nc\nd\n";
            IOException e = assertThrows(IOException.class,
                    () -> HellaSwag.score(m, 400, bad, new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("not a gold ending index"));
        }
    }

    @Test
    public void aTaskBeyondTheContextIsRefusedWithItsCounts() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            StringBuilder sb = new StringBuilder("A very long context sentence that keeps going and going.\n0\n");
            for (int s = 0; s < 4; s++) {
                sb.append("one two three four five six seven eight nine ten\n");
            }
            IOException e = assertThrows(IOException.class,
                    () -> HellaSwag.score(m, 400, sb.toString(), new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("does not fit in the context window"));
            assertTrue(e.getMessage(), e.getMessage().contains("tokens of the 64"));
        }
    }

    @Test
    public void theDatafileTextOfAFileReachesTheScorer() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path data = tmp.newFile("hs.txt").toPath();
        java.nio.file.Files.write(data, tasks(2).getBytes(StandardCharsets.UTF_8));
        try (GGUFModel m = new GGUFModel(model.toString(), config(64))) {
            String text = new String(java.nio.file.Files.readAllBytes(data), StandardCharsets.UTF_8);
            StringBuilder out = new StringBuilder();
            HellaSwag.score(m, 400, text, out, new StringBuilder());
            assertEquals(2, out.toString().split("\n").length - 2);
        }
    }
}
