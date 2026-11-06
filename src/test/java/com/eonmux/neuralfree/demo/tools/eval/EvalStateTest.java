package com.eonmux.neuralfree.demo.tools.eval;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.eonmux.neuralfree.utils.json.Json;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * A run's task layout (the reference's Python shuffle and passes), the state file round trip, the
 * Wilson interval and the pass@k estimator, and the HTML escaping.
 */
public class EvalStateTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theTaskLayoutMatchesTheReferencesShuffleAndPasses() {
        EvalState state = state(3, -1);
        state.setupTasks(5, 1);
        List<EvalState.TaskRef> tasks = state.tasks();
        assertEquals(5, tasks.size());
        assertRef(tasks.get(0), 1, "gsm8k_000_001");
        assertRef(tasks.get(1), 2, "gsm8k_000_002");
        assertRef(tasks.get(2), 0, "gsm8k_000_000");
        assertRef(tasks.get(3), 2, "gsm8k_001_002");
        assertRef(tasks.get(4), 0, "gsm8k_001_000");
    }

    @Test
    public void onePassUsesTheWholeSuiteShuffled() {
        EvalState state = state(3, -1);
        state.setupTasks(null, 1234);
        List<EvalState.TaskRef> tasks = state.tasks();
        assertEquals(3, tasks.size());
        assertRef(tasks.get(0), 2, "gsm8k_000_002");
        assertRef(tasks.get(1), 0, "gsm8k_000_000");
        assertRef(tasks.get(2), 1, "gsm8k_000_001");
    }

    @Test
    public void aZeroCaseCountIsRefused() {
        try {
            state(3, -1).setupTasks(0, 1);
            fail("zero cases must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--n_cases"));
        }
    }

    @Test
    public void theWilsonIntervalBracketsTheObservedRate() {
        double[] ci = EvalState.wilsonInterval(5, 10);
        assertEquals(0.2366, ci[0], 0.0002);
        assertEquals(0.7634, ci[1], 0.0002);
        assertEquals(0.0, EvalState.wilsonInterval(0, 0)[0], 0.0);
        assertEquals(1.0, EvalState.wilsonInterval(0, 0)[1], 0.0);
    }

    @Test
    public void passAtKMatchesTheEstimator() {
        assertEquals(0.5, EvalState.passAtK(10, 5, 1), 1e-9);
        assertEquals(1.0, EvalState.passAtK(10, 5, 10), 0.0);
        assertEquals(0.0, EvalState.passAtK(5, 0, 1), 0.0);
        assertEquals(1.0, EvalState.passAtK(5, 5, 1), 0.0);
    }

    @Test
    public void theStateFileRoundTripsAndKeepsPendingCases() throws Exception {
        Path dir = tmp.newFolder("run").toPath();
        Path file = dir.resolve("state.json");
        EvalState state = new EvalState(stateDataset(3), "m",
                new EvalSampling(null, null, null, null, null, -1), file);
        state.setupTasks(3, 1);
        state.addResult(okCase(state.tasks().get(0).taskId(), 1, true));
        state.addResult(okCase(state.tasks().get(1).taskId(), 2, false));
        state.dump();

        Object parsed = Json.parse(Files.readString(file), Json.Limits.UNTRUSTED);
        Map<?, ?> root = (Map<?, ?>) parsed;
        assertEquals("gsm8k", root.get("id"));
        assertEquals(3, ((List<?>) root.get("tasks")).size());
        Map<?, ?> states = (Map<?, ?>) root.get("task_states");
        Map<?, ?> cases = (Map<?, ?>) states.get("cases");
        assertEquals(3, cases.size());
        assertEquals("pending", ((Map<?, ?>) cases.get("gsm8k_000_000")).get("status"));

        EvalState loaded = EvalState.load(file, state.dataset(), -1);
        assertEquals(3, loaded.tasks().size());
        assertEquals(2, loaded.completed());
        assertEquals(1, loaded.correct());
        assertFalse(loaded.isComplete());
        assertEquals(1, loaded.pendingTasks().size());
        assertEquals("gsm8k_000_000", loaded.pendingTasks().get(0).taskId());
        assertNotNull(loaded.caseOf("gsm8k_000_001"));
        assertTrue(Files.isRegularFile(dir.resolve("state.json.html")));
        String html = Files.readString(dir.resolve("state.json.html"), StandardCharsets.UTF_8);
        assertTrue(html.contains("gsm8k_000_001"));
        assertTrue(html.contains("prompt 1"));
    }

    @Test
    public void theSamplingParametersArePersisted() throws Exception {
        Path file = tmp.newFolder("sampling").toPath().resolve("state.json");
        EvalState state = new EvalState(stateDataset(2), "m",
                new EvalSampling(0.5, 20, 0.9, null, "low", -1), file);
        state.setupTasks(2, 1);
        state.dump();
        EvalState loaded = EvalState.load(file, state.dataset(), 64);
        assertEquals(0.5, loaded.sampling().temperature(), 0.0);
        assertEquals(Integer.valueOf(20), loaded.sampling().topK());
        assertEquals(0.9, loaded.sampling().topP(), 0.0);
        assertEquals("low", loaded.sampling().reasoningEffort());
        assertEquals(64, loaded.sampling().nPredict());
    }

    @Test
    public void aStateOfAnotherSuiteIsRefused() throws Exception {
        Path file = tmp.newFolder("other").toPath().resolve("state.json");
        EvalState state = new EvalState(stateDataset(2), "m",
                new EvalSampling(null, null, null, null, null, -1), file);
        state.setupTasks(2, 1);
        state.dump();
        EvalDataset other = new EvalDataset() {
            @Override
            public String type() {
                return "aime";
            }

            @Override
            public int size() {
                return 1;
            }

            @Override
            public EvalQuestion question(int index) {
                return new EvalQuestion("q", "p", "1");
            }
        };
        try {
            EvalState.load(file, other, -1);
            fail("a state of another suite must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("gsm8k"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("aime"));
        }
    }

    @Test
    public void aStateNamingAQuestionOutsideTheDatasetIsRefused() throws Exception {
        Path file = tmp.newFolder("out-of-range").toPath().resolve("state.json");
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("id", "gsm8k");
        root.put("model_name", "m");
        root.put("tasks", java.util.List.of("gsm8k_000_009"));
        Map<String, Object> states = new LinkedHashMap<>();
        states.put("total", 1L);
        states.put("correct", 0L);
        states.put("total_time", 0.0);
        states.put("cases", new LinkedHashMap<>());
        root.put("task_states", states);
        root.put("sampling_config", new LinkedHashMap<>());
        Files.write(file, Json.toString(root, Json.Style.PYTHON_DEFAULT)
                .getBytes(StandardCharsets.UTF_8));
        try {
            EvalState.load(file, stateDataset(2), -1);
            fail("a task outside the dataset must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("outside the gsm8k"));
        }
    }

    @Test
    public void taskIdsExposeTheirPassAndProblem() {
        assertEquals(7, EvalState.problemIndex("gsm8k_002_007"));
        assertEquals(2, EvalState.chunkIndex("gsm8k_002_007"));
        assertEquals(0, EvalState.problemIndex("notatask"));
        assertEquals(0, EvalState.chunkIndex("notatask"));
    }

    @Test
    public void htmlEscapingCoversEveryMarkupCharacter() {
        assertEquals("&lt;a&gt;&amp;&quot;&#39;", EvalState.escape("<a>&\"'"));
    }

    private EvalState state(int size, int nPredict) {
        return new EvalState(stateDataset(size), "m", new EvalSampling(null, null, null, null, null,
                nPredict), tmp.getRoot().toPath().resolve("state-" + System.nanoTime() + ".json"));
    }

    private static EvalDataset stateDataset(int size) {
        return new EvalDataset() {
            @Override
            public String type() {
                return "gsm8k";
            }

            @Override
            public int size() {
                return size;
            }

            @Override
            public EvalQuestion question(int index) {
                return new EvalQuestion("Attempted question " + index, "prompt " + index, "42");
            }
        };
    }

    private static EvalCase okCase(String taskId, int problemIdx, boolean correct) {
        Map<String, Object> log = new LinkedHashMap<>();
        log.put("pred", "42");
        log.put("grader_type", "regex");
        return new EvalCase(taskId, "prompt " + problemIdx, "42", "q", "\\boxed{42}", "42", log,
                correct, EvalCase.OK, 7L, 10.0, 100.0, null, "one", 0, problemIdx);
    }

    private static void assertRef(EvalState.TaskRef ref, int index, String taskId) {
        assertEquals(index, ref.index());
        assertEquals(taskId, ref.taskId());
    }
}
