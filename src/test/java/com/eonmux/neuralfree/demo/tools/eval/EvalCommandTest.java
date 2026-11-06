package com.eonmux.neuralfree.demo.tools.eval;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

import com.eonmux.neuralfree.utils.json.Json;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The command line: the reference's flag names and count checks, the refusals by name, and whole
 * runs over real HTTP servers (two servers sharing the queue, a resume of failed cases, and the LLM
 * grader on the same endpoint).
 */
public class EvalCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void parseSplitsServersThreadsNamesAndKeys() throws Exception {
        EvalOptions opts = EvalCommand.parse(new String[]{
                "--server", "http://one:1,http://two:2", "--server-name", "first,second",
                "--threads", "2,3", "--api-key", "k1,k2", "--dataset", "gsm8k",
                "--grader-type", "regex", "--n_cases", "5", "--seed", "7",
                "--dataset-source", "/tmp/rows.jsonl", "--temperature", "0.3",
                "--top-k", "40", "--grader-model", "ignored"});
        assertEquals(2, opts.servers().size());
        assertEquals("first", opts.servers().get(0).name());
        assertEquals(2, opts.servers().get(0).threads());
        assertEquals("second", opts.servers().get(1).name());
        assertEquals(3, opts.servers().get(1).threads());
        assertEquals("Bearer k1", opts.servers().get(0).headers(null).get("Authorization"));
        assertEquals("Bearer k2", opts.servers().get(1).headers(null).get("Authorization"));
        assertEquals("gsm8k", opts.dataset());
        assertEquals("regex", opts.graderType());
        assertEquals(Integer.valueOf(5), opts.nCases());
        assertEquals(7L, opts.seed());
        assertEquals("/tmp/rows.jsonl", opts.datasetSource());
        assertEquals(0.3, opts.sampling().temperature(), 0.0);
        assertEquals(Integer.valueOf(40), opts.sampling().topK());
        assertEquals(opts.servers().get(0).url(), opts.graderServer());
    }

    @Test
    public void underscoresAreAcceptedWhereTheReferenceUsesThem() throws Exception {
        EvalOptions opts = EvalCommand.parse(new String[]{
                "--n_cases", "4", "--grader_type", "regex", "--top_k", "5", "--retry_delay", "0",
                "--reasoning_effort", "low", "--n_predict", "64"});
        assertEquals(Integer.valueOf(4), opts.nCases());
        assertEquals(Integer.valueOf(5), opts.sampling().topK());
        assertEquals("low", opts.sampling().reasoningEffort());
        assertEquals(64, opts.sampling().nPredict());
        assertEquals(0.0, opts.retryDelay(), 0.0);
    }

    @Test
    public void aSingleKeyCoversEveryServerAndTheGraderReusesItsServersKey() throws Exception {
        EvalOptions opts = EvalCommand.parse(new String[]{
                "--server", "http://one:1,http://two:2", "--threads", "1,1", "--api-key", "shared",
                "--grader-type", "llm", "--model", "m", "--grader-server", "http://two:2"});
        assertEquals("Bearer shared", opts.servers().get(1).headers(null).get("Authorization"));
        assertEquals("shared", opts.graderApiKey());
    }

    @Test
    public void mismatchedServerListsAreRefusedByName() {
        assertUsage(new String[]{"--server", "http://one:1,http://two:2", "--threads", "1",
                "--grader-type", "regex"}, "--threads");
        assertUsage(new String[]{"--server", "http://one:1,http://two:2", "--threads", "1,1",
                "--server-name", "one", "--grader-type", "regex"}, "--server-name");
        assertUsage(new String[]{"--server", "http://one:1,http://two:2", "--threads", "1,1",
                "--api-key", "a,b,c", "--grader-type", "regex"}, "--api-key");
    }

    @Test
    public void theExecGraderIsRefusedByNameAsACodeSuiteFeature() {
        assertUsage(new String[]{"--grader-type", "exec", "--model", "m"}, "exec");
        assertUsage(new String[]{"--grader-type", "exec", "--model", "m"}, "code suites");
    }

    @Test
    public void theLlmGraderNeedsAModel() {
        assertUsage(new String[]{"--grader-type", "llm"}, "--grader-model");
    }

    @Test
    public void unknownOptionsAndBadCountsAreRefused() {
        assertUsage(new String[]{"--grader-type", "regex", "--agentic-max-turns", "3"}, "unknown option");
        assertUsage(new String[]{"--grader-type", "regex", "--n_cases", "0"}, "--n_cases");
        assertUsage(new String[]{"--grader-type", "regex", "--threads", "0"}, "--threads");
        assertUsage(new String[]{"--grader-type", "regex", "stray"}, "stray");
    }

    @Test
    public void helpPrintsTheSurfaceAndReturnsNoOptions() throws Exception {
        PrintStream saved = System.out;
        ByteArrayOutputStream printed = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(printed, true, "UTF-8"));
            assertNull(EvalCommand.parse(new String[]{"--help"}));
        } finally {
            System.setOut(saved);
        }
        String usage = printed.toString("UTF-8");
        assertTrue(usage.contains("nftool eval"));
        assertTrue(usage.contains("--grader-type"));
        assertTrue(usage.contains("gsm8k"));
    }

    @Test
    public void aRunSpreadsOverTwoServersAndSavesItsVerdicts() throws Exception {
        File rows = rows("{\"question\":\"What is 2 + 3?\",\"answer\":\"#### 42\"}\n");
        try (FakeEvalServer one = new FakeEvalServer(); FakeEvalServer two = new FakeEvalServer()) {
            CountDownLatch both = new CountDownLatch(2);
            one.lobby = both;
            two.lobby = both;
            Path state = tmp.newFolder("run").toPath().resolve("state.json");
            EvalOptions opts = EvalCommand.parse(new String[]{
                    "--server", one.baseUrl() + "," + two.baseUrl(),
                    "--server-name", "one,two", "--threads", "1,1",
                    "--dataset", "gsm8k", "--dataset-source", rows.getAbsolutePath(),
                    "--n_cases", "2", "--grader-type", "regex", "--model", "m",
                    "--output", state.toString(), "--retry-delay", "0"});
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            EvalCommand.execute(opts, out, err);

            assertTrue(out.toString(), out.toString().contains("Results: 2/2 correct (100.0%)"));
            assertTrue(out.toString(), out.toString().contains("Servers (2):"));
            assertTrue(out.toString(), out.toString().contains("Eval state dumped to " + state));
            assertTrue("the first server took a task", one.chatCalls() >= 1);
            assertTrue("the second server took a task", two.chatCalls() >= 1);
            assertEquals(2, one.chatCalls() + two.chatCalls());

            Map<?, ?> root = (Map<?, ?>) Json.parse(Files.readString(state), Json.Limits.UNTRUSTED);
            assertEquals("gsm8k", root.get("id"));
            assertEquals("m", root.get("model_name"));
            Map<?, ?> cases = (Map<?, ?>) ((Map<?, ?>) root.get("task_states")).get("cases");
            assertEquals(2, cases.size());
            for (Object value : cases.values()) {
                Map<?, ?> c = (Map<?, ?>) value;
                assertEquals("ok", c.get("status"));
                assertEquals(Boolean.TRUE, c.get("correct"));
                assertEquals("42", c.get("answer"));
                assertTrue("one".equals(c.get("server_name")) || "two".equals(c.get("server_name")));
                Map<?, ?> log = (Map<?, ?>) c.get("grader_log");
                assertEquals("regex", log.get("grader_type"));
                assertEquals("\\b(\\d+)\\b", log.get("pattern"));
                assertTrue(log.get("pred") instanceof String);
            }
            assertTrue(Files.isRegularFile(state.resolveSibling("state.json.html")));
            assertEquals(2, ((Number) ((Map<?, ?>) root.get("task_states")).get("correct")).intValue());
        }
    }

    @Test
    public void resumeRetriesTheCasesAFormerRunCouldNotAnswer() throws Exception {
        File rows = rows("{\"question\":\"What is 2 + 3?\",\"answer\":\"#### 42\"}\n");
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.failuresBeforeSuccess = 100;
            Path state = tmp.newFolder("resume").toPath().resolve("state.json");
            String[] args = new String[]{
                    "--server", server.baseUrl(), "--threads", "1", "--dataset", "gsm8k",
                    "--dataset-source", rows.getAbsolutePath(), "--n_cases", "2",
                    "--grader-type", "regex", "--model", "m", "--output", state.toString(),
                    "--retries", "0", "--retry-delay", "0"};
            EvalCommand.execute(EvalCommand.parse(args), new StringBuilder(), new StringBuilder());
            Map<?, ?> first = readState(state);
            Map<?, ?> firstCases = (Map<?, ?>) ((Map<?, ?>) first.get("task_states")).get("cases");
            for (Object value : firstCases.values()) {
                assertEquals("server_error", ((Map<?, ?>) value).get("status"));
            }

            server.failuresBeforeSuccess = 0;
            String[] resumed = new String[]{
                    "--server", server.baseUrl(), "--threads", "1", "--dataset", "gsm8k",
                    "--dataset-source", rows.getAbsolutePath(), "--n_cases", "2",
                    "--grader-type", "regex", "--model", "m", "--output", state.toString(),
                    "--retries", "0", "--retry-delay", "0", "--resume"};
            StringBuilder out = new StringBuilder();
            EvalCommand.execute(EvalCommand.parse(resumed), out, new StringBuilder());
            assertTrue(out.toString(), out.toString().contains("Resuming from 2 pending tasks"));
            Map<?, ?> second = readState(state);
            Map<?, ?> secondCases = (Map<?, ?>) ((Map<?, ?>) second.get("task_states")).get("cases");
            assertEquals(2, ((Number) ((Map<?, ?>) second.get("task_states")).get("correct")).intValue());
            for (Object value : secondCases.values()) {
                assertEquals("ok", ((Map<?, ?>) value).get("status"));
            }
        }
    }

    @Test
    public void theLlmGraderRunsOnTheEvaluationServer() throws Exception {
        File rows = rows("{\"question\":\"What is 2 + 3?\",\"answer\":\"#### 42\"}\n");
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.content = "The answer is 42.";
            server.graderContent = "42";
            Path state = tmp.newFolder("llm").toPath().resolve("state.json");
            EvalOptions opts = EvalCommand.parse(new String[]{
                    "--server", server.baseUrl(), "--threads", "1", "--dataset", "gsm8k",
                    "--dataset-source", rows.getAbsolutePath(), "--n_cases", "1",
                    "--grader-type", "llm", "--model", "m", "--output", state.toString()});
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            EvalCommand.execute(opts, out, err);
            assertTrue(out.toString(), out.toString().contains("Results: 1/1 correct (100.0%)"));
            assertTrue(err.toString(), err.toString().contains(
                    "Warning: Using same server for LLM grader"));
            assertEquals("the generation and the extraction both reached the server", 2,
                    server.chatCalls());
        }
    }

    @Test
    public void aResumeWithAClippedModelIsRefused() throws Exception {
        File rows = rows("{\"question\":\"What is 2 + 3?\",\"answer\":\"#### 42\"}\n");
        try (FakeEvalServer server = new FakeEvalServer()) {
            Path state = tmp.newFolder("mismatch").toPath().resolve("state.json");
            String[] args = new String[]{
                    "--server", server.baseUrl(), "--threads", "1", "--dataset", "gsm8k",
                    "--dataset-source", rows.getAbsolutePath(), "--n_cases", "1",
                    "--grader-type", "regex", "--model", "first", "--output", state.toString()};
            EvalCommand.execute(EvalCommand.parse(args), new StringBuilder(), new StringBuilder());
            String[] other = new String[]{
                    "--server", server.baseUrl(), "--threads", "1", "--dataset", "gsm8k",
                    "--dataset-source", rows.getAbsolutePath(), "--n_cases", "1",
                    "--grader-type", "regex", "--model", "second", "--output", state.toString(),
                    "--resume"};
            try {
                EvalCommand.execute(EvalCommand.parse(other), new StringBuilder(),
                        new StringBuilder());
                fail("a resume of another model must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("first"));
                assertTrue(expected.getMessage(), expected.getMessage().contains("second"));
            }
        }
    }

    @Test
    public void aResumeWithoutAStateFileIsRefused() throws Exception {
        try {
            EvalOptions opts = EvalCommand.parse(new String[]{
                    "--dataset", "gsm8k", "--dataset-source",
                    rows("{\"question\":\"q\",\"answer\":\"#### 1\"}\n").getAbsolutePath(),
                    "--grader-type", "regex", "--output",
                    tmp.newFolder("missing").toPath().resolve("none.json").toString(),
                    "--resume"});
            EvalCommand.execute(opts, new StringBuilder(), new StringBuilder());
            fail("a resume without a state file must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("no existing eval state"));
        }
    }

    private File rows(String content) throws Exception {
        File file = tmp.newFile("rows-" + System.nanoTime() + ".jsonl");
        Files.write(file.toPath(), content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private static Map<?, ?> readState(Path file) throws Exception {
        return (Map<?, ?>) Json.parse(Files.readString(file), Json.Limits.UNTRUSTED);
    }

    private static void assertUsage(String[] args, String needle) {
        try {
            EvalCommand.parse(args);
            fail("the command line must be refused: " + String.join(" ", args));
        } catch (EvalCommand.UsageException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(needle));
        }
    }
}
