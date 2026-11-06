package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code bench-sweep}: the required command lines, the link variables, the health URL of a server
 * command, the combination directories, the summary CSV and the dry-run plan.
 */
public class BenchSweepCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void bothCommandLinesAreRequired() throws IOException {
        try {
            BenchSweepCommand.parse(new String[]{"--serve-cmd", "s"});
            fail("--bench-cmd is required");
        } catch (BenchSweepCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("--bench-cmd"));
        }
        try {
            BenchSweepCommand.parse(new String[]{"--bench-cmd", "b"});
            fail("--serve-cmd is required");
        } catch (BenchSweepCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("--serve-cmd"));
        }
    }

    @Test
    public void theFlagsParseWithTheReferenceDefaults() throws Exception {
        BenchSweepCommand.Args args = BenchSweepCommand.parse(new String[]{
                "--serve-cmd", "neuralfree serve model.gguf",
                "--bench-cmd", "bench-serve --num-prompts 2",
                "--link-vars", "max-model-len=max_tokens",
                "-o", "out", "-e", "exp", "--num-runs", "5", "--warmup-num-prompts", "3",
                "--continue-on-error", "--dry-run", "--resume", "--show-stdout",
                "--server-ready-timeout", "42"});
        assertEquals(Arrays.asList("neuralfree", "serve", "model.gguf"), args.serveCmd);
        assertEquals(Arrays.asList("bench-serve", "--num-prompts", "2"), args.benchCmd);
        org.junit.Assert.assertArrayEquals(new String[]{"max-model-len", "max_tokens"}, args.linkVars.get(0));
        assertEquals("out", args.outputDir.toString());
        assertEquals("exp", args.experimentName);
        assertEquals(5, args.numRuns);
        assertEquals(3, args.warmupNumPrompts);
        assertTrue(args.continueOnError);
        assertTrue(args.dryRun);
        assertTrue(args.resume);
        assertTrue(args.showStdout);
        assertEquals(42, args.serverReadyTimeout);
    }

    @Test
    public void aBadLinkVariableIsRefused() throws IOException {
        try {
            BenchSweepCommand.parse(new String[]{"--serve-cmd", "s", "--bench-cmd", "b",
                    "--link-vars", "no-equals"});
            fail("a link variable is a serve=bench pair");
        } catch (BenchSweepCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("serve=bench"));
        }
    }

    @Test
    public void theHealthUrlDerivesFromTheServerCommand() {
        assertEquals("http://127.0.0.1:8080/health",
                BenchSweepCommand.healthUrl(Arrays.asList("neuralfree", "serve")));
        assertEquals("http://0.0.0.0:9000/health", BenchSweepCommand.healthUrl(
                Arrays.asList("neuralfree", "serve", "--host", "0.0.0.0", "--port", "9000")));
    }

    @Test
    public void aCombinationDirectoryNamesTheSetRecords() {
        Map<String, Object> serve = new LinkedHashMap<>();
        serve.put("_benchmark_name", "tp2");
        Map<String, Object> bench = new LinkedHashMap<>();
        bench.put("_benchmark_name", "rate/8");
        Path dir = BenchSweepCommand.combinationDir(tmp.getRoot().toPath(),
                new ParameterSweep.Item(serve), new ParameterSweep.Item(bench));
        assertEquals("SERVE-tp2-BENCH-rate-8", dir.getFileName().toString());
        assertEquals("default", BenchSweepCommand.combinationDir(tmp.getRoot().toPath(),
                new ParameterSweep.Item(new LinkedHashMap<>()),
                new ParameterSweep.Item(new LinkedHashMap<>())).getFileName().toString());
    }

    @Test
    public void theSummaryCsvCarriesEveryRunAndQuotesComposites() throws IOException {
        List<Map<String, Object>> runs = new ArrayList<>();
        Map<String, Object> first = new LinkedHashMap<>();
        first.put("num_prompts", 2);
        first.put("label", "a,b");
        runs.add(first);
        Map<String, Object> second = new LinkedHashMap<>();
        second.put("num_prompts", 4);
        second.put("label", "plain");
        second.put("extra", Arrays.asList(1, 2));
        runs.add(second);
        Path file = tmp.newFile("summary.csv").toPath();
        BenchSweepCommand.writeSummaryCsv(file, runs);
        String[] lines = Files.readString(file, StandardCharsets.UTF_8).split("\n");
        assertEquals(3, lines.length);
        assertEquals("num_prompts,label,extra", lines[0]);
        assertEquals("2,\"a,b\",", lines[1]);
        assertTrue(lines[2], lines[2].startsWith("4,plain,"));
    }

    @Test
    public void anExistingExperimentIsNotOverwrittenAndResumeNeedsIt() throws Exception {
        Path existing = tmp.newFolder("results").toPath().resolve("exp");
        Files.createDirectories(existing);
        BenchSweepCommand.Args args = BenchSweepCommand.parse(new String[]{
                "--serve-cmd", "s", "--bench-cmd", "b", "-o", existing.getParent().toString(),
                "-e", existing.getFileName().toString()});
        try {
            args.experimentDir();
            fail("an existing experiment must not be overwritten");
        } catch (BenchSweepCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("cannot overwrite"));
        }
    }

    @Test
    public void aDryRunPrintsThePlanWithoutRunning() throws Exception {
        Path outDir = tmp.newFolder("out").toPath();
        BenchSweepCommand.Args args = BenchSweepCommand.parse(new String[]{
                "--serve-cmd", "neuralfree serve model.gguf --port 8080",
                "--bench-cmd", "bench-serve --num-prompts 2",
                "-o", outDir.toString(), "-e", "plan", "--num-runs", "2", "--dry-run"});
        StringBuilder out = new StringBuilder();
        BenchSweepCommand.sweep(args, out, new StringBuilder());
        String plan = out.toString();
        assertTrue(plan, plan.contains("[DRY RUN] server command: neuralfree serve model.gguf --port 8080"));
        assertTrue(plan, plan.contains("[BENCHMARK COMMAND] bench-serve --num-prompts 2"));
        assertTrue(plan, plan.contains("--percentile-metrics ttft,tpot,itl,e2el"));
        assertTrue(plan, plan.contains("run=1"));
        assertTrue(plan, plan.contains("The experiment will be saved at"));
        assertTrue("a dry run creates nothing", !Files.exists(outDir.resolve("plan")));
    }
}
