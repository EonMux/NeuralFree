package com.eonmux.neuralfree.cli;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The launcher's routing surface: every command and alias resolves, the {@code --flag} spelling of a
 * command resolves as llama.cpp's app binary allows, unknown commands resolve to nothing (main
 * refuses them with the surface), the usage names every visible command and the global flags while
 * hiding the tool-level ones until {@code help all}, and no two commands share a name or an alias.
 * {@code run-batch} runs the server jar's batch runner over the jar {@code serve} resolves, with the
 * same property forwarding.
 */
public class LauncherTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyCommandResolvesByItsName() {
        for (Launcher.Command c : Launcher.COMMANDS) {
            assertEquals(c.name(), c.name(), Launcher.commandOf(c.name()).name());
        }
    }

    @Test
    public void everyAliasResolvesToItsCommand() {
        assertEquals("serve", Launcher.commandOf("server").name());
        assertEquals("cli", Launcher.commandOf("client").name());
        assertEquals("download", Launcher.commandOf("get").name());
    }

    @Test
    public void aCommandMayBeSpelledAsAFlag() {
        assertEquals("serve", Launcher.commandOf("--serve").name());
        assertEquals("version", Launcher.commandOf("--version").name());
        assertEquals("cache-list", Launcher.commandOf("--cache-list").name());
    }

    @Test
    public void anUnknownCommandResolvesToNothing() {
        assertNull(Launcher.commandOf("no-such-command"));
        assertNull(Launcher.commandOf("--no-such-command"));
        assertNull(Launcher.commandOf(""));
    }

    @Test
    public void theUsageNamesTheVisibleCommandsAndTheGlobalFlags() {
        StringBuilder out = new StringBuilder();
        Launcher.usage(out, false);
        String usage = out.toString();
        assertTrue(usage.contains("Usage: neuralfree <command> [options]"));
        for (String command : new String[]{"serve", "run-batch", "cli", "chat", "complete", "bench", "download",
                "cache-list", "collect-env", "quantize", "perplexity", "version", "help", "bench-serve",
                "bench-throughput", "bench-latency", "bench-sweep", "bench-startup", "bench-compare"}) {
            assertTrue(usage.contains(command));
        }
        assertTrue(usage, usage.contains("--version"));
        assertTrue(usage, usage.contains("--list-devices"));
        assertTrue(usage, usage.contains("--cache-list"));
        assertTrue(usage, usage.contains("help all"));
    }

    @Test
    public void theDefaultUsageHidesTheToolLevelCommands() {
        StringBuilder out = new StringBuilder();
        Launcher.usage(out, false);
        String usage = out.toString();
        for (String hidden : new String[]{"tokenize", "dump", "hash", "batched-bench", "export-lora",
                "retrieval", "pack-experts", "train"}) {
            assertTrue(usage, !usage.contains(hidden));
        }
        StringBuilder all = new StringBuilder();
        Launcher.usage(all, true);
        for (String hidden : new String[]{"tokenize", "dump", "hash", "pack-experts", "train"}) {
            assertTrue(all.toString(), all.toString().contains(hidden));
        }
    }

    @Test
    public void noTwoCommandsShareANameOrAnAlias() {
        Set<String> seen = new HashSet<>();
        for (Launcher.Command c : Launcher.COMMANDS) {
            assertTrue(c.name(), seen.add(c.name()));
            for (String alias : c.aliases()) {
                assertTrue(alias, seen.add(alias));
            }
        }
    }

    @Test
    public void everyCommandDescribesItself() {
        for (Launcher.Command c : Launcher.COMMANDS) {
            assertNotNull(c.name(), c.description());
            assertTrue(c.name(), !c.description().trim().isEmpty());
        }
    }

    @Test
    public void everyNftoolCommandIsALauncherCommand() {
        for (String tool : com.eonmux.neuralfree.demo.tools.NfTool.COMMANDS) {
            assertNotNull(tool, Launcher.commandOf(tool));
        }
        assertTrue(com.eonmux.neuralfree.demo.tools.NfTool.COMMANDS
                .containsAll(com.eonmux.neuralfree.demo.tools.NfTool.IMPLEMENTED));
    }

    @Test
    public void theVersionLineCarriesTheProgramName() {
        String line = Launcher.versionLine();
        assertTrue(line, line.startsWith("neuralfree "));
        assertTrue(line, line.contains(Version.current()));
    }

    @Test
    public void runBatchIsAVisibleCommand() {
        assertEquals("run-batch", Launcher.commandOf("run-batch").name());
        assertEquals("run-batch", Launcher.commandOf("--run-batch").name());
        assertTrue(!Launcher.commandOf("run-batch").hidden());
    }

    @Test
    public void theInProcessEngineCommandsAreMarkedForTheVectorModuleRelaunch() {
        for (String name : new String[]{"cli", "bench", "quantize", "perplexity", "imatrix", "cvector",
                "export", "train", "download"}) {
            assertTrue(name, Launcher.runsEngineInProcess(Launcher.commandOf(name)));
        }
    }

    @Test
    public void theServerClientAndSelfDescribingCommandsDoNotRelaunchInProcess() {
        for (String name : new String[]{"serve", "run-batch", "chat", "complete", "version", "help",
                "cache-list", "collect-env"}) {
            assertFalse(name, Launcher.runsEngineInProcess(Launcher.commandOf(name)));
        }
    }

    @Test
    public void theOpenAiClientCommandsAreVisible() {
        for (String name : new String[]{"chat", "complete"}) {
            assertEquals(name, Launcher.commandOf(name).name());
            assertFalse(name, Launcher.commandOf(name).hidden());
        }
        assertEquals("chat", Launcher.commandOf("--chat").name());
        assertEquals("complete", Launcher.commandOf("--complete").name());
    }

    @Test
    public void runBatchRunsTheBatchRunnerClassOnTheServerJarsClasspath() throws Exception {
        Path jar = tmp.getRoot().toPath().resolve("server.jar");
        Files.write(jar, new byte[]{1});
        Properties forwarded = new Properties();
        forwarded.setProperty("neuralfree.parallel", "4");
        String[] program = ServeCommand.programArgsOf(new String[]{"-Dneuralfree.server.model=/m.gguf",
                "-i", "https://host/in.jsonl", "-o", "out.jsonl", "--enable-metrics"});
        List<String> cmd = ServeCommand.command(jar, ServeCommand.BATCH_RUNNER_CLASS, program, forwarded);
        String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
        assertEquals(javaBin, cmd.get(0));
        int at = 1;
        // the incubator Vector API module rides along on Java 17+ (CpuSimd); the JVM options precede the
        // classpath and the class: after them they would be program arguments
        if (Runtime.version().feature() >= 17) {
            assertEquals("--add-modules=jdk.incubator.vector", cmd.get(at));
            at++;
        }
        assertEquals("-Dneuralfree.parallel=4", cmd.get(at));
        assertEquals("-cp", cmd.get(at + 1));
        assertEquals(jar.toString(), cmd.get(at + 2));
        assertEquals("com.eonmux.neuralfree.server.BatchRunner", cmd.get(at + 3));
        assertEquals(Arrays.asList("-i", "https://host/in.jsonl", "-o", "out.jsonl",
                "--enable-metrics"), cmd.subList(at + 4, cmd.size()));
        assertTrue(cmd.toString(), !cmd.contains("-jar"));
    }

    @Test
    public void runBatchResolvesTheJarServeResolves() throws Exception {
        Path lib = tmp.newFolder("dist").toPath().resolve("lib");
        Path jar = tmp.getRoot().toPath().resolve("dist").resolve("server")
                .resolve("neuralfree-server-runner.jar");
        Files.createDirectories(jar.getParent());
        Files.write(jar, new byte[]{1});
        assertEquals(jar, ServeCommand.locate(null, null, Collections.singletonList(lib)));
        String missing = ServeCommand.missingMessage(Collections.singletonList(lib));
        assertTrue(missing, missing.contains("run-batch"));
    }
}
