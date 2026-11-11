package com.eonmux.neuralfree.cli;

import com.sun.jna.Platform;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The Unix start-script snippet ({@code gradle/start-scripts/vector-module.sh}) that resolves
 * {@value JvmVectorModule#MODULE} at start-up: it prepends {@link JvmVectorModule#OPTION} to
 * {@code DEFAULT_JVM_OPTS} exactly when the first argument names a command that runs the engine in the launcher's
 * JVM ({@link Launcher#runsEngineInProcess}) and the runtime's release file, read from {@code $JAVA_HOME/release} or
 * two directories above the resolved {@code java}, names Java 17 or later and, where it lists the image's modules,
 * the module itself. Otherwise the options stay as they are: on an older or unknown runtime the option would be an
 * error, and on an image linked without the module the JVM would not start. The generated start script
 * ({@code testStartScripts}, the build's {@code startScripts} over an empty class path) is run end to end with a stub
 * {@code java}, so the snippet's place in it is checked too.
 */
public class StartScriptVectorModuleTest {

    private static final String OPTS = "\"-Xmx4g\" \"--add-opens=java.base/java.util=ALL-UNNAMED\"";
    private static final String WITH_MODULE = "\"" + JvmVectorModule.OPTION + "\" " + OPTS;

    /** A release file's module list: a full JDK's names the incubator module among the others. */
    private static final String FULL_MODULES = "MODULES=\"java.base java.logging jdk.incubator.vector jdk.zipfs\"";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static Path snippet() {
        Path dir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        for (Path d = dir; d != null; d = d.getParent()) {
            Path file = d.resolve("gradle/start-scripts/vector-module.sh");
            if (Files.isRegularFile(file)) {
                return file;
            }
        }
        throw new IllegalStateException("gradle/start-scripts/vector-module.sh not found above " + dir);
    }

    /**
     * A runtime directory with a {@code bin/java} that prints its arguments one per line and, unless
     * {@code release} is null, a release file of those lines.
     */
    private Path runtime(String name, String... release) throws IOException {
        Path home = tmp.newFolder(name).toPath();
        Path java = Files.createDirectories(home.resolve("bin")).resolve("java");
        Files.write(java, "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done\n"
                .getBytes(StandardCharsets.US_ASCII));
        assertTrue(java.toFile().setExecutable(true));
        if (release != null) {
            Files.write(home.resolve("release"), ("IMPLEMENTOR=\"Test\"\n" + String.join("\n", release) + "\n")
                    .getBytes(StandardCharsets.US_ASCII));
        }
        return home;
    }

    private Path jdk(String name, String version) throws IOException {
        return runtime(name, "JAVA_VERSION=" + version, FULL_MODULES);
    }

    /** Runs {@code command} with {@code javaHome} (null: unset), {@code path} prepended to PATH, and its output. */
    private static String exec(List<String> command, Path javaHome, Path path) throws IOException,
            InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(command);
        Map<String, String> env = pb.environment();
        env.remove("JAVA_HOME");
        env.remove("JAVA_OPTS");
        env.remove("NEURALFREE_OPTS");
        if (javaHome != null) {
            env.put("JAVA_HOME", javaHome.toString());
        }
        if (path != null) {
            env.put("PATH", path + ":" + env.getOrDefault("PATH", "/usr/bin:/bin"));
        }
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue("the script finishes", p.waitFor(30, TimeUnit.SECONDS));
        assertEquals(out, 0, p.exitValue());
        return out;
    }

    /** DEFAULT_JVM_OPTS after the snippet runs for {@code args} with {@code javaHome} (null: unset), {@code javaCmd}. */
    private static String run(Path javaHome, String javaCmd, String... args) throws IOException,
            InterruptedException {
        List<String> cmd = new ArrayList<>(Arrays.asList("sh", "-c",
                "JAVACMD=$1; DEFAULT_JVM_OPTS=$2; shift 3; . \"$0\"; printf '%s' \"$DEFAULT_JVM_OPTS\"",
                snippet().toString(), javaCmd, OPTS, "--"));
        cmd.addAll(Arrays.asList(args));
        return exec(cmd, javaHome, null);
    }

    @Test
    public void javaHomeOfJava17OrLaterResolvesTheModule() throws Exception {
        Assume.assumeTrue(Platform.isLinux() || Platform.isMac());
        for (String version : new String[] {"\"17\"", "\"21.0.12.1\"", "\"25-ea\"", "17.0.2"}) {
            Path home = jdk("jdk" + version.replace("\"", ""), version);
            assertEquals(version, WITH_MODULE, run(home, home.resolve("bin/java").toString(), "bench"));
        }
        Path unlisted = runtime("unlisted", "JAVA_VERSION=\"21\"");
        assertEquals("a release file without a module list is a full JDK", WITH_MODULE,
                run(unlisted, unlisted.resolve("bin/java").toString(), "bench"));
    }

    @Test
    public void olderUnknownOrModuleLessRuntimesKeepTheirOptions() throws Exception {
        Assume.assumeTrue(Platform.isLinux() || Platform.isMac());
        for (String version : new String[] {"\"11.0.24\"", "\"16\"", "\"1.8.0_392\"", "\"\"", "\"abc\""}) {
            Path home = jdk("jdk" + version.replace("\"", "") + "x", version);
            assertEquals(version, OPTS, run(home, home.resolve("bin/java").toString(), "bench"));
        }
        Path noRelease = runtime("bare", (String[]) null);
        assertEquals("no release file", OPTS, run(noRelease, noRelease.resolve("bin/java").toString(), "bench"));
        assertEquals("no java at all", OPTS,
                run(null, tmp.getRoot().toPath().resolve("missing/java").toString(), "bench"));
        Path linked = runtime("jlinked", "JAVA_VERSION=\"21.0.2\"", "MODULES=\"java.base java.logging jdk.zipfs\"");
        assertEquals("an image linked without the module", OPTS,
                run(linked, linked.resolve("bin/java").toString(), "bench"));
        Path near = runtime("near", "JAVA_VERSION=\"21\"", "MODULES=\"java.base jdk.incubator.vectorx\"");
        assertEquals("a module name that only starts with it", OPTS,
                run(near, near.resolve("bin/java").toString(), "bench"));
        Path empty = runtime("empty", "JAVA_VERSION=\"21\"", "MODULES=");
        assertEquals("an empty module list", OPTS, run(empty, empty.resolve("bin/java").toString(), "bench"));
    }

    @Test
    public void withoutJavaHomeTheReleaseBesideTheResolvedJavaDecides() throws Exception {
        Assume.assumeTrue(Platform.isLinux());
        Path jdk = jdk("jdk21", "\"21.0.1\"");
        Path link = tmp.newFolder("path").toPath().resolve("java");
        Files.createSymbolicLink(link, jdk.resolve("bin/java"));
        assertEquals("a symlinked java resolves to its runtime", WITH_MODULE, run(null, link.toString(), "bench"));
        Path old = jdk("jdk11", "\"11.0.2\"");
        assertEquals(OPTS, run(null, old.resolve("bin/java").toString(), "bench"));
    }

    /**
     * Every first argument the launcher routes gets the option exactly when its command runs the engine in the
     * launcher's JVM: the commands with their aliases and {@code --} spellings, the flags that act and exit, and no
     * argument at all (the help).
     */
    @Test
    public void onlyCommandsThatRunTheEngineInTheLauncherResolveTheModule() throws Exception {
        Assume.assumeTrue(Platform.isLinux() || Platform.isMac());
        Path home = jdk("jdk21", "\"21\"");
        String java = home.resolve("bin/java").toString();
        for (Launcher.Command c : Launcher.COMMANDS) {
            List<String> spellings = new ArrayList<>();
            spellings.add(c.name());
            spellings.addAll(c.aliases());
            for (String s : new ArrayList<>(spellings)) {
                spellings.add("--" + s);
            }
            String expected = Launcher.runsEngineInProcess(c) ? WITH_MODULE : OPTS;
            for (String s : spellings) {
                assertEquals(s, expected, run(home, java, s, "--model", "m.gguf"));
            }
        }
        List<String> flags = new ArrayList<>();
        flags.addAll(Launcher.VERSION_FLAGS);
        flags.addAll(Launcher.LIST_DEVICES_FLAGS);
        flags.addAll(Launcher.CACHE_LIST_FLAGS);
        flags.addAll(Launcher.HELP_FLAGS);
        for (String flag : flags) {
            assertEquals(flag, OPTS, run(home, java, flag));
        }
        assertEquals("no argument runs the help", OPTS, run(home, java));
    }

    /** The generated Unix start script, or a skipped test outside the Gradle build that generates it. */
    private static Path generatedScript() {
        String dir = System.getProperty("neuralfree.test.startScripts");
        Assume.assumeTrue("the build's testStartScripts output is not given", dir != null);
        Path script = Paths.get(dir, "neuralfree");
        assertTrue(script + " is generated", Files.isRegularFile(script));
        return script;
    }

    /** The JVM arguments the generated start script hands {@code java} for {@code args}. */
    private static List<String> launch(Path script, Path javaHome, Path path, String... args) throws Exception {
        List<String> cmd = new ArrayList<>(Arrays.asList("sh", script.toString()));
        cmd.addAll(Arrays.asList(args));
        String out = exec(cmd, javaHome, path);
        List<String> argv = Arrays.asList(out.split("\n"));
        int main = argv.indexOf("com.eonmux.neuralfree.cli.Launcher");
        assertTrue(out, main > 0);
        assertEquals("the command's own arguments follow the main class", Arrays.asList(args),
                argv.subList(main + 1, argv.size()));
        return argv.subList(0, main);
    }

    @Test
    public void theGeneratedStartScriptPassesTheModuleAmongItsJvmOptions() throws Exception {
        Assume.assumeTrue(Platform.isLinux() || Platform.isMac());
        Path script = generatedScript();
        String text = Files.readString(script);
        int javaCmd = text.indexOf("JAVACMD=java");
        int defaults = text.indexOf("\nDEFAULT_JVM_OPTS=");
        int snippetAt = text.indexOf("NEURALFREE_VECTOR_COMMAND=");
        assertTrue("the snippet follows JAVACMD and DEFAULT_JVM_OPTS", javaCmd > 0 && defaults > javaCmd
                && snippetAt > defaults);
        Path home = jdk("jdk21", "\"21.0.1\"");
        List<String> bench = launch(script, home, null, "bench", "-m", "a b.gguf");
        assertTrue(bench.toString(), bench.contains(JvmVectorModule.OPTION));
        assertTrue("a JVM option, before the class path: " + bench,
                bench.indexOf(JvmVectorModule.OPTION) < bench.indexOf("-classpath"));
        assertTrue(bench.toString(), bench.contains("-Xmx4g"));
        assertFalse("version starts without it", launch(script, home, null, "version")
                .contains(JvmVectorModule.OPTION));
        Path linked = runtime("jlinked", "JAVA_VERSION=\"21.0.2\"", "MODULES=\"java.base java.logging\"");
        assertFalse("an image without the module", launch(script, linked, null, "bench")
                .contains(JvmVectorModule.OPTION));
    }

    @Test
    public void theGeneratedStartScriptResolvesTheJavaOnThePath() throws Exception {
        Assume.assumeTrue(Platform.isLinux());
        Path script = generatedScript();
        Path jdk = jdk("jdk21", "\"21.0.1\"");
        Path bin = tmp.newFolder("onpath").toPath();
        Files.createSymbolicLink(bin.resolve("java"), jdk.resolve("bin/java"));
        assertTrue("JAVACMD=java found through PATH and its symlink",
                launch(script, null, bin, "bench").contains(JvmVectorModule.OPTION));
        Path old = jdk("jdk11", "\"11.0.2\"");
        Path oldBin = tmp.newFolder("oldpath").toPath();
        Files.createSymbolicLink(oldBin.resolve("java"), old.resolve("bin/java"));
        assertFalse(launch(script, null, oldBin, "bench").contains(JvmVectorModule.OPTION));
    }
}
