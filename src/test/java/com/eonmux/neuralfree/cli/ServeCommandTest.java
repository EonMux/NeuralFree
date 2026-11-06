package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.SecretProperties;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The serve command's jar resolution and child command over a temporary layout: the property and the
 * environment variable win over the layout, a missing explicit pointer falls through to the layout,
 * the distribution's {@code server/} directory and the source checkout's {@code server/build/libs}
 * are both found, and the refusal names every searched place. The command's child JVM line carries
 * the jar, the forwarded {@code neuralfree.*} properties and the caller's arguments in that order.
 */
public class ServeCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final List<Path> NO_ROOTS = Collections.emptyList();

    /** Creates {@code path} as a regular file, creating its parents; the content is irrelevant. */
    private static Path touch(Path path) throws Exception {
        Files.createDirectories(path.getParent());
        Files.write(path, new byte[]{1});
        return path;
    }

    @Test
    public void anExistingPropertyWins() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("custom/server.jar"));
        Path layout = touch(tmp.getRoot().toPath().resolve("lib/server.jar"));
        assertEquals(jar, ServeCommand.locate(jar.toString(), null,
                Collections.singletonList(tmp.getRoot().toPath().resolve("lib"))));
        assertTrue(Files.exists(layout));
    }

    @Test
    public void aMissingPropertyFallsThroughToTheEnvironment() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("env/server.jar"));
        assertEquals(jar, ServeCommand.locate(tmp.getRoot().toPath().resolve("absent.jar").toString(),
                jar.toString(), NO_ROOTS));
    }

    @Test
    public void theEnvironmentVariableIsHonoured() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("env/server.jar"));
        assertEquals(jar, ServeCommand.locate(null, jar.toString(), NO_ROOTS));
    }

    @Test
    public void theDistributionServerDirectoryIsFound() throws Exception {
        Path lib = tmp.newFolder("neuralfree").toPath().resolve("lib");
        Path jar = touch(tmp.getRoot().toPath().resolve("neuralfree").resolve("server")
                .resolve("server.jar"));
        assertEquals(jar, ServeCommand.locate(null, null, Collections.singletonList(lib)));
    }

    @Test
    public void theRunnerJarOfTheDistributionServerDirectoryIsFound() throws Exception {
        Path lib = tmp.newFolder("dist").toPath().resolve("lib");
        Path jar = touch(tmp.getRoot().toPath().resolve("dist").resolve("server")
                .resolve("neuralfree-server-runner.jar"));
        assertEquals(jar, ServeCommand.locate(null, null, Collections.singletonList(lib)));
    }

    @Test
    public void theSourceCheckoutLayoutIsFoundFromTheClasspathRoot() throws Exception {
        Path root = tmp.newFolder("repo").toPath();
        Path classpathRoot = touch(root.resolve("build").resolve("classes").resolve("java")
                .resolve("main").resolve("Keep.class")).getParent();
        // the Quarkus build puts the runner jar in server/build, not server/build/libs
        Path jar = touch(root.resolve("server").resolve("build")
                .resolve("neuralfree-server-runner.jar"));
        assertEquals(jar, ServeCommand.locate(null, null, Collections.singletonList(classpathRoot)));
        Path libsJar = touch(root.resolve("server").resolve("build").resolve("libs")
                .resolve("neuralfree-server-runner.jar"));
        Files.delete(jar);
        assertEquals(libsJar, ServeCommand.locate(null, null, Collections.singletonList(classpathRoot)));
    }

    @Test
    public void aJarBesideAClasspathRootIsFound() throws Exception {
        Path lib = tmp.newFolder("home").toPath().resolve("lib");
        Path jar = touch(lib.resolve("server.jar"));
        assertEquals(jar, ServeCommand.locate(null, null, Collections.singletonList(lib)));
    }

    @Test
    public void withoutAnyCandidateTheResolutionFails() throws Exception {
        Path lib = tmp.newFolder("bare").toPath().resolve("lib");
        assertNull(ServeCommand.locate(null, null, Collections.singletonList(lib)));
        assertNull(ServeCommand.locate(null, null, NO_ROOTS));
    }

    @Test
    public void theRefusalNamesEveryPlaceAndTheBuildStep() throws Exception {
        Path lib = tmp.newFolder("bare2").toPath().resolve("lib");
        String message = ServeCommand.missingMessage(Collections.singletonList(lib));
        assertTrue(message, message.contains("-D" + ServeCommand.JAR_PROPERTY));
        assertTrue(message, message.contains(ServeCommand.JAR_ENV));
        assertTrue(message, message.contains(lib.resolve("server.jar").toString()));
        assertTrue(message, message.contains(":server:quarkusBuild"));
    }

    @Test
    public void theChildCommandCarriesJarPropertiesAndArguments() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("server.jar"));
        Properties forwarded = new Properties();
        forwarded.setProperty("neuralfree.server.model", "/models/m.gguf");
        forwarded.setProperty("quarkus.http.port", "8081");
        List<String> cmd = ServeCommand.command(jar, null, new String[]{"--host", "0.0.0.0"},
                forwarded);
        String java = System.getProperty("java.home");
        assertEquals(java + File.separator + "bin" + File.separator + "java", cmd.get(0));
        int at = 1;
        // Java 17 or later resolves the incubator Vector API module for the server's CPU kernels (CpuSimd);
        // the JVM options precede -jar: after it they would reach the server as program arguments
        if (Runtime.version().feature() >= 17) {
            assertEquals("--add-modules=jdk.incubator.vector", cmd.get(at));
            at++;
        }
        assertEquals("-Dneuralfree.server.model=/models/m.gguf", cmd.get(at));
        assertEquals("-Dquarkus.http.port=8081", cmd.get(at + 1));
        assertEquals("-jar", cmd.get(at + 2));
        assertEquals(jar.toString(), cmd.get(at + 3));
        assertEquals("--host", cmd.get(at + 4));
        assertEquals("0.0.0.0", cmd.get(at + 5));
        assertEquals(at + 6, cmd.size());
    }

    @Test
    public void theForwardedPropertiesAreTheEngineAndServerNamespaces() {
        String savedEngine = System.getProperty("neuralfree.test.probe");
        String savedServer = System.getProperty("quarkus.test.probe");
        String other = System.getProperty("example.probe");
        System.setProperty("neuralfree.test.probe", "1");
        System.setProperty("quarkus.test.probe", "2");
        System.setProperty("example.probe", "3");
        try {
            Properties forwarded = ServeCommand.forwardedProperties();
            assertEquals("1", forwarded.getProperty("neuralfree.test.probe"));
            assertEquals("2", forwarded.getProperty("quarkus.test.probe"));
            assertNull(forwarded.getProperty("example.probe"));
        } finally {
            restore("neuralfree.test.probe", savedEngine);
            restore("quarkus.test.probe", savedServer);
            restore("example.probe", other);
        }
    }

    /** Puts {@code value} back into {@code name}, clearing the property when the saved value was null. */
    private static void restore(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    @Test
    public void theClasspathRootsAreTheDirectoriesAndTheJarParents() throws Exception {
        Path home = tmp.newFolder("app-home").toPath();
        Path lib = touch(home.resolve("lib").resolve("core.jar")).getParent();
        Path classes = touch(home.resolve("build").resolve("classes").resolve("java")
                .resolve("main").resolve("Keep.class")).getParent();
        String separator = java.io.File.pathSeparator;
        List<Path> roots = ServeCommand.rootsOf(lib.resolve("core.jar") + separator + classes
                + separator + home.resolve("missing-dir"));
        assertEquals(2, roots.size());
        assertTrue(roots.contains(lib));
        assertTrue(roots.contains(classes));
    }

    @Test
    public void theProgramArgumentsKeepTheDefinitionsOutOfTheServersArgs() {
        String[] program = ServeCommand.programArgsOf(new String[]{"-Dquarkus.http.port=18098",
                "--host", "0.0.0.0", "-Dneuralfree.server.model=/m.gguf"});
        assertEquals(2, program.length);
        assertEquals("--host", program[0]);
        assertEquals("0.0.0.0", program[1]);
        assertTrue(ServeCommand.isPropertyDefinition("-Dquarkus.http.port=18098"));
        assertFalse(ServeCommand.isPropertyDefinition("--host"));
        assertFalse(ServeCommand.isPropertyDefinition("-Dnovalue"));
        assertFalse(ServeCommand.isPropertyDefinition("-D=novalue"));
    }

    @Test
    public void theWildcardGlobMatchesOnlyTheRunnerSuffix() throws Exception {
        Path lib = tmp.newFolder("glob").toPath().resolve("lib");
        Path serverDir = tmp.getRoot().toPath().resolve("glob").resolve("server");
        Path runner = touch(serverDir.resolve("neuralfree-server-runner.jar"));
        touch(serverDir.resolve("neuralfree-server-runner.jar.original"));
        // the .original file does not match the suffix, so the runner itself is the resolution
        assertEquals(runner, ServeCommand.locate(null, null, Collections.singletonList(lib)));
        Files.delete(runner);
        assertNull(ServeCommand.locate(null, null, Collections.singletonList(lib)));
    }

    @Test
    public void theChildCommandLineCarriesNoSecretValue() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("secret/server.jar"));
        Properties properties = new Properties();
        properties.setProperty("neuralfree.server.model", "/models/m.gguf");
        properties.setProperty(SecretProperties.API_KEY_PROPERTY, "sk-live-123");
        properties.setProperty(EngineConfig.PROP_RPC, "box:50052@rpcsecret");
        properties.setProperty(WatermarkConfig.PROP_CONFIG, "{\"key\": 987654321987}");
        properties.setProperty("neuralfree.remote.gcs.token", "ya29.gcs-token");
        properties.setProperty("neuralfree.remote.azure.sas", "sv=1&sig=azure-sas");
        properties.setProperty(SecretProperties.MCP_JSON_PROPERTY,
                "{\"mcpServers\":{\"x\":{\"command\":\"a\",\"env\":{\"API_KEY\":\"mcp-secret\"}}}}");
        properties.setProperty(KvTransferConfig.SERVER_PROPERTY,
                "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                        + " \"kv_connector_extra_config\": {\"auth_token\": \"kv-transfer-secret\"}}");

        List<String> cmd = ServeCommand.command(jar, null, new String[]{"--port", "8080"}, properties);
        String line = String.join("\n", cmd);
        for (String secret : new String[]{"sk-live-123", "rpcsecret", "987654321987", "ya29.gcs-token",
                "sig=azure-sas", "mcp-secret", "kv-transfer-secret", SecretProperties.API_KEY_PROPERTY,
                EngineConfig.PROP_RPC, WatermarkConfig.PROP_CONFIG, SecretProperties.MCP_JSON_PROPERTY,
                KvTransferConfig.SERVER_PROPERTY}) {
            assertFalse(secret + " reached the command line: " + line, line.contains(secret));
        }
        assertTrue(line, line.contains("-Dneuralfree.server.model=/models/m.gguf"));
        assertTrue(line, line.contains("--port"));
    }

    @Test
    public void theSecretLaunchFlagsLeaveTheArgumentListAndBecomeProperties() {
        Properties properties = new Properties();
        String watermark = System.getenv(WatermarkConfig.ENV_KEY) == null
                ? "{\"key\": 42}" : "{\"algorithm\": \"gumbel\"}";
        String kvTransfer = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\"}";
        String[] rest = ServeCommand.secretFlagsOf(new String[]{
                "--api-key", "sk-1", "--api-key", "sk-2", "--rpc", "box:50052@rpcsecret",
                "--watermark-config", watermark, "--mcp-servers-json", "{\"mcpServers\":{}}",
                "--kv-transfer-config", kvTransfer, "--port", "8080"}, properties);

        assertArrayEquals(new String[]{"--port", "8080"}, rest);
        assertEquals("sk-1,sk-2", properties.getProperty(SecretProperties.API_KEY_PROPERTY));
        assertEquals("box:50052@rpcsecret", properties.getProperty(EngineConfig.PROP_RPC));
        assertEquals("layer", properties.getProperty(EngineConfig.PROP_SPLIT_MODE));
        assertEquals(watermark, properties.getProperty(WatermarkConfig.PROP_CONFIG));
        assertEquals("{\"mcpServers\":{}}", properties.getProperty(SecretProperties.MCP_JSON_PROPERTY));
        assertEquals(kvTransfer, properties.getProperty(KvTransferConfig.SERVER_PROPERTY));
    }

    @Test
    public void aRefusedKvTransferConfigNamesTheFlag() {
        String json = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\", \"wat\": 1}";
        try {
            ServeCommand.secretFlagsOf(new String[]{"--kv-transfer-config", json}, new Properties());
            fail("accepted an unknown kv-transfer-config field");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--kv-transfer-config"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("wat"));
        }
    }

    @Test
    public void aRefusedSecretFlagNamesNoKey() {
        try {
            ServeCommand.secretFlagsOf(new String[]{"--rpc", "box:abc@rpcsecret"}, new Properties());
            fail("accepted a malformed rpc entry");
        } catch (IllegalArgumentException expected) {
            assertFalse(expected.getMessage(), expected.getMessage().contains("rpcsecret"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("box:abc@..."));
        }
        try {
            ServeCommand.secretFlagsOf(new String[]{"--api-key", " "}, new Properties());
            fail("accepted a blank key");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--api-key"));
        }
    }

    @Test
    public void theSecretsReachTheChildThroughAnOwnerOnlyFile() throws Exception {
        Path jar = touch(tmp.getRoot().toPath().resolve("secret-file/server.jar"));
        Properties properties = new Properties();
        properties.setProperty("neuralfree.server.model", "/models/m.gguf");
        properties.setProperty(SecretProperties.API_KEY_PROPERTY, "sk-live");
        properties.setProperty(EngineConfig.PROP_RPC, "box:50052@rpcsecret");

        ProcessBuilder pb = ServeCommand.processBuilder(jar, null, new String[0], properties);
        String line = String.join(" ", pb.command());
        assertFalse(line, line.contains("sk-live"));
        assertFalse(line, line.contains("rpcsecret"));
        assertTrue(line, line.contains("-Dneuralfree.server.model=/models/m.gguf"));

        String path = pb.environment().get(SecretProperties.FILE_ENV);
        assertNotNull("the child is told where the secret file is", path);
        Path file = Path.of(path);
        try {
            Set<PosixFilePermission> permissions = Files.getPosixFilePermissions(file);
            assertFalse("the group and others read no secret: " + permissions,
                    permissions.contains(PosixFilePermission.GROUP_READ)
                            || permissions.contains(PosixFilePermission.OTHERS_READ)
                            || permissions.contains(PosixFilePermission.GROUP_WRITE)
                            || permissions.contains(PosixFilePermission.OTHERS_WRITE));
            Properties loaded = new Properties();
            try (InputStream in = Files.newInputStream(file)) {
                loaded.load(in);
            }
            assertEquals("sk-live", loaded.getProperty(SecretProperties.API_KEY_PROPERTY));
            assertEquals("box:50052@rpcsecret", loaded.getProperty(EngineConfig.PROP_RPC));
            assertNull("a public setting stays on the command line", loaded.getProperty("neuralfree.server.model"));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theChildLoadsTheSecretFileAndDeletesIt() throws Exception {
        Properties secrets = new Properties();
        secrets.setProperty(SecretProperties.API_KEY_PROPERTY, "sk-child");
        Path file = SecretProperties.write(secrets);
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-cp", System.getProperty("java.class.path"),
                    SecretPropertiesProbe.class.getName(), SecretProperties.API_KEY_PROPERTY);
            pb.environment().put(SecretProperties.FILE_ENV, file.toString());
            pb.redirectErrorStream(true);
            Process child = pb.start();
            String output = new String(child.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            assertEquals(0, child.waitFor());
            assertEquals("sk-child", output);
            assertFalse("the child deletes the file it read", Files.exists(file));
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
