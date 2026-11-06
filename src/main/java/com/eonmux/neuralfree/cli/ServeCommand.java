package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.SecretProperties;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The commands that run over the server module's packaged uber-jar: {@code serve} launches the HTTP
 * server of {@code server/} and {@code run-batch} its offline batch runner ({@value
 * #BATCH_RUNNER_CLASS}), each waited for. The server is its own Quarkus application, so both run as
 * a child JVM over the jar — the same jar {@code server/build} produces and {@code java -jar} runs —
 * with this JVM's {@code neuralfree.*} and {@code quarkus.*} system properties and the command line
 * forwarded; {@code serve} runs the jar's {@code Main-Class}, {@code run-batch} names its class on
 * the jar's classpath. Leading
 * {@code -D<name>=<value>} arguments of the command itself define system properties for the child.
 * The settings {@link SecretProperties#isSecret(String)} marks secret — the API keys, the RPC entries
 * with their inline keys, the watermark configuration, the KV transfer configuration, the inline MCP
 * configuration and the cloud credentials — never ride on the child's command line: they go into a
 * temporary properties file that is readable by its owner only and is named to the child by
 * {@value SecretProperties#FILE_ENV}. The server reads and deletes that file at start-up. The secret
 * launch flags ({@code --api-key}, {@code --rpc}, {@code --watermark-config}, {@code
 * --kv-transfer-config} and {@code --mcp-servers-json}) are taken off the child's argument list the
 * same way, so a key given to {@code neuralfree serve} is not visible in {@code ps} as the child's
 * argument. The jar is looked up from the {@code neuralfree.cli.server.jar} property, the {@code
 * NEURALFREE_SERVER_JAR} environment variable, the distribution's {@code server} directory beside
 * the launcher's jars and the source checkout's Quarkus build output above any classpath root; when
 * no candidate exists the command refuses naming every place it searched and how the jar is
 * produced. Launching a jar that does not exist is an error too, so a stale pointer fails at
 * start-up instead of inside the child.
 */
public final class ServeCommand {

    /** The system property naming the server jar, ahead of the environment variable. */
    public static final String JAR_PROPERTY = "neuralfree.cli.server.jar";

    /** The environment variable naming the server jar, after the property. */
    public static final String JAR_ENV = "NEURALFREE_SERVER_JAR";

    /** The server jar's offline batch runner, the class {@code run-batch} launches. */
    public static final String BATCH_RUNNER_CLASS = "com.eonmux.neuralfree.server.BatchRunner";

    /** How far up from a classpath root the source checkout's Quarkus build output is looked for. */
    static final int MAX_ANCESTORS = 8;

    private ServeCommand() {
    }

    /**
     * Runs a command over the server jar: locates the jar, launches {@code mainClass} of it (the
     * jar's {@code Main-Class}, the HTTP server, when null) with the caller's {@code -D} definitions
     * and the forwarded properties, and exits with the child's exit code.
     */
    public static void launch(String[] args, String mainClass) throws IOException, InterruptedException {
        for (String a : args) {
            if (isPropertyDefinition(a)) {
                int eq = a.indexOf('=');
                System.setProperty(a.substring(2, eq), a.substring(eq + 1));
            }
        }
        List<Path> classpathRoots = classpathRoots();
        Path jar = locate(System.getProperty(JAR_PROPERTY), System.getenv(JAR_ENV), classpathRoots);
        if (jar == null) {
            System.err.println("Error: " + missingMessage(classpathRoots));
            System.exit(1);
            return;
        }
        Properties properties = forwardedProperties();
        String[] programArgs = programArgsOf(args);
        try {
            programArgs = secretFlagsOf(programArgs, properties);
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
            return;
        }
        Process process = start(jar, mainClass, programArgs, properties);
        Runtime.getRuntime().addShutdownHook(new Thread(process::destroy));
        int code = process.waitFor();
        System.exit(code);
    }

    /** Whether an argument is a {@code -D<name>=<value>} definition for the child's JVM. */
    static boolean isPropertyDefinition(String arg) {
        return arg.startsWith("-D") && arg.length() > 2 && arg.indexOf('=') > 2;
    }

    /** The caller's arguments without the {@code -D} definitions, which ride in as JVM options. */
    static String[] programArgsOf(String[] args) {
        List<String> out = new ArrayList<>();
        for (String a : args) {
            if (!isPropertyDefinition(a)) {
                out.add(a);
            }
        }
        return out.toArray(new String[0]);
    }

    /**
     * Takes the secret launch flags out of {@code args} and puts the settings they name into {@code properties}:
     * {@code --api-key} (repeatable, the keys join with commas as in the server), {@code --rpc} (which also sets
     * the layer split, as the flag does in the server), {@code --watermark-config}, {@code --kv-transfer-config}
     * and {@code --mcp-servers-json}. The values are validated here so a bad one is refused before the child
     * starts; an RPC refusal names the endpoints without their inline keys.
     *
     * @return {@code args} without those flags and their values, in order
     * @throws IllegalArgumentException when a flag has no value, a key is blank, an RPC entry is malformed or the
     *                                  watermark or KV transfer configuration is invalid
     */
    static String[] secretFlagsOf(String[] args, Properties properties) {
        List<String> rest = new ArrayList<>();
        List<String> apiKeys = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String flag = args[i];
            String property = SecretProperties.propertyOfFlag(flag);
            if (property == null) {
                rest.add(flag);
                continue;
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException(flag + " needs a value");
            }
            String value = args[++i];
            if (value.isBlank()) {
                throw new IllegalArgumentException(flag + " needs a value, got blank");
            }
            value = value.trim();
            if (flag.equals(SecretProperties.API_KEY_FLAG)) {
                apiKeys.add(value);
            } else if (flag.equals(SecretProperties.RPC_FLAG)) {
                properties.setProperty(EngineConfig.PROP_RPC, rpc(flag, value));
                properties.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
            } else if (flag.equals(SecretProperties.WATERMARK_CONFIG_FLAG)) {
                properties.setProperty(WatermarkConfig.PROP_CONFIG, watermarkConfig(flag, value));
            } else if (flag.equals(SecretProperties.KV_TRANSFER_CONFIG_FLAG)) {
                properties.setProperty(KvTransferConfig.SERVER_PROPERTY, kvTransferConfig(flag, value));
            } else {
                properties.setProperty(property, value);
            }
        }
        if (!apiKeys.isEmpty()) {
            properties.setProperty(SecretProperties.API_KEY_PROPERTY, String.join(",", apiKeys));
        }
        return rest.toArray(new String[0]);
    }

    /**
     * The {@code --rpc} value, every entry validated as the server validates it; a refusal names the entries with
     * their inline keys masked.
     */
    private static String rpc(String flag, String value) {
        String shown = RpcServerSpec.masked(value);
        for (String entry : value.split(",", -1)) {
            String server = entry.trim();
            if (server.isEmpty()) {
                throw new IllegalArgumentException(flag + " " + shown + " has an empty server entry");
            }
            try {
                RpcServerSpec.parse(server);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(flag + " " + shown + ": " + e.getMessage(), e);
            }
        }
        return value;
    }

    /** The {@code --watermark-config} value, checked to parse over this process's environment. */
    private static String watermarkConfig(String flag, String value) {
        Properties check = new Properties();
        check.setProperty(WatermarkConfig.PROP_CONFIG, value);
        try {
            WatermarkConfig.fromProperties(check, System.getenv());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
        return value;
    }

    /** The {@code --kv-transfer-config} value, checked to parse over this process's environment. */
    private static String kvTransferConfig(String flag, String value) {
        try {
            KvTransferConfig.parse(value, flag, System.getenv());
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
        return value;
    }

    /**
     * The child command: {@code java} with the forwarded {@code neuralfree.*} and {@code quarkus.*}
     * properties that carry no secret as {@code -D} definitions (JVM options, so they must precede
     * {@code -jar} or the main class), then {@code -jar <jar>} when {@code mainClass} is null or {@code -cp
     * <jar> <mainClass>} otherwise, and then the caller's arguments as the program's arguments. The secret
     * settings are not here: {@link #processBuilder} passes them in a file. On Java 17 or later the incubator
     * Vector API module rides along ({@link JvmVectorModule}), so the server's CPU kernels use it
     * ({@code CpuSimd}); on an older JVM the module does not exist and the launch stays untouched.
     */
    static List<String> command(Path jar, String mainClass, String[] args, Properties properties) {
        List<String> cmd = new ArrayList<>();
        cmd.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        cmd.addAll(JvmVectorModule.options(Runtime.version().feature()));
        for (String name : properties.stringPropertyNames()) {
            if (!SecretProperties.isSecret(name)) {
                cmd.add("-D" + name + "=" + properties.getProperty(name));
            }
        }
        if (mainClass == null) {
            cmd.add("-jar");
            cmd.add(jar.toString());
        } else {
            cmd.add("-cp");
            cmd.add(jar.toString());
            cmd.add(mainClass);
        }
        for (String a : args) {
            cmd.add(a);
        }
        return cmd;
    }

    /**
     * The child JVM's process: {@link #command}'s line, and, when the forwarded settings carry secrets, the
     * temporary owner-only properties file named by {@value SecretProperties#FILE_ENV}. Without secrets no file
     * is written and a stale variable of that name is kept out of the child's environment.
     */
    static ProcessBuilder processBuilder(Path jar, String mainClass, String[] args, Properties properties)
            throws IOException {
        ProcessBuilder pb = new ProcessBuilder(command(jar, mainClass, args, properties));
        pb.inheritIO();
        Properties secrets = SecretProperties.secrets(properties);
        if (secrets.isEmpty()) {
            pb.environment().remove(SecretProperties.FILE_ENV);
        } else {
            pb.environment().put(SecretProperties.FILE_ENV, SecretProperties.write(secrets).toString());
        }
        return pb;
    }

    /** Starts the child JVM over {@code jar}; its output stays on this process's streams. */
    static Process start(Path jar, String mainClass, String[] args, Properties properties)
            throws IOException {
        return processBuilder(jar, mainClass, args, properties).start();
    }

    /** The {@code neuralfree.*} and {@code quarkus.*} system properties of this JVM, forwarded to the child. */
    static Properties forwardedProperties() {
        Properties out = new Properties();
        for (String name : System.getProperties().stringPropertyNames()) {
            if (name.startsWith("neuralfree.") || name.startsWith("quarkus.")) {
                out.setProperty(name, System.getProperty(name));
            }
        }
        return out;
    }

    /**
     * The server jar of this installation: the explicit property and environment variable first,
     * then {@code server.jar} and every {@code *-runner.jar} of the distribution's {@code server}
     * directory, then the source checkout's Quarkus build output ({@code server/build} and its
     * {@code libs}) above any classpath root. The first candidate that exists as a regular file
     * wins; every other candidate is skipped, so a stale property falls through to the layout.
     */
    static Path locate(String jarProperty, String jarEnv, List<Path> classpathRoots) {
        List<Path> candidates = new ArrayList<>();
        if (jarProperty != null && !jarProperty.trim().isEmpty()) {
            candidates.add(Paths.get(jarProperty.trim()));
        }
        if (jarEnv != null && !jarEnv.trim().isEmpty()) {
            candidates.add(Paths.get(jarEnv.trim()));
        }
        for (Path root : classpathRoots) {
            candidates.add(root.resolve("server.jar"));
            Path parent = root.getParent();
            if (parent != null) {
                candidates.add(parent.resolve("server").resolve("server.jar"));
                candidates.addAll(glob(parent.resolve("server"), "*-runner.jar"));
            }
        }
        for (Path root : classpathRoots) {
            Path dir = root;
            for (int steps = 0; dir != null && steps < MAX_ANCESTORS; steps++) {
                // the source checkout's layout: the Quarkus build puts the runner jar in server/build
                // and the plain jar in server/build/libs
                candidates.addAll(glob(dir.resolve("server").resolve("build"), "*-runner.jar"));
                candidates.addAll(glob(dir.resolve("server").resolve("build").resolve("libs"),
                        "*-runner.jar"));
                dir = dir.getParent();
            }
        }
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        return null;
    }

    /** The refusal when no candidate exists: every searched place and how the jar is produced. */
    static String missingMessage(List<Path> classpathRoots) {
        StringBuilder sb = new StringBuilder("no server jar found; the serve and run-batch commands"
                + " run over the server's packaged uber-jar. Looked in:\n");
        sb.append("  -D").append(JAR_PROPERTY).append("=<path>\n");
        sb.append("  ").append(JAR_ENV).append("=<path>\n");
        for (Path root : classpathRoots) {
            sb.append("  ").append(root.resolve("server.jar")).append('\n');
            Path parent = root.getParent();
            if (parent != null) {
                sb.append("  ").append(parent.resolve("server").resolve("server.jar")).append('\n');
            }
        }
        sb.append("Build it with './gradlew :server:quarkusBuild' and install the distribution with"
                + " './gradlew installDist', which puts the jar in the distribution's server/ directory.");
        return sb.toString();
    }

    /** The sorted paths of {@code dir} matching {@code glob}, empty when the directory is absent. */
    private static List<Path> glob(Path dir, String glob) {
        if (dir == null || !Files.isDirectory(dir)) {
            return new ArrayList<>();
        }
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(p -> matches(p.getFileName().toString(), glob))
                    .sorted()
                    .collect(Collectors.toList());
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    /** The one-wildcard match of {@code name} against {@code glob}, whose only meta character is * . */
    private static boolean matches(String name, String glob) {
        int star = glob.indexOf('*');
        if (star < 0) {
            return name.equals(glob);
        }
        String head = glob.substring(0, star);
        String tail = glob.substring(star + 1);
        return name.length() >= head.length() + tail.length()
                && name.startsWith(head)
                && name.endsWith(tail);
    }

    /** The classpath's candidate roots: each directory entry, and each jar entry's parent directory. */
    static List<Path> classpathRoots() {
        return rootsOf(System.getProperty("java.class.path", ""));
    }

    /**
     * The candidate roots of a classpath: a directory entry is a root itself (the Gradle run), and a
     * jar entry's parent is a root (the start scripts run {@code APP_HOME/lib/*}, whose parent
     * {@code APP_HOME} holds the distribution's {@code server} directory).
     */
    static List<Path> rootsOf(String classPath) {
        List<Path> out = new ArrayList<>();
        for (String entry : classPath.split(java.io.File.pathSeparator)) {
            if (entry.isEmpty()) {
                continue;
            }
            Path path = Paths.get(entry);
            if (Files.isDirectory(path)) {
                out.add(path.toAbsolutePath().normalize());
            } else if (Files.isRegularFile(path)) {
                Path parent = path.toAbsolutePath().normalize().getParent();
                if (parent != null && !out.contains(parent)) {
                    out.add(parent);
                }
            }
        }
        return out;
    }
}
