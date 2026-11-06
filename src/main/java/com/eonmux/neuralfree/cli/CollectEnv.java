package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code collect-env} command, vLLM's {@code vllm collect-env} ({@code
 * vllm/entrypoints/cli/collect_env.py} over {@code vllm/collect_env.py}): the environment facts a
 * bug report needs — the NeuralFree version, the Java runtime, the operating system, the GPU devices
 * each API enumerates, the hub cache with its model count, the server jar the serve command resolves
 * and the engine's environment variables. Secret variables ({@code HF_TOKEN}, {@code
 * NEURALFREE_API_KEY}, {@code NEURALFREE_WATERMARK_KEY}, {@code NEURALFREE_KV_TRANSFER_TOKEN}, {@code
 * NEURALFREE_RPC_KEY}) report only whether they
 * are set, never their value; everything printed is read through the same accessors the engine itself uses. The
 * report goes to standard output; the command never writes a file and never touches the network.
 */
public final class CollectEnv {

    /** One immutable observation of this machine: every line of the report comes from it. */
    public static final class Snapshot {

        private final String version;
        private final Map<String, String> java;
        private final Map<String, String> os;
        private final List<DeviceListing.Entry> devices;
        private final String cacheRoot;
        private final int cachedModels;
        private final String serverJar;
        private final Map<String, String> env;

        /** A snapshot of the given observations; the maps are copied, so a snapshot never changes. */
        public Snapshot(String version, Map<String, String> java, Map<String, String> os,
                        List<DeviceListing.Entry> devices, String cacheRoot, int cachedModels,
                        String serverJar, Map<String, String> env) {
            this.version = version;
            this.java = new LinkedHashMap<>(java);
            this.os = new LinkedHashMap<>(os);
            this.devices = new ArrayList<>(devices);
            this.cacheRoot = cacheRoot;
            this.cachedModels = cachedModels;
            this.serverJar = serverJar;
            this.env = new LinkedHashMap<>(env);
        }

        /** The version {@link Version} reports. */
        public String version() {
            return version;
        }

        /** The Java runtime facts by label ({@code version}, {@code vendor}, {@code home}, {@code vm}). */
        public Map<String, String> java() {
            return java;
        }

        /** The operating system facts by label ({@code name}, {@code version}, {@code arch}). */
        public Map<String, String> os() {
            return os;
        }

        /** The devices every API enumerated ({@link DeviceListing#available}). */
        public List<DeviceListing.Entry> devices() {
            return devices;
        }

        /** The hub cache root's path, or the text of its absence. */
        public String cacheRoot() {
            return cacheRoot;
        }

        /** How many models the cache lists ({@link CacheListing#models}). */
        public int cachedModels() {
            return cachedModels;
        }

        /** The server jar the serve command resolves, or null when it resolves none. */
        public String serverJar() {
            return serverJar;
        }

        /** The environment variables by name, with their raw values; {@link #mask} hides the secrets. */
        public Map<String, String> env() {
            return env;
        }
    }

    /** The environment variables the report names, the engine's own and the hub's. */
    static final List<String> ENV_VARS = Arrays.asList(
            "NEURALFREE_MODEL", "NEURALFREE_API_KEY", "NEURALFREE_SERVER_JAR",
            WatermarkConfig.ENV_KEY, WatermarkConfig.ENV_KEY_FILE,
            KvTransferConfig.ENV_TOKEN, KvTransferConfig.ENV_TOKEN_FILE, RpcServerSpec.KEY_ENV, "HF_TOKEN",
            "HF_HUB_OFFLINE", "LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
            "XDG_CACHE_HOME", "TMPDIR");

    /** The variables whose value is never printed. */
    private static final List<String> SECRETS = Arrays.asList("NEURALFREE_API_KEY", "HF_TOKEN",
            WatermarkConfig.ENV_KEY, KvTransferConfig.ENV_TOKEN, RpcServerSpec.KEY_ENV);

    private CollectEnv() {
    }

    /** The snapshot of this machine; a failed cache read counts zero models, not a failed command. */
    public static Snapshot snapshot() {
        return new Snapshot(Version.current(), java(), os(), DeviceListing.available(), cacheRoot(),
                cachedModels(), serverJar(), env());
    }

    /** The Java runtime facts. */
    static Map<String, String> java() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("version", System.getProperty("java.version"));
        m.put("vendor", System.getProperty("java.vendor"));
        m.put("home", System.getProperty("java.home"));
        m.put("vm", System.getProperty("java.vm.name"));
        return m;
    }

    /** The operating system facts. */
    static Map<String, String> os() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("name", System.getProperty("os.name"));
        m.put("version", System.getProperty("os.version"));
        m.put("arch", System.getProperty("os.arch"));
        return m;
    }

    /** The hub cache root as {@link HubCache#defaultRoot} resolves it, or why it cannot be resolved. */
    static String cacheRoot() {
        try {
            return HubCache.defaultRoot().toString();
        } catch (IllegalStateException e) {
            return "<unresolvable: " + e.getMessage() + ">";
        }
    }

    /** The cached model count, 0 when the cache cannot be read. */
    static int cachedModels() {
        try {
            return CacheListing.models(new HubCache(HubCache.defaultRoot())).size();
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            return 0;
        }
    }

    /** The server jar of this installation, or null. */
    static String serverJar() {
        Path jar = ServeCommand.locate(System.getProperty(ServeCommand.JAR_PROPERTY),
                System.getenv(ServeCommand.JAR_ENV), ServeCommand.classpathRoots());
        return jar == null ? null : jar.toString();
    }

    /** The environment variables by name, with their raw values. */
    static Map<String, String> env() {
        Map<String, String> m = new LinkedHashMap<>();
        for (String name : ENV_VARS) {
            String value = System.getenv(name);
            m.put(name, value == null ? "<not set>" : value);
        }
        return m;
    }

    /** The printed value of one environment variable: a secret reports only whether it is set. */
    static String envValue(String name, String value) {
        if (value == null) {
            return "<not set>";
        }
        return SECRETS.contains(name) ? "<set>" : value;
    }

    /** The report as one string, the exact text {@code collect-env} prints. */
    public static String report(Snapshot s) {
        StringBuilder sb = new StringBuilder();
        sb.append("NeuralFree version: ").append(s.version()).append('\n');
        for (Map.Entry<String, String> e : s.java().entrySet()) {
            sb.append("Java ").append(e.getKey()).append(": ").append(e.getValue()).append('\n');
        }
        for (Map.Entry<String, String> e : s.os().entrySet()) {
            sb.append("OS ").append(e.getKey()).append(": ").append(e.getValue()).append('\n');
        }
        sb.append(DeviceListing.lines(s.devices()));
        sb.append("Hub cache: ").append(s.cacheRoot()).append(" (").append(s.cachedModels())
                .append(" models)\n");
        sb.append("Server jar: ").append(s.serverJar() == null ? "<not found>" : s.serverJar())
                .append('\n');
        sb.append("Environment:\n");
        for (Map.Entry<String, String> e : s.env().entrySet()) {
            sb.append("  ").append(e.getKey()).append(" = ")
                    .append(envValue(e.getKey(), e.getValue())).append('\n');
        }
        return sb.toString();
    }

    /** Runs the command: prints the environment report to standard output. */
    public static void run() {
        System.out.println("Collecting environment information...");
        System.out.print(report(snapshot()));
        System.out.flush();
    }
}
