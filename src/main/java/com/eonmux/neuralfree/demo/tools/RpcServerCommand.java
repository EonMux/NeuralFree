package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.rpc.RpcComputeServer;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;
import com.eonmux.neuralfree.gguf.rpc.RpcTls;
import com.eonmux.neuralfree.gguf.rpc.RpcWeightCache;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool rpc-server}: the remote compute server of a layer split (llama.cpp's {@code rpc-server}). It opens
 * the GPU devices of this machine and serves them over TCP to a client that runs a model with {@code neuralfree.rpc}
 * set, which then offloads a contiguous span of trunk layers to them. One process serves several devices
 * ({@code --device} with a comma-separated list, or every device of the backend); each answers its client by its
 * index in that list.
 *
 * <p>Every connection authenticates with the shared key ({@code --key} or {@value RpcServerSpec#KEY_ENV}); llama.cpp's
 * server has none, this one requires it, as the open transport is otherwise usable by anyone. The transport carries
 * optional TLS ({@code --tls-cert} and {@code --tls-key}, or {@link RpcTls#ENV_CERT} and {@link RpcTls#ENV_KEY};
 * the client names its trust store with {@link RpcTls#PROP_TRUSTSTORE}), and the HMAC handshake runs on top of it.
 * With {@code --cache} the server keeps uploaded weights in a directory ({@link RpcWeightCache#defaultCacheDir()} by
 * default) keyed by content hash, so a reconnecting client skips the re-upload. The server binds loopback by default;
 * pass {@code --host 0.0.0.0} to accept a remote client and keep the key secret between the machines.</p>
 */
final class RpcServerCommand {

    /** llama.cpp rpc-server's default port. */
    static final int DEFAULT_PORT = 50052;
    /** The interface the server binds unless {@code --host} names another; loopback keeps it off the network. */
    static final String DEFAULT_HOST = "127.0.0.1";
    /** The device list the server serves unless {@code --device} names one: every device of the backend. */
    static final String DEFAULT_DEVICE = "";

    /** The backend modes rpc-server serves: every {@link EngineConfig.BackendMode} but the CPU engine. */
    private static final String GPU_BACKENDS = gpuBackendNames();

    /** The usage text {@code --help} prints; the backend line names every backend mode but the CPU engine. */
    static final String USAGE =
            "nftool rpc-server [options]\n"
            + "options (llama.cpp rpc-server names):\n"
            + "  --host, -H <addr>                interface to bind, default " + DEFAULT_HOST + "\n"
            + "  --port, -p <port>                TCP port, default " + DEFAULT_PORT + "\n"
            + "  --backend, -b <mode>             " + GPU_BACKENDS + ", default auto\n"
            + "  --device, -d <list>              comma-separated device indices or name substrings to serve,\n"
            + "                                   default every device of the backend\n"
            + "  --cache, -c [ <dir> ]            keep uploaded weights in <dir> (default "
            + RpcWeightCache.defaultCacheDir() + ") keyed by content hash, so a\n"
            + "                                   reconnecting client skips the re-upload\n"
            + "  --tls-cert <file>                PEM certificate chain of the TLS transport (or "
            + RpcTls.ENV_CERT + ")\n"
            + "  --tls-key <file>                 unencrypted PKCS#8 PEM private key (or " + RpcTls.ENV_KEY + ")\n"
            + "  --key <token>                    shared key every client must hold (or " + RpcServerSpec.KEY_ENV
            + ")\n"
            + "  --help, -h                       this text\n"
            + "The client sets neuralfree.rpc=[tls://]host:port@key, neuralfree.split.mode=layer, and for TLS"
            + " neuralfree.rpc.tls.truststore=<store>.\n";

    private RpcServerCommand() {
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line. */
    static final class Options {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        EngineConfig.BackendMode backend = EngineConfig.BackendMode.AUTO;
        List<String> devices = new ArrayList<>();
        String key;
        boolean cache;
        Path cacheDir;
        String tlsCert;
        String tlsKey;

        /** The key of {@code --key}, else {@value RpcServerSpec#KEY_ENV}; never blank. */
        String key() throws UsageException {
            String resolved = key != null ? key : System.getenv(RpcServerSpec.KEY_ENV);
            if (resolved == null || resolved.trim().isEmpty()) {
                throw new UsageException("rpc-server needs a shared key: pass --key or set "
                        + RpcServerSpec.KEY_ENV + "; the transport is authenticated");
            }
            return resolved.trim();
        }
    }

    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                execute(o);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the command line.
     *
     * @return the options, or null after {@code --help}
     * @throws UsageException when a flag is unknown, a value is missing, or the backend is the CPU engine
     */
    static Options parse(String[] args) throws UsageException {
        Options o = new Options();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--help":
                case "-h":
                    System.out.print(USAGE);
                    return null;
                case "--host":
                case "-H":
                    o.host = value(args, ++i, arg);
                    break;
                case "--port":
                case "-p":
                    o.port = port(value(args, ++i, arg));
                    break;
                case "--backend":
                case "-b":
                    o.backend = backend(value(args, ++i, arg));
                    break;
                case "--device":
                case "-d":
                    o.devices = devices(value(args, ++i, arg));
                    break;
                case "--cache":
                case "-c":
                    o.cache = true;
                    // the directory is optional; a following token is its value unless it is another flag
                    if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                        o.cacheDir = Paths.get(value(args, ++i, arg));
                    }
                    break;
                case "--tls-cert":
                    o.tlsCert = value(args, ++i, arg);
                    break;
                case "--tls-key":
                    o.tlsKey = value(args, ++i, arg);
                    break;
                case "--key":
                    o.key = value(args, ++i, arg);
                    break;
                default:
                    throw new UsageException("unknown rpc-server option '" + arg + "'");
            }
        }
        o.key = o.key();
        return o;
    }

    /** The comma-separated device selectors of {@code --device}; an empty list serves every device. */
    private static List<String> devices(String text) throws UsageException {
        if (text.trim().isEmpty()) {
            throw new UsageException("--device needs a comma-separated device list, or no devices");
        }
        return Arrays.asList(text.trim().split(","));
    }

    private static String value(String[] args, int index, String flag) throws UsageException {
        if (index >= args.length || args[index].trim().isEmpty()) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index].trim();
    }

    private static int port(String text) throws UsageException {
        int port;
        try {
            port = Integer.parseInt(text);
        } catch (NumberFormatException e) {
            throw new UsageException("--port '" + text + "' is not a number");
        }
        if (port < 1 || port > 65535) {
            throw new UsageException("--port " + port + " is outside 1..65535");
        }
        return port;
    }

    /** The backend names rpc-server accepts, joined {@code " | "}: {@link EngineConfig.BackendMode#names()} without cpu. */
    private static String gpuBackendNames() {
        List<String> names = new ArrayList<>();
        for (String name : EngineConfig.BackendMode.names()) {
            if (!name.equals("cpu")) {
                names.add(name);
            }
        }
        return String.join(" | ", names);
    }

    private static EngineConfig.BackendMode backend(String text) throws UsageException {
        EngineConfig.BackendMode mode;
        try {
            mode = EngineConfig.BackendMode.parse(text);
        } catch (IllegalArgumentException e) {
            throw new UsageException("--backend '" + text + "' is not " + GPU_BACKENDS);
        }
        if (!mode.usesGpu()) {
            throw new UsageException("--backend " + text.toLowerCase(Locale.ROOT) + " is the CPU engine;"
                    + " rpc-server serves GPU devices");
        }
        return mode;
    }

    /**
     * Opens the selected devices and serves them until the process ends.
     *
     * @throws IOException when the listening socket cannot be bound
     */
    static void execute(Options o) throws IOException {
        String key = o.key;
        RpcTls.Server tls = tls(o);
        RpcWeightCache cache = o.cache ? new RpcWeightCache(
                o.cacheDir != null ? o.cacheDir : RpcWeightCache.defaultCacheDir()) : null;
        HWAcceleration acceleration = HWAcceleration.forMode(o.backend.toAccelerationMode());
        List<GpuDevices.Selected> devices;
        try {
            devices = GpuDevices.selectServerDevices(acceleration, o.devices, "--device");
        } catch (RuntimeException e) {
            acceleration.shutdown();
            throw e;
        }
        List<com.eonmux.neuralfree.acceleration.backend.ComputeBackend> backends = new ArrayList<>();
        for (GpuDevices.Selected device : devices) {
            backends.add(device.backend());
        }
        RpcComputeServer server = new RpcComputeServer(backends, key, o.host, o.port, cache,
                tls == null ? null : tls.context());
        Runtime.getRuntime().addShutdownHook(new Thread(server::close, "neuralfree-rpc-server-shutdown"));
        try {
            server.start();
            System.out.println("rpc-server: listening on " + o.host + ":" + server.port() + ", "
                    + backends.size() + " device(s): " + names(devices)
                    + (cache != null ? ", weight cache " + cache.directory() : "")
                    + (tls != null ? ", tls" : ""));
            System.out.flush();
            server.awaitTermination();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            server.close();
            acceleration.shutdown();
        }
    }

    /** The server TLS of {@code --tls-cert}/{@code --tls-key} and the environment names; null when unconfigured. */
    private static RpcTls.Server tls(Options o) {
        if (o.tlsCert != null && o.tlsKey != null) {
            return RpcTls.server(Paths.get(o.tlsCert), Paths.get(o.tlsKey));
        }
        if (o.tlsCert != null || o.tlsKey != null) {
            throw new IllegalArgumentException("rpc-server TLS needs both --tls-cert and --tls-key; only "
                    + (o.tlsCert != null ? "--tls-cert" : "--tls-key") + " is given");
        }
        return RpcTls.fromEnvironment();
    }

    /** The device names of {@code devices}, for the startup line. */
    private static String names(List<GpuDevices.Selected> devices) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < devices.size(); i++) {
            sb.append(i == 0 ? "" : ", ").append(devices.get(i).name());
        }
        return sb.append(']').toString();
    }
}
