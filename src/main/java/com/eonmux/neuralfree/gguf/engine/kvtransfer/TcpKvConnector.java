package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The socket KV connector (vLLM's pull/push connectors reduced to a pull): a producer stores every exported chain in
 * a bounded, leased in-memory store and listens on {@code kv_ip:kv_port}; a consumer connects there, names the
 * transfer id in a {@code GET} frame and reads the {@code PAYLOAD} frame ({@link KvTransferProtocol}).
 *
 * <p>The producer's listener starts with the connector, so a port conflict fails the server's start rather than its
 * first transfer. Each accepted connection is handled on a small daemon pool. A stored payload stays until a
 * consumer pulls it, its lease (default 300 seconds) runs out, or the store passes {@code kv_buffer_size}; the
 * consumer's {@link #release} drops it earlier. An optional {@code auth_token} extra (or the
 * {@link KvTransferConfig#ENV_TOKEN} / {@link KvTransferConfig#ENV_TOKEN_FILE} environment source) makes every pull
 * present the token, so a network peer that can reach the port cannot read another request's KV; presented and
 * expected tokens compare in constant time. Because a producer without a token serves every stored chain to any
 * peer that reaches the port, one whose {@code kv_ip} is not a loopback address refuses to start until a token is
 * set. The engine's thread only touches the store's map ({@link #save}); all network I/O happens on the pool's
 * threads or the consumer's caller.</p>
 *
 * <p>The consumer side only pulls from the peers {@code allowed_remote_hosts} allows (see {@link KvRemoteHosts}):
 * the request's {@code remote_host} is resolved once, the resolved address is checked, and that address is what
 * the socket connects to.</p>
 *
 * <p>Extra configuration ({@code kv_connector_extra_config}), all optional: {@code auth_token} (the shared token),
 * {@code allowed_remote_hosts} (the peers a decode side may pull from), {@code connect_timeout_ms} (default 10000),
 * {@code io_timeout_ms} (default 120000), {@code server_threads} (default 4), {@code kv_lease_seconds} (default
 * 300) and {@code max_payload_bytes} (the largest chain accepted, default 8 GiB). Any other key refuses by
 * name.</p>
 */
public final class TcpKvConnector implements KvConnector {

    private static final Logger LOGGER = Logger.getLogger(TcpKvConnector.class.getName());
    private static final Set<String> EXTRA_FIELDS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "auth_token", KvRemoteHosts.FIELD, "connect_timeout_ms", "io_timeout_ms", "server_threads",
            "kv_lease_seconds", "max_payload_bytes")));
    private static final long DEFAULT_MAX_PAYLOAD = 8L * 1024 * 1024 * 1024;

    /** One stored payload and the instant its lease ends. */
    private static final class Stored {
        final KvTransferPayload payload;
        final long expiresAtNanos;

        Stored(KvTransferPayload payload, long expiresAtNanos) {
            this.payload = payload;
            this.expiresAtNanos = expiresAtNanos;
        }
    }

    private final KvTransferConfig config;
    private final String authToken;
    private final KvRemoteHosts allowed;
    private final int connectTimeoutMs;
    private final int ioTimeoutMs;
    private final long leaseNanos;
    private final long maxPayloadBytes;
    private final long bufferBytes;
    private final ServerSocket listener;
    private final int boundPort;
    private final ExecutorService handlers;
    private final Thread acceptor;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicLong saved = new AtomicLong();
    private final AtomicLong loaded = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();
    private long storedBytes;
    /** Access-ordered, so the eldest entry is the least recently stored or pulled. */
    private final LinkedHashMap<String, Stored> store = new LinkedHashMap<>(16, 0.75f, true);

    /**
     * The connector of {@code config}: a producer binds its listener now, after the token rule held — a producer
     * whose {@code kv_ip} is not a loopback address needs a shared token, or it would serve every stored chain to
     * any peer that reaches the port.
     *
     * @throws IllegalArgumentException      for an unknown extra-config key, an out-of-range value, an
     *                                       {@code allowed_remote_hosts} entry that names no address, or a
     *                                       non-loopback producer without a token
     * @throws IOException                   when the listener cannot bind
     */
    public TcpKvConnector(KvTransferConfig config) throws IOException {
        this.config = Objects.requireNonNull(config, "config");
        checkExtra(config.extraConfig());
        this.authToken = config.authToken() != null ? config.authToken() : config.extraText("auth_token", null);
        this.allowed = KvRemoteHosts.of(config);
        this.connectTimeoutMs = positiveInt(config, "connect_timeout_ms", 10_000);
        this.ioTimeoutMs = positiveInt(config, "io_timeout_ms", 120_000);
        int threads = positiveInt(config, "server_threads", 4);
        long leaseSeconds = config.extraLong("kv_lease_seconds", 300);
        if (leaseSeconds <= 0) {
            throw new IllegalArgumentException("kv_connector_extra_config.kv_lease_seconds must be positive, got "
                    + leaseSeconds);
        }
        this.leaseNanos = TimeUnit.SECONDS.toNanos(leaseSeconds);
        this.maxPayloadBytes = positiveLong(config, "max_payload_bytes", DEFAULT_MAX_PAYLOAD);
        this.bufferBytes = config.kvBufferSize();
        if (config.produces()) {
            requireTokenOnNonLoopback();
            ServerSocket socket = new ServerSocket();
            socket.setReuseAddress(true);
            socket.bind(new InetSocketAddress(config.kvIp(), config.kvPort()));
            this.listener = socket;
            this.boundPort = socket.getLocalPort();
            this.handlers = Executors.newFixedThreadPool(threads, daemonFactory("kv-transfer-" + boundPort));
            this.acceptor = daemonFactory("kv-transfer-accept-" + boundPort).newThread(this::acceptLoop);
            this.acceptor.start();
        } else {
            this.listener = null;
            this.boundPort = -1;
            this.handlers = null;
            this.acceptor = null;
        }
    }

    /**
     * Refuses a producer whose listener would face a non-loopback network with no shared token: such a listener
     * serves every stored chain — other requests' prompt KV — to any peer that reaches the port.
     *
     * @throws IllegalArgumentException naming the fix (a token, or a loopback {@code kv_ip})
     */
    private void requireTokenOnNonLoopback() {
        if (authToken != null || KvRemoteHosts.isLoopback(config.kvIp())) {
            return;
        }
        throw new IllegalArgumentException("kv_ip " + config.kvIp() + " is not a loopback address and this"
                + " producer has no shared token: any peer that reaches the port could pull every stored chain"
                + " (other requests' prompt KV). Set kv_connector_extra_config.auth_token (or the "
                + KvTransferConfig.ENV_TOKEN + " or " + KvTransferConfig.ENV_TOKEN_FILE + " environment variable),"
                + " or bind kv_ip to a loopback address");
    }

    private static void checkExtra(Map<String, Object> extra) {
        for (String key : extra.keySet()) {
            if (!EXTRA_FIELDS.contains(key)) {
                throw new IllegalArgumentException("TcpKvConnector has no kv_connector_extra_config field " + key
                        + " (it reads " + String.join(", ", EXTRA_FIELDS) + ")");
            }
        }
    }

    private static int positiveInt(KvTransferConfig config, String key, int fallback) {
        long value = config.extraLong(key, fallback);
        if (value < 1 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("kv_connector_extra_config." + key + " must be positive, got " + value);
        }
        return (int) value;
    }

    private static long positiveLong(KvTransferConfig config, String key, long fallback) {
        long value = config.extraLong(key, fallback);
        if (value < 1) {
            throw new IllegalArgumentException("kv_connector_extra_config." + key + " must be positive, got " + value);
        }
        return value;
    }

    private static ThreadFactory daemonFactory(String name) {
        return runnable -> {
            Thread thread = new Thread(runnable, name);
            thread.setDaemon(true);
            return thread;
        };
    }

    private void acceptLoop() {
        while (!closed.get()) {
            try {
                Socket socket = listener.accept();
                handlers.execute(() -> handle(socket));
            } catch (IOException e) {
                if (!closed.get()) {
                    LOGGER.log(Level.WARNING, "KV transfer accept failed on port " + boundPort, e);
                }
                return;
            } catch (RuntimeException e) {
                // the pool refused (shutting down): leave the loop
                if (!closed.get()) {
                    LOGGER.log(Level.WARNING, "KV transfer listener stopped", e);
                }
                return;
            }
        }
    }

    private void handle(Socket socket) {
        try (Socket open = socket;
             DataInputStream in = new DataInputStream(new BufferedInputStream(open.getInputStream()));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(open.getOutputStream()))) {
            open.setSoTimeout(ioTimeoutMs);
            if (KvTransferProtocol.header(in, "the KV transfer client") != KvTransferProtocol.GET) {
                KvTransferProtocol.writeError(out, "the KV transfer server expects a GET frame");
                return;
            }
            String[] get = KvTransferProtocol.readGet(in);
            if (authToken != null && !tokenMatches(get[1])) {
                rejected.incrementAndGet();
                KvTransferProtocol.writeError(out, "the KV transfer token does not match");
                return;
            }
            Stored stored;
            synchronized (store) {
                stored = store.get(get[0]);
                if (stored != null && System.nanoTime() - stored.expiresAtNanos > 0) {
                    storedBytes -= stored.payload.bytes();
                    store.remove(get[0]);
                    stored = null;
                }
            }
            if (stored == null) {
                rejected.incrementAndGet();
                KvTransferProtocol.writeError(out, "no KV transfer is stored under transfer id " + get[0]);
                return;
            }
            KvTransferProtocol.writePayload(out, stored.payload);
            loaded.incrementAndGet();
        } catch (EOFException e) {
            LOGGER.log(Level.FINE, "KV transfer client disconnected mid-frame", e);
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "KV transfer to a client failed", e);
        }
    }

    /** Whether the presented token is the shared one; both sides compare in constant time, as bytes. */
    private boolean tokenMatches(String presented) {
        byte[] expected = authToken.getBytes(StandardCharsets.UTF_8);
        byte[] given = presented == null ? new byte[0] : presented.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expected, given);
    }

    @Override
    public String name() {
        return "TcpKvConnector";
    }

    @Override
    public String engineId() {
        return config.engineId();
    }

    @Override
    public KvRole role() {
        return config.role();
    }

    @Override
    public KvTransferEndpoint endpoint() {
        return boundPort > 0 ? KvTransferEndpoint.of(config.kvIp(), boundPort) : KvTransferEndpoint.none();
    }

    @Override
    public void save(KvTransferPayload payload) throws IOException {
        Objects.requireNonNull(payload, "payload");
        if (!config.produces()) {
            throw new UnsupportedOperationException("this instance's kv_role is " + config.role().wireName()
                    + "; it does not export KV");
        }
        if (payload.chain().length > maxPayloadBytes) {
            throw new IOException("the chain of transfer " + payload.transferId() + " holds "
                    + payload.chain().length + " bytes, over the configured max_payload_bytes " + maxPayloadBytes);
        }
        long now = System.nanoTime();
        synchronized (store) {
            Stored previous = store.remove(payload.transferId());
            if (previous != null) {
                storedBytes -= previous.payload.bytes();
            }
            expire(now);
            store.put(payload.transferId(), new Stored(payload, now + leaseNanos));
            storedBytes += payload.bytes();
            trimToBuffer();
        }
        saved.incrementAndGet();
    }

    private void expire(long now) {
        Iterator<Map.Entry<String, Stored>> it = store.entrySet().iterator();
        while (it.hasNext()) {
            Stored value = it.next().getValue();
            if (now - value.expiresAtNanos > 0) {
                storedBytes -= value.payload.bytes();
                it.remove();
            }
        }
    }

    private void trimToBuffer() {
        if (bufferBytes == 0) {
            return;
        }
        Iterator<Map.Entry<String, Stored>> it = store.entrySet().iterator();
        while (storedBytes > bufferBytes && it.hasNext()) {
            Stored value = it.next().getValue();
            storedBytes -= value.payload.bytes();
            it.remove();
        }
    }

    @Override
    public KvTransferPayload load(String transferId, String expectedEngineId, KvTransferEndpoint remote)
            throws IOException {
        Objects.requireNonNull(transferId, "transferId");
        if (!config.consumes()) {
            throw new UnsupportedOperationException("this instance's kv_role is " + config.role().wireName()
                    + "; it does not import KV");
        }
        if (remote == null || !remote.addressed()) {
            throw new IOException("the decode request names no remote KV transfer address (remote_host and"
                    + " remote_port), which TcpKvConnector needs");
        }
        InetAddress address = checkedAddress(remote);
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(address, remote.port()), connectTimeoutMs);
            socket.setSoTimeout(ioTimeoutMs);
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                 DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()))) {
                KvTransferProtocol.writeGet(out, transferId, authToken);
                int type;
                try {
                    type = KvTransferProtocol.header(in, "the KV transfer server");
                } catch (EOFException e) {
                    throw KvTransferProtocol.truncated(e, "the KV transfer server");
                }
                if (type == KvTransferProtocol.ERROR) {
                    throw new IOException("the KV transfer server refused transfer " + transferId + ": "
                            + KvTransferProtocol.readError(in));
                }
                if (type != KvTransferProtocol.PAYLOAD) {
                    throw new IOException("the KV transfer server answered frame type " + type);
                }
                KvTransferPayload payload = KvTransferProtocol.readPayload(in, maxPayloadBytes,
                        "the KV transfer server");
                if (!payload.transferId().equals(transferId)) {
                    throw new IOException("asked the KV transfer server for transfer " + transferId + " and got "
                            + payload.transferId());
                }
                if (expectedEngineId != null && !expectedEngineId.equals(payload.engineId())) {
                    throw new IOException("transfer " + transferId + " comes from engine " + payload.engineId()
                            + ", the request names remote_engine_id " + expectedEngineId);
                }
                return payload;
            }
        }
    }

    /**
     * The address of {@code remote} the connection uses: the request's {@code remote_host} resolved once, checked
     * against {@code allowed_remote_hosts}, and that checked address returned — so DNS cannot answer the check
     * with one address and the connect with another. Refusal happens before any connection attempt.
     *
     * @throws IllegalArgumentException naming {@code kv_connector_extra_config.allowed_remote_hosts} when the
     *                                  resolved address is not allowed
     * @throws IOException              when {@code remote_host} does not resolve
     */
    private InetAddress checkedAddress(KvTransferEndpoint remote) throws IOException {
        try {
            return allowed.select(remote.host(), remote.port());
        } catch (UnknownHostException e) {
            throw new IOException("cannot resolve the decode request's remote_host " + remote.host(), e);
        }
    }

    @Override
    public void release(String transferId) {
        synchronized (store) {
            Stored stored = store.remove(transferId);
            if (stored != null) {
                storedBytes -= stored.payload.bytes();
            }
        }
    }

    @Override
    public Map<String, Object> stats() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("connector", name());
        out.put("engine_id", config.engineId());
        out.put("kv_role", config.role().wireName());
        out.put("endpoint", endpoint().toString());
        synchronized (store) {
            out.put("stored_transfers", store.size());
            out.put("stored_bytes", storedBytes);
        }
        out.put("saved", saved.get());
        out.put("loaded", loaded.get());
        out.put("rejected", rejected.get());
        out.put("buffer_bytes", bufferBytes);
        return Collections.unmodifiableMap(out);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (listener != null) {
            try {
                listener.close();
            } catch (IOException e) {
                LOGGER.log(Level.FINE, "closing the KV transfer listener failed", e);
            }
        }
        if (handlers != null) {
            handlers.shutdownNow();
        }
        if (acceptor != null) {
            acceptor.interrupt();
        }
        synchronized (store) {
            store.clear();
            storedBytes = 0;
        }
    }

    @Override
    public String toString() {
        return "TcpKvConnector{" + config.role().wireName() + ", " + endpoint() + "}";
    }
}
