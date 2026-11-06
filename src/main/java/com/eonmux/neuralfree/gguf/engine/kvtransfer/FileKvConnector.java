package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The shared-storage KV connector: a producer writes each exported chain to a file under
 * {@code kv_connector_extra_config.storage_path}, and a consumer reads it from the same directory. This is the
 * deployment where the prefill and decode instances mount one network filesystem (an LMCache-style shared cache
 * without the Python store), so the payload travels through the filesystem instead of a socket.
 *
 * <p>A file is written under a temporary name and moved into place, so a consumer never reads a half-written chain.
 * The file name is the URL-safe base64 of the transfer id plus {@code .nfkv} — letters, digits, {@code -} and
 * {@code _} only, so no transfer id can name a file outside the directory. The payload's frame is the one
 * {@link KvTransferProtocol} defines. Extra configuration: {@code storage_path} (required),
 * {@code max_payload_bytes} (default 8 GiB) and {@code allowed_remote_hosts} (accepted for configurations shared
 * with a socket connector, and validated; shared storage connects nowhere, so it allows nothing here). Any other
 * key refuses by name.</p>
 */
public final class FileKvConnector implements KvConnector {

    private static final Set<String> EXTRA_FIELDS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "storage_path", "max_payload_bytes", KvRemoteHosts.FIELD)));
    private static final long DEFAULT_MAX_PAYLOAD = 8L * 1024 * 1024 * 1024;
    private static final String SUFFIX = ".nfkv";

    private final KvTransferConfig config;
    private final Path directory;
    private final long maxPayloadBytes;
    private final AtomicLong saved = new AtomicLong();
    private final AtomicLong loaded = new AtomicLong();

    /**
     * The connector over {@code config}'s storage directory; the directory is created when absent.
     *
     * @throws IllegalArgumentException for a missing or unknown extra-config field, or an out-of-range value
     * @throws IOException              when the directory cannot be created
     */
    public FileKvConnector(KvTransferConfig config) throws IOException {
        this.config = Objects.requireNonNull(config, "config");
        for (String key : config.extraConfig().keySet()) {
            if (!EXTRA_FIELDS.contains(key)) {
                throw new IllegalArgumentException("FileKvConnector has no kv_connector_extra_config field " + key
                        + " (it reads " + String.join(", ", EXTRA_FIELDS) + ")");
            }
        }
        KvRemoteHosts.of(config);
        String path = config.extraText("storage_path", null);
        if (path == null) {
            throw new IllegalArgumentException("FileKvConnector needs kv_connector_extra_config.storage_path");
        }
        this.directory = Path.of(path);
        Files.createDirectories(directory);
        long max = config.extraLong("max_payload_bytes", DEFAULT_MAX_PAYLOAD);
        if (max < 1) {
            throw new IllegalArgumentException("kv_connector_extra_config.max_payload_bytes must be positive, got "
                    + max);
        }
        this.maxPayloadBytes = max;
    }

    @Override
    public String name() {
        return "FileKvConnector";
    }

    @Override
    public String engineId() {
        return config.engineId();
    }

    @Override
    public KvRole role() {
        return config.role();
    }

    /** Shared storage has no address. */
    @Override
    public KvTransferEndpoint endpoint() {
        return KvTransferEndpoint.none();
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
        Path target = fileOf(payload.transferId());
        Path temp = Files.createTempFile(directory, "kv-", ".tmp");
        try {
            try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(temp))) {
                KvTransferProtocol.writeFile(out, payload);
            }
            try {
                Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temp);
        }
        saved.incrementAndGet();
    }

    @Override
    public KvTransferPayload load(String transferId, String expectedEngineId, KvTransferEndpoint remote)
            throws IOException {
        Objects.requireNonNull(transferId, "transferId");
        if (!config.consumes()) {
            throw new UnsupportedOperationException("this instance's kv_role is " + config.role().wireName()
                    + "; it does not import KV");
        }
        Path file = fileOf(transferId);
        if (!Files.isRegularFile(file)) {
            throw new IOException("no KV transfer is stored under transfer id " + transferId + " in " + directory);
        }
        KvTransferPayload payload;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(file))) {
            payload = KvTransferProtocol.readFile(in, maxPayloadBytes, file.toString());
        }
        if (!payload.transferId().equals(transferId)) {
            throw new IOException(file + " holds transfer " + payload.transferId() + ", not " + transferId);
        }
        if (expectedEngineId != null && !expectedEngineId.equals(payload.engineId())) {
            throw new IOException("transfer " + transferId + " comes from engine " + payload.engineId()
                    + ", the request names remote_engine_id " + expectedEngineId);
        }
        loaded.incrementAndGet();
        return payload;
    }

    @Override
    public void release(String transferId) {
        try {
            Files.deleteIfExists(fileOf(transferId));
        } catch (IOException e) {
            throw new java.io.UncheckedIOException("cannot drop transfer " + transferId + " from " + directory, e);
        }
    }

    @Override
    public Map<String, Object> stats() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("connector", name());
        out.put("engine_id", config.engineId());
        out.put("kv_role", config.role().wireName());
        out.put("storage_path", directory.toString());
        out.put("saved", saved.get());
        out.put("loaded", loaded.get());
        out.put("max_payload_bytes", maxPayloadBytes);
        return Collections.unmodifiableMap(out);
    }

    @Override
    public void close() {
        // nothing to release: the payloads belong to the shared storage and its owner
    }

    private Path fileOf(String transferId) {
        String name = Base64.getUrlEncoder().withoutPadding()
                .encodeToString(transferId.getBytes(StandardCharsets.UTF_8));
        return directory.resolve(name + SUFFIX);
    }

    @Override
    public String toString() {
        return "FileKvConnector{" + config.role().wireName() + ", " + directory + "}";
    }
}
