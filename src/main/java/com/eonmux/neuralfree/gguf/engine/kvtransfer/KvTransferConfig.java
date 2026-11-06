package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/**
 * vLLM's {@code KVTransferConfig} ({@code --kv-transfer-config}): which connector carries a prefill instance's KV
 * cache to a decode instance, which side of the transfer this instance is, and the transport settings the connector
 * reads.
 *
 * <p>Fields, under vLLM's names: {@code kv_connector} (the connector, {@code TcpKvConnector} or
 * {@code FileKvConnector}), {@code kv_role} ({@link KvRole}), {@code engine_id} (default a fresh UUID),
 * {@code kv_port} (default 14579, 0 picks an ephemeral port), {@code kv_ip} (the host a producer binds and
 * advertises, {@code 127.0.0.1} when absent), {@code kv_rank} and {@code kv_parallel_size} (default 0 and 1),
 * {@code kv_buffer_size} (the producer's in-memory transfer store, default 1 GiB), {@code kv_load_failure_policy}
 * ({@code fail} or {@code recompute}) and {@code kv_connector_extra_config} (the connector's own settings). A config
 * with a connector but no role is refused, as vLLM refuses it. The vLLM fields this build does not run refuse by
 * name: {@code kv_connector_module_path} (dynamic Python modules) and {@code enable_permute_local_kv} (the
 * HND/NHD layout permutation). Immutable.
 *
 * <p>The socket connector's shared token is best kept out of the configuration JSON, which rides the command line
 * and the {@value #SERVER_PROPERTY} property. {@link #parse(String, String, Map)} therefore also reads it from the
 * {@value #ENV_TOKEN} environment variable or from the file the {@value #ENV_TOKEN_FILE} environment variable
 * names; the JSON {@code auth_token} remains accepted for vLLM parity. {@value #ENV_TOKEN} takes precedence over
 * {@value #ENV_TOKEN_FILE}, and an environment source beside the JSON {@code auth_token} is refused rather than
 * silently choosing one.
 */
public final class KvTransferConfig {

    /** The vLLM launch flag that sets the configuration. */
    public static final String FLAG = "--kv-transfer-config";

    /** The server property carrying the configuration as a JSON object. */
    public static final String SERVER_PROPERTY = "neuralfree.server.kv.transfer.config";

    /** The environment variable carrying the socket connector's shared token. */
    public static final String ENV_TOKEN = "NEURALFREE_KV_TRANSFER_TOKEN";

    /** The environment variable naming a file whose text is the shared token (trimmed). */
    public static final String ENV_TOKEN_FILE = "NEURALFREE_KV_TRANSFER_TOKEN_FILE";

    private static final Set<String> FIELDS = fields();
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final long DEFAULT_BUFFER_BYTES = 1024L * 1024 * 1024;
    /** vLLM's {@code kv_port}. */
    public static final int DEFAULT_PORT = 14579;

    private final String kvConnector;
    private final String engineId;
    private final KvRole role;
    private final String kvIp;
    private final int kvPort;
    private final int kvRank;
    private final int kvParallelSize;
    private final long kvBufferSize;
    private final String loadFailurePolicy;
    private final Map<String, Object> extraConfig;
    private final String authToken;

    private KvTransferConfig(String kvConnector, String engineId, KvRole role, String kvIp, int kvPort, int kvRank,
                             int kvParallelSize, long kvBufferSize, String loadFailurePolicy,
                             Map<String, Object> extraConfig, String authToken) {
        this.kvConnector = kvConnector;
        this.engineId = engineId;
        this.role = role;
        this.kvIp = kvIp;
        this.kvPort = kvPort;
        this.kvRank = kvRank;
        this.kvParallelSize = kvParallelSize;
        this.kvBufferSize = kvBufferSize;
        this.loadFailurePolicy = loadFailurePolicy;
        this.extraConfig = extraConfig;
        this.authToken = authToken;
    }

    private static Set<String> fields() {
        Set<String> out = new LinkedHashSet<>();
        Collections.addAll(out, "kv_connector", "engine_id", "kv_role", "kv_ip", "kv_port", "kv_rank",
                "kv_parallel_size", "kv_buffer_size", "kv_connector_extra_config", "kv_load_failure_policy",
                "kv_connector_module_path", "enable_permute_local_kv");
        return Collections.unmodifiableSet(out);
    }

    /**
     * The configuration {@code json} holds, {@code refusedBy} (the flag or property) named in every refusal, over
     * no environment: the shared token can only come from the JSON's {@code kv_connector_extra_config.auth_token}.
     *
     * @throws IllegalArgumentException when the JSON is not an object, names an unknown or mistyped field, or a
     *                                  field is out of range
     * @throws UnsupportedOperationException for a vLLM field this build does not run, named with the reason
     */
    public static KvTransferConfig parse(String json, String refusedBy) {
        return parse(json, refusedBy, Collections.emptyMap());
    }

    /**
     * {@link #parse(String, String)} over an explicit environment, so the token sources are testable: the shared
     * token of the socket connector may come from the {@value #ENV_TOKEN} environment variable or from the file
     * named by {@value #ENV_TOKEN_FILE}, which is trimmed and read as its text; the variable wins over the file.
     * The JSON {@code kv_connector_extra_config.auth_token} is accepted for vLLM parity and may not be combined
     * with an environment source. An environment source without a JSON token does not configure a transfer on its
     * own (a transfer still needs the JSON's connector and role); it only supplies the token of the JSON that is
     * there.
     *
     * @throws IllegalArgumentException when the JSON is not an object, names an unknown or mistyped field, a field
     *                                  is out of range, or the token source is malformed or given twice
     * @throws UnsupportedOperationException for a vLLM field this build does not run, named with the reason
     */
    public static KvTransferConfig parse(String json, String refusedBy, Map<String, String> env) {
        Objects.requireNonNull(env, "env");
        String environmentToken = environmentToken(env);
        JsonView view = object(json, refusedBy);
        for (String key : view.keys()) {
            if (!FIELDS.contains(key)) {
                throw new IllegalArgumentException(refusedBy + " has no field " + key + " (it reads "
                        + String.join(", ", FIELDS) + ")");
            }
        }
        if (view.optBool("enable_permute_local_kv", false)) {
            throw new UnsupportedOperationException(refusedBy + " field enable_permute_local_kv permutes the KV"
                    + " layout between HND and NHD, which this build does not run");
        }
        if (view.has("kv_connector_module_path")) {
            throw new UnsupportedOperationException(refusedBy + " field kv_connector_module_path loads a Python"
                    + " connector module, which this build does not run");
        }
        String connector = text(view, "kv_connector", refusedBy);
        if (connector == null) {
            throw new IllegalArgumentException(refusedBy + " needs kv_connector");
        }
        String roleName = text(view, "kv_role", refusedBy);
        if (roleName == null) {
            throw new IllegalArgumentException(refusedBy + " needs kv_role when kv_connector is set (supported roles"
                    + " are kv_producer, kv_consumer and kv_both)");
        }
        KvRole role = KvRole.parse(roleName);
        String engineId = text(view, "engine_id", refusedBy);
        if (engineId == null) {
            engineId = UUID.randomUUID().toString();
        }
        String host = text(view, "kv_ip", refusedBy);
        if (host == null) {
            host = DEFAULT_HOST;
        }
        int port = view.optI32("kv_port", DEFAULT_PORT);
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException(refusedBy + " field kv_port must be in [0, 65535], got " + port);
        }
        int rank = view.optI32("kv_rank", 0);
        int parallel = view.optI32("kv_parallel_size", 1);
        if (rank < 0) {
            throw new IllegalArgumentException(refusedBy + " field kv_rank must be 0 or more, got " + rank);
        }
        if (parallel < 1) {
            throw new IllegalArgumentException(refusedBy + " field kv_parallel_size must be 1 or more, got "
                    + parallel);
        }
        if (rank >= parallel) {
            throw new IllegalArgumentException(refusedBy + " field kv_rank " + rank + " must be below"
                    + " kv_parallel_size " + parallel);
        }
        long buffer = view.optI64("kv_buffer_size", DEFAULT_BUFFER_BYTES);
        if (buffer < 0) {
            throw new IllegalArgumentException(refusedBy + " field kv_buffer_size must be 0 or more, got " + buffer);
        }
        String policy = view.optStr("kv_load_failure_policy", "fail");
        if (!policy.equals("fail") && !policy.equals("recompute")) {
            throw new IllegalArgumentException(refusedBy + " field kv_load_failure_policy must be fail or recompute,"
                    + " got " + policy);
        }
        Map<String, Object> extra = new LinkedHashMap<>();
        if (view.has("kv_connector_extra_config")) {
            Object raw = view.raw("kv_connector_extra_config");
            if (!(raw instanceof Map)) {
                throw new IllegalArgumentException(refusedBy + " field kv_connector_extra_config must be an object");
            }
            for (Map.Entry<?, ?> e : ((Map<?, ?>) raw).entrySet()) {
                extra.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        if (environmentToken != null && extra.containsKey("auth_token")) {
            throw new IllegalArgumentException("the KV transfer token is given by both the environment ("
                    + (env.containsKey(ENV_TOKEN) ? ENV_TOKEN : ENV_TOKEN_FILE)
                    + ") and kv_connector_extra_config.auth_token; keep one source");
        }
        return new KvTransferConfig(connector, engineId, role, host, port, rank, parallel, buffer, policy,
                Collections.unmodifiableMap(extra), environmentToken);
    }

    /**
     * The shared token of {@code env}: {@value #ENV_TOKEN} first, else the file {@value #ENV_TOKEN_FILE} names. A
     * variable that is set but blank is refused, as is an unreadable or empty file; no message repeats the token.
     */
    private static String environmentToken(Map<String, String> env) {
        if (env.containsKey(ENV_TOKEN)) {
            String value = env.get(ENV_TOKEN);
            if (value == null || value.trim().isEmpty()) {
                throw new IllegalArgumentException(ENV_TOKEN + " is blank; it must hold the KV transfer token");
            }
            return value.trim();
        }
        if (env.containsKey(ENV_TOKEN_FILE)) {
            String file = env.get(ENV_TOKEN_FILE);
            if (file == null || file.trim().isEmpty()) {
                throw new IllegalArgumentException(ENV_TOKEN_FILE + " is blank; it must name a file holding the"
                        + " KV transfer token");
            }
            String path = file.trim();
            String text;
            try {
                text = Files.readString(Path.of(path), StandardCharsets.UTF_8).trim();
            } catch (IOException | RuntimeException e) {
                throw new IllegalArgumentException(ENV_TOKEN_FILE + " names a file that cannot be read: " + path, e);
            }
            if (text.isEmpty()) {
                throw new IllegalArgumentException(ENV_TOKEN_FILE + " names an empty file: " + path);
            }
            return text;
        }
        return null;
    }

    /** The JSON object {@code json} holds, refusing anything else and every unknown field. */
    private static JsonView object(String json, String refusedBy) {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(refusedBy + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(refusedBy + " must be a JSON object, got " + json);
        }
        return JsonView.of(parsed, refusedBy);
    }

    /** A trimmed string field, or null when absent or blank. */
    private static String text(JsonView view, String key, String refusedBy) {
        Object raw = view.raw(key);
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof String)) {
            throw new IllegalArgumentException(refusedBy + " field " + key + " must be a string");
        }
        String value = ((String) raw).trim();
        return value.isEmpty() ? null : value;
    }

    /** The connector name as configured ({@code TcpKvConnector}, {@code FileKvConnector}, or a refused name). */
    public String kvConnector() {
        return kvConnector;
    }

    /** The instance id consumers name as {@code remote_engine_id}; a fresh UUID unless configured. */
    public String engineId() {
        return engineId;
    }

    /** This instance's side of the transfer. */
    public KvRole role() {
        return role;
    }

    /** The host a producer binds its listener on and advertises. */
    public String kvIp() {
        return kvIp;
    }

    /** The connector port (0 picks an ephemeral one). */
    public int kvPort() {
        return kvPort;
    }

    /** This instance's rank in a parallel KV transfer group. */
    public int kvRank() {
        return kvRank;
    }

    /** The transfer group's size. */
    public int kvParallelSize() {
        return kvParallelSize;
    }

    /** The producer's in-memory transfer store size in bytes. */
    public long kvBufferSize() {
        return kvBufferSize;
    }

    /** What a consumer does when a pull fails: {@code fail} or {@code recompute}. */
    public String loadFailurePolicy() {
        return loadFailurePolicy;
    }

    /** The connector's own settings, unmodifiable. */
    public Map<String, Object> extraConfig() {
        return extraConfig;
    }

    /**
     * The shared token an environment source ({@value #ENV_TOKEN} or {@value #ENV_TOKEN_FILE}) supplies, or null
     * when the JSON {@code kv_connector_extra_config.auth_token} or nothing supplies it. The socket connector
     * reads whichever of the two is set.
     */
    public String authToken() {
        return authToken;
    }

    /** Whether this instance exports prefill KV. */
    public boolean produces() {
        return role.produces();
    }

    /** Whether this instance imports prefill KV. */
    public boolean consumes() {
        return role.consumes();
    }

    /** The extra-config value of {@code key}, or {@code fallback} when absent. */
    public Object extra(String key, Object fallback) {
        return extraConfig.containsKey(key) ? extraConfig.get(key) : fallback;
    }

    /** The extra-config string of {@code key}, or {@code fallback} when absent or blank. */
    public String extraText(String key, String fallback) {
        Object value = extraConfig.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof String)) {
            throw new IllegalArgumentException("kv_connector_extra_config." + key + " must be a string, got " + value);
        }
        String text = ((String) value).trim();
        return text.isEmpty() ? fallback : text;
    }

    /** The extra-config whole number of {@code key}, or {@code fallback} when absent. */
    public long extraLong(String key, long fallback) {
        Object value = extraConfig.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof Number) || value instanceof Boolean) {
            throw new IllegalArgumentException("kv_connector_extra_config." + key + " must be a number, got " + value);
        }
        return ((Number) value).longValue();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvTransferConfig)) {
            return false;
        }
        KvTransferConfig other = (KvTransferConfig) o;
        return kvPort == other.kvPort && kvRank == other.kvRank && kvParallelSize == other.kvParallelSize
                && kvBufferSize == other.kvBufferSize && kvConnector.equals(other.kvConnector)
                && engineId.equals(other.engineId) && role == other.role && kvIp.equals(other.kvIp)
                && loadFailurePolicy.equals(other.loadFailurePolicy) && extraConfig.equals(other.extraConfig)
                && Objects.equals(authToken, other.authToken);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kvConnector, engineId, role, kvIp, kvPort, kvRank, kvParallelSize, kvBufferSize,
                loadFailurePolicy, extraConfig, authToken);
    }

    @Override
    public String toString() {
        return "KvTransferConfig{" + kvConnector + ", " + role.wireName() + ", engine " + engineId + ", " + kvIp
                + ":" + kvPort + "}";
    }
}
