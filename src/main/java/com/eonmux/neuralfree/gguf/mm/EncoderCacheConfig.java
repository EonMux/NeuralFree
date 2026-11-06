package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Where a server shares encoder outputs with the other servers of its deployment (vLLM's {@code
 * --ec-transfer-config}, {@code vllm/config/ec_transfer.py}): the shared directory the entries live in and the
 * {@link EncoderCacheConnector.Role role} this server takes, producer, consumer or both. An encoder instance of an
 * encoder-prefill-decode deployment is a producer, the prefill and decode instances are consumers, and a single
 * server that shares its encodings with a peer is both.
 *
 * <p>{@link #parseTransferConfig} reads vLLM's JSON form: the {@code ec_connector} ({@value #CONNECTOR_FILE}, or
 * vLLM's {@value #CONNECTOR_EXAMPLE} name for a shared filesystem), the {@code ec_role} ({@code ec_producer},
 * {@code ec_consumer} or {@code ec_both}, required with a connector as vLLM requires it) and
 * {@code ec_connector_extra_config.shared_storage_path}. A field naming a transport this build has no connector for
 * (an RDMA engine, a connector class path) is refused by name, as is an unknown connector. The equivalent
 * properties are {@value #PROP_DIR} and {@value #PROP_ROLE}. Immutable; {@link #open} binds the connector.
 */
public final class EncoderCacheConfig {

    /** The shared directory of the encoder cache entries. */
    public static final String PROP_DIR = "neuralfree.mm.encoder.cache.dir";

    /** The role of the server in the transfer ({@link EncoderCacheConnector.Role#parse}). */
    public static final String PROP_ROLE = "neuralfree.mm.encoder.cache.role";

    /** The connector name of the shared-directory store: vLLM's example connector. */
    public static final String CONNECTOR_EXAMPLE = "ECExampleConnector";

    /** The connector name of the shared-directory store in this build. */
    public static final String CONNECTOR_FILE = "file";

    /** The vLLM connector names this build does not have, each with the reason it is refused. */
    private static final Set<String> UNSUPPORTED_CONNECTORS = new LinkedHashSet<>(Arrays.asList(
            "ECMooncakeConnector", "ECCPUConnector"));

    private static final Set<String> FIELDS = new LinkedHashSet<>(Arrays.asList("ec_connector", "ec_role",
            "ec_connector_extra_config"));

    private static final Set<String> EXTRA_FIELDS = new LinkedHashSet<>(Arrays.asList("shared_storage_path"));

    private final Path directory;
    private final EncoderCacheConnector.Role role;

    private EncoderCacheConfig(Path directory, EncoderCacheConnector.Role role) {
        this.directory = Objects.requireNonNull(directory, "directory");
        this.role = Objects.requireNonNull(role, "role");
    }

    /**
     * A config over {@code directory} with {@code role}.
     *
     * @throws IllegalArgumentException when the directory is blank
     */
    public static EncoderCacheConfig of(Path directory, EncoderCacheConnector.Role role) {
        Objects.requireNonNull(directory, "directory");
        if (directory.toString().trim().isEmpty()) {
            throw new IllegalArgumentException("the encoder cache directory is blank");
        }
        return new EncoderCacheConfig(directory, role);
    }

    /**
     * The config of {@value #PROP_DIR} and {@value #PROP_ROLE}, or null when no directory is set.
     *
     * @throws IllegalArgumentException when the role is missing or unknown
     */
    public static EncoderCacheConfig fromProperties(Properties properties) {
        Objects.requireNonNull(properties, "properties");
        String dir = trimmed(properties.getProperty(PROP_DIR));
        if (dir == null) {
            return null;
        }
        String role = trimmed(properties.getProperty(PROP_ROLE));
        if (role == null) {
            throw new IllegalArgumentException(PROP_ROLE + " is required beside " + PROP_DIR
                    + "; use ec_producer, ec_consumer or ec_both");
        }
        try {
            return of(Path.of(dir), EncoderCacheConnector.Role.parse(role));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_ROLE + "=" + role + ": " + e.getMessage(), e);
        }
    }

    /**
     * The config of vLLM's {@code --ec-transfer-config} JSON.
     *
     * @param json      the JSON object
     * @param refusedBy the flag or property named in every refusal
     * @throws IllegalArgumentException when the JSON is malformed or names a connector, role or field this build
     *                                  does not have
     */
    public static EncoderCacheConfig parseTransferConfig(String json, String refusedBy) {
        Objects.requireNonNull(json, "json");
        Objects.requireNonNull(refusedBy, "refusedBy");
        Map<?, ?> config = object(json, refusedBy);
        for (Object key : config.keySet()) {
            if (!FIELDS.contains(String.valueOf(key))) {
                throw new IllegalArgumentException(refusedBy + " has no field " + key + " (it reads "
                        + String.join(", ", FIELDS) + ")");
            }
        }
        String connector = text(config, "ec_connector", refusedBy);
        if (connector == null) {
            throw new IllegalArgumentException(refusedBy + " needs ec_connector; this build has "
                    + CONNECTOR_FILE + " (a shared directory, vLLM's " + CONNECTOR_EXAMPLE + ")");
        }
        if (UNSUPPORTED_CONNECTORS.contains(connector)) {
            throw new IllegalArgumentException(refusedBy + " names the connector " + connector + ", which this"
                    + " build does not have; use " + CONNECTOR_FILE + " (a shared directory, vLLM's "
                    + CONNECTOR_EXAMPLE + ")");
        }
        if (!CONNECTOR_FILE.equals(connector) && !CONNECTOR_EXAMPLE.equals(connector)) {
            throw new IllegalArgumentException(refusedBy + " names the connector '" + connector + "', which this"
                    + " build does not have; use " + CONNECTOR_FILE + " (a shared directory, vLLM's "
                    + CONNECTOR_EXAMPLE + ")");
        }
        String role = text(config, "ec_role", refusedBy);
        if (role == null) {
            throw new IllegalArgumentException(refusedBy + " needs ec_role beside ec_connector; use ec_producer,"
                    + " ec_consumer or ec_both");
        }
        EncoderCacheConnector.Role parsed;
        try {
            parsed = EncoderCacheConnector.Role.parse(role);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(refusedBy + " ec_role=" + role + ": " + e.getMessage(), e);
        }
        String dir = sharedStoragePath(config, refusedBy);
        if (dir == null) {
            throw new IllegalArgumentException(refusedBy + " needs ec_connector_extra_config.shared_storage_path,"
                    + " the directory the encoder cache entries live in");
        }
        return of(Path.of(dir), parsed);
    }

    /** The JSON object {@code json} holds, refused unless every field is one vLLM's config declares. */
    private static Map<?, ?> object(String json, String refusedBy) {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(refusedBy + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(refusedBy + " must be a JSON object, got " + json);
        }
        return (Map<?, ?>) parsed;
    }

    /** The {@code ec_connector} name, trimmed; null when absent or JSON null. */
    private static String text(Map<?, ?> config, String field, String refusedBy) {
        Object value = config.get(field);
        if (value == null) {
            return null;
        }
        if (!(value instanceof String) || ((String) value).isBlank()) {
            throw new IllegalArgumentException(refusedBy + " " + field + " must be a non-empty string, got " + value);
        }
        return ((String) value).trim();
    }

    /** The {@code ec_connector_extra_config.shared_storage_path}, trimmed; null when absent. */
    private static String sharedStoragePath(Map<?, ?> config, String refusedBy) {
        Object extra = config.get("ec_connector_extra_config");
        if (extra == null) {
            return null;
        }
        if (!(extra instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(refusedBy + " ec_connector_extra_config must be a JSON object, got "
                    + extra);
        }
        Map<?, ?> fields = (Map<?, ?>) extra;
        for (Object key : fields.keySet()) {
            if (!EXTRA_FIELDS.contains(String.valueOf(key))) {
                throw new IllegalArgumentException(refusedBy + " ec_connector_extra_config has no field " + key
                        + " (it reads " + String.join(", ", EXTRA_FIELDS) + ")");
            }
        }
        Object path = fields.get("shared_storage_path");
        if (path == null) {
            return null;
        }
        if (!(path instanceof String) || ((String) path).isBlank()) {
            throw new IllegalArgumentException(refusedBy + " shared_storage_path must be a non-empty string, got "
                    + path);
        }
        return ((String) path).trim();
    }

    private static String trimmed(String value) {
        if (value == null) {
            return null;
        }
        String t = value.trim();
        return t.isEmpty() ? null : t;
    }

    /** The shared directory the entries live in. */
    public Path directory() {
        return directory;
    }

    /** The side of the transfer this server is on. */
    public EncoderCacheConnector.Role role() {
        return role;
    }

    /**
     * Binds the connector over {@link #directory()}, creating the directory when it does not exist.
     *
     * @throws IOException when the directory cannot be created or read
     */
    public EncoderCacheConnector open() throws IOException {
        return new DirectoryEncoderCacheConnector(directory, role);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof EncoderCacheConfig)) {
            return false;
        }
        EncoderCacheConfig other = (EncoderCacheConfig) o;
        return directory.equals(other.directory) && role == other.role;
    }

    @Override
    public int hashCode() {
        return Objects.hash(directory, role);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "EncoderCacheConfig{%s, role=%s}", directory, role.label());
    }
}
