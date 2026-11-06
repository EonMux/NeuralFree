package com.eonmux.neuralfree;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.remote.ObjectStores;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * The one definition of which {@code neuralfree.*} and {@code quarkus.*} settings carry a secret: the ones whose
 * name contains {@code key}, {@code secret}, {@code password} or {@code token}, plus the settings whose value
 * embeds one under a neutral name. Everything that prints, logs or forwards settings consults
 * {@link #isSecret(String)} instead of repeating the rule:
 *
 * <ul>
 *   <li>{@code neuralfree serve} forwards the secret settings to the child JVM through a temporary properties
 *       file, never as {@code -D} arguments, so the child's command line holds none of them. The file is created
 *       readable by its owner only, named by {@value #FILE_ENV}, and read and deleted by the server at startup
 *       ({@link #loadFromEnvironment()}).</li>
 *   <li>{@code /server_info}'s {@code neuralfree_env} leaves secret settings out.</li>
 *   <li>{@code collect-env} reports only whether the secret environment variables are set.</li>
 * </ul>
 *
 * <p>The named settings are {@code neuralfree.rpc}, whose entries may carry an inline {@code @key};
 * {@code neuralfree.watermark.config}, whose JSON may carry the watermark {@code key};
 * {@code neuralfree.server.kv.transfer.config}, whose JSON may carry the KV transfer {@code auth_token}; the
 * inline MCP server configuration of {@code neuralfree.server.mcp.json}, whose entries may carry credentials in
 * their {@code env}; and {@code neuralfree.remote.azure.sas}, a shared access signature. The S3 secret and the GCS
 * bearer token are already named as such ({@code secret}, {@code token}). Immutable and stateless.</p>
 */
public final class SecretProperties {

    /** The environment variable naming the temporary properties file of the secret settings. */
    public static final String FILE_ENV = "NEURALFREE_SECRET_PROPERTIES";

    /** The temporary file's prefix. */
    private static final String FILE_PREFIX = "neuralfree-secrets-";

    /** The temporary file's suffix. */
    private static final String FILE_SUFFIX = ".properties";

    /** The owner-only permissions of the temporary file. */
    private static final String OWNER_ONLY = "rw-------";

    /** The API key property of the server (llama.cpp's {@code --api-key}, vLLM's flag of the same name). */
    public static final String API_KEY_PROPERTY = "neuralfree.server.api.key";

    /** The inline MCP server configuration property, whose server entries may carry credentials. */
    public static final String MCP_JSON_PROPERTY = "neuralfree.server.mcp.json";

    /** The server launch flag of {@link #API_KEY_PROPERTY}; repeatable, keys join with commas. */
    public static final String API_KEY_FLAG = "--api-key";

    /** The server launch flag of {@link EngineConfig#PROP_RPC}. */
    public static final String RPC_FLAG = "--rpc";

    /** The server launch flag of {@link WatermarkConfig#PROP_CONFIG}. */
    public static final String WATERMARK_CONFIG_FLAG = "--watermark-config";

    /** The server launch flag of {@link KvTransferConfig#SERVER_PROPERTY}. */
    public static final String KV_TRANSFER_CONFIG_FLAG = KvTransferConfig.FLAG;

    /** The server launch flag of {@link #MCP_JSON_PROPERTY}. */
    public static final String MCP_JSON_FLAG = "--mcp-servers-json";

    /** The name fragments that mark a setting as secret, matched case-insensitively. */
    private static final Set<String> FRAGMENTS = Collections.unmodifiableSet(new HashSet<>(
            Arrays.asList("key", "secret", "password", "token")));

    /** The secret settings whose names carry no fragment of {@link #FRAGMENTS}. */
    private static final Set<String> NAMED = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            EngineConfig.PROP_RPC, WatermarkConfig.PROP_CONFIG, KvTransferConfig.SERVER_PROPERTY,
            MCP_JSON_PROPERTY, ObjectStores.PROP_AZURE_SAS)));

    private static final Logger LOGGER = LogManager.getLogger(SecretProperties.class);

    private SecretProperties() {
    }

    /**
     * Whether {@code name} is a setting whose value carries a secret and must not be printed, logged or passed as
     * an argument.
     */
    public static boolean isSecret(String name) {
        Objects.requireNonNull(name, "setting name");
        if (NAMED.contains(name)) {
            return true;
        }
        String lower = name.toLowerCase(Locale.ROOT);
        for (String fragment : FRAGMENTS) {
            if (lower.contains(fragment)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The settings property a server launch flag carries, or null for any other argument. The value must leave
     * the child's command line; repeatable values such as {@code --api-key} are joined by their caller.
     */
    public static String propertyOfFlag(String flag) {
        if (API_KEY_FLAG.equals(flag)) {
            return API_KEY_PROPERTY;
        }
        if (RPC_FLAG.equals(flag)) {
            return EngineConfig.PROP_RPC;
        }
        if (WATERMARK_CONFIG_FLAG.equals(flag)) {
            return WatermarkConfig.PROP_CONFIG;
        }
        if (KV_TRANSFER_CONFIG_FLAG.equals(flag)) {
            return KvTransferConfig.SERVER_PROPERTY;
        }
        if (MCP_JSON_FLAG.equals(flag)) {
            return MCP_JSON_PROPERTY;
        }
        return null;
    }

    /** A copy of the secret settings of {@code properties}, in their iteration order. */
    public static Properties secrets(Properties properties) {
        Objects.requireNonNull(properties, "properties");
        Properties out = new Properties();
        for (String name : properties.stringPropertyNames()) {
            if (isSecret(name)) {
                out.setProperty(name, properties.getProperty(name));
            }
        }
        return out;
    }

    /**
     * Writes {@code secrets} to a new temporary properties file readable by its owner only ({@code rw-------}).
     * The caller names the file to the child in {@value #FILE_ENV}; the child deletes it after reading it, and the
     * parent deletes it at exit if the child never got that far.
     *
     * @throws IOException when the file cannot be created or written
     */
    public static Path write(Properties secrets) throws IOException {
        Objects.requireNonNull(secrets, "secrets");
        Path file;
        try {
            FileAttribute<?> ownerOnly = PosixFilePermissions.asFileAttribute(
                    PosixFilePermissions.fromString(OWNER_ONLY));
            file = Files.createTempFile(FILE_PREFIX, FILE_SUFFIX, ownerOnly);
        } catch (UnsupportedOperationException e) {
            // a file system without POSIX permissions still gets the file; nothing better can be requested
            file = Files.createTempFile(FILE_PREFIX, FILE_SUFFIX);
        }
        try (OutputStream out = Files.newOutputStream(file)) {
            secrets.store(out, null);
        } catch (IOException e) {
            try {
                Files.deleteIfExists(file);
            } catch (IOException ignored) {
                e.addSuppressed(ignored);
            }
            throw e;
        }
        file.toFile().deleteOnExit();
        return file;
    }

    /** Loads the settings file {@value #FILE_ENV} names in this process's environment, if any. */
    public static void loadFromEnvironment() {
        loadFromEnvironment(System.getenv());
    }

    /**
     * Loads the settings file {@code env} names in {@value #FILE_ENV}, if any, into this process's system
     * properties and deletes it. The file is the one {@link #write(Properties)} produced; a name outside the
     * forwarded {@code neuralfree.} and {@code quarkus.} namespaces is refused, so the file cannot set unrelated
     * process settings.
     *
     * @throws IllegalStateException when the named file is missing or unreadable
     * @throws IllegalArgumentException when it names a setting outside the forwarded namespaces
     */
    public static void loadFromEnvironment(Map<String, String> env) {
        Objects.requireNonNull(env, "environment");
        String path = env.get(FILE_ENV);
        if (path == null || path.trim().isEmpty()) {
            return;
        }
        load(Paths.get(path.trim()));
    }

    /**
     * Loads {@code file} into this process's system properties and deletes it, whether or not it parsed.
     *
     * @throws IllegalStateException when the file is not a regular file or cannot be read
     * @throws IllegalArgumentException when it names a setting outside the forwarded namespaces
     */
    private static void load(Path file) {
        Objects.requireNonNull(file, "file");
        if (!Files.isRegularFile(file)) {
            throw new IllegalStateException(FILE_ENV + " names no regular file: " + file);
        }
        Properties settings = new Properties();
        try (InputStream in = Files.newInputStream(file)) {
            settings.load(in);
        } catch (IOException e) {
            throw new IllegalStateException(FILE_ENV + " names a file that cannot be read: " + file, e);
        } finally {
            delete(file);
        }
        for (String name : settings.stringPropertyNames()) {
            if (!name.startsWith("neuralfree.") && !name.startsWith("quarkus.")) {
                throw new IllegalArgumentException(FILE_ENV + " names a file with a setting outside the"
                        + " neuralfree. and quarkus. namespaces: " + name);
            }
        }
        for (String name : settings.stringPropertyNames()) {
            System.setProperty(name, settings.getProperty(name));
        }
    }

    /** Deletes a settings file the child has read; a failure is logged, never fatal. */
    private static void delete(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException e) {
            LOGGER.warn("could not delete the secret properties file {}; delete it by hand", file);
        }
    }
}
