package com.eonmux.neuralfree;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.remote.ObjectStores;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The single definition of secret-bearing settings: the name-fragment rule, the settings whose neutral names hide
 * an embedded secret, the launch flags that carry them, the owner-only properties file the serve relaunch uses, and
 * its load-and-delete in the child.
 */
public class SecretPropertiesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void nameFragmentsMarkTheSettingSecret() {
        for (String name : new String[]{"neuralfree.server.api.key", "neuralfree.server.api.key.file",
                "neuralfree.remote.gcs.token", "neuralfree.remote.s3.secret.access", "quarkus.datasource.password",
                "neuralfree.hf.tokenizer"}) {
            assertTrue(name, SecretProperties.isSecret(name));
        }
        assertTrue("the fragments match case-insensitively", SecretProperties.isSecret("neuralfree.API.Key"));
    }

    @Test
    public void namedSettingsMarkTheSettingSecret() {
        assertTrue(SecretProperties.isSecret(EngineConfig.PROP_RPC));
        assertTrue(SecretProperties.isSecret(WatermarkConfig.PROP_CONFIG));
        assertTrue(SecretProperties.isSecret(SecretProperties.MCP_JSON_PROPERTY));
        assertTrue(SecretProperties.isSecret(ObjectStores.PROP_AZURE_SAS));
        assertTrue("the KV transfer configuration embeds an auth_token",
                SecretProperties.isSecret(KvTransferConfig.SERVER_PROPERTY));
    }

    @Test
    public void ordinarySettingsAreNotSecret() {
        for (String name : new String[]{"neuralfree.server.model", "neuralfree.server.replicas",
                "quarkus.http.port", "neuralfree.remote.azure.account", "neuralfree.remote.s3.endpoint",
                "neuralfree.hf.cache.dir"}) {
            assertFalse(name, SecretProperties.isSecret(name));
        }
    }

    @Test
    public void theLaunchFlagsNameTheirProperties() {
        assertEquals(SecretProperties.API_KEY_PROPERTY, SecretProperties.propertyOfFlag("--api-key"));
        assertEquals(EngineConfig.PROP_RPC, SecretProperties.propertyOfFlag("--rpc"));
        assertEquals(WatermarkConfig.PROP_CONFIG, SecretProperties.propertyOfFlag("--watermark-config"));
        assertEquals(SecretProperties.MCP_JSON_PROPERTY, SecretProperties.propertyOfFlag("--mcp-servers-json"));
        assertEquals(KvTransferConfig.SERVER_PROPERTY, SecretProperties.propertyOfFlag("--kv-transfer-config"));
        assertNull(SecretProperties.propertyOfFlag("--port"));
        assertNull(SecretProperties.propertyOfFlag("-m"));
    }

    @Test
    public void theCopiesSplitSecretsFromPublicSettings() {
        Properties all = new Properties();
        all.setProperty("neuralfree.server.model", "/models/m.gguf");
        all.setProperty("neuralfree.rpc", "box:50052@s3cret");
        all.setProperty("neuralfree.server.api.key", "sk-live");

        Properties secrets = SecretProperties.secrets(all);
        assertEquals(2, secrets.size());
        assertEquals("box:50052@s3cret", secrets.getProperty("neuralfree.rpc"));
        assertEquals("sk-live", secrets.getProperty("neuralfree.server.api.key"));
        assertNull("a public setting is not forwarded as a secret", secrets.getProperty("neuralfree.server.model"));
    }

    @Test
    public void theFileIsOwnerOnlyAndHoldsTheSecrets() throws IOException {
        Properties secrets = new Properties();
        secrets.setProperty("neuralfree.server.api.key", "sk-live");
        secrets.setProperty("neuralfree.rpc", "box:50052@s3cret");
        Path file = SecretProperties.write(secrets);
        try {
            assertTrue(Files.isRegularFile(file));
            Set<PosixFilePermission> permissions = Files.getPosixFilePermissions(file);
            assertTrue("the owner reads and writes it: " + permissions,
                    permissions.contains(PosixFilePermission.OWNER_READ)
                            && permissions.contains(PosixFilePermission.OWNER_WRITE));
            assertFalse("the group and others read no secret: " + permissions,
                    permissions.contains(PosixFilePermission.GROUP_READ)
                            || permissions.contains(PosixFilePermission.OTHERS_READ)
                            || permissions.contains(PosixFilePermission.GROUP_WRITE)
                            || permissions.contains(PosixFilePermission.OTHERS_WRITE));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theEnvironmentFileIsLoadedAndDeleted() throws IOException {
        String apiKey = "neuralfree.test.secret.api.key";
        String rpc = "neuralfree.test.rpc";
        String savedKey = System.getProperty(apiKey);
        String savedRpc = System.getProperty(rpc);
        Path file = SecretProperties.write(settings(apiKey, "sk-child", rpc, "box:50052@s3cret"));
        try {
            Map<String, String> env = new LinkedHashMap<>();
            env.put(SecretProperties.FILE_ENV, file.toString());
            SecretProperties.loadFromEnvironment(env);
            assertEquals("sk-child", System.getProperty(apiKey));
            assertEquals("box:50052@s3cret", System.getProperty(rpc));
            assertFalse("the child deletes the file it read", Files.exists(file));
        } finally {
            restore(apiKey, savedKey);
            restore(rpc, savedRpc);
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void noEnvironmentVariableLoadsNothing() {
        SecretProperties.loadFromEnvironment(Collections.emptyMap());
    }

    @Test
    public void aMissingFileIsRefused() {
        Map<String, String> env = new LinkedHashMap<>();
        env.put(SecretProperties.FILE_ENV, tmp.getRoot().toPath().resolve("absent.properties").toString());
        try {
            SecretProperties.loadFromEnvironment(env);
            fail("loaded a missing file");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(SecretProperties.FILE_ENV));
        }
    }

    @Test
    public void aSettingOutsideTheForwardedNamespacesIsRefusedAndTheFileDeleted() throws IOException {
        String name = "example.not.forwarded";
        Path file = SecretProperties.write(settings(name, "x"));
        Map<String, String> env = new LinkedHashMap<>();
        env.put(SecretProperties.FILE_ENV, file.toString());
        try {
            SecretProperties.loadFromEnvironment(env);
            fail("loaded a setting outside the namespaces");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(name));
            assertNull(System.getProperty(name));
            assertFalse(Files.exists(file));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    private static Properties settings(String... pairs) {
        Properties p = new Properties();
        for (int i = 0; i + 1 < pairs.length; i += 2) {
            p.setProperty(pairs[i], pairs[i + 1]);
        }
        return p;
    }

    private static void restore(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
