package com.eonmux.neuralfree.gguf.mm;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EncoderCacheConfig}: vLLM's {@code --ec-transfer-config} JSON maps to the shared directory and role of the
 * deployment, a field, connector or role this build does not have is refused by name, and the equivalent
 * {@value EncoderCacheConfig#PROP_DIR} / {@value EncoderCacheConfig#PROP_ROLE} properties name a config too.
 */
public class EncoderCacheConfigTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static String json(Path dir, String role) {
        return "{\"ec_connector\": \"ECExampleConnector\", \"ec_role\": \"" + role + "\","
                + " \"ec_connector_extra_config\": {\"shared_storage_path\": \"" + dir + "\"}}";
    }

    private static EncoderCacheConfig parse(String json) {
        return EncoderCacheConfig.parseTransferConfig(json, "--ec-transfer-config");
    }

    @Test
    public void vllmsTransferConfigNamesTheSharedDirectoryAndRole() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        EncoderCacheConfig producer = parse(json(dir, "ec_producer"));
        assertEquals(dir, producer.directory());
        assertEquals(EncoderCacheConnector.Role.PRODUCER, producer.role());
        assertTrue(producer.open().isProducer());
        assertEquals(EncoderCacheConnector.Role.CONSUMER, parse(json(dir, "ec_consumer")).role());
        assertEquals(EncoderCacheConnector.Role.BOTH, parse(json(dir, "ec_both")).role());
        assertEquals(dir, parse("{\"ec_connector\": \"file\", \"ec_role\": \"ec_producer\","
                + " \"ec_connector_extra_config\": {\"shared_storage_path\": \"" + dir + "\"}}").directory());
    }

    @Test
    public void anUnsupportedConnectorOrFieldIsRefusedByName() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> parse(json(dir, "ec_producer").replace("ECExampleConnector", "ECMooncakeConnector")));
        assertTrue(e.getMessage(), e.getMessage().contains("ECMooncakeConnector"));
        assertTrue(e.getMessage(), e.getMessage().contains("--ec-transfer-config"));
        e = assertThrows(IllegalArgumentException.class, () -> parse("{\"ec_connector\": \"Nope\","
                + " \"ec_role\": \"ec_producer\", \"ec_connector_extra_config\": {\"shared_storage_path\": \""
                + dir + "\"}}"));
        assertTrue(e.getMessage(), e.getMessage().contains("Nope"));
        e = assertThrows(IllegalArgumentException.class, () -> parse("{\"ec_connector\": \"ECExampleConnector\","
                + " \"ec_role\": \"ec_producer\", \"ec_ip\": \"127.0.0.1\"}"));
        assertTrue(e.getMessage(), e.getMessage().contains("ec_ip"));
    }

    @Test
    public void aMissingRoleStoragePathOrConnectorIsRefused() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> parse("{\"ec_connector\": \"ECExampleConnector\","
                        + " \"ec_connector_extra_config\": {\"shared_storage_path\": \"" + dir + "\"}}"));
        assertTrue(e.getMessage(), e.getMessage().contains("ec_role"));
        e = assertThrows(IllegalArgumentException.class,
                () -> parse("{\"ec_connector\": \"ECExampleConnector\", \"ec_role\": \"ec_producer\"}"));
        assertTrue(e.getMessage(), e.getMessage().contains("shared_storage_path"));
        e = assertThrows(IllegalArgumentException.class,
                () -> EncoderCacheConfig.parseTransferConfig("{\"ec_connector\": \"ECExampleConnector\","
                        + " \"ec_role\": \"ec_both\", \"ec_connector_extra_config\": {\"shared_storage_path\":"
                        + " \"" + dir + "\", \"remote\": true}}", "-x"));
        assertTrue(e.getMessage(), e.getMessage().contains("-x"));
        assertTrue(e.getMessage(), e.getMessage().contains("remote"));
        assertThrows(IllegalArgumentException.class, () -> parse("not json"));
        assertThrows(IllegalArgumentException.class, () -> parse(json(dir, "ec_writer")));
    }

    @Test
    public void thePropertiesNameTheSameConfig() {
        Properties properties = new Properties();
        assertNull(EncoderCacheConfig.fromProperties(properties));
        properties.setProperty(EncoderCacheConfig.PROP_DIR, tmp.getRoot().toString());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EncoderCacheConfig.fromProperties(properties));
        assertTrue(e.getMessage(), e.getMessage().contains(EncoderCacheConfig.PROP_ROLE));
        properties.setProperty(EncoderCacheConfig.PROP_ROLE, "ec_both");
        EncoderCacheConfig config = EncoderCacheConfig.fromProperties(properties);
        assertEquals(tmp.getRoot().toPath(), config.directory());
        assertEquals(EncoderCacheConnector.Role.BOTH, config.role());
        assertEquals(config, EncoderCacheConfig.of(tmp.getRoot().toPath(), EncoderCacheConnector.Role.BOTH));
        assertThrows(IllegalArgumentException.class,
                () -> EncoderCacheConfig.of(Path.of(" "), EncoderCacheConnector.Role.BOTH));
        assertThrows(IllegalArgumentException.class,
                () -> EncoderCacheConnector.Role.parse("writer"));
    }
}
