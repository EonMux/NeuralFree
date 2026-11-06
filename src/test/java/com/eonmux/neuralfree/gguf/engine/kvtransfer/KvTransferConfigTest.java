package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV transfer configuration and request parameters (vLLM's {@code KVTransferConfig} and
 * {@code kv_transfer_params}): the fields and defaults a JSON object sets, and the refusals that name an unknown,
 * mistyped, out-of-range or unsupported field — including the reference connectors this build does not ship.
 */
public class KvTransferConfigTest {

    private static final String PRODUCER = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_producer\","
            + " \"engine_id\": \"engine-a\", \"kv_ip\": \"10.0.0.1\", \"kv_port\": 15000, \"kv_rank\": 0,"
            + " \"kv_parallel_size\": 2, \"kv_buffer_size\": 1024,"
            + " \"kv_connector_extra_config\": {\"auth_token\": \"shared\"},"
            + " \"kv_load_failure_policy\": \"recompute\"}";

    @Test
    public void aConfigReadsVllmsFields() {
        KvTransferConfig config = KvTransferConfig.parse(PRODUCER, "--kv-transfer-config");
        assertEquals("TcpKvConnector", config.kvConnector());
        assertEquals("engine-a", config.engineId());
        assertEquals(KvRole.PRODUCER, config.role());
        assertEquals("10.0.0.1", config.kvIp());
        assertEquals(15000, config.kvPort());
        assertEquals(2, config.kvParallelSize());
        assertEquals(1024, config.kvBufferSize());
        assertEquals("recompute", config.loadFailurePolicy());
        assertEquals("shared", config.extraText("auth_token", null));
        assertTrue(config.produces());
        assertFalse(config.consumes());
        assertEquals(KvRole.CONSUMER, KvRole.parse("kv_consumer"));
        assertEquals(KvRole.BOTH, KvRole.parse("kv_both"));
    }

    @Test
    public void defaultsFollowVllm() {
        KvTransferConfig config = KvTransferConfig.parse(
                "{\"kv_connector\": \"FileKvConnector\", \"kv_role\": \"kv_consumer\","
                        + " \"kv_connector_extra_config\": {\"storage_path\": \"/tmp/kv\"}}", "--kv-transfer-config");
        assertNotNull(config.engineId());
        assertEquals("127.0.0.1", config.kvIp());
        assertEquals(KvTransferConfig.DEFAULT_PORT, config.kvPort());
        assertEquals(0, config.kvRank());
        assertEquals(1, config.kvParallelSize());
        assertEquals("fail", config.loadFailurePolicy());
    }

    @Test
    public void unknownAndUnsupportedFieldsRefuseByName() {
        assertTrue(refusal("{\"kv_connector\": \"TcpKvConnector\"}").contains("kv_role"));
        assertTrue(refusal("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_wrong\"}")
                .contains("kv_role"));
        assertTrue(refusal("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\", \"extra\": 1}")
                .contains("extra"));
        assertTrue(refused("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                + " \"enable_permute_local_kv\": true}").contains("enable_permute_local_kv"));
        assertTrue(refused("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                + " \"kv_connector_module_path\": \"x.y\"}").contains("kv_connector_module_path"));
        assertTrue(refusal("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                + " \"kv_rank\": 2, \"kv_parallel_size\": 2}").contains("kv_rank"));
        assertTrue(refusal("{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                + " \"kv_load_failure_policy\": \"retry\"}").contains("kv_load_failure_policy"));
        assertTrue(refusal("not json").contains("not valid JSON"));
    }

    @Test
    public void theReferenceConnectorsRefuseByName() {
        String template = "{\"kv_connector\": \"%s\", \"kv_role\": \"kv_both\"}";
        for (String name : new String[] {"NixlConnector", "MooncakeConnector", "LMCacheConnectorV1"}) {
            UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                    () -> KvConnectors.create(KvTransferConfig.parse(String.format(template, name),
                            "--kv-transfer-config")));
            assertTrue(refused.getMessage(), refused.getMessage().contains(name));
        }
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> KvConnectors.create(KvTransferConfig.parse(String.format(template, "MyConnector"),
                        "--kv-transfer-config")));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("TcpKvConnector"));
    }

    @Test
    public void transferParamsReadTheRouterProtocol() {
        KvTransferParams prefill = body(
                "{\"do_remote_decode\": true, \"transfer_id\": \"t-1\"}");
        assertTrue(prefill.doRemoteDecode());
        assertFalse(prefill.doRemotePrefill());
        assertEquals("t-1", prefill.transferId());

        KvTransferParams decode = body(
                "{\"do_remote_prefill\": true, \"transfer_id\": \"t-1\", \"remote_engine_id\": \"engine-a\","
                        + " \"remote_host\": \"10.0.0.1\", \"remote_port\": 15000, \"remote_num_tokens\": 14}");
        assertTrue(decode.doRemotePrefill());
        assertEquals("engine-a", decode.remoteEngineId());
        assertEquals(15000, decode.remotePort());
        assertEquals(14, decode.remoteNumTokens());
        assertEquals(true, decode.toMap().get("do_remote_prefill"));
        assertEquals(15000, decode.toMap().get("remote_port"));
    }

    @Test
    public void contradictingOrUnknownTransferParamsRefuseByName() {
        assertTrue(transferRefusal("{}").contains("do_remote_decode"));
        assertTrue(transferRefusal("{\"do_remote_prefill\": true, \"do_remote_decode\": true}")
                .contains("do_remote_prefill"));
        assertTrue(transferRefusal("{\"do_remote_prefill\": true}").contains("transfer_id"));
        assertTrue(transferRefusal("{\"do_remote_decode\": true, \"remote_block_ids\": [[1]]}")
                .contains("remote_block_ids"));
        assertTrue(transferRefusal("{\"do_remote_decode\": true, \"do_remote_store\": true}")
                .contains("do_remote_store"));
        assertTrue(transferRefusal("{\"do_remote_prefill\": true, \"transfer_id\": \"x\","
                + " \"remote_port\": 70000}").contains("remote_port"));
    }

    private static KvTransferParams body(String json) {
        return KvTransferParams.parse(com.eonmux.neuralfree.utils.json.JsonView.of(
                com.eonmux.neuralfree.utils.json.Json.parse(json,
                        com.eonmux.neuralfree.utils.json.Json.Limits.UNTRUSTED), "body"));
    }

    private static String transferRefusal(String json) {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class, () -> body(json));
        return refused.getMessage();
    }

    private static String refused(String json) {
        UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                () -> KvTransferConfig.parse(json, "--kv-transfer-config"));
        return refused.getMessage();
    }

    private static String refusal(String json) {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> KvTransferConfig.parse(json, "--kv-transfer-config"));
        return refused.getMessage();
    }

    /** The environment a test passes: the named variables with their values. */
    private static Map<String, String> env(String... namesAndValues) {
        Map<String, String> out = new HashMap<>();
        for (int i = 0; i < namesAndValues.length; i += 2) {
            out.put(namesAndValues[i], namesAndValues[i + 1]);
        }
        return out;
    }

    private static final String CONSUMER_WITH_EXTRA = "{\"kv_connector\": \"TcpKvConnector\","
            + " \"kv_role\": \"kv_consumer\", \"kv_connector_extra_config\": {\"x\": 1}}";

    @Test
    public void theTokenComesFromTheEnvironmentVariable() {
        KvTransferConfig config = KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                env(KvTransferConfig.ENV_TOKEN, " shared "));
        assertEquals("shared", config.authToken());
        assertEquals("the JSON's other fields keep their values", 1L, config.extra("x", null));
    }

    @Test
    public void theTokenComesFromTheFileTheEnvironmentNames() throws IOException {
        Path file = Files.createTempFile("kv-transfer-token", ".txt");
        try {
            Files.write(file, " shared \n".getBytes(StandardCharsets.UTF_8));
            KvTransferConfig config = KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                    env(KvTransferConfig.ENV_TOKEN_FILE, " " + file + " "));
            assertEquals("shared", config.authToken());
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theEnvironmentVariableWinsOverTheFile() throws IOException {
        Path file = Files.createTempFile("kv-transfer-token", ".txt");
        try {
            Files.write(file, "from-the-file\n".getBytes(StandardCharsets.UTF_8));
            Map<String, String> both = env(KvTransferConfig.ENV_TOKEN, "from-the-variable",
                    KvTransferConfig.ENV_TOKEN_FILE, file.toString());
            assertEquals("from-the-variable", KvTransferConfig.parse(CONSUMER_WITH_EXTRA,
                    "--kv-transfer-config", both).authToken());
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void withoutAnEnvironmentSourceTheTokenStaysWithTheJson() {
        assertNull(KvTransferConfig.parse(PRODUCER, "--kv-transfer-config").authToken());
        assertNull(KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                Collections.emptyMap()).authToken());
    }

    @Test
    public void anEnvironmentTokenBesideTheJsonTokenIsRefusedByName() throws IOException {
        String variable = assertThrows(IllegalArgumentException.class,
                () -> KvTransferConfig.parse(PRODUCER, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN, "other"))).getMessage();
        assertTrue(variable, variable.contains(KvTransferConfig.ENV_TOKEN) && variable.contains("auth_token"));

        Path file = Files.createTempFile("kv-transfer-token", ".txt");
        try {
            Files.write(file, "other\n".getBytes(StandardCharsets.UTF_8));
            String fileSource = assertThrows(IllegalArgumentException.class,
                    () -> KvTransferConfig.parse(PRODUCER, "--kv-transfer-config",
                            env(KvTransferConfig.ENV_TOKEN_FILE, file.toString()))).getMessage();
            assertTrue(fileSource, fileSource.contains(KvTransferConfig.ENV_TOKEN_FILE)
                    && fileSource.contains("auth_token"));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void aBlankEnvironmentTokenIsRefusedWithoutRepeatingIt() {
        for (String value : new String[] {"", "   "}) {
            String message = assertThrows(IllegalArgumentException.class,
                    () -> KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                            env(KvTransferConfig.ENV_TOKEN, value))).getMessage();
            assertTrue(message, message.contains(KvTransferConfig.ENV_TOKEN));
        }
        assertThrows(IllegalArgumentException.class, () -> KvTransferConfig.parse(CONSUMER_WITH_EXTRA,
                "--kv-transfer-config", env(KvTransferConfig.ENV_TOKEN, null)));
    }

    @Test
    public void anUnreadableOrEmptyTokenFileIsRefusedByName() throws IOException {
        Path missing = Files.createTempDirectory("kv-transfer-token").resolve("absent");
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN_FILE, missing.toString()))).getMessage()
                .contains(KvTransferConfig.ENV_TOKEN_FILE));
        Path empty = Files.createTempFile("kv-transfer-token", ".txt");
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> KvTransferConfig.parse(CONSUMER_WITH_EXTRA, "--kv-transfer-config",
                            env(KvTransferConfig.ENV_TOKEN_FILE, empty.toString())));
        } finally {
            Files.deleteIfExists(empty);
        }
    }

    @Test
    public void theEnvironmentTokenIsPartOfTheValue() {
        String named = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\","
                + " \"engine_id\": \"engine-a\", \"kv_connector_extra_config\": {\"x\": 1}}";
        assertEquals(KvTransferConfig.parse(named, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN, "one")),
                KvTransferConfig.parse(named, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN, "one")));
        assertNotEquals(KvTransferConfig.parse(named, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN, "one")),
                KvTransferConfig.parse(named, "--kv-transfer-config",
                        env(KvTransferConfig.ENV_TOKEN, "two")));
    }

    @Test
    public void theServerPropertyCarriesTheConfigurationJson() {
        assertEquals("neuralfree.server.kv.transfer.config", KvTransferConfig.SERVER_PROPERTY);
        assertEquals("--kv-transfer-config", KvTransferConfig.FLAG);
    }
}
