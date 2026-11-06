package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The two KV transfer connectors: a producer's payload pulls from a shared storage directory and from a socket
 * listener, under the same transfer id; a missing id, another engine's payload and a wrong shared token refuse by
 * name; and each direction a role does not serve refuses by name.
 */
public class KvConnectorTest {

    private static final int[] PROMPT = {4, 11, 18, 25, 32, 39, 46, 53, 60, 67};

    private Path directory;

    @Before
    public void makeDirectory() throws IOException {
        directory = Files.createTempDirectory("nf-kv-transfer-");
    }

    @After
    public void removeDirectory() {
        try (java.util.stream.Stream<Path> paths = Files.walk(directory)) {
            paths.sorted(java.util.Comparator.reverseOrder()).forEach(file -> {
                try {
                    Files.deleteIfExists(file);
                } catch (IOException ignored) {
                    // a leftover temp file does not fail the test
                }
            });
        } catch (IOException ignored) {
            // the directory is temporary
        }
    }

    private static KvTransferPayload payload(String id, String engine, int[] tokens, byte[] chain) {
        return new KvTransferPayload(id, engine, tokens, chain);
    }

    private KvTransferConfig config(String connector, String role, String extra) {
        return config(connector, role, "127.0.0.1", 0, extra, java.util.Collections.emptyMap());
    }

    /** The configuration of a connector bound to {@code ip:port} whose token comes from {@code env}. */
    private KvTransferConfig config(String connector, String role, String ip, int port, String extra,
                                    Map<String, String> env) {
        return KvTransferConfig.parse("{\"kv_connector\": \"" + connector + "\", \"kv_role\": \"" + role + "\","
                + " \"engine_id\": \"" + role + "-engine\", \"kv_ip\": \"" + ip + "\", \"kv_port\": " + port + ","
                + " \"kv_connector_extra_config\": " + extra + "}", "--kv-transfer-config", env);
    }

    /** The configuration of a connector whose shared token comes from {@code NEURALFREE_KV_TRANSFER_TOKEN}. */
    private KvTransferConfig configWithEnvironmentToken(String connector, String role, String ip, String extra,
                                                        String token) {
        return config(connector, role, ip, 0, extra,
                java.util.Collections.singletonMap(KvTransferConfig.ENV_TOKEN, token));
    }

    private String storage() {
        return "{\"storage_path\": \"" + directory.toString().replace("\\", "\\\\") + "\"}";
    }

    @Test
    public void sharedStorageCarriesAPayloadBetweenRoles() throws IOException {
        byte[] chain = "chain-bytes".getBytes(StandardCharsets.UTF_8);
        try (FileKvConnector producer = new FileKvConnector(config("FileKvConnector", "kv_producer", storage()));
             FileKvConnector consumer = new FileKvConnector(config("FileKvConnector", "kv_consumer", storage()))) {
            producer.save(payload("transfer-1", "kv_producer-engine", PROMPT, chain));
            KvTransferPayload loaded = consumer.load("transfer-1", "kv_producer-engine",
                    KvTransferEndpoint.none());
            assertEquals("transfer-1", loaded.transferId());
            assertEquals("kv_producer-engine", loaded.engineId());
            assertArrayEquals(PROMPT, loaded.promptTokens());
            assertArrayEquals(chain, loaded.chain());
            assertTrue("a prompt that extends the exported tokens is covered",
                    loaded.covers(new int[] {4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74}));
            assertFalse(loaded.covers(new int[] {4, 11, 19, 25, 32, 39, 46, 53, 60, 67}));
            assertFalse("a prompt shorter than the export is not covered", loaded.covers(new int[] {4, 11, 18}));

            IOException otherEngine = assertThrows(IOException.class,
                    () -> consumer.load("transfer-1", "another-engine", KvTransferEndpoint.none()));
            assertTrue(otherEngine.getMessage(), otherEngine.getMessage().contains("another-engine"));
            IOException missing = assertThrows(IOException.class,
                    () -> consumer.load("missing", null, KvTransferEndpoint.none()));
            assertTrue(missing.getMessage(), missing.getMessage().contains("missing"));

            consumer.release("transfer-1");
            assertThrows(IOException.class, () -> consumer.load("transfer-1", null, KvTransferEndpoint.none()));
            assertThrows(UnsupportedOperationException.class, () -> consumer.save(
                    payload("x", "y", PROMPT, chain)));
            assertThrows(UnsupportedOperationException.class,
                    () -> producer.load("transfer-1", null, KvTransferEndpoint.none()));
        }
    }

    @Test
    public void theSocketCarrierPullsUnderTheTransferId() throws IOException {
        byte[] chain = new byte[4096];
        for (int i = 0; i < chain.length; i++) {
            chain[i] = (byte) (i * 31);
        }
        try (TcpKvConnector producer = new TcpKvConnector(config("TcpKvConnector", "kv_producer",
                "{\"auth_token\": \"shared\"}"));
             TcpKvConnector consumer = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                     "{\"auth_token\": \"shared\"}"))) {
            assertTrue(producer.endpoint().addressed());
            assertFalse("a consumer binds no listener", consumer.endpoint().addressed());
            producer.save(payload("transfer-2", "kv_producer-engine", PROMPT, chain));
            KvTransferPayload loaded = consumer.load("transfer-2", "kv_producer-engine", producer.endpoint());
            assertArrayEquals(chain, loaded.chain());
            assertArrayEquals(PROMPT, loaded.promptTokens());

            IOException unknown = assertThrows(IOException.class,
                    () -> consumer.load("missing", null, producer.endpoint()));
            assertTrue(unknown.getMessage(), unknown.getMessage().contains("missing"));
            IOException otherEngine = assertThrows(IOException.class,
                    () -> consumer.load("transfer-2", "another-engine", producer.endpoint()));
            assertTrue(otherEngine.getMessage(), otherEngine.getMessage().contains("another-engine"));

            producer.release("transfer-2");
            assertThrows(IOException.class, () -> consumer.load("transfer-2", null, producer.endpoint()));
        }
    }

    @Test
    public void aWrongSharedTokenRefusesThePull() throws IOException {
        try (TcpKvConnector producer = new TcpKvConnector(config("TcpKvConnector", "kv_producer",
                "{\"auth_token\": \"right\"}"));
             TcpKvConnector consumer = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                     "{\"auth_token\": \"wrong\"}"))) {
            producer.save(payload("transfer-3", "kv_producer-engine", PROMPT, new byte[] {1, 2, 3}));
            IOException refused = assertThrows(IOException.class,
                    () -> consumer.load("transfer-3", null, producer.endpoint()));
            assertTrue(refused.getMessage(), refused.getMessage().contains("token"));
        }
    }

    @Test
    public void aConsumerWithoutAnAddressRefusesByConnector() throws IOException {
        try (TcpKvConnector consumer = new TcpKvConnector(config("TcpKvConnector", "kv_consumer", "{}"))) {
            IOException refused = assertThrows(IOException.class,
                    () -> consumer.load("transfer-4", null, KvTransferEndpoint.none()));
            assertTrue(refused.getMessage(), refused.getMessage().contains("remote_host"));
        }
    }

    @Test
    public void unknownExtraConfigFieldsRefuseByConnector() {
        IllegalArgumentException tcp = assertThrows(IllegalArgumentException.class,
                () -> new TcpKvConnector(config("TcpKvConnector", "kv_consumer", "{\"wat\": 1}")));
        assertTrue(tcp.getMessage(), tcp.getMessage().contains("wat"));
        IllegalArgumentException file = assertThrows(IllegalArgumentException.class,
                () -> new FileKvConnector(config("FileKvConnector", "kv_consumer", "{\"wat\": 1}")));
        assertTrue(file.getMessage(), file.getMessage().contains("wat"));
        IllegalArgumentException noPath = assertThrows(IllegalArgumentException.class,
                () -> new FileKvConnector(config("FileKvConnector", "kv_consumer", "{}")));
        assertTrue(noPath.getMessage(), noPath.getMessage().contains("storage_path"));
    }

    @Test
    public void aPayloadRefusesEmptyParts() {
        assertThrows(IllegalArgumentException.class,
                () -> payload("", "engine", PROMPT, new byte[] {1}));
        assertThrows(IllegalArgumentException.class,
                () -> payload("id", "engine", new int[0], new byte[] {1}));
        assertThrows(IllegalArgumentException.class,
                () -> payload("id", "engine", new int[] {1, -1}, new byte[] {1}));
        assertThrows(IllegalArgumentException.class,
                () -> payload("id", "engine", PROMPT, new byte[0]));
        KvTransferPayload one = payload("id", "engine", new int[] {4}, new byte[] {1});
        assertEquals(5, one.bytes());
        assertTrue(Arrays.equals(new int[] {4}, one.promptTokens()));
    }

    @Test
    public void aProducerOnANonLoopbackAddressWithoutATokenRefusesToStart() {
        for (String role : new String[] {"kv_producer", "kv_both"}) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> new TcpKvConnector(config("TcpKvConnector", role, "0.0.0.0", 0, "{}",
                            java.util.Collections.emptyMap())));
            assertTrue(refused.getMessage(), refused.getMessage().contains("0.0.0.0"));
            assertTrue(refused.getMessage(), refused.getMessage().contains("auth_token"));
            assertTrue(refused.getMessage(), refused.getMessage().contains(KvTransferConfig.ENV_TOKEN));
            assertTrue(refused.getMessage(), refused.getMessage().contains(KvTransferConfig.ENV_TOKEN_FILE));
        }
    }

    @Test
    public void aProducerOnANonLoopbackAddressWithATokenStarts() throws IOException {
        try (TcpKvConnector producer = new TcpKvConnector(configWithEnvironmentToken("TcpKvConnector",
                "kv_producer", "0.0.0.0", "{}", "shared"))) {
            assertTrue(producer.endpoint().addressed());
            producer.save(payload("transfer-nonloopback", "kv_producer-engine", PROMPT, new byte[] {1, 2, 3}));
            producer.release("transfer-nonloopback");
        }
    }

    @Test
    public void theEnvironmentTokenCarriesBothSidesOfASocketTransfer() throws IOException {
        byte[] chain = "chain-from-env-token".getBytes(StandardCharsets.UTF_8);
        try (TcpKvConnector producer = new TcpKvConnector(configWithEnvironmentToken("TcpKvConnector",
                "kv_producer", "127.0.0.1", "{}", "shared"));
             TcpKvConnector consumer = new TcpKvConnector(configWithEnvironmentToken("TcpKvConnector",
                     "kv_consumer", "127.0.0.1", "{}", "shared"))) {
            producer.save(payload("transfer-env", "kv_producer-engine", PROMPT, chain));
            KvTransferPayload loaded = consumer.load("transfer-env", "kv_producer-engine", producer.endpoint());
            assertArrayEquals(chain, loaded.chain());
        }
    }

    @Test
    public void aRemoteHostOutsideTheAllowlistRefusesBeforeAnyConnection() throws IOException {
        try (TcpKvConnector producer = new TcpKvConnector(config("TcpKvConnector", "kv_producer",
                "{\"auth_token\": \"shared\"}"));
             TcpKvConnector consumer = new TcpKvConnector(config("TcpKvConnector", "kv_consumer", "{}"));
             TcpKvConnector emptyList = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                     "{\"allowed_remote_hosts\": []}"))) {
            producer.save(payload("transfer-5", "kv_producer-engine", PROMPT, new byte[] {1, 2, 3}));
            KvTransferEndpoint elsewhere = KvTransferEndpoint.of("10.9.8.7", 14579);
            for (TcpKvConnector pulling : new TcpKvConnector[] {consumer, emptyList}) {
                IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                        () -> pulling.load("transfer-5", "kv_producer-engine", elsewhere));
                assertTrue(refused.getMessage(), refused.getMessage().contains("allowed_remote_hosts"));
            }
            assertEquals("the producer saw no connection at all", 0L, producer.stats().get("loaded"));
            assertEquals("the producer saw no rejected pull either", 0L, producer.stats().get("rejected"));
        }
    }

    @Test
    public void anAllowlistEntryPermitsOnlyThePullsItNames() throws IOException {
        int port;
        try (TcpKvConnector producer = new TcpKvConnector(config("TcpKvConnector", "kv_producer",
                "{\"auth_token\": \"shared\"}"))) {
            port = producer.endpoint().port();
            producer.save(payload("transfer-6", "kv_producer-engine", PROMPT, new byte[] {5, 6, 7}));
            try (TcpKvConnector range = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                    "{\"auth_token\": \"shared\", \"allowed_remote_hosts\": [\"127.0.0.0/8\"]}"));
                 TcpKvConnector exactPort = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                         "{\"auth_token\": \"shared\", \"allowed_remote_hosts\":"
                                 + " [\"127.0.0.1:" + port + "\"]}"));
                 TcpKvConnector otherPort = new TcpKvConnector(config("TcpKvConnector", "kv_consumer",
                         "{\"auth_token\": \"shared\", \"allowed_remote_hosts\":"
                                 + " [\"127.0.0.1:" + (port + 1) + "\"]}"))) {
                KvTransferPayload loaded = range.load("transfer-6", "kv_producer-engine", producer.endpoint());
                assertArrayEquals(new byte[] {5, 6, 7}, loaded.chain());
                assertEquals(exactPort.load("transfer-6", "kv_producer-engine", producer.endpoint()).transferId(),
                        "transfer-6");
                IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                        () -> otherPort.load("transfer-6", "kv_producer-engine", producer.endpoint()));
                assertTrue(refused.getMessage(), refused.getMessage().contains("allowed_remote_hosts"));
                assertEquals(2L, producer.stats().get("loaded"));
                assertEquals("the refused consumer never reached the producer", 0L,
                        producer.stats().get("rejected"));
            }
        }
    }

    @Test
    public void theFileConnectorValidatesTheAllowlistItAccepts() throws IOException {
        try (FileKvConnector accepting = new FileKvConnector(config("FileKvConnector", "kv_consumer",
                storage().substring(0, storage().length() - 1)
                        + ", \"allowed_remote_hosts\": [\"10.0.0.0/8\"]}"))) {
            assertEquals("kv_consumer-engine", accepting.engineId());
        }
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> new FileKvConnector(config("FileKvConnector", "kv_consumer",
                        storage().substring(0, storage().length() - 1)
                                + ", \"allowed_remote_hosts\": [\"no host\"]}")));
        assertTrue(refused.getMessage(), refused.getMessage().contains("allowed_remote_hosts"));
    }

    @Test
    public void aTransferIdCannotNameAFileOutsideTheStorageDirectory() throws IOException {
        Path storage = directory.resolve("storage");
        byte[] sentinel = "untouched".getBytes(StandardCharsets.UTF_8);
        Path beside = directory.resolve("escape.nfkv");
        Files.write(beside, sentinel);
        String storageJson = "{\"storage_path\": \"" + storage.toString().replace("\\", "\\\\") + "\"}";
        byte[] chain = "chain-inside".getBytes(StandardCharsets.UTF_8);
        try (FileKvConnector producer = new FileKvConnector(config("FileKvConnector", "kv_producer",
                storageJson));
             FileKvConnector consumer = new FileKvConnector(config("FileKvConnector", "kv_consumer",
                     storageJson))) {
            for (String id : new String[] {"../../escape", "../../../escape", "..\\..\\escape", "a/b"}) {
                producer.save(payload(id, "kv_producer-engine", PROMPT, chain));
                KvTransferPayload loaded = consumer.load(id, "kv_producer-engine", KvTransferEndpoint.none());
                assertEquals(id, loaded.transferId());
                assertArrayEquals(chain, loaded.chain());
                if (!id.equals("a/b")) {
                    consumer.release(id);
                }
            }
            assertThrows(IOException.class,
                    () -> consumer.load("../../escape", null, KvTransferEndpoint.none()));
            assertArrayEquals("the file a traversal id would name outside the directory keeps its bytes",
                    sentinel, Files.readAllBytes(beside));
            try (java.util.stream.Stream<Path> files = Files.list(storage)) {
                for (Path file : files.collect(java.util.stream.Collectors.toList())) {
                    String name = file.getFileName().toString();
                    assertTrue(name, name.matches("[A-Za-z0-9_-]+\\.nfkv"));
                    byte[] decoded = Base64.getUrlDecoder().decode(
                            name.substring(0, name.length() - ".nfkv".length()));
                    assertArrayEquals("the file name is the base64 of the transfer id",
                            "a/b".getBytes(StandardCharsets.UTF_8), decoded);
                }
            }
        }
    }
}
