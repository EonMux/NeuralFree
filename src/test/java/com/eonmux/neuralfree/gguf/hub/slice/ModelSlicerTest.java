package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ModelSlicer} end to end against a local Hugging Face look-alike: the refs and tree APIs list a two-shard
 * repository, the shards and the auxiliary files are served with range support, and the run resolves the commit,
 * reads the headers, plans the slice, fetches exactly the kept spans and writes the shard, the patched config, the
 * auxiliary files and the metadata. The output is read back with {@link SafetensorsFile} and every kept tensor's
 * bytes are compared with the source tensor's prefix, so a wrong offset, a missed renumbering or a bad row slice
 * fails. Nothing leaves the machine.
 */
public class ModelSlicerTest {

    private static final String REPO = "user/model";
    private static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    private static final String SHARD_1 = "model-00001-of-00002.safetensors";
    private static final String SHARD_2 = "model-00002-of-00002.safetensors";
    private static final String INDEX = "model.safetensors.index.json";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final Map<String, byte[]> repository = new LinkedHashMap<>();
    private final Map<String, byte[]> source = new LinkedHashMap<>();
    private HttpServer server;
    private ExecutorService executor;
    private HubClient client;

    @Before
    public void start() throws IOException {
        Assume.assumeFalse("the slicer is online only", HubClient.offline());
        repository.put("config.json", config().getBytes(StandardCharsets.UTF_8));
        repository.put("tokenizer.json", "{\"model\":{}}".getBytes(StandardCharsets.UTF_8));
        repository.put("README.md", "# tiny model\n".getBytes(StandardCharsets.UTF_8));
        repository.put("model.bin", new byte[16]);
        repository.put("onnx/model.onnx", new byte[8]);
        repository.put(SHARD_1, shard1());
        repository.put(SHARD_2, shard2());
        repository.put(INDEX, index().getBytes(StandardCharsets.UTF_8));

        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        executor = Executors.newFixedThreadPool(4);
        server.setExecutor(executor);
        server.createContext("/", this::handle);
        server.start();
        String endpoint = "http://127.0.0.1:" + server.getAddress().getPort() + "/";
        client = new HubClient(new HubCache(tmp.newFolder("cache").toPath()), null, endpoint);
    }

    @After
    public void stop() {
        if (server != null) {
            server.stop(0);
        }
        if (executor != null) {
            executor.shutdownNow();
        }
    }

    // ================================================================== fixture

    private static String config() {
        return "{\"num_hidden_layers\": 2, \"vocab_size\": 8, \"num_experts\": 4,"
                + " \"num_experts_per_tok\": 2, \"hidden_size\": 2,"
                + " \"layer_types\": [\"dense\", \"moe\"]}";
    }

    private static String index() {
        List<String> names = new ArrayList<>();
        for (int e = 0; e < 4; e++) {
            names.add("model.layers.0.mlp.experts." + e + ".w1.weight");
        }
        names.add("model.embed_tokens.weight");
        names.add("model.layers.0.self_attn.q_proj.weight");
        names.add("model.layers.0.mlp.gate.weight");
        names.add("model.layers.0.mlp.experts.gate_up_proj");
        StringBuilder map = new StringBuilder();
        for (String name : names) {
            if (map.length() > 0) {
                map.append(',');
            }
            map.append('"').append(name).append("\":\"").append(SHARD_1).append('"');
        }
        for (String name : new String[]{"model.layers.1.self_attn.q_proj.weight", "model.norm.weight",
                "lm_head.weight"}) {
            map.append(',').append('"').append(name).append("\":\"").append(SHARD_2).append('"');
        }
        for (int e = 0; e < 4; e++) {
            map.append(',').append('"').append("model.layers.1.mlp.experts.").append(e)
                    .append(".w1.weight\":\"").append(SHARD_2).append('"');
        }
        for (String name : new String[]{"model.layers.1.mlp.gate.weight",
                "model.layers.1.mlp.experts.gate_up_proj"}) {
            map.append(',').append('"').append(name).append("\":\"").append(SHARD_2).append('"');
        }
        return "{\"metadata\":{\"total_size\":0},\"weight_map\":{" + map + "}}";
    }

    private byte[] shard1() throws IOException {
        SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        tensor(builder, "model.embed_tokens.weight", new long[]{8, 2}, 0, 64);
        tensor(builder, "model.layers.0.self_attn.q_proj.weight", new long[]{2}, 1, 8);
        for (int e = 0; e < 4; e++) {
            tensor(builder, "model.layers.0.mlp.experts." + e + ".w1.weight", new long[]{2}, 10 + e, 8);
        }
        tensor(builder, "model.layers.0.mlp.gate.weight", new long[]{4, 2}, 20, 32);
        tensor(builder, "model.layers.0.mlp.experts.gate_up_proj", new long[]{4, 2}, 60, 32);
        return write(builder, "shard1.safetensors");
    }

    private byte[] shard2() throws IOException {
        SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        tensor(builder, "model.layers.1.self_attn.q_proj.weight", new long[]{2}, 2, 8);
        for (int e = 0; e < 4; e++) {
            tensor(builder, "model.layers.1.mlp.experts." + e + ".w1.weight", new long[]{2}, 30 + e, 8);
        }
        tensor(builder, "model.layers.1.mlp.gate.weight", new long[]{4, 2}, 100, 32);
        tensor(builder, "model.layers.1.mlp.experts.gate_up_proj", new long[]{4, 2}, 140, 32);
        tensor(builder, "model.norm.weight", new long[]{2}, 3, 8);
        tensor(builder, "lm_head.weight", new long[]{8, 2}, 200, 64);
        return write(builder, "shard2.safetensors");
    }

    private void tensor(SafetensorsWriter.Builder builder, String name, long[] shape, int seed, int length) {
        byte[] data = ramp(seed, length);
        builder.add(name, StDType.F32, shape, data);
        source.put(name, data);
    }

    private byte[] write(SafetensorsWriter.Builder builder, String name) throws IOException {
        Path path = tmp.getRoot().toPath().resolve(name);
        builder.build().write(path);
        return Files.readAllBytes(path);
    }

    private static byte[] ramp(int start, int length) {
        byte[] data = new byte[length];
        for (int i = 0; i < length; i++) {
            data[i] = (byte) (start + i);
        }
        return data;
    }

    private Path output() {
        return tmp.getRoot().toPath().resolve("slice");
    }

    private SliceOptions.Builder options() {
        return SliceOptions.builder().output(output()).keepLayers(Arrays.asList(1)).numExperts(2).vocabSize(4)
                .jobs(4).maxBytes(1L << 20);
    }

    // ================================================================== the fake hub

    private void handle(HttpExchange exchange) throws IOException {
        try {
            String path = exchange.getRequestURI().getRawPath();
            String api = "/api/models/" + REPO;
            if (path.equals(api + "/refs")) {
                send(exchange, 200, refs());
            } else if (path.startsWith(api + "/tree/")) {
                send(exchange, 200, tree());
            } else if (path.startsWith("/" + REPO + "/resolve/" + COMMIT + "/")) {
                serve(exchange, path.substring(path.lastIndexOf('/') + 1));
            } else {
                exchange.sendResponseHeaders(404, -1);
            }
        } finally {
            exchange.close();
        }
    }

    private byte[] refs() {
        Map<String, Object> branch = new LinkedHashMap<>();
        branch.put("name", "main");
        branch.put("targetCommit", COMMIT);
        return Json.toString(java.util.Collections.singletonList(branch), Json.Style.CANONICAL)
                .getBytes(StandardCharsets.UTF_8);
    }

    private byte[] tree() {
        List<Map<String, Object>> entries = new ArrayList<>();
        for (Map.Entry<String, byte[]> e : repository.entrySet()) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("type", "file");
            entry.put("path", e.getKey());
            entry.put("size", (long) e.getValue().length);
            entries.add(entry);
        }
        return Json.toString(entries, Json.Style.CANONICAL).getBytes(StandardCharsets.UTF_8);
    }

    private void send(HttpExchange exchange, int status, byte[] body) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }

    private void serve(HttpExchange exchange, String name) throws IOException {
        byte[] body = repository.get(name);
        if (body == null) {
            exchange.sendResponseHeaders(404, -1);
            return;
        }
        String range = exchange.getRequestHeaders().getFirst("Range");
        if (range == null) {
            exchange.sendResponseHeaders(200, body.length);
            write(exchange, body, 0, body.length);
            return;
        }
        String span = range.substring("bytes=".length());
        int dash = span.indexOf('-');
        long start = Long.parseLong(span.substring(0, dash));
        String tail = span.substring(dash + 1);
        long end = tail.isEmpty() ? body.length - 1L : Long.parseLong(tail);
        long length = end - start + 1L;
        exchange.getResponseHeaders().set("Content-Range", "bytes " + start + "-" + end + "/" + body.length);
        exchange.sendResponseHeaders(206, length);
        write(exchange, body, (int) start, (int) length);
    }

    private static void write(HttpExchange exchange, byte[] body, int start, int length) throws IOException {
        OutputStream out = exchange.getResponseBody();
        out.write(body, start, length);
        out.flush();
    }

    // ================================================================== the slice

    @Test
    public void buildsTheSliceShardConfigAuxAndMetadata() throws Exception {
        SliceReport report = ModelSlicer.run(client, HubRef.parse(REPO), options().build());

        assertEquals(REPO, report.repository());
        assertEquals(COMMIT, report.commit());
        assertEquals(17, report.totalTensors());
        assertEquals(8, report.keptTensors());
        assertEquals(output(), report.directory());
        assertNull(report.inspect());

        Path shard = output().resolve("model.safetensors");
        assertTrue(Files.isRegularFile(shard));
        try (SafetensorsFile file = SafetensorsFile.open(shard)) {
            assertEquals("pt", file.metadata().get("format"));
            assertEquals(8, file.tensors().size());
            assertBytePrefix(file, "model.layers.0.self_attn.q_proj.weight",
                    source.get("model.layers.1.self_attn.q_proj.weight"), 8);
            assertBytePrefix(file, "model.layers.0.mlp.experts.0.w1.weight",
                    source.get("model.layers.1.mlp.experts.0.w1.weight"), 8);
            assertBytePrefix(file, "model.layers.0.mlp.experts.1.w1.weight",
                    source.get("model.layers.1.mlp.experts.1.w1.weight"), 8);
            assertBytePrefix(file, "model.layers.0.mlp.gate.weight",
                    source.get("model.layers.1.mlp.gate.weight"), 16);
            assertBytePrefix(file, "model.layers.0.mlp.experts.gate_up_proj",
                    source.get("model.layers.1.mlp.experts.gate_up_proj"), 16);
            assertBytePrefix(file, "model.embed_tokens.weight", source.get("model.embed_tokens.weight"), 32);
            assertBytePrefix(file, "lm_head.weight", source.get("lm_head.weight"), 32);
            assertBytePrefix(file, "model.norm.weight", source.get("model.norm.weight"), 8);
            assertArrayEquals(new long[]{2, 2},
                    file.tensor("model.layers.0.mlp.gate.weight").shape());
            assertArrayEquals(new long[]{4, 2}, file.tensor("model.embed_tokens.weight").shape());
            assertNull(file.tensor("model.layers.1.self_attn.q_proj.weight"));
        }

        Map<?, ?> config = (Map<?, ?>) Json.parse(
                new String(Files.readAllBytes(output().resolve("config.json")), StandardCharsets.UTF_8),
                Json.Limits.PYTHON);
        assertEquals(1L, config.get("num_hidden_layers"));
        assertEquals(4L, config.get("vocab_size"));
        assertEquals(2L, config.get("num_experts"));
        assertEquals(2L, config.get("num_experts_per_tok"));
        assertEquals(java.util.Collections.singletonList("moe"), config.get("layer_types"));
        assertTrue(report.configPatches().contains("num_hidden_layers: 2 -> 1"));
        assertTrue(report.configPatches().contains("vocab_size: 8 -> 4"));
        assertTrue(report.configPatches().contains("num_experts: 4 -> 2"));

        assertEquals(new String(repository.get("tokenizer.json"), StandardCharsets.UTF_8),
                new String(Files.readAllBytes(output().resolve("tokenizer.json")), StandardCharsets.UTF_8));
        assertTrue(Files.isRegularFile(output().resolve("README.md")));
        assertFalse("a weight in another format is not copied", Files.exists(output().resolve("model.bin")));
        assertFalse("an onnx export is not copied", Files.exists(output().resolve("onnx/model.onnx")));
        assertFalse("the index is replaced by the single shard", Files.exists(output().resolve(INDEX)));
        assertFalse("no source shard is copied",
                Files.exists(output().resolve(SHARD_1)) || Files.exists(output().resolve(SHARD_2)));

        Map<?, ?> meta = (Map<?, ?>) Json.parse(
                new String(Files.readAllBytes(output().resolve(".slice_meta.json")), StandardCharsets.UTF_8),
                Json.Limits.PYTHON);
        assertEquals(REPO, meta.get("repo"));
        assertEquals(COMMIT, meta.get("revision"));
        assertEquals(8L, meta.get("tensors"));
        assertEquals(17L, meta.get("total_tensors"));
        assertEquals(report.bytes(), meta.get("size"));
        assertTrue(meta.get("config_patches") instanceof List);
    }

    @Test
    public void dryRunPlansWithoutWriting() throws Exception {
        SliceReport report = ModelSlicer.run(client, HubRef.parse(REPO), options().dryRun(true).build());
        assertEquals(8, report.keptTensors());
        assertNull(report.directory());
        assertTrue(report.dryRun());
        assertFalse("a dry run creates nothing", Files.exists(output()));
    }

    @Test
    public void inspectReportsTheFamiliesWithoutWriting() throws Exception {
        SliceReport report = ModelSlicer.run(client, HubRef.parse(REPO), options().inspect(true).build());
        assertNotNull(report.inspect());
        assertTrue(report.inspect().contains("numeric families in tensor names"));
        assertTrue(report.inspect().contains("kind=layer"));
        assertTrue(report.inspect().contains("kind=expert"));
        assertTrue(report.inspect().contains("8/17 tensors kept"));
        assertFalse(Files.exists(output()));
    }

    @Test
    public void refusesASliceOverTheSizeBudget() {
        IOException e = assertThrows(IOException.class,
                () -> ModelSlicer.run(client, HubRef.parse(REPO), options().maxBytes(4L).build()));
        assertTrue(e.getMessage(), e.getMessage().contains("max-size"));
        assertFalse(Files.exists(output()));
    }

    @Test
    public void refusesANonEmptyOutputDirectoryWithoutOverwrite() throws Exception {
        ModelSlicer.run(client, HubRef.parse(REPO), options().build());
        IOException e = assertThrows(IOException.class,
                () -> ModelSlicer.run(client, HubRef.parse(REPO), options().build()));
        assertTrue(e.getMessage(), e.getMessage().contains("not empty"));
        SliceReport again = ModelSlicer.run(client, HubRef.parse(REPO), options().overwrite(true).build());
        assertEquals(output(), again.directory());
    }

    @Test
    public void refusesARepositoryWithoutSafetensorsWeights() throws Exception {
        repository.remove(SHARD_1);
        repository.remove(SHARD_2);
        repository.remove(INDEX);
        IOException e = assertThrows(IOException.class,
                () -> ModelSlicer.run(client, HubRef.parse(REPO), options().build()));
        assertTrue(e.getMessage(), e.getMessage().contains("no .safetensors weights"));
    }

    @Test
    public void refusesAnIndexNamingAShardTheListingDoesNotCarry() throws Exception {
        repository.put(INDEX, "{\"weight_map\":{\"t\":\"gone.safetensors\"}}"
                .getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class,
                () -> ModelSlicer.run(client, HubRef.parse(REPO), options().build()));
        assertTrue(e.getMessage(), e.getMessage().contains("gone.safetensors"));
    }

    @Test
    public void auxiliaryBlacklistCoversOtherWeightFormatsAndMedia() {
        assertTrue(SliceWriter.auxBlacklisted("model.bin"));
        assertTrue(SliceWriter.auxBlacklisted("onnx/model.onnx"));
        assertTrue(SliceWriter.auxBlacklisted("images/logo.png"));
        assertTrue(SliceWriter.auxBlacklisted(".gitattributes"));
        assertTrue(SliceWriter.auxBlacklisted("original/consolidated.pth"));
        assertTrue(SliceWriter.auxBlacklisted("weights/model.mlpackage/data"));
        assertFalse(SliceWriter.auxBlacklisted("tokenizer.json"));
        assertFalse(SliceWriter.auxBlacklisted("README.md"));
        assertFalse(SliceWriter.auxBlacklisted("chat_template.jinja"));
    }

    // ================================================================== helpers

    /** Asserts the output tensor's bytes are the first {@code length} bytes of {@code expected}. */
    private static void assertBytePrefix(SafetensorsFile file, String name, byte[] expected, int length)
            throws IOException {
        StTensor tensor = file.tensor(name);
        assertNotNull("the output holds " + name, tensor);
        ByteBuffer buffer = file.readAll(tensor);
        byte[] got = new byte[buffer.remaining()];
        buffer.get(got);
        assertArrayEquals(name, Arrays.copyOf(expected, length), got);
    }
}
