package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * vLLM's {@code sharded_state} and {@code tensorizer} checkpoints load to the same GGUF view as the Hugging Face
 * safetensors checkpoint their fixtures were generated from.
 *
 * <p>The fixtures under {@code src/test/resources/vllm/fixtures} are written by
 * {@code src/test/resources/vllm/make_load_format_fixtures.py} in the exact on-disk formats: per-rank vLLM module
 * state dicts with the rank-local packing of {@code QKVParallelLinear} (including replicated key/value heads),
 * {@code MergedColumnParallelLinear} and {@code FusedMoE}, the padded vocabulary and vLLM's tied-weight filtering;
 * tensorizer files with the 256 KiB metadata index, sequential headers and, for one fixture, XSalsa20-Poly1305
 * encrypted data.</p>
 */
public class VllmLoadFormatTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static Path fixture(String rel) throws URISyntaxException {
        URL url = VllmLoadFormatTest.class.getResource("/vllm/fixtures/" + rel);
        assertNotNull("fixture resource " + rel, url);
        return Paths.get(url.toURI());
    }

    @Test
    public void shardedStateLlamaMatchesHuggingFace() throws Exception {
        assertParity(fixture("llama-tp2/hf"), fixture("llama-tp2/vllm"));
    }

    @Test
    public void shardedStateQwen3MoeMatchesHuggingFace() throws Exception {
        assertParity(fixture("qwen3moe-tp2/hf"), fixture("qwen3moe-tp2/vllm"));
    }

    @Test
    public void shardedTensorizerQwen3MoeMatchesHuggingFace() throws Exception {
        assertParity(fixture("qwen3moe-tp2/hf"), fixture("qwen3moe-tp2/tensorizer"));
    }

    @Test
    public void tensorizerHoldingAHuggingFaceStateDictMatchesIt() throws Exception {
        assertParity(fixture("llama-untied/hf"), fixture("llama-untied/tensorizer"));
    }

    @Test
    public void aTruncatedTensorizerFileIsRefusedBeforeItsDataIsRead() throws Exception {
        Path copy = tmp.newFolder("truncated").toPath().resolve("model.tensors");
        byte[] whole = java.nio.file.Files.readAllBytes(
                fixture("qwen3moe-tp2/tensorizer/model-0000.tensors"));
        // the magic, version, file header and the 256 KiB metadata area, then only part of the first tensor
        java.nio.file.Files.write(copy, Arrays.copyOf(whole, 5 + 4 + 56 + 8 + 256 * 1024 + 200));
        IOException e = assertThrows(IOException.class, () -> TensorizerFile.open(copy));
        assertTrue(e.getMessage(), e.getMessage().contains("tensorizer"));
        assertTrue(e.getMessage(), e.getMessage().contains("outside the file")
                || e.getMessage().contains("past the file"));
    }

    @Test
    public void aTensorizerMetadataLengthPastTheFileIsRefused() throws Exception {
        Path copy = tmp.newFolder("metadata").toPath().resolve("model.tensors");
        java.nio.file.Files.copy(fixture("qwen3moe-tp2/tensorizer/model-0000.tensors"), copy);
        try (java.io.RandomAccessFile file = new java.io.RandomAccessFile(copy.toFile(), "rw")) {
            file.seek(5 + 4 + 56);
            file.writeLong(0x7fffffffffffffffL);
        }
        IOException e = assertThrows(IOException.class, () -> TensorizerFile.open(copy));
        assertTrue(e.getMessage(), e.getMessage().contains("metadata index length"));
        assertTrue(e.getMessage(), e.getMessage().contains("does not fit"));
    }

    @Test
    public void chunkedEncryptedTensorDecryptsRangesAcrossChunks() throws Exception {
        byte[] key = hex("000102030405060708090a0b0c0d0e0f"
                + "101112131415161718191a1b1c1d1e1f");
        try (TensorizerFile file = TensorizerFile.open(fixture("tensorizer-chunked/model.tensors"), key)) {
            assertEquals(1, file.tensors().size());
            StoredTensor t = file.tensors().get(0);
            assertEquals("wide.weight", t.name());
            assertArrayEquals(new long[]{64, 8}, t.shape());
            // rows 20..40 span the 1024-byte chunk boundary at row 32
            ByteBuffer b = ByteBuffer.allocate(20 * 8 * 4).order(ByteOrder.LITTLE_ENDIAN);
            t.source().readRows(20, 20, b);
            b.flip();
            for (int i = 20 * 8; i < 40 * 8; i++) {
                assertEquals("element " + i, i / 1000f, b.getFloat(), 0f);
            }
        }
    }

    @Test
    public void encryptedTensorizerIsRefusedWithoutAKey() throws Exception {
        assumeTrue("no tensorizer key is configured", System.getenv(TensorizerFile.KEY_ENV) == null
                && System.getenv(TensorizerFile.KEY_FILE_ENV) == null);
        Path file = fixture("llama-encrypted/tensorizer/model.tensors");
        IOException e = assertThrows(IOException.class, () -> TensorizerFile.open(file));
        assertTrue(e.getMessage(), e.getMessage().contains(TensorizerFile.KEY_ENV));
        assertTrue(e.getMessage(), e.getMessage().contains(TensorizerFile.KEY_FILE_ENV));
    }

    @Test
    public void encryptedTensorizerDecryptsWithTheConfiguredKey() throws Exception {
        Path dir = fixture("llama-encrypted/tensorizer");
        byte[] key = hex("000102030405060708090a0b0c0d0e0f"
                + "101112131415161718191a1b1c1d1e1f");
        try (TensorizerFile file = TensorizerFile.open(dir.resolve("model.tensors"), key);
             SafetensorsFile expected = SafetensorsFile.open(fixture("llama-untied/hf/model.safetensors"))) {
            assertEquals(expected.tensors().size(), file.tensors().size());
            for (StoredTensor t : file.tensors()) {
                StTensor want = expected.tensor(t.name());
                assertNotNull("tensor " + t.name(), want);
                assertArrayEquals("tensor " + t.name(), bytes(expected.readAll(want)), bytes(t.source()));
            }
        }
    }

    @Test
    public void encryptedTensorizerCheckpointLoadsWithTheKey() throws Exception {
        Path dir = fixture("llama-encrypted/tensorizer");
        byte[] key = hex("000102030405060708090a0b0c0d0e0f"
                + "101112131415161718191a1b1c1d1e1f");
        HfConfig config = HfConfigLoader.load(dir, (Map<String, Object>) null);
        try (CheckpointContainer container = TensorizerCheckpoint.openCheckpoint(dir, dir, config, key);
             SafetensorsFile expected = SafetensorsFile.open(fixture("llama-untied/hf/model.safetensors"))) {
            assertEquals(expected.tensors().size(), container.tensors().size());
            for (StTensor want : expected.tensors()) {
                StTensor got = container.tensor(want.name());
                assertNotNull("tensor " + want.name(), got);
                assertArrayEquals("tensor " + want.name(), bytes(expected.readAll(want)),
                        bytes(container.source(want.name())));
            }
        }
    }

    // ------------------------------------------------------------------ unit: the packed layouts

    @Test
    public void qkvShardsReassembleInQkvOrderWithReplicatedKvHeads() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        // full packed [q (2 heads of 4), k (1 head of 4), v (1 head of 4)] x hidden 8
        float[][] full = new float[16][8];
        for (int r = 0; r < full.length; r++) {
            for (int c = 0; c < 8; c++) {
                full[r][c] = r * 100 + c;
            }
        }
        List<RowSource> ranks = new ArrayList<>();
        for (int rank = 0; rank < 2; rank++) {
            float[][] local = new float[12][8];
            for (int r = 0; r < 4; r++) {
                local[r] = full[rank * 4 + r];      // q rows of this rank
                local[4 + r] = full[8 + r];         // the one k head, replicated
                local[8 + r] = full[12 + r];        // the one v head, replicated
            }
            ranks.add(source(local));
        }
        List<StoredTensor> assembled = VllmShardAssembly.assemble(ranks("model.layers.0.self_attn.qkv_proj.weight", ranks), config, "test");
        assertEquals(1, assembled.size());
        assertArrayEquals(flatten(full), floats(assembled.get(0).source()), 0f);

        List<StoredTensor> hf = VllmStateDict.toHf(assembled, config);
        Map<String, StoredTensor> byName = storedByName(hf);
        assertArrayEquals(rows(full, 0, 8), floats(byName.get("model.layers.0.self_attn.q_proj.weight").source()), 0f);
        assertArrayEquals(rows(full, 8, 12), floats(byName.get("model.layers.0.self_attn.k_proj.weight").source()), 0f);
        assertArrayEquals(rows(full, 12, 16),
                floats(byName.get("model.layers.0.self_attn.v_proj.weight").source()), 0f);
    }

    @Test
    public void gateUpShardsReassembleGateThenUp() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        // full gate_up [16 gate rows; 16 up rows] x hidden 8
        float[][] full = new float[32][8];
        for (int r = 0; r < full.length; r++) {
            for (int c = 0; c < 8; c++) {
                full[r][c] = r * 100 + c;
            }
        }
        List<RowSource> ranks = new ArrayList<>();
        for (int rank = 0; rank < 2; rank++) {
            float[][] local = new float[16][8];
            for (int r = 0; r < 8; r++) {
                local[r] = full[rank * 8 + r];          // gate rows of this rank
                local[8 + r] = full[16 + rank * 8 + r]; // up rows of this rank
            }
            ranks.add(source(local));
        }
        List<StoredTensor> assembled = VllmShardAssembly.assemble(
                ranks("model.layers.0.mlp.gate_up_proj.weight", ranks), config, "test");
        assertArrayEquals(flatten(full), floats(assembled.get(0).source()), 0f);
    }

    @Test
    public void expertW13ShardsReassembleGateThenUpPerExpert() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("qwen3moe-tp2/hf"), (Map<String, Object>) null);
        int experts = 2;
        int width = 8;
        int hidden = 8;
        float[][] full = new float[experts * width][hidden];
        for (int r = 0; r < full.length; r++) {
            for (int c = 0; c < hidden; c++) {
                full[r][c] = r * 100 + c;
            }
        }
        List<RowSource> ranks = new ArrayList<>();
        for (int rank = 0; rank < 2; rank++) {
            float[][][] local = new float[experts][4][hidden];
            for (int e = 0; e < experts; e++) {
                for (int r = 0; r < 2; r++) {
                    local[e][r] = full[e * width + rank * 2 + r];          // gate rows
                    local[e][2 + r] = full[e * width + 4 + rank * 2 + r];  // up rows
                }
            }
            ranks.add(source(local));
        }
        List<StoredTensor> assembled = VllmShardAssembly.assemble(
                ranks("model.layers.0.mlp.experts.w13_weight", ranks), config, "test");
        assertEquals(1, assembled.size());
        assertArrayEquals(flatten(full), floats(assembled.get(0).source()), 0f);

        List<StoredTensor> hf = VllmStateDict.toHf(assembled, config);
        Map<String, StoredTensor> byName = storedByName(hf);
        for (int e = 0; e < experts; e++) {
            assertArrayEquals("expert " + e + " gate", rows(full, e * width, e * width + 4),
                    floats(byName.get("model.layers.0.mlp.experts." + e + ".gate_proj.weight").source()), 0f);
            assertArrayEquals("expert " + e + " up", rows(full, e * width + 4, e * width + 8),
                    floats(byName.get("model.layers.0.mlp.experts." + e + ".up_proj.weight").source()), 0f);
        }
    }

    @Test
    public void pipelineStagesMergeDisjointTensors() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
        Map<String, StoredTensor> first = new LinkedHashMap<>();
        first.put("model.embed_tokens.weight", StoredTensor.of("model.embed_tokens.weight",
                source(new float[8][8])));
        first.put("model.layers.0.input_layernorm.weight",
                StoredTensor.of("model.layers.0.input_layernorm.weight", source(new float[1][8])));
        ranks.add(new VllmShardAssembly.Rank(0, 0, first));
        Map<String, StoredTensor> last = new LinkedHashMap<>();
        last.put("model.norm.weight", StoredTensor.of("model.norm.weight", source(new float[1][8])));
        last.put("lm_head.weight", StoredTensor.of("lm_head.weight", source(new float[8][8])));
        ranks.add(new VllmShardAssembly.Rank(1, 0, last));

        List<StoredTensor> assembled = VllmShardAssembly.assemble(ranks, config, "test");
        assertEquals(4, assembled.size());
        assertEquals(java.util.Arrays.asList("model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight", "model.norm.weight", "lm_head.weight"),
                names(assembled));

        last.put("model.norm.weight", StoredTensor.of("model.norm.weight",
                source(new float[][]{{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}})));
        ranks.set(1, new VllmShardAssembly.Rank(1, 0, last));
        first.put("model.norm.weight", StoredTensor.of("model.norm.weight",
                source(new float[][]{{8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f}})));
        ranks.set(0, new VllmShardAssembly.Rank(0, 0, first));
        IOException e = assertThrows(IOException.class, () -> VllmShardAssembly.assemble(ranks, config, "test"));
        assertTrue(e.getMessage(), e.getMessage().contains("pipeline stages"));
    }

    @Test
    public void replicatedTensorDifferingBetweenRanksIsRefused() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
        ranks.add(rank(0, 0, "model.norm.weight", source(new float[][]{{1f}})));
        ranks.add(rank(0, 1, "model.norm.weight", source(new float[][]{{2f}})));
        IOException e = assertThrows(IOException.class, () -> VllmShardAssembly.assemble(ranks, config, "test"));
        assertTrue(e.getMessage(), e.getMessage().contains("replicated"));
        assertTrue(e.getMessage(), e.getMessage().contains("model.norm.weight"));
    }

    @Test
    public void aTensorMissingFromOneRankIsRefused() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        RowSource source = source(new float[][]{{1f}});
        List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
        ranks.add(rank(0, 0, "model.layers.0.input_layernorm.weight", source));
        ranks.add(rank(0, 1, "model.layers.0.post_attention_layernorm.weight", source));
        IOException e = assertThrows(IOException.class, () -> VllmShardAssembly.assemble(ranks, config, "test"));
        assertTrue(e.getMessage(), e.getMessage().contains("missing from rank"));
    }

    @Test
    public void anExtraTensorOnOneRankIsRefused() throws Exception {
        HfConfig config = HfConfigLoader.load(fixture("llama-tp2/hf"), (Map<String, Object>) null);
        RowSource source = source(new float[][]{{1f}});
        List<VllmShardAssembly.Rank> ranks = new ArrayList<>();
        ranks.add(rank(0, 0, "model.layers.0.input_layernorm.weight", source));
        ranks.add(rank(0, 1, "model.layers.0.input_layernorm.weight", source,
                "model.layers.0.extra.weight"));
        IOException e = assertThrows(IOException.class, () -> VllmShardAssembly.assemble(ranks, config, "test"));
        assertTrue(e.getMessage(), e.getMessage().contains("model.layers.0.extra.weight"));
    }

    @Test
    public void aFusedUpArchitectureKeepsItsGateUpTensor() throws Exception {
        Path dir = tmp.newFolder("phi3").toPath();
        java.nio.file.Files.write(dir.resolve("config.json"),
                "{\"architectures\":[\"Phi3ForCausalLM\"],\"hidden_size\":8,\"intermediate_size\":16,"
                        .concat("\"num_attention_heads\":2,\"num_key_value_heads\":1,\"num_hidden_layers\":1}")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        HfConfig config = HfConfigLoader.load(dir, (Map<String, Object>) null);
        RowSource gateUp = source(new float[32][8]);
        List<StoredTensor> out = VllmStateDict.toHf(java.util.Collections.singletonList(
                StoredTensor.of("model.layers.0.mlp.gate_up_proj.weight", gateUp)), config);
        assertEquals(1, out.size());
        assertEquals("model.layers.0.mlp.gate_up_proj.weight", out.get(0).name());
        assertArrayEquals(new long[]{32, 8}, out.get(0).shape());
    }

    @Test
    public void llama4FusedExpertsAreTransposedToTheirConverterNames() throws Exception {
        Path dir = tmp.newFolder("llama4").toPath();
        java.nio.file.Files.write(dir.resolve("config.json"),
                ("{\"architectures\":[\"Llama4ForCausalLM\"],\"hidden_size\":8,\"num_experts\":2,"
                        + "\"intermediate_size_moe\":4,\"intermediate_size\":16,\"num_attention_heads\":2,"
                        + "\"num_key_value_heads\":1,\"num_hidden_layers\":1}")
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        HfConfig config = HfConfigLoader.load(dir, (Map<String, Object>) null);
        // vLLM layout [E, gate||up, hidden]; the converter's [E, hidden, gate||up]
        float[][][] w13 = new float[2][8][8];
        for (int e = 0; e < 2; e++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    w13[e][r][c] = e * 1000 + r * 10 + c;
                }
            }
        }
        List<StoredTensor> out = VllmStateDict.toHf(java.util.Collections.singletonList(
                StoredTensor.of("model.layers.0.feed_forward.experts.w13_weight", source(w13))), config);
        assertEquals(1, out.size());
        assertEquals("model.layers.0.feed_forward.experts.gate_up_proj", out.get(0).name());
        assertArrayEquals(new long[]{2, 8, 8}, out.get(0).shape());
        float[] values = floats(out.get(0).source());
        int at = 0;
        for (int e = 0; e < 2; e++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    assertEquals("[" + e + "," + r + "," + c + "]", w13[e][c][r], values[at++], 0f);
                }
            }
        }
    }

    @Test
    public void anExpertParallelShardIsRefusedByName() throws Exception {
        Path dir = tmp.newFolder("ep").toPath();
        java.nio.file.Files.write(dir.resolve("config.json"),
                ("{\"architectures\":[\"Qwen3MoeForCausalLM\"],\"num_experts\":4,\"hidden_size\":8,"
                        + "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":4,"
                        + "\"intermediate_size\":16,\"num_hidden_layers\":1}").getBytes(java.nio.charset.StandardCharsets.UTF_8));
        HfConfig config = HfConfigLoader.load(dir, (Map<String, Object>) null);
        RowSource source = source(new float[2][4][8]);
        List<RowSource> ranks = Arrays.asList(source, source);
        IOException e = assertThrows(IOException.class, () -> VllmShardAssembly.assemble(
                ranks("model.layers.0.mlp.experts.w13_weight", ranks), config, "test"));
        assertTrue(e.getMessage(), e.getMessage().contains("expert-parallel"));
    }

    // ------------------------------------------------------------------ helpers

    private static void assertParity(Path expectedDir, Path actualDir) throws IOException {
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(expectedDir.getParent().resolve("cache-" + actualDir.getFileName())).build();
        try (ModelSource expected = ModelSources.open(expectedDir.toString(), options, null);
             ModelSource actual = ModelSources.open(actualDir.toString(), options, null)) {
            Map<String, String> e = byName(expected.getTensors());
            Map<String, String> a = byName(actual.getTensors());
            assertEquals("tensor names, shapes and types", e, a);
            assertEquals(expected.getString("general.architecture"), actual.getString("general.architecture"));
            Map<String, GGUFTensorInfo> actualInfos = new LinkedHashMap<>();
            for (GGUFTensorInfo t : actual.getTensors()) {
                actualInfos.put(t.name(), t);
            }
            try (WeightStore es = expected.createMappedStore();
                 WeightStore as = actual.createMappedStore()) {
                for (GGUFTensorInfo t : expected.getTensors()) {
                    GGUFTensorInfo other = actualInfos.get(t.name());
                    assertNotNull("tensor " + t.name(), other);
                    byte[] want = new byte[(int) t.sizeInBytes()];
                    es.readTensor(t.offset(), want.length, want);
                    byte[] got = new byte[(int) other.sizeInBytes()];
                    as.readTensor(other.offset(), got.length, got);
                    assertArrayEquals("tensor " + t.name(), want, got);
                }
            }
        }
    }

    private static Map<String, String> byName(List<GGUFTensorInfo> infos) {
        Map<String, String> out = new TreeMap<>();
        for (GGUFTensorInfo t : infos) {
            assertTrue("tensor " + t.name() + " is listed once", out.put(t.name(),
                    Arrays.toString(t.dims()) + " " + t.type()) == null);
        }
        return out;
    }

    private static List<String> names(List<StoredTensor> tensors) {
        List<String> out = new ArrayList<>(tensors.size());
        for (StoredTensor t : tensors) {
            out.add(t.name());
        }
        return out;
    }

    private static Map<String, StoredTensor> storedByName(List<StoredTensor> tensors) {
        Map<String, StoredTensor> out = new LinkedHashMap<>();
        for (StoredTensor t : tensors) {
            out.put(t.name(), t);
        }
        return out;
    }

    private static List<VllmShardAssembly.Rank> ranks(String name, List<RowSource> sources) {
        List<VllmShardAssembly.Rank> out = new ArrayList<>();
        for (int tp = 0; tp < sources.size(); tp++) {
            Map<String, StoredTensor> tensors = new LinkedHashMap<>();
            tensors.put(name, StoredTensor.of(name, sources.get(tp)));
            out.add(new VllmShardAssembly.Rank(0, tp, tensors));
        }
        return out;
    }

    private static VllmShardAssembly.Rank rank(int pp, int tp, String name, RowSource source, String... extraNames) {
        Map<String, StoredTensor> tensors = new LinkedHashMap<>();
        tensors.put(name, StoredTensor.of(name, source));
        for (String extra : extraNames) {
            tensors.put(extra, StoredTensor.of(extra, source));
        }
        return new VllmShardAssembly.Rank(pp, tp, tensors);
    }

    private static RowSource source(float[][] rows) {
        int cols = rows.length == 0 ? 0 : rows[0].length;
        return RopeTensors.fromFloats(new long[]{rows.length, cols}, flatten(rows));
    }

    private static RowSource source(float[][][] experts) {
        int cols = experts.length == 0 || experts[0].length == 0 ? 0 : experts[0][0].length;
        float[] values = new float[experts.length * experts[0].length * cols];
        int at = 0;
        for (float[][] e : experts) {
            for (float[] r : e) {
                System.arraycopy(r, 0, values, at, cols);
                at += cols;
            }
        }
        return RopeTensors.fromFloats(new long[]{experts.length, experts[0].length, cols}, values);
    }

    private static float[] flatten(float[][] rows) {
        float[] out = new float[rows.length * rows[0].length];
        for (int r = 0; r < rows.length; r++) {
            System.arraycopy(rows[r], 0, out, r * rows[0].length, rows[0].length);
        }
        return out;
    }

    private static float[] rows(float[][] rows, int from, int to) {
        return flatten(Arrays.copyOfRange(rows, from, to));
    }

    private static float[] floats(RowSource source) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (source.rows() * source.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(0, (int) source.rows(), b);
        b.flip();
        float[] out = new float[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    private static byte[] bytes(ByteBuffer buffer) {
        byte[] out = new byte[buffer.remaining()];
        buffer.duplicate().get(out);
        return out;
    }

    private static byte[] bytes(RowSource source) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (source.rows() * source.rowBytes()));
        source.readRows(0, (int) source.rows(), b);
        return bytes(b.flip());
    }

    private static byte[] hex(String text) {
        byte[] out = new byte[text.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(text.substring(2 * i, 2 * i + 2), 16);
        }
        return out;
    }
}
