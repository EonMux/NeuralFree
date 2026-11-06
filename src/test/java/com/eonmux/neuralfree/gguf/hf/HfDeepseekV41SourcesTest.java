package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfExtraTensors;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDeepseekV41Fixtures;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek-V4.1 text converter ({@code DeepseekV41Model}, deepseek41.py:117-292) on a full synthetic checkpoint:
 * the fixed configuration, the name filter, the V4.1 tensor names, the compressed-cache and candidate-selection
 * metadata and the Engram layout with its concatenated {@code per_layer_token_embd} table. The test drives the entry's
 * hooks; {@link HfDeepseekV41EngramRoundTripTest} loads the exported file and compares its forward logits with an
 * independent evaluation of the Engram math.
 */
public class HfDeepseekV41SourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theV41FixtureConvertsToTheTextTensorsAndMetadata() throws IOException {
        Path dir = HfDeepseekV41Fixtures.deepseekV41().write(tmp.newFolder("v41").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        HfConfig cfg = HfConfigLoader.load(dir, options);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        HfConfig fixed = entry.configFixup().apply(cfg);
        assertEquals(2, entry.blockCount(fixed));

        try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
            HfTensorIndex raw = HfTensorIndex.of(ckpt);
            HfLoadContext ctx = HfLoadContext.builder(fixed, entry, Arch.DEEPSEEK41)
                    .index(raw).checkpoint(ckpt).options(options).build();
            HfNameFilter filter = entry.nameFilter(ctx);
            Map<String, RowSource> filtered = new LinkedHashMap<>();
            for (String name : raw.names()) {
                String mapped = filter.filter(name, n -> n);
                if (mapped != null) {
                    filtered.put(mapped, raw.get(name));
                }
            }
            HfTensorIndex index = HfTensorIndex.of(filtered);
            HfLoadContext planCtx = HfLoadContext.builder(fixed, entry, Arch.DEEPSEEK41)
                    .index(index).checkpoint(ckpt).options(options).build();
            Sink sink = new Sink();
            for (String name : index.names()) {
                entry.tensors().apply(planCtx, name, index.get(name), sink);
            }
            if (entry.extra() != HfExtraTensors.NONE) {
                entry.extra().generate(planCtx, sink);
            }

            Set<String> t = sink.added.keySet();
            for (String name : new String[]{"token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_q_a.weight", "blk.0.attn_q_b.weight", "blk.0.attn_q_a_norm.weight",
                    "blk.0.attn_kv.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.attn_output_a.weight",
                    "blk.0.attn_output_b.weight", "blk.0.attn_sinks.weight", "blk.0.ffn_gate_inp.weight",
                    "blk.0.exp_probs_b.bias", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                    "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.hc_attn_fn.weight",
                    "blk.0.hc_ffn_fn.weight", "blk.1.attn_compressor_kv.weight", "blk.1.indexer.attn_q_b.weight",
                    "blk.1.indexer.attn_k.weight", "blk.1.indexer.proj.weight"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("an MTP tensor leaked: " + name, name.startsWith("mtp"));
            }

            HfMetadataBuilder kv = new HfMetadataBuilder(Arch.DEEPSEEK41.ggufName());
            entry.meta().apply(planCtx, kv, null);
            HfMetadata m = kv.build();
            assertEquals(2, m.get("deepseek41.block_count"));
            assertEquals(64, m.get("deepseek41.embedding_length"));
            assertEquals(Arrays.asList(0, 1), m.get("deepseek41.attention.kv_source_layers"));
            assertEquals(Arrays.asList(1, 1), m.get("deepseek41.attention.indexer.source_layers"));
            assertEquals(0, m.get("deepseek41.attention.indexer.candidate_source_layer"));
            assertEquals(4, m.get("deepseek41.attention.indexer.candidate_top_k_blocks"));
            assertEquals(64, m.get("deepseek41.attention.indexer.candidate_block_size"));
            assertEquals(0, m.get("deepseek41.hash_layer_count"));
        }
    }

    @Test
    public void theEngramFixtureWritesThePleLayoutAndTheConcatenatedTable() throws IOException {
        Path dir = HfDeepseekV41Fixtures.deepseekV41Engram().write(tmp.newFolder("engram").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-engram").toPath()).build();
        HfConfig cfg = HfConfigLoader.load(dir, options);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        HfConfig fixed = entry.configFixup().apply(cfg);
        try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
            HfTensorIndex raw = HfTensorIndex.of(ckpt);
            HfLoadContext ctx = HfLoadContext.builder(fixed, entry, Arch.DEEPSEEK41)
                    .index(raw).checkpoint(ckpt).options(options).build();
            HfNameFilter filter = entry.nameFilter(ctx);
            Map<String, RowSource> filtered = new LinkedHashMap<>();
            for (String name : raw.names()) {
                String mapped = filter.filter(name, n -> n);
                if (mapped != null) {
                    filtered.put(mapped, raw.get(name));
                }
            }
            HfTensorIndex index = HfTensorIndex.of(filtered);
            HfLoadContext planCtx = HfLoadContext.builder(fixed, entry, Arch.DEEPSEEK41)
                    .index(index).checkpoint(ckpt).options(options)
                    .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();
            Sink sink = new Sink();
            for (String name : index.names()) {
                entry.tensors().apply(planCtx, name, index.get(name), sink);
            }
            entry.extra().generate(planCtx, sink);

            Set<String> t = sink.added.keySet();
            for (String name : new String[]{"per_layer_token_embd.weight", "blk.0.engram_gate.weight",
                    "blk.0.engram_kv.weight"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("an Engram table leaked as its own tensor: " + name, name.contains("engram.embed"));
            }

            RowSource table = sink.added.get("per_layer_token_embd.weight");
            assertArrayEquals(new long[]{928, 64}, table.shape());
            assertEquals(StDType.BF16, table.format().stDType());
            assertTableValues(table);
            assertEngramMetadata(planCtx);
        }
    }

    /** The table concatenates the two layers' dequantized E4M3 rows in layer order. */
    private static void assertTableValues(RowSource table) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate((int) Math.multiplyExact(table.rowBytes(), table.rows()))
                .order(ByteOrder.LITTLE_ENDIAN);
        table.readRows(0, (int) table.rows(), bytes);
        bytes.flip();
        int dim = 64;
        for (int r = 0; r < 928; r++) {
            int row = r < 420 ? r : r - 420;
            for (int c = 0; c < dim; c++) {
                int code = (row * 7 + c * 13) % 127;
                int exponent = 3 + (c < 32 ? row : row + 71) % 244;
                float expected = FpBlockDecoders.e4m3ToFp32(code)
                        * Float.intBitsToFloat(exponent << 23);
                short actual = bytes.getShort();
                assertEquals("row " + r + ", column " + c, (short) HalfFloats.floatToBFloat16(expected), actual);
            }
        }
    }

    /** deepseek41.py:158-175: the Engram layout, the multipliers and the compressed token map. */
    private static void assertEngramMetadata(HfLoadContext ctx) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.DEEPSEEK41.ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList(0, 1), m.get("deepseek41.ple.layers"));
        assertEquals(GGUFValueType.UINT32, m.elementType("deepseek41.ple.layers"));
        assertEquals(3, m.get("deepseek41.ple.ngram_size"));
        assertEquals(2, m.get("deepseek41.ple.heads_per_ngram"));
        assertEquals(64, m.get("deepseek41.embedding_length_per_layer_input"));
        assertEquals(Arrays.asList(0L, 101L, 204L, 311L, 420L, 533L, 660L, 791L),
                m.get("deepseek41.ple.head_offsets"));
        assertEquals(Arrays.asList(101L, 103L, 107L, 109L, 113L, 127L, 131L, 137L),
                m.get("deepseek41.ple.head_vocab_sizes"));
        assertEquals(Arrays.asList(839276373626933875L, 355477604520660829L, 53987722132071145L,
                        1084803951630661891L, 68513285976946179L, 509045121352709105L),
                m.get("deepseek41.ple.layer_multipliers"));
        assertEquals(7, m.get("deepseek41.engram.pad_token_id"));
        List<Integer> tokenMap = cast(m.get("deepseek41.engram.token_map"));
        assertEquals(Arrays.asList(0, 1, 2, 3, 4, 5, 6), tokenMap);
    }

    @SuppressWarnings("unchecked")
    private static List<Integer> cast(Object value) {
        return (List<Integer>) value;
    }

    /** Records the planned tensors; the conversion writes no tensor-derived metadata. */
    private static final class Sink implements TensorSink {
        private final Map<String, RowSource> added = new LinkedHashMap<>();
        private final Set<String> consumed = new LinkedHashSet<>();

        @Override
        public void add(String ggufName, RowSource src) {
            if (added.put(ggufName, src) != null) {
                throw new AssertionError("planned twice: " + ggufName);
            }
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            throw new AssertionError("unexpected tensor-derived key " + key);
        }

        @Override
        public void drop(String hfName, String reason) {
        }

        @Override
        public boolean isConsumed(String hfName) {
            return consumed.contains(hfName);
        }

        @Override
        public void consume(String hfName) {
            consumed.add(hfName);
        }
    }
}
