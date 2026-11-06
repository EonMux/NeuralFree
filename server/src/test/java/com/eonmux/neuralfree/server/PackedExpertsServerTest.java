package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.packed.PackedExpertView;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An experts-packed model serves through the real engine exactly as its unpacked original does: a
 * synthetic llama MoE model is packed with {@code nftool pack-experts}'s pipeline, and the packed
 * file and the original answer a greedy {@code /completion} with the same tokens. A packed file
 * with broken {@code avalon.experts_packed} metadata is refused at open.
 */
class PackedExpertsServerTest {

    private static EngineService original;
    private static EngineService packed;
    private static LlamaCompletionResource originalApi;
    private static LlamaCompletionResource packedApi;
    private static Path packedPath;

    @BeforeAll
    static void open() throws IOException {
        Path model = llamaMoe().writeTempFile();
        model.toFile().deleteOnExit();
        packedPath = Files.createTempFile("nf-server-packed-", ".gguf");
        packedPath.toFile().deleteOnExit();
        try (ModelSource src = ModelSources.open(model.toString())) {
            GgufWriter.write(PackedExpertView.packedViewIfMoe(src), packedPath,
                    new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(true));
        }
        original = new EngineService(model.toString(), pagedConfig(512, 2));
        packed = new EngineService(packedPath.toString(), pagedConfig(512, 2));
        originalApi = new LlamaCompletionResource(original);
        packedApi = new LlamaCompletionResource(packed);
    }

    /**
     * The llama fixture with routed experts on every layer (4 experts, 2 used, expert FFN 32) in
     * place of its dense FFN: the llama graph builds {@code build_moe_ffn} whenever
     * {@code ffn_gate_inp.weight} is present, as llama.cpp does.
     */
    private static GgufTestWriter llamaMoe() {
        GgufTestWriter w = SyntheticModels.llama();
        w.u32("llama.expert_count", 4).u32("llama.expert_used_count", 2)
                .u32("llama.expert_feed_forward_length", 32);
        for (int il = 0; il < 2; il++) {
            w.removeTensor("blk." + il + ".ffn_gate.weight");
            w.removeTensor("blk." + il + ".ffn_up.weight");
            w.removeTensor("blk." + il + ".ffn_down.weight");
            w.tensorRandom("blk." + il + ".ffn_gate_inp.weight", GGMLType.F32, 1000L + il, -0.1f, 0.1f, 64, 4);
            w.tensorRandom("blk." + il + ".ffn_up_exps.weight", GGMLType.F32, 2000L + il, -0.1f, 0.1f, 64, 32, 4);
            w.tensorRandom("blk." + il + ".ffn_gate_exps.weight", GGMLType.F32, 3000L + il, -0.1f, 0.1f, 64, 32, 4);
            w.tensorRandom("blk." + il + ".ffn_down_exps.weight", GGMLType.F32, 4000L + il, -0.1f, 0.1f, 32, 64, 4);
        }
        return w;
    }

    @AfterAll
    static void close() {
        original.close();
        packed.close();
    }

    @Test
    void packedModelGeneratesLikeTheOriginal() {
        List<?> originalTokens = tokens(originalApi);
        List<?> packedTokens = tokens(packedApi);
        assertEquals(8, originalTokens.size());
        assertEquals(originalTokens, packedTokens);
    }

    private static List<?> tokens(LlamaCompletionResource api) {
        try {
            return (List<?>) view(api.completion("""
                    {"prompt": "one two three", "n_predict": 8, "ignore_eos": true, "temperature": 0,
                     "return_tokens": true}"""
                    ).getEntity()).raw("tokens");
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new IllegalStateException("completion failed: " + e.getResponse().getEntity(), e);
        }
    }

    @Test
    void packedModelEncodesLikeTheOriginal() {
        assertArrayEquals(original.encode("one two three", true, true),
                packed.encode("one two three", true, true));
    }

    @Test
    void brokenPackedMetadataIsRefusedAtOpen() throws IOException {
        Path bad = new GgufTestWriter()
                .architecture("llama")
                .u64Array("avalon.experts_packed.blk.0", Arrays.copyOf(new long[14], 13))
                .tensor("blk.0.ffn_exps_packed", GGMLType.I8, new long[]{30208, 4}, new byte[30208 * 4])
                .writeTempFile();
        bad.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class,
                () -> new EngineService(bad.toString(), pagedConfig(512, 2)));
        assertTrue(e.getMessage().contains("holds 13 values"), e.getMessage());
        Files.deleteIfExists(bad);
    }
}
