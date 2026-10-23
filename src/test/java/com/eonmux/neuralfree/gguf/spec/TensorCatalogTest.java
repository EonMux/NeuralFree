package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Tensor lookup through {@link TensorCatalog}: GGUF and Hugging Face alias names, required-tensor errors naming the
 * architecture, layer and tensor, and the llama.cpp {@code create_tensor_qkv} / {@code create_tensor_gate_up_exps}
 * resolution of fused and separate projections.
 */
public class TensorCatalogTest {

    private static GGUFTensorInfo info(String name, GGMLType type, long... ne) {
        return new GGUFTensorInfo(name, ne, type, 0L);
    }

    private static TensorCatalog catalog(GGUFTensorInfo... infos) {
        return TensorCatalog.of("llama", Arrays.asList(infos));
    }

    @Test
    public void layerLookupPrefersGgufNamesAndFallsBackToHfAliases() {
        TensorCatalog c = catalog(
                info("blk.0.attn_q.weight", GGMLType.F16, 64, 64),
                info("model.layers.1.self_attn.q_proj.weight", GGMLType.F16, 64, 64),
                info("model.layers.1.mlp.gate_proj.weight", GGMLType.F16, 64, 128),
                info("model.layers.1.input_layernorm.weight", GGMLType.F32, 64));
        assertEquals("blk.0.attn_q.weight", c.layer(0, "attn_q.weight").name());
        assertEquals("model.layers.1.self_attn.q_proj.weight", c.layer(1, "attn_q.weight").name());
        assertEquals("model.layers.1.mlp.gate_proj.weight", c.layer(1, "ffn_gate.weight").name());
        assertEquals("model.layers.1.input_layernorm.weight", c.layer(1, "attn_norm.weight").name());
        assertNull(c.layer(1, "attn_k.weight"));
        assertTrue(c.hasLayer(0, "attn_q.weight"));
        assertEquals(2, c.blockCount());
    }

    @Test
    public void modelLevelLookupUsesTheModelAliasTable() {
        TensorCatalog c = catalog(
                info("model.embed_tokens.weight", GGMLType.F16, 64, 1000),
                info("lm_head.weight", GGMLType.F16, 64, 1000));
        assertEquals("model.embed_tokens.weight", c.get("token_embd.weight").name());
        assertEquals("lm_head.weight", c.require("output.weight").name());
        assertFalse(c.has("output.weight"));
        assertNull(c.get("output_norm.weight"));
    }

    @Test
    public void requireNamesArchLayerAndTensor() {
        TensorCatalog c = catalog(info("blk.0.attn_q.weight", GGMLType.F16, 64, 64));
        MissingTensorException e = assertThrows(MissingTensorException.class, () -> c.require(3, "ffn_down.weight"));
        assertEquals("llama", e.arch());
        assertEquals(3, e.layer());
        assertEquals("blk.3.ffn_down.weight", e.tensor());
        assertTrue(e.getMessage().contains("llama"));
        assertTrue(e.getMessage().contains("layer 3"));
        assertTrue(e.getMessage().contains("blk.3.ffn_down.weight"));

        MissingTensorException top = assertThrows(MissingTensorException.class, () -> c.require("output_norm.weight"));
        assertEquals(-1, top.layer());
    }

    @Test
    public void otherStacksResolveWithoutAliases() {
        TensorCatalog c = catalog(info("enc.blk.2.attn_q.weight", GGMLType.F16, 64, 64));
        assertNotNull(c.layer(TensorNames.STACK_ENC, 2, "attn_q.weight"));
        assertNull(c.layer(2, "attn_q.weight"));
        assertThrows(MissingTensorException.class, () -> c.require(TensorNames.STACK_DEC, 2, "attn_q.weight"));
    }

    @Test
    public void fusedQkvWithFusedBias() {
        TensorCatalog c = catalog(
                info("blk.0.attn_qkv.weight", GGMLType.Q8_0, 64, 64 + 32 + 32),
                info("blk.0.attn_qkv.bias", GGMLType.F32, 128),
                info("blk.0.attn_q.bias", GGMLType.F32, 64));
        TensorCatalog.FusedQkv f = c.fusedQkv(0);
        assertNotNull(f);
        assertEquals(128, f.rows());
        assertEquals("blk.0.attn_qkv.bias", f.bias().name());
        assertNull(f.qBias());
        f.validate(64, 32, 32);
        assertThrows(IllegalStateException.class, () -> f.validate(64, 64, 64));
    }

    @Test
    public void fusedQkvWithSeparateBiases() {
        TensorCatalog c = catalog(
                info("blk.1.attn_qkv.weight", GGMLType.F16, 64, 128),
                info("blk.1.attn_q.bias", GGMLType.F32, 64),
                info("blk.1.attn_k.bias", GGMLType.F32, 32),
                info("blk.1.attn_v.bias", GGMLType.F32, 32));
        TensorCatalog.FusedQkv f = c.fusedQkv(1);
        assertNull(f.bias());
        assertEquals("blk.1.attn_k.bias", f.kBias().name());
        f.validate(64, 32, 32);
        assertNull(c.fusedQkv(0));
    }

    @Test
    public void gateUpExpsPrefersTheFusedTensor() {
        TensorCatalog c = catalog(
                info("blk.0.ffn_gate_up_exps.weight", GGMLType.MXFP4, 64, 2 * 96, 8),
                info("blk.0.ffn_gate_up_exps.bias", GGMLType.F32, 2 * 96, 8),
                info("blk.0.ffn_up_exps.scale", GGMLType.F32, 8));
        TensorCatalog.GateUpExps g = c.gateUpExps(0);
        assertTrue(g.fused());
        assertTrue(g.gated());
        assertEquals(96, g.nFf());
        assertEquals(8, g.nExpert());
        assertNotNull(g.gateUpBias());
        assertEquals("blk.0.ffn_up_exps.scale", g.upScale().name());
        assertNull(g.gateScale());
    }

    @Test
    public void gateUpExpsSeparateAndUngated() {
        TensorCatalog c = catalog(
                info("blk.0.ffn_gate_exps.weight", GGMLType.Q4_K, 256, 96, 8),
                info("blk.0.ffn_up_exps.weight", GGMLType.Q4_K, 256, 96, 8),
                info("blk.0.ffn_gate_exps.scale", GGMLType.F32, 8),
                info("blk.1.ffn_up_exps.weight", GGMLType.Q4_K, 256, 96, 8),
                info("blk.2.ffn_gate_exps.weight", GGMLType.Q4_K, 256, 96, 8));
        TensorCatalog.GateUpExps sep = c.gateUpExps(0);
        assertFalse(sep.fused());
        assertTrue(sep.gated());
        assertEquals(96, sep.nFf());
        assertNotNull(sep.gateScale());

        TensorCatalog.GateUpExps ungated = c.gateUpExps(1);
        assertFalse(ungated.gated());

        assertThrows(MissingTensorException.class, () -> c.gateUpExps(2));
        assertNull(c.gateUpExps(3));
    }

    @Test
    public void denseUpFusedNeedsDoubleRowsAndNoGate() {
        TensorCatalog c = catalog(
                info("blk.0.ffn_up.weight", GGMLType.F16, 64, 256),
                info("blk.1.ffn_up.weight", GGMLType.F16, 64, 256),
                info("blk.1.ffn_gate.weight", GGMLType.F16, 64, 256));
        assertTrue(c.denseUpFused(0, 128));
        assertFalse(c.denseUpFused(0, 256));
        assertFalse(c.denseUpFused(1, 128));
    }

    @Test
    public void duplicateNamesAreRejected() {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        infos.add(info("blk.0.attn_q.weight", GGMLType.F16, 64, 64));
        infos.add(info("blk.0.attn_q.weight", GGMLType.F16, 64, 64));
        assertThrows(IllegalArgumentException.class, () -> TensorCatalog.of("llama", infos));
    }

    @Test
    public void catalogFromAParsedFileUsesTheArchitectureKey() throws IOException {
        Path path = new GgufTestWriter()
                .architecture("qwen3")
                .tensorZeros("token_embd.weight", GGMLType.F32, 8, 16)
                .tensorZeros("blk.0.attn_q.weight", GGMLType.F32, 8, 8)
                .writeTempFile();
        try {
            GGUFFileParser parser = new GGUFFileParser(path.toString());
            try {
                TensorCatalog c = TensorCatalog.of(parser);
                assertEquals("qwen3", c.archName());
                assertEquals(2, c.size());
                TensorRef q = c.require(0, "attn_q.weight");
                assertEquals(GGMLType.F32, q.type());
                assertEquals(8, q.ne(1));
                assertEquals(1, q.ne(2));
                MissingTensorException e = assertThrows(MissingTensorException.class, () -> c.require(0, "attn_k.weight"));
                assertTrue(e.getMessage().contains("qwen3"));
            } finally {
                parser.close();
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
