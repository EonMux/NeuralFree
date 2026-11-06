package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end load and forward of a synthetic Qwen2 model carrying the NeuralFree {@code attention.dual_chunk.*}
 * GGUF keys: the hparams loader reads them back into the attention spec (every softmax layer runs the dense
 * dual-chunk attention), a context of several chunk lengths prefills and decodes with finite logits, a context shift
 * is refused by name, and an enabled sparse attention is refused by name.
 */
public class DualChunkModelTest {

    private static final int CHUNK = 8;
    private static final int LOCAL = 2;
    private static final int ORIGINAL_MAX = 4;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32)
                .typeV(KvDType.F32).build()).build();
    }

    private static Path model() throws IOException {
        return SyntheticModels.qwen2DualChunk(SyntheticModels.Config.qwen2(), CHUNK, LOCAL, ORIGINAL_MAX)
                .writeTempFile();
    }

    private static int[] tokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 3 + (i * 17) % 250;
        }
        return t;
    }

    @Test
    public void theGgufKeysReachTheAttentionSpec() throws IOException {
        try (GGUFModel model = new GGUFModel(model().toString(), config())) {
            int dual = 0;
            for (LayerSpec l : model.spec().layers()) {
                AttentionSpec a = l.attention();
                if (a == null) {
                    continue;
                }
                assertNotNull("every qwen2 layer is dual-chunk: " + l.il(), a.dualChunk());
                assertEquals(CHUNK, a.dualChunk().chunkSize());
                assertEquals(LOCAL, a.dualChunk().localSize());
                assertEquals(ORIGINAL_MAX, a.dualChunk().originalMaxPositionEmbeddings());
                assertEquals(5L * a.qDim(), a.qRopeDim());
                dual++;
            }
            assertTrue(dual > 0);
            String shift = model.spec().contextShiftRefusal(model.spec().layers().get(0));
            assertNotNull("a dual-chunk context cannot shift", shift);
            assertTrue(shift, shift.contains("dual-chunk"));
            String modelWide = model.spec().contextShiftRefusal();
            assertNotNull("the model-wide check refuses a dual-chunk context", modelWide);
            assertTrue(modelWide, modelWide.contains("dual-chunk"));
        }
    }

    @Test
    public void aContextOfSeveralChunksPrefillsAndDecodes() throws IOException {
        try (GGUFModel model = new GGUFModel(model().toString(), config())) {
            ForwardBackend f = model.forwardBackend();
            int[] prompt = tokens(14);
            int[] rows = new int[prompt.length];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = i;
            }
            float[][] prefill = f.forwardRows(prompt, 0, rows);
            float[][] decode = f.forwardRows(new int[]{prompt[0]}, prompt.length, new int[]{0});
            float[][] decode2 = f.forwardRows(new int[]{prompt[1]}, prompt.length + 1, new int[]{0});
            assertEquals(prompt.length, prefill.length);
            assertFinite(prefill);
            assertFinite(decode);
            assertFinite(decode2);
            boolean moved = false;
            for (int v = 0; v < prefill[prefill.length - 1].length; v++) {
                moved |= Math.abs(prefill[prefill.length - 1][v] - decode[0][v]) > 1e-4f;
            }
            assertTrue("the decode's logits follow the last prefill row", moved);
        }
    }

    private static EngineConfig quantized(boolean rotation) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.Q8_0)
                .typeV(KvDType.Q8_0).kvRotation(rotation).build()).build();
    }

    /**
     * A rotated quantized cache rotates every query plane like the keys it is scored against: with the rotation off
     * the same cache type must produce the same logits up to quantization noise. The fixture has 64-value heads, the
     * narrowest rotation block.
     */
    @Test
    public void aRotatedQuantizedCacheKeepsTheDualChunkLogits() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.qwen2().toBuilder()
                .nEmbd(128).nHead(2).nHeadKv(1).headDim(64).nRot(64).nFf(256).build();
        Path path = SyntheticModels.qwen2DualChunk(c, CHUNK, LOCAL, ORIGINAL_MAX).writeTempFile();
        int[] prompt = tokens(14);
        int[] rows = new int[prompt.length];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        float[][] rotated;
        try (GGUFModel model = new GGUFModel(path.toString(), quantized(true))) {
            rotated = model.forwardBackend().forwardRows(prompt, 0, rows);
        }
        float[][] plain;
        try (GGUFModel model = new GGUFModel(path.toString(), quantized(false))) {
            plain = model.forwardBackend().forwardRows(prompt, 0, rows);
        }
        assertEquals(plain.length, rotated.length);
        for (int r = 0; r < plain.length; r++) {
            assertEquals(plain[r].length, rotated[r].length);
            for (int v = 0; v < plain[r].length; v++) {
                float a = plain[r][v];
                assertTrue("row " + r + " value " + v + ": rotated " + rotated[r][v] + " plain " + a,
                        Math.abs(a - rotated[r][v]) <= 5e-2f * Math.max(1f, Math.abs(a)));
            }
        }
    }

    @Test
    public void anEnabledSparseAttentionIsRefusedByName() throws IOException {
        Path path = SyntheticModels.qwen2DualChunk(SyntheticModels.Config.qwen2(), CHUNK, LOCAL, ORIGINAL_MAX,
                true).writeTempFile();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> new GGUFModel(path.toString(), config()).close());
        assertTrue(e.getMessage(), e.getMessage().contains("sparse_attention_enabled"));
    }

    private static void assertFinite(float[][] rows) {
        for (int r = 0; r < rows.length; r++) {
            for (int v = 0; v < rows[r].length; v++) {
                assertTrue("row " + r + " value " + v + " is " + rows[r][v],
                        Float.isFinite(rows[r][v]));
            }
        }
    }
}
