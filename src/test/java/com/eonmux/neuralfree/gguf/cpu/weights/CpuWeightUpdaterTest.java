package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Live weight updates into {@link CpuWeights}: an update swaps the view the weights serve for one tensor and drops
 * its small-vector cache, so the next lookup decodes the new bytes; a wrong tensor, shape or dtype refuses by name,
 * and a tensor that was never mapped holds nothing to update.
 */
public class CpuWeightUpdaterTest {

    private static final int COLS = 64;
    private static final int TOKENS = 5;

    private Path path;
    private CpuWeights weights;
    private CpuWeightUpdater updater;

    @Before
    public void open() throws IOException {
        path = model().writeTempFile();
        ModelSource source = ModelSources.open(path.toString(), ModelSourceOptions.defaults());
        weights = CpuWeights.open(source, TensorCatalog.of(source), null);
        updater = new CpuWeightUpdater(weights, weights.archName());
    }

    @After
    public void close() throws IOException {
        try {
            if (weights != null) {
                weights.close();
            }
        } finally {
            weights = null;
            Files.deleteIfExists(path);
        }
    }

    private static GgufTestWriter model() {
        return new GgufTestWriter()
                .architecture("llama")
                .tensorRandom("token_embd.weight", GGMLType.Q8_0, 11L, -1f, 1f, COLS, TOKENS)
                .tensorFilled("blk.0.attn_norm.weight", GGMLType.F32, 1.0f, COLS)
                .tensorRandom("lm_head.weight", GGMLType.F32, 13L, -0.1f, 0.1f, COLS, TOKENS)
                .tensorFilled("output_norm.weight", GGMLType.F32, 1.0f, COLS);
    }

    private static WeightUpdateTensor f32(String name, long[] ne, float v) {
        float[] fs = new float[(int) (ne[0] * (ne.length == 1 ? 1 : ne[1]))];
        java.util.Arrays.fill(fs, v);
        return WeightUpdateTensor.f32(name, ne, fs);
    }

    private static float[] decoded(CpuWeights w, String name) {
        CpuTensorRef t = w.get(name);
        float[] out = new float[(int) (t.rows() * t.cols())];
        t.decodeRows(0L, (int) t.rows(), out, 0);
        return out;
    }

    @Test
    public void anUpdateSwapsTheServedViewAndDropsTheVectorCache() {
        weights.get("blk.0.attn_norm.weight");
        assertArrayEquals("the fixture tensor is filled with ones", fill(COLS, 1f), decoded(weights,
                "blk.0.attn_norm.weight"), 0f);
        assertArrayEquals(fill(COLS, 1f), weights.floats("blk.0.attn_norm.weight"), 0f);

        updater.apply(f32("blk.0.attn_norm.weight", new long[]{COLS}, 3f));

        assertArrayEquals("the view serves the new bytes", fill(COLS, 3f), decoded(weights,
                "blk.0.attn_norm.weight"), 0f);
        assertArrayEquals("the small-vector cache decoded again", fill(COLS, 3f),
                weights.floats("blk.0.attn_norm.weight"), 0f);
    }

    @Test
    public void theOldViewKeepsTheBytesItServed() {
        CpuTensorRef before = weights.get("blk.0.attn_norm.weight");
        updater.apply(f32("blk.0.attn_norm.weight", new long[]{COLS}, 5f));
        CpuTensorRef after = weights.get("blk.0.attn_norm.weight");
        assertNotSame(before, after);
        assertArrayEquals("a forward pass holding the old view still reads it", fill(COLS, 1f),
                decodedRow(before), 0f);
    }

    private static float[] decodedRow(CpuTensorRef t) {
        float[] out = new float[(int) t.cols()];
        t.decodeRows(0L, 1, out, 0);
        return out;
    }

    @Test
    public void updatesAddressModelAliases() {
        weights.get("lm_head.weight");
        updater.apply(f32("output.weight", new long[]{COLS, TOKENS}, 0.5f));
        assertArrayEquals("output.weight resolves to the stored lm_head.weight", fill(COLS * TOKENS, 0.5f),
                decoded(weights, "lm_head.weight"), 0f);
    }

    @Test
    public void theF32UpdateOfAQuantTensorIsRefused() {
        weights.get("token_embd.weight");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> updater.apply(f32("token_embd.weight", new long[]{COLS, TOKENS}, 1f)));
        assertTrue(e.getMessage(), e.getMessage().contains("token_embd.weight")
                && e.getMessage().contains("would requantize") && e.getMessage().contains("Q8_0"));
    }

    @Test
    public void aWrongShapeIsRefused() {
        weights.get("blk.0.attn_norm.weight");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> updater.apply(f32("blk.0.attn_norm.weight", new long[]{COLS / 2}, 1f)));
        assertTrue(e.getMessage(), e.getMessage().contains("[64]") && e.getMessage().contains("[32]"));
    }

    @Test
    public void aWrongDtypeIsRefused() {
        weights.get("blk.0.attn_norm.weight");
        WeightUpdateTensor bf16 = WeightUpdateTensor.of("blk.0.attn_norm.weight", GGMLType.BF16, new long[]{COLS},
                new byte[COLS * 2]);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> updater.apply(bf16));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_norm.weight") && e.getMessage().contains("BF16"));
    }

    @Test
    public void anUnknownTensorIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> updater.apply(f32("blk.9.attn_norm.weight", new long[]{COLS}, 1f)));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.9.attn_norm.weight"));
    }

    @Test
    public void aNeverMappedTensorRefuses() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> updater.apply(f32("output_norm.weight", new long[]{COLS}, 1f)));
        assertTrue(e.getMessage(), e.getMessage().contains("output_norm.weight") && e.getMessage().contains("never"
                + " mapped"));
    }

    @Test
    public void theUpdaterFollowsTheTransferLifecycle() {
        assertEquals(WeightUpdater.DEFAULT_VERSION, updater.version());
        assertFalse(updater.updating());

        updater.start();
        assertTrue(updater.updating());
        weights.get("blk.0.attn_norm.weight");
        updater.apply(f32("blk.0.attn_norm.weight", new long[]{COLS}, 4f));
        updater.finish("rl-step-3");
        assertFalse(updater.updating());
        assertEquals("rl-step-3", updater.version());

        updater.setVersion("v8");
        assertEquals("v8", updater.version());

        updater.shutdown();
        IllegalStateException e = assertThrows(IllegalStateException.class, updater::start);
        assertTrue(e.getMessage(), e.getMessage().contains("shut down"));
        e = assertThrows(IllegalStateException.class, () -> updater.apply(f32("blk.0.attn_norm.weight",
                new long[]{COLS}, 4f)));
        assertTrue(e.getMessage(), e.getMessage().contains("shut down"));
    }

    @Test
    public void anUpdateAfterTheWeightsClosedIsRefused() throws IOException {
        weights.close();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> updater.apply(f32("blk.0.attn_norm.weight", new long[]{COLS}, 4f)));
        assertTrue(e.getMessage(), e.getMessage().contains("closed"));
    }

    private static float[] fill(int n, float v) {
        float[] fs = new float[n];
        java.util.Arrays.fill(fs, v);
        return fs;
    }
}
