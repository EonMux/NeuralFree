package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The parakeet tower against a double-precision transcription of llama.cpp tools/mtmd/models/parakeet.cpp
 * ({@code clip_graph_parakeet}, {@link ConformerReference}): the subsampling stem, the bias-free blocks with the
 * batch-norm conv module and the {@code n_embd}-wide relative-position input, the local attention past 8192 rows,
 * the RMS-normed ReLU-squared projector; the {@code ceil(n / 8)} row count and the refusals.
 */
public class ParakeetTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 16;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int TEXT = 6;
    static final int ADAPTER = 10;
    static final int KERNEL = 5;
    static final float EPS = 1e-5f;

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("parakeet", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, EPS);
        m.put("clip.audio.subsampling_factor", 8);
        m.put("clip.audio.conv_kernel_size", KERNEL);
        return m;
    }

    static MmprojFixture fixture(int layers) {
        MmprojFixture f = new MmprojFixture(71);
        f.put(ParakeetTower.MEL_FILTERS, 0f, 0.05f, 257, N_MEL);
        f.put(ParakeetTower.WINDOW, 0f, 1f, 400);
        ConformerReference.stem(f, N_MEL, HIDDEN, 4, 5, 3);
        for (int il = 0; il < layers; il++) {
            ConformerReference.block(f, il, false, HIDDEN, HEADS, FF, HIDDEN, KERNEL);
        }
        f.put("mm.a.norm_pre.weight", 0.8f, 1.2f, HIDDEN);
        f.weight("mm.a.mlp.1.weight", HIDDEN, ADAPTER);
        f.weight("mm.a.mlp.2.weight", ADAPTER, TEXT);
        return f;
    }

    /** {@code pos_freqs} of clip.cpp: {@code expf(-(2k * logf(1e4) / n_state))}. */
    static double[] posFreq() {
        double[] fr = new double[HIDDEN / 2];
        for (int k = 0; k < fr.length; k++) {
            fr[k] = Math.exp(-(2.0 * k * Math.log(10000.0) / HIDDEN));
        }
        return fr;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk, int layers) {
        double[][] x = ConformerReference.stem(f, chunk);
        int n = x.length;
        int reach = n > 8192 ? 128 : n - 1;
        double[][] pos = ConformerReference.posEmb(2 * reach + 1, HIDDEN, posFreq());
        for (int il = 0; il < layers; il++) {
            x = ConformerReference.block(f, il, false, x, pos, HEADS, KERNEL, EPS);
        }
        x = rmsNorm(x, f.get("mm.a.norm_pre.weight"), null, 1e-6);
        return ffn(x, f.get("mm.a.mlp.1.weight"), null, AudioTowerSupport::reluSqr, f.get("mm.a.mlp.2.weight"), null);
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture(LAYERS);
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 45, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk, LAYERS), out, TOL);
            assertEquals(6, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void attendsLocallyPastTheThreshold() {
        // 8193 rows: each row sees the 128 rows on either side
        MmprojFixture f = fixture(1);
        Map<String, Object> m = meta();
        m.put("clip.audio.block_count", 1);
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 8 * 8192 + 1, 9);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, m)) {
            float[][] out = enc.encode(chunk);
            assertEquals(8193, out.length);
            AudioTowerSupport.assertRows(reference(f, chunk, 1), out, TOL);
        }
        assertEquals(8191, ParakeetTower.attentionReach(8192));
        assertEquals(128, ParakeetTower.attentionReach(8193));
    }

    @Test
    public void rowCountIsTheCeilingOfAnEighth() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(LAYERS), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 1; n <= 40; n++) {
                assertEquals((n + 7) / 8, m.nTokens(n));
                assertEquals(enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length, m.nTokens(n));
            }
        }
    }

    @Test
    public void refusesAKernelOtherThanTheConvKernelSize() {
        MmprojFixture f = fixture(LAYERS);
        f.weight("a.blk.1.conv_dw.weight", KERNEL + 2, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.1.conv_dw.weight"));
    }

    @Test
    public void refusesAFilterbankOfAnotherSize() {
        MmprojFixture f = fixture(LAYERS);
        f.put(ParakeetTower.MEL_FILTERS, 0f, 0.05f, 256, N_MEL);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains(ParakeetTower.MEL_FILTERS));
    }

    @Test
    public void refusesMissingBatchNormStatisticsAndProjector() {
        MmprojFixture f = fixture(LAYERS);
        f.remove("a.blk.0.conv_norm_var");
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("conv_norm_var"));
        MmprojFixture g = fixture(LAYERS);
        g.remove("mm.a.norm_pre.weight");
        MissingTensorException p = assertThrows(MissingTensorException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), g.catalog()));
        assertTrue(p.getMessage(), p.getMessage().contains("mm.a.norm_pre.weight"));
    }
}
