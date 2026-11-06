package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.layerNorm;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lfm2a / d1omni_a conformer tower against a double-precision transcription of llama.cpp
 * tools/mtmd/models/conformer.cpp ({@code clip_graph_conformer}, {@link ConformerReference}): the subsampling
 * stem, the blocks with the 512-wide relative-position input and the graph's rel shift, the GELU-erf adapter and
 * the d1omni_a residual block; the {@code ((((n + 1) / 2) + 1) / 2 + 1) / 2} row count and the shape refusals.
 */
public class ConformerTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 16;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int TEXT = 6;
    static final int ADAPTER = 10;
    static final int RESIDUAL = 7;
    static final int POSITION_ROWS = 1600;

    static Map<String, Object> meta(String projector) {
        return AudioTowerSupport.meta(projector, N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, 1e-5f);
    }

    static MmprojFixture fixture(boolean d1omni) {
        MmprojFixture f = new MmprojFixture(d1omni ? 62 : 61);
        ConformerReference.stem(f, N_MEL, HIDDEN, 4, 5, 3);
        for (int il = 0; il < LAYERS; il++) {
            ConformerReference.block(f, il, true, HIDDEN, HEADS, FF, ConformerTower.POS_WIDTH,
                    ConformerTower.CONV_KERNEL);
        }
        f.norm("mm.a.mlp.0.weight", "mm.a.mlp.0.bias", HIDDEN);
        f.weight("mm.a.mlp.1.weight", HIDDEN, ADAPTER);
        f.weight("mm.a.mlp.1.bias", ADAPTER);
        f.weight("mm.a.mlp.3.weight", ADAPTER, TEXT);
        f.weight("mm.a.mlp.3.bias", TEXT);
        if (!d1omni) {
            // lfm2a files carry the LFM2 audio-output embedding here; llama.cpp reads its width as the output width
            f.weight(ConformerTower.POSITION_EMBD, TEXT, POSITION_ROWS);
        }
        if (d1omni) {
            f.norm("mm.a.mlp.4.weight", "mm.a.mlp.4.bias", TEXT);
            f.weight("mm.a.mlp.5.weight", TEXT, RESIDUAL);
            f.weight("mm.a.mlp.5.bias", RESIDUAL);
            f.weight("mm.a.mlp.6.weight", RESIDUAL, TEXT);
            f.weight("mm.a.mlp.6.bias", TEXT);
        }
        return f;
    }

    /** {@code inv_freq} of the clip.cpp LFM2A position fill. */
    static double[] posFreq() {
        double[] fr = new double[ConformerTower.POS_WIDTH / 2];
        for (int i = 0; i < fr.length; i++) {
            fr[i] = Math.exp(-(Math.log(10000.0) / 512) * (2.0 * i));
        }
        return fr;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk, boolean d1omni) {
        double[][] x = ConformerReference.stem(f, chunk);
        double[][] pos = ConformerReference.posEmb(2 * x.length - 1, ConformerTower.POS_WIDTH, posFreq());
        for (int il = 0; il < LAYERS; il++) {
            x = ConformerReference.block(f, il, true, x, pos, HEADS, ConformerTower.CONV_KERNEL, 1e-5);
        }
        x = adapter(f, x, 0, 1, 3);
        if (d1omni) {
            x = add(x, adapter(f, x, 4, 5, 6));
        }
        return x;
    }

    private static double[][] adapter(MmprojFixture f, double[][] x, int norm, int up, int down) {
        String p = "mm.a.mlp.";
        double[][] h = layerNorm(x, f.get(p + norm + ".weight"), f.get(p + norm + ".bias"), 1e-5);
        return ffn(h, f.get(p + up + ".weight"), f.get(p + up + ".bias"), AudioTowerSupport::geluErf,
                f.get(p + down + ".weight"), f.get(p + down + ".bias"));
    }

    @Test
    public void lfm2aMatchesReference() {
        MmprojFixture f = fixture(false);
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 37, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta("lfm2a"))) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk, false), out, TOL);
            assertEquals(5, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void d1omniAddsTheResidualBlock() {
        MmprojFixture f = fixture(true);
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 29, 6);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta("d1omni_a"))) {
            AudioTowerSupport.assertRows(reference(f, chunk, true), enc.encode(chunk), TOL);
        }
    }

    @Test
    public void rowCountIsThreeStrideTwoConvs() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(false), meta("lfm2a"))) {
            ClipAudioModel m = enc.model();
            for (int n = 1; n <= 40; n++) {
                assertEquals(((((n + 1) / 2) + 1) / 2 + 1) / 2, m.nTokens(n));
                assertEquals(enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length, m.nTokens(n));
            }
            assertEquals(375, m.nTokens(3000));
        }
    }

    @Test
    public void refusesARelativePositionProjectionOfAnotherWidth() {
        MmprojFixture f = fixture(false);
        f.weight("a.blk.1.linear_pos.weight", HIDDEN, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta("lfm2a")), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.1.linear_pos.weight"));
    }

    @Test
    public void refusesAConvModuleKernelOtherThanNine() {
        MmprojFixture f = fixture(false);
        f.weight("a.blk.0.conv_dw.weight", 7, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta("lfm2a")), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.0.conv_dw.weight"));
    }

    @Test
    public void refusesAMissingAttentionBiasAndD1omniWithoutItsResidualBlock() {
        MmprojFixture f = fixture(false);
        f.remove("a.blk.0.attn_k.bias");
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta("lfm2a")), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("attn_k.bias"));
        MmprojFixture g = fixture(false);
        MissingTensorException d = assertThrows(MissingTensorException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta("d1omni_a")), g.catalog()));
        assertTrue(d.getMessage(), d.getMessage().contains("mm.a.mlp.4"));
    }

    @Test
    public void refusesMoreFramesThanTwiceThePositionRows() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(false), meta("lfm2a"))) {
            int most = 2 * POSITION_ROWS + 1;
            assertEquals(enc.model().nTokens(most), enc.encode(AudioTowerSupport.chunk(N_MEL, most, 1)).length);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(AudioTowerSupport.chunk(N_MEL, 2 * POSITION_ROWS + 2, 1)));
            assertTrue(e.getMessage(), e.getMessage().contains(ConformerTower.POSITION_EMBD));
        }
    }

    @Test
    public void readsTheOptionalBlockBiasesWhenAbsent() {
        MmprojFixture f = fixture(false);
        for (int il = 0; il < LAYERS; il++) {
            for (String t : new String[]{"ln1.bias", "ln2.bias", "ffn_up.bias", "ffn_down.bias"}) {
                f.remove("a.blk." + il + "." + t);
            }
        }
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 19, 7);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta("lfm2a"))) {
            AudioTowerSupport.assertRows(reference(f, chunk, false), enc.encode(chunk), TOL);
        }
    }
}
