package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise1d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.frames;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The raw-waveform audio towers against double-precision transcriptions of llama.cpp tools/mtmd/models:
 * gemma4ua.cpp (RMS norm over each 640-sample frame, then the input projection) and muse-spark-audio.cpp (the
 * folded patch projections, the causal depthwise-conv residual blocks and the RMS-normed projection), with their
 * mtmd-audio.cpp framings and row-count formulas.
 */
public class RawWaveTowersTest {

    private static final double TOL = 2e-4;
    private static final int TEXT = 5;
    private static final float EPS = 1e-5f;

    // ================================================================== gemma4ua

    private static final int G4_FRAME = 640;

    static Map<String, Object> gemma4uaMeta() {
        // the converter writes no encoder blocks: zero heads, FFN width and block count
        Map<String, Object> m = AudioTowerSupport.meta("gemma4ua", G4_FRAME, 8, 0, 0, 0, TEXT, 1e-3f);
        m.remove("clip.audio.num_mel_bins");
        return m;
    }

    static MmprojFixture gemma4uaFixture() {
        MmprojFixture f = new MmprojFixture(41);
        f.weight("mm.a.input_projection.weight", G4_FRAME, TEXT);
        return f;
    }

    static double[][] gemma4uaReference(MmprojFixture f, MelChunk chunk) {
        // eps is fixed at 1e-6 whatever the file says
        double[][] x = rmsNorm(frames(chunk), null, null, 1e-6);
        return dense(x, f.get("mm.a.input_projection.weight"), null);
    }

    @Test
    public void gemma4uaNormalizesEachFrameAndProjects() {
        MmprojFixture f = gemma4uaFixture();
        MelChunk chunk = AudioTowerSupport.chunk(G4_FRAME, 3, 7);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, gemma4uaMeta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(gemma4uaReference(f, chunk), out, TOL);
            assertEquals(3, enc.model().nTokens(3));
            assertEquals(1e-6f, enc.model().hparams().eps(), 0f);
        }
    }

    @Test
    public void gemma4uaFramesTheWaveformInto640SampleTokens() {
        float[] samples = samples(1500, 3);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(gemma4uaFixture(), gemma4uaMeta())) {
            List<MelChunk> chunks = enc.chunks(new AudioClip(samples, 16000));
            assertEquals(1, chunks.size());
            MelChunk c = chunks.get(0);
            assertEquals(G4_FRAME, c.nMel());
            assertEquals(3, c.nLen());
            assertEquals(samples[641], c.value(1, 1), 0f);
            assertEquals(samples[1499], c.value(1499 - 1280, 2), 0f);
            assertEquals(0f, c.value(639, 2), 0f);
            assertEquals(3, enc.nTokens(new AudioClip(samples, 16000)));
        }
    }

    @Test
    public void gemma4uaRefusesAProjectionOfAnotherFrameWidth() {
        MmprojFixture f = new MmprojFixture(41);
        f.weight("mm.a.input_projection.weight", 320, TEXT);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(gemma4uaMeta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("mm.a.input_projection.weight"));
    }

    // ================================================================== muse spark

    private static final int[] MS_PATCH = {4, 2};
    private static final int[] MS_HIDDEN = {6, 8};
    private static final int[] MS_DEPTH = {1, 2};
    private static final int MS_KERNEL = 3;

    static Map<String, Object> museMeta() {
        Map<String, Object> m = AudioTowerSupport.meta("muse-spark-audio", 1, 8, 16, 3, 1, TEXT, EPS);
        m.put("clip.audio.conv_kernel_size", MS_KERNEL);
        m.put("clip.audio.hidden_sizes", MS_HIDDEN.clone());
        m.put("clip.audio.patch_sizes", MS_PATCH.clone());
        m.put("clip.audio.depths", MS_DEPTH.clone());
        return m;
    }

    static MmprojFixture museFixture() {
        MmprojFixture f = new MmprojFixture(43);
        int prev = 1;
        int il = 0;
        for (int s = 0; s < MS_PATCH.length; s++) {
            int c = MS_HIDDEN[s];
            f.weight("a.patch." + s + ".weight", (long) prev * MS_PATCH[s], c);
            f.weight("a.patch." + s + ".bias", c);
            f.put("a.patch." + s + ".norm.weight", 0.8f, 1.2f, c);
            for (int i = 0; i < MS_DEPTH[s]; i++, il++) {
                int ff = 2 * c;
                f.weight("a.conv1d." + il + ".weight", MS_KERNEL, 1, c);
                f.weight("a.conv1d." + il + ".bias", c);
                f.put("a.conv1d." + il + ".norm.weight", 0.8f, 1.2f, c);
                f.weight("a.blk." + il + ".ffn_up.weight", c, ff);
                f.weight("a.blk." + il + ".ffn_up.bias", ff);
                f.weight("a.blk." + il + ".ffn_down.weight", ff, c);
                f.weight("a.blk." + il + ".ffn_down.bias", c);
            }
            prev = c;
        }
        f.put("a.pre_ln.weight", 0.8f, 1.2f, prev);
        f.weight("mm.a.fc.weight", prev, TEXT);
        f.weight("mm.a.fc.bias", TEXT);
        f.put("a.post_ln.weight", 0.8f, 1.2f, TEXT);
        return f;
    }

    /** muse-spark-audio.cpp over one waveform row. */
    static double[][] museReference(MmprojFixture f, float[] wave) {
        double[][] x = new double[wave.length][1];
        for (int i = 0; i < wave.length; i++) {
            x[i][0] = wave[i];
        }
        int il = 0;
        for (int s = 0; s < MS_PATCH.length; s++) {
            x = fold(x, MS_PATCH[s]);
            x = dense(x, f.get("a.patch." + s + ".weight"), f.get("a.patch." + s + ".bias"));
            x = rmsNorm(x, f.get("a.patch." + s + ".norm.weight"), null, EPS);
            for (int i = 0; i < MS_DEPTH[s]; i++, il++) {
                String cv = "a.conv1d." + il + ".";
                String b = "a.blk." + il + ".";
                double[][] conv = depthwise1d(x, f.get(cv + "weight"), MS_KERNEL, MS_KERNEL - 1, 0);
                float[] convB = f.get(cv + "bias");
                for (double[] row : conv) {
                    for (int c = 0; c < row.length; c++) {
                        row[c] += convB[c];
                    }
                }
                double[][] h = rmsNorm(conv, f.get(cv + "norm.weight"), null, EPS);
                h = ffn(h, f.get(b + "ffn_up.weight"), f.get(b + "ffn_up.bias"), AudioTowerSupport::geluErf,
                        f.get(b + "ffn_down.weight"), f.get(b + "ffn_down.bias"));
                x = add(h, x);
            }
        }
        x = rmsNorm(x, f.get("a.pre_ln.weight"), null, EPS);
        x = dense(x, f.get("mm.a.fc.weight"), f.get("mm.a.fc.bias"));
        return rmsNorm(x, f.get("a.post_ln.weight"), null, EPS);
    }

    /** {@code ggml_reshape_2d(x, C * patch, n / patch)}: row {@code t} is rows {@code t * patch ..} concatenated. */
    private static double[][] fold(double[][] x, int patch) {
        int c = x[0].length;
        double[][] y = new double[x.length / patch][c * patch];
        for (int t = 0; t < y.length; t++) {
            for (int j = 0; j < patch; j++) {
                System.arraycopy(x[t * patch + j], 0, y[t], j * c, c);
            }
        }
        return y;
    }

    @Test
    public void museSparkPyramidMatchesReference() {
        MmprojFixture f = museFixture();
        float[] wave = samples(40, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, museMeta())) {
            float[][] out = enc.encode(MelChunk.of(1, 40, wave));
            AudioTowerSupport.assertRows(museReference(f, wave), out, TOL);
            assertEquals(5, out.length);
            assertEquals(TEXT, out[0].length);
            assertEquals(5, enc.model().nTokens(40));
        }
    }

    @Test
    public void museSparkPadsTheWaveformToWholeTokens() {
        float[] samples = samples(37, 9);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(museFixture(), museMeta())) {
            List<MelChunk> chunks = enc.chunks(new AudioClip(samples, 24000));
            assertEquals(1, chunks.size());
            MelChunk c = chunks.get(0);
            assertEquals(1, c.nMel());
            assertEquals(40, c.nLen());
            float[] want = java.util.Arrays.copyOf(samples, 40);
            assertArrayEquals(want, c.data(), 0f);
            // ceil(n / fold) with n the sample count
            assertEquals(5, enc.nTokens(new AudioClip(samples, 24000)));
            assertEquals(5, enc.model().nTokens(37));
            assertEquals(1, enc.model().nTokens(8));
        }
    }

    @Test
    public void museSparkRefusesAChunkThatIsNotWholeTokens() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(museFixture(), museMeta())) {
            assertThrows(IllegalArgumentException.class, () -> enc.encode(MelChunk.of(1, 37, samples(37, 1))));
        }
    }

    @Test
    public void museSparkHparamsFollowTheLoader() {
        Map<String, Object> m = museMeta();
        m.put("clip.audio.depths", new int[]{1, 1});
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(m));
        Map<String, Object> mel = museMeta();
        mel.put("clip.audio.num_mel_bins", 2);
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(mel));
        Map<String, Object> stages = museMeta();
        stages.put("clip.audio.hidden_sizes", new int[]{6});
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(stages));
        Map<String, Object> missing = museMeta();
        missing.remove("clip.audio.conv_kernel_size");
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(missing));
        ClipAudioHparams a = AudioTowerSupport.hparams(museMeta()).audio();
        assertEquals(8, a.samplesPerToken());
        assertEquals(24000, a.sampleRate());
        assertTrue(!a.hasMelFrontend());
    }

    @Test
    public void museSparkRefusesAStageProjectionOfAnotherWidth() {
        MmprojFixture f = museFixture();
        f.weight("a.patch.1.weight", 12, 7);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(museMeta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.patch.1.weight"));
    }

    static float[] samples(int n, long seed) {
        Random r = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextFloat() * 2f - 1f;
        }
        return v;
    }
}
