package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.assertVector;
import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.image;
import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.lin;
import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.tower;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.CONTEXT;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.EPS;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.FF;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.HEADS;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.SIGLIP_PAD;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The SigLIP dual encoder (vLLM's {@code SiglipEmbeddingModel}) against an independent double-precision
 * transcription of HF {@code SiglipModel}'s {@code get_text_features} / {@code get_image_features}: the
 * bidirectional text tower over the input padded to the context, pooled at the last row and passed through the
 * text head; the vision tower without a class token or pre-norm, stretched to the tile, its post-normed rows pooled
 * by the multihead attention pooling head; the sigmoid score of the scaled cosine plus bias; and the refusals by
 * name.
 */
public class SiglipDualEncoderTest {

    private static final double LOGIT_SCALE = 2.3;
    private static final double LOGIT_BIAS = -2;

    // ================================================================== reference

    /** HF {@code SiglipTextTransformer} over {@code ids} padded to the context: the last row through the head. */
    static double[] refText(MmprojFixture f, int[] ids) {
        int[] padded = Arrays.copyOf(ids, CONTEXT);
        Arrays.fill(padded, ids.length, CONTEXT, SIGLIP_PAD);
        double[][] x = new double[CONTEXT][E];
        float[] tok = f.get("t.token_embd.weight");
        float[] pos = f.get("t.position_embd.weight");
        for (int t = 0; t < CONTEXT; t++) {
            for (int i = 0; i < E; i++) {
                x[t][i] = tok[padded[t] * E + i] + pos[t * E + i];
            }
        }
        tower(f, "t", x, false, true);
        MmprojFixture.Reference.layerNorm(x, f.get("t.post_ln.weight"), f.get("t.post_ln.bias"), EPS);
        return lin(new double[][]{x[CONTEXT - 1]}, f, "t.head", E)[0];
    }

    /** HF {@code SiglipVisionTransformer} over an 8x8 normalized tile: patches, blocks, post-norm, pooling head. */
    static double[] refImage(MmprojFixture f, float[] px) {
        ClipHparams hp = ClipHparams.twoTower(GgufKv.ofMap(withArch(ClipDualFixtures.siglipMeta())),
                ProjectorModality.VISION);
        double[][] x = MmprojFixture.Reference.embed(f, hp, px, 8, 8);
        tower(f, "v", x, false, true);
        MmprojFixture.Reference.layerNorm(x, f.get("v.post_ln.weight"), f.get("v.post_ln.bias"), EPS);
        return mapHead(f, x);
    }

    /** {@code SiglipMultiheadAttentionPoolingHead}: the probe attends to every row, then the residual MLP. */
    static double[] mapHead(MmprojFixture f, double[][] rows) {
        int n = rows.length;
        int dh = E / HEADS;
        double[] probe = new double[E];
        for (int i = 0; i < E; i++) {
            probe[i] = f.get("v.head.probe")[i];
        }
        double[][] qkv = new double[n + 1][3 * E];
        double[][] in = new double[n + 1][];
        in[0] = probe;
        System.arraycopy(rows, 0, in, 1, n);
        MmprojFixture.Reference.linear(in, f.get("v.head.attention.in_weight"), f.get("v.head.attention.in_bias"),
                qkv);
        double[] o = new double[E];
        for (int h = 0; h < HEADS; h++) {
            double[] s = new double[n];
            double max = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < dh; i++) {
                    s[t] += qkv[0][h * dh + i] * qkv[1 + t][E + h * dh + i];
                }
                s[t] /= Math.sqrt(dh);
                max = Math.max(max, s[t]);
            }
            double sum = 0;
            for (int t = 0; t < n; t++) {
                s[t] = Math.exp(s[t] - max);
                sum += s[t];
            }
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < dh; i++) {
                    o[h * dh + i] += s[t] / sum * qkv[1 + t][2 * E + h * dh + i];
                }
            }
        }
        double[] attn = lin(new double[][]{o}, f, "v.head.attention.out", E)[0];
        double[][] normed = {attn.clone()};
        MmprojFixture.Reference.layerNorm(normed, f.get("v.head.ln.weight"), f.get("v.head.ln.bias"), EPS);
        double[][] mid = lin(normed, f, "v.head.ffn_down", FF);
        for (int i = 0; i < FF; i++) {
            double v = mid[0][i];
            mid[0][i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
        }
        double[] mlp = lin(mid, f, "v.head.ffn_up", E)[0];
        for (int i = 0; i < E; i++) {
            attn[i] += mlp[i];
        }
        return attn;
    }

    // ================================================================== fixtures

    private static Map<String, Object> withArch(Map<String, Object> meta) {
        Map<String, Object> m = new LinkedHashMap<>(meta);
        m.put("general.architecture", "clip");
        return m;
    }

    private static Path file(MmprojFixture f, Map<String, Object> meta) throws IOException {
        return ClipDualFixtures.writer(f, meta).writeTempFile();
    }

    private static ClipDualEncoder open(MmprojFixture f) throws IOException {
        return ClipDualEncoder.open(file(f, ClipDualFixtures.siglipMeta()).toString(), ModelSourceOptions.defaults());
    }

    private static float[] normalized(RasterImage img) {
        return img.normalized(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});
    }

    // ================================================================== tests

    @Test
    public void textEmbeddingIsTheBidirectionalTowerOverThePaddedInputThroughTheHead() throws IOException {
        MmprojFixture f = ClipDualFixtures.siglipTensors(51);
        try (ClipDualEncoder enc = open(f)) {
            assertEquals(ClipFamily.SIGLIP, enc.family());
            assertEquals(E, enc.embeddingLength());
            int[] ids = {6, 11, 15};
            float[] got = enc.embedText(ids);
            assertVector(refText(f, ids), got);
            int[] full = Arrays.copyOf(ids, CONTEXT);
            Arrays.fill(full, ids.length, CONTEXT, SIGLIP_PAD);
            assertArrayEquals("the input pads to the context", got, enc.embedText(full), 0f);
            int[] changedFirst = {9, 11, 15};
            assertNotEquals("the last row attends to the first token", got[0], enc.embedText(changedFirst)[0],
                    1e-7f);
        }
    }

    @Test
    public void imageEmbeddingIsTheAttentionPoolingHeadOverThePostNormedPatches() throws IOException {
        MmprojFixture f = ClipDualFixtures.siglipTensors(52);
        RasterImage img = image(8, 8, 3);
        try (ClipDualEncoder enc = open(f)) {
            assertEquals(8, enc.imageSize());
            assertVector(refImage(f, normalized(img)), enc.embedImage(img));
        }
    }

    @Test
    public void imagesStretchToTheTileWithoutCropping() throws IOException {
        MmprojFixture f = ClipDualFixtures.siglipTensors(53);
        RasterImage wide = image(16, 8, 4);
        RasterImage squashed = ClipImageTower.squash(wide, 8);
        assertEquals(new ImageSize(8, 8), squashed.size());
        assertTrue("the squash keeps both halves of the wide image",
                !Arrays.equals(squashed.normalized(new float[]{0, 0, 0}, new float[]{1, 1, 1}),
                        ClipImageTower.preprocess(wide, 8).normalized(new float[]{0, 0, 0}, new float[]{1, 1, 1})));
        try (ClipDualEncoder enc = open(f)) {
            assertArrayEquals(enc.embedImage(squashed), enc.embedImage(wide), 0f);
            assertVector(refImage(f, normalized(squashed)), enc.embedImage(wide));
        }
    }

    @Test
    public void scoreIsTheSigmoidOfTheScaledCosinePlusBias() throws IOException {
        try (ClipDualEncoder enc = open(ClipDualFixtures.siglipTensors(54))) {
            for (double cos : new double[]{-1, 0, 0.25, 1}) {
                double want = 1 / (1 + Math.exp(-(cos * Math.exp(LOGIT_SCALE) + LOGIT_BIAS)));
                assertEquals(want, enc.score(cos), 1e-6);
            }
        }
        try (ClipDualEncoder clip = ClipDualEncoder.open(ClipDualFixtures.twoTower().toString(),
                ModelSourceOptions.defaults())) {
            assertEquals(ClipFamily.CLIP, clip.family());
            assertEquals("CLIP scores by the cosine itself", 0.25, clip.score(0.25), 0);
        }
    }

    @Test
    public void malformedSiglipFilesRefuseByName() throws IOException {
        MmprojFixture withClass = ClipDualFixtures.siglipTensors(55);
        withClass.weight("v.class_embd", E);
        Path classFile = file(withClass, ClipDualFixtures.siglipMeta());
        IllegalArgumentException cls = assertThrows(IllegalArgumentException.class,
                () -> ClipDualEncoder.open(classFile.toString(), ModelSourceOptions.defaults()));
        assertTrue(cls.getMessage(), cls.getMessage().contains("v.class_embd"));

        MmprojFixture noBias = ClipDualFixtures.siglipTensors(56);
        noBias.remove("logit_bias");
        Path biasFile = file(noBias, ClipDualFixtures.siglipMeta());
        MissingTensorException bias = assertThrows(MissingTensorException.class,
                () -> ClipDualEncoder.open(biasFile.toString(), ModelSourceOptions.defaults()));
        assertTrue(bias.getMessage(), bias.getMessage().contains("logit_bias"));

        Map<String, Object> unknown = ClipDualFixtures.siglipMeta();
        unknown.put(ClipFamily.KEY, "align");
        Path unknownFile = file(ClipDualFixtures.siglipTensors(57), unknown);
        GgufFormatException family = assertThrows(GgufFormatException.class,
                () -> ClipDualEncoder.open(unknownFile.toString(), ModelSourceOptions.defaults()));
        assertTrue(family.getMessage(), family.getMessage().contains(ClipFamily.KEY));

        MmprojFixture shortProbe = ClipDualFixtures.siglipTensors(58);
        shortProbe.weight("v.head.probe", E - 1);
        Path probeFile = file(shortProbe, ClipDualFixtures.siglipMeta());
        IllegalArgumentException probe = assertThrows(IllegalArgumentException.class,
                () -> ClipDualEncoder.open(probeFile.toString(), ModelSourceOptions.defaults()));
        assertTrue(probe.getMessage(), probe.getMessage().contains("v.head.probe"));
    }
}
