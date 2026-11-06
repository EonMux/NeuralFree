package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.image.ImageOps;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.EPS;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.FF;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.HEADS;
import static com.eonmux.neuralfree.gguf.mm.ClipDualFixtures.LAYERS;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CLIP dual encoder against an independent double-precision transcription of HF {@code CLIPModel}'s
 * {@code get_text_features} / {@code get_image_features} (vLLM's {@code CLIPEmbeddingModel}): the causal pre-norm
 * text tower pooled at the last token and projected, the vision tower with the class token first, pooled at the
 * class row (or averaged under MEAN pooling) after the post-norm and projected; the CLIP tokenizer through the
 * tokenizer source; and the refusals by name.
 */
public class ClipDualEncoderTest {

    private static final float TOL = 2e-4f;

    // ================================================================== reference

    /** HF {@code CLIPTextTransformer} + {@code text_projection} over {@code ids}, pooled by {@code pooling}. */
    static double[] refText(MmprojFixture f, int[] ids, PoolingKind pooling) {
        int n = ids.length;
        double[][] x = new double[n][E];
        float[] tok = f.get("t.token_embd.weight");
        float[] pos = f.get("t.position_embd.weight");
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < E; i++) {
                x[t][i] = tok[ids[t] * E + i] + pos[t * E + i];
            }
        }
        tower(f, "t", x, true, false);
        MmprojFixture.Reference.layerNorm(x, f.get("t.post_ln.weight"), f.get("t.post_ln.bias"), EPS);
        return project(pool(x, pooling), f.get("text_projection.weight"));
    }

    /** HF {@code CLIPVisionTransformer} + {@code visual_projection} over an {@code IMAGE}-square normalized tile. */
    static double[] refImage(MmprojFixture f, float[] px, PoolingKind pooling) {
        int side = 2;
        int ps = 4;
        double[][] x = new double[side * side + 1][E];
        float[] kernel = f.get("v.patch_embd.weight");
        for (int e = 0; e < E; e++) {
            x[0][e] = f.get("v.class_embd")[e];
        }
        for (int py = 0; py < side; py++) {
            for (int pxi = 0; pxi < side; pxi++) {
                double[] row = x[1 + py * side + pxi];
                for (int e = 0; e < E; e++) {
                    double s = 0;
                    for (int c = 0; c < 3; c++) {
                        for (int ky = 0; ky < ps; ky++) {
                            for (int kx = 0; kx < ps; kx++) {
                                s += kernel[((e * 3 + c) * ps + ky) * ps + kx]
                                        * MmprojFixture.Reference.pixel(px, 8, 8, pxi * ps + kx, py * ps + ky, c);
                            }
                        }
                    }
                    row[e] = s;
                }
            }
        }
        float[] pos = f.get("v.position_embd.weight");
        for (int t = 0; t < x.length; t++) {
            for (int i = 0; i < E; i++) {
                x[t][i] += pos[t * E + i];
            }
        }
        MmprojFixture.Reference.layerNorm(x, f.get("v.pre_ln.weight"), f.get("v.pre_ln.bias"), EPS);
        tower(f, "v", x, false, false);
        MmprojFixture.Reference.layerNorm(x, f.get("v.post_ln.weight"), f.get("v.post_ln.bias"), EPS);
        double[] pooled = pooling == PoolingKind.MEAN ? pool(x, PoolingKind.MEAN) : x[0];
        return project(pooled, f.get("visual_projection.weight"));
    }

    /**
     * The CLIP encoder layers in place: pre-norm attention (causal for the CLIP text tower) and the quick-GELU MLP,
     * or the tanh-GELU MLP of SigLIP when {@code geluTanh}.
     */
    static void tower(MmprojFixture f, String prefix, double[][] x, boolean causal, boolean geluTanh) {
        int n = x.length;
        int dh = E / HEADS;
        for (int il = 0; il < LAYERS; il++) {
            String p = prefix + ".blk." + il + ".";
            double[][] h = new double[n][E];
            MmprojFixture.Reference.layerNorm(h, x, f.get(p + "ln1.weight"), f.get(p + "ln1.bias"), EPS);
            double[][] q = lin(h, f, p + "attn_q", E);
            double[][] k = lin(h, f, p + "attn_k", E);
            double[][] v = lin(h, f, p + "attn_v", E);
            double[][] o = new double[n][E];
            for (int head = 0; head < HEADS; head++) {
                for (int t = 0; t < n; t++) {
                    int end = causal ? t + 1 : n;
                    double[] s = new double[end];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int u = 0; u < end; u++) {
                        double d = 0;
                        for (int i = 0; i < dh; i++) {
                            d += q[t][head * dh + i] * k[u][head * dh + i];
                        }
                        s[u] = d / Math.sqrt(dh);
                        max = Math.max(max, s[u]);
                    }
                    double sum = 0;
                    for (int u = 0; u < end; u++) {
                        s[u] = Math.exp(s[u] - max);
                        sum += s[u];
                    }
                    for (int u = 0; u < end; u++) {
                        for (int i = 0; i < dh; i++) {
                            o[t][head * dh + i] += s[u] / sum * v[u][head * dh + i];
                        }
                    }
                }
            }
            add(x, lin(o, f, p + "attn_out", E));
            MmprojFixture.Reference.layerNorm(h, x, f.get(p + "ln2.weight"), f.get(p + "ln2.bias"), EPS);
            double[][] mid = lin(h, f, p + "ffn_down", FF);
            for (double[] row : mid) {
                for (int i = 0; i < FF; i++) {
                    double a = row[i];
                    row[i] = geluTanh ? 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a)))
                            : a / (1 + Math.exp(-1.702 * a));
                }
            }
            add(x, lin(mid, f, p + "ffn_up", E));
        }
    }

    static double[][] lin(double[][] x, MmprojFixture f, String name, int out) {
        double[][] y = new double[x.length][out];
        MmprojFixture.Reference.linear(x, f.get(name + ".weight"), f.get(name + ".bias"), y);
        return y;
    }

    static void add(double[][] x, double[][] d) {
        for (int t = 0; t < x.length; t++) {
            for (int i = 0; i < x[t].length; i++) {
                x[t][i] += d[t][i];
            }
        }
    }

    private static double[] pool(double[][] x, PoolingKind pooling) {
        switch (pooling) {
            case LAST:
                return x[x.length - 1];
            case CLS:
                return x[0];
            default:
                double[] mean = new double[x[0].length];
                for (double[] row : x) {
                    for (int i = 0; i < row.length; i++) {
                        mean[i] += row[i] / x.length;
                    }
                }
                return mean;
        }
    }

    private static double[] project(double[] v, float[] w) {
        double[][] y = new double[1][w.length / v.length];
        MmprojFixture.Reference.linear(new double[][]{v}, w, null, y);
        return y[0];
    }

    static void assertVector(double[] want, float[] got) {
        assertEquals(want.length, got.length);
        for (int i = 0; i < want.length; i++) {
            assertEquals("component " + i, want[i], got[i], TOL);
        }
    }

    // ================================================================== fixtures

    private static Path file(MmprojFixture f, Map<String, Object> meta) throws IOException {
        return ClipDualFixtures.writer(f, meta).writeTempFile();
    }

    static RasterImage image(int width, int height, long seed) {
        byte[] rgb = new byte[width * height * 3];
        new Random(seed).nextBytes(rgb);
        return new RasterImage(width, height, rgb);
    }

    // ================================================================== tests

    @Test
    public void textEmbeddingMatchesTheCausalTowerPooledAtTheLastToken() throws IOException {
        MmprojFixture f = ClipDualFixtures.tensors(41, true, true);
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, ClipDualFixtures.meta(true, true)).toString(),
                ModelSourceOptions.defaults())) {
            assertEquals(PoolingKind.LAST, enc.pooling());
            assertEquals(ClipDualFixtures.DIM, enc.embeddingLength());
            assertEquals(ClipDualFixtures.CONTEXT, enc.contextLength());
            int[] ids = {14, 6, 11, 13, 15};
            assertVector(refText(f, ids, PoolingKind.LAST), enc.embedText(ids));
            int[] prefix = {14, 6, 11};
            assertVector(refText(f, prefix, PoolingKind.LAST), enc.embedText(prefix));
        }
    }

    @Test
    public void causalMaskKeepsLaterTokensOutOfEarlierRows() throws IOException {
        MmprojFixture f = ClipDualFixtures.tensors(42, true, false);
        Map<String, Object> meta = ClipDualFixtures.meta(true, false);
        meta.put(ClipDualEncoder.POOLING_KEY, PoolingKind.CLS.ggufCode());
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, meta).toString(), ModelSourceOptions.defaults())) {
            assertEquals(PoolingKind.CLS, enc.pooling());
            float[] one = enc.embedText(new int[]{14, 3});
            float[] two = enc.embedText(new int[]{14, 9, 1, 2});
            assertArrayEquals("the first row attends to itself only", one, two, 1e-6f);
            assertVector(refText(f, new int[]{14, 9, 1, 2}, PoolingKind.CLS), two);
        }
    }

    @Test
    public void meanPoolingAveragesTextAndImageRows() throws IOException {
        MmprojFixture f = ClipDualFixtures.tensors(43, true, true);
        Map<String, Object> meta = ClipDualFixtures.meta(true, true);
        meta.put(ClipDualEncoder.POOLING_KEY, PoolingKind.MEAN.ggufCode());
        RasterImage img = image(8, 8, 5);
        float[] px = img.normalized(new float[]{0.48f, 0.46f, 0.41f}, new float[]{0.27f, 0.26f, 0.28f});
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, meta).toString(), ModelSourceOptions.defaults())) {
            int[] ids = {14, 0, 4, 15};
            assertVector(refText(f, ids, PoolingKind.MEAN), enc.embedText(ids));
            assertVector(refImage(f, px, PoolingKind.MEAN), enc.embedImage(img));
        }
    }

    @Test
    public void imageEmbeddingMatchesTheClassRowAfterThePostNorm() throws IOException {
        MmprojFixture f = ClipDualFixtures.tensors(44, true, true);
        RasterImage img = image(8, 8, 7);
        float[] px = img.normalized(new float[]{0.48f, 0.46f, 0.41f}, new float[]{0.27f, 0.26f, 0.28f});
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, ClipDualFixtures.meta(true, true)).toString(),
                ModelSourceOptions.defaults())) {
            assertTrue(enc.hasText() && enc.hasVision());
            assertEquals(8, enc.imageSize());
            assertVector(refImage(f, px, PoolingKind.LAST), enc.embedImage(img));
        }
    }

    @Test
    public void largerImagesResizeTheShortestEdgeAndCenterCrop() {
        RasterImage wide = image(16, 8, 9);
        RasterImage tile = ClipImageTower.preprocess(wide, 8);
        assertEquals(ImageOps.crop(wide, 4, 0, 8, 8), tile);
        RasterImage tall = image(10, 20, 10);
        RasterImage cropped = ClipImageTower.preprocess(tall, 8);
        assertEquals(8, cropped.width());
        assertEquals(8, cropped.height());
        RasterImage resized = ImageOps.resize(tall, new com.eonmux.neuralfree.gguf.image.ImageSize(8, 16),
                com.eonmux.neuralfree.gguf.image.ResizeAlgo.BICUBIC, com.eonmux.neuralfree.gguf.image.PadStyle.NONE,
                new int[]{0, 0, 0});
        assertEquals(ImageOps.crop(resized, 0, 4, 8, 8), cropped);
    }

    @Test
    public void clipTokenizerComesFromTheTokenizerSource() throws IOException {
        MmprojFixture f = ClipDualFixtures.tensors(45, true, true);
        ModelSourceOptions options = ModelSourceOptions.builder().tokenizerDir(ClipDualFixtures.tokenizerDir()).build();
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, ClipDualFixtures.meta(true, true)).toString(),
                options)) {
            assertTrue(enc.hasTokenizer());
            int[] ids = enc.tokenizer().encode("A cat  DOG", true, true);
            assertArrayEquals(ClipDualFixtures.catDogIds(), ids);
            assertVector(refText(f, ids, PoolingKind.LAST), enc.embedText(ids));
        }
    }

    @Test
    public void legacyTokenListAloneNamesTheTokenizerSource() throws IOException {
        try (ClipDualEncoder enc = ClipDualEncoder.open(ClipDualFixtures.twoTower().toString(),
                ModelSourceOptions.defaults())) {
            assertFalse(enc.hasTokenizer());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, enc::tokenizer);
            assertTrue(e.getMessage(), e.getMessage().contains(ModelSourceOptions.PROP_TOKENIZER));
        }
    }

    @Test
    public void missingTowerRefusesByName() throws IOException {
        try (ClipDualEncoder enc = ClipDualEncoder.open(ClipDualFixtures.visionOnly().toString(),
                ModelSourceOptions.defaults())) {
            assertFalse(enc.hasText());
            assertEquals(0, enc.contextLength());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> enc.embedText(new int[]{1}));
            assertTrue(e.getMessage(), e.getMessage().contains("clip.has_text_encoder"));
            assertEquals(ClipDualFixtures.DIM, enc.embedImage(image(8, 8, 1)).length);
        }
        MmprojFixture f = ClipDualFixtures.tensors(46, true, false);
        try (ClipDualEncoder enc = ClipDualEncoder.open(file(f, ClipDualFixtures.meta(true, false)).toString(),
                ModelSourceOptions.defaults())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> enc.embedImage(image(8, 8, 1)));
            assertTrue(e.getMessage(), e.getMessage().contains("clip.has_vision_encoder"));
        }
    }

    @Test
    public void textInputsOutsideTheTowerRefuse() throws IOException {
        try (ClipDualEncoder enc = ClipDualEncoder.open(ClipDualFixtures.twoTower().toString(),
                ModelSourceOptions.defaults())) {
            IllegalArgumentException tooLong = assertThrows(IllegalArgumentException.class,
                    () -> enc.embedText(new int[ClipDualFixtures.CONTEXT + 1]));
            assertTrue(tooLong.getMessage(), tooLong.getMessage().contains("clip.text.context_length"));
            IllegalArgumentException oov = assertThrows(IllegalArgumentException.class,
                    () -> enc.embedText(new int[]{14, 16}));
            assertTrue(oov.getMessage(), oov.getMessage().contains("token id 16"));
            assertThrows(IllegalArgumentException.class, () -> enc.embedText(new int[0]));
        }
    }

    @Test
    public void unsupportedPoolingAndMismatchedTowersRefuseAtOpen() throws IOException {
        Map<String, Object> rank = ClipDualFixtures.meta(true, true);
        rank.put(ClipDualEncoder.POOLING_KEY, PoolingKind.RANK.ggufCode());
        Path rankFile = file(ClipDualFixtures.tensors(47, true, true), rank);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ClipDualEncoder.open(rankFile.toString(), ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(ClipDualEncoder.POOLING_KEY));

        MmprojFixture f = ClipDualFixtures.tensors(48, true, true);
        f.weight("visual_projection.weight", E, ClipDualFixtures.DIM + 1);
        Map<String, Object> meta = ClipDualFixtures.meta(true, true);
        meta.put("clip.vision.projection_dim", ClipDualFixtures.DIM + 1);
        Path mismatched = file(f, meta);
        IllegalArgumentException m = assertThrows(IllegalArgumentException.class,
                () -> ClipDualEncoder.open(mismatched.toString(), ModelSourceOptions.defaults()));
        assertTrue(m.getMessage(), m.getMessage().contains("one embedding space"));
    }

    @Test
    public void twoTowerFilesAreTellApartFromMmprojFiles() throws IOException {
        Path twoTower = ClipDualFixtures.twoTower();
        assertTrue(ClipDualEncoder.isTwoTower(twoTower.toString()));
        Path llava = MmprojTestFiles.llava(8);
        assertFalse(ClipDualEncoder.isTwoTower(llava.toString()));
        assertFalse(ClipDualEncoder.isTwoTower((String) null));
        IllegalArgumentException asMmproj = assertThrows(IllegalArgumentException.class,
                () -> MmprojModel.open(twoTower.toString()));
        assertTrue(asMmproj.getMessage(), asMmproj.getMessage().contains("ClipDualEncoder"));
        IllegalArgumentException asDual = assertThrows(IllegalArgumentException.class,
                () -> ClipDualEncoder.open(llava.toString(), ModelSourceOptions.defaults()));
        assertTrue(asDual.getMessage(), asDual.getMessage().contains("not a CLIP two-tower file"));
    }

    @Test
    public void twoTowerHparamsReadTheTowerKeys() {
        GgufKv kv = GgufKv.ofMap(withArch(ClipDualFixtures.meta(true, true)));
        ClipHparams text = ClipHparams.twoTower(kv, ProjectorModality.TEXT);
        assertTrue(text.twoTower());
        assertEquals(ProjectorModality.TEXT, text.modality());
        assertEquals(E, text.nEmbd());
        assertEquals(ClipDualFixtures.DIM, text.projectionDim());
        assertEquals(ClipFfnOp.GELU_QUICK, text.ffnOp());
        ClipHparams vision = ClipHparams.twoTower(kv, ProjectorModality.VISION);
        assertEquals(8, vision.imageSize());
        assertEquals(4, vision.patchSize());
        assertThrows(IllegalArgumentException.class, () -> ClipHparams.twoTower(kv, ProjectorModality.AUDIO));
        GgufKv visionOnly = GgufKv.ofMap(withArch(ClipDualFixtures.meta(false, true)));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipHparams.twoTower(visionOnly, ProjectorModality.TEXT));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.has_text_encoder"));
    }

    @Test
    public void featureSelectFollowsVllmPoolingStrategies() {
        assertEquals(ImageFeatureSelect.CLASS, ImageFeatureSelect.of(PoolingKind.CLS));
        assertEquals(ImageFeatureSelect.CLASS, ImageFeatureSelect.of(PoolingKind.LAST));
        assertEquals(ImageFeatureSelect.FULL, ImageFeatureSelect.of(PoolingKind.MEAN));
        assertThrows(IllegalArgumentException.class, () -> ImageFeatureSelect.of(PoolingKind.NONE));
        assertThrows(IllegalArgumentException.class, () -> ImageFeatureSelect.of(PoolingKind.RANK));
        float[] rows = {1, 2, 3, 4, 5, 6};
        assertArrayEquals(new float[]{1, 2}, ImageFeatureSelect.CLASS.select(rows, 3, 2), 0f);
        assertArrayEquals(new float[]{3, 4}, ImageFeatureSelect.FULL.select(rows, 3, 2), 1e-6f);
    }

    private static Map<String, Object> withArch(Map<String, Object> meta) {
        meta.put("general.architecture", "clip");
        return meta;
    }
}
