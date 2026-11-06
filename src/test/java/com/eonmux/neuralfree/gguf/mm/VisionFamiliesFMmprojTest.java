package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The glm5v, lfm3vl, ling3vl, minimax_m3, muse-glimmer, nemotron_v2_vl, nemotron_omni and exaone4_5 families
 * through a real mmproj GGUF file: {@link MmprojModel#open}, the registry row and tensor table of the family, and
 * the CPU encoder of {@link MmprojModel#openVisionEncoder} against the rows llama.cpp's clip encodes from the same
 * file and pixels ({@link VisionFamiliesFFixtures#REFERENCE_RESOURCE}), within {@link #REL_TOL} relative; files
 * missing a required tensor and inputs the graphs cannot run are refused by name. The reference is llama.cpp's CPU
 * clip kept in F32 throughout: stock ggml rounds the patch-conv im2col to F16 and tabulates the tanh GELU in F16,
 * which moves its rows by up to about 1e-3 from the F32 graph.
 */
public class VisionFamiliesFMmprojTest {

    /** Relative tolerance against the llama.cpp rows. */
    static final double REL_TOL = 1e-4;
    /** Magnitude below which an element is compared absolutely, at {@code REL_TOL * REL_FLOOR}. */
    static final double REL_FLOOR = 1e-2;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path write(GgufTestWriter w) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    /** The llama.cpp rows of every case, by case name. */
    static Map<String, float[][]> reference() throws IOException {
        InputStream in = VisionFamiliesFMmprojTest.class.getResourceAsStream(
                VisionFamiliesFFixtures.REFERENCE_RESOURCE);
        assertNotNull(VisionFamiliesFFixtures.REFERENCE_RESOURCE, in);
        Map<String, float[][]> out = new HashMap<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] head = line.split(" ");
                assertEquals("case header: " + line, "case", head[0]);
                float[][] rows = new float[Integer.parseInt(head[2])][];
                int width = Integer.parseInt(head[3]);
                for (int t = 0; t < rows.length; t++) {
                    String[] cols = r.readLine().trim().split(" ");
                    assertEquals(head[1] + " row " + t, width, cols.length);
                    rows[t] = new float[width];
                    for (int c = 0; c < width; c++) {
                        rows[t][c] = Float.parseFloat(cols[c]);
                    }
                }
                out.put(head[1], rows);
            }
        }
        return out;
    }

    /** Asserts every element of {@code got} within {@link #REL_TOL} of {@code want}, relative above the floor. */
    static void assertClose(String name, float[][] want, float[][] got) {
        assertEquals(name + " token count", want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(name + " row width", want[t].length, got[t].length);
            for (int c = 0; c < want[t].length; c++) {
                double w = want[t][c];
                double err = Math.abs(got[t][c] - w);
                if (err > REL_TOL * Math.max(Math.abs(w), REL_FLOOR)) {
                    fail(name + " row " + t + " col " + c + ": want " + w + ", got " + got[t][c]);
                }
            }
        }
    }

    private float[][] encode(VisionFamiliesFFixtures.Case c, MmprojFixture f, Map<String, Object> meta, int width,
                             int height, float[] px) throws IOException {
        Path file = write(VisionFamilyFixtures.gguf(f, meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertTrue(mm.caps().inpVision());
            assertEquals(c.type, mm.visionHparams().projectorType());
            try (VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
                assertEquals(VisionFamiliesFFixtures.OUT, enc.model().outputDim());
                return enc.encode(ClipImage.of(width, height, px));
            }
        }
    }

    private float[][] encode(VisionFamiliesFFixtures.Case c) throws IOException {
        return encode(c, c.fixture, c.meta, c.width, c.height, c.pixels());
    }

    @Test
    public void everyFamilyMatchesLlamaCppClip() throws IOException {
        Map<String, float[][]> ref = reference();
        List<String> names = new ArrayList<>();
        for (VisionFamiliesFFixtures.Case c : VisionFamiliesFFixtures.cases()) {
            float[][] want = ref.get(c.name);
            assertNotNull("reference rows of " + c.name, want);
            assertClose(c.name, want, encode(c));
            names.add(c.name);
        }
        assertEquals(ref.keySet().size(), names.size());
    }

    /**
     * Every family through {@link MediaContext}: a decoded PNG becomes image chunks whose rows are the direct
     * encodes of the family's preprocessed tiles, in order.
     */
    @Test
    public void mediaContextServesEveryFamily() throws IOException {
        byte[] png = MediaContextTest.png(40, 24);
        for (VisionFamiliesFFixtures.Case c : VisionFamiliesFFixtures.cases()) {
            Path file = write(VisionFamilyFixtures.gguf(c.fixture, c.meta));
            List<float[]> rows = new ArrayList<>();
            try (MediaContext ctx = MediaContext.open(file.toString(), VisionFamiliesFFixtures.OUT, false,
                    ImageTokenBudget.none(), t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES)) {
                assertTrue(c.name, ctx.supportsVision());
                MediaPrompt.Builder b = MediaPrompt.builder();
                ctx.appendImage(b, png, 0, text -> new int[text.length()]);
                for (MediaChunk chunk : b.build().chunks()) {
                    if (!chunk.isText()) {
                        rows.addAll(Arrays.asList(chunk.embeddings()));
                    }
                }
            }
            List<float[]> direct = new ArrayList<>();
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
                for (ImageTile tile : ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png))
                        .encodedTiles()) {
                    direct.addAll(Arrays.asList(enc.encode(ClipImage.of(tile.width(), tile.height(),
                            tile.normalized(params.mean(), params.std())))));
                }
            }
            assertFalse(c.name, direct.isEmpty());
            assertEquals(c.name + " image rows", direct.size(), rows.size());
            for (int r = 0; r < rows.size(); r++) {
                assertArrayEquals(c.name + " row " + r, direct.get(r), rows.get(r), 0f);
            }
        }
    }

    @Test
    public void glm5vClampsItsSwiGlu() throws IOException {
        VisionFamiliesFFixtures.Case c = VisionFamiliesFFixtures.byName("glm5v");
        float[][] clamped = encode(c);
        Map<String, Object> unclamped = new HashMap<>(c.meta);
        unclamped.put("clip.vision.swiglu_clamp", 1e6f);
        float[][] open = encode(c, c.fixture, unclamped, c.width, c.height, c.pixels());
        boolean differs = false;
        for (int t = 0; t < clamped.length && !differs; t++) {
            for (int i = 0; i < clamped[t].length; i++) {
                differs |= Math.abs(clamped[t][i] - open[t][i]) > 1e-3f;
            }
        }
        assertTrue("the clamp limit must change the rows", differs);
    }

    // ------------------------------------------------------------------ refusals

    private void missing(String name, String tensor) throws IOException {
        VisionFamiliesFFixtures.Case c = VisionFamiliesFFixtures.byName(name);
        c.fixture.remove(tensor);
        Path file = write(VisionFamilyFixtures.gguf(c.fixture, c.meta));
        Exception e = assertThrows(name + " without " + tensor, Exception.class, () -> {
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertNotNull(enc);
            }
        });
        assertTrue(name + ": " + e, e instanceof MissingTensorException || e instanceof IllegalArgumentException);
        assertTrue(name + ": " + e.getMessage(), e.getMessage().contains(tensor));
    }

    @Test
    public void filesMissingARequiredTensorAreRefusedNamingIt() throws IOException {
        missing("glm5v", "mm.patch_merger.bias");
        missing("lfm3vl", "v.embedding_norm.weight");
        missing("lfm3vl", "mm.input_norm.bias");
        missing("ling3vl", "mm.input_norm.weight");
        missing("ling3vl", "v.patch_embd.weight.1");
        missing("minimax_m3", "mm.merger.fc2.bias");
        missing("minimax_m3", "v.patch_embd.weight.1");
        missing("muse-glimmer", "mm.1.weight");
        missing("muse-glimmer", "v.position_embd.weight");
        missing("nemotron_v2_vl", "v.class_embd");
        missing("nemotron_omni", "mm.model.mlp.3.weight");
        missing("exaone4_5", "v.blk.0.attn_qkv.weight");
        missing("exaone4_5", "v.post_ln.weight");
    }

    private void refusedAtOpen(String name, MmprojFixture f, Map<String, Object> meta, String expected)
            throws IOException {
        Path file = write(VisionFamilyFixtures.gguf(f, meta));
        Exception e = assertThrows(name, Exception.class, () -> {
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertNotNull(enc);
            }
        });
        assertTrue(name + ": " + e.getMessage(), e.getMessage().contains(expected));
    }

    @Test
    public void incompleteOrMalformedFilesAreRefusedByName() throws IOException {
        VisionFamiliesFFixtures.Case lfm = VisionFamiliesFFixtures.byName("lfm3vl");
        Map<String, Object> merge3 = new HashMap<>(lfm.meta);
        merge3.put("clip.vision.spatial_merge_size", 3);
        refusedAtOpen("lfm3vl merge 3", lfm.fixture, merge3, "spatial_merge_size");
        Map<String, Object> noKv = new HashMap<>(lfm.meta);
        noKv.remove("clip.vision.attention.head_count_kv");
        refusedAtOpen("lfm3vl without head_count_kv", lfm.fixture, noKv, "head_count_kv");

        MmprojFixture both = VisionFamiliesFFixtures.lfm3vl(902);
        both.weight("v.blk.0.attn_q.weight", VisionFamilyFixtures.E, VisionFamilyFixtures.E);
        both.weight("v.blk.0.attn_k.weight", VisionFamilyFixtures.E, VisionFamilyFixtures.E / 2);
        both.weight("v.blk.0.attn_v.weight", VisionFamilyFixtures.E, VisionFamilyFixtures.E / 2);
        both.weight("v.blk.0.attn_out.weight", VisionFamilyFixtures.E, VisionFamilyFixtures.E);
        refusedAtOpen("lfm3vl conv and attention block", both, lfm.meta, "lfm3vl block 0");

        VisionFamiliesFFixtures.Case glm = VisionFamiliesFFixtures.byName("glm5v");
        Map<String, Object> noClamp = new HashMap<>(glm.meta);
        noClamp.remove("clip.vision.swiglu_clamp");
        refusedAtOpen("glm5v without the clamp", glm.fixture, noClamp, "clip.vision.swiglu_clamp");

        VisionFamiliesFFixtures.Case exaone = VisionFamiliesFFixtures.byName("exaone4_5");
        MmprojFixture biased = VisionFamiliesFFixtures.exaone45(908);
        biased.put("v.patch_embd.bias", -0.1f, 0.1f, VisionFamilyFixtures.E);
        refusedAtOpen("exaone4_5 with a patch bias", biased, exaone.meta, "v.patch_embd.bias");

        VisionFamiliesFFixtures.Case minimax = VisionFamiliesFFixtures.byName("minimax_m3");
        MmprojFixture classToken = VisionFamiliesFFixtures.minimaxM3(904);
        classToken.weight("v.class_embd", VisionFamiliesFFixtures.MINIMAX_E, 1);
        refusedAtOpen("minimax_m3 with a class token", classToken, minimax.meta, "v.class_embd");

        VisionFamiliesFFixtures.Case v2 = VisionFamiliesFFixtures.byName("nemotron_v2_vl");
        Map<String, Object> noMerge = new HashMap<>(v2.meta);
        noMerge.remove("clip.vision.projector.scale_factor");
        refusedAtOpen("nemotron_v2_vl without merge", v2.fixture, noMerge, "scale_factor");
    }

    private void refusedAtEncode(String name, int width, int height, String expected) throws IOException {
        VisionFamiliesFFixtures.Case c = VisionFamiliesFFixtures.byName(name);
        Path file = write(VisionFamilyFixtures.gguf(c.fixture, c.meta));
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            float[] px = MmprojFixture.pixels(width, height, 5);
            IllegalArgumentException e = assertThrows(name + " " + width + "x" + height,
                    IllegalArgumentException.class, () -> enc.encode(ClipImage.of(width, height, px)));
            assertTrue(name + ": " + e.getMessage(), e.getMessage().contains(expected));
        }
    }

    @Test
    public void imagesTheGraphsCannotRunAreRefusedByName() throws IOException {
        refusedAtEncode("nemotron_v2_vl", 12, 8, "image_size");
        refusedAtEncode("lfm3vl", 6, 8, "lfm3vl");
        refusedAtEncode("minimax_m3", 6, 8, "minimax_m3");
        refusedAtEncode("muse-glimmer", 6, 8, "muse-glimmer");
        refusedAtEncode("exaone4_5", 14 * 3, 14 * 4, "multiple");
    }

    @Test
    public void ling3vlTakesThePatchBiasAsOptionalWhileQwen3vlRequiresIt() throws IOException {
        VisionFamiliesFFixtures.Case c = VisionFamiliesFFixtures.byName("ling3vl");
        MmprojFixture unbiased = VisionFamiliesFFixtures.ling3vl(903);
        unbiased.remove("v.patch_embd.bias");
        float[][] rows = encode(c, unbiased, c.meta, c.width, c.height, c.pixels());
        assertEquals(6, rows.length);

        MmprojFixture qwen = VisionFamiliesCFixtures.qwen3vl(91);
        qwen.remove("v.patch_embd.bias");
        Path file = write(VisionFamilyFixtures.gguf(qwen, VisionFamiliesCFixtures.qwen3vlMeta()));
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("v.patch_embd.bias"));
    }

    @Test
    public void everyCaseIsAFamilyOfThisSet() {
        List<ProjectorType> types = new ArrayList<>();
        for (VisionFamiliesFFixtures.Case c : VisionFamiliesFFixtures.cases()) {
            assertFalse(c.name, types.contains(c.type));
            types.add(c.type);
            assertTrue(c.name, ProjectorRegistry.get(c.type.typeName()).runnable());
        }
        assertEquals(8, types.size());
    }
}
