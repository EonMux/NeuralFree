package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixturesB.OUT;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The stacked-feature llava, ldp, ldpv2, glm-edge, cogvlm, glm4v, kimivl, lfm2, d1omni_v, dots_ocr, youtuvl and
 * granite4_vision families through a real mmproj GGUF file: {@link MmprojModel#open}, the registry row and tensor
 * table of the family, and the encoder of {@link MmprojModel#openVisionEncoder} against the double-precision
 * reference graph.
 */
public class VisionFamiliesBMmprojTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The reference graph of one family. */
    private interface Reference {
        float[][] encode(MmprojFixture f, ClipHparams hp, float[] px, int width, int height);
    }

    private Path write(GgufTestWriter w) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    private void openAndEncode(MmprojFixture f, Map<String, Object> meta, ProjectorType type, Reference ref,
                               int width, int height, int tokens, int outWidth) throws IOException {
        Path file = write(VisionFamilyFixtures.gguf(f, meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertTrue(mm.caps().inpVision());
            ClipHparams hp = mm.visionHparams();
            assertEquals(type, hp.projectorType());
            try (VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
                assertEquals(outWidth, enc.model().outputDim());
                float[] px = MmprojFixture.pixels(width, height, 77);
                float[][] got = enc.encode(ClipImage.of(width, height, px));
                assertEquals(tokens, got.length);
                assertEquals(outWidth, got[0].length);
                VisionFamiliesTest.assertRows(ref.encode(f, hp, px, width, height), got);
            }
        }
    }

    @Test
    public void stackedLlavaOpensAndEncodes() throws IOException {
        Map<String, Object> meta = VisionFamilyFixturesB.llavaStackedMeta();
        openAndEncode(VisionFamilyFixturesB.llavaStacked(21), meta, ProjectorType.MLP,
                MmprojFixture.Reference::clip, 16, 16, 16, OUT);
        Path file = write(VisionFamilyFixtures.gguf(VisionFamilyFixturesB.llavaStacked(21), meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertArrayEquals(new int[]{0, 1, 2}, mm.visionHparams().featureLayers());
        }
    }

    @Test
    public void ldpAndLdpv2OpenAndEncode() throws IOException {
        openAndEncode(VisionFamilyFixturesB.ldp(22), VisionFamilyFixturesB.llavaMeta("ldp"), ProjectorType.LDP,
                (f, hp, px, w, h) -> VisionFamilyReferenceB.ldp(f, hp, px, w), 16, 16, 4, OUT);
        openAndEncode(VisionFamilyFixturesB.ldpv2(23), VisionFamilyFixturesB.llavaMeta("ldpv2"),
                ProjectorType.LDPV2, (f, hp, px, w, h) -> VisionFamilyReferenceB.ldpv2(f, hp, px, w), 16, 16, 4,
                OUT);
    }

    @Test
    public void glmEdgeOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixturesB.glmEdge(24), VisionFamilyFixturesB.llavaMeta("adapter"),
                ProjectorType.GLM_EDGE, (f, hp, px, w, h) -> VisionFamilyReferenceB.glmEdge(f, hp, px, w), 16, 16,
                6, OUT);
    }

    @Test
    public void cogvlmOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixturesB.cogvlm(25), VisionFamilyFixturesB.cogvlmMeta(), ProjectorType.COGVLM,
                (f, hp, px, w, h) -> VisionFamilyReferenceB.cogvlm(f, hp, px, w), 8, 8, 6, OUT);
    }

    @Test
    public void glm4vOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixturesB.glm4v(26, false), VisionFamilyFixturesB.glm4vMeta(), ProjectorType.GLM4V,
                VisionFamilyReferenceB::glm4v, 16, 8, 8, OUT);
    }

    @Test
    public void kimivlLfm2D1omniAndDotsOcrOpenAndEncode() throws IOException {
        openAndEncode(VisionFamilyFixturesB.kimivl(27), VisionFamilyFixturesB.kimivlMeta(), ProjectorType.KIMIVL,
                VisionFamilyReferenceB::kimivl, 12, 8, 6, OUT);
        openAndEncode(VisionFamilyFixturesB.lfm2(28, false), VisionFamilyFixturesB.lfm2Meta("lfm2"),
                ProjectorType.LFM2, (f, hp, px, w, h) -> VisionFamilyReferenceB.lfm2(f, hp, px, w, h, false), 12, 8,
                6, OUT);
        openAndEncode(VisionFamilyFixturesB.lfm2(29, true), VisionFamilyFixturesB.lfm2Meta("d1omni_v"),
                ProjectorType.D1OMNI_V, (f, hp, px, w, h) -> VisionFamilyReferenceB.lfm2(f, hp, px, w, h, true), 8,
                8, 4, OUT);
        openAndEncode(VisionFamilyFixturesB.dotsOcr(30), VisionFamilyFixturesB.dotsOcrMeta(),
                ProjectorType.DOTS_OCR, VisionFamilyReferenceB::dotsOcr, 8, 12, 6, OUT);
    }

    @Test
    public void youtuvlOpensAndEncodesWithItsWindowLayers() throws IOException {
        Map<String, Object> meta = VisionFamilyFixturesB.youtuvlMeta(1);
        openAndEncode(VisionFamilyFixturesB.youtuvl(31), meta, ProjectorType.YOUTUVL,
                (f, hp, px, w, h) -> VisionFamilyReferenceB.youtuvl(f, hp, px, w, h, new int[]{1}), 16, 8, 8, OUT);
        Path file = write(VisionFamilyFixtures.gguf(VisionFamilyFixturesB.youtuvl(31), meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertArrayEquals(new int[]{1}, mm.visionHparams().visionFamily().fullAttnLayers());
            assertEquals(8, mm.visionHparams().attnWindowSize());
        }
    }

    @Test
    public void granite4OpensAndEncodesOneTile() throws IOException {
        Map<String, Object> meta = VisionFamilyFixturesB.granite4Meta();
        openAndEncode(VisionFamilyFixturesB.granite4(32), meta, ProjectorType.GRANITE4_VISION,
                (f, hp, px, w, h) -> MmprojFixture.Reference.toFloat(VisionFamilyReferenceB.granite4Tile(f, hp, px,
                        w)), 16, 16, 16, 2 * OUT);
        Path file = write(VisionFamilyFixtures.gguf(VisionFamilyFixturesB.granite4(32), meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            VisionFamilyHparams fam = mm.visionHparams().visionFamily();
            assertArrayEquals(new int[]{3, -1}, fam.spatialOffsets());
            assertEquals(VisionFamilyFixturesB.GRANITE_QUERY, fam.querySide());
            assertEquals(VisionFamilyFixturesB.GRANITE_WINDOW, fam.windowSide());
        }
    }

    @Test
    public void missingFamilyTensorsFailAtOpen() throws IOException {
        String[][] cases = {
                {"ldp", "mm.model.mb_block.2.block.2.1.bias"},
                {"ldpv2", "mm.model.peg.0.bias"},
                {"adapter", "adapter.eoi"},
                {"cogvlm", "mm.post_fc_norm.bias"},
                {"glm4v", "mm.patch_merger.bias"},
                {"kimivl", "mm.input_norm.bias"},
                {"lfm2", "mm.2.bias"},
                {"dots_ocr", "mm.post_norm.weight"},
                {"youtuvl", "v.blk.1.attn_k.bias"},
                {"granite4_vision", "v.proj_blk.0.ffn_norm.bias"},
        };
        for (String[] c : cases) {
            MmprojFixture f = fixture(c[0]);
            f.remove(c[1]);
            Path file = write(VisionFamilyFixtures.gguf(f, meta(c[0])));
            MissingTensorException e = assertThrows(c[0], MissingTensorException.class,
                    () -> MmprojModel.open(file.toString()));
            assertTrue(e.getMessage(), e.getMessage().contains(c[1]));
        }
    }

    private static MmprojFixture fixture(String projector) {
        switch (projector) {
            case "ldp":
                return VisionFamilyFixturesB.ldp(41);
            case "ldpv2":
                return VisionFamilyFixturesB.ldpv2(42);
            case "adapter":
                return VisionFamilyFixturesB.glmEdge(43);
            case "cogvlm":
                return VisionFamilyFixturesB.cogvlm(44);
            case "glm4v":
                return VisionFamilyFixturesB.glm4v(45, false);
            case "kimivl":
                return VisionFamilyFixturesB.kimivl(46);
            case "lfm2":
                return VisionFamilyFixturesB.lfm2(47, false);
            case "dots_ocr":
                return VisionFamilyFixturesB.dotsOcr(48);
            case "youtuvl":
                return VisionFamilyFixturesB.youtuvl(49);
            default:
                return VisionFamilyFixturesB.granite4(50);
        }
    }

    private static Map<String, Object> meta(String projector) {
        switch (projector) {
            case "ldp":
            case "ldpv2":
            case "adapter":
                return VisionFamilyFixturesB.llavaMeta(projector);
            case "cogvlm":
                return VisionFamilyFixturesB.cogvlmMeta();
            case "glm4v":
                return VisionFamilyFixturesB.glm4vMeta();
            case "kimivl":
                return VisionFamilyFixturesB.kimivlMeta();
            case "lfm2":
                return VisionFamilyFixturesB.lfm2Meta("lfm2");
            case "dots_ocr":
                return VisionFamilyFixturesB.dotsOcrMeta();
            case "youtuvl":
                return VisionFamilyFixturesB.youtuvlMeta(1);
            default:
                return VisionFamilyFixturesB.granite4Meta();
        }
    }
}
