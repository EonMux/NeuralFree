package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The minicpmv, internvl, pixtral, idefics3, llama4 and phi4 families through a real mmproj GGUF file:
 * {@link MmprojModel#open}, the registry row and tensor table of the family, and the encoder of
 * {@link MmprojModel#openCpuVisionEncoder} against the double-precision reference graph.
 */
public class VisionFamiliesMmprojTest {

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
            try (CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
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
    public void minicpmvOpensAndEncodes() throws IOException {
        Map<String, Object> meta = VisionFamilyFixtures.minicpmvMeta();
        openAndEncode(VisionFamilyFixtures.minicpmv(11), meta, ProjectorType.MINICPMV,
                VisionFamilyReference::minicpmv, 12, 8, VisionFamilyFixtures.QUERIES, VisionFamilyFixtures.RESAMPLER_D);
        Path file = write(VisionFamilyFixtures.gguf(VisionFamilyFixtures.minicpmv(11), meta));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertEquals(4, mm.visionHparams().minicpmvVersion());
            assertEquals(VisionFamilyFixtures.QUERIES, mm.visionHparams().minicpmvQueryNum());
        }
    }

    @Test
    public void internvlOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixtures.internvl(12, "per_head"), VisionFamilyFixtures.internvlMeta(),
                ProjectorType.INTERNVL, VisionFamilyReference::internvl, 8, 8, 4, 6);
    }

    @Test
    public void pixtralOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixtures.pixtral(13, true), VisionFamilyFixtures.pixtralMeta(true),
                ProjectorType.PIXTRAL, VisionFamilyReference::pixtral, 8, 12, 2 * 3 + 2, 6);
        openAndEncode(VisionFamilyFixtures.pixtral(14, false), VisionFamilyFixtures.pixtralMeta(false),
                ProjectorType.PIXTRAL, VisionFamilyReference::pixtral, 4, 4, 2 * 2 + 1, 6);
    }

    @Test
    public void idefics3OpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixtures.idefics3(15), VisionFamilyFixtures.idefics3Meta(), ProjectorType.IDEFICS3,
                VisionFamilyReference::idefics3, 16, 16, 4, 6);
    }

    @Test
    public void llama4OpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixtures.llama4(16), VisionFamilyFixtures.llama4Meta(), ProjectorType.LLAMA4,
                VisionFamilyReference::llama4, 8, 8, 4, 6);
    }

    @Test
    public void phi4OpensAndEncodes() throws IOException {
        openAndEncode(VisionFamilyFixtures.phi4(17), VisionFamilyFixtures.phi4Meta(), ProjectorType.PHI4,
                VisionFamilyReference::phi4, 20, 12, 15, 6);
    }

    @Test
    public void missingFamilyTensorsFailAtOpen() throws IOException {
        String[][] cases = {
                {"resampler", "resampler.ln_post.weight"},
                {"internvl", "mm.model.mlp.3.bias"},
                {"pixtral", "v.token_embd.img_break"},
                {"idefics3", "mm.model.fc.weight"},
                {"llama4", "mm.model.mlp.2.weight"},
                {"phi4", "mm.2.bias"},
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
            case "resampler":
                return VisionFamilyFixtures.minicpmv(21);
            case "internvl":
                return VisionFamilyFixtures.internvl(22, null);
            case "pixtral":
                return VisionFamilyFixtures.pixtral(23, true);
            case "idefics3":
                return VisionFamilyFixtures.idefics3(24);
            case "llama4":
                return VisionFamilyFixtures.llama4(25);
            default:
                return VisionFamilyFixtures.phi4(26);
        }
    }

    private static Map<String, Object> meta(String projector) {
        switch (projector) {
            case "resampler":
                return VisionFamilyFixtures.minicpmvMeta();
            case "internvl":
                return VisionFamilyFixtures.internvlMeta();
            case "pixtral":
                return VisionFamilyFixtures.pixtralMeta(true);
            case "idefics3":
                return VisionFamilyFixtures.idefics3Meta();
            case "llama4":
                return VisionFamilyFixtures.llama4Meta();
            default:
                return VisionFamilyFixtures.phi4Meta();
        }
    }
}
