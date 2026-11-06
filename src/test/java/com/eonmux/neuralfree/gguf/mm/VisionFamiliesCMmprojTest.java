package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesCFixtures.OUT;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.LAYERS;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The qwen3vl, paddleocr, kimivl, dots_ocr, dots3note_v, hunyuanvl, janus_pro, lfm2, d1omni_v, cohere2v and
 * lightonocr families through a real mmproj GGUF file: {@link MmprojModel#open}, the registry row and tensor table
 * of the family, and the CPU encoder of {@link MmprojModel#openVisionEncoder} against the double-precision
 * reference graph; a file missing a tensor the family's table requires is refused at open, naming it.
 */
public class VisionFamiliesCMmprojTest {

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
                float[] px = MmprojFixture.pixels(width, height, 79);
                float[][] got = enc.encode(ClipImage.of(width, height, px));
                assertEquals(tokens, got.length);
                assertEquals(outWidth, got[0].length);
                VisionFamiliesTest.assertRows(ref.encode(f, hp, px, width, height), got);
            }
        }
    }

    @Test
    public void qwen3vlOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.qwen3vl(91), VisionFamiliesCFixtures.qwen3vlMeta(),
                ProjectorType.QWEN3VL, VisionFamiliesCReference::qwen3vl, 12, 8, 6, OUT * (1 + LAYERS));
    }

    @Test
    public void paddleocrAndKimivlOpenAndEncode() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.paddleocr(92), VisionFamiliesCFixtures.paddleocrMeta(),
                ProjectorType.PADDLEOCR, VisionFamiliesCReference::paddleocr, 8, 12, 6, OUT);
        openAndEncode(VisionFamiliesCFixtures.kimivl(93), VisionFamiliesCFixtures.kimivlMeta(),
                ProjectorType.KIMIVL, VisionFamiliesCReference::kimivl, 10, 8, 6, OUT);
    }

    @Test
    public void dotsFamiliesOpenAndEncode() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.dotsOcr(94), VisionFamiliesCFixtures.dotsOcrMeta(),
                ProjectorType.DOTS_OCR, VisionFamiliesCReference::dots, 8, 8, 4, OUT);
        openAndEncode(VisionFamiliesCFixtures.dots3note(95), VisionFamiliesCFixtures.dots3noteMeta(),
                ProjectorType.DOTS3NOTE_V, VisionFamiliesCReference::dots, 12, 8, 6, OUT);
    }

    @Test
    public void hunyuanvlOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.hunyuanvl(96), VisionFamiliesCFixtures.hunyuanvlMeta(),
                ProjectorType.HUNYUANVL, VisionFamiliesCReference::hunyuanvl, 8, 8, (2 + 1) * 2 + 2, OUT);
    }

    @Test
    public void siglipProjectorFamiliesOpenAndEncode() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.janus(97), VisionFamiliesCFixtures.janusMeta(),
                ProjectorType.JANUS_PRO, VisionFamiliesCReference::janus, 8, 8, 16, OUT);
        openAndEncode(VisionFamiliesCFixtures.lfm2(98, true), VisionFamiliesCFixtures.lfm2Meta("lfm2"),
                ProjectorType.LFM2, VisionFamiliesCReference::lfm2, 8, 12, 6, OUT);
        openAndEncode(VisionFamiliesCFixtures.lfm2(99, false), VisionFamiliesCFixtures.lfm2Meta("d1omni_v"),
                ProjectorType.D1OMNI_V, VisionFamiliesCReference::lfm2, 8, 8, 4, OUT);
        openAndEncode(VisionFamiliesCFixtures.cohere2v(100), VisionFamiliesCFixtures.cohere2vMeta(),
                ProjectorType.COHERE2V, VisionFamiliesCReference::cohere2v, 8, 8, 4, OUT);
    }

    @Test
    public void lightonocrOpensAndEncodes() throws IOException {
        openAndEncode(VisionFamiliesCFixtures.lightonocr(101), VisionFamiliesCFixtures.lightonocrMeta(),
                ProjectorType.LIGHTONOCR, VisionFamilyReference::pixtral, 12, 8, 6, OUT);
    }

    @Test
    public void missingFamilyTensorsFailAtOpen() throws IOException {
        check(VisionFamiliesCFixtures::qwen3vlMeta, () -> VisionFamiliesCFixtures.qwen3vl(111), "v.patch_embd.bias");
        check(VisionFamiliesCFixtures::paddleocrMeta, () -> VisionFamiliesCFixtures.paddleocr(112),
                "mm.input_norm.bias");
        check(VisionFamiliesCFixtures::kimivlMeta, () -> VisionFamiliesCFixtures.kimivl(113),
                "v.position_embd.weight");
        check(VisionFamiliesCFixtures::dotsOcrMeta, () -> VisionFamiliesCFixtures.dotsOcr(114),
                "mm.post_norm.weight");
        check(VisionFamiliesCFixtures::dots3noteMeta, () -> VisionFamiliesCFixtures.dots3note(115),
                "v.blk.1.ffn_up_exps.weight");
        check(VisionFamiliesCFixtures::hunyuanvlMeta, () -> VisionFamiliesCFixtures.hunyuanvl(116),
                "v.image_newline");
        check(VisionFamiliesCFixtures::janusMeta, () -> VisionFamiliesCFixtures.janus(117), "mm.1.bias");
        check(() -> VisionFamiliesCFixtures.lfm2Meta("lfm2"), () -> VisionFamiliesCFixtures.lfm2(118, true),
                "mm.2.weight");
        check(VisionFamiliesCFixtures::cohere2vMeta, () -> VisionFamiliesCFixtures.cohere2v(119), "mm.2.bias");
        check(VisionFamiliesCFixtures::lightonocrMeta, () -> VisionFamiliesCFixtures.lightonocr(120),
                "mm.1.weight");
    }

    @Test
    public void aPartialDeepstackBlockIsRefusedWhenTheEncoderBinds() throws IOException {
        // the deepstack tensors are optional per block, so the table accepts the file; resolving the merger
        // needs every tensor of a block that carries any
        MmprojFixture f = VisionFamiliesCFixtures.qwen3vl(121);
        f.remove("v.deepstack.1.fc2.weight");
        Path file = write(VisionFamilyFixtures.gguf(f, VisionFamiliesCFixtures.qwen3vlMeta()));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            MissingTensorException cpu = assertThrows(MissingTensorException.class, mm::openCpuVisionEncoder);
            assertTrue(cpu.getMessage(), cpu.getMessage().contains("v.deepstack.1.fc2.weight"));
            MissingTensorException offload = assertThrows(MissingTensorException.class,
                    () -> mm.openVisionEncoder(MmprojOptions.defaults()));
            assertTrue(offload.getMessage(), offload.getMessage().contains("v.deepstack.1.fc2.weight"));
        }
    }

    private void check(Supplier<Map<String, Object>> meta, Supplier<MmprojFixture> fixture, String tensor)
            throws IOException {
        MmprojFixture f = fixture.get();
        f.remove(tensor);
        Path file = write(VisionFamilyFixtures.gguf(f, meta.get()));
        MissingTensorException e = assertThrows(tensor, MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains(tensor));
    }
}
