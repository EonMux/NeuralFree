package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.image;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Hugging Face {@code CLIPModel} and {@code SiglipModel} checkpoints through the Hugging Face loader, against the
 * two-tower GGUF files llama.cpp's legacy {@code convert_image_encoder_to_gguf.py} layout gives the same weights:
 * the converter's tensor names (the {@code position_ids} buffers and CLIP's {@code logit_scale} skipped), the
 * {@code clip.*} hyperparameters with the deeper vision tower kept whole, the processor's mean and std, the CLIP
 * tokenizer from the checkpoint, SigLIP's family and pad token, and the same embeddings and scores.
 */
public class ClipDualHfTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder().toPath()).outType(ModelSourceOptions.OutType.F32)
                .build();
    }

    /** The CLIP fixture with a third vision block, so the vision tower is deeper than the text tower. */
    private static MmprojFixture clipTensors() {
        MmprojFixture f = ClipDualFixtures.tensors(61, true, true);
        ClipDualFixtures.block(f, "v", 2);
        return f;
    }

    private static Set<String> names(TensorCatalog tc) {
        return new HashSet<>(tc.names());
    }

    @Test
    public void clipModelCheckpointEmbedsAsItsTwoTowerFile() throws IOException {
        MmprojFixture f = clipTensors();
        Map<String, Object> meta = ClipDualFixtures.meta(true, true);
        meta.put("clip.vision.block_count", 3);
        Path gguf = ClipDualFixtures.writer(f, meta).writeTempFile();
        Path dir = ClipHfFixtures.write(f, ClipHfFixtures.config(false, 3), false, null, tmp.newFolder("clip").toPath());
        ModelSourceOptions o = options();
        assertTrue(ClipDualEncoder.isTwoTower(dir.toString()));
        try (ModelSource src = ModelSources.open(dir.toString(), o, null)) {
            GgufKv kv = GgufKv.of(src);
            assertTrue(ClipDualEncoder.isTwoTower(kv));
            assertEquals(f.shapes.keySet(), names(TensorCatalog.of(src)));
            assertEquals(3, kv.u32("clip.vision.block_count", 0));
            assertEquals(2, kv.u32("clip.text.block_count", 0));
            assertEquals(ClipDualFixtures.DIM, kv.u32("clip.text.projection_dim", 0));
            assertFalse(kv.bool("clip.use_gelu", true));
            assertArrayEquals(new float[]{0.48f, 0.46f, 0.41f},
                    ClipHparams.twoTower(kv, ProjectorModality.VISION).imageMean(), 0f);
        }
        RasterImage img = image(8, 8, 11);
        int[] ids = ClipDualFixtures.catDogIds();
        try (ClipDualEncoder hf = ClipDualEncoder.open(dir.toString(), o);
             ClipDualEncoder ref = ClipDualEncoder.open(gguf.toString(), ModelSourceOptions.builder()
                     .tokenizerDir(ClipDualFixtures.tokenizerDir()).build())) {
            assertEquals(ClipFamily.CLIP, hf.family());
            assertArrayEquals(ids, hf.tokenizer().encode("A cat  DOG", true, false));
            assertArrayEquals(ref.embedText(ids), hf.embedText(ids), 0f);
            assertArrayEquals(ref.embedImage(img), hf.embedImage(img), 0f);
        }
    }

    @Test
    public void siglipModelCheckpointEmbedsAsItsTwoTowerFile() throws IOException {
        MmprojFixture f = ClipDualFixtures.siglipTensors(62);
        Path gguf = ClipDualFixtures.writer(f, ClipDualFixtures.siglipMeta()).writeTempFile();
        Path dir = ClipHfFixtures.write(f, ClipHfFixtures.config(true, 2), true, null, tmp.newFolder("siglip").toPath());
        ModelSourceOptions o = options();
        assertTrue(ClipDualEncoder.isTwoTower(dir.toString()));
        try (ModelSource src = ModelSources.open(dir.toString(), o, null)) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("siglip", kv.str(ClipFamily.KEY, null));
            assertEquals(f.shapes.keySet(), names(TensorCatalog.of(src)));
            assertEquals(0, kv.u32("clip.vision.projection_dim", -1));
            assertTrue(kv.bool("clip.use_gelu", false));
            assertEquals(ClipDualFixtures.SIGLIP_PAD, kv.u32("tokenizer.ggml.padding_token_id", -1));
        }
        RasterImage img = image(8, 8, 12);
        int[] ids = {6, 11, 15};
        try (ClipDualEncoder hf = ClipDualEncoder.open(dir.toString(), o);
             ClipDualEncoder ref = ClipDualEncoder.open(gguf.toString(), ModelSourceOptions.defaults())) {
            assertEquals(ClipFamily.SIGLIP, hf.family());
            assertArrayEquals(ref.embedText(ids), hf.embedText(ids), 0f);
            assertArrayEquals(ref.embedImage(img), hf.embedImage(img), 0f);
            assertEquals(ref.score(0.3), hf.score(0.3), 0);
            assertFalse("a checkpoint without spiece.model carries no tokenizer", hf.hasTokenizer());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, hf::tokenizer);
            assertTrue(e.getMessage(), e.getMessage().contains(ModelSourceOptions.PROP_TOKENIZER));
        }
    }

    @Test
    public void siglipSentencePieceModelBecomesTheTextTokenizer() throws IOException {
        Path dir = ClipDualFixtures.siglipCheckpoint(tmp.newFolder("siglip-spiece").toPath());
        try (ClipDualEncoder enc = ClipDualEncoder.open(dir.toString(), options())) {
            assertTrue(enc.hasTokenizer());
            int[] ids = enc.tokenizer().encode("a b", true, false);
            assertArrayEquals("the pieces of a and b, then the end token", new int[]{3, 4, 1}, ids);
            assertEquals(ClipDualFixtures.SIGLIP_PAD, enc.tokenizer().padId());
            assertEquals(ClipDualFixtures.E, enc.embedText(ids).length);
        }
    }

    @Test
    public void languageModelsAreNoDualEncodersAndTheDualEncoderIsNoLanguageModel() throws IOException {
        Path llama = SyntheticHf.llama(1, 8, 2, 2, 8).write(tmp.newFolder("llama").toPath());
        assertFalse(ClipDualEncoder.isTwoTower(llama.toString()));
        Path dir = ClipHfFixtures.write(clipTensors(), ClipHfFixtures.config(false, 3), false, null,
                tmp.newFolder("clip-lm").toPath());
        Exception e = assertThrows(Exception.class, () -> new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build()).close());
        assertTrue(e.getMessage(), e.getMessage().contains("clip"));
    }
}
