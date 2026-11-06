package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The kimik25, gemma4v, gemma4uv, MiniCPM-V 4.6, MiMo-VL and Gemma 3n families through a real mmproj GGUF file:
 * {@link MmprojModel#open}, the registry row and tensor table, and the CPU encoder of
 * {@link MmprojModel#openVisionEncoder} against the rows llama.cpp's own clip graphs produce for the same file and
 * image ({@link LlamaCppParity}); files the reference loader refuses are refused at open, naming the tensor or key.
 * A mimovl and a Gemma 3n image also run end to end through {@link MediaContext}; Gemma 3n F16 kernels encode
 * like F32 kernels of the same values.
 */
public class VisionFamiliesDTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path write(MmprojFixture f, Map<String, Object> meta) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(f, meta).writeTo(file);
        return file;
    }

    /** Encodes case {@code name} on the CPU and checks it against the llama.cpp rows. */
    private void matchesLlamaCpp(String name, ProjectorType type) throws IOException {
        LlamaCppParity.Case c = LlamaCppParity.find(name);
        float[][] want = LlamaCppParity.rows(name);
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        c.gguf().writeTo(file);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            assertTrue(mm.caps().inpVision());
            assertEquals(type, mm.visionHparams().projectorType());
            try (VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
                assertEquals(want[0].length, enc.model().outputDim());
                LlamaCppParity.assertMatches(name, want, enc.encode(c.image()));
            }
        }
    }

    private void refusedAtOpen(MmprojFixture f, Map<String, Object> meta, Class<? extends Exception> type,
                               String named) throws IOException {
        Path file = write(f, meta);
        Exception e = assertThrows(type, () -> {
            try (MmprojModel mm = MmprojModel.open(file.toString())) {
                mm.openCpuVisionEncoder().close();
            }
        });
        assertTrue(e.getMessage(), e.getMessage().contains(named));
    }

    // ================================================================== kimik25

    @Test
    public void kimik25MatchesLlamaCppWithTheResizedPositionGrid() throws IOException {
        matchesLlamaCpp("kimik25", ProjectorType.KIMIK25);
    }

    @Test
    public void kimik25MatchesLlamaCppOnTheStoredPositionGrid() throws IOException {
        matchesLlamaCpp("kimik25.stored_grid", ProjectorType.KIMIK25);
    }

    @Test
    public void kimik25RefusesAFlatPositionTableAndAMissingProjectorNorm() throws IOException {
        MmprojFixture flat = VisionFamiliesDFixtures.kimik25(203);
        flat.remove("v.position_embd.weight");
        flat.weight("v.position_embd.weight", E + 1, 6);
        refusedAtOpen(flat, VisionFamiliesDFixtures.kimik25Meta(), IllegalArgumentException.class,
                "v.position_embd.weight");
        MmprojFixture noNorm = VisionFamiliesDFixtures.kimik25(204);
        noNorm.remove("mm.input_norm.bias");
        refusedAtOpen(noNorm, VisionFamiliesDFixtures.kimik25Meta(), MissingTensorException.class,
                "mm.input_norm.bias");
    }

    // ================================================================== gemma4v / gemma4uv

    @Test
    public void gemma4vMatchesLlamaCppWithClampsPostNormsAndTheTwoByTwoPool() throws IOException {
        matchesLlamaCpp("gemma4v", ProjectorType.GEMMA4V);
    }

    @Test
    public void gemma4vMatchesLlamaCppWithTheDefaultThreeByThreePool() throws IOException {
        matchesLlamaCpp("gemma4v.pool3", ProjectorType.GEMMA4V);
    }

    @Test
    public void gemma4uvMatchesLlamaCpp() throws IOException {
        matchesLlamaCpp("gemma4uv", ProjectorType.GEMMA4UV);
    }

    @Test
    public void gemma4vRefusesAPatchBiasALoneStandardizationHalfAndFlatPositionTables() throws IOException {
        MmprojFixture biased = VisionFamiliesDFixtures.gemma4v(214);
        biased.weight("v.patch_embd.bias", VisionFamiliesDFixtures.G);
        refusedAtOpen(biased, VisionFamiliesDFixtures.gemma4vMeta(2), IllegalArgumentException.class,
                "v.patch_embd.bias");
        MmprojFixture lone = VisionFamiliesDFixtures.gemma4v(215);
        lone.remove("v.std_scale");
        refusedAtOpen(lone, VisionFamiliesDFixtures.gemma4vMeta(2), IllegalArgumentException.class, "v.std_bias");
        MmprojFixture flat = VisionFamiliesDFixtures.gemma4v(216);
        flat.remove("v.position_embd.weight");
        flat.weight("v.position_embd.weight", VisionFamiliesDFixtures.G, 2 * VisionFamiliesDFixtures.AXIS_ROWS);
        refusedAtOpen(flat, VisionFamiliesDFixtures.gemma4vMeta(2), IllegalArgumentException.class,
                "v.position_embd.weight");
    }

    @Test
    public void gemma4vRefusesAGridBeyondThePositionTablesOrThePool() throws IOException {
        Path file = write(VisionFamiliesDFixtures.gemma4v(217), VisionFamiliesDFixtures.gemma4vMeta(2));
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            int wide = 2 * 2 * (VisionFamiliesDFixtures.AXIS_ROWS + 1);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(ClipImage.of(wide, 4, MmprojFixture.pixels(wide, 4, 7))));
            assertTrue(e.getMessage(), e.getMessage().contains("v.position_embd.weight"));
            e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(ClipImage.of(6, 4, MmprojFixture.pixels(6, 4, 8))));
            assertTrue(e.getMessage(), e.getMessage().contains("pool"));
        }
    }

    @Test
    public void gemma4uvRefusesAMissingPatchNorm() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.gemma4uv(218);
        f.remove("v.patch_norm.3.bias");
        refusedAtOpen(f, VisionFamiliesDFixtures.gemma4uvMeta(), MissingTensorException.class, "v.patch_norm.3.bias");
    }

    // ================================================================== minicpmv4_6 / 4_7 / ocr

    @Test
    public void minicpmv46MatchesLlamaCppWithTheVitMergerAfterTheFirstBlock() throws IOException {
        matchesLlamaCpp("minicpmv4_6", ProjectorType.MINICPMV4_6);
    }

    @Test
    public void minicpmv47MatchesLlamaCppWithTheOverviewSeparatorRows() throws IOException {
        matchesLlamaCpp("minicpmv4_7.ov_slice", ProjectorType.MINICPMV4_7);
    }

    @Test
    public void minicpmv47MatchesLlamaCppWithTheRowEndSeparatorRows() throws IOException {
        matchesLlamaCpp("minicpmv4_7.row_end", ProjectorType.MINICPMV4_7);
    }

    @Test
    public void minicpmOcrMatchesLlamaCppWithATwoByTwoMergeAndSwappedFfnTensors() throws IOException {
        matchesLlamaCpp("minicpm_ocr_1_0.merge2", ProjectorType.MINICPM_OCR_1_0);
    }

    @Test
    public void minicpmvRefusesMissingMergerTensorsSeparatorsAndBadKeys() throws IOException {
        MmprojFixture noMerger = VisionFamiliesDFixtures.minicpmv46(225, false, false);
        refusedAtOpen(noMerger, VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_6", null, null),
                MissingTensorException.class, "v.vit_merger.ln1.weight");
        MmprojFixture noSep = VisionFamiliesDFixtures.minicpmv46(226, true, false);
        noSep.remove(MiniCpmv46Graph.SEPARATORS);
        refusedAtOpen(noSep, VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_7", null, null),
                MissingTensorException.class, MiniCpmv46Graph.SEPARATORS);
        MmprojFixture ok = VisionFamiliesDFixtures.minicpmv46(227, true, false);
        refusedAtOpen(ok, VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_6", 3, null),
                GgufFormatException.class, "clip.vision.projector.scale_factor");
        refusedAtOpen(ok, VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_6", null, 2),
                GgufFormatException.class, "clip.vision.wa_layer_indexes");
    }

    @Test
    public void separatorRowsAreRefusedOutsideTheMinicpmvMergersAndGridsMustFitTheMerge() throws IOException {
        Path gemma = write(VisionFamiliesDFixtures.gemma4v(228), VisionFamiliesDFixtures.gemma4vMeta(2));
        try (MmprojModel mm = MmprojModel.open(gemma.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            ClipImage img = ClipImage.of(8, 8, MmprojFixture.pixels(8, 8, 9)).withSuffix(ClipSuffix.MINICPMV_LAST);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> enc.encode(img));
            assertTrue(e.getMessage(), e.getMessage().contains("gemma4v"));
        }
        Path ocr = write(VisionFamiliesDFixtures.minicpmv46(229, true, false),
                VisionFamiliesDFixtures.minicpmv46Meta("minicpm_ocr_1_0", null, null));
        try (MmprojModel mm = MmprojModel.open(ocr.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(ClipImage.of(12, 8, MmprojFixture.pixels(12, 8, 10))));
            assertTrue(e.getMessage(), e.getMessage().contains("scale_factor"));
            assertEquals(1 + ClipSuffix.MINICPMV_ROW_END.size(), enc.encode(ClipImage.of(8, 8,
                    MmprojFixture.pixels(8, 8, 11)).withSuffix(ClipSuffix.MINICPMV_ROW_END)).length);
        }
    }

    // ================================================================== mimovl

    @Test
    public void mimovlMatchesLlamaCppWithRowAndColumnWindowsSinksAndGroupedHeads() throws IOException {
        matchesLlamaCpp("mimovl", ProjectorType.MIMOVL);
    }

    @Test
    public void mimovlMatchesLlamaCppOnAPortraitGridWithANarrowWindow() throws IOException {
        matchesLlamaCpp("mimovl.portrait", ProjectorType.MIMOVL);
    }

    @Test
    public void mimovlRefusesMissingSinksAndKernelsAndBiasesTheGraphDoesNotAdd() throws IOException {
        Map<String, Object> meta = VisionFamiliesDFixtures.mimovlMeta(5);
        MmprojFixture noSinks = VisionFamiliesDFixtures.mimovl(233);
        noSinks.remove("v.blk.2.attn_sinks");
        refusedAtOpen(noSinks, meta, MissingTensorException.class, "v.blk.2.attn_sinks");
        MmprojFixture noSecondKernel = VisionFamiliesDFixtures.mimovl(234);
        noSecondKernel.remove("v.patch_embd.weight.1");
        refusedAtOpen(noSecondKernel, meta, MissingTensorException.class, "v.patch_embd.weight.1");
        MmprojFixture patchBias = VisionFamiliesDFixtures.mimovl(235);
        patchBias.weight("v.patch_embd.bias", E);
        refusedAtOpen(patchBias, meta, IllegalArgumentException.class, "v.patch_embd.bias");
        MmprojFixture mergerBias = VisionFamiliesDFixtures.mimovl(236);
        mergerBias.weight("mm.0.bias", VisionFamiliesDFixtures.HID);
        refusedAtOpen(mergerBias, meta, IllegalArgumentException.class, "mm.0.bias");
        MmprojFixture oddQkv = VisionFamiliesDFixtures.mimovl(237);
        oddQkv.remove("v.blk.0.attn_qkv.weight");
        oddQkv.weight("v.blk.0.attn_qkv.weight", E, 63);
        refusedAtOpen(oddQkv, meta, IllegalArgumentException.class, "v.blk.0.attn_qkv.weight");
    }

    @Test
    public void mimovlRefusesMissingOrMalformedWindowKeys() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.mimovl(238);
        Map<String, Object> shortModes = VisionFamiliesDFixtures.mimovlMeta(5);
        shortModes.put("clip.vision.wa_pattern_mode", new int[]{0, 1, -1});
        refusedAtOpen(f, shortModes, GgufFormatException.class, "clip.vision.wa_pattern_mode");
        Map<String, Object> badMode = VisionFamiliesDFixtures.mimovlMeta(5);
        badMode.put("clip.vision.wa_pattern_mode", new int[]{0, 2, -1, 1});
        refusedAtOpen(f, badMode, GgufFormatException.class, "clip.vision.wa_pattern_mode");
        Map<String, Object> noWindow = VisionFamiliesDFixtures.mimovlMeta(5);
        noWindow.remove("clip.vision.window_size");
        refusedAtOpen(f, noWindow, GgufFormatException.class, "clip.vision.window_size");
        refusedAtOpen(f, VisionFamiliesDFixtures.mimovlMeta(0), GgufFormatException.class,
                "clip.vision.window_size");
        Map<String, Object> noKvHeads = VisionFamiliesDFixtures.mimovlMeta(5);
        noKvHeads.remove("clip.vision.attention.head_count_kv");
        refusedAtOpen(f, noKvHeads, GgufFormatException.class, "clip.vision.attention.head_count_kv");
        Map<String, Object> merge4 = VisionFamiliesDFixtures.mimovlMeta(5);
        merge4.put("clip.vision.spatial_merge_size", 4);
        refusedAtOpen(f, merge4, GgufFormatException.class, "clip.vision.spatial_merge_size");
    }

    @Test
    public void mimovlRefusesAPatchGridThatIsNotWholeMergeUnits() throws IOException {
        Path file = write(VisionFamiliesDFixtures.mimovl(239), VisionFamiliesDFixtures.mimovlMeta(5));
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            assertEquals(VisionFamiliesDFixtures.MIMO_HEAD_DIM, enc.model().dHead());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(ClipImage.of(10, 8, MmprojFixture.pixels(10, 8, 12))));
            assertTrue(e.getMessage(), e.getMessage().contains("multiple of 2"));
        }
    }

    @Test
    public void mimovlServesAnImageThroughTheMediaContextOnItsTokenGrid() throws IOException {
        Path file = write(VisionFamiliesDFixtures.mimovl(240), VisionFamiliesDFixtures.mimovlMeta(5));
        byte[] png = MediaContextTest.png(48, 32);
        try (MediaContext ctx = MediaContext.open(file.toString(), VisionFamiliesDFixtures.OUT, true,
                ImageTokenBudget.none(), t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES)) {
            List<String> seen = new ArrayList<>();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text -> {
                seen.add(text);
                return new int[text.length()];
            });
            MediaPrompt prompt = b.build();
            assertEquals(Arrays.asList("<|vision_start|>", "<|vision_end|>"), seen);
            MediaChunk image = prompt.chunks().get(1);
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
                ImageTile tile = ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png))
                        .encodedTiles().get(0);
                ImageSize grid = params.tokenGridFor(tile.size());
                float[][] want = enc.encode(ClipImage.of(tile.width(), tile.height(),
                        tile.normalized(params.mean(), params.std())));
                assertEquals(grid.width() * grid.height(), want.length);
                assertEquals(MediaRopePositions.Kind.IMAGE_2D, image.positions().kind());
                assertEquals(grid.width(), image.positions().nx());
                assertEquals(grid.height(), image.positions().ny());
                float[][] got = image.embeddings();
                assertEquals(want.length, got.length);
                for (int i = 0; i < want.length; i++) {
                    assertArrayEquals("row " + i, want[i], got[i], 0f);
                }
            }
        }
    }

    // ================================================================== gemma3nv

    @Test
    public void gemma3nvMatchesLlamaCppAcrossEveryBlockKindAndTheFusionUpscale() throws IOException {
        matchesLlamaCpp("gemma3nv", ProjectorType.GEMMA3NV);
    }

    @Test
    public void gemma3nvMatchesLlamaCppWhenTheFusionMapPoolsToSixteenASide() throws IOException {
        matchesLlamaCpp("gemma3nv.pool", ProjectorType.GEMMA3NV);
    }

    @Test
    public void gemma3nvMatchesLlamaCppWithThreeStages() throws IOException {
        matchesLlamaCpp("gemma3nv.three_stages", ProjectorType.GEMMA3NV);
    }

    @Test
    public void gemma3nvF16KernelsEncodeLikeF32KernelsOfTheSameValues() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.gemma3nv(254, true);
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            if (VisionFamiliesDFixtures.gemma3nvHalfTensor(e.getKey(), e.getValue())) {
                float[] v = f.values.get(e.getKey());
                for (int i = 0; i < v.length; i++) {
                    v[i] = HalfFloats.halfToFloat(HalfFloats.floatToHalf(v[i]));
                }
            }
        }
        Map<String, Object> meta = VisionFamiliesDFixtures.gemma3nvMeta(16);
        Path f32 = write(f, meta);
        Path f16 = tmp.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(f, meta, (name, shape) -> VisionFamiliesDFixtures.gemma3nvHalfTensor(name, shape)
                ? GGMLType.F16 : GGMLType.F32).writeTo(f16);
        ClipImage image = ClipImage.of(64, 32, MmprojFixture.pixels(64, 32, 17));
        float[][] want = encodeCpu(f32, image);
        float[][] got = encodeCpu(f16, image);
        assertEquals(8, want.length);
        LlamaCppParity.assertMatches("gemma3nv F16 kernels", want, got);
    }

    @Test
    public void gemma3nvRefusesAConvKernelOtherThanF32OrF16() throws IOException {
        String kernel = "v.blk.1.0." + MobileNetV5Layout.PW_EXP;
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(VisionFamiliesDFixtures.gemma3nv(255, true), VisionFamiliesDFixtures.gemma3nvMeta(16),
                (name, shape) -> name.equals(kernel) ? GGMLType.BF16 : GGMLType.F32).writeTo(file);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> {
            try (MmprojModel mm = MmprojModel.open(file.toString())) {
                mm.openCpuVisionEncoder().close();
            }
        });
        assertTrue(e.getMessage(), e.getMessage().contains(kernel) && e.getMessage().contains("BF16"));
    }

    @Test
    public void gemma3nvRefusesABlockHoldingTwoBlockGraphs() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.gemma3nv(256, true);
        f.weight("v.blk.1.0." + MobileNetV5Layout.CONV_EXP, 3, 3, 4, 8);
        refusedAtOpen(f, VisionFamiliesDFixtures.gemma3nvMeta(16), IllegalArgumentException.class, "v.blk.1.0.");
    }

    @Test
    public void gemma3nvRefusesFeaturesThatDoNotEndWithTheEmbeddingWidth() throws IOException {
        Map<String, Object> meta = VisionFamiliesDFixtures.gemma3nvMeta(16);
        meta.put("clip.vision.embedding_length", VisionFamiliesDFixtures.MNV5_EMBD + 1);
        refusedAtOpen(VisionFamiliesDFixtures.gemma3nv(257, true), meta, IllegalArgumentException.class,
                "clip.vision.embedding_length");
    }

    @Test
    public void gemma3nvRefusesAKeyDownsampleWithoutTheValueOne() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.gemma3nv(258, true);
        f.remove("v.blk.2.1." + MobileNetV5Layout.VALUE_DOWN);
        refusedAtOpen(f, VisionFamiliesDFixtures.gemma3nvMeta(16), MissingTensorException.class,
                "v.blk.2.1." + MobileNetV5Layout.VALUE_DOWN);
    }

    @Test
    public void gemma3nvRefusesAFileWithoutTheInputProjection() throws IOException {
        MmprojFixture f = VisionFamiliesDFixtures.gemma3nv(259, true);
        f.remove(MobileNetV5Layout.INPUT_PROJECTION);
        refusedAtOpen(f, VisionFamiliesDFixtures.gemma3nvMeta(16), MissingTensorException.class,
                MobileNetV5Layout.INPUT_PROJECTION);
    }

    @Test
    public void gemma3nvServesAnImageThroughTheMediaContextAtTheFixedSize() throws IOException {
        // a 64-pixel square image: the fusion map of 4x4 pixels, 64 / 4 = 16 tokens
        Path file = write(VisionFamiliesDFixtures.gemma3nv(260, true), VisionFamiliesDFixtures.gemma3nvMeta(64, 4));
        byte[] png = MediaContextTest.png(96, 40);
        try (MediaContext ctx = MediaContext.open(file.toString(), VisionFamiliesDFixtures.OUT, true,
                ImageTokenBudget.none(), t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES)) {
            List<String> seen = new ArrayList<>();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text -> {
                seen.add(text);
                return new int[text.length()];
            });
            MediaPrompt prompt = b.build();
            assertEquals(Arrays.asList("<start_of_image>", "<end_of_image>"), seen);
            MediaChunk image = prompt.chunks().get(1);
            try (MmprojModel mm = MmprojModel.open(file.toString())) {
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
                ImageTile tile = ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png))
                        .encodedTiles().get(0);
                assertEquals(64, tile.width());
                assertEquals(64, tile.height());
                float[][] want = encodeCpu(file, ClipImage.of(tile.width(), tile.height(),
                        tile.normalized(params.mean(), params.std())));
                assertEquals(16, want.length);
                float[][] got = image.embeddings();
                assertEquals(want.length, got.length);
                for (int i = 0; i < want.length; i++) {
                    assertArrayEquals("row " + i, want[i], got[i], 0f);
                }
            }
        }
    }

    private static float[][] encodeCpu(Path file, ClipImage image) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            assertEquals(ProjectorType.GEMMA3NV, mm.visionHparams().projectorType());
            return enc.encode(image);
        }
    }
}
