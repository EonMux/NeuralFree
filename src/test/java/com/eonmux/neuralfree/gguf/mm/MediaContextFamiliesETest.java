package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.image.PreparedImages;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaContext} end to end over the sensenova_u1, step3vl, deepseek4v, deepseek41v, deepseekocr,
 * deepseekocr2 and granite4_vision fixtures: the family's markers, its chunk layout and positions as llama.cpp's
 * {@code mtmd_tokenizer} lays them out, and rows equal to a direct encode of the family's preprocessed image.
 */
public class MediaContextFamiliesETest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A text encoder recording every segment it tokenizes, one token per character. */
    static final class RecordingText implements MediaContext.TextEncoder {
        final List<String> seen = new ArrayList<>();

        @Override
        public int[] encode(String text) {
            seen.add(text);
            int[] out = new int[text.length()];
            for (int i = 0; i < out.length; i++) {
                out[i] = text.charAt(i);
            }
            return out;
        }
    }

    private Path write(MmprojFixture f, Map<String, Object> meta) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(f, meta).writeTo(file);
        return file;
    }

    private static MediaContext open(Path file, int width, boolean mrope) throws IOException {
        return MediaContext.open(file.toString(), width, mrope, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES);
    }

    /** The family's preprocessing of {@code png}. */
    private static PreparedImages prepare(Path file, byte[] png) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            return ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png));
        }
    }

    /** A direct CPU encode of one preprocessed tile. */
    private static float[][] encodeTile(Path file, ImageTile tile) throws IOException {
        return encodeTile(file, tile, 0);
    }

    /** A direct CPU encode of one preprocessed tile whose block opens with {@code lead} lead pads. */
    private static float[][] encodeTile(Path file, ImageTile tile, int lead) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            return enc.encode(ClipImage.of(tile.width(), tile.height(),
                    tile.normalized(params.mean(), params.std())).withLeadPad(lead));
        }
    }

    private static void assertRows(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertArrayEquals("row " + t, want[t], got[t], 0f);
        }
    }

    @Test
    public void sensenovaImageTakesZeroBasedGridPositionsAndOnePosition() throws IOException {
        Path file = write(VisionFamiliesEFixtures.sensenova(301), VisionFamiliesEFixtures.sensenovaMeta());
        byte[] png = MediaContextTest.png(32, 16);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, true)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            b.appendText(new int[]{1, 2, 3});
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<img>", "</img>"), text.seen);
            MediaChunk image = prompt.chunks().get(1);
            assertTrue(image.nonCausal());
            MediaRopePositions pos = image.positions();
            assertEquals(MediaRopePositions.Kind.IMAGE_2D_ZERO_BASED, pos.kind());
            int pos0 = 3 + "<img>".length();
            assertEquals(pos0, pos.pos0());
            assertEquals(8, pos.nx());
            assertEquals(4, pos.ny());
            for (int i = 0; i < pos.nTokens(); i++) {
                assertEquals(pos0, pos.position(0, i));
                assertEquals(i / 8, pos.position(1, i));
                assertEquals(i % 8, pos.position(2, i));
                assertEquals(0, pos.position(3, i));
            }
            assertEquals(1, pos.nPositions());
            assertEquals(pos0 + 1 + "</img>".length(), prompt.nPositions());
            ImageTile tile = prepare(file, png).encodedTiles().get(0);
            assertRows(encodeTile(file, tile), image.embeddings());
        }
    }

    @Test
    public void step3vlSmallImageIsItsOverviewBetweenTheOverviewMarkers() throws IOException {
        Map<String, Object> meta = VisionFamiliesEFixtures.step3vlMeta();
        meta.put("clip.vision.image_size", 16);
        Path file = write(VisionFamiliesEFixtures.step3vl(311, true), meta);
        byte[] png = MediaContextTest.png(12, 10);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<im_start>", "<im_end>"), text.seen);
            MediaChunk image = prompt.chunks().get(1);
            assertEquals(MediaRopePositions.Kind.NORMAL, image.positions().kind());
            assertEquals(4, image.nTokens());
            PreparedImages prepared = prepare(file, png);
            assertEquals(0, prepared.tiles().size());
            assertRows(encodeTile(file, prepared.overview()), image.embeddings());
        }
    }

    @Test
    public void deepseek4vBlockTakesTheLeadPadsThatAlignItAfterThePrompt() throws IOException {
        Path file = write(VisionFamiliesEFixtures.deepseekV4(321, true),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"));
        byte[] png = MediaContextTest.png(20, 12);
        ImageTile tile = prepare(file, png).encodedTiles().get(0);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            assertEquals(4, ctx.markers().imageLeadPadAlign());
            for (int before = 0; before < 8; before++) {
                RecordingText text = new RecordingText();
                MediaPrompt.Builder b = MediaPrompt.builder();
                if (before > 0) {
                    b.appendText(new int[before]);
                }
                ctx.appendImage(b, png, 0, text);
                MediaPrompt prompt = b.build();

                assertTrue("no marker text", text.seen.isEmpty());
                MediaChunk image = prompt.chunks().get(before > 0 ? 1 : 0);
                int lead = 3 - before % 4;
                assertTrue(image.id(), image.id().endsWith("#0@lead" + lead));
                assertTrue(image.nonCausal());
                assertEquals(MediaRopePositions.Kind.NORMAL, image.positions().kind());
                assertEquals(before, image.positions().pos0());
                assertEquals("the image start sentinel opens a block of 4", 0, (before + lead + 1) % 4);
                assertRows(encodeTile(file, tile, lead), image.embeddings());
            }
            assertEquals("one encode serves every lead pad count", 1, ctx.cachedItems());
        }
    }

    @Test
    public void deepseek41vBlockIsCausalAndTakesNoLeadPads() throws IOException {
        Path file = write(VisionFamiliesEFixtures.deepseekV4(331, false),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek41v"));
        byte[] png = MediaContextTest.png(20, 12);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            assertEquals(0, ctx.markers().imageLeadPadAlign());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            b.appendText(new int[]{1, 2});
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertTrue("no marker text", text.seen.isEmpty());
            assertEquals(2, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            assertTrue(image.id(), image.id().endsWith("#0"));
            assertFalse(image.nonCausal());
            assertEquals(2, image.positions().pos0());
            ImageTile tile = prepare(file, png).encodedTiles().get(0);
            float[][] want = encodeTile(file, tile);
            assertRows(want, image.embeddings());
            int w = DeepseekV4Layout.gridSide(tile.width() / 2, 2);
            int h = DeepseekV4Layout.gridSide(tile.height() / 2, 2);
            assertEquals(DeepseekV4Layout.tokens41v(w, h), want.length);
        }
    }

    /** A direct CPU encode of the global view of a prepared image. */
    private static float[][] encodeGlobalView(Path file, ImageTile overview) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            return enc.encode(ClipImage.of(overview.width(), overview.height(),
                    overview.normalized(params.mean(), params.std())).withViewSeparator(true));
        }
    }

    @Test
    public void deepseekOcrSmallImageIsTheGlobalViewThenANewline() throws IOException {
        Path file = write(VisionFamiliesEFixtures.deepseekOcr(341),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"));
        byte[] png = MediaContextTest.png(40, 30);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("\n"), text.seen);
            assertEquals(2, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(0);
            // the 1024 global view: 16x16 tower rows, a newline per row and the view separator
            assertEquals(16 * 17 + 1, image.nTokens());
            assertEquals(MediaRopePositions.Kind.NORMAL, image.positions().kind());
            PreparedImages prepared = prepare(file, png);
            assertEquals(0, prepared.tiles().size());
            assertRows(encodeGlobalView(file, prepared.overview()), image.embeddings());
        }
    }

    @Test
    public void deepseekOcrWideImageEncodesItsTileRowsThenTheGlobalView() throws IOException {
        Path file = write(VisionFamiliesEFixtures.deepseekOcr(342),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"));
        byte[] png = MediaContextTest.png(1300, 640);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, new RecordingText());
            MediaPrompt prompt = b.build();

            PreparedImages prepared = prepare(file, png);
            assertEquals("a 2x1 grid fuses into one tile row", 1, prepared.tiles().size());
            assertEquals(3, prompt.chunks().size());
            MediaChunk row = prompt.chunks().get(0);
            // two 640 tiles of 10x10 tower rows side by side, a newline per row
            assertEquals((10 * 2 + 1) * 10, row.nTokens());
            assertRows(encodeTile(file, prepared.tiles().get(0)), row.embeddings());
            MediaChunk global = prompt.chunks().get(1);
            assertEquals(16 * 17 + 1, global.nTokens());
            assertRows(encodeGlobalView(file, prepared.overview()), global.embeddings());
            assertTrue(row.id() + " / " + global.id(), row.id().endsWith("#0") && global.id().endsWith("#1"));
        }
    }

    @Test
    public void deepseekOcr2EncodesTheTilesThenTheGlobalViewWithItsSeparator() throws IOException {
        Path file = write(VisionFamiliesEFixtures.deepseekOcr2(351),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2"));
        byte[] png = MediaContextTest.png(1500, 770);
        try (MediaContext ctx = open(file, VisionFamiliesEFixtures.OUT, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("\n"), text.seen);
            PreparedImages prepared = prepare(file, png);
            int nTiles = prepared.tiles().size();
            assertEquals(2, nTiles);
            assertEquals(nTiles + 2, prompt.chunks().size());
            for (int i = 0; i < nTiles; i++) {
                MediaChunk tile = prompt.chunks().get(i);
                assertEquals(DeepseekOcrGraph.QUERIES_768, tile.nTokens());
                assertRows(encodeTile(file, prepared.tiles().get(i)), tile.embeddings());
            }
            MediaChunk global = prompt.chunks().get(nTiles);
            assertEquals(DeepseekOcrGraph.QUERIES_1024 + 1, global.nTokens());
            assertRows(encodeGlobalView(file, prepared.overview()), global.embeddings());
        }
    }

    /** The granite4_vision fixture's metadata with anyres candidates of one to two 16-pixel tiles per side. */
    private static Map<String, Object> granite4Meta() {
        Map<String, Object> meta = VisionFamilyFixturesB.granite4Meta();
        meta.put("clip.vision.image_grid_pinpoints", new int[]{16, 16, 32, 16, 16, 32, 32, 32});
        return meta;
    }

    /** A direct CPU {@code encodeAnyres} of a prepared anyres group. */
    private static float[][] encodeAnyres(Path file, PreparedImages prepared) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            List<ClipImage> tiles = new ArrayList<>();
            for (ImageTile tile : prepared.tiles()) {
                tiles.add(ClipImage.of(tile.width(), tile.height(), tile.normalized(params.mean(), params.std())));
            }
            ImageTile ov = prepared.overview();
            return enc.encodeAnyres(ClipImage.of(ov.width(), ov.height(), ov.normalized(params.mean(), params.std())),
                    tiles, prepared.gridX(), prepared.gridY(), prepared.anyresOriginal().width(),
                    prepared.anyresOriginal().height());
        }
    }

    @Test
    public void granite4WideImageIsOneChunkOfTheOverviewAndTheCroppedGrid() throws IOException {
        Path file = write(VisionFamilyFixturesB.granite4(361), granite4Meta());
        byte[] png = MediaContextTest.png(30, 10);
        int width = 2 * VisionFamilyFixturesB.OUT;
        try (MediaContext ctx = open(file, width, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            b.appendText(new int[]{1, 2});
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("\n"), text.seen);
            PreparedImages prepared = prepare(file, png);
            assertTrue(prepared.isAnyres());
            assertEquals(2, prepared.gridX());
            assertEquals(1, prepared.gridY());
            assertEquals(3, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            // 4x4 overview tokens, then the 8x4 grid cropped to 8x2 with a newline per row
            assertEquals(16 + 2 * 9, image.nTokens());
            assertEquals(prepared.totalTokens(), image.nTokens());
            assertEquals(MediaRopePositions.Kind.NORMAL, image.positions().kind());
            assertEquals(2, image.positions().pos0());
            assertFalse(image.nonCausal());
            assertRows(encodeAnyres(file, prepared), image.embeddings());
            assertEquals(2 + image.nTokens() + 1, prompt.nPositions());
        }
    }

    @Test
    public void granite4SmallImageStillEncodesItsOneTileGridOnOneRowUnderMrope() throws IOException {
        Path file = write(VisionFamilyFixturesB.granite4(362), granite4Meta());
        byte[] png = MediaContextTest.png(8, 8);
        try (MediaContext ctx = open(file, 2 * VisionFamilyFixturesB.OUT, true)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, new RecordingText());
            MediaPrompt prompt = b.build();

            PreparedImages prepared = prepare(file, png);
            assertEquals(1, prepared.gridX());
            assertEquals(1, prepared.gridY());
            MediaChunk image = prompt.chunks().get(0);
            // the overview, then the whole 4x4 tile with a newline per row
            assertEquals(16 + 4 * 5, image.nTokens());
            MediaRopePositions pos = image.positions();
            assertEquals(MediaRopePositions.Kind.IMAGE_2D, pos.kind());
            assertEquals(image.nTokens(), pos.nx());
            assertEquals(1, pos.ny());
            assertRows(encodeAnyres(file, prepared), image.embeddings());
        }
    }
}
