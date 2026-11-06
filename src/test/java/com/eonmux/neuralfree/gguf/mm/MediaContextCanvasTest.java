package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
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
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaContext} over the MiniCPM-V 4.6 and 4.7 projectors, as llama.cpp's {@code mtmd_tokenizer} lays their
 * sliced images out: a minicpmv4_6 image is the image-id tag and the overview and slices framed by the
 * {@code <image>} / {@code <slice>} marker text; a minicpmv4_7 image is the image-id tag, {@code <image>} and one
 * canvas chunk ({@code add_canvas_chunk}) of the overview's and every slice's rows, each followed by its separator
 * rows, at the canvas positions of {@link MediaRopePositions#canvas}.
 */
public class MediaContextCanvasTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A text encoder recording every segment it tokenizes, one token per character. */
    private static final class RecordingText implements MediaContext.TextEncoder {
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

    private Path write(String type) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(VisionFamiliesDFixtures.minicpmv46(301, true, false),
                VisionFamiliesDFixtures.minicpmv46Meta(type, null, null)).writeTo(file);
        return file;
    }

    private static MediaContext open(Path file, boolean mrope) throws IOException {
        return MediaContext.open(file.toString(), VisionFamiliesDFixtures.OUT, mrope, ImageTokenBudget.none(),
                t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES);
    }

    /** The family's own preprocessing of {@code png}. */
    private static PreparedImages prepare(MmprojModel mm, byte[] png) throws IOException {
        ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
        return ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png));
    }

    private static float[][] encode(MmprojModel mm, CpuVisionEncoder enc, ImageTile tile, ClipSuffix suffix) {
        ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
        return enc.encode(ClipImage.of(tile.width(), tile.height(), tile.normalized(params.mean(), params.std()))
                .withSuffix(suffix));
    }

    @Test
    public void aMinicpmv46ImageFramesTheOverviewAndSlicesWithMarkerText() throws IOException {
        Path file = write("minicpmv4_6");
        byte[] png = MediaContextTest.png(48, 16);
        try (MediaContext ctx = open(file, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 1, text);
            MediaPrompt prompt = b.build();
            try (MmprojModel mm = MmprojModel.open(file.toString())) {
                PreparedImages prepared = prepare(mm, png);
                int slices = prepared.tiles().size();
                assertTrue("the 3:1 image slices", slices > 1);
                assertEquals(1 + slices, prompt.nMediaChunks());
                List<String> want = new ArrayList<>(Arrays.asList("<image_id>1</image_id>", "<image>", "</image>"));
                for (int y = 0; y < prepared.gridY(); y++) {
                    for (int x = 0; x < prepared.gridX(); x++) {
                        want.add("<slice>");
                        want.add("</slice>");
                    }
                    if (y != prepared.gridY() - 1) {
                        want.add("\n");
                    }
                }
                assertEquals(want, text.seen);
            }
        }
    }

    @Test
    public void aMinicpmv47ImageIsOneCanvasChunkWithItsSeparatorRows() throws IOException {
        Path file = write("minicpmv4_7");
        byte[] png = MediaContextTest.png(48, 16);
        try (MediaContext ctx = open(file, true)) {
            assertTrue(ctx.markers().canvasImages());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder().appendText(text.encode("ab"));
            ctx.appendImage(b, png, 2, text);
            b.appendText(text.encode("c"));
            MediaPrompt prompt = b.build();

            assertEquals(Arrays.asList("ab", "<image_id>2</image_id>", "<image>", "c"), text.seen);
            assertEquals(1, prompt.nMediaChunks());
            MediaChunk canvas = null;
            for (MediaChunk c : prompt.chunks()) {
                if (!c.isText()) {
                    canvas = c;
                }
            }
            int pos0 = "ab".length() + "<image_id>2</image_id>".length() + "<image>".length();
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
                PreparedImages prepared = prepare(mm, png);
                List<ImageTile> slices = prepared.tiles();
                int cols = prepared.gridX();
                int rows = prepared.gridY();
                assertTrue("the 3:1 image slices", slices.size() > 1);
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
                List<float[][]> parts = new ArrayList<>();
                List<MediaRopePositions.CanvasTile> tiles = new ArrayList<>();
                addTile(mm, enc, params, prepared.overview(), ClipSuffix.MINICPMV_OV_SLICE, parts, tiles);
                for (int y = 0; y < rows; y++) {
                    for (int x = 0; x < cols; x++) {
                        ClipSuffix s = y == rows - 1 && x == cols - 1 ? ClipSuffix.MINICPMV_LAST
                                : x == cols - 1 ? ClipSuffix.MINICPMV_ROW_END : ClipSuffix.MINICPMV_SLICE;
                        addTile(mm, enc, params, slices.get(y * cols + x), s, parts, tiles);
                    }
                }
                float[][] got = canvas.embeddings();
                int at = 0;
                for (float[][] part : parts) {
                    for (float[] row : part) {
                        assertArrayEquals("row " + at, row, got[at++], 0f);
                    }
                }
                assertEquals(at, got.length);
                MediaRopePositions want = MediaRopePositions.canvas(pos0, cols, rows, tiles);
                assertEquals(want, canvas.positions());
                // one position per tile, plus the overview's two separators, two per slice but the last, and one
                // more for every row end
                int nPos = 1 + slices.size() + 2 + 2 * (slices.size() - 1) + (rows - 1) + 1;
                assertEquals(nPos, want.nPositions());
                assertEquals(pos0 + nPos + "c".length(), prompt.nPositions());
            }
        }
    }

    private static void addTile(MmprojModel mm, CpuVisionEncoder enc, ImageProcessorParams params, ImageTile tile,
                                ClipSuffix suffix, List<float[][]> parts, List<MediaRopePositions.CanvasTile> tiles) {
        float[][] rows = encode(mm, enc, tile, suffix);
        ImageSize grid = params.tokenGridFor(tile.size());
        assertEquals(grid.width() * grid.height() + suffix.size(), rows.length);
        parts.add(rows);
        tiles.add(new MediaRopePositions.CanvasTile(grid.width(), grid.height(), suffix.size()));
    }

    @Test
    public void aMinicpmv47ProjectorNeedsAnMropeTextModel() throws IOException {
        Path file = write("minicpmv4_7");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> open(file, false).close());
        assertTrue(e.getMessage(), e.getMessage().contains("M-RoPE"));
    }
}
