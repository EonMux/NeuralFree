package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The preprocessors of the families ported from llama.cpp tools/mtmd against the output of the reference code
 * itself: mtmd-image.cpp run with each family's load_hparams settings on a synthetic image
 * (mtmd-image-reference.txt). Per run the slice grid, the overview, every entry's size, token count and pixel
 * checksums must match, and an anyres run's token count of the assembled image.
 */
public class ReferencePreprocessingTest {

    private static final String REFERENCE = "com/eonmux/neuralfree/gguf/image/mtmd-image-reference.txt";

    /** The GGUF metadata of each reference case, the keys the converter writes for that family. */
    private static Map<String, Object> metadata(String name) {
        switch (name) {
            case "lfm2":
                return meta("lfm2", 16, 512, "clip.vision.projector.scale_factor", 2);
            case "lfm2bicubic":
                return meta("lfm2", 16, 512, "clip.vision.projector.scale_factor", 2,
                        "clip.vision.image_resize_algo", "bicubic");
            case "step3vl":
                return meta("step3vl", 14, 728);
            case "glm5v":
                return meta("glm5v", 14, 336, "clip.vision.image_min_pixels", 3136,
                        "clip.vision.image_max_pixels", 1003520);
            case "youtuvl":
                return meta("youtuvl", 16, 560);
            case "lfm3vl":
                return meta("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                        "clip.vision.image_resize_algo", "bilinear", "clip.vision.image_min_pixels", 65536,
                        "clip.vision.image_max_pixels", 1048576);
            case "muse":
                return meta("muse-glimmer", 16, 512);
            case "dsocr":
                return meta("deepseekocr", 16, 1024);
            case "dsocr2":
                return meta("deepseekocr2", 16, 1024);
            case "lightonocr":
                return meta("lightonocr", 14, 1540, "clip.vision.spatial_merge_size", 2);
            case "llama4":
                return meta("llama4", 14, 336, "clip.vision.projector.scale_factor", 2);
            case "gemma4v":
                return meta("gemma4v", 16, 768);
            case "gemma4uv":
                return meta("gemma4uv", 16, 768);
            case "minicpmv46":
                return meta("minicpmv4_6", 14, 448);
            case "minicpmocr":
                return meta("minicpm_ocr_1_0", 14, 448);
            case "minicpmocr2":
                return meta("minicpm_ocr_1_0", 14, 448, "clip.vision.projector.scale_factor", 2);
            case "glmedge":
                return meta("adapter", 14, 672);
            case "janus":
                return meta("janus_pro", 16, 384);
            case "minimax":
                return meta("minimax_m3", 14, 672);
            case "sensenova":
                return meta("sensenova_u1", 16, 0, "clip.vision.spatial_merge_size", 2,
                        "clip.vision.image_min_pixels", 3136, "clip.vision.image_max_pixels", 12845056);
            case "deepseek4v":
                return meta("deepseek4v", 16, 1024, "clip.vision.projector.scale_factor", 2,
                        "clip.vision.image_min_pixels", 65536);
            case "deepseek41v":
                return meta("deepseek41v", 16, 1024, "clip.vision.projector.scale_factor", 2,
                        "clip.vision.image_min_pixels", 65536);
            case "granite4":
                return meta("granite4_vision", 16, 384, "clip.vision.projector.window_side", 8,
                        "clip.vision.projector.query_side", 4, "clip.vision.image_grid_pinpoints",
                        new int[] {384, 384, 384, 768, 768, 384, 768, 768, 384, 1152, 1152, 384, 1152, 768,
                                768, 1152});
            default:
                throw new IllegalArgumentException("no metadata for reference case " + name);
        }
    }

    private static Map<String, Object> meta(String projector, int patch, int imageSize, Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", projector);
        m.put("clip.vision.patch_size", patch);
        m.put("clip.vision.image_size", imageSize);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    /** The synthetic image of the reference run. */
    static RasterImage pattern(int width, int height) {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                rgb[i] = (byte) ((x * 7 + y * 13) & 255);
                rgb[i + 1] = (byte) ((x * 3 + y * 5 + 17) & 255);
                rgb[i + 2] = (byte) (((x ^ y) * 11) & 255);
            }
        }
        return new RasterImage(width, height, rgb);
    }

    @Test
    public void everyReferenceRunMatches() throws IOException {
        Set<String> families = new LinkedHashSet<>();
        List<String> mismatches = new ArrayList<>();
        for (Run run : readReference()) {
            families.add(run.name);
            try {
                check(run);
            } catch (AssertionError | RuntimeException e) {
                mismatches.add(e.toString());
            }
        }
        assertEquals(String.join("\n", mismatches), 0, mismatches.size());
        assertEquals("[lfm2, lfm2bicubic, step3vl, glm5v, youtuvl, lfm3vl, muse, dsocr, dsocr2, lightonocr, "
                + "llama4, gemma4v, gemma4uv, minicpmv46, minicpmocr, minicpmocr2, glmedge, janus, minimax, "
                + "sensenova, deepseek4v, deepseek41v, granite4]",
                families.toString());
    }

    private static void check(Run run) {
        String label = run.name + " " + run.width + "x" + run.height;
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(run.name)),
                ImageTokenBudget.none());
        PreparedImages out = ImagePreprocessors.forParams(params).prepare(pattern(run.width, run.height));
        assertEquals(label + " grid", run.gridX + "x" + run.gridY, out.gridX() + "x" + out.gridY());
        assertEquals(label + " overview", run.overview != null, out.hasOverview());
        assertEquals(label + " entries", run.entries.size(), out.tiles().size());
        if (run.overview != null) {
            checkTile(label + " ov", run.overview, out.overview());
        }
        for (int i = 0; i < run.entries.size(); i++) {
            checkTile(label + " e" + i, run.entries.get(i), out.tiles().get(i));
        }
        int total = 0;
        int slice = 0;
        for (int i = 0; i < out.encodedTiles().size(); i++) {
            Entry expected = out.isOverview(i) ? run.overview : run.entries.get(slice++);
            assertEquals(label + " tokens of encoded tile " + i, expected.tokens, out.tokensFor(i));
            total += expected.tokens;
        }
        assertEquals(label + " anyres", run.anyresTotal > 0, out.isAnyres());
        assertEquals(label + " total tokens", run.anyresTotal > 0 ? run.anyresTotal : total, out.totalTokens());
    }

    private static void checkTile(String label, Entry expected, ImageTile tile) {
        assertNotNull(label, tile);
        assertEquals(label + " size", expected.width + "x" + expected.height, tile.width() + "x" + tile.height());
        float[] v = tile.normalized(new float[] {0, 0, 0}, new float[] {1, 1, 1});
        double sum = 0;
        double weighted = 0;
        for (int i = 0; i < v.length; i++) {
            sum += v[i];
            weighted += v[i] * (double) ((i % 97) + 1);
        }
        assertClose(label + " sum", expected.sum, sum);
        assertClose(label + " weighted sum", expected.weightedSum, weighted);
    }

    /** The printed six decimals plus float rounding of a resample done in a different evaluation order. */
    private static void assertClose(String label, double expected, double actual) {
        double tolerance = 2e-6;
        assertTrue(label + ": expected " + expected + " but was " + actual, Math.abs(expected - actual) <= tolerance);
    }

    private static List<Run> readReference() throws IOException {
        List<Run> runs = new ArrayList<>();
        try (InputStream in = ReferencePreprocessingTest.class.getClassLoader().getResourceAsStream(REFERENCE)) {
            assertNotNull(REFERENCE, in);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            Run current = null;
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                if (!line.startsWith(" ")) {
                    current = Run.parse(line);
                    runs.add(current);
                } else if (line.trim().startsWith("ov ")) {
                    current.overview = Entry.parse(line);
                } else {
                    current.entries.add(Entry.parse(line));
                }
            }
        }
        return runs;
    }

    /** One preprocessor run of the reference file. */
    private static final class Run {
        String name;
        int width;
        int height;
        int gridX;
        int gridY;
        Entry overview;
        /** The token count of the assembled anyres image; 0 for every other run. */
        int anyresTotal;
        final List<Entry> entries = new ArrayList<>();

        static Run parse(String line) {
            String[] f = line.trim().split(" ");
            Run r = new Run();
            r.name = f[0];
            int[] size = pair(f[1]);
            r.width = size[0];
            r.height = size[1];
            int[] grid = pair(f[2].substring("grid=".length()));
            r.gridX = grid[0];
            r.gridY = grid[1];
            if (f.length > 4 && f[4].startsWith("total=")) {
                r.anyresTotal = Integer.parseInt(f[4].substring("total=".length()));
            }
            return r;
        }
    }

    /** One output image of a run: its size, token count and checksums. */
    private static final class Entry {
        int width;
        int height;
        int tokens;
        double sum;
        double weightedSum;

        static Entry parse(String line) {
            String[] f = line.trim().split(" ");
            Entry e = new Entry();
            int[] size = pair(f[1]);
            e.width = size[0];
            e.height = size[1];
            e.tokens = Integer.parseInt(f[2].substring("tokens=".length()));
            e.sum = Double.parseDouble(f[3].substring("sum=".length()));
            e.weightedSum = Double.parseDouble(f[4].substring("wsum=".length()));
            return e;
        }
    }

    private static int[] pair(String wxh) {
        int x = wxh.indexOf('x');
        return new int[] {Integer.parseInt(wxh.substring(0, x)), Integer.parseInt(wxh.substring(x + 1))};
    }
}
