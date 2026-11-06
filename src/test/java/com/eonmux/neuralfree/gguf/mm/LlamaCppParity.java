package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * The encoder rows llama.cpp's own clip graphs produce for the synthetic mmproj files of the families checked
 * against llama.cpp directly: each {@link Case} names a fixture, its metadata, the image size and the suffix
 * row type; the rows {@code clip_image_batch_encode} returned for it are stored in the resource
 * {@value #RESOURCE}, whose header records how they were produced. {@link #main} writes every case's mmproj file
 * and normalized image for that run. Package-private test support of the {@code mm} tests.
 */
final class LlamaCppParity {

    /** The stored llama.cpp rows, one block per case. */
    static final String RESOURCE = "clip-encoder-reference.txt";

    /** Tolerance of every value, relative to the largest magnitude of its row. */
    static final float REL_TOL = 1e-4f;

    /** One synthetic mmproj file and the image encoded with it. */
    static final class Case {
        final String name;
        final Supplier<MmprojFixture> fixture;
        final Supplier<Map<String, Object>> meta;
        final int width;
        final int height;
        /** The {@code clip_suffix_type} of the image entry (0: none). */
        final int suffix;

        Case(String name, Supplier<MmprojFixture> fixture, Supplier<Map<String, Object>> meta, int width, int height,
             int suffix) {
            this.name = name;
            this.fixture = fixture;
            this.meta = meta;
            this.width = width;
            this.height = height;
            this.suffix = suffix;
        }

        /** The case's mmproj file. */
        GgufTestWriter gguf() {
            return VisionFamilyFixtures.gguf(fixture.get(), meta.get());
        }

        /** The normalized image of the case, interleaved RGB in {@code [-1, 1)}. */
        float[] pixels() {
            return MmprojFixture.pixels(width, height, 1000L + 31L * width + height);
        }

        /** The image of the case with its suffix ({@link ClipSuffix} declares the {@code clip_suffix_type} order). */
        ClipImage image() {
            return ClipImage.of(width, height, pixels()).withSuffix(ClipSuffix.values()[suffix]);
        }
    }

    static final List<Case> CASES = Collections.unmodifiableList(Arrays.asList(
            // a 4x6 patch grid from the 3x2 position grid: bicubic resize, merged to 2x3
            new Case("kimik25", () -> VisionFamiliesDFixtures.kimik25(201), VisionFamiliesDFixtures::kimik25Meta,
                    8, 12, 0),
            // the 3x2 patch grid of the stored positions: no resize, the merge pads it to 4x2
            new Case("kimik25.stored_grid", () -> VisionFamiliesDFixtures.kimik25(202),
                    VisionFamiliesDFixtures::kimik25Meta, 6, 4, 0),
            // a 4x6 patch grid pooled 2x2 to 2x3
            new Case("gemma4v", () -> VisionFamiliesDFixtures.gemma4v(211),
                    () -> VisionFamiliesDFixtures.gemma4vMeta(2), 8, 12, 0),
            // the default 3x3 pool over a 6x3 grid
            new Case("gemma4v.pool3", () -> VisionFamiliesDFixtures.gemma4v(212),
                    () -> VisionFamiliesDFixtures.gemma4vMeta(null), 12, 6, 0),
            // 4x4 merged patches, a 4x3 grid
            new Case("gemma4uv", () -> VisionFamiliesDFixtures.gemma4uv(213), VisionFamiliesDFixtures::gemma4uvMeta,
                    16, 12, 0),
            // a 12x8 patch grid, the ViT merger after block 0 halves it to 6x4, the final merger to 3x2
            new Case("minicpmv4_6", () -> VisionFamiliesDFixtures.minicpmv46(221, true, false),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_6", null, null), 24, 16, 0),
            // the ViT merger after the last block of an 8x12 grid; the overview's </image><slice> rows follow
            new Case("minicpmv4_7.ov_slice", () -> VisionFamiliesDFixtures.minicpmv46(222, true, false),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_7", 4, 1), 16, 24,
                    ClipSuffix.MINICPMV_OV_SLICE.ordinal()),
            // one token from a 4x4 grid, then the </slice>\n<slice> rows of a row's last slice
            new Case("minicpmv4_7.row_end", () -> VisionFamiliesDFixtures.minicpmv46(223, true, false),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_7", 4, 0), 8, 8,
                    ClipSuffix.MINICPMV_ROW_END.ordinal()),
            // a 2x merge: no ViT merger, the final merger over the 6x4 grid; the blocks store their FFN swapped
            new Case("minicpm_ocr_1_0.merge2", () -> VisionFamiliesDFixtures.minicpmv46(224, false, true),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpm_ocr_1_0", 2, null), 12, 8, 0),
            // 6x4 patches in 3x2 merge units, a window of 5 rows each way
            new Case("mimovl", () -> VisionFamiliesDFixtures.mimovl(231),
                    () -> VisionFamiliesDFixtures.mimovlMeta(5), 12, 8, 0),
            // 4x8 patches in 2x4 merge units, a window of 2 rows each way
            new Case("mimovl.portrait", () -> VisionFamiliesDFixtures.mimovl(232),
                    () -> VisionFamiliesDFixtures.mimovlMeta(2), 8, 16, 0),
            // 64x32 pixels: the fusion features of 4x2 and 2x1 pixels meet at 4x2, no pool
            new Case("gemma3nv", () -> VisionFamiliesDFixtures.gemma3nv(251, true),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(16), 64, 32, 0),
            // 512x512 pixels: the 32x32 fusion map pools 2x2 down to 16x16
            new Case("gemma3nv.pool", () -> VisionFamiliesDFixtures.gemma3nv(252, true),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(512), 512, 512, 0),
            // three stages: the last block alone feeds the fusion adapter
            new Case("gemma3nv.three_stages", () -> VisionFamiliesDFixtures.gemma3nv(253, false),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(8), 32, 32, 0)));

    private static Map<String, float[][]> stored;

    private LlamaCppParity() {
    }

    static Case find(String name) {
        for (Case c : CASES) {
            if (c.name.equals(name)) {
                return c;
            }
        }
        throw new IllegalArgumentException("no llama.cpp parity case " + name);
    }

    /** The rows llama.cpp produced for case {@code name}. */
    static synchronized float[][] rows(String name) {
        if (stored == null) {
            stored = load();
        }
        float[][] r = stored.get(name);
        if (r == null) {
            throw new IllegalStateException(RESOURCE + " holds no rows for case " + name);
        }
        return r;
    }

    /** {@code got} against the llama.cpp rows of {@code what}, every value within {@link #REL_TOL}. */
    static void assertMatches(String what, float[][] want, float[][] got) {
        assertEquals(what + ": rows", want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(what + ": row " + t + " width", want[t].length, got[t].length);
            float scale = 0f;
            for (float v : want[t]) {
                scale = Math.max(scale, Math.abs(v));
            }
            for (int c = 0; c < want[t].length; c++) {
                float diff = Math.abs(want[t][c] - got[t][c]);
                if (!(diff <= REL_TOL * scale)) {
                    fail(what + ": row " + t + " col " + c + ": llama.cpp " + want[t][c] + ", NeuralFree "
                            + got[t][c] + " (row scale " + scale + ")");
                }
            }
        }
    }

    private static Map<String, float[][]> load() {
        InputStream in = LlamaCppParity.class.getResourceAsStream(RESOURCE);
        if (in == null) {
            throw new IllegalStateException("missing test resource " + RESOURCE);
        }
        Map<String, float[][]> m = new HashMap<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] head = line.trim().split(" ");
                if (head.length != 4 || !"case".equals(head[0])) {
                    throw new IllegalStateException(RESOURCE + ": expected 'case <name> <rows> <width>', got " + line);
                }
                m.put(head[1], block(r, Integer.parseInt(head[2]), Integer.parseInt(head[3])));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return m;
    }

    private static float[][] block(BufferedReader r, int n, int width) throws IOException {
        float[][] rows = new float[n][width];
        for (int t = 0; t < n; t++) {
            String[] v = r.readLine().trim().split(" ");
            if (v.length != width) {
                throw new IllegalStateException(RESOURCE + ": row " + t + " holds " + v.length + " values, not "
                        + width);
            }
            for (int c = 0; c < width; c++) {
                rows[t][c] = Float.parseFloat(v[c]);
            }
        }
        return rows;
    }

    /**
     * Writes every case to the directory {@code args[0]}: {@code <name>.gguf}, the image as little-endian floats
     * {@code <name>.f32}, and one line {@code <name> <width> <height> <suffix>} per case in {@code cases.txt}.
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: LlamaCppParity <output directory>");
        }
        Path dir = Files.createDirectories(Paths.get(args[0]));
        List<String> lines = new ArrayList<>();
        for (Case c : CASES) {
            c.gguf().writeTo(dir.resolve(c.name + ".gguf"));
            float[] px = c.pixels();
            ByteBuffer buf = ByteBuffer.allocate(px.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(px);
            Files.write(dir.resolve(c.name + ".f32"), buf.array());
            lines.add(c.name + " " + c.width + " " + c.height + " " + c.suffix);
        }
        try (Writer w = Files.newBufferedWriter(dir.resolve("cases.txt"), StandardCharsets.UTF_8)) {
            for (String l : lines) {
                w.write(l);
                w.write('\n');
            }
        }
    }
}
