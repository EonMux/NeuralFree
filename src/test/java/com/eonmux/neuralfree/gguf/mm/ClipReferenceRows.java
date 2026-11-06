package com.eonmux.neuralfree.gguf.mm;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

/**
 * Encoder rows of llama.cpp's own clip for the synthetic mmproj fixtures of the {@code mm} tests, stored as test
 * resources ({@code clip-reference/<case>.f32}: int32 row count, int32 row width, then the rows as little-endian
 * floats). The rows come from a harness compiled outside the repository against llama.cpp's {@code libmtmd}: it
 * loads the fixture GGUF with {@code clip_init}, encodes the fixture's normalized pixels as one
 * {@code clip_image_f32} (with its {@code add_viewsep} and {@code lead_pad} flags) through
 * {@code clip_image_batch_encode} on the CPU backend, and writes the rows.
 *
 * <p>With the environment variable {@value #EXPORT_ENV} naming a directory, {@link #export} copies each case's
 * GGUF and pixels there and appends a line {@code <case> <width> <height> [flags]} to {@code cases.txt}, the input
 * list of the harness. Package-private test support.</p>
 */
final class ClipReferenceRows {

    /** Environment variable naming the directory the fixtures export to; unset in normal runs. */
    static final String EXPORT_ENV = "NF_CLIP_REFERENCE_EXPORT";

    /** Relative tolerance of the comparison, scaled by {@code max(1, |reference|)}. */
    static final double TOLERANCE = 1e-4;

    private ClipReferenceRows() {
    }

    /** The reference rows of {@code caseName}. */
    static float[][] load(String caseName) throws IOException {
        String res = "clip-reference/" + caseName + ".f32";
        try (InputStream in = ClipReferenceRows.class.getResourceAsStream(res)) {
            assertNotNull("missing reference resource " + res, in);
            byte[] all = in.readAllBytes();
            ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
            int n = bb.getInt();
            int w = bb.getInt();
            assertEquals("size of " + res, 8L + 4L * n * w, all.length);
            float[][] rows = new float[n][w];
            for (float[] row : rows) {
                for (int i = 0; i < w; i++) {
                    row[i] = bb.getFloat();
                }
            }
            return rows;
        }
    }

    /**
     * Writes the case's inputs for the harness when {@value #EXPORT_ENV} is set (the run then only exports, the
     * callers skip their comparisons); does nothing otherwise.
     *
     * @return whether the case was exported
     */
    static boolean export(String caseName, Path gguf, ClipImage image, String flags) throws IOException {
        String dir = System.getenv(EXPORT_ENV);
        if (dir == null || dir.isEmpty()) {
            return false;
        }
        Path out = Paths.get(dir);
        Files.createDirectories(out);
        Files.copy(gguf, out.resolve(caseName + ".gguf"), StandardCopyOption.REPLACE_EXISTING);
        float[] px = image.pixels();
        ByteBuffer bb = ByteBuffer.allocate(px.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : px) {
            bb.putFloat(v);
        }
        Files.write(out.resolve(caseName + ".px"), bb.array());
        String line = String.format(Locale.ROOT, "%s %d %d %s%n", caseName, image.width(), image.height(),
                flags == null ? "" : flags);
        Files.write(out.resolve("cases.txt"), line.getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE,
                StandardOpenOption.APPEND);
        return true;
    }

    /** Asserts {@code got} matches the reference rows of {@code caseName} within {@link #TOLERANCE}. */
    static void assertMatches(String caseName, float[][] got) throws IOException {
        float[][] want = load(caseName);
        assertEquals(caseName + " rows", want.length, got.length);
        double worst = 0;
        int worstRow = -1;
        int worstCol = -1;
        for (int t = 0; t < want.length; t++) {
            assertEquals(caseName + " row width", want[t].length, got[t].length);
            for (int i = 0; i < want[t].length; i++) {
                double err = Math.abs(got[t][i] - want[t][i]) / Math.max(1.0, Math.abs(want[t][i]));
                if (!(err <= worst)) {
                    worst = err;
                    worstRow = t;
                    worstCol = i;
                }
            }
        }
        if (!(worst <= TOLERANCE)) {
            fail(String.format(Locale.ROOT, "%s: %d rows, worst relative error %.3g at row %d value %d: expected"
                            + " %.7g but was %.7g", caseName, want.length, worst, worstRow, worstCol,
                    want[worstRow][worstCol], got[worstRow][worstCol]));
        }
    }
}
