package com.eonmux.neuralfree.gguf.mm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Encoder rows stored by an external reference run: a class-path text file of {@code #} comment lines and blocks,
 * each a header line ({@code <name> <source> ...}) followed by indented lines of space-separated floats, one per
 * stored row. Package-private test support.
 */
final class ReferenceRows {

    private ReferenceRows() {
    }

    /** The rows of the block whose header starts with {@code "<name> <source> "} in {@code resource}. */
    static float[][] read(String resource, String name, String source) throws IOException {
        String header = name + " " + source + " ";
        try (InputStream in = ReferenceRows.class.getClassLoader().getResourceAsStream(resource)) {
            assertNotNull(resource, in);
            BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            List<float[]> rows = new ArrayList<>();
            boolean inBlock = false;
            for (String line = r.readLine(); line != null; line = r.readLine()) {
                if (line.startsWith("#")) {
                    continue;
                }
                if (!line.startsWith(" ")) {
                    if (inBlock) {
                        break;
                    }
                    inBlock = line.startsWith(header);
                } else if (inBlock) {
                    rows.add(parse(line));
                }
            }
            assertFalse("no block '" + header.trim() + "' in " + resource, rows.isEmpty());
            return rows.toArray(new float[0][]);
        }
    }

    private static float[] parse(String line) {
        String[] parts = line.trim().split(" ");
        float[] row = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            row[i] = Float.parseFloat(parts[i]);
        }
        return row;
    }

    /**
     * Asserts that every value of {@code got} lies within {@code tolerance} times the largest magnitude of
     * {@code want}; {@code got} row {@code i * stride} is compared with {@code want} row {@code i}.
     */
    static void assertClose(String what, float[][] want, float[][] got, int stride, double tolerance) {
        assertEquals(what + " rows", want.length, (got.length + stride - 1) / stride);
        double scale = 0;
        for (float[] row : want) {
            for (float v : row) {
                scale = Math.max(scale, Math.abs(v));
            }
        }
        double worst = 0;
        for (int r = 0; r < want.length; r++) {
            float[] g = got[r * stride];
            assertEquals(what + " width", want[r].length, g.length);
            for (int c = 0; c < g.length; c++) {
                worst = Math.max(worst, Math.abs(want[r][c] - g[c]));
            }
        }
        assertTrue(what + ": largest difference " + worst + " of scale " + scale, worst <= tolerance * scale);
    }
}
