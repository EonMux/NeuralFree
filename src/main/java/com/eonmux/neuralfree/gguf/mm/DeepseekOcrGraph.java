package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The DeepSeek-OCR and DeepSeek-OCR 2 graphs (models/deepseekocr.cpp {@code clip_graph_deepseekocr},
 * models/deepseekocr2.cpp {@code clip_graph_deepseekocr2}), written once against {@link VisionOps}; the SAM
 * backbone runs through {@link VisionOps#sam}. Package-private executor support.
 *
 * <p>DeepSeek-OCR: per square tile, the SAM rows, then the CLIP tower over them (the class row first, the learned
 * positions added, quick-GELU FFN), the class row dropped, the CLIP rows concatenated with the SAM rows and
 * {@code mm.model.fc}. A global view ({@link ClipImage#viewSeparator()}) is one tile whose rows end with a newline
 * per token row and the view separator; any other image is a tile row (tiles stacked top to bottom), whose rows
 * interleave the tiles row by row with a newline after each token row.</p>
 *
 * <p>DeepSeek-OCR 2: the SAM rows of the 768 (144 rows) or 1024 (256 rows) image, then the learned queries of that
 * size appended, the Qwen2 encoder over both (RMS norm, SiLU-gated FFN, NEOX rope over positions {@code 0..},
 * base 1e6; the image rows attend to the image rows, each query to the image rows and the queries up to itself),
 * the query rows through {@code mm.model.fc}, and the view separator after the global view.</p>
 */
final class DeepseekOcrGraph {

    /** The SAM downsample per side: two stride-2 convs. */
    static final int SAM_DOWNSAMPLE = 4;

    /** The NEOX rope base of the DeepSeek-OCR 2 encoder. */
    static final float QWEN2_ROPE_BASE = 1_000_000f;

    /** The queries of the 1024 global view (16 x 16 SAM rows). */
    static final int QUERIES_1024 = 256;

    /** The queries of a 768 tile (12 x 12 SAM rows). */
    static final int QUERIES_768 = 144;

    private DeepseekOcrGraph() {
    }

    /** True for the encoder graphs that run the SAM backbone. */
    static boolean runsSam(String encoderGraph) {
        return ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP.equals(encoderGraph)
                || ProjectorEntry.ENCODER_DEEPSEEKOCR2_SAM_QWEN2.equals(encoderGraph);
    }

    /** The tower rows of an image whose SAM grid is {@code side x side}: the class row or the queries added. */
    static int towerTokens(String encoderGraph, int side) {
        int n = side * side;
        return ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP.equals(encoderGraph) ? n + 1 : 2 * n;
    }

    /** The embedding rows of {@code image}. */
    static float[][] encode(VisionOps ops, ClipImage image) {
        if (ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP.equals(ops.model().encoderGraph())) {
            return image.viewSeparator() ? globalView(ops, image) : tileRow(ops, image);
        }
        return queries(ops, image);
    }

    // ------------------------------------------------------------------ DeepSeek-OCR

    private static float[][] globalView(VisionOps ops, ClipImage image) {
        float[][] rows = clipTile(ops, image);
        int side = side(rows.length);
        List<float[]> out = new ArrayList<>(rows.length + side + 1);
        float[] newline = ops.hostVector("v.image_newline");
        for (int y = 0; y < side; y++) {
            out.addAll(Arrays.asList(rows).subList(y * side, (y + 1) * side));
            out.add(newline.clone());
        }
        out.add(ops.hostVector("v.view_seperator").clone());
        return out.toArray(new float[0][]);
    }

    /** A row of tiles stacked top to bottom: token row {@code y} of every tile in turn, then a newline. */
    private static float[][] tileRow(VisionOps ops, ClipImage image) {
        int w = image.width();
        int h = image.height();
        if (h < w || h % w != 0) {
            throw new IllegalArgumentException("a DeepSeek-OCR tile row stacks square tiles top to bottom, got "
                    + w + "x" + h);
        }
        int nTiles = h / w;
        float[] pixels = image.pixels();
        int tileValues = w * w * 3;
        List<float[][]> tiles = new ArrayList<>(nTiles);
        for (int t = 0; t < nTiles; t++) {
            float[] px = Arrays.copyOfRange(pixels, t * tileValues, (t + 1) * tileValues);
            tiles.add(clipTile(ops, ClipImage.of(w, w, px)));
        }
        int side = side(tiles.get(0).length);
        float[] newline = ops.hostVector("v.image_newline");
        List<float[]> out = new ArrayList<>((side * nTiles + 1) * side);
        for (int y = 0; y < side; y++) {
            for (float[][] tile : tiles) {
                out.addAll(Arrays.asList(tile).subList(y * side, (y + 1) * side));
            }
            out.add(newline.clone());
        }
        return out.toArray(new float[0][]);
    }

    /** One square tile: SAM, the CLIP tower over the SAM rows, the concatenation and {@code mm.model.fc}. */
    private static float[][] clipTile(VisionOps ops, ClipImage tile) {
        VisionOps.Block sam = ops.sam(tile);
        int n = sam.n();
        int e = ops.hp().nEmbd();
        requireTowerWidth(ops, sam);
        ClipVisionModel model = ops.model();
        VisionOps.Block x = ops.insertRow(sam, 0, ops.hostWeight(model.classEmbd()));
        float[] table = ops.hostWeight(model.posEmbd());
        if (table.length != (n + 1) * e && model.posEmbd().type() != GGMLType.F32) {
            throw new IllegalArgumentException("the DeepSeek-OCR position table " + model.posEmbd().name()
                    + " resizes to " + side(n) + "x" + side(n) + " only in F32 (ggml_interpolate), it is "
                    + model.posEmbd().type());
        }
        ops.addHost(x, positions(table, e, side(n)));
        x = ops.tower(x, n + 1, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        int[] patches = new int[n];
        for (int t = 0; t < n; t++) {
            patches[t] = t + 1;
        }
        VisionOps.Block clip = ops.gatherRows(x, patches, e);
        VisionOps.Block cat = ops.concatChannels(Arrays.asList(clip, sam), n, e);
        VisionOps.Block out = ops.linear("mm.model.fc.weight", cat, "mm.model.fc.bias");
        return VisionRows.tokenRows(ops.download(out), model.outputDim());
    }

    /**
     * The learned position rows of a tower over {@code tgt * tgt} SAM rows and the class row, as the reference
     * graph builds them from the F32 table of {@code src * src + 1} rows: the table itself when {@code src == tgt};
     * otherwise the {@code ggml_interpolate} bicubic resize of its first {@code src * src} rows read as one
     * {@code n_embd x (src * src)} plane to {@code tgt x tgt}, that plane repeated {@code n_embd} times and read back
     * as {@code tgt * tgt} rows of {@code n_embd}, then the {@code n_embd} floats the graph's class-row view reads:
     * its {@code ggml_view_2d} offset is {@code src * src} bytes into the table.
     */
    static float[] positions(float[] table, int nEmbd, int tgt) {
        int rows = table.length / nEmbd;
        int src = (int) Math.round(Math.sqrt(rows - 1));
        if (src == tgt) {
            return table;
        }
        int grid = src * src;
        float[] plane = Glm4vGraph.resizeBicubic(Arrays.copyOf(table, grid * nEmbd), 1, nEmbd, grid, tgt, tgt);
        int area = tgt * tgt;
        float[] out = new float[(area + 1) * nEmbd];
        for (int f = 0; f < area * nEmbd; f++) {
            out[f] = plane[f % area];
        }
        ByteBuffer bytes = ByteBuffer.allocate(table.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(table);
        for (int c = 0; c < nEmbd; c++) {
            out[area * nEmbd + c] = bytes.getFloat(grid + c * Float.BYTES);
        }
        return out;
    }

    // ------------------------------------------------------------------ DeepSeek-OCR 2

    private static float[][] queries(VisionOps ops, ClipImage image) {
        VisionOps.Block sam = ops.sam(image);
        requireTowerWidth(ops, sam);
        int n = sam.n();
        int e = ops.hp().nEmbd();
        String name;
        if (n == QUERIES_1024) {
            name = "v.resample_query_1024.weight";
        } else if (n == QUERIES_768) {
            name = "v.resample_query_768.weight";
        } else {
            throw new IllegalArgumentException("the DeepSeek-OCR 2 encoder reads 768 or 1024 images (144 or 256 SAM"
                    + " rows), got " + image.width() + "x" + image.height() + " (" + n + " rows)");
        }
        float[] q = ops.hostVectorOrNull(name);
        if (q == null) {
            throw new IllegalArgumentException("the projector carries no " + name + " for a " + image.width() + "x"
                    + image.height() + " image");
        }
        float[] rows = Arrays.copyOf(ops.download(sam), 2 * n * e);
        System.arraycopy(q, 0, rows, n * e, n * e);
        int seq = 2 * n;
        int[] pos = new int[seq];
        for (int t = 0; t < seq; t++) {
            pos[t] = t;
        }
        VisionRows.KeyRanges mask = new VisionRows.KeyRanges() {
            @Override
            public int begin(int row) {
                return 0;
            }

            @Override
            public int end(int row) {
                return row < n ? n : row + 1;
            }
        };
        VisionOps.Block x = ops.tower(ops.upload(rows, e), seq, NormKind.RMS, VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.neox(Collections.singletonList(
                        new VisionOps.NeoxSegment(0, ops.model().dHead(), pos)), QWEN2_ROPE_BASE))
                .withWindows(il -> mask));
        int[] tail = new int[n];
        for (int t = 0; t < n; t++) {
            tail[t] = n + t;
        }
        VisionOps.Block out = ops.linear("mm.model.fc.weight", ops.gatherRows(x, tail, e), "mm.model.fc.bias");
        float[][] tokens = VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
        if (!image.viewSeparator()) {
            return tokens;
        }
        float[][] framed = Arrays.copyOf(tokens, n + 1);
        framed[n] = ops.hostVector("v.view_seperator").clone();
        return framed;
    }

    // ------------------------------------------------------------------ helpers

    private static void requireTowerWidth(VisionOps ops, VisionOps.Block sam) {
        if (sam.width() != ops.hp().nEmbd()) {
            throw new IllegalStateException("the SAM backbone outputs " + sam.width() + " channels, the tower reads "
                    + ops.hp().nEmbd());
        }
    }

    private static int side(int n) {
        int side = (int) Math.round(Math.sqrt(n));
        if (side * side != n) {
            throw new IllegalStateException(n + " rows are not a square grid");
        }
        return side;
    }
}
