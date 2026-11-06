package com.eonmux.neuralfree.gguf.mm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.erf;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.geluTanh;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.glu;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.linear;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.orNull;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.sigmoid;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.toFloat;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.unary;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.attention;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.norm;

/**
 * Independent double-precision transcriptions of the llama.cpp clip graphs of the ldp, ldpv2, glm-edge, cogvlm,
 * glm4v, kimivl, lfm2, d1omni_v, dots_ocr, youtuvl and granite4_vision families (tools/mtmd/models/{llava,cogvlm,
 * glm4v,kimivl,siglip,dotsocr,youtuvl,granite4-vision}.cpp, the graph inputs clip.cpp fills, and the Hugging Face
 * modules they port), written from the reference graphs, not from the executor. Everything runs in
 * {@code double}; results narrow once at the end.
 */
final class VisionFamilyReferenceB {

    private VisionFamilyReferenceB() {
    }

    /** The per-call shape of the shared tower: norm type, rotary positions, per-layer masks, block layout. */
    static final class Tower {
        boolean rms;
        VisionFamilyReference.Rope rope;
        double[][][] masks;
        boolean postNormLayout;
        int nLayers = -1;
        String postNorm = "v.post_ln.weight";
        /** Filled with the stream after every block when non-null. */
        List<double[][]> taps;
    }

    // ================================================================== llava graph projectors

    /** The llava graph's patch rows: the tower, the stack, and rows 1.. when a class token exists. */
    static double[][] llavaRows(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        double[][] x = MmprojFixture.Reference.embed(f, hp, px, width, width);
        x = MmprojFixture.Reference.llavaTower(f, hp, x);
        int side = width / hp.patchSize();
        if (hp.projectorType() == ProjectorType.GLM_EDGE) {
            return x;
        }
        int off = f.has("v.class_embd") ? 1 : 0;
        return Arrays.copyOfRange(x, off, off + side * side);
    }

    static float[][] ldp(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        int side = width / hp.patchSize();
        double[][] h = mlp(llavaRows(f, hp, px, width), f, "mm.model.mlp.1", "mm.model.mlp.3", false);
        double[][] b1 = mobileBlock(f, hp, "mm.model.mb_block.1.block.", h, side, 1);
        for (int t = 0; t < h.length; t++) {
            for (int c = 0; c < h[t].length; c++) {
                b1[t][c] += h[t][c];
            }
        }
        return toFloat(mobileBlock(f, hp, "mm.model.mb_block.2.block.", b1, side, 2));
    }

    private static double[][] mobileBlock(MmprojFixture f, ClipHparams hp, String p, double[][] x, int side,
                                          int stride) {
        int out = (side - 1) / stride + 1;
        double[][] cur = depthwise(x, side, f.get(p + "0.0.weight"), stride);
        cur = norm(cur, f.get(p + "0.1.weight"), f.get(p + "0.1.bias"), false, hp.eps());
        int d = cur[0].length;
        double[] pooled = new double[d];
        for (double[] row : cur) {
            for (int c = 0; c < d; c++) {
                row[c] = row[c] * Math.min(Math.max(row[c] + 3, 0), 6) / 6;
                pooled[c] += row[c] / (out * out);
            }
        }
        double[][] se = new double[1][(int) f.shapes.get(p + "1.fc1.weight")[1]];
        linear(new double[][]{pooled}, f.get(p + "1.fc1.weight"), f.get(p + "1.fc1.bias"), se);
        for (int i = 0; i < se[0].length; i++) {
            se[0][i] = Math.max(se[0][i], 0);
        }
        double[][] gate = new double[1][d];
        linear(se, f.get(p + "1.fc2.weight"), f.get(p + "1.fc2.bias"), gate);
        for (double[] row : cur) {
            for (int c = 0; c < d; c++) {
                row[c] *= Math.min(Math.max(gate[0][c] + 3, 0), 6) / 6;
            }
        }
        double[][] pw = new double[cur.length][d];
        linear(cur, f.get(p + "2.0.weight"), null, pw);
        return norm(pw, f.get(p + "2.1.weight"), f.get(p + "2.1.bias"), false, hp.eps());
    }

    /** Depthwise 3x3 conv, zero padding 1: {@code y[c] = sum w[c][ky][kx] * x[c](oy*s + ky - 1, ox*s + kx - 1)}. */
    private static double[][] depthwise(double[][] x, int side, float[] w, int stride) {
        int out = (side - 1) / stride + 1;
        int d = x[0].length;
        double[][] y = new double[out * out][d];
        for (int oy = 0; oy < out; oy++) {
            for (int ox = 0; ox < out; ox++) {
                for (int c = 0; c < d; c++) {
                    double s = 0;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = oy * stride + ky - 1;
                            int ix = ox * stride + kx - 1;
                            if (iy >= 0 && iy < side && ix >= 0 && ix < side) {
                                s += w[c * 9 + ky * 3 + kx] * x[iy * side + ix][c];
                            }
                        }
                    }
                    y[oy * out + ox][c] = s;
                }
            }
        }
        return y;
    }

    static float[][] ldpv2(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        int side = width / hp.patchSize();
        double[][] h = mlp(llavaRows(f, hp, px, width), f, "mm.model.mlp.0", "mm.model.mlp.2", false);
        int half = side / 2;
        int d = h[0].length;
        double[][] pooled = new double[half * half][d];
        for (int t = 0; t < side * side; t++) {
            int y = t / side;
            int xx = t % side;
            for (int c = 0; c < d; c++) {
                pooled[(y / 2) * half + xx / 2][c] += h[t][c] / 4;
            }
        }
        double[][] peg = depthwise(pooled, half, f.get("mm.model.peg.0.weight"), 1);
        float[] bias = f.get("mm.model.peg.0.bias");
        for (int t = 0; t < peg.length; t++) {
            for (int c = 0; c < d; c++) {
                peg[t][c] += bias[c] + pooled[t][c];
            }
        }
        return toFloat(peg);
    }

    static float[][] glmEdge(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        double[][] x = llavaRows(f, hp, px, width);
        int side = width / hp.patchSize();
        int e = x[0].length;
        float[] w = f.get("adapter.conv.weight");
        float[] b = f.get("adapter.conv.bias");
        int hc = b.length;
        int half = side / 2;
        double[][] conv = new double[half * half][hc];
        for (int oy = 0; oy < half; oy++) {
            for (int ox = 0; ox < half; ox++) {
                for (int o = 0; o < hc; o++) {
                    double s = b[o];
                    for (int c = 0; c < e; c++) {
                        for (int ky = 0; ky < 2; ky++) {
                            for (int kx = 0; kx < 2; kx++) {
                                s += w[((o * e + c) * 2 + ky) * 2 + kx] * x[(2 * oy + ky) * side + 2 * ox + kx][c];
                            }
                        }
                    }
                    conv[oy * half + ox][o] = s;
                }
            }
        }
        double[][] h = matmul(conv, f, "adapter.linear.linear.weight", null);
        h = norm(h, f.get("adapter.linear.norm1.weight"), f.get("adapter.linear.norm1.bias"), false, hp.eps());
        apply(h, VisionFamilyReferenceB::gelu);
        double[][] g = swiglu(matmul(h, f, "adapter.linear.gate.weight", null),
                matmul(h, f, "adapter.linear.dense_h_to_4h.weight", null));
        return framed(f, "adapter.boi", matmul(g, f, "adapter.linear.dense_4h_to_h.weight", null), "adapter.eoi");
    }

    // ================================================================== cogvlm, glm4v

    static float[][] cogvlm(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        int side = width / hp.patchSize();
        double[][] x = MmprojFixture.Reference.embed(f, hp, px, width, width);
        Tower t = new Tower();
        t.postNormLayout = true;
        t.postNorm = null;
        x = tower(f, hp, x, t);
        x = Arrays.copyOf(x, side * side);
        double[][] h = matmul(x, f, "mm.model.fc.weight", null);
        h = norm(h, f.get("mm.post_fc_norm.weight"), f.get("mm.post_fc_norm.bias"), false, 1e-5);
        apply(h, VisionFamilyReferenceB::gelu);
        double[][] g = swiglu(matmul(h, f, "mm.gate.weight", null), matmul(h, f, "mm.up.weight", null));
        return framed(f, "v.boi", matmul(g, f, "mm.down.weight", null), "v.eoi");
    }

    static float[][] glm4v(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        int n = pw * ph;
        int e = hp.nEmbd();
        int[] order = mergeOrder(pw, ph);
        double[][] a = convPatches(f, "v.patch_embd.weight", hp, px, width, pw, ph);
        double[][] b = convPatches(f, "v.patch_embd.weight.1", hp, px, width, pw, ph);
        float[] bias = f.get("v.patch_embd.bias");
        double[][] x = new double[n][e];
        int[] rowPos = new int[n];
        int[] colPos = new int[n];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < e; c++) {
                x[t][c] = a[order[t]][c] + b[order[t]][c] + bias[c];
            }
            rowPos[t] = order[t] / pw;
            colPos[t] = order[t] % pw;
        }
        if (f.has("v.norm_embd.weight")) {
            x = norm(x, f.get("v.norm_embd.weight"), f.get("v.norm_embd.bias"), true, hp.eps());
        }
        if (f.has("v.position_embd.weight")) {
            double[][] pos = bicubic(f.get("v.position_embd.weight"), e, 3, pw, ph);
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < e; c++) {
                    x[t][c] += pos[order[t]][c];
                }
            }
        }
        Tower tw = new Tower();
        tw.rms = true;
        tw.rope = q -> qwenRope(q, rowPos, colPos, hp.dHead());
        x = tower(f, hp, x, tw);
        int eo = f.get("mm.patch_merger.bias").length;
        float[] mw = f.get("mm.patch_merger.weight");
        double[][] merged = new double[n / 4][eo];
        for (int g = 0; g < n / 4; g++) {
            for (int o = 0; o < eo; o++) {
                double s = f.get("mm.patch_merger.bias")[o];
                for (int c = 0; c < e; c++) {
                    for (int r = 0; r < 4; r++) {
                        s += mw[(o * e + c) * 4 + r] * x[g * 4 + r][c];
                    }
                }
                merged[g][o] = s;
            }
        }
        double[][] h = matmul(merged, f, "mm.model.fc.weight", null);
        h = norm(h, f.get("mm.post_norm.weight"), orNull(f, "mm.post_norm.bias"), false, 1e-5);
        apply(h, VisionFamilyReferenceB::geluErf);
        double[][] gate = matmul(h, f, "mm.gate.weight", "mm.gate.bias");
        double[][] up = matmul(h, f, "mm.up.weight", "mm.up.bias");
        for (int t = 0; t < up.length; t++) {
            for (int i = 0; i < up[t].length; i++) {
                up[t][i] = glu(hp.ffnOp(), gate[t][i], up[t][i]);
            }
        }
        return toFloat(matmul(up, f, "mm.down.weight", "mm.down.bias"));
    }

    /**
     * PyTorch {@code interpolate(mode="bicubic", align_corners=False)} of the square grid: cubic convolution
     * with {@code A = -0.75} over the four neighbours of {@code (i + 0.5) / scale - 0.5}, indices clamped.
     */
    static double[][] bicubic(float[] pos, int e, int side, int width, int height) {
        double[][] out = new double[width * height][e];
        for (int oy = 0; oy < height; oy++) {
            double sy = (oy + 0.5) * side / height - 0.5;
            int y0 = (int) Math.floor(sy);
            for (int ox = 0; ox < width; ox++) {
                double sx = (ox + 0.5) * side / width - 0.5;
                int x0 = (int) Math.floor(sx);
                for (int j = -1; j <= 2; j++) {
                    double wy = cubic(sy - (y0 + j));
                    int iy = Math.max(0, Math.min(side - 1, y0 + j));
                    for (int i = -1; i <= 2; i++) {
                        double wgt = wy * cubic(sx - (x0 + i));
                        int ix = Math.max(0, Math.min(side - 1, x0 + i));
                        for (int c = 0; c < e; c++) {
                            out[oy * width + ox][c] += wgt * pos[(iy * side + ix) * e + c];
                        }
                    }
                }
            }
        }
        return out;
    }

    private static double cubic(double d) {
        double a = -0.75;
        double x = Math.abs(d);
        if (x <= 1) {
            return ((a + 2) * x - (a + 3)) * x * x + 1;
        }
        return x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0;
    }

    // ================================================================== kimivl, lfm2, dots_ocr

    static float[][] kimivl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        int e = hp.nEmbd();
        double[][] x = resizedSiglipInput(f, hp, px, width, pw, ph);
        int[] posH = new int[pw * ph];
        int[] posW = new int[pw * ph];
        for (int t = 0; t < posH.length; t++) {
            posH[t] = t / pw;
            posW[t] = t % pw;
        }
        Tower tw = new Tower();
        tw.rope = q -> VisionFamilyReference.rope2d(q, posW, posH, hp.dHead(), 10000.0, false);
        x = tower(f, hp, x, tw);
        double[][] merged = VisionFamilyReference.spaceToDepth(x, pw, ph, hp.nMerge(), false);
        merged = chunkNorm(merged, e, f.get("mm.input_norm.weight"), f.get("mm.input_norm.bias"), 1e-5);
        return toFloat(mlp(merged, f, "mm.1", "mm.2", false));
    }

    static float[][] lfm2(MmprojFixture f, ClipHparams hp, float[] px, int width, int height, boolean erfGelu) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = tower(f, hp, resizedSiglipInput(f, hp, px, width, pw, ph), new Tower());
        double[][] merged = VisionFamilyReference.spaceToDepth(x, pw, ph, hp.nMerge(), false);
        if (f.has("mm.input_norm.weight")) {
            merged = norm(merged, f.get("mm.input_norm.weight"), null, false, 1e-5);
        }
        float[] bias = orNull(f, "mm.input_norm.bias");
        if (bias != null) {
            for (double[] row : merged) {
                for (int c = 0; c < row.length; c++) {
                    row[c] += bias[c];
                }
            }
        }
        return toFloat(mlp(merged, f, "mm.1", "mm.2", erfGelu));
    }

    static float[][] dotsOcr(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = VisionFamilyReference.patches(f, hp, px, width, pw, ph);
        int[] rowPos = new int[pw * ph];
        int[] colPos = new int[pw * ph];
        for (int t = 0; t < rowPos.length; t++) {
            rowPos[t] = t / pw;
            colPos[t] = t % pw;
        }
        Tower tw = new Tower();
        tw.rms = true;
        tw.rope = q -> qwenRope(q, rowPos, colPos, hp.dHead());
        tw.postNorm = "mm.post_norm.weight";
        x = tower(f, hp, x, tw);
        x = norm(x, f.get("mm.input_norm.weight"), f.get("mm.input_norm.bias"), false, 1e-6);
        double[][] merged = VisionFamilyReference.spaceToDepth(x, pw, ph, hp.nMerge(), false);
        return toFloat(mlp(merged, f, "mm.0", "mm.2", true));
    }

    private static double[][] resizedSiglipInput(MmprojFixture f, ClipHparams hp, float[] px, int width, int pw,
                                                 int ph) {
        double[][] x = VisionFamilyReference.patches(f, hp, px, width, pw, ph);
        int e = hp.nEmbd();
        float[] pos = f.get("v.position_embd.weight");
        int side = (int) Math.round(Math.sqrt(pos.length / (double) e));
        double[][] p = side == pw && side == ph ? VisionFamilyReference.rows(pos, pw * ph, e)
                : VisionFamilyReference.resizeAntialiasBilinear(pos, e, side, pw, ph);
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < e; c++) {
                x[t][c] += p[t][c];
            }
        }
        return x;
    }

    // ================================================================== youtuvl

    static float[][] youtuvl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height, int[] fullAttn) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        int n = pw * ph;
        int e = hp.nEmbd();
        int[] order = mergeOrder(pw, ph);
        float[] w = f.get("v.patch_embd.weight");
        float[] bias = f.get("v.patch_embd.bias");
        double[][] x = new double[n][e];
        for (int t = 0; t < n; t++) {
            int y0 = order[t] / pw * ps;
            int x0 = order[t] % pw * ps;
            for (int c = 0; c < e; c++) {
                double s = bias[c];
                for (int ky = 0; ky < ps; ky++) {
                    for (int kx = 0; kx < ps; kx++) {
                        for (int ic = 0; ic < 3; ic++) {
                            s += w[c * 3 * ps * ps + (ky * ps + kx) * 3 + ic]
                                    * px[((y0 + ky) * width + x0 + kx) * 3 + ic];
                        }
                    }
                }
                x[t][c] = s;
            }
        }
        int[] rowPos = new int[n];
        int[] colPos = new int[n];
        for (int t = 0; t < n; t++) {
            rowPos[t] = order[t] / pw;
            colPos[t] = order[t] % pw;
        }
        Tower tw = new Tower();
        int[] idx = null;
        if (fullAttn.length > 0) {
            int mw = pw / 2;
            int mh = ph / 2;
            int window = hp.attnWindowSize() / ps / 2;
            idx = new int[mw * mh];
            int[] inv = new int[mw * mh];
            double[][] mask = windows(mw, mh, window, idx, inv);
            x = gatherBlocks(x, inv, 4);
            int[] r = new int[n];
            int[] cl = new int[n];
            for (int p = 0; p < n; p++) {
                int remap = idx[p / 4] * 4 + p % 4;
                r[remap] = rowPos[p];
                cl[remap] = colPos[p];
            }
            System.arraycopy(r, 0, rowPos, 0, n);
            System.arraycopy(cl, 0, colPos, 0, n);
            tw.masks = new double[hp.nLayer()][][];
            for (int il = 0; il < hp.nLayer(); il++) {
                tw.masks[il] = Arrays.binarySearch(fullAttn, il) >= 0 ? null : mask;
            }
        }
        tw.rope = q -> qwenRope(q, rowPos, colPos, hp.dHead());
        tw.postNorm = null;
        x = tower(f, hp, x, tw);
        if (idx != null) {
            x = gatherBlocks(x, idx, 4);
        }
        x = norm(x, f.get("v.post_ln.weight"), f.get("v.post_ln.bias"), false, hp.eps());
        x = norm(x, f.get("mm.input_norm.weight"), null, true, 1e-6);
        double[][] merged = new double[n / 4][4 * e];
        for (int t = 0; t < n; t++) {
            System.arraycopy(x[t], 0, merged[t / 4], (t % 4) * e, e);
        }
        return toFloat(mlp(merged, f, "mm.0", "mm.2", false));
    }

    /** clip.cpp's window plan: merged tokens grouped window by window, and the block-diagonal patch mask. */
    private static double[][] windows(int mw, int mh, int window, int[] idx, int[] inv) {
        int nRows = mw * mh * 4;
        double[][] mask = new double[nRows][nRows];
        for (double[] row : mask) {
            Arrays.fill(row, Double.NEGATIVE_INFINITY);
        }
        int dst = 0;
        int row = 0;
        for (int y = 0; y < mh; y += window) {
            for (int xx = 0; xx < mw; xx += window) {
                int h = Math.min(window, mh - y);
                int w = Math.min(window, mw - xx);
                int dst0 = dst;
                for (int dy = 0; dy < h; dy++) {
                    for (int dx = 0; dx < w; dx++) {
                        int src = (y + dy) * mw + xx + dx;
                        idx[src] = dst;
                        inv[dst] = src;
                        dst++;
                    }
                }
                for (int r = 0; r < h * w * 4; r++) {
                    Arrays.fill(mask[row++], dst0 * 4, dst * 4, 0.0);
                }
            }
        }
        return mask;
    }

    // ================================================================== granite4_vision

    /** One tile through the tower and every WindowQFormer, the streams concatenated along the channels. */
    static double[][] granite4Tile(MmprojFixture f, ClipHparams hp, float[] px, int width) {
        int side = width / hp.patchSize();
        int e = hp.nEmbd();
        double[][] x = VisionFamilyReference.patches(f, hp, px, width, side, side);
        float[] pos = f.get("v.position_embd.weight");
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < e; c++) {
                x[t][c] += pos[t * e + c];
            }
        }
        Tower tw = new Tower();
        tw.taps = new ArrayList<>();
        tw.postNorm = null;
        tower(f, hp, x, tw);
        int[] layers = hp.featureLayers();
        int[] offsets = hp.visionFamily().spatialOffsets();
        List<double[][]> streams = new ArrayList<>();
        for (int bid = 0; bid < layers.length; bid++) {
            streams.add(qformer(f, hp, bid, tw.taps.get(layers[bid]), side, offsets[bid]));
        }
        int p = streams.get(0)[0].length;
        double[][] out = new double[streams.get(0).length][p * streams.size()];
        for (int b = 0; b < streams.size(); b++) {
            for (int t = 0; t < out.length; t++) {
                System.arraycopy(streams.get(b)[t], 0, out[t], b * p, p);
            }
        }
        return out;
    }

    private static double[][] qformer(MmprojFixture f, ClipHparams hp, int bid, double[][] h, int side,
                                      int offset) {
        String p = "v.proj_blk." + bid + ".";
        int e = hp.nEmbd();
        int win = hp.visionFamily().windowSide();
        int qs = hp.visionFamily().querySide();
        int n = side / win;
        int newSide = n * qs;
        double[][] x = norm(h, f.get(p + "norm.weight"), f.get(p + "norm.bias"), false, hp.eps());
        double[][] down = new double[newSide * newSide][];
        int k = side / newSide;
        for (int y = 0; y < newSide; y++) {
            for (int xx = 0; xx < newSide; xx++) {
                if (offset >= 0) {
                    down[y * newSide + xx] = x[(2 * y + (offset >> 1 & 1)) * side + 2 * xx + (offset & 1)].clone();
                } else {
                    double[] acc = new double[e];
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            for (int c = 0; c < e; c++) {
                                acc[c] += x[(y * k + ky) * side + xx * k + kx][c] / (k * k);
                            }
                        }
                    }
                    down[y * newSide + xx] = acc;
                }
            }
        }
        double[][] out = new double[newSide * newSide][];
        for (int wy = 0; wy < n; wy++) {
            for (int wx = 0; wx < n; wx++) {
                double[][] enc = new double[win * win][];
                for (int i = 0; i < win * win; i++) {
                    enc[i] = add(x[(wy * win + i / win) * side + wx * win + i % win], f.get(p + "img_pos"), i);
                }
                double[][] q = new double[qs * qs][];
                for (int i = 0; i < qs * qs; i++) {
                    q[i] = add(down[(wy * qs + i / qs) * newSide + wx * qs + i % qs], f.get(p + "query"), i);
                }
                double[][] res = qformerWindow(f, p, q, enc);
                for (int i = 0; i < qs * qs; i++) {
                    out[(wy * qs + i / qs) * newSide + wx * qs + i % qs] = res[i];
                }
            }
        }
        return matmul(out, f, p + "linear.weight", p + "linear.bias");
    }

    /** One window of the one-layer QFormer: post-norm self-attention, cross-attention and GELU-erf FFN. */
    private static double[][] qformerWindow(MmprojFixture f, String p, double[][] q, double[][] enc) {
        double eps = 1e-12;
        int nHead = q[0].length / 64;
        q = norm(q, f.get(p + "post_norm.weight"), f.get(p + "post_norm.bias"), false, eps);
        double[][] sa = matmul(attention(matmul(q, f, p + "self_attn_q.weight", p + "self_attn_q.bias"),
                matmul(q, f, p + "self_attn_k.weight", p + "self_attn_k.bias"),
                matmul(q, f, p + "self_attn_v.weight", p + "self_attn_v.bias"), nHead, 64, null),
                f, p + "self_attn_out.weight", p + "self_attn_out.bias");
        sa = norm(sum(sa, q), f.get(p + "self_attn_norm.weight"), f.get(p + "self_attn_norm.bias"), false, eps);
        double[][] ca = matmul(attention(matmul(sa, f, p + "cross_attn_q.weight", p + "cross_attn_q.bias"),
                matmul(enc, f, p + "cross_attn_k.weight", p + "cross_attn_k.bias"),
                matmul(enc, f, p + "cross_attn_v.weight", p + "cross_attn_v.bias"), nHead, 64, null),
                f, p + "cross_attn_out.weight", p + "cross_attn_out.bias");
        ca = norm(sum(ca, sa), f.get(p + "cross_attn_norm.weight"), f.get(p + "cross_attn_norm.bias"), false, eps);
        double[][] up = matmul(ca, f, p + "ffn_up.weight", p + "ffn_up.bias");
        apply(up, VisionFamilyReferenceB::geluErf);
        double[][] ffn = sum(matmul(up, f, p + "ffn_down.weight", p + "ffn_down.bias"), ca);
        return norm(ffn, f.get(p + "ffn_norm.weight"), f.get(p + "ffn_norm.bias"), false, eps);
    }

    /**
     * The llava-next anyres layout: the overview tile, then the tile grid read over the whole tiled image,
     * cropped to the original aspect ratio, a newline row after every row.
     */
    static float[][] granite4Anyres(MmprojFixture f, ClipHparams hp, float[][] tilePixels, int width, int gridX,
                                    int gridY, int origW, int origH) {
        List<double[][]> tiles = new ArrayList<>();
        for (float[] px : tilePixels) {
            tiles.add(granite4Tile(f, hp, px, width));
        }
        int out = (int) Math.round(Math.sqrt(tiles.get(0).length));
        int curW = gridX * out;
        int curH = gridY * out;
        int x0 = 0;
        int y0 = 0;
        int w = curW;
        int h = curH;
        double origAspect = (double) origW / origH;
        if (origAspect > (double) curW / curH) {
            int newH = (int) Math.floor((double) origH * curW / origW + 1e-7);
            y0 = (curH - newH) / 2;
            h = curH - 2 * y0;
        } else {
            int newW = (int) Math.floor((double) origW * curH / origH + 1e-7);
            x0 = (curW - newW) / 2;
            w = curW - 2 * x0;
        }
        List<double[]> rows = new ArrayList<>(Arrays.asList(tiles.get(0)));
        float[] nl = f.get("v.image_newline");
        int dim = tiles.get(0)[0].length;
        for (int y = y0; y < y0 + h; y++) {
            for (int xx = x0; xx < x0 + w; xx++) {
                rows.add(tiles.get(1 + (y / out) * gridX + xx / out)[(y % out) * out + xx % out]);
            }
            double[] newline = new double[dim];
            for (int c = 0; c < dim; c++) {
                newline[c] = nl[c % nl.length];
            }
            rows.add(newline);
        }
        return toFloat(rows.toArray(new double[0][]));
    }

    // ================================================================== the tower

    /** build_vit (or the CogVLM post-norm layout) over {@code x}: pre-norm, blocks, optional post-norm. */
    static double[][] tower(MmprojFixture f, ClipHparams hp, double[][] x, Tower o) {
        if (f.has("v.pre_ln.weight")) {
            x = norm(x, f.get("v.pre_ln.weight"), orNull(f, "v.pre_ln.bias"), o.rms, hp.eps());
        }
        int nLayers = o.nLayers < 0 ? hp.nLayer() : o.nLayers;
        for (int il = 0; il < nLayers; il++) {
            x = block(f, hp, il, x, o, o.masks == null ? null : o.masks[il]);
            if (o.taps != null) {
                o.taps.add(x);
            }
        }
        if (o.postNorm != null && f.has(o.postNorm)) {
            x = norm(x, f.get(o.postNorm), orNull(f, "v.post_ln.bias"), o.rms, hp.eps());
        }
        return x;
    }

    private static double[][] block(MmprojFixture f, ClipHparams hp, int il, double[][] x, Tower o,
                                    double[][] mask) {
        String p = "v.blk." + il + ".";
        int e = hp.nEmbd();
        double[][] cur = o.postNormLayout ? x : norm(x, f.get(p + "ln1.weight"), orNull(f, p + "ln1.bias"), o.rms,
                hp.eps());
        double[][] q;
        double[][] k;
        double[][] v;
        if (f.has(p + "attn_qkv.weight")) {
            double[][] qkv = matmul(cur, f, p + "attn_qkv.weight", p + "attn_qkv.bias");
            q = new double[x.length][];
            k = new double[x.length][];
            v = new double[x.length][];
            for (int t = 0; t < x.length; t++) {
                q[t] = Arrays.copyOfRange(qkv[t], 0, e);
                k[t] = Arrays.copyOfRange(qkv[t], e, 2 * e);
                v[t] = Arrays.copyOfRange(qkv[t], 2 * e, 3 * e);
            }
        } else {
            q = matmul(cur, f, p + "attn_q.weight", p + "attn_q.bias");
            k = matmul(cur, f, p + "attn_k.weight", p + "attn_k.bias");
            v = matmul(cur, f, p + "attn_v.weight", p + "attn_v.bias");
        }
        if (o.rope != null) {
            o.rope.apply(q);
            o.rope.apply(k);
        }
        double[][] attn = matmul(attention(q, k, v, hp.nHead(), hp.dHead(), mask), f, p + "attn_out.weight",
                p + "attn_out.bias");
        if (o.postNormLayout) {
            attn = norm(attn, f.get(p + "ln1.weight"), orNull(f, p + "ln1.bias"), o.rms, hp.eps());
        }
        double[][] res = sum(attn, x);
        cur = o.postNormLayout ? res : norm(res, f.get(p + "ln2.weight"), orNull(f, p + "ln2.bias"), o.rms,
                hp.eps());
        double[][] up = matmul(cur, f, p + "ffn_up.weight", p + "ffn_up.bias");
        if (f.has(p + "ffn_gate.weight")) {
            double[][] gate = matmul(cur, f, p + "ffn_gate.weight", p + "ffn_gate.bias");
            for (int t = 0; t < up.length; t++) {
                for (int i = 0; i < up[t].length; i++) {
                    up[t][i] = glu(hp.ffnOp(), gate[t][i], up[t][i]);
                }
            }
        } else {
            apply(up, val -> unary(hp.ffnOp(), val));
        }
        double[][] ffn = matmul(up, f, p + "ffn_down.weight", p + "ffn_down.bias");
        if (o.postNormLayout) {
            ffn = norm(ffn, f.get(p + "ln2.weight"), orNull(f, p + "ln2.bias"), o.rms, hp.eps());
        }
        return sum(ffn, res);
    }

    /**
     * The qwen vision m-rope: pair {@code (j, j + d/2)} of every head rotates by the row position times
     * {@code 10000^(-4j/d)} for {@code j < d/4}, and by the column position times {@code 10000^(-4(j - d/4)/d)}
     * above (GGML_ROPE_TYPE_VISION restarts the exponent at each section).
     */
    static void qwenRope(double[][] x, int[] rowPos, int[] colPos, int dh) {
        int half = dh / 2;
        int quarter = dh / 4;
        for (int t = 0; t < x.length; t++) {
            for (int base = 0; base < x[t].length; base += dh) {
                for (int j = 0; j < half; j++) {
                    boolean row = j < quarter;
                    double angle = (row ? rowPos[t] : colPos[t])
                            * Math.pow(10000.0, -4.0 * (row ? j : j - quarter) / dh);
                    double a = x[t][base + j];
                    double b = x[t][base + j + half];
                    x[t][base + j] = a * Math.cos(angle) - b * Math.sin(angle);
                    x[t][base + j + half] = a * Math.sin(angle) + b * Math.cos(angle);
                }
            }
        }
    }

    // ================================================================== primitives

    /** Raster index of merge-group position {@code t}: 2x2 groups in raster order, raster inside a group. */
    static int[] mergeOrder(int pw, int ph) {
        int[] order = new int[pw * ph];
        int t = 0;
        for (int ty = 0; ty < ph / 2; ty++) {
            for (int tx = 0; tx < pw / 2; tx++) {
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        order[t++] = (ty * 2 + dy) * pw + tx * 2 + dx;
                    }
                }
            }
        }
        return order;
    }

    /** Raster conv patch rows of the kernel {@code name}, no bias. */
    private static double[][] convPatches(MmprojFixture f, String name, ClipHparams hp, float[] px, int width,
                                          int pw, int ph) {
        int ps = hp.patchSize();
        int e = hp.nEmbd();
        float[] w = f.get(name);
        double[][] x = new double[pw * ph][e];
        for (int t = 0; t < pw * ph; t++) {
            int y0 = t / pw * ps;
            int x0 = t % pw * ps;
            for (int c = 0; c < e; c++) {
                double s = 0;
                for (int ic = 0; ic < 3; ic++) {
                    for (int ky = 0; ky < ps; ky++) {
                        for (int kx = 0; kx < ps; kx++) {
                            s += w[((c * 3 + ic) * ps + ky) * ps + kx] * px[((y0 + ky) * width + x0 + kx) * 3 + ic];
                        }
                    }
                }
                x[t][c] = s;
            }
        }
        return x;
    }

    private static double[][] gatherBlocks(double[][] x, int[] idx, int block) {
        double[][] out = new double[x.length][];
        for (int b = 0; b < idx.length; b++) {
            for (int r = 0; r < block; r++) {
                out[b * block + r] = x[idx[b] * block + r];
            }
        }
        return out;
    }

    /** Layer norm of every {@code width}-wide chunk of every row. */
    private static double[][] chunkNorm(double[][] x, int width, float[] w, float[] b, double eps) {
        double[][] out = new double[x.length][x[0].length];
        for (int t = 0; t < x.length; t++) {
            for (int base = 0; base < x[t].length; base += width) {
                double[][] chunk = {Arrays.copyOfRange(x[t], base, base + width)};
                System.arraycopy(norm(chunk, w, b, false, eps)[0], 0, out[t], base, width);
            }
        }
        return out;
    }

    /** {@code down(gelu(up(x) + b) ) + b}, the tanh GELU or the erf GELU. */
    private static double[][] mlp(double[][] x, MmprojFixture f, String up, String down, boolean erfGelu) {
        double[][] h = matmul(x, f, up + ".weight", up + ".bias");
        apply(h, erfGelu ? VisionFamilyReferenceB::geluErf : VisionFamilyReferenceB::gelu);
        return matmul(h, f, down + ".weight", down + ".bias");
    }

    /** {@code x W^T + b} with the output width of {@code name}; a bias name that is absent adds nothing. */
    private static double[][] matmul(double[][] x, MmprojFixture f, String name, String bias) {
        long[] ne = f.shapes.get(name);
        double[][] out = new double[x.length][(int) ne[ne.length - 1]];
        linear(x, f.get(name), bias == null ? null : orNull(f, bias), out);
        return out;
    }

    private static double[][] swiglu(double[][] gate, double[][] up) {
        for (int t = 0; t < up.length; t++) {
            for (int i = 0; i < up[t].length; i++) {
                up[t][i] *= gate[t][i] * sigmoid(gate[t][i]);
            }
        }
        return up;
    }

    private static float[][] framed(MmprojFixture f, String first, double[][] rows, String last) {
        double[][] out = new double[rows.length + 2][];
        out[0] = toDouble(f.get(first));
        System.arraycopy(rows, 0, out, 1, rows.length);
        out[rows.length + 1] = toDouble(f.get(last));
        return toFloat(out);
    }

    private static double[] add(double[] row, float[] table, int i) {
        double[] out = row.clone();
        for (int c = 0; c < out.length; c++) {
            out[c] += table[i * out.length + c];
        }
        return out;
    }

    private static double[][] sum(double[][] a, double[][] b) {
        double[][] out = new double[a.length][a[0].length];
        for (int t = 0; t < a.length; t++) {
            for (int c = 0; c < a[t].length; c++) {
                out[t][c] = a[t][c] + b[t][c];
            }
        }
        return out;
    }

    private interface Op {
        double apply(double v);
    }

    private static void apply(double[][] x, Op op) {
        for (double[] row : x) {
            for (int i = 0; i < row.length; i++) {
                row[i] = op.apply(row[i]);
            }
        }
    }

    private static double gelu(double v) {
        return geluTanh(v);
    }

    private static double geluErf(double v) {
        return 0.5 * v * (1 + erf(v / Math.sqrt(2)));
    }

    private static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }
}
