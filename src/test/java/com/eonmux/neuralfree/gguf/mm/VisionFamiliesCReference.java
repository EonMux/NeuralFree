package com.eonmux.neuralfree.gguf.mm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.erf;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.linear;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.orNull;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.sigmoid;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.toFloat;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.unary;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.norm;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.patches;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.rows;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyReference.toDouble;

/**
 * Independent double-precision transcriptions of the llama.cpp clip graphs of the qwen3vl, paddleocr, kimivl,
 * dots, hunyuanvl, janus_pro, lfm2 / d1omni_v and cohere2v families (tools/mtmd/models/{qwen3vl,paddleocr,kimivl,
 * dotsocr,hunyuanvl,siglip}.cpp, the {@code build_vit} / {@code build_moe_ffn} / {@code build_patch_merge_permute}
 * builders and the HunyuanVL host position resampling of clip.cpp, the {@code ggml_interpolate} bilinear kernel),
 * written from the reference graphs and the Hugging Face modules they port, not from the executor. Everything
 * runs in {@code double}; results narrow once at the end.
 */
final class VisionFamiliesCReference {

    private VisionFamiliesCReference() {
    }

    private static final DoubleUnaryOperator GELU = MmprojFixture.Reference::geluTanh;
    private static final DoubleUnaryOperator GELU_ERF = v -> 0.5 * v * (1 + erf(v / Math.sqrt(2)));

    // ================================================================== families

    /**
     * Qwen3-VL: both conv kernels plus bias, the align-corners resized positions, merge-group token order, the
     * layer-norm tower with the vision m-rope, deepstack after every block that has one, the post norm, the merger,
     * the deepstack rows concatenated after the merger row.
     */
    static float[][] qwen3vl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        int e = hp.nEmbd();
        double[][] x = patches(f, hp, px, width, pw, ph);
        add(x, conv(f, "v.patch_embd.weight.1", px, width, pw, ph, ps, e));
        float[] pos = f.get("v.position_embd.weight");
        int side = (int) Math.round(Math.sqrt(pos.length / (double) e));
        add(x, alignCorners(pos, e, side, pw, ph));
        int[] order = windowOrder(pw, ph, 2);
        x = MmprojFixture.Reference.gather(x, order);
        int n = pw * ph;
        int[] rope = new int[2 * n];
        for (int t = 0; t < n; t++) {
            rope[t] = order[t] / pw;
            rope[n + t] = order[t] % pw;
        }
        List<double[][]> deepstack = new ArrayList<>();
        for (int il = 0; il < hp.nLayer(); il++) {
            VisionFamilyReference.block(f, hp, il, x, false, r -> MmprojFixture.Reference.rope2d(r, rope, hp.dHead()));
            String p = "v.deepstack." + il + ".";
            if (f.has(p + "fc1.weight")) {
                double[][] feat = norm(group(x, 4), f.get(p + "norm.weight"), f.get(p + "norm.bias"), false,
                        hp.eps());
                deepstack.add(mlp(feat, f, p + "fc1", p + "fc2", GELU));
            }
        }
        x = norm(x, f.get("v.post_ln.weight"), f.get("v.post_ln.bias"), false, hp.eps());
        double[][] merged = mlp(group(x, 4), f, "mm.0", "mm.2", GELU);
        double[][] out = new double[merged.length][];
        for (int t = 0; t < merged.length; t++) {
            int w = merged[t].length;
            out[t] = new double[w * (1 + deepstack.size())];
            System.arraycopy(merged[t], 0, out[t], 0, w);
            for (int i = 0; i < deepstack.size(); i++) {
                System.arraycopy(deepstack.get(i)[t], 0, out[t], (i + 1) * w, w);
            }
        }
        return toFloat(out);
    }

    /** PaddleOCR-VL: resized positions, raster m-rope, layer norm, merge permute, MLP with the encoder op. */
    static float[][] paddleocr(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = withResizedPositions(f, hp, px, width, pw, ph);
        int[] rope = rasterRope(pw, ph);
        VisionFamilyReference.vit(f, hp, x, false, r -> MmprojFixture.Reference.rope2d(r, rope, hp.dHead()));
        double[][] cur = norm(x, f.get("mm.input_norm.weight"), f.get("mm.input_norm.bias"), false, 1e-5);
        cur = mergePermute(cur, pw, ph, 2);
        return toFloat(mlp(cur, f, "mm.1", "mm.2", v -> unary(hp.ffnOp(), v)));
    }

    /**
     * Kimi-VL: resized positions, 2D RoPE (columns on the first half, rows on the second), merge permute with zero
     * padding, a layer norm over every {@code n_embd} chunk of the merged rows, GELU MLP.
     */
    static float[][] kimivl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        int n = pw * ph;
        double[][] x = withResizedPositions(f, hp, px, width, pw, ph);
        int[] posH = new int[n];
        int[] posW = new int[n];
        for (int t = 0; t < n; t++) {
            posH[t] = t / pw;
            posW[t] = t % pw;
        }
        int dh = hp.dHead();
        VisionFamilyReference.vit(f, hp, x, false, r -> VisionFamilyReference.rope2d(r, posW, posH, dh, 10000.0,
                false));
        int s = hp.nMerge();
        int e = hp.nEmbd();
        double[][] cur = mergePermute(x, pw, ph, s);
        for (int t = 0; t < cur.length; t++) {
            double[][] chunks = new double[s * s][];
            for (int k = 0; k < s * s; k++) {
                chunks[k] = Arrays.copyOfRange(cur[t], k * e, (k + 1) * e);
            }
            chunks = norm(chunks, f.get("mm.input_norm.weight"), f.get("mm.input_norm.bias"), false, 1e-5);
            for (int k = 0; k < s * s; k++) {
                System.arraycopy(chunks[k], 0, cur[t], k * e, e);
            }
        }
        return toFloat(mlp(cur, f, "mm.1", "mm.2", GELU));
    }

    /** dots.ocr / dots3-note: RMS tower with raster m-rope, mm.post_norm, layer norm, merge permute, erf-GELU MLP. */
    static float[][] dots(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = patches(f, hp, px, width, pw, ph);
        int[] rope = rasterRope(pw, ph);
        for (int il = 0; il < hp.nLayer(); il++) {
            VisionFamilyReference.block(f, hp, il, x, true, r -> MmprojFixture.Reference.rope2d(r, rope, hp.dHead()));
        }
        x = norm(x, f.get("mm.post_norm.weight"), orNull(f, "v.post_ln.bias"), true, hp.eps());
        double[][] cur = norm(x, f.get("mm.input_norm.weight"), f.get("mm.input_norm.bias"), false, 1e-6);
        cur = mergePermute(cur, pw, ph, hp.nMerge());
        return toFloat(mlp(cur, f, "mm.0", "mm.2", GELU_ERF));
    }

    /**
     * HunyuanVL: the host-resampled positions, the layer-norm tower, RMS pre-norm, the stride-s conv with bias,
     * GELU, the pointwise conv with bias, the newline column, the fc, begin and end rows, RMS post-norm.
     */
    static float[][] hunyuanvl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        int e = hp.nEmbd();
        double[][] x = patches(f, hp, px, width, pw, ph);
        add(x, hunyuanPositions(f.get("v.position_embd.weight"), e, pw, ph));
        VisionFamilyReference.vit(f, hp, x, false, null);
        x = norm(x, f.get("mm.pre_norm.weight"), null, true, hp.eps());
        int s = hp.nMerge();
        int ow = pw / s;
        int oh = ph / s;
        double[][] h = VisionFamilyReference.unfold(cropTo(x, pw, ow * s, oh * s), ow * s, oh * s, s);
        h = lin(h, f.get("mm.0.weight"), f.get("mm.0.bias"));
        apply(h, GELU);
        h = lin(h, f.get("mm.2.weight"), f.get("mm.2.bias"));
        double[] nl = toDouble(f.get("v.image_newline"));
        double[][] tokens = new double[(ow + 1) * oh][];
        for (int y = 0; y < oh; y++) {
            for (int xx = 0; xx < ow; xx++) {
                tokens[y * (ow + 1) + xx] = h[y * ow + xx];
            }
            tokens[y * (ow + 1) + ow] = nl;
        }
        double[][] proj = lin(tokens, f.get("mm.model.fc.weight"), f.get("mm.model.fc.bias"));
        double[][] out = new double[proj.length + 2][];
        out[0] = toDouble(f.get("mm.image_begin"));
        System.arraycopy(proj, 0, out, 1, proj.length);
        out[proj.length + 1] = toDouble(f.get("mm.image_end"));
        return toFloat(norm(out, f.get("mm.post_norm.weight"), null, true, hp.eps()));
    }

    /** Janus-Pro: the SigLIP tower over the image-size grid, then {@code mm.0} / {@code mm.1} with the encoder op. */
    static float[][] janus(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        double[][] x = siglip(f, hp, px, width, height);
        return toFloat(mlp(x, f, "mm.0", "mm.1", v -> unary(hp.ffnOp(), v)));
    }

    /** LFM2-VL / d1-omni: resized positions, merge permute (padded), optional layer norm, GELU (erf) MLP. */
    static float[][] lfm2(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = withResizedPositions(f, hp, px, width, pw, ph);
        VisionFamilyReference.vit(f, hp, x, false, null);
        double[][] cur = mergePermute(x, pw, ph, hp.nMerge());
        if (f.has("mm.input_norm.weight")) {
            cur = norm(cur, f.get("mm.input_norm.weight"), orNull(f, "mm.input_norm.bias"), false, 1e-5);
        } else if (f.has("mm.input_norm.bias")) {
            add(cur, new double[][]{toDouble(f.get("mm.input_norm.bias"))});
        }
        DoubleUnaryOperator act = hp.projectorType() == ProjectorType.D1OMNI_V ? GELU_ERF : GELU;
        return toFloat(mlp(cur, f, "mm.1", "mm.2", act));
    }

    /** Cohere2 vision: the SigLIP tower, merge permute, {@code mm.1}, {@code x * silu(gate)}, {@code mm.2}. */
    static float[][] cohere2v(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = siglip(f, hp, px, width, height);
        double[][] h = lin(mergePermute(x, pw, ph, hp.nMerge()), f.get("mm.1.weight"), f.get("mm.1.bias"));
        int half = h[0].length / 2;
        double[][] g = new double[h.length][half];
        for (int t = 0; t < h.length; t++) {
            for (int i = 0; i < half; i++) {
                double gate = h[t][half + i];
                g[t][i] = h[t][i] * gate * sigmoid(gate);
            }
        }
        return toFloat(lin(g, f.get("mm.2.weight"), f.get("mm.2.bias")));
    }

    // ================================================================== pieces

    /** The SigLIP tower over the {@code image_size} grid with the stored positions. */
    private static double[][] siglip(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int pw = width / hp.patchSize();
        int ph = height / hp.patchSize();
        double[][] x = patches(f, hp, px, width, pw, ph);
        add(x, rows(f.get("v.position_embd.weight"), pw * ph, hp.nEmbd()));
        VisionFamilyReference.vit(f, hp, x, false, null);
        return x;
    }

    /** Patches plus the position grid resized by PyTorch's antialiased bilinear interpolation. */
    private static double[][] withResizedPositions(MmprojFixture f, ClipHparams hp, float[] px, int width, int pw,
                                                   int ph) {
        int e = hp.nEmbd();
        double[][] x = patches(f, hp, px, width, pw, ph);
        float[] pos = f.get("v.position_embd.weight");
        int side = (int) Math.round(Math.sqrt(pos.length / (double) e));
        add(x, side == pw && side == ph ? rows(pos, side * side, e)
                : VisionFamilyReference.resizeAntialiasBilinear(pos, e, side, pw, ph));
        return x;
    }

    /** Conv of one patch kernel (no bias), raster patch order. */
    private static double[][] conv(MmprojFixture f, String name, float[] px, int width, int pw, int ph, int ps,
                                   int e) {
        float[] w = f.get(name);
        double[][] x = new double[pw * ph][e];
        for (int py = 0; py < ph; py++) {
            for (int pxi = 0; pxi < pw; pxi++) {
                for (int c = 0; c < e; c++) {
                    double s = 0;
                    for (int ic = 0; ic < 3; ic++) {
                        for (int ky = 0; ky < ps; ky++) {
                            for (int kx = 0; kx < ps; kx++) {
                                s += w[((c * 3 + ic) * ps + ky) * ps + kx]
                                        * px[((py * ps + ky) * width + pxi * ps + kx) * 3 + ic];
                            }
                        }
                    }
                    x[py * pw + pxi][c] = s;
                }
            }
        }
        return x;
    }

    /**
     * PyTorch {@code interpolate(mode="bilinear", align_corners=True)}: output {@code i} samples source
     * {@code i * (in - 1) / (out - 1)}, the grid itself when the sizes match.
     */
    static double[][] alignCorners(float[] pos, int e, int side, int width, int height) {
        if (side == width && side == height) {
            return rows(pos, side * side, e);
        }
        double sx = width > 1 ? (double) (side - 1) / (width - 1) : 0;
        double sy = height > 1 ? (double) (side - 1) / (height - 1) : 0;
        return bilinear(pos, e, side, width, height, i -> i * sx, i -> i * sy);
    }

    /**
     * The HunyuanVL host resampling of clip.cpp: scale {@code (out + 0.1) / n_grid}, pixel-centre source
     * {@code (i + 0.5) / scale - 0.5}.
     */
    static double[][] hunyuanPositions(float[] pos, int e, int width, int height) {
        int side = (int) Math.round(Math.sqrt(pos.length / (double) e));
        double sx = (width + 0.1) / side;
        double sy = (height + 0.1) / side;
        return bilinear(pos, e, side, width, height, i -> (i + 0.5) / sx - 0.5, i -> (i + 0.5) / sy - 0.5);
    }

    /** Clamped bilinear sampling of the square grid at the source coordinates {@code fx(i0)}, {@code fy(i1)}. */
    private static double[][] bilinear(float[] pos, int e, int side, int width, int height, DoubleUnaryOperator fx,
                                       DoubleUnaryOperator fy) {
        double[][] out = new double[width * height][e];
        for (int oy = 0; oy < height; oy++) {
            double y = fy.applyAsDouble(oy);
            int y0 = clamp((int) Math.floor(y), side);
            int y1 = clamp((int) Math.floor(y) + 1, side);
            double dy = Math.max(0, Math.min(y - y0, 1));
            for (int ox = 0; ox < width; ox++) {
                double x = fx.applyAsDouble(ox);
                int x0 = clamp((int) Math.floor(x), side);
                int x1 = clamp((int) Math.floor(x) + 1, side);
                double dx = Math.max(0, Math.min(x - x0, 1));
                for (int c = 0; c < e; c++) {
                    out[oy * width + ox][c] = pos[(y0 * side + x0) * e + c] * (1 - dx) * (1 - dy)
                            + pos[(y0 * side + x1) * e + c] * dx * (1 - dy)
                            + pos[(y1 * side + x0) * e + c] * (1 - dx) * dy
                            + pos[(y1 * side + x1) * e + c] * dx * dy;
                }
            }
        }
        return out;
    }

    private static int clamp(int i, int n) {
        return Math.max(0, Math.min(i, n - 1));
    }

    /**
     * The raster index of every token of the qwen merge order: the {@code s x s} windows row-major, each window's
     * patches row-major (Qwen2-VL / Qwen3-VL {@code PatchEmbed} input order).
     */
    private static int[] windowOrder(int pw, int ph, int s) {
        int[] order = new int[pw * ph];
        int t = 0;
        for (int wy = 0; wy < ph; wy += s) {
            for (int wx = 0; wx < pw; wx += s) {
                for (int y = wy; y < wy + s; y++) {
                    for (int x = wx; x < wx + s; x++) {
                        order[t++] = y * pw + x;
                    }
                }
            }
        }
        return order;
    }

    /** The qwen m-rope positions of a raster grid: rows, then columns. */
    private static int[] rasterRope(int pw, int ph) {
        int n = pw * ph;
        int[] pos = new int[2 * n];
        for (int t = 0; t < n; t++) {
            pos[t] = t / pw;
            pos[n + t] = t % pw;
        }
        return pos;
    }

    /**
     * {@code build_patch_merge_permute} (the Idefics3 / LFM2 pixel unshuffle): the grid zero-padded to multiples of
     * {@code s}, then every {@code s x s} window concatenated row-major ({@code (dy * s + dx) * e + c}).
     */
    static double[][] mergePermute(double[][] x, int pw, int ph, int s) {
        int wp = (pw + s - 1) / s * s;
        int hp = (ph + s - 1) / s * s;
        int e = x[0].length;
        double[][] padded = new double[wp * hp][];
        for (int y = 0; y < hp; y++) {
            for (int xx = 0; xx < wp; xx++) {
                padded[y * wp + xx] = y < ph && xx < pw ? x[y * pw + xx] : new double[e];
            }
        }
        return VisionFamilyReference.spaceToDepth(padded, wp, hp, s, false);
    }

    /** The top-left {@code w x h} patches of a raster grid {@code pw} wide. */
    private static double[][] cropTo(double[][] x, int pw, int w, int h) {
        double[][] out = new double[w * h][];
        for (int y = 0; y < h; y++) {
            for (int xx = 0; xx < w; xx++) {
                out[y * w + xx] = x[y * pw + xx];
            }
        }
        return out;
    }

    /** Groups of {@code k} consecutive rows concatenated into one row. */
    private static double[][] group(double[][] x, int k) {
        int e = x[0].length;
        double[][] out = new double[x.length / k][k * e];
        for (int t = 0; t < out.length; t++) {
            for (int i = 0; i < k; i++) {
                System.arraycopy(x[t * k + i], 0, out[t], i * e, e);
            }
        }
        return out;
    }

    private static double[][] mlp(double[][] x, MmprojFixture f, String up, String down, DoubleUnaryOperator act) {
        double[][] h = lin(x, f.get(up + ".weight"), orNull(f, up + ".bias"));
        apply(h, act);
        return lin(h, f.get(down + ".weight"), orNull(f, down + ".bias"));
    }

    /** {@code x W^T + b} with {@code W} flat {@code [rows][cols]}, {@code cols} the row width of {@code x}. */
    private static double[][] lin(double[][] x, float[] w, float[] b) {
        double[][] y = new double[x.length][w.length / x[0].length];
        linear(x, w, b, y);
        return y;
    }

    private static void apply(double[][] x, DoubleUnaryOperator op) {
        for (double[] row : x) {
            for (int i = 0; i < row.length; i++) {
                row[i] = op.applyAsDouble(row[i]);
            }
        }
    }

    /** {@code x[t] += y[t]}, or {@code x[t] += y[0]} for every row when {@code y} holds one row. */
    private static void add(double[][] x, double[][] y) {
        for (int t = 0; t < x.length; t++) {
            double[] src = y.length == 1 ? y[0] : y[t];
            for (int c = 0; c < x[t].length; c++) {
                x[t][c] += src[c];
            }
        }
    }
}
