package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Synthetic mmproj tensors, shapes and values of one vision family fixture, plus an independent
 * double-precision transcription of the llama.cpp clip graphs the executor must match. Package-private test
 * support of the {@code mm} tests.
 */
final class MmprojFixture {

    final Map<String, long[]> shapes = new LinkedHashMap<>();
    final Map<String, float[]> values = new HashMap<>();
    private final Random rnd;

    MmprojFixture(long seed) {
        rnd = new Random(seed);
    }

    void put(String name, float lo, float hi, long... ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = lo + rnd.nextFloat() * (hi - lo);
        }
        shapes.put(name, ne);
        values.put(name, v);
    }

    void weight(String name, long... ne) {
        put(name, -0.5f, 0.5f, ne);
    }

    void norm(String weight, String bias, long... ne) {
        put(weight, 0.8f, 1.2f, ne);
        put(bias, -0.1f, 0.1f, ne);
    }

    float[] get(String name) {
        float[] v = values.get(name);
        if (v == null) {
            throw new MissingTensorException("clip", -1, name);
        }
        return v;
    }

    boolean has(String name) {
        return values.containsKey(name);
    }

    void remove(String name) {
        shapes.remove(name);
        values.remove(name);
    }

    /** The stored tensor as a 2D matmul view: cols = product of the input dims ne[0..], rows = ne[rank-1]. */
    CpuTensorRef ref(String name) {
        float[] v = get(name);
        long[] ne = shapes.get(name);
        long cols = 1;
        for (int i = 0; i < ne.length - 1; i++) {
            cols *= ne[i];
        }
        long rows = ne[ne.length - 1];
        ByteBuffer bb = ByteBuffer.allocate(v.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) {
            bb.putFloat(f);
        }
        bb.flip();
        return CpuTensorRef.of(GGMLType.F32, bb, rows, cols);
    }

    TensorCatalog catalog() {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        for (Map.Entry<String, long[]> e : shapes.entrySet()) {
            infos.add(new GGUFTensorInfo(e.getKey(), e.getValue(), GGMLType.F32, 0L));
        }
        return TensorCatalog.of("clip", infos);
    }

    /** Deterministic normalized image pixels in the interleaved RGB layout of {@link ClipImage}. */
    static float[] pixels(int width, int height, long seed) {
        Random r = new Random(seed);
        float[] px = new float[width * height * 3];
        for (int i = 0; i < px.length; i++) {
            px[i] = r.nextFloat() * 2f - 1f;
        }
        return px;
    }

    /**
     * The clip graphs in double precision, transcribed from llama.cpp tools/mtmd/models/{llava,siglip,
     * qwen2vl}.cpp: everything computed in {@code double}, results narrowed once at the end.
     */
    static final class Reference {

        /**
         * Full CLIP / SigLIP + square-projector forward. The llava projectors (mlp, mlp_norm) run the llava graph:
         * the blocks up to the deepest feature layer ({@code n_layer - 1} without feature layers), the optional
         * post-norm, the stacked feature layers, and {@code n_patches} rows read from row 1 on when a class token
         * exists.
         */
        static float[][] clip(MmprojFixture f, ClipHparams hp, float[] pixels, int width, int height) {
            int side = width / hp.patchSize();
            int patches = side * side;
            boolean hasClass = f.has("v.class_embd");
            double[][] x = embed(f, hp, pixels, width, height);
            boolean llava = hp.projectorType() == ProjectorType.MLP || hp.projectorType() == ProjectorType.MLP_NORM;
            if (llava) {
                x = llavaTower(f, hp, x);
                x = java.util.Arrays.copyOfRange(x, hasClass ? 1 : 0, (hasClass ? 1 : 0) + patches);
            } else {
                for (int il = 0; il < hp.nLayer(); il++) {
                    block(f, hp, il, x, false, null, null);
                }
                layerNorm(x, f.get("v.post_ln.weight"), orNull(f, "v.post_ln.bias"), hp.eps());
            }
            switch (hp.projectorType()) {
                case MLP:
                    return llavaMlp(f, hp, x);
                case MLP_NORM:
                    return llavaMlpNorm(f, hp, x);
                case GEMMA3:
                    return gemma3(f, hp, x, side);
                default:
                    throw new GgufFormatException("reference: unsupported projector " + hp.projectorType());
            }
        }

        /**
         * The square-tile input of the llava and siglip graphs: conv patches in raster order, the class token
         * appended last, the learned position rows, and the optional layer-norm pre-norm.
         */
        static double[][] embed(MmprojFixture f, ClipHparams hp, float[] pixels, int width, int height) {
            int ps = hp.patchSize();
            int side = width / ps;
            int nEmbd = hp.nEmbd();
            int patches = side * side;
            boolean hasClass = f.has("v.class_embd");
            int nPos = patches + (hasClass ? 1 : 0);
            double[][] x = new double[nPos][nEmbd];
            for (int py = 0; py < side; py++) {
                for (int px = 0; px < side; px++) {
                    int t = py * side + px;
                    for (int c = 0; c < nEmbd; c++) {
                        double s = 0;
                        int at = 0;
                        for (int ic = 0; ic < 3; ic++) {
                            for (int ky = 0; ky < ps; ky++) {
                                for (int kx = 0; kx < ps; kx++) {
                                    s += f.get("v.patch_embd.weight")[(c * 3 + ic) * ps * ps + ky * ps + kx]
                                            * pixel(pixels, width, height, px * ps + kx, py * ps + ky, ic);
                                }
                            }
                        }
                        x[t][c] = s;
                        if (f.has("v.patch_embd.bias")) {
                            x[t][c] += f.get("v.patch_embd.bias")[c];
                        }
                    }
                }
            }
            if (hasClass) {
                for (int c = 0; c < nEmbd; c++) {
                    x[patches][c] = f.get("v.class_embd")[c];
                }
            }
            if (f.has("v.position_embd.weight")) {
                float[] pos = f.get("v.position_embd.weight");
                for (int t = 0; t < nPos; t++) {
                    for (int c = 0; c < nEmbd; c++) {
                        x[t][c] += pos[t * nEmbd + c];
                    }
                }
            }
            if (f.has("v.pre_ln.weight")) {
                layerNorm(x, f.get("v.pre_ln.weight"), f.get("v.pre_ln.bias"), hp.eps());
            }
            return x;
        }

        /** The llava graph's blocks, post-norm and feature-layer stack over the embedded rows. */
        static double[][] llavaTower(MmprojFixture f, ClipHparams hp, double[][] x) {
            java.util.Set<Integer> features = new java.util.TreeSet<>();
            int deepest = -1;
            for (int v : hp.featureLayers()) {
                deepest = Math.max(deepest, v);
                if (v >= 0) {
                    features.add(v);
                }
            }
            boolean wholeTower = hp.projectorType() == ProjectorType.GLM_EDGE
                    || hp.projectorType() == ProjectorType.MINICPMV;
            int max = deepest < 0 ? hp.nLayer() - (wholeTower ? 0 : 1) : deepest;
            List<double[][]> stack = new ArrayList<>();
            for (int il = 0; il < max; il++) {
                if (features.contains(il)) {
                    stack.add(copy(x));
                }
                block(f, hp, il, x, false, null, null);
            }
            if (f.has("v.post_ln.weight")) {
                layerNorm(x, f.get("v.post_ln.weight"), orNull(f, "v.post_ln.bias"), hp.eps());
            }
            if (features.contains(max)) {
                stack.add(copy(x));
            }
            if (stack.isEmpty()) {
                return x;
            }
            int width = x[0].length;
            double[][] out = new double[x.length][width * stack.size()];
            for (int b = 0; b < stack.size(); b++) {
                for (int t = 0; t < x.length; t++) {
                    System.arraycopy(stack.get(b)[t], 0, out[t], b * width, width);
                }
            }
            return out;
        }

        static double[][] copy(double[][] x) {
            double[][] out = new double[x.length][];
            for (int t = 0; t < x.length; t++) {
                out[t] = x[t].clone();
            }
            return out;
        }

        /** Full qwen2vl / qwen2.5vl forward; RMS norms and window attention for 2.5. */
        static float[][] qwen(MmprojFixture f, ClipHparams hp, float[] pixels, int width, int height) {
            int ps = hp.patchSize();
            int merge = hp.nMerge();
            int pw = width / ps;
            int ph = height / ps;
            int mw = pw / merge;
            int mh = ph / merge;
            int nEmbd = hp.nEmbd();
            int nPatches = pw * ph;
            boolean rms = hp.projectorType() == ProjectorType.QWEN25VL;
            int[] pos = new int[2 * nPatches];
            double[][] x = new double[nPatches][nEmbd];
            float[] kernel = f.get("v.patch_embd.weight");
            // build_inp_temporal_merge: the second Conv3d frame runs over the same still image and adds
            float[] kernel1 = f.has("v.patch_embd.weight.1") ? f.get("v.patch_embd.weight.1") : null;
            int t = 0;
            for (int ty = 0; ty < mh; ty++) {
                for (int tx = 0; tx < mw; tx++) {
                    for (int dy = 0; dy < merge; dy++) {
                        for (int dx = 0; dx < merge; dx++) {
                            int y = ty * merge + dy;
                            int xx = tx * merge + dx;
                            pos[t] = y;
                            pos[nPatches + t] = xx;
                            for (int c = 0; c < nEmbd; c++) {
                                double s = 0;
                                int at = 0;
                                for (int ic = 0; ic < 3; ic++) {
                                    for (int ky = 0; ky < ps; ky++) {
                                        for (int kx = 0; kx < ps; kx++) {
                                            int k = (c * 3 + ic) * ps * ps + ky * ps + kx;
                                            double wk = kernel[k] + (kernel1 != null ? (double) kernel1[k] : 0.0);
                                            s += wk * pixel(pixels, width, height, xx * ps + kx, y * ps + ky, ic);
                                        }
                                    }
                                }
                                x[t][c] = s;
                            }
                            t++;
                        }
                    }
                }
            }
            Window w = hp.nWaPattern() > 0 ? windowPlan(hp, mw, mh) : Window.NONE;
            if (w.reorder) {
                // the graph reorders merged tokens, each a block of merge^2 contiguous patch rows, and the
                // positions input is filled at the reordered rows (clip.cpp set_input, qwen2.5vl)
                int mpow = hp.nMerge() * hp.nMerge();
                x = gatherBlocks(x, w.invIdx, mpow);
                pos = reorderPositions(pos, w.invIdx, mpow);
            }
            if (f.has("v.pre_ln.weight")) {
                if (rms) {
                    rmsNorm(x, f.get("v.pre_ln.weight"), hp.eps());
                } else {
                    layerNorm(x, f.get("v.pre_ln.weight"), f.get("v.pre_ln.bias"), hp.eps());
                }
            }
            for (int il = 0; il < hp.nLayer(); il++) {
                boolean full = !w.reorder || (il + 1) % hp.nWaPattern() == 0;
                block(f, hp, il, x, rms, full ? null : w.mask, pos);
            }
            if (rms) {
                rmsNorm(x, f.get("v.post_ln.weight"), hp.eps());
            } else {
                layerNorm(x, f.get("v.post_ln.weight"), orNull(f, "v.post_ln.bias"), hp.eps());
            }
            float[][] out = merger(f, hp, x, mw, mh);
            return w.reorder ? gatherRows(out, w.idx) : out;
        }

        private static final class Window {
            static final Window NONE = new Window(false, null, null, null);
            final boolean reorder;
            final int[] idx;
            final int[] invIdx;
            final float[] mask;

            Window(boolean reorder, int[] idx, int[] invIdx, float[] mask) {
                this.reorder = reorder;
                this.idx = idx;
                this.invIdx = invIdx;
                this.mask = mask;
            }
        }

        private static Window windowPlan(ClipHparams hp, int mw, int mh) {
            int nTokens = mw * mh;
            int mpow = hp.nMerge() * hp.nMerge();
            int window = (hp.attnWindowSize() > 0 ? hp.attnWindowSize() : 112) / hp.patchSize() / hp.nMerge();
            int[] idx = new int[nTokens];
            int[] invIdx = new int[nTokens];
            int nRows = nTokens * mpow;
            float[] mask = new float[nRows * nRows];
            java.util.Arrays.fill(mask, Float.NEGATIVE_INFINITY);
            int dst = 0;
            int maskRow = 0;
            for (int y = 0; y < mh; y += window) {
                for (int x = 0; x < mw; x += window) {
                    int winH = Math.min(window, mh - y);
                    int winW = Math.min(window, mw - x);
                    int dst0 = dst;
                    for (int dy = 0; dy < winH; dy++) {
                        for (int dx = 0; dx < winW; dx++) {
                            int src = (y + dy) * mw + (x + dx);
                            idx[src] = dst;
                            invIdx[dst] = src;
                            dst++;
                        }
                    }
                    for (int r = 0; r < winH * winW * mpow; r++) {
                        int rowOff = maskRow * nRows;
                        java.util.Arrays.fill(mask, rowOff + dst0 * mpow, rowOff + dst * mpow, 0f);
                        maskRow++;
                    }
                }
            }
            return new Window(true, idx, invIdx, mask);
        }

        /** One encoder block in place; rope positions and window mask only for the qwen graphs. */
        static void block(MmprojFixture f, ClipHparams hp, int il, double[][] x, boolean rms,
                                  float[] mask, int[] pos) {
            int nEmbd = hp.nEmbd();
            int n = x.length;
            int dh = hp.nEmbdHead() > 0 ? hp.nEmbdHead() : nEmbd / hp.nHead();
            String p = "v.blk." + il + ".";
            double[][] cur = new double[n][nEmbd];
            if (rms) {
                rmsNorm(cur, x, f.get(p + "ln1.weight"), hp.eps());
            } else {
                layerNorm(cur, x, f.get(p + "ln1.weight"), orNull(f, p + "ln1.bias"), hp.eps());
            }
            boolean fused = f.has(p + "attn_qkv.weight");
            double[][] q = new double[n][nEmbd];
            double[][] k = new double[n][nEmbd];
            double[][] v = new double[n][nEmbd];
            if (fused) {
                float[] wgt = f.get(p + "attn_qkv.weight");
                float[] bias = f.has(p + "attn_qkv.bias") ? f.get(p + "attn_qkv.bias") : null;
                for (int t = 0; t < n; t++) {
                    for (int i = 0; i < 3 * nEmbd; i++) {
                        double s = bias != null ? bias[i] : 0;
                        for (int c = 0; c < nEmbd; c++) {
                            s += wgt[i * nEmbd + c] * cur[t][c];
                        }
                        if (i < nEmbd) {
                            q[t][i] = s;
                        } else if (i < 2 * nEmbd) {
                            k[t][i - nEmbd] = s;
                        } else {
                            v[t][i - 2 * nEmbd] = s;
                        }
                    }
                }
            } else {
                linear(cur, f.get(p + "attn_q.weight"), f.get(p + "attn_q.bias"), q);
                linear(cur, f.get(p + "attn_k.weight"), f.get(p + "attn_k.bias"), k);
                linear(cur, f.get(p + "attn_v.weight"), f.get(p + "attn_v.bias"), v);
            }
            if (pos != null) {
                rope2d(q, pos, dh);
                rope2d(k, pos, dh);
            }
            double[][] out = new double[n][nEmbd];
            float scale = (float) (1.0 / Math.sqrt(dh));
            int kvGroup = hp.nHead() / hp.nHeadKv();
            double[] scores = new double[n];
            for (int h = 0; h < hp.nHead(); h++) {
                int kvh = h / kvGroup;
                for (int tq = 0; tq < n; tq++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int tk = 0; tk < n; tk++) {
                        double s = 0;
                        for (int d = 0; d < dh; d++) {
                            s += q[tq][h * dh + d] * k[tk][kvh * dh + d];
                        }
                        s *= scale;
                        if (mask != null) {
                            s += mask[tq * n + tk];
                        }
                        scores[tk] = s;
                        max = Math.max(max, s);
                    }
                    double sum = 0;
                    for (int tk = 0; tk < n; tk++) {
                        scores[tk] = Math.exp(scores[tk] - max);
                        sum += scores[tk];
                    }
                    for (int d = 0; d < dh; d++) {
                        double acc = 0;
                        for (int tk = 0; tk < n; tk++) {
                            acc += scores[tk] / sum * v[tk][kvh * dh + d];
                        }
                        out[tq][h * dh + d] = acc;
                    }
                }
            }
            double[][] attn = new double[n][nEmbd];
            linear(out, f.get(p + "attn_out.weight"), orNull(f, p + "attn_out.bias"), attn);
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < nEmbd; c++) {
                    x[t][c] += attn[t][c];
                }
            }
            if (rms) {
                rmsNorm(cur, x, f.get(p + "ln2.weight"), hp.eps());
            } else {
                layerNorm(cur, x, f.get(p + "ln2.weight"), orNull(f, p + "ln2.bias"), hp.eps());
            }
            double[] up = new double[n * hp.nFf()];
            double[] gate = new double[n * hp.nFf()];
            Pair ffn = ffnPair(f, p, hp.projectorType(), hp.nEmbd(), hp.nFf());
            linearFlat(cur, ffn.upW, ffn.upB, up, hp.nFf());
            boolean gated = f.has(p + "ffn_gate.weight");
            if (gated) {
                linearFlat(cur, f.get(p + "ffn_gate.weight"), orNull(f, p + "ffn_gate.bias"), gate, hp.nFf());
            }
            double[] act = new double[n * hp.nFf()];
            for (int i = 0; i < act.length; i++) {
                act[i] = gated ? glu(hp.ffnOp(), gate[i], up[i]) : unary(hp.ffnOp(), up[i]);
            }
            double[] down = new double[n * nEmbd];
            linearFlatFlat(act, ffn.downW, ffn.downB, down, nEmbd);
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < nEmbd; c++) {
                    x[t][c] += down[t * nEmbd + c];
                }
            }
        }

        /** Row r of the reordered grid takes the positions of patch r of the merged token it holds. */
        private static int[] reorderPositions(int[] pos, int[] invIdx, int mpow) {
            int n = pos.length / 2;
            int[] out = new int[pos.length];
            for (int b = 0; b < invIdx.length; b++) {
                for (int r = 0; r < mpow; r++) {
                    out[b * mpow + r] = pos[invIdx[b] * mpow + r];
                    out[n + b * mpow + r] = pos[n + invIdx[b] * mpow + r];
                }
            }
            return out;
        }

        /**
         * The vision rope (ggml_rope_multi, GGML_ROPE_TYPE_VISION, sections d/4 each): per head, pair (j, j+d/2)
         * rotates by y * base^(-4j/d) for j < d/4 and by x * base^(-4(j - d/4)/d) above, each section restarting
         * its theta at the section start (ggml_mrope_cache_init indep_sects), as the Hugging Face vision rotary
         * concatenates one frequency set per axis.
         */
        static void rope2d(double[][] x, int[] pos, int dh) {
            int half = dh / 2;
            int quarter = dh / 4;
            int nEmbd = x[0].length;
            int n = x.length;
            for (int t = 0; t < n; t++) {
                int row = pos[t];
                int col = pos[n + t];
                for (int base = 0; base < nEmbd; base += dh) {
                    for (int j = 0; j < half; j++) {
                        boolean first = j < quarter;
                        int exponent = first ? j : j - quarter;
                        double angle = (first ? row : col) * Math.pow(10000.0, -4.0 * exponent / dh);
                        double cos = Math.cos(angle);
                        double sin = Math.sin(angle);
                        double x0 = x[t][base + j];
                        double x1 = x[t][base + j + half];
                        x[t][base + j] = x0 * cos - x1 * sin;
                        x[t][base + j + half] = x0 * sin + x1 * cos;
                    }
                }
            }
        }


        private static float[][] llavaMlp(MmprojFixture f, ClipHparams hp, double[][] x) {
            int d = hp.projectionDim();
            int n = x.length;
            double[][] h = new double[n][d];
            linear(x, f.get("mm.0.weight"), f.get("mm.0.bias"), h);
            for (double[] row : h) {
                for (int i = 0; i < d; i++) {
                    row[i] = geluTanh(row[i]);
                }
            }
            if (f.has("mm.2.weight")) {
                double[][] out = new double[n][d];
                linear(h, f.get("mm.2.weight"), f.get("mm.2.bias"), out);
                return toFloat(out);
            }
            return toFloat(h);
        }

        private static float[][] llavaMlpNorm(MmprojFixture f, ClipHparams hp, double[][] x) {
            int d = hp.projectionDim();
            int n = x.length;
            double[][] h = new double[n][d];
            linear(x, f.get("mm.0.weight"), f.get("mm.0.bias"), h);
            layerNorm(h, f.get("mm.1.weight"), f.get("mm.1.bias"), hp.eps());
            for (double[] row : h) {
                for (int i = 0; i < d; i++) {
                    row[i] = geluTanh(row[i]);
                }
            }
            double[][] out = new double[n][d];
            linear(h, f.get("mm.3.weight"), f.get("mm.3.bias"), out);
            layerNorm(out, f.get("mm.4.weight"), f.get("mm.4.bias"), hp.eps());
            return toFloat(out);
        }

        private static float[][] merger(MmprojFixture f, ClipHparams hp, double[][] x, int mw, int mh) {
            int nEmbd = hp.nEmbd();
            int d = hp.projectionDim();
            int mpow = hp.nMerge() * hp.nMerge();
            int n = mw * mh;
            double[][] merged = new double[n][mpow * nEmbd];
            for (int t = 0; t < n; t++) {
                for (int k = 0; k < mpow; k++) {
                    System.arraycopy(x[t * mpow + k], 0, merged[t], k * nEmbd, nEmbd);
                }
            }
            int hidden = f.get("mm.0.bias").length;
            double[][] h = new double[n][hidden];
            linear(merged, f.get("mm.0.weight"), f.get("mm.0.bias"), h);
            for (double[] row : h) {
                for (int i = 0; i < hidden; i++) {
                    row[i] = geluTanh(row[i]);
                }
            }
            double[][] out = new double[n][d];
            linear(h, f.get("mm.2.weight"), f.get("mm.2.bias"), out);
            return toFloat(out);
        }

        private static float[][] gemma3(MmprojFixture f, ClipHparams hp, double[][] x, int side) {
            int nEmbd = hp.nEmbd();
            int d = hp.projectionDim();
            int k = hp.nMerge();
            int outSide = side / k;
            int n = outSide * outSide;
            double[][] pooled = new double[n][nEmbd];
            for (int oy = 0; oy < outSide; oy++) {
                for (int ox = 0; ox < outSide; ox++) {
                    for (int dy = 0; dy < k; dy++) {
                        for (int dx = 0; dx < k; dx++) {
                            double[] src = x[(oy * k + dy) * side + (ox * k + dx)];
                            for (int c = 0; c < nEmbd; c++) {
                                pooled[oy * outSide + ox][c] += src[c];
                            }
                        }
                    }
                    for (int c = 0; c < nEmbd; c++) {
                        pooled[oy * outSide + ox][c] /= k * k;
                    }
                }
            }
            rmsNorm(pooled, f.get("mm.soft_emb_norm.weight"), hp.eps());
            double[][] out = new double[n][d];
            linear(pooled, f.get("mm.input_projection.weight"), null, out);
            return toFloat(out);
        }

        /** The FFN up/down pair with the legacy swap of clip.cpp is_ffn_swapped applied. */
        private static Pair ffnPair(MmprojFixture f, String p, ProjectorType type, int nEmbd, int nFf) {
            float[] upW = f.get(p + "ffn_up.weight");
            float[] upB = orNull(f, p + "ffn_up.bias");
            float[] downW = f.get(p + "ffn_down.weight");
            float[] downB = orNull(f, p + "ffn_down.bias");
            switch (type) {
                case MLP:
                case MLP_NORM:
                case QWEN2VL:
                case QWEN25VL:
                case GEMMA3:
                    if (f.shapes.get(p + "ffn_down.weight")[0] == nEmbd) {
                        float[] w = upW;
                        upW = downW;
                        downW = w;
                        float[] b = upB;
                        upB = downB;
                        downB = b;
                    }
                    break;
                default:
                    break;
            }
            if (upW.length != nEmbd * nFf || downW.length != nFf * nEmbd) {
                throw new IllegalArgumentException("fixture ffn shapes do not fit n_embd " + nEmbd + " n_ff " + nFf);
            }
            return new Pair(upW, upB, downW, downB);
        }

        private static final class Pair {
            final float[] upW;
            final float[] upB;
            final float[] downW;
            final float[] downB;

            Pair(float[] upW, float[] upB, float[] downW, float[] downB) {
                this.upW = upW;
                this.upB = upB;
                this.downW = downW;
                this.downB = downB;
            }
        }

        // ---- primitive math

        static float pixel(float[] px, int width, int height, int x, int y, int c) {
            return px[(y * width + x) * 3 + c];
        }

        static void layerNorm(double[][] y, float[] w, float[] b, float eps) {
            layerNorm(y, y, w, b, eps);
        }

        /** Row-wise layer norm of {@code x} into {@code y} ({@code ggml_norm}). */
        static void layerNorm(double[][] y, double[][] x, float[] w, float[] b, float eps) {
            for (int r = 0; r < x.length; r++) {
                double[] row = x[r];
                double mean = 0;
                for (double v : row) {
                    mean += v;
                }
                mean /= row.length;
                double variance = 0;
                for (double v : row) {
                    variance += (v - mean) * (v - mean);
                }
                double inv = 1.0 / Math.sqrt(variance / row.length + eps);
                for (int c = 0; c < row.length; c++) {
                    y[r][c] = (row[c] - mean) * inv * w[c] + (b != null ? b[c] : 0);
                }
            }
        }

        static void rmsNorm(double[][] y, float[] w, float eps) {
            for (double[] row : y) {
                double ss = 0;
                for (double v : row) {
                    ss += v * v;
                }
                double inv = 1.0 / Math.sqrt(ss / row.length + eps);
                for (int c = 0; c < row.length; c++) {
                    row[c] = row[c] * inv * w[c];
                }
            }
        }

        /** In-place RMS norm of x into y ({@code ggml_rms_norm}). */
        static void rmsNorm(double[][] y, double[][] x, float[] w, float eps) {
            for (int t = 0; t < x.length; t++) {
                double ss = 0;
                for (double v : x[t]) {
                    ss += v * v;
                }
                double inv = 1.0 / Math.sqrt(ss / x[t].length + eps);
                for (int c = 0; c < x[t].length; c++) {
                    y[t][c] = x[t][c] * inv * w[c];
                }
            }
        }

        static void linear(double[][] x, float[] w, float[] b, double[][] y) {
            int cols = x[0].length;
            int rows = y[0].length;
            for (int t = 0; t < x.length; t++) {
                for (int r = 0; r < rows; r++) {
                    double s = b != null ? b[r] : 0;
                    for (int c = 0; c < cols; c++) {
                        s += w[r * cols + c] * x[t][c];
                    }
                    y[t][r] = s;
                }
            }
        }

        static void linearFlat(double[][] x, float[] w, float[] b, double[] y, int rows) {
            int cols = x[0].length;
            for (int t = 0; t < x.length; t++) {
                for (int r = 0; r < rows; r++) {
                    double s = b != null ? b[r] : 0;
                    for (int c = 0; c < cols; c++) {
                        s += w[r * cols + c] * x[t][c];
                    }
                    y[t * rows + r] = s;
                }
            }
        }

        static void linearFlatFlat(double[] x, float[] w, float[] b, double[] y, int rows) {
            int cols = w.length / rows;
            int n = y.length / rows;
            for (int t = 0; t < n; t++) {
                for (int r = 0; r < rows; r++) {
                    double s = b != null ? b[r] : 0;
                    for (int c = 0; c < cols; c++) {
                        s += w[r * cols + c] * x[t * cols + c];
                    }
                    y[t * rows + r] = s;
                }
            }
        }

        static double unary(ClipFfnOp op, double v) {
            switch (op) {
                case GELU:
                    return geluTanh(v);
                case GELU_QUICK:
                    return v * sigmoid(1.702 * v);
                case SILU:
                    return v * sigmoid(v);
                case GELU_ERF:
                    return 0.5 * v * (1 + erf(v / Math.sqrt(2)));
                default:
                    throw new IllegalArgumentException("unhandled op " + op);
            }
        }

        static double glu(ClipFfnOp op, double gate, double up) {
            switch (op) {
                case SILU:
                    return gate * sigmoid(gate) * up;
                case GELU:
                    return geluTanh(gate) * up;
                case GELU_QUICK:
                    return gate * sigmoid(1.702 * gate) * up;
                case GELU_ERF:
                    return 0.5 * gate * (1 + erf(gate / Math.sqrt(2))) * up;
                default:
                    throw new IllegalArgumentException("unhandled op " + op);
            }
        }

        static double sigmoid(double v) {
            return 1.0 / (1.0 + Math.exp(-v));
        }

        static double geluTanh(double v) {
            return 0.5 * v * (1 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (v + 0.044715 * v * v * v)));
        }

        /** Abramowitz-Stegun 7.1.26, exact to 1.5e-7: the erf the executor's kernel approximates. */
        static double erf(double v) {
            boolean neg = v < 0;
            double a = Math.abs(v);
            double t = 1.0 / (1.0 + 0.3275911 * a);
            double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592) * t;
            double y = 1.0 - poly * Math.exp(-a * a);
            return neg ? -y : y;
        }

        static double[][] gather(double[][] x, int[] idx) {
            double[][] out = new double[idx.length][];
            for (int i = 0; i < idx.length; i++) {
                out[i] = x[idx[i]];
            }
            return out;
        }

        /** Gathers blocks of {@code block} consecutive rows: output block b holds input rows of {@code idx[b]}. */
        static double[][] gatherBlocks(double[][] x, int[] idx, int block) {
            double[][] out = new double[idx.length * block][];
            for (int b = 0; b < idx.length; b++) {
                for (int r = 0; r < block; r++) {
                    out[b * block + r] = x[idx[b] * block + r];
                }
            }
            return out;
        }

        static float[][] gatherRows(float[][] x, int[] idx) {
            float[][] out = new float[idx.length][];
            for (int i = 0; i < idx.length; i++) {
                out[i] = x[idx[i]];
            }
            return out;
        }

        static float[] orNull(MmprojFixture f, String name) {
            return f.has(name) ? f.get(name) : null;
        }

        static float[][] toFloat(double[][] x) {
            float[][] out = new float[x.length][x[0].length];
            for (int t = 0; t < x.length; t++) {
                for (int c = 0; c < x[0].length; c++) {
                    out[t][c] = (float) x[t][c];
                }
            }
            return out;
        }
    }
}
