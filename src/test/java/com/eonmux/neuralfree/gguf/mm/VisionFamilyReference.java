package com.eonmux.neuralfree.gguf.mm;

import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.geluTanh;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.glu;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.linear;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.orNull;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.toFloat;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.unary;

/**
 * Independent double-precision transcriptions of the llama.cpp clip graphs of the minicpmv, internvl, pixtral
 * (and lightonocr, without the break rows), idefics3, llama4 and phi4 families
 * (tools/mtmd/models/{minicpmv,internvl,pixtral,siglip,llama4}.cpp and the {@code build_vit} /
 * {@code build_moe_ffn} / {@code build_rope_2d} / {@code resize_position_embeddings} builders of clip.cpp), written
 * from the reference graphs and the Hugging Face modules they port, not from the executor. Everything runs in
 * {@code double}; results narrow once at the end.
 */
final class VisionFamilyReference {

    private VisionFamilyReference() {
    }

    /** Per-token rotation of Q or K rows. */
    interface Rope {
        void apply(double[][] x);
    }

    // ================================================================== families

    static float[][] minicpmv(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        double[][] x = patches(f, hp, px, width, pw, ph);
        float[] pos = f.get("v.position_embd.weight");
        for (int y = 0; y < ph; y++) {
            for (int xx = 0; xx < pw; xx++) {
                int idx = (int) Math.floor(70.0 * y / ph) * 70 + (int) Math.floor(70.0 * xx / pw);
                for (int c = 0; c < hp.nEmbd(); c++) {
                    x[y * pw + xx][c] += pos[idx * hp.nEmbd() + c];
                }
            }
        }
        vit(f, hp, x, false, null);
        return resampler(f, hp, x, pw);
    }

    private static float[][] resampler(MmprojFixture f, ClipHparams hp, double[][] x, int pw) {
        int d = (int) f.shapes.get("resampler.proj.weight")[0];
        int nq = hp.minicpmvQueryNum();
        int n = x.length;
        double[][] q = rows(f.get("resampler.query"), nq, d);
        q = norm(q, f.get("resampler.ln_q.weight"), f.get("resampler.ln_q.bias"), false, hp.eps());
        double[][] v = new double[n][d];
        linear(x, f.get("resampler.kv.weight"), null, v);
        v = norm(v, f.get("resampler.ln_kv.weight"), f.get("resampler.ln_kv.bias"), false, hp.eps());
        double[][] k = new double[n][d];
        int quarter = d / 4;
        for (int t = 0; t < n; t++) {
            double col = t % pw;
            double row = t / pw;
            for (int i = 0; i < quarter; i++) {
                double omega = 1.0 / Math.pow(10000.0, (double) i / quarter);
                k[t][i] = v[t][i] + Math.sin(col * omega);
                k[t][quarter + i] = v[t][quarter + i] + Math.cos(col * omega);
                k[t][2 * quarter + i] = v[t][2 * quarter + i] + Math.sin(row * omega);
                k[t][3 * quarter + i] = v[t][3 * quarter + i] + Math.cos(row * omega);
            }
        }
        double[][] qp = lin(q, f, "resampler.attn.q", d);
        double[][] kp = lin(k, f, "resampler.attn.k", d);
        double[][] vp = lin(v, f, "resampler.attn.v", d);
        double[][] att = attention(qp, kp, vp, d / 128, 128, null);
        double[][] out = lin(att, f, "resampler.attn.out", d);
        out = norm(out, f.get("resampler.ln_post.weight"), f.get("resampler.ln_post.bias"), false, hp.eps());
        double[][] proj = new double[nq][d];
        linear(out, f.get("resampler.proj.weight"), null, proj);
        return toFloat(proj);
    }

    static float[][] internvl(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        double[][] p = patches(f, hp, px, width, pw, ph);
        int n = pw * ph;
        double[][] x = new double[n + 1][];
        x[0] = toDouble(f.get("v.class_embd"));
        System.arraycopy(p, 0, x, 1, n);
        addPos(x, f.get("v.position_embd.weight"));
        boolean rms = hp.nEmbd() == 3200 && hp.nLayer() == 45;
        vit(f, hp, x, rms, null);
        double[][] body = new double[n][];
        System.arraycopy(x, 1, body, 0, n);
        double[][] cur = spaceToDepth(body, pw, ph, hp.nMerge(), false);
        cur = norm(cur, f.get("mm.model.mlp.0.weight"), f.get("mm.model.mlp.0.bias"), false, 1e-5);
        return toFloat(mlp(cur, f, "mm.model.mlp.1", "mm.model.mlp.3"));
    }

    static float[][] pixtral(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        double[][] x = patches(f, hp, px, width, pw, ph);
        int n = pw * ph;
        int[] posH = new int[n];
        int[] posW = new int[n];
        for (int t = 0; t < n; t++) {
            posH[t] = t / pw;
            posW[t] = t % pw;
        }
        int dh = hp.dHead();
        vit(f, hp, x, true, r -> rope2d(r, posH, posW, dh, 10000.0, true));
        int s = hp.nMerge();
        double[][] cur = x;
        if (f.has("mm.patch_merger.weight")) {
            double[][] normed = norm(x, f.get("mm.input_norm.weight"), null, true, hp.eps());
            cur = unfold(normed, pw, ph, s);
            double[][] merged = new double[cur.length][hp.nEmbd()];
            linear(cur, f.get("mm.patch_merger.weight"), null, merged);
            cur = merged;
        }
        double[][] tokens = mlp(cur, f, "mm.1", "mm.2");
        if (!f.has("v.token_embd.img_break")) {
            return toFloat(tokens);
        }
        int tx = pw / s;
        int ty = ph / s;
        double[][] out = new double[tx * ty + ty - 1][];
        int at = 0;
        for (int y = 0; y < ty; y++) {
            for (int xx = 0; xx < tx; xx++) {
                out[at++] = tokens[y * tx + xx];
            }
            if (y < ty - 1) {
                out[at++] = toDouble(f.get("v.token_embd.img_break"));
            }
        }
        return toFloat(out);
    }

    static float[][] idefics3(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        double[][] x = patches(f, hp, px, width, pw, ph);
        addPos(x, f.get("v.position_embd.weight"));
        vit(f, hp, x, false, null);
        double[][] cur = spaceToDepth(x, pw, ph, hp.nMerge(), false);
        double[][] out = new double[cur.length][(int) f.shapes.get("mm.model.fc.weight")[1]];
        linear(cur, f.get("mm.model.fc.weight"), null, out);
        return toFloat(out);
    }

    static float[][] llama4(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        int n = pw * ph;
        double[][] p = patches(f, hp, px, width, pw, ph);
        double[][] x = new double[n + 1][];
        System.arraycopy(p, 0, x, 0, n);
        x[n] = toDouble(f.get("v.class_embd"));
        addPos(x, f.get("v.position_embd.weight"));
        int[] posH = new int[n + 1];
        int[] posW = new int[n + 1];
        for (int t = 0; t < n; t++) {
            posH[t] = t / pw + 1;
            posW[t] = t % pw + 1;
        }
        int dh = hp.dHead();
        vit(f, hp, x, false, r -> rope2d(r, posW, posH, dh, 10000.0, false));
        double[][] body = new double[n][];
        System.arraycopy(x, 0, body, 0, n);
        double[][] cur = spaceToDepth(body, pw, ph, hp.nMerge(), true);
        double[][] h = matmul(cur, f, "mm.model.mlp.1.weight");
        gelu(h);
        h = matmul(h, f, "mm.model.mlp.2.weight");
        gelu(h);
        return toFloat(matmul(h, f, "mm.model.fc.weight"));
    }

    static float[][] phi4(MmprojFixture f, ClipHparams hp, float[] px, int width, int height) {
        int ps = hp.patchSize();
        int pw = width / ps;
        int ph = height / ps;
        double[][] x = patches(f, hp, px, width, pw, ph);
        float[] pos = f.get("v.position_embd.weight");
        int side = (int) Math.round(Math.sqrt(pos.length / (double) hp.nEmbd()));
        double[][] resized = side == pw && side == ph ? rows(pos, side * side, hp.nEmbd())
                : resizeAntialiasBilinear(pos, hp.nEmbd(), side, pw, ph);
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < hp.nEmbd(); c++) {
                x[t][c] += resized[t][c];
            }
        }
        vit(f, hp, x, false, null);
        return toFloat(mlp(x, f, "mm.0", "mm.2"));
    }

    // ================================================================== ViT

    /** Conv patch embedding (or the unfold-linear of Llama 4: the same flat weight order), raster patch order. */
    static double[][] patches(MmprojFixture f, ClipHparams hp, float[] px, int width, int pw, int ph) {
        int ps = hp.patchSize();
        int e = hp.nEmbd();
        float[] wgt = f.get("v.patch_embd.weight");
        float[] bias = orNull(f, "v.patch_embd.bias");
        double[][] x = new double[pw * ph][e];
        for (int py = 0; py < ph; py++) {
            for (int pxi = 0; pxi < pw; pxi++) {
                for (int c = 0; c < e; c++) {
                    double s = bias != null ? bias[c] : 0;
                    for (int ic = 0; ic < 3; ic++) {
                        for (int ky = 0; ky < ps; ky++) {
                            for (int kx = 0; kx < ps; kx++) {
                                int pix = ((py * ps + ky) * width + pxi * ps + kx) * 3 + ic;
                                s += wgt[((c * 3 + ic) * ps + ky) * ps + kx] * px[pix];
                            }
                        }
                    }
                    x[py * pw + pxi][c] = s;
                }
            }
        }
        return x;
    }

    /** build_vit: pre-norm, blocks (layer scale, Q/K norms, rope), post-norm; {@code rms} picks the norm type. */
    static void vit(MmprojFixture f, ClipHparams hp, double[][] x, boolean rms, Rope rope) {
        if (f.has("v.pre_ln.weight")) {
            copyInto(norm(x, f.get("v.pre_ln.weight"), orNull(f, "v.pre_ln.bias"), rms, hp.eps()), x);
        }
        for (int il = 0; il < hp.nLayer(); il++) {
            block(f, hp, il, x, rms, rope);
        }
        if (f.has("v.post_ln.weight")) {
            copyInto(norm(x, f.get("v.post_ln.weight"), orNull(f, "v.post_ln.bias"), rms, hp.eps()), x);
        }
    }

    /** One build_vit block in place: attention (ls1) and the dense or routed-expert FFN (ls2), with residuals. */
    static void block(MmprojFixture f, ClipHparams hp, int il, double[][] x, boolean rms, Rope rope) {
        String p = "v.blk." + il + ".";
        int e = hp.nEmbd();
        int dh = hp.dHead();
        int n = x.length;
        double[][] cur = norm(x, f.get(p + "ln1.weight"), orNull(f, p + "ln1.bias"), rms, hp.eps());
        double[][] q = new double[n][e];
        double[][] k = new double[n][e];
        double[][] v = new double[n][e];
        if (f.has(p + "attn_qkv.weight")) {
            double[][] qkv = new double[n][3 * e];
            linear(cur, f.get(p + "attn_qkv.weight"), orNull(f, p + "attn_qkv.bias"), qkv);
            for (int t = 0; t < n; t++) {
                System.arraycopy(qkv[t], 0, q[t], 0, e);
                System.arraycopy(qkv[t], e, k[t], 0, e);
                System.arraycopy(qkv[t], 2 * e, v[t], 0, e);
            }
        } else {
            linear(cur, f.get(p + "attn_q.weight"), orNull(f, p + "attn_q.bias"), q);
            linear(cur, f.get(p + "attn_k.weight"), orNull(f, p + "attn_k.bias"), k);
            linear(cur, f.get(p + "attn_v.weight"), orNull(f, p + "attn_v.bias"), v);
        }
        q = qkNorm(q, orNull(f, p + "attn_q_norm.weight"), dh, rms, hp.eps());
        k = qkNorm(k, orNull(f, p + "attn_k_norm.weight"), dh, rms, hp.eps());
        if (rope != null) {
            rope.apply(q);
            rope.apply(k);
        }
        double[][] att = attention(q, k, v, hp.nHead(), dh, null);
        double[][] o = new double[n][e];
        linear(att, f.get(p + "attn_out.weight"), orNull(f, p + "attn_out.bias"), o);
        scale(o, orNull(f, p + "ls1.weight"));
        add(x, o);
        cur = norm(x, f.get(p + "ln2.weight"), orNull(f, p + "ln2.bias"), rms, hp.eps());
        double[][] ffn = ffn(f, hp, p, cur);
        scale(ffn, orNull(f, p + "ls2.weight"));
        add(x, ffn);
    }

    private static double[][] ffn(MmprojFixture f, ClipHparams hp, String p, double[][] cur) {
        if (f.has(p + "ffn_gate_exps.weight")) {
            return moe(f, hp, p, cur);
        }
        String up = "ffn_up";
        String down = "ffn_down";
        boolean swapped = (hp.projectorType() == ProjectorType.IDEFICS3
                || hp.projectorType() == ProjectorType.MINICPMV)
                && f.shapes.get(p + "ffn_down.weight")[0] == hp.nEmbd();
        if (swapped) {
            up = "ffn_down";
            down = "ffn_up";
        }
        int n = cur.length;
        double[][] u = new double[n][hp.nFf()];
        linear(cur, f.get(p + up + ".weight"), orNull(f, p + up + ".bias"), u);
        if (f.has(p + "ffn_gate.weight")) {
            double[][] g = new double[n][hp.nFf()];
            linear(cur, f.get(p + "ffn_gate.weight"), orNull(f, p + "ffn_gate.bias"), g);
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < hp.nFf(); i++) {
                    u[t][i] = glu(hp.ffnOp(), g[t][i], u[t][i]);
                }
            }
        } else {
            for (double[] row : u) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = unary(hp.ffnOp(), row[i]);
                }
            }
        }
        double[][] out = new double[n][hp.nEmbd()];
        linear(u, f.get(p + down + ".weight"), orNull(f, p + down + ".bias"), out);
        return out;
    }

    /**
     * build_moe_ffn: sigmoid router probabilities, the top {@code expert_used_count} experts of probability plus
     * the optional selection bias, weights of the chosen probabilities over their sum, and the weighted sum of
     * {@code down(silu(gate(x)) * up(x))} of the chosen experts.
     */
    private static double[][] moe(MmprojFixture f, ClipHparams hp, String p, double[][] cur) {
        long[] gs = f.shapes.get(p + "ffn_gate_exps.weight");
        int e = hp.nEmbd();
        int nFf = (int) gs[1];
        int nExp = (int) gs[2];
        int k = Math.min(hp.nExpertUsed(), nExp);
        float[] router = f.get(p + "ffn_gate_inp.weight");
        float[] bias = orNull(f, p + "exp_probs_b.weight");
        float[] gate = f.get(p + "ffn_gate_exps.weight");
        float[] up = f.get(p + "ffn_up_exps.weight");
        float[] down = f.get(p + "ffn_down_exps.weight");
        double[][] out = new double[cur.length][e];
        for (int t = 0; t < cur.length; t++) {
            double[] probs = new double[nExp];
            for (int x = 0; x < nExp; x++) {
                double s = 0;
                for (int c = 0; c < e; c++) {
                    s += router[x * e + c] * cur[t][c];
                }
                probs[x] = MmprojFixture.Reference.sigmoid(s);
            }
            boolean[] taken = new boolean[nExp];
            int[] chosen = new int[k];
            double sum = 0;
            for (int i = 0; i < k; i++) {
                int best = -1;
                for (int x = 0; x < nExp; x++) {
                    double v = probs[x] + (bias != null ? bias[x] : 0);
                    if (!taken[x] && (best < 0 || v > probs[best] + (bias != null ? bias[best] : 0))) {
                        best = x;
                    }
                }
                taken[best] = true;
                chosen[i] = best;
                sum += probs[best];
            }
            for (int x : chosen) {
                double[] h = new double[nFf];
                for (int i = 0; i < nFf; i++) {
                    double g = 0;
                    double u = 0;
                    for (int c = 0; c < e; c++) {
                        g += gate[(x * nFf + i) * e + c] * cur[t][c];
                        u += up[(x * nFf + i) * e + c] * cur[t][c];
                    }
                    h[i] = g * MmprojFixture.Reference.sigmoid(g) * u;
                }
                for (int c = 0; c < e; c++) {
                    double d = 0;
                    for (int i = 0; i < nFf; i++) {
                        d += down[(x * e + c) * nFf + i] * h[i];
                    }
                    out[t][c] += probs[x] / sum * d;
                }
            }
        }
        return out;
    }

    private static double[][] qkNorm(double[][] x, float[] w, int dh, boolean rms, float eps) {
        if (w == null) {
            return x;
        }
        int width = w.length;
        double[][] out = new double[x.length][x[0].length];
        for (int t = 0; t < x.length; t++) {
            for (int base = 0; base < x[t].length; base += width) {
                double[][] chunk = {java.util.Arrays.copyOfRange(x[t], base, base + width)};
                System.arraycopy(norm(chunk, w, null, rms, eps)[0], 0, out[t], base, width);
            }
        }
        return out;
    }

    /** Softmax attention per head, {@code 1/sqrt(dh)}, optional additive mask. */
    static double[][] attention(double[][] q, double[][] k, double[][] v, int nHead, int dh, double[][] mask) {
        int nq = q.length;
        int nk = k.length;
        double[][] out = new double[nq][nHead * dh];
        double scale = 1.0 / Math.sqrt(dh);
        for (int h = 0; h < nHead; h++) {
            for (int i = 0; i < nq; i++) {
                double[] s = new double[nk];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < nk; j++) {
                    double dot = 0;
                    for (int c = 0; c < dh; c++) {
                        dot += q[i][h * dh + c] * k[j][h * dh + c];
                    }
                    s[j] = dot * scale + (mask != null ? mask[i][j] : 0);
                    max = Math.max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < nk; j++) {
                    s[j] = Math.exp(s[j] - max);
                    sum += s[j];
                }
                for (int c = 0; c < dh; c++) {
                    double acc = 0;
                    for (int j = 0; j < nk; j++) {
                        acc += s[j] / sum * v[j][h * dh + c];
                    }
                    out[i][h * dh + c] = acc;
                }
            }
        }
        return out;
    }

    /**
     * build_rope_2d over every head: first half, adjacent pairs {@code (2i, 2i+1)} rotate by
     * {@code posA * base^(-2i / (d/2))}; second half likewise with {@code posB}, frequencies scaled by
     * {@code base^(-2/d)} when interleaved.
     */
    static void rope2d(double[][] x, int[] posA, int[] posB, int dh, double base, boolean interleave) {
        int half = dh / 2;
        double odd = interleave ? Math.pow(base, -2.0 / dh) : 1.0;
        for (int t = 0; t < x.length; t++) {
            for (int head = 0; head < x[t].length; head += dh) {
                for (int i0 = 0; i0 < half; i0 += 2) {
                    double theta = Math.pow(base, -(double) i0 / half);
                    rot(x[t], head + i0, posA[t] * theta);
                    rot(x[t], head + half + i0, odd * posB[t] * theta);
                }
            }
        }
    }

    private static void rot(double[] x, int i, double angle) {
        double a = x[i];
        double b = x[i + 1];
        x[i] = a * Math.cos(angle) - b * Math.sin(angle);
        x[i + 1] = a * Math.sin(angle) + b * Math.cos(angle);
    }

    // ================================================================== projector pieces

    /**
     * Space-to-depth of an {@code s x s} window: feature {@code (dy * s + dx) * e + c} of the window at block
     * row {@code by}, block column {@code bx}. Tokens are row-major over the blocks (Idefics3 / InternVL), or
     * column-major as the llama.cpp Llama 4 graph lays them out (it omits the second permute of the HF shuffle).
     */
    static double[][] spaceToDepth(double[][] x, int pw, int ph, int s, boolean columnMajor) {
        int e = x[0].length;
        int bw = pw / s;
        int bh = ph / s;
        double[][] out = new double[bw * bh][s * s * e];
        for (int by = 0; by < bh; by++) {
            for (int bx = 0; bx < bw; bx++) {
                int t = columnMajor ? by + bh * bx : by * bw + bx;
                for (int dy = 0; dy < s; dy++) {
                    for (int dx = 0; dx < s; dx++) {
                        double[] src = x[(by * s + dy) * pw + bx * s + dx];
                        System.arraycopy(src, 0, out[t], (dy * s + dx) * e, e);
                    }
                }
            }
        }
        return out;
    }

    /** torch {@code F.unfold(kernel=s, stride=s)} of the {@code (e, ph, pw)} grid, transposed to tokens. */
    static double[][] unfold(double[][] x, int pw, int ph, int s) {
        int e = x[0].length;
        int ow = pw / s;
        int oh = ph / s;
        double[][] out = new double[ow * oh][e * s * s];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                for (int c = 0; c < e; c++) {
                    for (int ky = 0; ky < s; ky++) {
                        for (int kx = 0; kx < s; kx++) {
                            out[oy * ow + ox][c * s * s + ky * s + kx] = x[(oy * s + ky) * pw + ox * s + kx][c];
                        }
                    }
                }
            }
        }
        return out;
    }

    /**
     * PyTorch {@code interpolate(mode="bilinear", align_corners=False, antialias=True)} of the square position
     * grid to {@code width x height}, rows raster order.
     */
    static double[][] resizeAntialiasBilinear(float[] pos, int e, int side, int width, int height) {
        double sx = (double) width / side;
        double sy = (double) height / side;
        double supX = Math.max(1.0, 1.0 / sx);
        double supY = Math.max(1.0, 1.0 / sy);
        double[][] out = new double[width * height][e];
        for (int oy = 0; oy < height; oy++) {
            double cy = (oy + 0.5) / sy;
            for (int ox = 0; ox < width; ox++) {
                double cx = (ox + 0.5) / sx;
                double total = 0;
                for (int iy = Math.max((int) (cy - supY + 0.5), 0); iy < Math.min((int) (cy + supY + 0.5), side);
                     iy++) {
                    double wy = Math.max(0, 1 - Math.abs((iy - cy + 0.5) / supY));
                    for (int ix = Math.max((int) (cx - supX + 0.5), 0);
                         ix < Math.min((int) (cx + supX + 0.5), side); ix++) {
                        double wgt = wy * Math.max(0, 1 - Math.abs((ix - cx + 0.5) / supX));
                        if (wgt <= 0) {
                            continue;
                        }
                        total += wgt;
                        for (int c = 0; c < e; c++) {
                            out[oy * width + ox][c] += wgt * pos[(iy * side + ix) * e + c];
                        }
                    }
                }
                for (int c = 0; c < e; c++) {
                    out[oy * width + ox][c] /= total;
                }
            }
        }
        return out;
    }

    /** A biased two-layer GELU (tanh) MLP, widths from the fixture shapes. */
    static double[][] mlp(double[][] x, MmprojFixture f, String up, String down) {
        double[][] h = new double[x.length][(int) f.shapes.get(up + ".weight")[1]];
        linear(x, f.get(up + ".weight"), orNull(f, up + ".bias"), h);
        gelu(h);
        double[][] out = new double[x.length][(int) f.shapes.get(down + ".weight")[1]];
        linear(h, f.get(down + ".weight"), orNull(f, down + ".bias"), out);
        return out;
    }

    private static double[][] matmul(double[][] x, MmprojFixture f, String name) {
        double[][] out = new double[x.length][(int) f.shapes.get(name)[1]];
        linear(x, f.get(name), null, out);
        return out;
    }

    private static double[][] lin(double[][] x, MmprojFixture f, String base, int d) {
        double[][] out = new double[x.length][d];
        linear(x, f.get(base + ".weight"), f.get(base + ".bias"), out);
        return out;
    }

    // ================================================================== primitives

    /** Layer norm (or RMS norm) of every row, times {@code w}, plus the optional {@code b}. */
    static double[][] norm(double[][] x, float[] w, float[] b, boolean rms, double eps) {
        double[][] out = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double[] row = x[t];
            int n = row.length;
            double mean = 0;
            if (!rms) {
                for (double v : row) {
                    mean += v;
                }
                mean /= n;
            }
            double ss = 0;
            for (double v : row) {
                ss += (v - mean) * (v - mean);
            }
            double inv = 1.0 / Math.sqrt(ss / n + eps);
            out[t] = new double[n];
            for (int c = 0; c < n; c++) {
                out[t][c] = (row[c] - mean) * inv * w[c] + (b != null ? b[c] : 0);
            }
        }
        return out;
    }

    private static void gelu(double[][] x) {
        for (double[] row : x) {
            for (int i = 0; i < row.length; i++) {
                row[i] = geluTanh(row[i]);
            }
        }
    }

    private static void scale(double[][] x, float[] s) {
        if (s == null) {
            return;
        }
        for (double[] row : x) {
            for (int c = 0; c < row.length; c++) {
                row[c] *= s[c];
            }
        }
    }

    private static void add(double[][] x, double[][] y) {
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < x[t].length; c++) {
                x[t][c] += y[t][c];
            }
        }
    }

    private static void addPos(double[][] x, float[] pos) {
        int e = x[0].length;
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < e; c++) {
                x[t][c] += pos[t * e + c];
            }
        }
    }

    private static void copyInto(double[][] src, double[][] dst) {
        for (int t = 0; t < src.length; t++) {
            dst[t] = src[t];
        }
    }

    static double[][] rows(float[] flat, int n, int width) {
        double[][] out = new double[n][width];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[t][c] = flat[t * width + c];
            }
        }
        return out;
    }

    static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }
}
