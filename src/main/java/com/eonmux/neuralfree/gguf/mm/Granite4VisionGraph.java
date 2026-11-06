package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The Granite Vision 4.1 graph (models/granite4-vision.cpp {@code clip_graph_granite4_vision}): a SigLIP tower over
 * the square {@code image_size} tile with learned positions and no pre/post norm, then one WindowQFormer projector
 * per {@code clip.vision.feature_layer} entry, reading that layer's output: layer norm, the tile split into
 * {@code window_side} windows, a downsampler (one 2x2 sampling offset, or an average pool) split into
 * {@code query_side} windows as the queries, a one-layer QFormer (self-attention, cross-attention to the window,
 * GELU-erf FFN, each post-norm at {@code 1e-12}, 64-wide heads), the windows back in raster order and the output
 * linear. The projector streams concatenate along the channels. {@link #assemble} lays out an anyres image: the
 * overview tile, then the tile grid row by row with the padding tokens removed and a newline row after each row.
 * Written once against {@link VisionOps}. Package-private executor support.
 */
final class Granite4VisionGraph {

    /** The weight / bias tensors of every projector block {@code v.proj_blk.<bid>.}. */
    static final String[] BLOCK_TENSORS = {"linear", "norm", "post_norm", "self_attn_q", "self_attn_k",
            "self_attn_v", "self_attn_out", "self_attn_norm", "cross_attn_q", "cross_attn_k", "cross_attn_v",
            "cross_attn_out", "cross_attn_norm", "ffn_up", "ffn_down", "ffn_norm"};

    /** Head width of the QFormer attention, fixed by the reference graph. */
    static final int HEAD_DIM = 64;

    /** Epsilon of the QFormer layer norms. */
    static final float QFORMER_EPS = 1e-12f;

    private Granite4VisionGraph() {
    }

    /** Name of a projector block tensor. */
    static String blockTensor(int bid, String name) {
        return "v.proj_blk." + bid + "." + name;
    }

    /** Tokens per side of one tile's output: {@code (side / window_side) * query_side}. */
    static int outSide(ClipHparams hp, int side) {
        VisionFamilyHparams fam = hp.visionFamily();
        return side / fam.windowSide() * fam.querySide();
    }

    /** One tile to {@code [outSide^2][K * projection_dim]} rows, flat. */
    static float[] encodeTile(VisionOps ops, ClipImage tile) {
        int[] grid = ClipVit.grid(ops.hp(), tile);
        ClipVit.requireImageSizeGrid(ops.model(), grid[0], grid[1]);
        int side = grid[0];
        int n = side * side;
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.patchRows(tile, side, side);
        ops.addHost(x, ops.hostWeight(ops.model().posEmbd()));
        int[] layers = ops.hp().featureLayers();
        float[][] taps = new float[layers.length][];
        int deepest = 0;
        for (int f : layers) {
            deepest = Math.max(deepest, f);
        }
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE.withLayers(deepest + 1).withTap((done, cur, nt) -> {
            for (int i = 0; i < layers.length; i++) {
                if (layers[i] + 1 == done) {
                    taps[i] = ops.download(cur).clone();
                }
            }
        });
        x = ops.tower(x, n, NormKind.LAYER, opts);
        List<VisionOps.Block> streams = new ArrayList<>(layers.length);
        int width = 0;
        for (int bid = 0; bid < layers.length; bid++) {
            VisionOps.Block stream = block(ops, bid, taps[bid], side);
            width = stream.width();
            streams.add(stream);
        }
        int outSide = outSide(ops.hp(), side);
        return ops.download(ops.concatChannels(streams, outSide * outSide, width));
    }

    /** One WindowQFormer projector over the {@code side x side} rows {@code h} of its feature layer. */
    private static VisionOps.Block block(VisionOps ops, int bid, float[] h, int side) {
        VisionFamilyHparams fam = ops.hp().visionFamily();
        int nEmbd = ops.hp().nEmbd();
        int win = fam.windowSide();
        int qs = fam.querySide();
        int newSide = side / win * qs;
        int nWindows = (side / win) * (side / win);
        VisionOps.Block x = ops.upload(h.clone(), nEmbd);
        ops.norm(x, NormKind.LAYER, blockTensor(bid, "norm.weight"), blockTensor(bid, "norm.bias"),
                ops.hp().eps(), side * side, nEmbd);
        VisionOps.Block enc = ops.gatherRows(x, windowIndex(side, win), nEmbd);
        int offset = fam.spatialOffsets()[bid];
        VisionOps.Block down = offset >= 0 ? ops.gatherRows(x, spatialIndex(side, offset), nEmbd)
                : poolDown(ops, x, nEmbd, side, newSide);
        VisionOps.Block q = ops.gatherRows(down, windowIndex(newSide, qs), nEmbd);
        ops.addHost(q, repeated(ops.hostVector(blockTensor(bid, "query")), q.n() * nEmbd));
        ops.addHost(enc, repeated(ops.hostVector(blockTensor(bid, "img_pos")), enc.n() * nEmbd));
        postNorm(ops, bid, "post_norm", q, nEmbd);

        VisionOps.Block sa = attention(ops, bid, "self_attn_", q, q, nWindows, qs * qs, qs * qs);
        ops.addBlock(sa, q);
        postNorm(ops, bid, "self_attn_norm", sa, nEmbd);
        VisionOps.Block ca = attention(ops, bid, "cross_attn_", sa, enc, nWindows, qs * qs, win * win);
        ops.addBlock(ca, sa);
        postNorm(ops, bid, "cross_attn_norm", ca, nEmbd);
        VisionOps.Block ffn = ops.linear(blockTensor(bid, "ffn_up.weight"), ca, blockTensor(bid, "ffn_up.bias"));
        ops.activate(ffn, ActOp.GELU_ERF);
        ffn = ops.linear(blockTensor(bid, "ffn_down.weight"), ffn, blockTensor(bid, "ffn_down.bias"));
        ops.addBlock(ffn, ca);
        postNorm(ops, bid, "ffn_norm", ffn, nEmbd);

        VisionOps.Block raster = ops.gatherRows(ffn, inverse(windowIndex(newSide, qs)), nEmbd);
        return ops.linear(blockTensor(bid, "linear.weight"), raster, blockTensor(bid, "linear.bias"));
    }

    private static void postNorm(VisionOps ops, int bid, String name, VisionOps.Block x, int nEmbd) {
        ops.norm(x, NormKind.LAYER, blockTensor(bid, name + ".weight"), blockTensor(bid, name + ".bias"),
                QFORMER_EPS, x.n() * x.width() / nEmbd, nEmbd);
    }

    /** {@code bias} tiled to {@code total} values (a row bias over the whole block). */
    private static float[] repeated(float[] bias, int total) {
        if (bias.length == 0 || total % bias.length != 0) {
            throw new IllegalArgumentException("a bias of " + bias.length + " values does not tile to " + total);
        }
        if (total == bias.length) {
            return bias;
        }
        float[] out = new float[total];
        for (int at = 0; at < total; at += bias.length) {
            System.arraycopy(bias, 0, out, at, bias.length);
        }
        return out;
    }

    /**
     * Biased Q/K/V projections, softmax attention of every window's {@code nq} queries over its {@code nk} keys
     * ({@code n_embd / 64} heads), then the biased output projection.
     */
    private static VisionOps.Block attention(VisionOps ops, int bid, String p, VisionOps.Block xq,
                                             VisionOps.Block xkv, int nWindows, int nq, int nk) {
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block q = ops.linear(blockTensor(bid, p + "q.weight"), xq, blockTensor(bid, p + "q.bias"));
        VisionOps.Block k = ops.linear(blockTensor(bid, p + "k.weight"), xkv, blockTensor(bid, p + "k.bias"));
        VisionOps.Block v = ops.linear(blockTensor(bid, p + "v.weight"), xkv, blockTensor(bid, p + "v.bias"));
        int nHead = nEmbd / HEAD_DIM;
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[] out = new float[nWindows * nq * nEmbd];
        for (int win = 0; win < nWindows; win++) {
            VisionOps.Block o = ops.attend(q, win * nq, nq, k, v, win * nk, nk, nEmbd, nHead, 1, HEAD_DIM, scale);
            System.arraycopy(ops.download(o), 0, out, win * nq * nEmbd, nq * nEmbd);
        }
        return ops.linear(blockTensor(bid, p + "out.weight"), ops.upload(out, nEmbd),
                blockTensor(bid, p + "out.bias"));
    }

    /**
     * The raster-to-window permutation of a {@code side x side} grid split into windows of {@code win x win}:
     * entry {@code w * win^2 + p} is the raster index of patch {@code p} of window {@code w}.
     */
    static int[] windowIndex(int side, int win) {
        int nn = side / win;
        int[] idx = new int[side * side];
        for (int wy = 0; wy < nn; wy++) {
            for (int wx = 0; wx < nn; wx++) {
                for (int iy = 0; iy < win; iy++) {
                    for (int ix = 0; ix < win; ix++) {
                        idx[(wy * nn + wx) * win * win + iy * win + ix] = (wy * win + iy) * side + wx * win + ix;
                    }
                }
            }
        }
        return idx;
    }

    private static int[] inverse(int[] idx) {
        int[] inv = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            inv[idx[i]] = i;
        }
        return inv;
    }

    /** The 2x2 sampling of the spatial downsampler at {@code offset = (dy << 1) | dx}. */
    static int[] spatialIndex(int side, int offset) {
        int offY = (offset >> 1) & 1;
        int offX = offset & 1;
        int ns = side / 2;
        int[] idx = new int[ns * ns];
        for (int y = 0; y < ns; y++) {
            for (int x = 0; x < ns; x++) {
                idx[y * ns + x] = (y * 2 + offY) * side + x * 2 + offX;
            }
        }
        return idx;
    }

    /** {@code ggml_pool_2d} AVG with kernel and stride {@code side / newSide}. */
    private static VisionOps.Block poolDown(VisionOps ops, VisionOps.Block x, int nEmbd, int side, int newSide) {
        int k = side / newSide;
        float inv = 1f / (k * k);
        float[] values = ops.download(x);
        float[] out = new float[newSide * newSide * nEmbd];
        for (int oy = 0; oy < newSide; oy++) {
            for (int ox = 0; ox < newSide; ox++) {
                int dst = (oy * newSide + ox) * nEmbd;
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        int src = ((oy * k + ky) * side + ox * k + kx) * nEmbd;
                        for (int c = 0; c < nEmbd; c++) {
                            out[dst + c] += values[src + c];
                        }
                    }
                }
                for (int c = 0; c < nEmbd; c++) {
                    out[dst + c] *= inv;
                }
            }
        }
        return ops.upload(out, nEmbd);
    }

    /**
     * The anyres token sequence: the overview tile's rows, then the {@code gridX x gridY} tile grid read in raster
     * order over the whole tiled image, cropped to the area of the original aspect ratio
     * ({@code clip_anyres_unpad}), with the newline row ({@code v.image_newline} repeated per projector) after
     * every row.
     *
     * @param tiles the encoded tiles in grid raster order, each {@code [outSide^2][dim]} flat
     */
    static float[][] assemble(float[] overview, List<float[]> tiles, int outSide, int dim, int gridX, int gridY,
                              int origWidth, int origHeight, float[] imageNewline) {
        int curW = gridX * outSide;
        int curH = gridY * outSide;
        int[] crop = AnyresGrid.unpad(curW, curH, origWidth, origHeight);
        int offX = crop[0];
        int offY = crop[1];
        int w = crop[2];
        int h = crop[3];
        float[] newline = new float[dim];
        for (int at = 0; at < dim; at += imageNewline.length) {
            System.arraycopy(imageNewline, 0, newline, at, imageNewline.length);
        }
        int nBase = outSide * outSide;
        float[][] out = new float[nBase + h * (w + 1)][];
        for (int t = 0; t < nBase; t++) {
            out[t] = Arrays.copyOfRange(overview, t * dim, (t + 1) * dim);
        }
        int at = nBase;
        for (int y = offY; y < offY + h; y++) {
            for (int x = offX; x < offX + w; x++) {
                float[] tile = tiles.get((y / outSide) * gridX + x / outSide);
                int local = (y % outSide) * outSide + x % outSide;
                out[at++] = Arrays.copyOfRange(tile, local * dim, (local + 1) * dim);
            }
            out[at++] = newline.clone();
        }
        return out;
    }
}
