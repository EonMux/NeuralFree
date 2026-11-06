package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ops.audio.VocosDecoder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/**
 * Synthetic soprano decoder GGUF (llama.cpp {@code PROJECTOR_TYPE_SOPRANO}): the clip generator keys
 * ({@code clip.gen.audio.projector_type = soprano}, the block count and the layer norm epsilon) and the
 * {@code a.gen.wav.*} Vocos tensors of tools/mtmd/models/soprano.cpp, over an input projection of
 * {@link #INP} hidden-state features, two blocks and a head of {@link #FRAME_VALS} values per frame
 * ({@code n_fft = 512}, the smallest size the fixed hop of 512 samples accepts). Test support of the
 * {@code mm} tests.
 */
final class SopranoFixtures {

    /** Hidden-state feature width ({@code clip.gen.audio.projection_dim}). */
    static final int INP = 3;
    /** Vocos width. */
    static final int C = 4;
    /** Pointwise hidden width. */
    static final int FF = 6;
    /** Values per output frame: 257 log-magnitudes then 257 phases ({@code n_fft} 512). */
    static final int FRAME_VALS = 514;
    /** Vocos block count. */
    static final int BLOCKS = 2;
    static final float EPS = 1e-6f;

    private SopranoFixtures() {
    }

    /** The fixture writer; {@code projectorType} null omits the {@code clip.gen.audio.projector_type} key. */
    static GgufTestWriter soprano(String projectorType) {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        if (projectorType != null) {
            w.str("clip.gen.audio.projector_type", projectorType);
        }
        w.u32("clip.gen.audio.block_count", BLOCKS);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", EPS);
        w.u32("clip.gen.audio.projection_dim", INP);
        Random rnd = new Random(41);
        put(w, rnd, VocosDecoder.weight(VocosDecoder.INPUT), -0.5f, 0.5f, INP, C);
        put(w, rnd, VocosDecoder.bias(VocosDecoder.INPUT), -0.1f, 0.1f, C);
        put(w, rnd, VocosDecoder.weight(VocosDecoder.NORM), 0.8f, 1.2f, C);
        put(w, rnd, VocosDecoder.bias(VocosDecoder.NORM), -0.1f, 0.1f, C);
        for (int il = 0; il < BLOCKS; il++) {
            put(w, rnd, VocosDecoder.dwconv(il, "weight"), -0.3f, 0.3f, VocosDecoder.DW_K, 1, C);
            put(w, rnd, VocosDecoder.dwconv(il, "bias"), -0.05f, 0.05f, C);
            put(w, rnd, VocosDecoder.block(il, "norm", "weight"), 0.8f, 1.2f, C);
            put(w, rnd, VocosDecoder.block(il, "norm", "bias"), -0.1f, 0.1f, C);
            put(w, rnd, VocosDecoder.block(il, "pw1", "weight"), -0.5f, 0.5f, C, FF);
            put(w, rnd, VocosDecoder.block(il, "pw1", "bias"), -0.1f, 0.1f, FF);
            put(w, rnd, VocosDecoder.block(il, "pw2", "weight"), -0.5f, 0.5f, FF, C);
            put(w, rnd, VocosDecoder.block(il, "pw2", "bias"), -0.1f, 0.1f, C);
            put(w, rnd, "a.gen.wav.up.blk." + il + ".gamma", 0.3f, 0.7f, C);
        }
        put(w, rnd, VocosDecoder.weight(VocosDecoder.OUTPUT_NORM), 0.8f, 1.2f, C);
        put(w, rnd, VocosDecoder.bias(VocosDecoder.OUTPUT_NORM), -0.1f, 0.1f, C);
        put(w, rnd, VocosDecoder.weight(VocosDecoder.OUTPUT), -0.5f, 0.5f, C, FRAME_VALS);
        put(w, rnd, VocosDecoder.bias(VocosDecoder.OUTPUT), -0.1f, 0.1f, FRAME_VALS);
        return w;
    }

    private static void put(GgufTestWriter w, Random rnd, String name, float lo, float hi, long... ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = lo + rnd.nextFloat() * (hi - lo);
        }
        w.tensorFromFloats(name, GGMLType.F32, ne, v);
    }

    /** A file at a temp path; the caller deletes it. */
    static Path write() throws IOException {
        return soprano("soprano").writeTempFile();
    }
}
