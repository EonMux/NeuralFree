package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkNorm;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.elements;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The dots3note audio encoder and its adapter (llama.cpp tools/mtmd/models/dots3note.cpp,
 * {@code clip_graph_dots3note_a}): three conv2d blocks (kernel 3, stride 2, padding 1) with GELU-erf over the
 * {@code [frames, mel]} image, the {@code [freq x channel]} flatten through the bias-free {@code a.conv_out}, the
 * shared ViT tower ({@link CpuVitTower}) with RMS norms and NEOX rotary positions over the first half of every
 * head (base 1e4), then the adapter: a layer norm at 1e-5 ({@code mm.a.norm_pre}) and the GELU-erf MLP
 * {@code mm.a.mlp.1} / {@code mm.a.mlp.3}. Immutable weights; package-private executor support.
 */
final class Dots3NoteATower implements FamilyEncoder {

    private static final int CONVS = 3;
    private static final int K = 3;
    private static final int STRIDE = 2;
    private static final int PAD = 1;

    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final CpuVitTower tower;
    private final CpuTensorRef[] convW;
    private final float[][] convB;
    private final CpuTensorRef convOut;
    private final float[] normPreW;
    private final float[] normPreB;
    private final CpuTensorRef mm1;
    private final float[] mm1B;
    private final CpuTensorRef mm3;
    private final float[] mm3B;
    private final int outWidth;

    private Dots3NoteATower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                            Function<String, CpuTensorRef> refs) {
        this.hp = model.hparams();
        this.matVec = matVec;
        TensorCatalog tc = model.catalog();
        this.tower = new CpuVitTower(hp, matVec, layers(tc, hp), tc.get("a.pre_ln.weight"), tc.get("a.pre_ln.bias"),
                tc.get("a.post_ln.weight"), tc.get("a.post_ln.bias"), floats, refs, null);
        int[] c = convChannels(tc);
        this.convW = new CpuTensorRef[CONVS];
        this.convB = new float[CONVS][];
        for (int i = 0; i < CONVS; i++) {
            int cIn = i == 0 ? 1 : c[i - 1];
            convW[i] = CpuVitTower.matrix(refs.apply(conv(i, "weight")), c[i], (long) K * K * cIn);
            convB[i] = floats.apply(conv(i, "bias"));
        }
        this.convOut = refs.apply("a.conv_out.weight");
        this.normPreW = floats.apply("mm.a.norm_pre.weight");
        this.normPreB = floats.apply("mm.a.norm_pre.bias");
        this.mm1 = refs.apply("mm.a.mlp.1.weight");
        this.mm1B = floats.apply("mm.a.mlp.1.bias");
        this.mm3 = refs.apply("mm.a.mlp.3.weight");
        this.mm3B = floats.apply("mm.a.mlp.3.bias");
        this.outWidth = model.outputWidth();
    }

    /** {@code a.conv2d.(i + 1).suffix}: the stem convs are numbered from 1. */
    private static String conv(int i, String suffix) {
        return "a.conv2d." + (i + 1) + "." + suffix;
    }

    private static List<ClipLayer> layers(TensorCatalog tc, ClipHparams hp) {
        List<ClipLayer> layers = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            layers.add(ClipLayer.resolve(tc, hp, il, true));
        }
        return layers;
    }

    private static int[] convChannels(TensorCatalog tc) {
        int[] c = new int[CONVS];
        for (int i = 0; i < CONVS; i++) {
            c[i] = (int) tc.require(conv(i, "weight")).ne(3);
        }
        return c;
    }

    /** The mel height after the three stride-2 convs. */
    private static int convHeight(int nMel) {
        int h = nMel;
        for (int i = 0; i < CONVS; i++) {
            h = AudioOps.convOut(h, K, STRIDE, PAD);
        }
        return h;
    }

    /** Shape-checks the conv stem, conv_out, the ViT blocks and the adapter against the hparams. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (hp.dHead() % 4 != 0) {
            throw new IllegalArgumentException("the dots3note tower rotates half of each head in NEOX pairs, the"
                    + " head width " + hp.dHead() + " is not a multiple of 4");
        }
        int[] c = convChannels(tc);
        for (int i = 0; i < CONVS; i++) {
            shape(tc, tc.require(conv(i, "weight")), K, K, i == 0 ? 1 : c[i - 1], c[i]);
            elements(tc, tc.require(conv(i, "bias")), c[i]);
        }
        shape(tc, tc.require("a.conv_out.weight"), (long) convHeight(hp.audio().nMelBins()) * c[CONVS - 1], e);
        checkNorm(tc, "a.pre_ln", tc.get("a.pre_ln.weight"), tc.get("a.pre_ln.bias"), e);
        checkNorm(tc, "a.post_ln", tc.get("a.post_ln.weight"), tc.get("a.post_ln.bias"), e);
        layers(tc, hp);
        shape(tc, tc.require("mm.a.norm_pre.weight"), e);
        shape(tc, tc.require("mm.a.norm_pre.bias"), e);
        TensorRef m1 = tc.require("mm.a.mlp.1.weight");
        long ff = m1.ne(1);
        shape(tc, m1, e, ff);
        shape(tc, tc.require("mm.a.mlp.1.bias"), ff);
        shape(tc, tc.require("mm.a.mlp.3.weight"), ff, hp.projectionDim());
        shape(tc, tc.require("mm.a.mlp.3.bias"), hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static Dots3NoteATower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                                Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new Dots3NoteATower(model, matVec, floats, refs);
    }

    /** The conv stem, the flatten, the tower and the adapter: {@code ceil(nLen / 8)} rows. */
    @Override
    public float[][] encode(MelChunk chunk) {
        int nMel = hp.audio().nMelBins();
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        // the mel-major chunk is the planar [frames, mel, 1] image
        float[] img = chunk.data();
        int w = chunk.nLen();
        int h = nMel;
        int c = 1;
        for (int i = 0; i < CONVS; i++) {
            img = AudioOps.conv2d(matVec, img, w, h, c, convW[i], K, K, STRIDE, PAD, convB[i]);
            AudioOps.activate(ActOp.GELU_ERF, img);
            w = AudioOps.convOut(w, K, STRIDE, PAD);
            h = AudioOps.convOut(h, K, STRIDE, PAD);
            c = (int) convW[i].rows();
        }
        int n = w;
        float[] rows = AudioOps.linear(matVec, convOut, null, AudioOps.flattenFreqMajor(img, w, h, c), n);
        int[] pos = AudioOps.iota(n);
        int dh = hp.dHead();
        float theta = hp.audio().ropeTheta();
        CpuVitTower.Options opts = CpuVitTower.Options.NONE.withPos((x, nTok, nHeads) ->
                AudioOps.ropeNeox(x, nTok, nHeads, dh, dh / 2, pos, theta));
        rows = tower.forward(rows, n, NormKind.RMS, opts);
        AudioOps.layerNorm(rows, n, hp.nEmbd(), normPreW, normPreB, AudioOps.GRAPH_EPS);
        float[] out = AudioOps.ffn(matVec, rows, n, mm1, mm1B, ActOp.GELU_ERF, mm3, mm3B);
        return VisionRows.tokenRows(out, outWidth);
    }
}
