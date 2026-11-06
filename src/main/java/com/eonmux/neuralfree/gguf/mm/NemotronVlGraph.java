package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.Arrays;

/**
 * The Nemotron Nano VL graphs (models/nemotron-v2-vl.cpp {@code clip_graph_nemotron_v2_vl} and
 * models/nemotron-omni.cpp {@code clip_graph_nemotron_omni}): the patch conv plus the optional patch bias; the
 * learned position rows added as stored (v2-VL, whose rows are the {@code image_size} grid) or resized to the patch
 * grid by the half-pixel bilinear kernel (Omni); the register rows of {@code v.class_embd} prepended; the
 * layer-norm ViT; the registers dropped; then the projector: {@code build_patch_merge_permute}, the RMS norm on
 * {@code mm.model.mlp.0} (eps 1e-6 for v2-VL, 1e-5 for Omni), {@code mlp.1}, ReLU^2 and {@code mlp.3}, all bias-free.
 * Written once against {@link VisionOps}. Package-private executor support.
 */
final class NemotronVlGraph {

    /** The projector norm epsilon of nemotron-v2-vl.cpp. */
    static final float V2_VL_NORM_EPS = 1e-6f;
    /** The projector norm epsilon of nemotron-omni.cpp. */
    static final float OMNI_NORM_EPS = 1e-5f;

    private NemotronVlGraph() {
    }

    /**
     * The encoder rows of {@code image}: the {@code pw * ph} patch rows after the tower, registers dropped.
     *
     * @throws IllegalArgumentException when a v2-VL image is not the {@code image_size} square (named)
     */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        int nEmbd = hp.nEmbd();
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        ops.addHost(x, positions(ops, pw, ph));
        float[] registers = ops.hostWeight(ops.model().classEmbd());
        int nReg = registers.length / nEmbd;
        for (int r = 0; r < nReg; r++) {
            x = ops.insertRow(x, r, Arrays.copyOfRange(registers, r * nEmbd, (r + 1) * nEmbd));
        }
        x = ops.tower(x, n + nReg, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        int[] patches = new int[n];
        for (int t = 0; t < n; t++) {
            patches[t] = nReg + t;
        }
        return new Encoded(ops.download(ops.gatherRows(x, patches, nEmbd)), n, pw, ph);
    }

    /** The position rows of the {@code pw x ph} grid, raster order. */
    private static float[] positions(VisionOps ops, int pw, int ph) {
        int nEmbd = ops.hp().nEmbd();
        float[] pos = ops.hostWeight(ops.model().posEmbd());
        if (ops.hp().projectorType() == ProjectorType.NEMOTRON_V2_VL) {
            ClipVit.requireImageSizeGrid(ops.model(), pw, ph);
            return pos;
        }
        return SiglipResizedGraph.halfPixel(pos, nEmbd, SiglipResizedGraph.side(pos, nEmbd), pw, ph);
    }

    /** The projector: {@code ceil(pw / s) * ceil(ph / s)} rows of the output width. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the nemotron projector");
        ClipHparams hp = ops.hp();
        int s = hp.nMerge();
        VisionOps.Block x = ops.mergePermute(ops.upload(enc.rows, enc.width), hp.nEmbd(), enc.pw, enc.ph, s);
        float eps = hp.projectorType() == ProjectorType.NEMOTRON_V2_VL ? V2_VL_NORM_EPS : OMNI_NORM_EPS;
        ops.norm(x, NormKind.RMS, "mm.model.mlp.0.weight", null, eps, x.n(), x.width());
        VisionOps.Block h = ops.linear("mm.model.mlp.1.weight", x, null);
        ops.activate(h, ActOp.RELU_SQR);
        VisionOps.Block out = ops.linear("mm.model.mlp.3.weight", h, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}
