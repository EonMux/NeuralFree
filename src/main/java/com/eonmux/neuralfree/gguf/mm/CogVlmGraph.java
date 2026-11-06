package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.Arrays;

/**
 * The CogVLM graph (models/cogvlm.cpp {@code clip_graph_cogvlm}): the patches of the square {@code image_size}
 * tile with the class token appended last and the learned position rows, post-norm blocks over a fused Q/K/V
 * (layer norm after attention and after the FFN, each before its residual), the class token dropped, then
 * {@code mm.model.fc}, layer norm, GELU, the SwiGLU of {@code mm.gate} and {@code mm.up}, {@code mm.down}, and the
 * BOI and EOI rows around the image. Written once against {@link VisionOps}. Package-private executor support.
 */
final class CogVlmGraph {

    /** Epsilon of the projector's layer norm (nn.LayerNorm default). */
    static final float PROJECTOR_EPS = 1e-5f;

    private CogVlmGraph() {
    }

    /** The patch rows of the tile after the encoder, class token removed, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        ClipVit.requireImageSizeGrid(ops.model(), grid[0], grid[1]);
        int nEmbd = ops.hp().nEmbd();
        int nPatches = ops.model().nPatches();
        int nPos = ops.model().nPos();
        VisionOps.Block x = ops.patchRows(image, grid[0], grid[1]);
        x = ops.insertRow(x, nPatches, ops.hostWeight(ops.model().classEmbd()));
        ops.addHost(x, ops.hostWeight(ops.model().posEmbd()));
        x = ops.tower(x, nPos, NormKind.LAYER, VisionOps.TowerOptions.NONE.withPostNorm());
        float[] rows = ops.download(x);
        return new Encoded(Arrays.copyOf(rows, nPatches * nEmbd), nPatches, grid[0], grid[1]);
    }

    /** The GLU projector and the BOI / EOI rows. */
    static float[][] project(VisionOps ops, Encoded enc) {
        int n = enc.nTokens;
        VisionOps.Block h = ops.linear("mm.model.fc.weight", ops.upload(enc.rows, enc.width), null);
        int width = h.width();
        ops.norm(h, NormKind.LAYER, "mm.post_fc_norm.weight", "mm.post_fc_norm.bias", PROJECTOR_EPS, n, width);
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block gate = ops.linear("mm.gate.weight", h, null);
        VisionOps.Block up = ops.linear("mm.up.weight", h, null);
        ops.glu(gate, up, GluOp.SWIGLU, n, up.width());
        VisionOps.Block out = ops.linear("mm.down.weight", up, null);
        return VisionRows.framed(ops.hostVector("v.boi"), ops.download(out), ops.hostVector("v.eoi"),
                ops.model().outputDim());
    }
}
