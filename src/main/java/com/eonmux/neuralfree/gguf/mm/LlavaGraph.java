package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * The llava graph (models/llava.cpp {@code clip_graph_llava}) shared by the mlp, mlp_norm, ldp, ldpv2 and
 * glm-edge families: the patches of the square {@code image_size} tile with the class token appended last, the
 * learned position rows, the optional layer-norm pre-norm, then the blocks up to the deepest feature layer
 * ({@code clip.vision.feature_layer}; without it the second to last block, or every block for glm-edge), the
 * optional post-norm, and the feature layers stacked along the channels (Granite Vision). The llava projectors
 * then read {@code n_patches} rows gathered from row {@code 1} on when a class token exists, as clip.cpp fills
 * the {@code patches} input; glm-edge reads every row. Also the GLM-Edge adapter and the LLaVA MLP projectors.
 * The graph is written once against {@link VisionOps} and runs on the CPU and the GPU executor. Package-private
 * executor support.
 */
final class LlavaGraph {

    private LlavaGraph() {
    }

    /** The encoder rows of the tile, stacked feature layers, gathered for the llava projectors. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        ClipVisionModel model = ops.model();
        ClipVit.requireImageSizeGrid(model, grid[0], grid[1]);
        int nEmbd = hp.nEmbd();
        int nPos = model.nPos();
        VisionOps.Block x = ops.patchRows(image, grid[0], grid[1]);
        float[] classEmbd = ops.hostWeight(model.classEmbd());
        if (classEmbd != null) {
            x = ops.insertRow(x, model.nPatches(), classEmbd);
        }
        ops.addHost(x, ops.hostWeight(model.posEmbd()));

        int max = maxFeatureLayer(hp);
        TreeSet<Integer> features = stackedLayers(hp);
        List<float[]> stack = new ArrayList<>();
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE.withLayers(max);
        if (!features.isEmpty()) {
            opts = opts.withTap((done, cur, n) -> {
                if (done < max && features.contains(done)) {
                    stack.add(ops.download(cur).clone());
                }
            });
        }
        x = ops.tower(x, nPos, NormKind.LAYER, opts);
        if (!stack.isEmpty()) {
            if (features.contains(max)) {
                stack.add(ops.download(x).clone());
            }
            List<VisionOps.Block> blocks = new ArrayList<>(stack.size());
            for (float[] part : stack) {
                blocks.add(ops.upload(part, nEmbd));
            }
            x = ops.concatChannels(blocks, nPos, nEmbd);
        }
        int width = featureWidth(hp);
        if (!hasLlavaProjector(hp.projectorType())) {
            return new Encoded(ops.download(x), nPos, grid[0], grid[1]);
        }
        int offset = classEmbd != null ? 1 : 0;
        float[] rows = ops.download(x);
        int n = model.nPatches();
        float[] picked = new float[n * width];
        System.arraycopy(rows, offset * width, picked, 0, n * width);
        return new Encoded(picked, n, grid[0], grid[1]);
    }

    /**
     * The block count the graph runs: the deepest feature layer, or without feature layers {@code n_layer - 1}
     * ({@code n_layer} for glm-edge and minicpmv).
     */
    static int maxFeatureLayer(ClipHparams hp) {
        int ilLast = hp.nLayer() - 1;
        if (hp.projectorType() == ProjectorType.GLM_EDGE || hp.projectorType() == ProjectorType.MINICPMV) {
            ilLast++;
        }
        int deepest = -1;
        for (int f : hp.featureLayers()) {
            deepest = Math.max(deepest, f);
        }
        return deepest < 0 ? ilLast : deepest;
    }

    /** The feature layers that stack ({@code is_feature_layer} over {@code [0, max]}), ascending. */
    static TreeSet<Integer> stackedLayers(ClipHparams hp) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int f : hp.featureLayers()) {
            if (f >= 0) {
                s.add(f);
            }
        }
        return s;
    }

    /** Width of the encoder rows the projector reads: {@code n_embd} per stacked feature layer, at least one. */
    static int featureWidth(ClipHparams hp) {
        return hp.nEmbd() * Math.max(1, stackedLayers(hp).size());
    }

    /** The families whose projector reads the gathered patch rows ({@code has_llava_projector}). */
    static boolean hasLlavaProjector(ProjectorType type) {
        switch (type) {
            case MLP:
            case MLP_NORM:
            case LDP:
            case LDPV2:
                return true;
            default:
                return false;
        }
    }

    /** LLaVA MLP: {@code mm.0} + GELU + the optional {@code mm.2}. */
    static float[][] mlp(VisionOps ops, Encoded enc) {
        VisionOps.Block h = ops.linear("mm.0.weight", ops.upload(enc.rows, enc.width), "mm.0.bias");
        ops.activate(h, ActOp.GELU_TANH);
        if (ops.hasMatrix("mm.2.weight")) {
            h = ops.linear("mm.2.weight", h, "mm.2.bias");
        }
        return VisionRows.tokenRows(ops.download(h), ops.model().outputDim());
    }

    /** LLaVA MLP with norms: {@code mm.0}, norm, GELU, {@code mm.3}, norm. */
    static float[][] mlpNorm(VisionOps ops, Encoded enc) {
        int n = enc.nTokens;
        int d = ops.model().outputDim();
        float eps = ops.hp().eps();
        VisionOps.Block h = ops.linear("mm.0.weight", ops.upload(enc.rows, enc.width), "mm.0.bias");
        ops.norm(h, NormKind.LAYER, "mm.1.weight", "mm.1.bias", eps, n, d);
        ops.activate(h, ActOp.GELU_TANH);
        h = ops.linear("mm.3.weight", h, "mm.3.bias");
        ops.norm(h, NormKind.LAYER, "mm.4.weight", "mm.4.bias", eps, n, d);
        return VisionRows.tokenRows(ops.download(h), d);
    }

    /**
     * The GLM-Edge adapter: the rows as a square image, a 2x2 stride-2 conv ({@code adapter.conv}), the linear
     * layer, layer norm, GELU, the SwiGLU of {@code gate} and {@code dense_h_to_4h}, {@code dense_4h_to_h}, then
     * the BOI and EOI rows around the image.
     */
    static float[][] glmEdge(VisionOps ops, Encoded enc) {
        int side = (int) Math.round(Math.sqrt(enc.nTokens));
        if (side * side != enc.nTokens || side % 2 != 0) {
            throw new IllegalArgumentException("the glm-edge adapter needs a square grid of even side, the"
                    + " encoder produced " + enc.nTokens + " rows");
        }
        int nEmbd = enc.width;
        int n = (side / 2) * (side / 2);
        VisionOps.Block cols = ops.unfold(ops.upload(enc.rows, nEmbd), nEmbd, side, side, 2);
        int convIn = 4 * nEmbd;
        int convOut = ops.hostVector("adapter.conv.bias").length;
        VisionOps.Block h = ops.linearKernel("adapter.conv.weight", convOut, convIn, cols, "adapter.conv.bias");
        h = ops.linear("adapter.linear.linear.weight", h, null);
        int width = h.width();
        ops.norm(h, NormKind.LAYER, "adapter.linear.norm1.weight", "adapter.linear.norm1.bias", ops.hp().eps(),
                n, width);
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block gate = ops.linear("adapter.linear.gate.weight", h, null);
        VisionOps.Block up = ops.linear("adapter.linear.dense_h_to_4h.weight", h, null);
        ops.glu(gate, up, GluOp.SWIGLU, n, up.width());
        VisionOps.Block out = ops.linear("adapter.linear.dense_4h_to_h.weight", up, null);
        return VisionRows.framed(ops.hostVector("adapter.boi"), ops.download(out), ops.hostVector("adapter.eoi"),
                ops.model().outputDim());
    }
}
