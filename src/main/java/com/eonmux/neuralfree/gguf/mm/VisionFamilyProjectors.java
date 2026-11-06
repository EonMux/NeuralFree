package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The validated projector tensors of the MobileVLM, GLM-Edge, CogVLM, GLM-4V, Youtu-VL, Granite Vision 4.1,
 * Nemotron, Ling3-VL, MiniMax-M3, Muse Glimmer, LFM3-VL, SenseNova-U1, Step3-VL, DeepSeek-V4 / V4.1,
 * DeepSeek-OCR / OCR 2, Gemma 4, MiniCPM-V 4.6, MiMo-VL and Gemma 3n projector graphs (clip.cpp
 * {@code load_tensors} per projector type; the DeepSeek-OCR SAM backbone through {@link SamTensors}),
 * shape-checked where their graphs read them, and the projector output width ({@code clip_n_mmproj_embd}). Static
 * methods only; the counterpart of the shared families' switch in {@link ClipProjectorTensors}.
 */
final class VisionFamilyProjectors {

    /** Kernel side of the MobileVLM depthwise convolutions. */
    static final int DEPTHWISE_KERNEL = 3;

    private VisionFamilyProjectors() {
    }

    /**
     * The tensors of {@code graph} into {@code c}.
     *
     * @throws IllegalArgumentException when a shape differs from the reference loader's, or the hparams cannot
     *                                  drive the graph (named)
     * @throws IllegalStateException    for a projector graph no family resolves
     */
    static ClipProjectorTensors resolve(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp,
                                        String graph) {
        int e = hp.nEmbd();
        switch (graph) {
            case ProjectorEntry.PROJECTOR_LDP:
                return ldp(c, LlavaGraph.featureWidth(hp));
            case ProjectorEntry.PROJECTOR_LDPV2: {
                int d = mlp(c, "mm.model.mlp.0", "mm.model.mlp.2", LlavaGraph.featureWidth(hp));
                c.depthwise("mm.model.peg.0.weight", DEPTHWISE_KERNEL, d).vector("mm.model.peg.0.bias", d);
                return c.build(d);
            }
            case ProjectorEntry.PROJECTOR_GLM_EDGE:
                return glmEdge(c, tc, LlavaGraph.featureWidth(hp));
            case ProjectorEntry.PROJECTOR_COGVLM: {
                int h = c.outDim("mm.model.fc.weight", e);
                c.matrix("mm.model.fc.weight", e, h).vector("mm.post_fc_norm.weight", h)
                        .vector("mm.post_fc_norm.bias", h);
                int out = gated(c, h, false);
                c.vector("v.boi", out).vector("v.eoi", out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_GLM4V:
                return glm4v(c, tc, hp);
            case ProjectorEntry.PROJECTOR_YOUTUVL:
                c.vector("mm.input_norm.weight", e);
                return c.build(mlp(c, "mm.0", "mm.2", (long) e * YoutuVlGraph.MERGE * YoutuVlGraph.MERGE));
            case ProjectorEntry.PROJECTOR_GRANITE4:
                return granite4(c, tc, hp);
            case ProjectorEntry.PROJECTOR_NEMOTRON:
                return nemotron(c, hp);
            case ProjectorEntry.PROJECTOR_LING3VL:
                requireMergeTwo(hp, "ling3vl");
                c.vector("mm.input_norm.weight", e).vector("mm.input_norm.bias", e);
                return c.build(mlp(c, "mm.0", "mm.2", 4L * e));
            case ProjectorEntry.PROJECTOR_MINIMAX_M3:
                return minimaxM3(c, hp);
            case ProjectorEntry.PROJECTOR_MUSE_GLIMMER:
                return museGlimmer(c, hp);
            case ProjectorEntry.PROJECTOR_LFM3VL:
                requireMergeTwo(hp, "lfm3vl");
                Lfm3vlGraph.resolve(c, tc, hp);
                c.vector("mm.input_norm.weight", e).vector("mm.input_norm.bias", e);
                return c.build(mlp(c, "mm.0", "mm.2", 4L * e));
            case ProjectorEntry.PROJECTOR_SENSENOVA:
                return sensenova(c, tc, hp);
            case ProjectorEntry.PROJECTOR_STEP3VL:
                return step3vl(c, tc, hp);
            case ProjectorEntry.PROJECTOR_DEEPSEEK4V:
            case ProjectorEntry.PROJECTOR_DEEPSEEK41V:
                return deepseekV4(c, hp, ProjectorEntry.PROJECTOR_DEEPSEEK4V.equals(graph));
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR:
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR2:
                return deepseekOcr(c, tc, hp, ProjectorEntry.PROJECTOR_DEEPSEEKOCR2.equals(graph));
            case ProjectorEntry.PROJECTOR_GEMMA4V:
                return gemma4v(c, tc, hp);
            case ProjectorEntry.PROJECTOR_GEMMA4UV: {
                int k = hp.patchSize() * hp.nMerge();
                c.vector("v.patch_norm.1.weight", 3L * k * k).vector("v.patch_norm.1.bias", 3L * k * k);
                for (int n = 2; n <= 3; n++) {
                    c.vector("v.patch_norm." + n + ".weight", e).vector("v.patch_norm." + n + ".bias", e);
                }
                int out = c.outDim(Gemma4vGraph.INPUT_PROJECTION, e);
                c.matrix(Gemma4vGraph.INPUT_PROJECTION, e, out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_MINICPMV46_MERGER:
                return minicpmv46(c, tc, hp);
            case ProjectorEntry.PROJECTOR_MIMOVL_MERGER:
                return mimovl(c, tc, hp);
            case ProjectorEntry.PROJECTOR_GEMMA3NV:
                return MobileNetV5Tensors.resolve(c, tc, hp);
            default:
                throw new IllegalStateException("unknown projector graph " + graph);
        }
    }

    /**
     * The MiniCPM-V 4.6 tensors: the ViT merger (required with a 4x merge), the final merger with its optional
     * biases, and the separator rows (required for minicpmv4_7, {@code [output width, n_sep]}).
     */
    private static ClipProjectorTensors minicpmv46(ClipProjectorTensors.Collector c, TensorCatalog tc,
                                                   ClipHparams hp) {
        long e = hp.nEmbd();
        long merged = 4 * e;
        if (hp.nMerge() == 4) {
            for (String norm : new String[]{MiniCpmv46Graph.MERGER_LN1, MiniCpmv46Graph.MERGER_DS_LN}) {
                long width = norm.equals(MiniCpmv46Graph.MERGER_LN1) ? e : merged;
                c.vector(norm + ".weight", width).vector(norm + ".bias", width);
            }
            for (String p : MiniCpmv46Graph.MERGER_PROJECTIONS) {
                String name = MiniCpmv46Graph.MERGER_ATTN + p;
                c.matrix(name + ".weight", e, e).optionalVector(name + ".bias", e);
            }
            optionalBiasMlp(c, MiniCpmv46Graph.MERGER_DS_UP, MiniCpmv46Graph.MERGER_DS_DOWN, merged, e);
        }
        c.vector("mm.input_norm.weight", merged).optionalVector("mm.input_norm.bias", merged);
        int out = optionalBiasMlp(c, "mm.up", "mm.down", merged, -1);
        TensorRef sep = tc.get(MiniCpmv46Graph.SEPARATORS);
        if (sep == null && hp.projectorType() == ProjectorType.MINICPMV4_7) {
            throw new MissingTensorException(tc.archName(), -1, MiniCpmv46Graph.SEPARATORS);
        }
        if (sep != null) {
            if (sep.rank() > 2 || sep.ne(0) != out || sep.ne(1) <= 0) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + sep.name()
                        + " has shape " + sep + ", the loader expects {" + out + ", n_sep} separator rows");
            }
            c.vector(sep.name(), out, sep.ne(1));
        }
        return c.build(out);
    }

    /**
     * An up/down pair with optional biases: {@code up} from {@code in} to its own width, {@code down} from that
     * width to {@code out}, or to its own width when {@code out} is -1; returns the output width.
     */
    private static int optionalBiasMlp(ClipProjectorTensors.Collector c, String up, String down, long in,
                                       long out) {
        int hidden = c.outDim(up + ".weight", in);
        c.matrix(up + ".weight", in, hidden).optionalVector(up + ".bias", hidden);
        int width = out < 0 ? c.outDim(down + ".weight", hidden) : Math.toIntExact(out);
        c.matrix(down + ".weight", hidden, width).optionalVector(down + ".bias", width);
        return width;
    }

    /**
     * The Gemma 4 projector: the input projection with its optional clippable-linear bounds and the optional
     * standardization pair (models/gemma4v.cpp applies it only as a pair, so a lone half is refused by name).
     */
    private static ClipProjectorTensors gemma4v(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        int e = hp.nEmbd();
        if (hp.nMerge() <= 0) {
            throw new IllegalArgumentException("the gemma4v pool needs a positive kernel"
                    + " (clip.vision.projector.scale_factor), got " + hp.nMerge());
        }
        if (tc.has(Gemma4vGraph.STD_BIAS) != tc.has(Gemma4vGraph.STD_SCALE)) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the gemma4v projector"
                    + " standardizes with " + Gemma4vGraph.STD_BIAS + " and " + Gemma4vGraph.STD_SCALE
                    + " together, but the file carries only "
                    + (tc.has(Gemma4vGraph.STD_BIAS) ? Gemma4vGraph.STD_BIAS : Gemma4vGraph.STD_SCALE));
        }
        c.optionalVector(Gemma4vGraph.STD_BIAS, e).optionalVector(Gemma4vGraph.STD_SCALE, e);
        int out = c.outDim(Gemma4vGraph.INPUT_PROJECTION, e);
        c.matrix(Gemma4vGraph.INPUT_PROJECTION, e, out);
        for (String bound : Gemma4vGraph.PROJECTION_BOUNDS) {
            c.optionalVector(bound, 1);
        }
        return c.build(out);
    }

    /**
     * The MiMo-VL merger: {@code mm.0} over every 2x2 group of rows and {@code mm.2}, both bias-free; the output
     * width is {@code mm.2}'s ({@code clip_n_mmproj_embd}). models/mimovl.cpp adds no bias, so a file carrying
     * one is refused by name.
     */
    private static ClipProjectorTensors mimovl(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        for (String bias : new String[]{"mm.0.bias", "mm.2.bias"}) {
            if (tc.has(bias)) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': the mimovl merger adds"
                        + " no bias to mm.0 and mm.2, but the file carries " + bias);
            }
        }
        long in = (long) hp.nEmbd() * MimoVlGraph.MERGE * MimoVlGraph.MERGE;
        int hidden = c.outDim("mm.0.weight", in);
        c.matrix("mm.0.weight", in, hidden);
        int out = c.outDim("mm.2.weight", hidden);
        c.matrix("mm.2.weight", hidden, out);
        return c.build(out);
    }

    /** A biased two-layer MLP {@code up} / {@code down} over {@code in} values; returns its output width. */
    private static int mlp(ClipProjectorTensors.Collector c, String up, String down, long in) {
        int hidden = c.outDim(up + ".weight", in);
        c.matrix(up + ".weight", in, hidden).vector(up + ".bias", hidden);
        int out = c.outDim(down + ".weight", hidden);
        c.matrix(down + ".weight", hidden, out).vector(down + ".bias", out);
        return out;
    }

    /** {@code mm.gate} / {@code mm.up} over {@code in} values and {@code mm.down}; returns the output width. */
    private static int gated(ClipProjectorTensors.Collector c, int in, boolean optionalBiases) {
        int f = c.outDim("mm.up.weight", in);
        c.matrix("mm.up.weight", in, f).matrix("mm.gate.weight", in, f);
        int out = c.outDim("mm.down.weight", f);
        c.matrix("mm.down.weight", f, out);
        if (optionalBiases) {
            c.optionalVector("mm.up.bias", f).optionalVector("mm.gate.bias", f).optionalVector("mm.down.bias", out);
        }
        return out;
    }

    /** The graphs that hard-code a 2x2 merge refuse any other {@code clip.vision.spatial_merge_size}. */
    private static void requireMergeTwo(ClipHparams hp, String family) {
        if (hp.nMerge() != 2) {
            throw new IllegalArgumentException("the " + family + " graph merges 2x2 patches, the file sets"
                    + " clip.vision.spatial_merge_size " + hp.nMerge());
        }
    }

    /** Nemotron: the RMS norm over the shuffled rows, then the bias-free {@code mlp.1} / {@code mlp.3} pair. */
    private static ClipProjectorTensors nemotron(ClipProjectorTensors.Collector c, ClipHparams hp) {
        ClipProjectorTensors.requireMerge(hp, "nemotron");
        long merged = (long) hp.nEmbd() * hp.nMerge() * hp.nMerge();
        c.vector("mm.model.mlp.0.weight", merged);
        int hidden = c.outDim("mm.model.mlp.1.weight", merged);
        c.matrix("mm.model.mlp.1.weight", merged, hidden);
        int out = c.outDim("mm.model.mlp.3.weight", hidden);
        c.matrix("mm.model.mlp.3.weight", hidden, out);
        return c.build(out);
    }

    /** MiniMax-M3: the per-patch {@code mm.1} / {@code mm.2} MLP, then the merger MLP over merged rows. */
    private static ClipProjectorTensors minimaxM3(ClipProjectorTensors.Collector c, ClipHparams hp) {
        int m = hp.nMerge();
        int proj = mlp(c, "mm.1", "mm.2", hp.nEmbd());
        return c.build(mlp(c, "mm.merger.fc1", "mm.merger.fc2", (long) proj * m * m));
    }

    /** Muse Glimmer: three bias-free linears over the {@code n_merge^2}-shuffled rows. */
    private static ClipProjectorTensors museGlimmer(ClipProjectorTensors.Collector c, ClipHparams hp) {
        ClipProjectorTensors.requireMerge(hp, "muse-glimmer");
        long in = (long) hp.nEmbd() * hp.nMerge() * hp.nMerge();
        int h0 = c.outDim("mm.0.weight", in);
        c.matrix("mm.0.weight", in, h0);
        int h1 = c.outDim("mm.1.weight", h0);
        c.matrix("mm.1.weight", h0, h1);
        int out = c.outDim("mm.2.weight", h1);
        c.matrix("mm.2.weight", h1, out);
        return c.build(out);
    }

    private static ClipProjectorTensors ldp(ClipProjectorTensors.Collector c, int in) {
        int d = mlp(c, "mm.model.mlp.1", "mm.model.mlp.3", in);
        for (int blk = 1; blk <= 2; blk++) {
            String p = "mm.model.mb_block." + blk + ".block.";
            int r = c.outDim(p + "1.fc1.weight", d);
            c.depthwise(p + "0.0.weight", DEPTHWISE_KERNEL, d)
                    .vector(p + "0.1.weight", d).vector(p + "0.1.bias", d)
                    .matrix(p + "1.fc1.weight", d, r).vector(p + "1.fc1.bias", r)
                    .matrix(p + "1.fc2.weight", r, d).vector(p + "1.fc2.bias", d)
                    .matrix(p + "2.0.weight", d, d)
                    .vector(p + "2.1.weight", d).vector(p + "2.1.bias", d);
        }
        return c.build(d);
    }

    private static ClipProjectorTensors glmEdge(ClipProjectorTensors.Collector c, TensorCatalog tc, int in) {
        long hc = tc.require("adapter.conv.weight").ne(3);
        c.kernel("adapter.conv.weight", 2, 2, in, hc).vector("adapter.conv.bias", hc);
        int h = c.outDim("adapter.linear.linear.weight", hc);
        c.matrix("adapter.linear.linear.weight", hc, h)
                .vector("adapter.linear.norm1.weight", h).vector("adapter.linear.norm1.bias", h);
        int f = c.outDim("adapter.linear.dense_h_to_4h.weight", h);
        c.matrix("adapter.linear.dense_h_to_4h.weight", h, f).matrix("adapter.linear.gate.weight", h, f);
        int out = c.outDim("adapter.linear.dense_4h_to_h.weight", f);
        c.matrix("adapter.linear.dense_4h_to_h.weight", f, out)
                .vector("adapter.boi", out).vector("adapter.eoi", out);
        return c.build(out);
    }

    private static ClipProjectorTensors glm4v(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        int e = hp.nEmbd();
        int m = Glm4vGraph.MERGE;
        long eo = tc.require("mm.patch_merger.weight").ne(3);
        c.kernel("mm.patch_merger.weight", m, m, e, eo).vector("mm.patch_merger.bias", eo);
        int h = c.outDim("mm.model.fc.weight", eo);
        c.matrix("mm.model.fc.weight", eo, h).vector("mm.post_norm.weight", h).optionalVector("mm.post_norm.bias", h);
        return c.build(gated(c, h, true));
    }

    /**
     * SenseNova-U1: the 2x2 conv {@code mm.0} from {@code n_embd} to {@code projection_dim} channels and its bias
     * (clip.cpp reads the output width from {@code projection_dim}); the graph needs a merge of 2.
     */
    private static ClipProjectorTensors sensenova(ClipProjectorTensors.Collector c, TensorCatalog tc,
                                                  ClipHparams hp) {
        int m = SenseNovaGraph.MERGE;
        if (hp.nMerge() != m) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': SenseNova-U1 requires"
                    + " clip.vision.spatial_merge_size " + m + ", got " + hp.nMerge());
        }
        int out = hp.projectionDim();
        c.kernel("mm.0.weight", m, m, hp.nEmbd(), out).vector("mm.0.bias", out);
        return c.build(out);
    }

    /**
     * Step3-VL: the 3x3 downsamplers {@code mm.0} ({@code n_embd} to {@code c1}) and {@code mm.1} ({@code c1} to
     * {@code c2}) with optional biases, then {@code mm.model.fc} to the output width; the two stride-2 convs merge
     * 4 patches per side, which {@code clip.vision.projector.scale_factor} must state.
     */
    private static ClipProjectorTensors step3vl(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        int k = Step3vlGraph.KERNEL;
        if (hp.nMerge() != Step3vlGraph.MERGE) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the step3vl downsamplers merge "
                    + Step3vlGraph.MERGE + " patches per side, but clip.vision.projector.scale_factor is "
                    + hp.nMerge());
        }
        long c1 = tc.require("mm.0.weight").ne(3);
        c.kernel("mm.0.weight", k, k, hp.nEmbd(), c1).optionalVector("mm.0.bias", c1);
        long c2 = tc.require("mm.1.weight").ne(3);
        c.kernel("mm.1.weight", k, k, c1, c2).optionalVector("mm.1.bias", c2);
        int out = c.outDim("mm.model.fc.weight", c2);
        c.matrix("mm.model.fc.weight", c2, out);
        return c.build(out);
    }

    /**
     * DeepSeek-V4 and V4.1: the aligner MLP over the {@code n_merge x n_merge} unfold and the sentinel rows of the
     * token block (the pad row for V4 only).
     */
    private static ClipProjectorTensors deepseekV4(ClipProjectorTensors.Collector c, ClipHparams hp, boolean pad) {
        long m = hp.nMerge();
        int out = mlp(c, "mm.1", "mm.2", hp.nEmbd() * m * m);
        c.vector("v.token_embd.img_start", out).vector("v.token_embd.img_end", out).vector("v.image_newline", out);
        if (pad) {
            c.vector("v.token_embd.img_pad", out);
        }
        return c.build(out);
    }

    /**
     * DeepSeek-OCR and OCR 2: the SAM backbone ({@link SamTensors}), whose output width must be the tower's,
     * {@code mm.model.fc} over the CLIP and SAM rows (OCR) or the query rows (OCR 2), the view separator, and the
     * newline (OCR) or the learned queries the file carries (OCR 2: 144 of the 768 tiles, 256 of the global view).
     */
    private static ClipProjectorTensors deepseekOcr(ClipProjectorTensors.Collector c, TensorCatalog tc,
                                                    ClipHparams hp, boolean v2) {
        int e = hp.nEmbd();
        int sam = SamTensors.collect(c, tc, hp);
        if (sam != e) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the SAM backbone outputs "
                    + sam + " channels, the tower reads " + e);
        }
        long in = v2 ? e : 2L * e;
        int out = c.outDim("mm.model.fc.weight", in);
        c.matrix("mm.model.fc.weight", in, out).vector("mm.model.fc.bias", out).vector("v.view_seperator", out);
        if (!v2) {
            c.vector("v.image_newline", out);
            return c.build(out);
        }
        boolean any = false;
        for (int[] q : new int[][]{{768, DeepseekOcrGraph.QUERIES_768}, {1024, DeepseekOcrGraph.QUERIES_1024}}) {
            String name = "v.resample_query_" + q[0] + ".weight";
            if (tc.has(name)) {
                c.vector(name, e, q[1]);
                any = true;
            }
        }
        if (!any) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the deepseekocr2 projector"
                    + " carries neither v.resample_query_768.weight nor v.resample_query_1024.weight");
        }
        return c.build(out);
    }

    private static ClipProjectorTensors granite4(ClipProjectorTensors.Collector c, TensorCatalog tc,
                                                 ClipHparams hp) {
        int e = hp.nEmbd();
        int p = hp.projectionDim();
        if (e % Granite4VisionGraph.HEAD_DIM != 0 || p <= 0) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the granite4_vision QFormer"
                    + " needs n_embd a multiple of " + Granite4VisionGraph.HEAD_DIM + " and a positive"
                    + " projection_dim, got " + e + " and " + p);
        }
        VisionFamilyHparams fam = hp.visionFamily();
        int side = hp.imageSize() / hp.patchSize();
        int win = fam.windowSide();
        if (side % win != 0) {
            throw new IllegalArgumentException("clip.vision.projector.window_side " + win + " does not divide the"
                    + " patch grid " + side);
        }
        int newSide = Granite4VisionGraph.outSide(hp, side);
        int[] offsets = fam.spatialOffsets();
        c.vector("v.image_newline", p);
        for (int bid = 0; bid < offsets.length; bid++) {
            boolean fits = offsets[bid] >= 0 ? 2 * newSide == side : newSide <= side && side % newSide == 0;
            if (!fits) {
                throw new IllegalArgumentException("granite4_vision projector " + bid + " downsamples the "
                        + side + "-patch grid to " + newSide + " per side, which its "
                        + (offsets[bid] >= 0 ? "2x2 sampling" : "average pool") + " cannot produce");
            }
            qformerBlock(c, bid, e, p, win, fam.querySide());
        }
        return c.build(offsets.length * p);
    }

    private static void qformerBlock(ClipProjectorTensors.Collector c, int bid, int e, int p, int win, int qs) {
        c.vector(Granite4VisionGraph.blockTensor(bid, "img_pos"), e, (long) win * win)
                .vector(Granite4VisionGraph.blockTensor(bid, "query"), e, (long) qs * qs);
        for (String norm : new String[]{"norm", "post_norm", "self_attn_norm", "cross_attn_norm", "ffn_norm"}) {
            c.vector(Granite4VisionGraph.blockTensor(bid, norm + ".weight"), e)
                    .vector(Granite4VisionGraph.blockTensor(bid, norm + ".bias"), e);
        }
        for (String attn : new String[]{"self_attn_", "cross_attn_"}) {
            for (String proj : new String[]{"q", "k", "v", "out"}) {
                c.matrix(Granite4VisionGraph.blockTensor(bid, attn + proj + ".weight"), e, e)
                        .vector(Granite4VisionGraph.blockTensor(bid, attn + proj + ".bias"), e);
            }
        }
        String up = Granite4VisionGraph.blockTensor(bid, "ffn_up.weight");
        int f = c.outDim(up, e);
        c.matrix(up, e, f).vector(Granite4VisionGraph.blockTensor(bid, "ffn_up.bias"), f)
                .matrix(Granite4VisionGraph.blockTensor(bid, "ffn_down.weight"), f, e)
                .vector(Granite4VisionGraph.blockTensor(bid, "ffn_down.bias"), e)
                .matrix(Granite4VisionGraph.blockTensor(bid, "linear.weight"), e, p)
                .vector(Granite4VisionGraph.blockTensor(bid, "linear.bias"), p);
    }
}
