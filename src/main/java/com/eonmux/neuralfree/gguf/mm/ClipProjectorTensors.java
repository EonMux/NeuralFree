package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The validated projector tensors of one projector graph (the per-type {@code get_tensor} lists of llama.cpp
 * clip.cpp {@code load_tensors}, shape-checked where the graphs read them) and the projector output width
 * ({@code clip_n_mmproj_embd}). Every tensor is filed as a matrix (read by a matmul in its stored type) or a
 * vector (read as floats: biases, norm weights, embedding rows), so the executor loads it once in the form it
 * reads. Immutable.
 */
final class ClipProjectorTensors {

    private final Map<String, TensorRef> matrices;
    private final Map<String, TensorRef> vectors;
    private final int outputDim;

    private ClipProjectorTensors(Map<String, TensorRef> matrices, Map<String, TensorRef> vectors, int outputDim) {
        this.matrices = Collections.unmodifiableMap(matrices);
        this.vectors = Collections.unmodifiableMap(vectors);
        this.outputDim = outputDim;
    }

    /** Tensors read by a matmul, by name (unmodifiable). */
    Map<String, TensorRef> matrices() {
        return matrices;
    }

    /** Tensors read as float values, by name (unmodifiable). */
    Map<String, TensorRef> vectors() {
        return vectors;
    }

    /** Width of every output row, the text model's input embedding width. */
    int outputDim() {
        return outputDim;
    }

    /**
     * The tensors of {@code projectorGraph} over {@code tc}.
     *
     * @throws MissingTensorException   when a tensor the graph reads is absent
     * @throws IllegalArgumentException when a shape differs from the reference loader's, or the hparams cannot
     *                                  drive the graph (named)
     */
    static ClipProjectorTensors resolve(TensorCatalog tc, ClipHparams hp, String projectorGraph) {
        Collector c = new Collector(tc);
        int d = hp.projectionDim();
        int e = hp.nEmbd();
        int merged = e * hp.nMerge() * hp.nMerge();
        int fw = LlavaGraph.featureWidth(hp);
        switch (projectorGraph) {
            case ProjectorEntry.PROJECTOR_LLAVA_MLP:
                c.matrix("mm.0.weight", fw, d).vector("mm.0.bias", d);
                if (tc.has("mm.2.weight")) {
                    c.matrix("mm.2.weight", d, d).vector("mm.2.bias", d);
                }
                return c.build(d);
            case ProjectorEntry.PROJECTOR_LLAVA_MLP_NORM:
                c.matrix("mm.0.weight", fw, d).vector("mm.0.bias", d)
                        .vector("mm.1.weight", d).vector("mm.1.bias", d)
                        .matrix("mm.3.weight", d, d).vector("mm.3.bias", d)
                        .vector("mm.4.weight", d).vector("mm.4.bias", d);
                return c.build(d);
            case ProjectorEntry.PROJECTOR_QWEN2VL_MERGER:
                return mlp(c, "mm.0", "mm.2", 4L * e);
            case ProjectorEntry.PROJECTOR_QWEN3VL_MERGER:
                return qwen3vl(c, tc, hp);
            case ProjectorEntry.PROJECTOR_PADDLEOCR:
            case ProjectorEntry.PROJECTOR_KIMIVL:
            case ProjectorEntry.PROJECTOR_KIMIK25:
                requireMerge(hp, projectorGraph);
                c.vector("mm.input_norm.weight", e).vector("mm.input_norm.bias", e);
                return mlp(c, "mm.1", "mm.2", merged);
            case ProjectorEntry.PROJECTOR_DOTS:
                requireMerge(hp, projectorGraph);
                c.vector("mm.input_norm.weight", e).vector("mm.input_norm.bias", e);
                return mlp(c, "mm.0", "mm.2", merged);
            case ProjectorEntry.PROJECTOR_HUNYUANVL:
                return hunyuanvl(c, tc, hp);
            case ProjectorEntry.PROJECTOR_JANUS:
                return mlp(c, "mm.0", "mm.1", e);
            case ProjectorEntry.PROJECTOR_LFM2:
                requireMerge(hp, projectorGraph);
                c.optionalVector("mm.input_norm.weight", merged).optionalVector("mm.input_norm.bias", merged);
                return mlp(c, "mm.1", "mm.2", merged);
            case ProjectorEntry.PROJECTOR_COHERE2V:
                return cohere2v(c, tc, hp);
            case ProjectorEntry.PROJECTOR_LIGHTONOCR:
                return pixtral(c, tc, hp, false);
            case ProjectorEntry.PROJECTOR_GEMMA3:
                c.matrix("mm.input_projection.weight", e, d).vector("mm.soft_emb_norm.weight", e);
                return c.build(d);
            case ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER:
                return resampler(c, tc, hp);
            case ProjectorEntry.PROJECTOR_INTERNVL: {
                c.vector("mm.model.mlp.0.weight", merged).vector("mm.model.mlp.0.bias", merged);
                int hidden = c.outDim("mm.model.mlp.1.weight", merged);
                c.matrix("mm.model.mlp.1.weight", merged, hidden).vector("mm.model.mlp.1.bias", hidden);
                int out = c.outDim("mm.model.mlp.3.weight", hidden);
                c.matrix("mm.model.mlp.3.weight", hidden, out).vector("mm.model.mlp.3.bias", out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_PIXTRAL:
                return pixtral(c, tc, hp, true);
            case ProjectorEntry.PROJECTOR_IDEFICS3: {
                requireMerge(hp, "idefics3");
                int out = c.outDim("mm.model.fc.weight", merged);
                c.matrix("mm.model.fc.weight", merged, out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_LLAMA4: {
                int h1 = c.outDim("mm.model.mlp.1.weight", merged);
                c.matrix("mm.model.mlp.1.weight", merged, h1);
                int h2 = c.outDim("mm.model.mlp.2.weight", h1);
                c.matrix("mm.model.mlp.2.weight", h1, h2);
                int out = c.outDim("mm.model.fc.weight", h2);
                c.matrix("mm.model.fc.weight", h2, out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_PHI4: {
                int hidden = c.outDim("mm.0.weight", e);
                c.matrix("mm.0.weight", e, hidden).vector("mm.0.bias", hidden);
                int out = c.outDim("mm.2.weight", hidden);
                c.matrix("mm.2.weight", hidden, out).vector("mm.2.bias", out);
                return c.build(out);
            }
            case ProjectorEntry.PROJECTOR_YASA2:
                return Yasa2Graph.resolve(c, tc, hp);
            default:
                return VisionFamilyProjectors.resolve(c, tc, hp, projectorGraph);
        }
    }

    /** No projector tensors, rows of {@code width} out ({@link ProjectorEntry#PROJECTOR_NONE}). */
    static ClipProjectorTensors none(int width) {
        return new ClipProjectorTensors(new LinkedHashMap<>(), new LinkedHashMap<>(), width);
    }

    /** Head width of the resampler attention, fixed by the reference graph. */
    static final int RESAMPLER_HEAD_DIM = 128;

    private static ClipProjectorTensors resampler(Collector c, TensorCatalog tc, ClipHparams hp) {
        TensorRef proj = tc.require("resampler.proj.weight");
        long d = proj.ne(0);
        if (d <= 0 || d % RESAMPLER_HEAD_DIM != 0) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': resampler width " + d
                    + " (resampler.proj.weight) is not a multiple of the resampler head width "
                    + RESAMPLER_HEAD_DIM);
        }
        int nQuery = hp.minicpmvQueryNum();
        if (nQuery <= 0) {
            throw new IllegalArgumentException("the resampler needs clip.minicpmv_query_num > 0, got " + nQuery);
        }
        TensorRef posK = tc.require("resampler.pos_embed_k");
        if (posK.ne(0) != d) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + posK.name()
                    + " has shape " + posK + ", the resampler width is " + d);
        }
        c.matrix("resampler.proj.weight", d, d)
                .vector("resampler.query", d, nQuery)
                .matrix("resampler.kv.weight", hp.nEmbd(), d);
        for (String p : new String[]{"q", "k", "v", "out"}) {
            c.matrix("resampler.attn." + p + ".weight", d, d).vector("resampler.attn." + p + ".bias", d);
        }
        for (String p : new String[]{"q", "kv", "post"}) {
            c.vector("resampler.ln_" + p + ".weight", d).vector("resampler.ln_" + p + ".bias", d);
        }
        return c.build((int) d);
    }

    /** A biased two-layer MLP {@code up} / {@code down} over {@code in}-wide rows; widths from the tensors. */
    private static ClipProjectorTensors mlp(Collector c, String up, String down, long in) {
        int hidden = c.outDim(up + ".weight", in);
        c.matrix(up + ".weight", in, hidden).vector(up + ".bias", hidden);
        int out = c.outDim(down + ".weight", hidden);
        c.matrix(down + ".weight", hidden, out).vector(down + ".bias", out);
        return c.build(out);
    }

    /**
     * The Qwen3-VL merger over {@code 4 * n_embd} rows and the deepstack blocks of the layers that carry them
     * (norm, fc1, fc2 over {@code n_merge^2 * n_embd} rows); the output width is the merger width times one plus
     * the deepstack count (clip_n_mmproj_embd).
     */
    private static ClipProjectorTensors qwen3vl(Collector c, TensorCatalog tc, ClipHparams hp) {
        if (hp.nMerge() != 2) {
            throw new IllegalArgumentException("the qwen3vl merger merges 2x2 patches, the file sets"
                    + " clip.vision.spatial_merge_size " + hp.nMerge());
        }
        long in = 4L * hp.nEmbd();
        int hidden = c.outDim("mm.0.weight", in);
        c.matrix("mm.0.weight", in, hidden).vector("mm.0.bias", hidden);
        int out = c.outDim("mm.2.weight", hidden);
        c.matrix("mm.2.weight", hidden, out).vector("mm.2.bias", out);
        int deepstack = 0;
        for (int il = 0; il < hp.nLayer(); il++) {
            String p = "v.deepstack." + il + ".";
            if (!tc.has(p + "fc1.weight") && !tc.has(p + "norm.weight") && !tc.has(p + "fc2.weight")) {
                continue;
            }
            c.vector(p + "norm.weight", in).vector(p + "norm.bias", in);
            int h = c.outDim(p + "fc1.weight", in);
            c.matrix(p + "fc1.weight", in, h).vector(p + "fc1.bias", h)
                    .matrix(p + "fc2.weight", h, out).vector(p + "fc2.bias", out);
            deepstack++;
        }
        return c.build(Math.multiplyExact(out, 1 + deepstack));
    }

    /**
     * The HunyuanVL perceiver: {@code mm.0} a {@code n_merge x n_merge} stride-{@code n_merge} conv from
     * {@code n_embd} channels, {@code mm.2} a pointwise conv, the newline column, {@code mm.model.fc} to the
     * output width, the begin/end rows and the closing norm.
     */
    private static ClipProjectorTensors hunyuanvl(Collector c, TensorCatalog tc, ClipHparams hp) {
        int e = hp.nEmbd();
        int s = hp.nMerge();
        TensorRef conv0 = tc.require("mm.0.weight");
        long c1 = conv0.ne(3);
        c.vector("mm.pre_norm.weight", e).conv("mm.0.weight", s, s, e, c1).vector("mm.0.bias", c1);
        TensorRef conv1 = tc.require("mm.2.weight");
        long c2 = conv1.ne(3);
        c.conv("mm.2.weight", 1, 1, c1, c2).vector("mm.2.bias", c2).vector("v.image_newline", c2);
        int out = c.outDim("mm.model.fc.weight", c2);
        c.matrix("mm.model.fc.weight", c2, out).vector("mm.model.fc.bias", out)
                .vector("mm.image_begin", out).vector("mm.image_end", out).vector("mm.post_norm.weight", out);
        return c.build(out);
    }

    /** Cohere2 vision: {@code mm.1} to {@code [x, gate]} halves, the swapped SwiGLU, then {@code mm.2}. */
    private static ClipProjectorTensors cohere2v(Collector c, TensorCatalog tc, ClipHparams hp) {
        requireMerge(hp, "cohere2v");
        long merged = (long) hp.nEmbd() * hp.nMerge() * hp.nMerge();
        int twice = c.outDim("mm.1.weight", merged);
        if (twice % 2 != 0) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor mm.1.weight has an"
                    + " odd output width " + twice + ", the swapped SwiGLU splits it in halves");
        }
        c.matrix("mm.1.weight", merged, twice).vector("mm.1.bias", twice);
        int out = c.outDim("mm.2.weight", twice / 2);
        c.matrix("mm.2.weight", twice / 2, out).vector("mm.2.bias", out);
        return c.build(out);
    }

    private static ClipProjectorTensors pixtral(Collector c, TensorCatalog tc, ClipHparams hp, boolean imgBreak) {
        int e = hp.nEmbd();
        int s = hp.nMerge();
        boolean merger = tc.has("mm.patch_merger.weight");
        if (merger) {
            if (!tc.has("mm.input_norm.weight")) {
                throw new MissingTensorException(tc.archName(), -1, "mm.input_norm.weight");
            }
            c.vector("mm.input_norm.weight", e).matrix("mm.patch_merger.weight", (long) e * s * s, e);
        } else if (s > 1) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': clip.vision.spatial_merge_size "
                    + s + " needs the patch merger mm.patch_merger.weight, which the file does not carry");
        }
        int hidden = c.outDim("mm.1.weight", e);
        c.matrix("mm.1.weight", e, hidden).optionalVector("mm.1.bias", hidden);
        int out = c.outDim("mm.2.weight", hidden);
        c.matrix("mm.2.weight", hidden, out).optionalVector("mm.2.bias", out);
        if (imgBreak) {
            c.vector("v.token_embd.img_break", out);
        }
        return c.build(out);
    }

    static void requireMerge(ClipHparams hp, String family) {
        if (hp.nMerge() <= 1) {
            throw new IllegalArgumentException("the " + family + " pixel shuffle needs"
                    + " clip.vision.projector.scale_factor > 1, got " + hp.nMerge());
        }
    }

    /** Shape-checked collection of one graph's tensors. */
    static final class Collector {
        private final TensorCatalog tc;
        private final Map<String, TensorRef> matrices = new LinkedHashMap<>();
        private final Map<String, TensorRef> vectors = new LinkedHashMap<>();

        Collector(TensorCatalog tc) {
            this.tc = tc;
        }

        Collector matrix(String name, long in, long out) {
            TensorRef r = tc.require(name);
            ClipShapes.shape(tc, r, in, out);
            matrices.put(name, r);
            return this;
        }

        Collector vector(String name, long... ne) {
            TensorRef r = tc.require(name);
            ClipShapes.shape(tc, r, ne);
            vectors.put(name, r);
            return this;
        }

        /** A convolution kernel read by an im2col matmul, filed as a matrix; {@code ne} is its ggml shape. */
        Collector conv(String name, long... ne) {
            TensorRef r = tc.require(name);
            ClipShapes.shape(tc, r, ne);
            matrices.put(name, r);
            return this;
        }

        Collector optionalVector(String name, long n) {
            if (tc.has(name)) {
                vector(name, n);
            }
            return this;
        }

        /**
         * A conv kernel read by a matmul over its im2col input: every listed dim checked, filed as a matrix the
         * executor re-wraps to {@code [out, in]}.
         */
        Collector kernel(String name, long... ne) {
            TensorRef r = tc.require(name);
            ClipShapes.shape(tc, r, ne);
            matrices.put(name, r);
            return this;
        }

        /**
         * A depthwise {@code k x k} conv kernel over {@code channels} channels ({@code ggml_conv_2d_dw}: the first
         * two dims {@code k}, {@code k * k * channels} values in all), read as floats.
         */
        Collector depthwise(String name, int k, long channels) {
            TensorRef r = tc.require(name);
            if (r.ne(0) != k || r.ne(1) != k) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + name
                        + " has shape " + r + ", the loader expects a " + k + "x" + k + " depthwise kernel");
            }
            ClipShapes.elements(tc, r, (long) k * k * channels);
            vectors.put(name, r);
            return this;
        }

        /** The output width of matrix {@code name}, after checking its input width. */
        int outDim(String name, long in) {
            TensorRef r = tc.require(name);
            if (r.ne(0) != in || r.ne(1) <= 0 || r.ne(1) > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + name
                        + " has shape " + r + ", the loader expects an input width of " + in);
            }
            return (int) r.ne(1);
        }

        ClipProjectorTensors build(int outputDim) {
            if (outputDim <= 0) {
                throw new IllegalArgumentException("the projector output width must be positive, got " + outputDim);
            }
            return new ClipProjectorTensors(new LinkedHashMap<>(matrices), new LinkedHashMap<>(vectors), outputDim);
        }
    }
}
