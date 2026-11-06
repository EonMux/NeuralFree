package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkNorm;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The text tower of a CLIP two-tower file and its projection, vLLM's {@code CLIPTextTransformer} plus
 * {@code text_projection} (vllm/model_executor/models/clip.py), or {@code SiglipTextTransformer} with its
 * {@code head} (vllm/model_executor/models/siglip.py): token rows ({@code t.token_embd}) plus learned position rows
 * ({@code t.position_embd}), the shared pre-norm block of {@link CpuVitTower} under a causal mask (HF
 * {@code CLIPTextTransformer} attends causally) or bidirectionally (SigLIP's encoder-only attention), the final
 * layer norm ({@code t.post_ln}), one row pooled by the model's pooling kind (LAST: the end-of-text token the
 * tokenizer appends; CLS: the first; MEAN: the average), the optional text head ({@code t.head}, a linear layer
 * with bias) and the optional bias-free {@code text_projection}. Matmuls run in the stored weight types.
 * Immutable weights; not thread-safe across calls. Package-private executor of {@link ClipDualEncoder}.
 */
final class ClipTextTower {

    static final String TOKEN_EMBD = "t.token_embd.weight";
    static final String POSITION_EMBD = "t.position_embd.weight";
    static final String POST_LN = "t.post_ln";
    static final String PROJECTION = "text_projection.weight";
    static final String HEAD = "t.head";

    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final CpuTensorRef tokenEmbd;
    private final float[] posEmbd;
    private final int nCtx;
    private final boolean causal;
    private final CpuTensorRef head;
    private final float[] headBias;
    private final CpuTensorRef projection;
    private final CpuVitTower tower;

    private ClipTextTower(ClipHparams hp, CpuMatVec matVec, CpuTensorRef tokenEmbd, float[] posEmbd, int nCtx,
                          boolean causal, CpuTensorRef head, float[] headBias, CpuTensorRef projection,
                          CpuVitTower tower) {
        this.hp = hp;
        this.matVec = matVec;
        this.tokenEmbd = tokenEmbd;
        this.posEmbd = posEmbd;
        this.nCtx = nCtx;
        this.causal = causal;
        this.head = head;
        this.headBias = headBias;
        this.projection = projection;
        this.tower = tower;
    }

    /**
     * Resolves and shape-checks the text tensors of {@code tc} for {@code hp} over {@code nCtx} positions, then
     * binds them through {@code floats} and {@code refs}; the blocks attend causally when {@code causal}, else
     * bidirectionally.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalArgumentException when a shape differs from the tower's
     */
    static ClipTextTower bind(ClipHparams hp, TensorCatalog tc, int nCtx, boolean causal, CpuMatVec matVec,
                              Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(hp, "hparams");
        int nEmbd = hp.nEmbd();
        TensorRef tok = tc.require(TOKEN_EMBD);
        if (tok.rank() < 2 || tok.ne(0) != nEmbd) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + TOKEN_EMBD
                    + " has shape " + tok + ", the loader expects {" + nEmbd + ", n_vocab}");
        }
        TensorRef pos = tc.require(POSITION_EMBD);
        shape(tc, pos, nEmbd, nCtx);
        TensorRef postW = tc.get(POST_LN + ".weight");
        TensorRef postB = tc.get(POST_LN + ".bias");
        checkNorm(tc, POST_LN, postW, postB, nEmbd);
        TensorRef headW = head(tc, nEmbd);
        int headOut = headW == null ? nEmbd : (int) headW.ne(1);
        TensorRef proj = projection(tc, hp, PROJECTION, headOut);
        List<ClipLayer> layers = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            layers.add(ClipLayer.resolve(tc, hp, il, false));
        }
        CpuVitTower tower = new CpuVitTower(hp, matVec, layers, null, null, postW, postB, floats, refs, null);
        return new ClipTextTower(hp, matVec, refs.apply(TOKEN_EMBD), floats.apply(POSITION_EMBD), nCtx, causal,
                headW == null ? null : refs.apply(headW.name()), headW == null ? null : floats.apply(HEAD + ".bias"),
                proj == null ? null : refs.apply(proj.name()), tower);
    }

    /**
     * The text head {@value #HEAD} ({@code [n_embd, n_out]} with its {@code n_out} bias), or null when the file
     * has none.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when the weight comes without its bias
     */
    private static TensorRef head(TensorCatalog tc, int nEmbd) {
        TensorRef w = tc.get(HEAD + ".weight");
        if (w == null) {
            if (tc.get(HEAD + ".bias") != null) {
                throw new MissingTensorException(tc.archName(), -1, HEAD + ".weight");
            }
            return null;
        }
        if (w.rank() < 2 || w.ne(0) != nEmbd) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + w.name()
                    + " has shape " + w + ", the loader expects {" + nEmbd + ", n_out}");
        }
        shape(tc, w, nEmbd, w.ne(1));
        shape(tc, tc.require(HEAD + ".bias"), w.ne(1));
        return w;
    }

    /**
     * The tower's projection {@code name} over {@code width} pooled values: {@code [width, projection_dim]} when
     * the file declares a projection width, absent (the pooled row is the embedding) when it declares 0.
     */
    static TensorRef projection(TensorCatalog tc, ClipHparams hp, String name, int width) {
        if (hp.projectionDim() == 0) {
            TensorRef stray = tc.get(name);
            if (stray != null) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': the file declares "
                        + "projection_dim 0 but carries " + name);
            }
            return null;
        }
        TensorRef proj = tc.require(name);
        shape(tc, proj, width, hp.projectionDim());
        return proj;
    }

    /** Rows of the token embedding: every input id is below it. */
    int vocabRows() {
        return (int) tokenEmbd.rows();
    }

    /** The positions the tower holds ({@code clip.text.context_length}). */
    int contextLength() {
        return nCtx;
    }

    /** The width of the embedding: the projection's, else the head's, else the tower's. */
    int outputDim() {
        if (projection != null) {
            return (int) projection.rows();
        }
        return head == null ? hp.nEmbd() : (int) head.rows();
    }

    /** The tower's blocks on the host: causal or bidirectional as the tower was bound. */
    TowerBlocks cpuBlocks() {
        CpuVitTower.Options opts = causal ? CpuVitTower.Options.NONE.withLayerWindows(il -> VisionRows.CAUSAL)
                : CpuVitTower.Options.NONE;
        return (rows, n) -> tower.forward(rows, n, NormKind.LAYER, opts);
    }

    /**
     * The pooled, projected embedding of {@code tokens}, the tower's blocks run by {@code blocks}.
     *
     * @throws IllegalArgumentException when the input is empty, longer than the context, or an id is outside the
     *                                  vocabulary, or the pooling kind gives no pooled vector
     */
    float[] embed(int[] tokens, PoolingKind pooling, TowerBlocks blocks) {
        int n = tokens.length;
        if (n == 0) {
            throw new IllegalArgumentException("the CLIP text tower needs at least one token");
        }
        if (n > nCtx) {
            throw new IllegalArgumentException("the input holds " + n + " tokens, the CLIP text tower holds "
                    + nCtx + " positions (clip.text.context_length)");
        }
        int nEmbd = hp.nEmbd();
        float[] rows = new float[n * nEmbd];
        for (int t = 0; t < n; t++) {
            int id = tokens[t];
            if (id < 0 || id >= tokenEmbd.rows()) {
                throw new IllegalArgumentException("token id " + id + " at position " + t + " is outside the"
                        + " vocabulary of " + tokenEmbd.rows());
            }
            tokenEmbd.decodeRow(id, rows, t * nEmbd);
            for (int i = 0; i < nEmbd; i++) {
                rows[t * nEmbd + i] += posEmbd[t * nEmbd + i];
            }
        }
        float[] out = blocks.forward(rows, n);
        return project(head(pooled(out, n, nEmbd, pooling)));
    }

    /** The row of the pooling kind over the final-normed rows. */
    private static float[] pooled(float[] rows, int n, int width, PoolingKind pooling) {
        switch (pooling) {
            case LAST: {
                float[] last = new float[width];
                System.arraycopy(rows, (n - 1) * width, last, 0, width);
                return last;
            }
            case CLS:
                return ImageFeatureSelect.CLASS.select(rows, n, width);
            case MEAN:
                return ImageFeatureSelect.FULL.select(rows, n, width);
            default:
                throw new IllegalArgumentException("the CLIP text tower pools by cls, last or mean, not "
                        + pooling);
        }
    }

    private float[] head(float[] pooled) {
        if (head == null) {
            return pooled;
        }
        float[] out = new float[(int) head.rows()];
        matVec.matmul(head, pooled, 0, out, 0, 1);
        VisionRows.addBias(out, headBias);
        return out;
    }

    private float[] project(float[] pooled) {
        if (projection == null) {
            return pooled;
        }
        float[] out = new float[(int) projection.rows()];
        matVec.matmul(projection, pooled, 0, out, 0, 1);
        return out;
    }
}
