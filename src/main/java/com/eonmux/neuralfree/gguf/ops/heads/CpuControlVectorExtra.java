package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * The control-vector extra of layer {@code il} on the host forward pass ({@code llm_graph_context::build_cvec}
 * applied to the layer output after its residual add, llama.cpp models/llama.cpp:223-224): {@code x[t] += v} per token
 * {@code t} of the layer output, for {@code v = scale * direction.il} of the {@linkplain ControlVector#add active
 * control vectors} — the {@link CpuLayerExtra afterBlock} hook, a token-wise add with {@link #batchRefusal()} null
 * because the add does not depend on the step's sequence layout.
 *
 * <p>The extras are the residual/hidden-state application path the endpoint uses ({@link ControlVector}): llama.cpp
 * builds one per decoder layer alike, and the layer executor calls the {@code afterBlock} of every installed extra of
 * a layer at the end of the block, batched and contiguous alike (the {@code afterBlock(StepLayout, float[])} override).
 * The layer's sum comes from the set's current {@link ControlVector.Snapshot}: the extra refreshes against the set's
 * revision per step, so a load, an unload or a scale change takes effect between two steps, on the engine thread (the
 * {@link ControlVector} listener delivery is synchronous on the same thread), and a step is never computed half
 * against one revision. The sum is computed once per change, per layer, in F32 — the bounded host shadow of the GPU
 * extra's buffer rewrite.</p>
 *
 * <p>Mutable, owned by the engine thread like every extra. Must be used only from the forward pass that
 * bound it; {@link #bind} refuses twice.</p>
 */
public final class CpuControlVectorExtra implements CpuLayerExtra {

    private static final int ELEMENT_GRAIN = 1 << 15;

    private final ControlVector set;
    private final int il;
    private ControlVector.Snapshot held;
    private int width;

    public CpuControlVectorExtra(ControlVector set, LayerSpec layer) {
        this.set = Objects.requireNonNull(set, "set");
        this.il = Objects.requireNonNull(layer, "layer").il();
    }

    /** The graph layer whose output this extra adds to. */
    public int layer() {
        return il;
    }

    /** The set this extra sums over. */
    public ControlVector set() {
        return set;
    }

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (layer.il() != il) {
            throw new IllegalArgumentException("the control-vector extra is for layer " + il + " and cannot bind at "
                    + "layer " + layer.il() + " of " + describe(ctx, layer));
        }
        if (held != null) {
            throw new IllegalStateException("the control-vector extra of layer " + il + " is already bound");
        }
        int streamWidth = layer.residual().streamWidth();
        if (streamWidth != set.nEmbd()) {
            throw new IllegalStateException("control vector n_embd does not match model: the residual stream of "
                    + describe(ctx, layer) + " is " + streamWidth + " wide, the set is over " + set.nEmbd());
        }
        this.width = streamWidth;
        this.held = set.snapshot();
    }

    /** {@code x[t] += v} of the layer's sum for every row {@code t} of the layer output. */
    @Override
    public void afterBlock(int pos0, int n, float[] x) {
        float[] v = refresh();
        if (v.length == 0) {
            return;
        }
        addRows(n, x, 0, v);
    }

    /**
     * Batched form: the add is a token-wise add, the same {@code v} per row, independent of the segments' sequences and
     * positions, so the batched step runs one loop over the step's rows.
     */
    @Override
    public void afterBlock(StepLayout s, float[] x) {
        float[] v = refresh();
        if (v.length == 0) {
            return;
        }
        addRows(s.nTokens(), x, 0, v);
    }

    /**
     * The add touches only the end of the block (the {@code afterBlock} hooks), and both are layout independent, so
     * the batched forms of the earlier hooks are the no-ops the contiguous forms delegate to.
     */
    @Override
    public void beforeBlock(StepLayout s, float[] x) {
    }

    @Override
    public boolean afterMixer(StepLayout s, float[] x, float[] branch) {
        return false;
    }

    @Override
    public boolean afterFfn(StepLayout s, float[] x, float[] branch) {
        return false;
    }

    /** The add needs no per-sequence layout, so batching is supported (the batched hooks run). */
    @Override
    public String batchRefusal() {
        return null;
    }

    private float[] refresh() {
        if (held == null) {
            throw new IllegalStateException("control-vector extra of layer " + il + " used before bind()");
        }
        if (held.revision() != set.revision()) {
            held = set.snapshot();
        }
        return held.layer(il);
    }

    private void addRows(int n, float[] x, int xOff, float[] v) {
        if (n < 0 || n * (long) width > x.length - (long) xOff) {
            throw new IllegalStateException("control-vector extra of layer " + il + ": " + n + " rows of " + width
                    + " do not fit the residual stream array of " + x.length + " floats");
        }
        int w = width;
        CpuParallel.parallelFor(n, Math.max(1, ELEMENT_GRAIN / Math.max(1, w)), (from, to) -> {
            for (int t = from; t < to; t++) {
                int xb = xOff + t * w;
                for (int i = 0; i < w; i++) {
                    x[xb + i] += v[i];
                }
            }
        });
    }

    @Override
    public void close() {
        held = null;
        width = 0;
    }

    private static String describe(CpuForwardContext ctx, LayerSpec layer) {
        return ctx.spec().arch().ggufName() + " " + layer.stack() + "." + layer.il();
    }

    @Override
    public String toString() {
        return "CpuControlVectorExtra{layer " + il + " of " + set + "}";
    }
}
