package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * The control-vector extra of layer {@code il} on the device forward pass ({@code llm_graph_context::build_cvec}
 * applied to the layer output after its residual add, llama.cpp models/llama.cpp:223-224): {@code x[t] += v} per token
 * {@code t} of the layer output, for {@code v = scale * direction.il} of the {@linkplain ControlVector#add active
 * control vectors} — the {@link GpuLayerExtra afterBlock} hook, a token-wise add with {@link #batchRefusal()} null
 * because the add does not depend on the step's sequence layout.
 *
 * <p>The extras are the residual/hidden-state application path the endpoint uses ({@link ControlVector}): llama.cpp
 * builds one per decoder layer alike, and the device layer executor calls the {@code afterBlock} of every installed
 * extra of a layer at the end of the block, batched and contiguous alike (the {@code afterBlock(StepLayout,
 * ScratchSlot)} override). Each row add is one {@code GpuOps#scaleAdd} of {@code width} elements with the layer's
 * {@code v} buffer; no new kernel and no kernel-ABI change, the add rides {@code LLM_Axpby_F32} as every element-wise
 * device add of the engine does.
 *
 * <p>The layer's sum lives in one writable F32 buffer the extra owns and rewrites in place between steps: the extra
 * follows its set as a {@link ControlVector.Listener} — the set delivers the events synchronously on the engine thread
 * the forward pass runs on, so a load, an unload or a scale change rewrites {@code v} there, in the
 * {@code LoraDeltas.rescale} form, and a later step runs entirely on the new epoch. {@link #afterBlock} also
 * rewrites when it first sees the set's revision move past the one the buffer was written for, which keeps a buffer
 * bound before a hot change correct without relying on event order. Mutable, owned by the engine thread like every
 * extra; {@link #close} must not run while a step's queues still hold its dispatches.
 */
public final class ControlVectorExtra implements GpuLayerExtra, ControlVector.Listener {

    private final ControlVector set;
    private final int il;
    private GpuOps ops;
    /** The layer's sum, written per revision; owned and released by {@link #close}. */
    private GpuBuffer vec;
    private ControlVector.Snapshot held;
    private int width;
    private int ubatch;
    private boolean following;

    public ControlVectorExtra(ControlVector set, LayerSpec layer) {
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

    /** The layer's sum buffer if the extra has bound ({@code null} otherwise). */
    public GpuBuffer vector() {
        return vec;
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (ops != null) {
            throw new IllegalStateException("the control-vector extra of layer " + il + " is already bound");
        }
        if (layer.il() != il) {
            throw new IllegalArgumentException("the control-vector extra is for layer " + il + " and cannot bind at "
                    + "layer " + layer.il());
        }
        int streamWidth = layer.residual().streamWidth();
        if (streamWidth != set.nEmbd()) {
            throw new IllegalStateException("control vector n_embd does not match model: the residual stream of "
                    + layer.stack() + "." + layer.il() + " of " + ctx.spec().arch().ggufName() + " is " + streamWidth
                    + " wide, the set is over " + set.nEmbd());
        }
        this.width = streamWidth;
        this.ubatch = ctx.scratch().ubatch();
        this.held = set.snapshot();
        try {
            this.ops = ctx.ops();
            this.vec = GpuOpsFloatBuffer.allocate(ops, width);
            writeVector();
        } catch (RuntimeException e) {
            release();
            ops = null;
            throw e;
        }
        set.addListener(this);
        following = true;
    }

    /** {@code x[t] += v} of the layer's sum for every row {@code t} of the layer output. */
    @Override
    public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
        GpuOps o = ensureOperational();
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("control-vector afterBlock of layer " + il + ": token count "
                    + nTokens + " is outside [1, " + ubatch + "]");
        }
        checkRevision();
        GpuBuffer v = vec;
        if (held.layer(il).length == 0) {
            return;
        }
        ScratchSlot rows = x.require((long) nTokens * width);
        for (int t = 0; t < nTokens; t++) {
            o.scaleAdd(v, 0, 1.0f, rows.buffer(), rows.offset() + t * width, width);
        }
    }

    /**
     * Batched form: the add is a token-wise add, the same {@code v} per row, independent of the segments' sequences and
     * positions, so the batched step runs one loop over the step's rows.
     */
    @Override
    public void afterBlock(StepLayout s, ScratchSlot x) {
        afterBlock(s.pos0(0), s.nTokens(), x);
    }

    /**
     * The add touches only the end of the block (the {@code afterBlock} hooks), and both are layout independent, so
     * the batched forms of the earlier hooks are the no-ops the contiguous forms delegate to.
     */
    @Override
    public void beforeBlock(StepLayout s, ScratchSlot x) {
    }

    @Override
    public boolean afterMixer(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        return false;
    }

    @Override
    public boolean afterFfn(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        return false;
    }

    /** The add needs no per-sequence layout, so batching is supported (the batched hooks run). */
    @Override
    public String batchRefusal() {
        return null;
    }

    private GpuOps ensureOperational() {
        if (ops == null) {
            throw new IllegalStateException("control-vector extra of layer " + il + " used before bind()");
        }
        return ops;
    }

    /** Rewrites {@code v} when the set moved past the revision the buffer holds. Idempotent between steps. */
    private void checkRevision() {
        if (held.revision() != set.revision()) {
            held = set.snapshot();
            writeVector();
        }
    }

    /**
     * The {@link ControlVector} events, delivered synchronously on the engine thread: rewrite {@code v} in
     * place for the new identity of the set. A loaded or unloaded vector outside this layer's range changes nothing
     * here, and the rewrite still refreshes the revision so no step runs on the old bytes by accident.
     */
    @Override
    public void loaded(ControlVector.Entry entry) {
        ensureOperational();
        refreshFromListener();
    }

    @Override
    public void unloaded(ControlVector.Entry entry) {
        ensureOperational();
        refreshFromListener();
    }

    @Override
    public void scaleChanged(ControlVector.Entry entry, float previous) {
        ensureOperational();
        refreshFromListener();
    }

    private void refreshFromListener() {
        ControlVector.Snapshot now = set.snapshot();
        if (now.revision() == held.revision()) {
            return;
        }
        held = now;
        writeVector();
    }

    /** Uploads the layer's {@code held} sum into {@code vec} in place (the rescale form). */
    private void writeVector() {
        float[] sum = held.layer(il);
        if (sum.length == 0) {
            ops.fill(vec, 0L, (long) width * Float.BYTES, 0);
            return;
        }
        if (sum.length != width) {
            throw new IllegalStateException("the control-vector sum of layer " + il + " is " + sum.length
                    + " wide, the buffer is " + width);
        }
        GpuEvent e = ops.queue().enqueueWriteBuffer(vec, 0L, (long) width * Float.BYTES, sum);
        if (e != null) {
            e.close();
        }
    }

    /** Releases the buffer and the listener registration; the shared context resources stay. Idempotent. */
    @Override
    public void close() {
        if (following) {
            set.removeListener(this);
            following = false;
        }
        release();
    }

    private void release() {
        if (vec != null) {
            vec.close();
            vec = null;
        }
        held = null;
    }

    /** The host-side buffer allocation and upload the extra keeps in one place. */
    private static final class GpuOpsFloatBuffer {
        /** A writable F32 buffer of {@code floats} owned by its caller. */
        static GpuBuffer allocate(GpuOps ops, int floats) {
            return SwitchedLoraExtra.allocateFloats(ops, "control vector " + floats + " floats", floats, false);
        }
    }
}
