package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Objects;

/**
 * RoPE half of the context shift for one attention layer on the GPU (llama.cpp {@code build_graph_shift} /
 * {@code build_rope_shift}, src/llama-kv-cache.cpp:1945-2070). The memory has already dropped positions
 * {@code [nKeep, nKeep + nDiscard)} and moved the later rows ({@code DeviceModelMemory.shiftPositions}); this rotates
 * the K rows of the layer's own cache by the per-row deltas of its KV group ({@code DeviceModelMemory.shiftDelta},
 * indexed by the row after the shift) with {@code LLM_KV_RopeShift_<type>}:
 * <ul>
 *   <li>rotated dims {@code [nOffs, nOffs + n_rot)} of every KV head with the layer's RoPE (base, scale, YaRN and
 *       frequency factors); multi-section types shift as NEOX ({@link com.eonmux.neuralfree.gguf.ops.rope.RopeMath#shiftType});</li>
 *   <li>a block-type cache is dequantized, rotated back (when Hadamard-rotated), shifted, rotated and requantized in
 *       the kernel ({@code rot_k = nRotK}, llama-kv-cache.cpp:1971-1985);</li>
 *   <li>layers without RoPE and layers that read another layer's cache shift nothing (the owner shifts,
 *       llama-kv-cache.cpp:2029-2033).</li>
 * </ul>
 * The kernel is prepared for its work-group size while the layer binds, so a backend that builds one pipeline per
 * work-group size (Vulkan) never builds it during generation, and a kernel the device cannot run is reported before
 * any cache row moves. The dispatch itself is prepared on the first shift, when the memory has created the delta
 * buffers, and re-dispatched for later shifts. Not thread-safe.
 */
public final class GpuContextShift implements AutoCloseable {

    private final GpuOps ops;
    private final GpuMemory memory;
    private final GpuKvLayer kv;
    private final RopeSpec rope;
    private final GpuBuffer freqFactors;
    private final int nHeadKv;
    private final int headDimK;
    private final int rotK;
    private PreparedDispatch dispatch;
    private GpuBuffer boundDelta;

    private GpuContextShift(GpuOps ops, GpuMemory memory, GpuKvLayer kv, RopeSpec rope, GpuBuffer freqFactors,
                            int nHeadKv, int headDimK, int rotK) {
        this.ops = ops;
        this.memory = memory;
        this.kv = kv;
        this.rope = rope;
        this.freqFactors = freqFactors;
        this.nHeadKv = nHeadKv;
        this.headDimK = headDimK;
        this.rotK = rotK;
    }

    /**
     * Why the context of {@code layer} cannot be shifted, or null when it can
     * ({@link ModelSpec#contextShiftRefusal(LayerSpec)}, llama.cpp {@code get_can_shift}).
     */
    public static String refusal(ModelSpec model, LayerSpec layer) {
        Objects.requireNonNull(model, "model");
        return model.contextShiftRefusal(layer);
    }

    /**
     * True when the layer's own cache needs a RoPE shift: it uses memory, owns its cache and applies RoPE.
     */
    public static boolean shifts(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        return a != null && a.usesMemory() && !a.sharesKv(layer.il()) && !a.linearReplace()
                && layer.rope() != null && !a.noRope();
    }

    /**
     * Shift helper of an owning RoPE layer's cache.
     *
     * @param freqFactors the layer's F32 frequency factors ({@code n_dims / 2} values), or null
     * @throws IllegalArgumentException when the cache geometry does not fit the shift kernel, or the backend cannot
     *                                  provide or prepare the shift kernel for its work-group size
     */
    public static GpuContextShift of(GpuOps ops, GpuMemory memory, LayerSpec layer, GpuKvLayer kv,
                                     GpuBuffer freqFactors) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(kv, "kv");
        if (!shifts(layer)) {
            throw new IllegalArgumentException("layer " + layer.il() + " has no RoPE cache of its own to shift");
        }
        AttentionSpec a = layer.attention();
        KvLayout layout = kv.layout();
        KvDType type = layout.dtypeK();
        RotationPlan rot = kv.rotation();
        int rotK = type.quantized() && rot.rotateK() ? rot.nRotK() : 0;
        RopeSpec rope = layer.rope();
        RopeKernels.validateKvShift(type, rope.nDims(), rope.nOffs(), layout.rows(), a.nHeadKv(), a.headDimK(),
                layout.widthK(), rotK);
        String kernel = RopeKernels.kvRopeShift(type);
        int local = RopeKernels.kvRopeShiftLocal(type, ops.local());
        if (!ops.prewarm(kernel, new long[]{local})) {
            throw new IllegalArgumentException("context shift kernel " + kernel + " is unavailable on the "
                    + ops.kernels().backend().backendType() + " backend for work-groups of " + local);
        }
        return new GpuContextShift(ops, memory, kv, rope, freqFactors, a.nHeadKv(), a.headDimK(), rotK);
    }

    /**
     * Rotates the cached K rows by the deltas of the shift the memory just applied.
     *
     * @throws UnsupportedOperationException when the memory does not record shift deltas
     *                                       ({@link DeviceModelMemory} does)
     */
    public void apply() {
        if (!(memory instanceof DeviceModelMemory)) {
            throw new UnsupportedOperationException("context shift needs DeviceModelMemory deltas, the memory is "
                    + memory.getClass().getSimpleName());
        }
        GpuBuffer delta = ((DeviceModelMemory) memory).shiftDelta(kv.group());
        if (dispatch == null || delta != boundDelta) {
            if (dispatch != null) {
                dispatch.close();
                dispatch = null;
            }
            dispatch = prepare(delta);
            boundDelta = delta;
        }
        dispatch.dispatch();
    }

    private PreparedDispatch prepare(GpuBuffer delta) {
        KvLayout layout = kv.layout();
        KvDType type = layout.dtypeK();
        boolean hasFf = freqFactors != null;
        GpuBuffer ff = hasFf ? freqFactors : ops.placeholders().readOnlyFloats();
        Object[] args = RopeKernels.kvRopeShiftArgs(type, rope, layout.rows(), nHeadKv, headDimK, layout.widthK(),
                rope.nOffs(), rotK, hasFf, delta, ff, kv.k());
        int local = RopeKernels.kvRopeShiftLocal(type, ops.local());
        long items = RopeKernels.kvRopeShiftWorkItems(type, layout.rows(), nHeadKv, rope.nDims(), local);
        return ops.prepare(RopeKernels.kvRopeShift(type), new long[]{RopeKernels.roundUp(items, local)},
                new long[]{local}, args);
    }

    @Override
    public void close() {
        if (dispatch != null) {
            dispatch.close();
            dispatch = null;
        }
    }
}
