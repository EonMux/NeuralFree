package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvFormats;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.List;

/**
 * The cache writes of {@link GpuAttentionMixer}: the {@code LLM_KV_Store_<type>} steps of K and V at the step's
 * {@code cell_idx}, and their fused forms — {@link FusedKernels#NORM_ROPE_STORE} when the K rows are per-head RMS
 * normalized before RoPE, {@link FusedKernels#ROPE_STORE} without a QK norm — which write the cache rows directly and
 * let the layer skip its separate K norm/rotation and K store. Not thread-safe; driven by the owning mixer's bind.
 */
final class GpuKvStoreStages {

    private final GpuAttentionMixer mixer;
    private final List<TokenStep> storeSteps = new ArrayList<>();
    /** The cache stores' rebind points a prefill context-parallel shadow store overrides. */
    private final List<StoreRebind> storeRebinds = new ArrayList<>();

    GpuKvStoreStages(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /** The store steps of the bound layer, in bind order (K first, then V). */
    List<TokenStep> steps() {
        return storeSteps;
    }

    /** The stores' rebind points, the buffers {@code shadowStore} overrides. */
    List<StoreRebind> rebinds() {
        return storeRebinds;
    }

    /**
     * The fused K-row kernel this layer can use, or null: {@link FusedKernels#NORM_ROPE_STORE} when the K rows are
     * per-head RMS normalized before RoPE, {@link FusedKernels#ROPE_STORE} without a QK norm. Both write the cache
     * rows directly, so the layer skips its separate K norm/rotation and K store. Only plain F32/F16 caches without
     * a K rotation or per-token-head scale qualify, as llama.cpp's rope + set_rows fusion does; a V-only rotation is
     * still applied before the store.
     */
    String fusedKernel(GpuKvLayer kv) {
        RotationPlan rotation = kv != null ? kv.rotation() : null;
        if (kv == null || mixer.hostKv() || !mixer.proj().projectsKv() || mixer.proj().vFromK()
                || mixer.spec().qkNormAfterRope() || (rotation != null && rotation.rotateK())) {
            return null;
        }
        KvLayout layout = kv.layout();
        if (layout.vAliasesK() || (layout.dtypeK() != KvDType.F32 && layout.dtypeK() != KvDType.F16)
                || layout.widthK() != mixer.spec().kDim() || mixer.kvScales().kDevice() != null
                || kv.rowScales() != null) {
            return null;
        }
        RopeSpec rope = mixer.layer().rope();
        if (rope == null || mixer.spec().noRope() || rope.type() == RopeType.NONE || rope.fope() != null) {
            /* The fused store kernels evaluate the analytic rotation; FoPE pairs come from coefficients. */
            return null;
        }
        QkNormKind kind = mixer.spec().qkNorm();
        if (kind == QkNormKind.NONE) {
            return mixer.ops().fused(FusedKernels.ROPE_STORE) ? FusedKernels.ROPE_STORE : null;
        }
        boolean hasQ = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        GpuNormRopeStages norms = mixer.normRope();
        if (!QkNormGeometry.applies(kind, mixer.ctx().spec().arch(), true, hasQ, hasK)
                || !GpuNormRopeStages.perHeadRms(norms.geometry(kind, mixer.spec().nHeadKv()), mixer.spec().nHeadKv(),
                        mixer.spec().headDimK())) {
            return null;
        }
        return mixer.ops().fused(FusedKernels.NORM_ROPE_STORE) ? FusedKernels.NORM_ROPE_STORE : null;
    }

    /** Binds the K and V store steps of the layer's cache, fused or plain. */
    void bind(GpuKvLayer kv, String fusedKernel, GpuBuffer ropePos) {
        KvLayout layout = kv.layout();
        GpuBuffer cellIdx = mixer.stepInputs().cellIdx(kv.group());
        GpuBuffer side = kv.vendorSide();
        int sideStride = side != null ? DeviceKvFormats.sideRowBytes(layout) : 0;
        if (fusedKernel == null) {
            storeSteps.add(storeStep(layout.dtypeK(), mixer.spec().kDim(), layout.widthK(), mixer.kBuf(), cellIdx,
                    kv.k(), mixer.kvScales().kDevice(), kv.rowScales(), side, sideStride,
                    DeviceKvFormats.sidePlaneOffset(layout, false), mixer.spec().headDimK(), 0));
        } else {
            storeSteps.add(fusedKStoreStep(kv, fusedKernel, ropePos));
        }
        storeSteps.add(storeStep(layout.dtypeV(), mixer.spec().vDim(), layout.widthV(), mixer.vBuf(), cellIdx, kv.v(),
                mixer.kvScales().vDevice(), kv.rowScales(), side, sideStride,
                DeviceKvFormats.sidePlaneOffset(layout, true), mixer.spec().headDimV(), 1));
    }

    /** The fused K-row step of {@link #fusedKernel}: normalize (first form only) and rotate into the cache. */
    private TokenStep fusedKStoreStep(GpuKvLayer kv, String kernel, GpuBuffer ropePos) {
        RopeSpec rope = mixer.layer().rope();
        KvLayout layout = kv.layout();
        int nHeads = mixer.spec().nHeadKv();
        int headDim = mixer.spec().headDimK();
        GpuBuffer ff = mixer.normRope().factors();
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : mixer.ops().placeholders().readOnlyFloats();
        GpuBuffer none = mixer.ops().placeholders().readOnlyFloats();
        GpuBuffer cellIdx = mixer.stepInputs().cellIdx(kv.group());
        int dstType = layout.dtypeK() == KvDType.F16 ? FusedKernels.KV_STORE_F16 : FusedKernels.KV_STORE_F32;
        Object[] args;
        int cellArg;
        int dstArg;
        int posStrideArg;
        if (FusedKernels.NORM_ROPE_STORE.equals(kernel)) {
            GpuNormRopeStages norms = mixer.normRope();
            QkNormGeometry kg = norms.geometry(mixer.spec().qkNorm(), nHeads);
            GpuNormRopeStages.QkNormOperand kop = norms.operand(kg, TensorNames.ATTN_K_NORM);
            args = FusedKernels.normRopeStoreArgs(rope, mixer.ubatch(), nHeads, headDim, mixer.spec().kDim(), headDim,
                    rope.nOffs(), mixer.ubatch(), 0, kop.flags(), kop.wRows, kg.eps(), hasFf, headDim,
                    layout.widthK(), dstType, ropePos, ffArg, kop.weight != null ? kop.weight : none,
                    kop.bias != null ? kop.bias : none, mixer.kBuf(), cellIdx, kv.k());
            cellArg = FusedKernels.STORE_ARG_CELL;
            dstArg = FusedKernels.STORE_ARG_DST;
            posStrideArg = FusedKernels.STORE_ARG_POS_STRIDE;
        } else {
            args = FusedKernels.ropeStoreArgs(rope, mixer.ubatch(), nHeads, headDim, mixer.spec().kDim(),
                    rope.nOffs(), mixer.ubatch(), 0, hasFf, headDim, layout.widthK(), dstType, ropePos, ffArg,
                    mixer.kBuf(), cellIdx, kv.k());
            cellArg = FusedKernels.ROPE_STORE_ARG_CELL;
            dstArg = FusedKernels.ROPE_STORE_ARG_DST;
            posStrideArg = FusedKernels.ROPE_STORE_ARG_POS_STRIDE;
        }
        TokenStep store = mixer.step(kernel, args, FusedKernels.STORE_ARG_N_TOKENS, 1, (long) nHeads * mixer.local(),
                0, posStrideArg);
        storeRebinds.add(new StoreRebind(store, cellArg, cellIdx, dstArg, kv.k(), false));
        return store;
    }

    private TokenStep storeStep(KvDType type, int rowElems, int dstRowStride, GpuBuffer src, GpuBuffer cellIdx,
                                GpuBuffer dst, GpuBuffer kvScale, GpuBuffer rowScales, GpuBuffer side,
                                int sideStride, int sideOff, int headElems, int vSide) {
        GpuBuffer scales = kvScale != null ? kvScale : mixer.ops().placeholders().kvScalesOnes();
        GpuBuffer rowScaleArg = rowScales != null ? rowScales : mixer.ops().placeholders().kvScalesOnes();
        GpuBuffer sideArg = side != null ? side : mixer.ops().placeholders().scratchInts();
        Object[] args = RopeKernels.kvStoreArgs(type, mixer.ubatch(), rowElems, rowElems, 0, dstRowStride, headElems,
                src, cellIdx, scales, dst, rowScaleArg, sideArg, sideStride, sideOff, vSide);
        long perToken = type.perTokenHead() || type.vendorFormat()
                ? RopeKernels.kvStoreWorkItemsPerTokenHead(type, 1, rowElems, headElems)
                : RopeKernels.kvStoreWorkItems(type, 1, rowElems);
        TokenStep store = mixer.step(RopeKernels.kvStore(type), args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0, -1);
        storeRebinds.add(new StoreRebind(store, RopeKernels.kvStoreCellArg(type), cellIdx,
                RopeKernels.kvStoreDstArg(type), dst, vSide != 0));
        return store;
    }
}
