package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * The GPU a dflash drafter drafts on: the target's forward context, whose queue the draft's work is enqueued on, and
 * the draft's matrices uploaded to that device (llama.cpp runs the draft context on the target's backend,
 * {@code --spec-draft-backend-sampling}). The drafter's stack, cache and working buffers then live on the device
 * ({@link GpuDflashBackbone}, {@link GpuDflashKvCache}). Immutable; owns neither the context nor the weights.
 */
public final class DflashDevice {

    /** Per-layer matrices the device stack reads from the draft file. */
    private static final Set<String> LAYER_MATRICES = new HashSet<>(Arrays.asList(
            TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
            TensorNames.weight(TensorNames.ATTN_V), TensorNames.weight(TensorNames.ATTN_OUT),
            TensorNames.weight(TensorNames.FFN_GATE), TensorNames.weight(TensorNames.FFN_UP),
            TensorNames.weight(TensorNames.FFN_DOWN), TensorNames.weight(TensorNames.DFLASH_ATTN_CONV_PROJ),
            TensorNames.weight(TensorNames.DFLASH_FFN_CONV_PROJ)));

    /** The DFlash2 selector tables, which the selector uploads beside the weight pool as F32 rows. */
    private static final Set<String> SELECTOR_TABLES = new HashSet<>(Arrays.asList(
            TensorNames.weight(TensorNames.DFLASH_SELECTOR_PREV), TensorNames.weight(TensorNames.DFLASH_SELECTOR_NEXT),
            TensorNames.weight(TensorNames.DFLASH_SELECTOR_HIDDEN)));

    private final GpuForwardContext ctx;
    private final GpuWeights draft;

    /**
     * @param ctx   the target's forward context
     * @param draft the draft's matrices on the context's device ({@link #reads})
     */
    public DflashDevice(GpuForwardContext ctx, GpuWeights draft) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.draft = Objects.requireNonNull(draft, "draft");
    }

    GpuForwardContext context() {
        return ctx;
    }

    GpuWeights draft() {
        return draft;
    }

    /**
     * Why a draft of {@code spec} with a {@code kvType} cache drafts on the host instead, or null when the device
     * stack runs it: the DeepSeek-V4 DSpark backbone stays on the host, and a quantized cache whose head size is not
     * a multiple of 32 (its rows must fit the device kernels' 32-value blocks).
     */
    public static String refusal(DflashSpec spec, KvDType kvType) {
        return GpuDflashBackbone.refusal(Objects.requireNonNull(spec, "spec"), Objects.requireNonNull(kvType,
                "kvType"));
    }

    /**
     * Whether the device stack takes the draft tensor {@code name} onto the device, so the uploader loads everything
     * else only: the per-layer attention and FFN matrices, the encoder {@code fc}, the draft's own {@code output} and
     * the DFlash2 selector tables, which the selector uploads beside the weight pool as the F32 rows its kernels
     * read. Norm vectors, sinks and RoPE factors are uploaded from the host weights; the token embeddings and the
     * other readout heads ({@code d2t}, DSpark, LiLiCorr) stay on the host.
     */
    public static boolean reads(String name) {
        if (name.equals(TensorNames.weight(TensorNames.FC)) || name.equals(TensorNames.weight(TensorNames.OUTPUT))
                || SELECTOR_TABLES.contains(name)) {
            return true;
        }
        String prefix = TensorNames.STACK_BLK + ".";
        if (!name.startsWith(prefix)) {
            return false;
        }
        int dot = name.indexOf('.', prefix.length());
        return dot > 0 && LAYER_MATRICES.contains(name.substring(dot + 1));
    }

    /**
     * Device bytes a drafter of {@code spec} allocates beside its weights when it binds to a target with a context
     * of {@code nCtx} tokens: its KV cache in {@code kvType} ({@link GpuDflashKvCache}), its working buffers, its
     * DSpark Markov tables, its DFlash2 convolutions' scratch and selector readout, and its top-k readout candidates
     * (bound for a LiLiCorr draft and for a plain DFlash draft with a {@code pMin} above 0).
     *
     * @param catalog the draft's tensors, for the FFN and encoder widths
     * @throws IllegalArgumentException when the context is not positive or a width tensor is missing
     */
    public static long deviceBytes(DflashSpec spec, TensorCatalog catalog, int nCtx, KvDType kvType, float pMin) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(catalog, "catalog");
        if (nCtx <= 0) {
            throw new IllegalArgumentException("dflash device cache of a context of " + nCtx + " tokens");
        }
        int maxRows = Math.max(DflashDrafter.INJECT_CHUNK, spec.blockSize());
        boolean[] swa = new boolean[spec.nLayer()];
        int[] heads = new int[spec.nLayer()];
        int qWidth = 0;
        int kvWidth = 0;
        long nFf = 0;
        for (int il = 0; il < swa.length; il++) {
            swa[il] = spec.isSwa(il);
            heads[il] = spec.nHeadKv(il);
            qWidth = Math.max(qWidth, spec.nHead(il) * spec.headDim());
            kvWidth = Math.max(kvWidth, heads[il] * spec.headDim());
            nFf = Math.max(nFf, catalog.require(il, TensorNames.weight(TensorNames.FFN_GATE)).rows());
        }
        TensorRef fc = catalog.require(TensorNames.weight(TensorNames.FC));
        long cache = GpuDflashKvCache.bytes(Math.addExact(nCtx, spec.blockSize()), spec.blockSize(), spec.nSwa(),
                swa, heads, spec.headDim(), kvType, maxRows);
        int topKK = spec.lilicorr() || spec.dflash2() || pMin > 0.0f ? GpuDflashBackbone.topKWidth(spec) : 0;
        long work = GpuDflashBackbone.workBytes(spec, catalog, Math.toIntExact(fc.cols()), qWidth, kvWidth,
                Math.toIntExact(nFf), spec.nVocabDraft(), topKK);
        return Math.addExact(cache, work);
    }

    /** The draft's device {@code output}, else the target's ({@code output}, then the tied {@code token_embd}). */
    GpuWeight output() {
        GpuWeight w = draft.getOrNull(TensorNames.weight(TensorNames.OUTPUT));
        if (w == null) {
            w = ctx.weights().getOrNull(TensorNames.weight(TensorNames.OUTPUT));
        }
        if (w == null) {
            w = ctx.weights().getOrNull(TensorNames.weight(TensorNames.TOKEN_EMBD));
        }
        if (w == null) {
            throw new UnsupportedOperationException("dflash device drafter: neither the draft nor the target holds"
                    + " an output head on the device");
        }
        return w;
    }

    @Override
    public String toString() {
        return "DflashDevice{" + ctx.backend().backendType() + '}';
    }
}
