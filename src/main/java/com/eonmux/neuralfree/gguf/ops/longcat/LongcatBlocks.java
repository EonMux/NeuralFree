package com.eonmux.neuralfree.gguf.ops.longcat;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.ops.act.CpuDenseFfn;
import com.eonmux.neuralfree.gguf.ops.act.GpuDenseFfn;
import com.eonmux.neuralfree.gguf.ops.moe.CpuMoeFfn;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The LongCat-Flash family: the two composed FFN kinds of its dual-attention blocks, and the n-gram input
 * embedding of the Lite variant.
 *
 * <p>A logical LongCat layer is stored as two physical GGUF blocks (vLLM longcat_flash.py
 * {@code FlashDecoderLayer}). The even block runs the first MLA attention and then, on the same normalized input,
 * the first dense SwiGLU MLP and the shortcut-connected MoE: {@link FfnKind#MOE_SHORTCUT} writes the MLP to the
 * block output and keeps the expert output in a model-wide scratch. The odd block runs the second MLA attention and
 * MLP: {@link FfnKind#SHORTCUT_ADD} adds the kept expert output to its MLP. The two kinds therefore compose the
 * dense FFN block with the routed-expert block; the shared scratch is bound by size only, as every
 * {@code CpuScratch}/ {@code GpuScratch} temporary of one name and size is one buffer.</p>
 *
 * <p>The n-gram embedding ({@link LayerSpec.Extras#NGRAM_EMBD}) rewrites the model input on the first block: the
 * token embedding is averaged with the projection of one hashed row per embedder, over the preceding
 * {@code emb_neighbor_num - 1} tokens of the request (vLLM longcat_flash_ngram.py {@code NgramEmbedding}).</p>
 */
public final class LongcatBlocks {

    /** Scratch temporary holding the shortcut expert output between the two blocks of a LongCat layer. */
    static final String SHORTCUT_TEMP = "longcat.shortcut";

    private static final FfnKind[] FFN_KINDS = {FfnKind.MOE_SHORTCUT, FfnKind.SHORTCUT_ADD};

    private LongcatBlocks() {
    }

    /** The wanted kinds are the two LongCat FFN kinds the LongCat architectures resolve. */
    public static boolean isLongcatKind(FfnKind kind) {
        return kind == FfnKind.MOE_SHORTCUT || kind == FfnKind.SHORTCUT_ADD;
    }

    public static List<GpuFfnFactory> gpuFfns() {
        List<GpuFfnFactory> out = new ArrayList<>(FFN_KINDS.length);
        for (FfnKind kind : FFN_KINDS) {
            out.add(new GpuFfnFactory() {
                @Override
                public FfnKind kind() {
                    return kind;
                }

                @Override
                public GpuFfnBlock create(LayerSpec layer) {
                    return new GpuLongcatFfn(kind, layer);
                }
            });
        }
        return Collections.unmodifiableList(out);
    }

    public static List<CpuFfnFactory> cpuFfns() {
        List<CpuFfnFactory> out = new ArrayList<>(FFN_KINDS.length);
        for (FfnKind kind : FFN_KINDS) {
            out.add(new CpuFfnFactory() {
                @Override
                public FfnKind kind() {
                    return kind;
                }

                @Override
                public CpuFfnBlock create(LayerSpec layer) {
                    return new CpuLongcatFfn(kind, layer);
                }
            });
        }
        return Collections.unmodifiableList(out);
    }

    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.singletonList(new GpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LayerSpec.Extras.NGRAM_EMBD;
            }

            @Override
            public GpuLayerExtra create(LayerSpec layer) {
                return new GpuNgramEmbeddingExtra();
            }
        });
    }

    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.singletonList(new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LayerSpec.Extras.NGRAM_EMBD;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                return new CpuNgramEmbeddingExtra();
            }
        });
    }

    /**
     * The layer as an FFN-only view over the same block: only the FFN sub-specs a composed block hands to its
     * dense and routed halves, with no mixer, attention or RoPE.
     */
    static LayerSpec ffnView(LayerSpec layer, FfnKind kind, FfnSpec ffn, MoeSpec moe) {
        return LayerSpec.builder(layer.il(), MixerKind.NONE, kind)
                .weightLayer(layer.weightLayer())
                .stack(layer.stack())
                .norm(layer.norm())
                .residual(layer.residual())
                .ffn(ffn)
                .moe(moe)
                .build();
    }

    /** The CPU blocks of one LongCat FFN kind. */
    static final class CpuLongcatFfn implements CpuFfnBlock {
        private final FfnKind kind;
        private final LayerSpec layer;
        private CpuDenseFfn dense;
        private CpuMoeFfn moe;
        private float[] carry;
        private int nEmbd;
        private boolean isBound;

        CpuLongcatFfn(FfnKind kind, LayerSpec layer) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.layer = Objects.requireNonNull(layer, "layer");
            if (!isLongcatKind(kind)) {
                throw new IllegalArgumentException("layer " + layer.il() + " of kind " + kind
                        + " is not a LongCat FFN kind");
            }
            if (layer.ffnKind() != kind || layer.ffn() == null) {
                throw new IllegalArgumentException("layer " + layer.il() + " has FFN kind " + layer.ffnKind()
                        + " and FFN " + layer.ffn() + ", not " + kind);
            }
            if ((kind == FfnKind.MOE_SHORTCUT) != (layer.moe() != null)) {
                throw new IllegalArgumentException("layer " + layer.il() + " of kind " + kind + " has MoE "
                        + layer.moe());
            }
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec target) {
            if (isBound) {
                throw new IllegalStateException("LongCat FFN block of layer " + layer.il() + " is already bound");
            }
            if (target != layer) {
                throw new IllegalArgumentException("block created for layer " + layer.il() + " bound to layer "
                        + target.il());
            }
            this.nEmbd = ctx.spec().hparams().nEmbd();
            int ubatch = ctx.scratch().ubatch();
            this.carry = ctx.scratch().temp(SHORTCUT_TEMP, Math.multiplyExact(ubatch, nEmbd));
            this.dense = new CpuDenseFfn();
            this.dense.bind(ctx, ffnView(layer, FfnKind.DENSE, layer.ffn(), null));
            if (kind == FfnKind.MOE_SHORTCUT) {
                LayerSpec moeView = ffnView(layer, FfnKind.MOE, null, layer.moe());
                this.moe = new CpuMoeFfn(FfnKind.MOE, moeView);
                this.moe.bind(ctx, moeView);
            }
            this.isBound = true;
        }

        @Override
        public void forward(int n, float[] in, float[] out) {
            if (!isBound) {
                throw new IllegalStateException("LongCat FFN block of layer " + layer.il() + " used before bind");
            }
            dense.forward(n, in, out);
            if (moe != null) {
                moe.forward(n, in, carry);
                return;
            }
            int count = Math.multiplyExact(n, nEmbd);
            if (out.length < count || carry.length < count) {
                throw new IllegalArgumentException("LongCat rows need " + count + " values, got out " + out.length
                        + ", carry " + carry.length);
            }
            for (int i = 0; i < count; i++) {
                out[i] += carry[i];
            }
        }

        @Override
        public void close() {
            if (moe != null) {
                moe.close();
            }
            if (dense != null) {
                dense.close();
            }
        }

        @Override
        public String toString() {
            return "CpuLongcatFfn{layer=" + layer.il() + ", " + kind + '}';
        }
    }

    /** The GPU blocks of one LongCat FFN kind. */
    static final class GpuLongcatFfn implements GpuFfnBlock {
        private final FfnKind kind;
        private final LayerSpec layer;
        private GpuOps ops;
        private GpuDenseFfn dense;
        private GpuMoeFfn moe;
        private ScratchSlot carry;
        private int nEmbd;
        private boolean isBound;

        GpuLongcatFfn(FfnKind kind, LayerSpec layer) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.layer = Objects.requireNonNull(layer, "layer");
            if (!isLongcatKind(kind)) {
                throw new IllegalArgumentException("layer " + layer.il() + " of kind " + kind
                        + " is not a LongCat FFN kind");
            }
            if (layer.ffnKind() != kind || layer.ffn() == null) {
                throw new IllegalArgumentException("layer " + layer.il() + " has FFN kind " + layer.ffnKind()
                        + " and FFN " + layer.ffn() + ", not " + kind);
            }
            if ((kind == FfnKind.MOE_SHORTCUT) != (layer.moe() != null)) {
                throw new IllegalArgumentException("layer " + layer.il() + " of kind " + kind + " has MoE "
                        + layer.moe());
            }
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec target) {
            if (isBound) {
                throw new IllegalStateException("LongCat FFN block of layer " + layer.il() + " is already bound");
            }
            if (target != layer) {
                throw new IllegalArgumentException("block created for layer " + layer.il() + " bound to layer "
                        + target.il());
            }
            this.ops = ctx.ops();
            this.nEmbd = ctx.spec().hparams().nEmbd();
            GpuScratch scratch = ctx.scratch();
            this.carry = scratch.temp(SHORTCUT_TEMP, (long) scratch.ubatch() * nEmbd);
            this.dense = new GpuDenseFfn();
            this.dense.bind(ctx, ffnView(layer, FfnKind.DENSE, layer.ffn(), null));
            if (kind == FfnKind.MOE_SHORTCUT) {
                LayerSpec moeView = ffnView(layer, FfnKind.MOE, null, layer.moe());
                this.moe = new GpuMoeFfn(FfnKind.MOE, moeView);
                this.moe.bind(ctx, moeView);
            }
            this.isBound = true;
        }

        @Override
        public void forward(int n, ScratchSlot in, ScratchSlot out) {
            if (!isBound) {
                throw new IllegalStateException("LongCat FFN block of layer " + layer.il() + " used before bind");
            }
            dense.forward(n, in, out);
            if (moe != null) {
                moe.forward(n, in, carry);
                return;
            }
            int count = Math.multiplyExact(n, nEmbd);
            out.require(count);
            carry.require(count);
            ops.add(out.buffer(), out.offset(), carry.buffer(), carry.offset(), out.buffer(), out.offset(), count);
        }

        @Override
        public boolean capturableDecode() {
            return moe == null || moe.capturableDecode();
        }

        @Override
        public void close() {
            if (moe != null) {
                moe.close();
            }
            if (dense != null) {
                dense.close();
            }
        }

        @Override
        public String toString() {
            return "GpuLongcatFfn{layer=" + layer.il() + ", " + kind + '}';
        }
    }
}
