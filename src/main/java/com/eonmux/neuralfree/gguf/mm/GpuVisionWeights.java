package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device tensors of one vision encoder and its projector, the GPU counterpart of {@link ClipVisionWeights}:
 * projection matrices as resident (in their device types, read by {@link GpuOps#matmul}), the patch-embedding conv
 * weight viewed as its {@code [n_embd][3*ps*ps]} matrix (and any projector conv kernel as its im2col matrix,
 * {@link #kernelMatrix}), the routed experts of MoE blocks as stacked resident tensors, and every tensor an
 * element-wise kernel reads (norm weights, biases, layer scales, position and class rows) as dense single-buffer
 * F32, decoded on the device when the file stores it otherwise. The projector tensors are filed by name as {@link ClipVisionModel#projectorMatrices}
 * and {@link ClipVisionModel#projectorVectors} file them. Absent optionals are null. Immutable after construction;
 * the buffers belong to the weight source and the ops scratch.
 */
final class GpuVisionWeights {

    private static final String TEMP = "vision.f32.";

    /** The patch kernel viewed as its matrix; null for a two-tower text tower, which reads tokens. */
    final GpuWeight patchEmbd;
    /** The second-frame patch kernel of the qwen encoders viewed as its matrix, or null. */
    final GpuWeight patchEmbd1;
    final GpuBuffer patchBias;
    final GpuWeight preLnW;
    final GpuWeight preLnB;
    final GpuWeight postLnW;
    final GpuWeight postLnB;
    final List<Layer> layers;
    private final Map<String, GpuWeight> matrices;
    private final Map<String, GpuWeight> vectors;
    private final Map<String, GpuWeight> towerVectors;
    /** The LoRA site of every dense matrix the encoder serves, by tensor name; empty without deltas. */
    private final Map<String, GpuProjectionDeltas.Site> sites;

    /** Device tensors of one encoder block. */
    static final class Layer {
        final GpuWeight qkv;
        final GpuBuffer qkvB;
        final GpuWeight q;
        final GpuBuffer qB;
        final GpuWeight k;
        final GpuBuffer kB;
        final GpuWeight v;
        final GpuBuffer vB;
        final GpuWeight o;
        final GpuBuffer oB;
        final GpuWeight ln1W;
        final GpuWeight ln1B;
        final GpuWeight ln2W;
        final GpuWeight ln2B;
        final GpuWeight ffnUp;
        final GpuBuffer ffnUpB;
        final GpuWeight ffnGate;
        final GpuBuffer ffnGateB;
        final GpuWeight ffnDown;
        final GpuBuffer ffnDownB;
        /** Layer scales of the attention and FFN outputs ({@code ls1} / {@code ls2}), or null. */
        final GpuBuffer ls1;
        final GpuBuffer ls2;
        /** Q and K norm weights, {@code d_head} (per head) or {@code n_embd} (whole row) wide, or null. */
        final GpuWeight qNorm;
        final GpuWeight kNorm;
        /** The routed experts of a MoE block, or null for a dense FFN. */
        final Moe moe;
        /** Post-attention and post-FFN norm weights and the block output scale (Gemma 4), or null. */
        final GpuWeight attnPostNorm;
        final GpuWeight ffnPostNorm;
        final GpuBuffer outScale;
        /** Clamp bounds {@code {inMin, inMax, outMin, outMax}} per matrix ({@link ClipClamp}), null when unclamped. */
        final float[] qkvClamp;
        final float[] qClamp;
        final float[] kClamp;
        final float[] vClamp;
        final float[] oClamp;
        final float[] upClamp;
        final float[] gateClamp;
        final float[] downClamp;

        Layer(GpuVisionWeights.Resolver r, ClipLayer l) {
            this.moe = l.moe() ? new Moe(r, l) : null;
            boolean dense = moe == null;
            this.qkv = r.matrix(l.qkvWeight());
            this.qkvB = r.buffer(l.qkvBias());
            this.q = r.matrix(l.qWeight());
            this.qB = r.buffer(l.qBias());
            this.k = r.matrix(l.kWeight());
            this.kB = r.buffer(l.kBias());
            this.v = r.matrix(l.vWeight());
            this.vB = r.buffer(l.vBias());
            this.o = r.matrix(l.oWeight());
            this.oB = r.buffer(l.oBias());
            this.ln1W = r.vector(l.ln1Weight());
            this.ln1B = r.vector(l.ln1Bias());
            this.ln2W = r.vector(l.ln2Weight());
            this.ln2B = r.vector(l.ln2Bias());
            this.ffnUp = dense ? r.matrix(l.ffnUpWeight()) : null;
            this.ffnUpB = dense ? r.buffer(l.ffnUpBias()) : null;
            this.ffnGate = dense ? r.matrix(l.ffnGateWeight()) : null;
            this.ffnGateB = dense ? r.buffer(l.ffnGateBias()) : null;
            this.ffnDown = dense ? r.matrix(l.ffnDownWeight()) : null;
            this.ffnDownB = dense ? r.buffer(l.ffnDownBias()) : null;
            this.ls1 = r.buffer(l.ls1Weight());
            this.ls2 = r.buffer(l.ls2Weight());
            this.qNorm = r.vector(l.qNormWeight());
            this.kNorm = r.vector(l.kNormWeight());
            this.attnPostNorm = r.vector(l.attnPostNormWeight());
            this.ffnPostNorm = r.vector(l.ffnPostNormWeight());
            this.outScale = r.buffer(l.outScaleWeight());
            this.qkvClamp = r.clamp(l, l.qkvWeight());
            this.qClamp = r.clamp(l, l.qWeight());
            this.kClamp = r.clamp(l, l.kWeight());
            this.vClamp = r.clamp(l, l.vWeight());
            this.oClamp = r.clamp(l, l.oWeight());
            this.upClamp = dense ? r.clamp(l, l.ffnUpWeight()) : null;
            this.gateClamp = dense ? r.clamp(l, l.ffnGateWeight()) : null;
            this.downClamp = dense ? r.clamp(l, l.ffnDownWeight()) : null;
        }

        boolean fusedQkv() {
            return qkv != null;
        }
    }

    /**
     * The routed experts of one MoE block ({@code build_moe_ffn}): the router matrix, the optional selection bias
     * and the stacked SiLU expert tensors {@code [n_expert][rows][cols]} as resident, each read whole by the
     * id-matmul kernels.
     */
    static final class Moe {
        final GpuWeight router;
        final GpuBuffer selectBias;
        final GpuWeight gate;
        final GpuWeight up;
        final GpuWeight down;
        final int nExpert;
        final int nFf;

        Moe(GpuVisionWeights.Resolver r, ClipLayer l) {
            this.nExpert = l.nExpert();
            this.nFf = l.nFfExpert();
            this.router = r.matrix(l.moeRouterWeight());
            this.selectBias = r.buffer(l.moeSelectBias());
            this.gate = requirePlainExperts(r.matrix(l.moeGateWeight()));
            this.up = requirePlainExperts(r.matrix(l.moeUpWeight()));
            this.down = requirePlainExperts(r.matrix(l.moeDownWeight()));
        }

        /** The id-matmul kernels read the stored bytes as they are: no input transform, rotation or scale. */
        private static GpuWeight requirePlainExperts(GpuWeight w) {
            if (w.hasInputTransform() || w.hadamard() != null || w.hasScale()) {
                throw new UnsupportedOperationException("expert tensor " + w.name() + " carries an input transform,"
                        + " a rotation or a scale, which the GPU vision experts do not apply");
            }
            return w;
        }
    }

    /**
     * Resolves every tensor of {@code model} from {@code weights}.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is not resident
     * @throws IllegalArgumentException when a sharded patch-embedding weight splits an output channel
     */
    GpuVisionWeights(ClipVisionModel model, GpuOps ops, GpuWeights weights, GpuProjectionDeltas deltas) {
        Resolver r = new Resolver(Objects.requireNonNull(ops, "ops"), Objects.requireNonNull(weights, "weights"));
        ClipHparams hp = model.hparams();
        this.patchEmbd = model.patchEmbd() == null ? null
                : r.patchMatrix(model.patchEmbd(), model.patchChannels(),
                        ClipPatches.rowWidth(model.patchKernelSize()));
        this.patchEmbd1 = model.patchEmbd1() == null ? null
                : r.patchMatrix(model.patchEmbd1(), hp.nEmbd(), ClipPatches.rowWidth(hp.patchSize()));
        this.patchBias = r.buffer(model.patchBias());
        this.preLnW = r.vector(model.preLnWeight());
        this.preLnB = r.vector(model.preLnBias());
        this.postLnW = r.vector(model.postLnWeight());
        this.postLnB = r.vector(model.postLnBias());
        List<Layer> ls = new ArrayList<>(model.layers().size());
        Map<String, GpuProjectionDeltas.Site> served = new HashMap<>();
        for (ClipLayer l : model.layers()) {
            ls.add(new Layer(r, l));
            if (deltas != null) {
                serveLayer(served, deltas, r, l);
            }
        }
        this.layers = Collections.unmodifiableList(ls);
        Map<String, GpuWeight> mats = new HashMap<>();
        for (Map.Entry<String, TensorRef> e : model.projectorMatrices().entrySet()) {
            GpuWeight matrix = r.matrix(e.getValue());
            mats.put(e.getKey(), matrix);
            if (deltas != null) {
                serve(served, deltas, e.getValue(), matrix);
            }
        }
        Map<String, GpuWeight> vecs = new HashMap<>();
        for (Map.Entry<String, TensorRef> e : model.projectorVectors().entrySet()) {
            vecs.put(e.getKey(), r.vector(e.getValue()));
        }
        this.matrices = Collections.unmodifiableMap(mats);
        this.vectors = Collections.unmodifiableMap(vecs);
        this.sites = Collections.unmodifiableMap(served);
        Map<String, GpuWeight> tower = new HashMap<>();
        for (TensorRef ref : new TensorRef[]{model.classEmbd(), model.posEmbd(), model.normEmbdWeight(),
                model.normEmbdBias()}) {
            if (ref != null) {
                tower.put(ref.name(), r.vector(ref));
            }
        }
        for (GpuWeight norm : new GpuWeight[]{preLnW, preLnB, postLnW, postLnB}) {
            if (norm != null) {
                tower.put(norm.name(), norm);
            }
        }
        this.towerVectors = Collections.unmodifiableMap(tower);
    }

    /**
     * The projector matrix {@code name}.
     *
     * @throws IllegalStateException when the projector graph resolved no such matrix
     */
    GpuWeight matrix(String name) {
        GpuWeight m = matrices.get(name);
        if (m == null) {
            throw new IllegalStateException("the projector resolved no matrix " + name);
        }
        return m;
    }

    /** The projector matrix {@code name}, or null when the file does not carry the optional tensor. */
    GpuWeight matrixOrNull(String name) {
        return matrices.get(name);
    }

    /**
     * The projector vector {@code name} as a one-row F32 weight (a norm weight or bias).
     *
     * @throws IllegalStateException when the projector graph resolved no such vector
     */
    GpuWeight vector(String name) {
        GpuWeight v = vectors.get(name);
        if (v == null) {
            throw new IllegalStateException("the projector resolved no vector " + name);
        }
        return v;
    }

    /** The projector vector {@code name} as a one-row F32 weight, or null when the file does not carry it. */
    GpuWeight vectorOrNull(String name) {
        return vectors.get(name);
    }

    /** The F32 buffer of projector vector {@code name}, or null when the file does not carry it. */
    GpuBuffer bufferOrNull(String name) {
        GpuWeight v = vectors.get(name);
        return v == null ? null : v.shard(0);
    }

    /**
     * A projector or input tensor the graphs read by reference (the class and position rows, the embedding norm,
     * a projector vector) as a one-row F32 weight, or null for a null reference.
     *
     * @throws IllegalStateException when the tensor was not resolved
     */
    GpuWeight vectorOf(TensorRef ref) {
        if (ref == null) {
            return null;
        }
        GpuWeight v = towerVectors.containsKey(ref.name()) ? towerVectors.get(ref.name()) : vectors.get(ref.name());
        if (v == null) {
            throw new IllegalStateException("the vision encoder resolved no tensor " + ref.name());
        }
        return v;
    }

    /**
     * The projector conv kernel {@code name} viewed as the {@code [outRows][inWidth]} matrix its contiguous bytes
     * already are (a stride-{@code s} conv read as the matmul of its im2col rows).
     *
     * @throws IllegalStateException    when the projector graph resolved no such matrix
     * @throws IllegalArgumentException when the kernel does not hold {@code outRows * inWidth} values, or a shard
     *                                  splits an output channel
     */
    GpuWeight kernelMatrix(String name, int outRows, int inWidth) {
        GpuWeight w = matrix(name);
        if (w.rows() * w.cols() != (long) outRows * inWidth) {
            throw new IllegalArgumentException("conv kernel " + name + " holds " + w.rows() * w.cols()
                    + " values, not a " + outRows + "x" + inWidth + " matrix");
        }
        return Resolver.asMatrix(w, outRows, inWidth);
    }

    /**
     * The LoRA site of dense product {@code name}, or null when the encoder does not serve it as one (an absent
     * tensor, an expert tensor, or an encoder bound without deltas).
     */
    GpuProjectionDeltas.Site site(String name) {
        return sites.get(name);
    }

    /** Serves the dense product of {@code ref} through {@code weight}, when both are present. */
    private static void serve(Map<String, GpuProjectionDeltas.Site> served, GpuProjectionDeltas deltas,
                              TensorRef ref, GpuWeight weight) {
        if (ref != null && weight != null) {
            served.put(ref.name(), deltas.serveModel(ref.name(), weight));
        }
    }

    /**
     * Serves the dense matrices of one block: the fused or separate Q/K/V, the output projection, the dense FFN
     * and the MoE router. The routed experts are 3-D products the encoder reads per expert, which no vision adapter
     * targets; they stay unserved and an adapter naming one is refused.
     */
    private static void serveLayer(Map<String, GpuProjectionDeltas.Site> served, GpuProjectionDeltas deltas,
                                   Resolver r, ClipLayer l) {
        for (TensorRef t : new TensorRef[]{l.qkvWeight(), l.qWeight(), l.kWeight(), l.vWeight(), l.oWeight(),
                l.ffnUpWeight(), l.ffnGateWeight(), l.ffnDownWeight(), l.moeRouterWeight()}) {
            if (t != null) {
                serve(served, deltas, t, r.matrix(t));
            }
        }
    }

    /** Looks tensors up and converts the element-wise ones to dense F32. */
    static final class Resolver {
        private final GpuOps ops;
        private final GpuWeights weights;

        Resolver(GpuOps ops, GpuWeights weights) {
            this.ops = ops;
            this.weights = weights;
        }

        /** A projection matrix as resident, or null for an absent optional. */
        GpuWeight matrix(TensorRef ref) {
            return ref == null ? null : weights.get(ref.name());
        }

        /** A vector or small table as a single-buffer F32 weight of one row, or null for an absent optional. */
        GpuWeight vector(TensorRef ref) {
            return ref == null ? null : vector(ref.name());
        }

        private GpuWeight vector(String name) {
            GpuWeight w = weights.get(name);
            long n = w.rows() * w.cols();
            if (w.type() == GGMLType.F32 && w.isSingleShard()) {
                return w.rows() == 1 ? w : GpuWeight.of(w.name(), GGMLType.F32, new long[]{n}, w.shard(0));
            }
            return GpuWeight.of(w.name(), GGMLType.F32, new long[]{n}, decoded(w));
        }

        /** The dense F32 buffer of a tensor, or null for an absent optional. */
        GpuBuffer buffer(TensorRef ref) {
            GpuWeight w = vector(ref);
            return w == null ? null : w.shard(0);
        }

        /** The clamp bounds of block matrix {@code weight}, read to the host, or null when it has none. */
        float[] clamp(ClipLayer l, TensorRef weight) {
            ClipClamp c = l.clamp(weight);
            return c == null ? null : c.values(this::hostFloats);
        }

        /** The values of the small tensor {@code name}, read from the device. */
        private float[] hostFloats(String name) {
            GpuWeight w = vector(name);
            float[] host = new float[Math.toIntExact(w.rows() * w.cols())];
            GpuEvent e = ops.queue().enqueueReadBuffer(w.shard(0), 0, (long) host.length * Float.BYTES, host);
            if (e != null) {
                e.close();
            }
            ops.queue().finish();
            return host;
        }

        /**
         * The patch-embedding conv weight ggml {@code {ps, ps, 3, n_embd}} viewed as the {@code [n_embd][3*ps*ps]}
         * matrix its contiguous bytes already are.
         */
        GpuWeight patchMatrix(TensorRef ref, int nEmbd, int patchWidth) {
            return asMatrix(weights.get(ref.name()), nEmbd, patchWidth);
        }

        /** {@code w}'s bytes viewed as {@code [outRows][inWidth]}, shard by shard over whole output rows. */
        static GpuWeight asMatrix(GpuWeight w, int outRows, int inWidth) {
            long[] ne = {inWidth, outRows};
            if (w.isSingleShard()) {
                return GpuWeight.of(w.name(), w.type(), ne, w.shard(0));
            }
            long rowsPerChannel = w.rows() / outRows;
            List<GpuBuffer> bufs = new ArrayList<>(w.shardCount());
            long[] rows = new long[w.shardCount()];
            for (int s = 0; s < w.shardCount(); s++) {
                if (w.shardRows(s) % rowsPerChannel != 0) {
                    throw new IllegalArgumentException("tensor " + w.name() + ": shard " + s + " holds "
                            + w.shardRows(s) + " rows, not whole output channels of " + rowsPerChannel);
                }
                bufs.add(w.shard(s));
                rows[s] = w.shardRows(s) / rowsPerChannel;
            }
            return GpuWeight.sharded(w.name(), w.type(), ne, bufs, rows);
        }

        /** Decodes every shard of {@code w} into a dense F32 scratch buffer ({@link GpuOps#embed} of one row). */
        private GpuBuffer decoded(GpuWeight w) {
            long cols = w.cols();
            ScratchSlot dst = ops.scratch().temp(TEMP + w.name(), w.rows() * cols);
            if (dst.offset() != 0) {
                throw new IllegalStateException("scratch temporary " + dst.name() + " does not start at element 0");
            }
            for (int s = 0; s < w.shardCount(); s++) {
                long n = w.shardRows(s) * cols;
                GpuWeight flat = GpuWeight.of(w.name(), w.type(), new long[]{n}, w.shard(s));
                ops.embed(flat, new int[]{0}, dst.buffer(), Math.toIntExact(w.shardRowStart(s) * cols));
            }
            return dst.buffer();
        }
    }
}
