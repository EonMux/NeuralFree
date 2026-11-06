package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.CpuVitTower.floatsOrNull;

/**
 * The weights of one {@link ClipVisionModel}, resolved once at bind: the encoder blocks and pre/post norms as the
 * shared {@link CpuVitTower}, matrices as {@link CpuTensorRef} views read by {@link CpuMatVec} in their stored
 * types, vectors (biases, norms, embedding rows) as floats. Absent optionals are null. Package-private executor
 * support.
 */
final class ClipVisionWeights {

    final CpuMatVec matVec;
    /** The patch kernel, or null for the encoders that embed no patches (the DeepSeek-OCR towers read SAM rows,
     *  the MobileNetV5 conv stack its own stem). */
    final CpuTensorRef patchEmbd;
    /** The second patch kernel of the qwen encoders (second frame) and GLM-4V, or null. */
    final CpuTensorRef patchEmbd1;
    final float[] patchBias;
    final float[] normEmbdW;
    final float[] normEmbdB;
    final float[] classEmbd;
    final float[] posEmbd;
    final CpuVitTower tower;
    private final Function<String, float[]> floats;
    private final Map<String, CpuTensorRef> matrices;
    private final Map<String, float[]> vectors;

    ClipVisionWeights(ClipVisionModel m, CpuMatVec matVec, Function<String, float[]> floats,
                      Function<String, CpuTensorRef> refs, CpuProjectionDeltas deltas) {
        this.matVec = Objects.requireNonNull(matVec, "matVec");
        this.floats = floats;
        ClipHparams hp = m.hparams();
        long patchWidth = 3L * m.patchKernelSize() * m.patchKernelSize();
        this.patchEmbd = m.patchEmbd() == null ? null
                : CpuVitTower.matrix(refs.apply(m.patchEmbd().name()), m.patchChannels(), patchWidth);
        this.patchEmbd1 = m.patchEmbd1() == null ? null : CpuVitTower.matrix(refs.apply(m.patchEmbd1().name()),
                hp.nEmbd(), 3L * hp.patchSize() * hp.patchSize());
        this.patchBias = floatsOrNull(m.patchBias(), floats);
        this.normEmbdW = floatsOrNull(m.normEmbdWeight(), floats);
        this.normEmbdB = floatsOrNull(m.normEmbdBias(), floats);
        this.classEmbd = floatsOrNull(m.classEmbd(), floats);
        this.posEmbd = floatsOrNull(m.posEmbd(), floats);
        this.tower = new CpuVitTower(hp, matVec, m.layers(), m.preLnWeight(), m.preLnBias(), m.postLnWeight(),
                m.postLnBias(), floats, refs, deltas);
        Map<String, CpuTensorRef> mats = new HashMap<>();
        for (Map.Entry<String, TensorRef> e : m.projectorMatrices().entrySet()) {
            mats.put(e.getKey(), refs.apply(e.getValue().name()));
        }
        Map<String, float[]> vecs = new HashMap<>();
        for (Map.Entry<String, TensorRef> e : m.projectorVectors().entrySet()) {
            vecs.put(e.getKey(), floats.apply(e.getValue().name()));
        }
        this.matrices = Collections.unmodifiableMap(mats);
        this.vectors = Collections.unmodifiableMap(vecs);
    }

    /**
     * The projector matrix {@code name}.
     *
     * @throws IllegalStateException when the projector graph resolved no such matrix
     */
    CpuTensorRef matrix(String name) {
        CpuTensorRef r = matrices.get(name);
        if (r == null) {
            throw new IllegalStateException("the projector resolved no matrix " + name);
        }
        return r;
    }

    /** The projector matrix {@code name}, or null when the file does not carry the optional tensor. */
    CpuTensorRef matrixOrNull(String name) {
        return matrices.get(name);
    }

    /**
     * The projector vector {@code name}.
     *
     * @throws IllegalStateException when the projector graph resolved no such vector
     */
    float[] vector(String name) {
        float[] v = vectors.get(name);
        if (v == null) {
            throw new IllegalStateException("the projector resolved no vector " + name);
        }
        return v;
    }

    /** The projector vector {@code name}, or null when the file does not carry the optional tensor. */
    float[] vectorOrNull(String name) {
        return vectors.get(name);
    }

    /** The floats of a tower tensor, or null for an absent optional. */
    float[] floats(TensorRef ref) {
        return ref == null ? null : floats.apply(ref.name());
    }
}
