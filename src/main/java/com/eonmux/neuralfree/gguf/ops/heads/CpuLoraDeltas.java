package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;

import java.util.Objects;

/**
 * Host deltas of one LoRA pair bound to its scale ({@link LoraAdapter.Delta}), as the three kinds of
 * {@link CpuProjectionDeltas}: a projection ({@link LoraAdapter.Delta#add}), the selected experts
 * ({@link LoraAdapter.Delta#addExperts}) and token-embedding rows ({@link LoraAdapter.Delta#addEmbedding}). Each delta
 * owns a scratch row grown on demand; one forward pass at a time.
 */
public final class CpuLoraDeltas {

    private CpuLoraDeltas() {
    }

    /** The host delta of {@code d} for its pair's kind. */
    public static Object of(LoraAdapter.Delta d) {
        switch (Objects.requireNonNull(d, "delta").pair().kind()) {
            case EXPERTS:
                return new Experts(d);
            case EMBEDDING:
                return new Embedding(d);
            default:
                return new Projection(d);
        }
    }

    /** A dense projection delta. */
    public static final class Projection implements CpuProjectionDeltas.Delta {
        private final LoraAdapter.Delta delta;
        private float[] tmp = new float[0];

        Projection(LoraAdapter.Delta delta) {
            this.delta = delta;
        }

        @Override
        public void add(int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride) {
            int need = n * delta.pair().rank();
            if (tmp.length < need) {
                tmp = new float[need];
            }
            delta.add(n, x, xOff, xStride, tmp, 0, y, yOff, yStride);
        }
    }

    /** A per-expert delta over the selected pairs. */
    public static final class Experts implements CpuProjectionDeltas.ExpertDelta {
        private final LoraAdapter.Delta delta;
        private final float[] tmp;

        Experts(LoraAdapter.Delta delta) {
            this.delta = delta;
            this.tmp = new float[delta.pair().rank() + delta.pair().outDim()];
        }

        @Override
        public void add(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] y, int yOff) {
            delta.addExperts(pairs, k, ids, x, xOff, tokenRows, tmp, y, yOff);
        }
    }

    /** A token-embedding delta over looked-up rows. */
    public static final class Embedding implements CpuProjectionDeltas.EmbeddingDelta {
        private final LoraAdapter.Delta delta;

        Embedding(LoraAdapter.Delta delta) {
            this.delta = delta;
        }

        @Override
        public void add(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            delta.addEmbedding(n, ids, idsOff, y, yOff, yStride);
        }
    }
}
