package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * Adaptive RMS norm of Voxtral realtime ({@code NormSpec.ffnAda()}, llama.cpp src/models/mistral3.cpp): the
 * {@code ffn_norm} output {@code cur} of a dense layer becomes {@code cur + cur * ada} with
 * <pre>
 * ada    = ffn_norm_ada_up · gelu_erf(ffn_norm_ada_down · t_cond)           (build_ffn SEQ, :227-236)
 * t_cond = [cos(f_0) .. cos(f_{h-1}), sin(f_0) .. sin(f_{h-1})]             (h = n_embd / 2, :99-108)
 * f_i    = exp(-ln(10000) * i / h) * n_tokens_delay
 * </pre>
 * {@code t_cond} is constant for the session; {@code ada} is recomputed per step, as llama.cpp does, so LoRA deltas on
 * the ada projections take effect. This class holds the host math and the CPU application; the device twin is
 * {@link GpuAdaNorm}. Stateless.
 */
public final class AdaNormMath {

    private AdaNormMath() {
    }

    /**
     * The sinusoidal time condition of {@code delay} tokens over {@code width} values; an odd last value stays 0
     * (mistral3.cpp:101-108).
     *
     * @throws IllegalArgumentException when {@code width < 1} or {@code delay < 0}
     */
    public static float[] timeCondition(int delay, int width) {
        if (width < 1 || delay < 0) {
            throw new IllegalArgumentException("time condition of delay " + delay + " over width " + width);
        }
        float[] t = new float[width];
        int half = width / 2;
        for (int i = 0; i < half; i++) {
            double freq = Math.exp(-Math.log(10000.0) * i / half) * delay;
            t[i] = (float) Math.cos(freq);
            t[half + i] = (float) Math.sin(freq);
        }
        return t;
    }

    /** One layer's ada projections on the CPU, with their LoRA delta sites. */
    public static final class CpuLayer {
        private final CpuTensorRef down;
        private final CpuTensorRef up;
        private final CpuProjectionDeltas.Site downDeltas;
        private final CpuProjectionDeltas.Site upDeltas;
        private final int width;
        private final int hidden;

        /**
         * @throws IllegalArgumentException when the projections are not a {@code [width, h]} / {@code [h, width]} pair
         */
        public CpuLayer(CpuTensorRef down, CpuTensorRef up, CpuProjectionDeltas.Site downDeltas,
                        CpuProjectionDeltas.Site upDeltas, int width) {
            this.down = Objects.requireNonNull(down, "down");
            this.up = Objects.requireNonNull(up, "up");
            this.downDeltas = Objects.requireNonNull(downDeltas, "downDeltas");
            this.upDeltas = Objects.requireNonNull(upDeltas, "upDeltas");
            if (down.cols() != width || up.rows() != width || up.cols() != down.rows() || down.rows() < 1
                    || down.rows() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("ada projections " + down.cols() + "x" + down.rows() + " and "
                        + up.cols() + "x" + up.rows() + " do not map width " + width + " through a hidden width");
            }
            this.width = width;
            this.hidden = (int) down.rows();
        }

        /** Hidden width of the ada MLP. */
        public int hidden() {
            return hidden;
        }

        /**
         * Applies {@code x = x + x * ada} in place to {@code nRows} rows of {@code width} floats at {@code xOff}.
         *
         * @param tCond   {@link #timeCondition} of the layer width
         * @param hiddenT scratch of at least {@link #hidden()} floats
         * @param ada     scratch of at least {@code width} floats
         * @throws IllegalArgumentException when a buffer is too small
         */
        public void apply(CpuMatVec mv, float[] tCond, float[] hiddenT, float[] ada, float[] x, int xOff, int nRows) {
            if (tCond.length < width || hiddenT.length < hidden || ada.length < width || nRows < 0
                    || xOff < 0 || (long) xOff + (long) nRows * width > x.length) {
                throw new IllegalArgumentException("ada norm buffers too small for " + nRows + " rows of " + width);
            }
            mv.matmul(down, tCond, 0, hiddenT, 0, 1);
            downDeltas.applyRows(1, tCond, 0, hiddenT, 0);
            CpuOps.unary(ActOp.GELU_ERF, ActParams.NONE, hiddenT, 0, hiddenT, 0, hidden);
            mv.matmul(up, hiddenT, 0, ada, 0, 1);
            upDeltas.applyRows(1, hiddenT, 0, ada, 0);
            for (int r = 0; r < nRows; r++) {
                int base = xOff + r * width;
                for (int c = 0; c < width; c++) {
                    float v = x[base + c];
                    x[base + c] = v + v * ada[c];
                }
            }
        }
    }
}
