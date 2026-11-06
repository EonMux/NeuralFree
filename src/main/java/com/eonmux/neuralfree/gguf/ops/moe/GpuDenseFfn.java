package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * A dense FFN ({@code build_ffn}, llama-graph.cpp:1923-2140) through {@link GpuOps}: the shared experts and the
 * parallel dense FFN of the MoE blocks. Per-tensor {@code w_s} scales are applied by {@link GpuOps} from
 * {@link GpuWeight#scale()}; a product with deltas runs unscaled, adds them and then applies {@code w_s}, as
 * {@code build_ffn} scales after {@code build_lora_mm} (llama-graph.cpp:1979-2010). Its scratch slots and weights
 * belong to the context; the block that binds it holds nothing to close.
 */
final class GpuDenseFfn {

    private static final String T_FFN_GATE = "moe.ffn_gate";
    private static final String T_FFN_UP = "moe.ffn_up";
    private static final String T_FFN_OUT = "moe.ffn_out";

    private final FfnSpec f;
    private final LayerSpec layer;
    private final GpuOps ops;
    private final GpuWeight gate;
    private final GpuWeight up;
    private final GpuWeight down;
    private final GpuProjectionDeltas.Site gateSite;
    private final GpuProjectionDeltas.Site upSite;
    private final GpuProjectionDeltas.Site downSite;
    /** The weights without {@code w_s}, for the products with deltas. */
    private final GpuWeight gatePlain;
    private final GpuWeight upPlain;
    private final GpuWeight downPlain;
    private final ScratchSlot g;
    private final ScratchSlot u;
    private final ScratchSlot out;
    private final int ubatch;

    /**
     * Binds the FFN of {@code f} over the layer's weights.
     *
     * @param nEmbd width of the FFN input rows (the block's stream width)
     * @throws UnsupportedOperationException for an activation-sparse or sequentially gated FFN, which the MoE graphs
     *                                       do not build
     * @throws IllegalStateException         when the tensors do not match {@code nEmbd} and {@code n_ff}
     */
    GpuDenseFfn(GpuForwardContext ctx, GpuWeights w, LayerSpec layer, FfnSpec f, int nEmbd) {
        this.f = Objects.requireNonNull(f, "ffn");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        if (f.activationSparsityStdMul() != 0f) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": " + f.role()
                    + " FFN with activation sparsity is not part of a MoE graph");
        }
        if (f.gating() == FfnGating.SEQ && f.gated()) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": sequentially gated " + f.role()
                    + " FFN is not built by the MoE graphs");
        }
        int nFf = f.nFf();
        boolean fused = f.gating() == FfnGating.FUSED_UP;
        this.gate = f.gating() == FfnGating.PAR ? w.layer(layer, f.gateWeight()) : null;
        this.up = w.layer(layer, f.upWeight());
        this.down = w.layer(layer, f.downWeight());
        long upRows = fused ? 2L * nFf : nFf;
        if (up.cols() != nEmbd || up.rows() != upRows || down.cols() != nFf || down.rows() != nEmbd
                || (gate != null && (gate.cols() != nEmbd || gate.rows() != nFf))) {
            throw new IllegalStateException("layer " + layer.il() + ": " + f.role() + " FFN tensors " + gate + ", "
                    + up + ", " + down + " do not match n_embd " + nEmbd + " and n_ff " + nFf);
        }
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        this.gateSite = gate != null ? deltas.serve(layer, f.gateWeight(), gate) : null;
        this.upSite = deltas.serve(layer, f.upWeight(), up);
        this.downSite = deltas.serve(layer, f.downWeight(), down);
        this.gatePlain = plain(gate);
        this.upPlain = plain(up);
        this.downPlain = plain(down);
        GpuScratch scratch = ctx.scratch();
        this.g = gate != null || fused ? scratch.temp(T_FFN_GATE, ubatch * upRows) : null;
        this.u = scratch.temp(T_FFN_UP, ubatch * nFf);
        this.out = scratch.temp(T_FFN_OUT, ubatch * nEmbd);
    }

    /** The output rows: {@code ubatch} rows of {@code n_embd}, written by the last {@link #run}. */
    ScratchSlot out() {
        return out;
    }

    /** {@link #out} = FFN of {@code n} rows at {@code x[xOff]}. */
    void run(int n, GpuBuffer x, int xOff) {
        int nFf = f.nFf();
        switch (f.gating()) {
            case PAR:
                project(up, upPlain, upSite, x, xOff, u, n);
                project(gate, gatePlain, gateSite, x, xOff, g, n);
                ops.glu(f.glu(), f.params(), g.buffer(), g.offset(), nFf, u.buffer(), u.offset(), nFf,
                        u.buffer(), u.offset(), nFf, n, nFf);
                break;
            case FUSED_UP:
                project(up, upPlain, upSite, x, xOff, g, n);
                ops.glu(f.glu(), f.params(), g.buffer(), g.offset(), 2 * nFf, g.buffer(), g.offset() + nFf,
                        2 * nFf, u.buffer(), u.offset(), nFf, n, nFf);
                break;
            case SEQ:
                project(up, upPlain, upSite, x, xOff, u, n);
                ops.unary(f.act(), f.params(), u.buffer(), u.offset(), u.buffer(), u.offset(), n * nFf);
                break;
            default:
                throw new IllegalStateException("unhandled FFN gating " + f.gating());
        }
        project(down, downPlain, downSite, u.buffer(), u.offset(), out, n);
    }

    private static GpuWeight plain(GpuWeight w) {
        return w != null && w.hasScale() ? w.withScale(1.0f) : w;
    }

    /** {@code y = (W x + deltas) * w_s} over {@code n} rows; {@code plain} is {@code w} without {@code w_s}. */
    private void project(GpuWeight w, GpuWeight plain, GpuProjectionDeltas.Site site, GpuBuffer x, int xOff,
                         ScratchSlot y, int n) {
        if (site.isEmpty()) {
            ops.matmul(w, x, xOff, y.buffer(), y.offset(), n);
            return;
        }
        ops.matmul(plain, x, xOff, y.buffer(), y.offset(), n);
        site.applyRows(n, x, xOff, y.buffer(), y.offset());
        if (w.hasScale()) {
            ops.scale(y.buffer(), y.offset(), Math.toIntExact((long) n * w.rows()), w.scale());
        }
    }

    @Override
    public String toString() {
        return "GpuDenseFfn{layer=" + layer.il() + ", " + f.role() + ", n_ff=" + f.nFf() + "}";
    }
}
