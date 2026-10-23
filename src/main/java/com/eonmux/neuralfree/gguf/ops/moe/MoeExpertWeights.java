package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The expert tensors of one expert set of a MoE layer, as {@code build_moe_ffn} reads them
 * (src/llama-graph.cpp:2367-2530): the gate and up projections (separate {@code ffn_gate_exps} / {@code ffn_up_exps},
 * the fused {@code ffn_gate_up_exps} whose expert {@code e} holds {@code 2 * n_ff} rows with the gate rows first, or
 * an ungated {@code ffn_up_exps}), {@code ffn_down_exps}, the per-expert scales {@code <exps>.scale}
 * ({@code {n_expert}}, llama-model.cpp:1764-1773) and the expert biases {@code <exps>.bias}
 * ({@code {rows, n_expert}}).
 *
 * <p>Two sets exist: the routed experts and the grovemoe chunk experts {@code ffn_*_chexps}, which llama.cpp runs
 * without scales or biases (grovemoe.cpp:156-166). The fused tensor is scaled by {@code ffn_up_exps.scale} only and
 * never by a gate scale (llama-graph.cpp:2370-2377). Shapes follow ggml: an expert tensor is
 * {@code [in_dim, rows, n_expert]}, so expert {@code e} row {@code r} is tensor row {@code e * rows + r}.</p>
 *
 * <p>A device expert tensor larger than one binding is held in row shards of whole experts
 * ({@link GpuWeight}); {@link Gpu} exposes each projection as its {@link Shard}s, one id-matvec binding each. A host
 * tensor ({@link com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef}) may be split into row shards anywhere on a row
 * boundary; {@link CpuExpertMatmul} reads expert {@code e}'s rows {@code [e * rows, (e + 1) * rows)} from whichever
 * shards hold them.</p>
 *
 * <p>{@link #resolve(GpuWeights, LayerSpec)} and {@link #resolve(CpuWeights, LayerSpec)} look the tensors up at the
 * layer's weight layer and check every shape; a required tensor that is absent throws
 * {@link com.eonmux.neuralfree.gguf.spec.MissingTensorException}. Immutable.</p>
 */
final class MoeExpertWeights {

    /** Which expert set of the layer. */
    enum Set {
        /** {@code ffn_{gate,up,gate_up,down}_exps} with scales and biases. */
        ROUTED(TensorNames.FFN_GATE_EXPS, TensorNames.FFN_UP_EXPS, TensorNames.FFN_DOWN_EXPS, true),
        /** grovemoe {@code ffn_{gate,up,down}_chexps}, gated, without scales or biases. */
        CHUNK(TensorNames.FFN_GATE_CHEXPS, TensorNames.FFN_UP_CHEXPS, TensorNames.FFN_DOWN_CHEXPS, false);

        final String gate;
        final String up;
        final String down;
        final boolean extras;

        Set(String gate, String up, String down, boolean extras) {
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.extras = extras;
        }
    }

    final Set set;
    final int nExpert;
    final int nFf;
    final int inDim;
    final int outDim;
    final boolean gated;
    final boolean fused;
    final GluOp glu;
    final ActOp act;
    final ActParams params;
    final boolean expectScales;
    final boolean expectBiases;

    private MoeExpertWeights(Set set, int nExpert, int nFf, int inDim, int outDim, boolean gated, boolean fused,
                             GluOp glu, ActOp act, ActParams params, boolean expectScales, boolean expectBiases) {
        if (nExpert <= 0 || nFf <= 0 || inDim <= 0 || outDim <= 0) {
            throw new IllegalArgumentException("expert set " + set + " needs positive sizes, got n_expert " + nExpert
                    + ", n_ff " + nFf + ", in " + inDim + ", out " + outDim);
        }
        if (gated == (glu == null) || (!gated && act == null)) {
            throw new IllegalArgumentException("expert set " + set + ": a gated set needs a GLU op, an ungated one an"
                    + " activation (glu " + glu + ", act " + act + ")");
        }
        this.set = set;
        this.nExpert = nExpert;
        this.nFf = nFf;
        this.inDim = inDim;
        this.outDim = outDim;
        this.gated = gated;
        this.fused = fused;
        this.glu = glu;
        this.act = act;
        this.params = Objects.requireNonNull(params, "params");
        this.expectScales = expectScales;
        this.expectBiases = expectBiases;
    }

    /**
     * The routed experts of a layer whose experts read and write rows of {@code dim} values (the stream width, or
     * the latent width of a latent MoE).
     */
    static MoeExpertWeights routed(MoeSpec spec, int dim) {
        return new MoeExpertWeights(Set.ROUTED, spec.nExpert(), spec.nFfExp(), dim, dim, spec.gated(),
                spec.fusedGateUp(), spec.glu(), spec.act(), spec.params(), spec.expertScales(), spec.expertBiases());
    }

    /**
     * The grovemoe chunk experts: the routed experts' GLU on {@code nChunkExperts} experts of {@code nFfChunk} rows
     * (grovemoe.cpp:156-166 passes the same op with the routed clamp rules and no scales or biases).
     *
     * @throws IllegalStateException when the spec has no chunk experts or the routed experts are not separately gated
     */
    static MoeExpertWeights chunks(MoeSpec spec, int dim) {
        if (!spec.chunked()) {
            throw new IllegalStateException("MoE spec has no chunk experts: " + spec);
        }
        if (!spec.gated() || spec.fusedGateUp()) {
            throw new IllegalStateException("chunk experts need separately gated routed experts: " + spec);
        }
        return new MoeExpertWeights(Set.CHUNK, spec.nChunkExperts(), spec.nFfChunk(), dim, dim, true, false,
                spec.glu(), null, spec.params(), false, false);
    }

    /** Rows of one expert of the gate-up product: {@code 2 * n_ff} for the fused tensor, else {@code n_ff}. */
    int gateUpRows() {
        return fused ? 2 * nFf : nFf;
    }

    // ------------------------------------------------------------------ names

    String gateName() {
        return TensorNames.weight(set.gate);
    }

    String upName() {
        return TensorNames.weight(set.up);
    }

    String gateUpName() {
        return TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS);
    }

    String downName() {
        return TensorNames.weight(set.down);
    }

    // ------------------------------------------------------------------ GPU

    /**
     * One device shard of an expert tensor: the experts {@code [expertBase, expertBase + expertCount)} stored from
     * byte 0 of {@link #buffer}, bound by one id-matvec dispatch ({@code expert_base}, {@code expert_count}).
     * Immutable.
     */
    static final class Shard {
        final GpuBuffer buffer;
        final int expertBase;
        final int expertCount;

        Shard(GpuBuffer buffer, int expertBase, int expertCount) {
            this.buffer = Objects.requireNonNull(buffer, "buffer");
            if (expertBase < 0 || expertCount <= 0) {
                throw new IllegalArgumentException("expert shard needs base >= 0 and count > 0, got " + expertBase
                        + ", " + expertCount);
            }
            this.expertBase = expertBase;
            this.expertCount = expertCount;
        }

        /** True when both shards hold the same experts (they may live in different buffers). */
        boolean sameExperts(Shard o) {
            return expertBase == o.expertBase && expertCount == o.expertCount;
        }

        @Override
        public String toString() {
            return "experts [" + expertBase + ", " + (expertBase + expertCount) + ")";
        }
    }

    /**
     * True when two shard lists split the experts identically, so one dispatch per shard can bind a shard of each
     * (the separate gate and up tensors of {@code {T}_MatVecIdGateUp}).
     */
    static boolean samePartition(List<Shard> a, List<Shard> b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (int i = 0; i < a.size(); i++) {
            if (!a.get(i).sameExperts(b.get(i))) {
                return false;
            }
        }
        return true;
    }

    /** Device tensors of one expert set; null fields are absent tensors. */
    static final class Gpu {
        final MoeExpertWeights plan;
        /** Separate gate, or null (fused or ungated). */
        final GpuWeight gate;
        /** Separate or ungated up, or null (fused). */
        final GpuWeight up;
        /** Fused gate_up, or null. */
        final GpuWeight gateUp;
        final GpuWeight down;
        /** Expert shards of {@link #gate}, {@link #up}, {@link #gateUp} and {@link #down}; null with the tensor. */
        final List<Shard> gateShards;
        final List<Shard> upShards;
        final List<Shard> gateUpShards;
        final List<Shard> downShards;
        final GpuWeight gateScale;
        /** {@code ffn_up_exps.scale}; scales the fused product too. */
        final GpuWeight upScale;
        final GpuWeight downScale;
        final GpuWeight gateBias;
        final GpuWeight upBias;
        final GpuWeight gateUpBias;
        final GpuWeight downBias;

        private Gpu(MoeExpertWeights plan, GpuWeight gate, GpuWeight up, GpuWeight gateUp, GpuWeight down,
                    List<Shard> gateShards, List<Shard> upShards, List<Shard> gateUpShards, List<Shard> downShards,
                    GpuWeight gateScale, GpuWeight upScale, GpuWeight downScale, GpuWeight gateBias, GpuWeight upBias,
                    GpuWeight gateUpBias, GpuWeight downBias) {
            this.plan = plan;
            this.gate = gate;
            this.up = up;
            this.gateUp = gateUp;
            this.down = down;
            this.gateShards = gateShards;
            this.upShards = upShards;
            this.gateUpShards = gateUpShards;
            this.downShards = downShards;
            this.gateScale = gateScale;
            this.upScale = upScale;
            this.downScale = downScale;
            this.gateBias = gateBias;
            this.upBias = upBias;
            this.gateUpBias = gateUpBias;
            this.downBias = downBias;
        }

        boolean hasBiases() {
            return gateBias != null || upBias != null || gateUpBias != null || downBias != null;
        }
    }

    /**
     * Resolves and checks the device tensors of this set for {@code layer}.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalStateException         for a shape, type or scale that contradicts the spec
     * @throws UnsupportedOperationException for an expert tensor of a type without expert matvec kernels
     */
    Gpu resolve(GpuWeights w, LayerSpec layer) {
        GpuWeight gateUp = fused ? expertGpu(w.layer(layer, gateUpName()), 2 * nFf, inDim, layer) : null;
        GpuWeight gate = gated && !fused ? expertGpu(w.layer(layer, gateName()), nFf, inDim, layer) : null;
        GpuWeight up = fused ? null : expertGpu(w.layer(layer, upName()), nFf, inDim, layer);
        GpuWeight down = expertGpu(w.layer(layer, downName()), outDim, nFf, layer);
        GpuWeight gateScale = null;
        GpuWeight upScale = null;
        GpuWeight downScale = null;
        GpuWeight gateBias = null;
        GpuWeight upBias = null;
        GpuWeight gateUpBias = null;
        GpuWeight downBias = null;
        if (set.extras) {
            gateScale = gate != null ? vectorGpu(w.layerOrNull(layer, TensorNames.scale(set.gate)), 1, layer) : null;
            upScale = vectorGpu(w.layerOrNull(layer, TensorNames.scale(set.up)), 1, layer);
            downScale = vectorGpu(w.layerOrNull(layer, TensorNames.scale(set.down)), 1, layer);
            gateBias = gate != null ? vectorGpu(w.layerOrNull(layer, TensorNames.bias(set.gate)), nFf, layer) : null;
            upBias = up != null ? vectorGpu(w.layerOrNull(layer, TensorNames.bias(set.up)), nFf, layer) : null;
            gateUpBias = fused
                    ? vectorGpu(w.layerOrNull(layer, TensorNames.bias(TensorNames.FFN_GATE_UP_EXPS)), 2 * nFf, layer)
                    : null;
            downBias = vectorGpu(w.layerOrNull(layer, TensorNames.bias(set.down)), outDim, layer);
        }
        checkPresence(layer, gateScale != null || upScale != null || downScale != null,
                gateBias != null || upBias != null || gateUpBias != null || downBias != null);
        return new Gpu(this, gate, up, gateUp, down, expertShards(gate, nFf, layer), expertShards(up, nFf, layer),
                expertShards(gateUp, 2 * nFf, layer), expertShards(down, outDim, layer), gateScale, upScale, downScale,
                gateBias, upBias, gateUpBias, downBias);
    }

    /**
     * The expert shards of a checked expert tensor of {@code rowsPerExpert} rows per expert, in expert order, or null
     * for an absent tensor.
     *
     * @throws IllegalStateException when a shard does not hold whole experts
     */
    private List<Shard> expertShards(GpuWeight t, int rowsPerExpert, LayerSpec layer) {
        if (t == null) {
            return null;
        }
        List<Shard> out = new ArrayList<>(t.shardCount());
        for (int i = 0; i < t.shardCount(); i++) {
            long start = t.shardRowStart(i);
            long rows = t.shardRows(i);
            if (start % rowsPerExpert != 0 || rows % rowsPerExpert != 0) {
                throw new IllegalStateException(where(layer) + t.name() + " shard " + i + " holds rows [" + start
                        + ", " + (start + rows) + "), which split an expert of " + rowsPerExpert + " rows");
            }
            out.add(new Shard(t.shard(i), Math.toIntExact(start / rowsPerExpert),
                    Math.toIntExact(rows / rowsPerExpert)));
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * An expert tensor {@code [cols, rows, n_expert]} with expert kernels for its type, in one buffer or in shards of
     * whole experts.
     */
    private GpuWeight expertGpu(GpuWeight t, int rows, int cols, LayerSpec layer) {
        long[] ne = t.ne();
        if (ne.length != 3 || ne[0] != cols || ne[1] != rows || ne[2] != nExpert) {
            throw new IllegalStateException(where(layer) + t.name() + " has shape " + Arrays.toString(ne)
                    + ", expected [" + cols + ", " + rows + ", " + nExpert + "]");
        }
        if (!MoeKernels.supports(t.type())) {
            throw new UnsupportedOperationException(where(layer) + t.name() + " has type " + t.type()
                    + ", which has no expert matvec kernels");
        }
        if (t.hasScale() || t.hadamard() != null) {
            throw new IllegalStateException(where(layer) + t.name() + " carries a per-tensor scale or Hadamard"
                    + " transform; expert tensors are scaled per expert by their {n_expert} scale vectors");
        }
        return t;
    }

    /** An F32 per-expert vector of {@code width * n_expert} values in one buffer, or null when absent. */
    private GpuWeight vectorGpu(GpuWeight t, int width, LayerSpec layer) {
        if (t == null) {
            return null;
        }
        long n = t.rows() * t.cols();
        if (t.type() != GGMLType.F32 || !t.isSingleShard() || n != (long) width * nExpert) {
            throw new IllegalStateException(where(layer) + t.name() + " must be one F32 buffer of " + width + " x "
                    + nExpert + " values, got " + t);
        }
        return t;
    }

    // ------------------------------------------------------------------ CPU

    /** Host tensors of one expert set; null fields are absent tensors. Vectors are decoded F32 (shared, read only). */
    static final class Cpu {
        final MoeExpertWeights plan;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef gateUp;
        final CpuTensorRef down;
        /** {@code ffn_up_exps.scale} of the fused tensor, applied by the block; null otherwise. */
        final float[] fusedUpScale;
        final float[] gateBias;
        final float[] upBias;
        final float[] gateUpBias;
        final float[] downBias;

        private Cpu(MoeExpertWeights plan, CpuTensorRef gate, CpuTensorRef up, CpuTensorRef gateUp, CpuTensorRef down,
                    float[] fusedUpScale, float[] gateBias, float[] upBias, float[] gateUpBias, float[] downBias) {
            this.plan = plan;
            this.gate = gate;
            this.up = up;
            this.gateUp = gateUp;
            this.down = down;
            this.fusedUpScale = fusedUpScale;
            this.gateBias = gateBias;
            this.upBias = upBias;
            this.gateUpBias = gateUpBias;
            this.downBias = downBias;
        }
    }

    /**
     * Resolves and checks the host tensors of this set for {@code layer}. The per-expert scales of separate
     * tensors are applied by {@link com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec} through {@link CpuWeights};
     * the fused tensor's {@code ffn_up_exps.scale} belongs to another tensor name and is returned for the block.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalStateException for a shape that contradicts the spec
     */
    Cpu resolve(CpuWeights w, LayerSpec layer) {
        CpuTensorRef gateUp = fused ? expertCpu(w, w.layer(layer, gateUpName()), 2 * nFf, inDim, layer) : null;
        CpuTensorRef gate = gated && !fused ? expertCpu(w, w.layer(layer, gateName()), nFf, inDim, layer) : null;
        CpuTensorRef up = fused ? null : expertCpu(w, w.layer(layer, upName()), nFf, inDim, layer);
        CpuTensorRef down = expertCpu(w, w.layer(layer, downName()), outDim, nFf, layer);
        float[] fusedUpScale = null;
        float[] gateBias = null;
        float[] upBias = null;
        float[] gateUpBias = null;
        float[] downBias = null;
        boolean scales = false;
        if (set.extras) {
            float[] upScale = vectorCpu(w.layerFloatsOrNull(layer, TensorNames.scale(set.up)), 1, layer,
                    TensorNames.scale(set.up));
            fusedUpScale = fused ? upScale : null;
            if (fused && w.matVec().transformsOf(gateUp).hasScale()) {
                throw new IllegalStateException(where(layer) + gateUpName() + " has its own scale tensor; llama.cpp"
                        + " scales the fused experts by " + TensorNames.scale(set.up) + " only");
            }
            scales = upScale != null
                    || (gate != null && w.layerFloatsOrNull(layer, TensorNames.scale(set.gate)) != null)
                    || w.layerFloatsOrNull(layer, TensorNames.scale(set.down)) != null;
            gateBias = gate != null ? vectorCpu(w.layerFloatsOrNull(layer, TensorNames.bias(set.gate)), nFf, layer,
                    TensorNames.bias(set.gate)) : null;
            upBias = up != null ? vectorCpu(w.layerFloatsOrNull(layer, TensorNames.bias(set.up)), nFf, layer,
                    TensorNames.bias(set.up)) : null;
            gateUpBias = fused ? vectorCpu(w.layerFloatsOrNull(layer, TensorNames.bias(TensorNames.FFN_GATE_UP_EXPS)),
                    2 * nFf, layer, TensorNames.bias(TensorNames.FFN_GATE_UP_EXPS)) : null;
            downBias = vectorCpu(w.layerFloatsOrNull(layer, TensorNames.bias(set.down)), outDim, layer,
                    TensorNames.bias(set.down));
        }
        checkPresence(layer, scales, gateBias != null || upBias != null || gateUpBias != null || downBias != null);
        return new Cpu(this, gate, up, gateUp, down, fusedUpScale, gateBias, upBias, gateUpBias, downBias);
    }

    private CpuTensorRef expertCpu(CpuWeights w, CpuTensorRef t, int rows, int cols, LayerSpec layer) {
        TensorRef ref = w.describe(t);
        if (ref.rank() != 3 || ref.ne(0) != cols || ref.ne(1) != rows || ref.ne(2) != nExpert) {
            throw new IllegalStateException(where(layer) + ref.name() + " has shape "
                    + Arrays.toString(ref.ne()) + ", expected [" + cols + ", " + rows + ", " + nExpert + "]");
        }
        return t;
    }

    private float[] vectorCpu(float[] v, int width, LayerSpec layer, String suffix) {
        if (v != null && v.length != (long) width * nExpert) {
            throw new IllegalStateException(where(layer) + layer.tensorName(suffix) + " holds " + v.length
                    + " values, expected " + width + " x " + nExpert);
        }
        return v;
    }

    // ------------------------------------------------------------------ shared checks

    private void checkPresence(LayerSpec layer, boolean scales, boolean biases) {
        if (scales != expectScales) {
            throw new IllegalStateException(where(layer) + "expert scale tensors " + (scales ? "present" : "absent")
                    + " but the MoE spec says " + (expectScales ? "present" : "absent"));
        }
        if (biases != expectBiases) {
            throw new IllegalStateException(where(layer) + "expert bias tensors " + (biases ? "present" : "absent")
                    + " but the MoE spec says " + (expectBiases ? "present" : "absent"));
        }
    }

    private String where(LayerSpec layer) {
        return "layer " + layer.il() + " " + set.name().toLowerCase(Locale.ROOT) + " experts: ";
    }

    @Override
    public String toString() {
        return "MoeExpertWeights{" + set + ", n_expert=" + nExpert + ", n_ff=" + nFf + ", in=" + inDim + ", out="
                + outDim + (fused ? ", fused" : gated ? ", gated" : ", ungated") + ", glu=" + glu + ", act=" + act
                + '}';
    }
}
