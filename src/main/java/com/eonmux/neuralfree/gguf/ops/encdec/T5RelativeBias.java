package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * The T5 relative-position bias table of an attention layer, resolved for the flash kernels and
 * {@code CpuAttentionCore}. The bucket computation and the bias add run inside the kernels (bias modes
 * {@link BiasMode#T5_CAUSAL} = 2 and {@link BiasMode#T5_BIDIR} = 3); this class only resolves and converts the
 * tensor.
 *
 * <p>Source tensor: {@code <stack>.<t5BiasLayer>.attn_rel_b.weight} of shape {@code {n_head, n_rel_attn_bkts}}
 * (llama.cpp src/models/t5.cpp:67 encoder, :85 decoder), taken from the layer itself when it has one and from layer 0
 * of its stack otherwise (t5.cpp:146 decoder, :303 encoder; resolved into {@link AttentionSpec#t5BiasLayer()}).
 * llama.cpp adds {@code ggml_get_rows(attn_rel_b, bucket)} per (key, query) pair (llama-graph.cpp:2931-2944), so the
 * bias of head {@code h} for bucket {@code b} is element {@code b * n_head + h} of the table, the {@code rel_bias}
 * layout of the kernels ({@code kernels/llm/common/attn_common.h}). The kernels read F32; a table stored in another
 * type is converted once at bind.</p>
 */
public final class T5RelativeBias {

    /** Tensor suffix of the bias table in both the encoder and the decoder stack. */
    public static final String SUFFIX = TensorNames.weight(TensorNames.DEC_ATTN_REL_B);

    private T5RelativeBias() {
    }

    /** True when the attention adds the T5 relative-position bias. */
    public static boolean uses(AttentionSpec a) {
        Objects.requireNonNull(a, "attention");
        return a.biasMode() == BiasMode.T5_CAUSAL || a.biasMode() == BiasMode.T5_BIDIR;
    }

    /** Stored name of the bias table the attention of {@code layer} reads. */
    public static String tensorName(LayerSpec layer, AttentionSpec a) {
        return TensorNames.layerName(layer.stack(), a.t5BiasLayer(), SUFFIX);
    }

    /**
     * Uploads the bias table of a T5-biased attention as a device F32 buffer of {@code n_buckets * n_head} floats,
     * owned by the returned handle. The table's weight layer is made resident first
     * ({@link LayerWeightSource#prefetch}, {@link LayerWeightSource#await}); the copy is one embedding lookup of rows
     * {@code 0 .. n_buckets-1} enqueued on the context queue, so the buffer is valid for every later dispatch.
     *
     * @throws IllegalArgumentException when the attention has no T5 bias
     * @throws MissingTensorException   when the table is absent
     * @throws IllegalStateException    when its shape is not {@code {n_head, n_buckets}}
     */
    public static Device device(GpuForwardContext ctx, LayerSpec layer, AttentionSpec a) {
        Objects.requireNonNull(ctx, "ctx");
        checkUses(layer, a);
        int biasLayer = a.t5BiasLayer();
        LayerWeightSource source = ctx.weightSource();
        source.prefetch(biasLayer);
        source.await(biasLayer);
        GpuWeights weights = source.weightsFor(biasLayer);
        GpuWeight table = weights.stackOrNull(layer.stack(), biasLayer, SUFFIX);
        if (table == null) {
            throw new MissingTensorException(weights.archName(), biasLayer, tensorName(layer, a));
        }
        checkShape(table.name(), table.cols(), table.rows(), a);
        int nBuckets = a.t5Buckets();
        long floats = (long) nBuckets * a.nHead();
        GpuBuffer buffer = ctx.backend().allocateBuffer(floats * Float.BYTES);
        try {
            int[] rows = new int[nBuckets];
            for (int b = 0; b < nBuckets; b++) {
                rows[b] = b;
            }
            ctx.ops().embed(table, rows, buffer, 0);
        } catch (RuntimeException e) {
            buffer.close();
            throw e;
        }
        return new Device(table.name(), buffer, a);
    }

    /**
     * The bias table of a T5-biased attention as host F32 values {@code [n_buckets][n_head]}, the {@code relBias}
     * argument of {@code CpuAttentionCore.Params.of}.
     *
     * @throws IllegalArgumentException when the attention has no T5 bias
     * @throws MissingTensorException   when the table is absent
     * @throws IllegalStateException    when its shape is not {@code {n_head, n_buckets}}
     */
    public static float[] host(CpuWeights weights, LayerSpec layer, AttentionSpec a) {
        Objects.requireNonNull(weights, "weights");
        checkUses(layer, a);
        CpuTensorRef table = weights.stackOrNull(layer.stack(), a.t5BiasLayer(), SUFFIX);
        if (table == null) {
            throw new MissingTensorException(weights.archName(), a.t5BiasLayer(), tensorName(layer, a));
        }
        String name = tensorName(layer, a);
        checkShape(name, table.cols(), table.rows(), a);
        float[] values = new float[a.t5Buckets() * a.nHead()];
        table.decodeRows(0, a.t5Buckets(), values, 0);
        return values;
    }

    private static void checkUses(LayerSpec layer, AttentionSpec a) {
        Objects.requireNonNull(layer, "layer");
        if (!uses(a)) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " attention has bias mode "
                    + a.biasMode() + ", not a T5 relative-position bias");
        }
    }

    private static void checkShape(String name, long cols, long rows, AttentionSpec a) {
        if (cols != a.nHead() || rows != a.t5Buckets()) {
            throw new IllegalStateException("T5 bias table " + name + " is {" + cols + ", " + rows + "}, expected {n_head "
                    + a.nHead() + ", n_rel_attn_bkts " + a.t5Buckets() + "}");
        }
    }

    /**
     * A device copy of a bias table with the flash-kernel arguments that go with it ({@code bias_mode},
     * {@code n_buckets}, {@code max_distance}). Closing releases the buffer; dispatches enqueued before must have
     * completed (the owner closes it after the queue drained, as every bind-time resource).
     */
    public static final class Device implements AutoCloseable {
        private final String name;
        private final GpuBuffer buffer;
        private final int biasMode;
        private final int nBuckets;
        private final int maxDistance;
        private boolean closed;

        private Device(String name, GpuBuffer buffer, AttentionSpec a) {
            this.name = name;
            this.buffer = buffer;
            this.biasMode = a.biasMode().code();
            this.nBuckets = a.t5Buckets();
            this.maxDistance = a.t5MaxDistance();
        }

        /** The {@code rel_bias} buffer: {@code n_buckets * n_head} F32 values from element 0. */
        public GpuBuffer buffer() {
            if (closed) {
                throw new IllegalStateException("T5 bias table " + name + " is closed");
            }
            return buffer;
        }

        /** {@code bias_mode} argument: 2 (causal decoder) or 3 (bidirectional encoder). */
        public int biasMode() { return biasMode; }

        /** {@code n_buckets} argument ({@code n_rel_attn_bkts}). */
        public int nBuckets() { return nBuckets; }

        /** {@code max_distance} argument (128 in llama.cpp, llama-graph.cpp {@code llama_relative_position_bucket}). */
        public int maxDistance() { return maxDistance; }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                buffer.close();
            }
        }

        @Override
        public String toString() {
            return "T5RelativeBias.Device{" + name + ", mode=" + biasMode + ", buckets=" + nBuckets + ", maxDistance="
                    + maxDistance + (closed ? ", closed" : "") + '}';
        }
    }
}
