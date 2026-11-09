package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * The static sink tokens of {@link GpuAttentionMixer}: the learned per-head {@code attn_sinks} logits of the flash
 * kernels, and the normalized learned sink keys and values of the openPangu sink attention (vLLM
 * {@code OpenPanguSinkAttention.post_weight_load}) — the {@code [n_sink * n_head_kv, head_dim]} keys normalized per
 * KV head by the layer's K norm and left un-rotated, the matching values (zeros when the checkpoint omits
 * {@code attn_sink_v}). Host copies serve a host attention; the device buffers serve flash attention. Not
 * thread-safe.
 */
final class GpuStaticSinks {

    private final GpuAttentionMixer mixer;

    /** Per-head attention sinks in host memory for a host attention, or null. */
    private float[] hostSinks;
    /** Normalized learned static sink keys and values of the openPangu sink attention, or null. */
    private float[] hostSinkK;
    private float[] hostSinkV;
    /** The device copies of the static sink keys and values; null without sink tokens. */
    private GpuBuffer sinkKBuf;
    private GpuBuffer sinkVBuf;

    GpuStaticSinks(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /**
     * Loads the learned static sink tokens of the openPangu sink attention: the sink keys normalized per KV head by
     * the layer's K norm and their values, as host arrays and, on a device cache, as device buffers.
     *
     * @throws IllegalStateException    when the sink keys are missing
     * @throws IllegalArgumentException when a sink tensor or the K norm weight has the wrong size
     */
    void bind() {
        int n = mixer.spec().sinkTokens();
        if (n == 0) {
            return;
        }
        int heads = mixer.spec().nHeadKv();
        int dK = mixer.spec().headDimK();
        int dV = mixer.spec().headDimV();
        GpuWeight keyWeight = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_SINK_K));
        if (keyWeight == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + ": tensor "
                    + TensorNames.weight(TensorNames.ATTN_SINK_K) + " is missing");
        }
        float[] rawKeys = hostVector(keyWeight, (long) n * heads * dK);
        GpuWeight normWeight = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_K_NORM));
        float[] weight = normWeight != null ? hostVector(normWeight, dK) : null;
        hostSinkK = new float[n * heads * dK];
        NormMath.rowNorm(NormKind.RMS, mixer.rmsEps(), n * heads, dK, dK, 1, rawKeys, 0, weight, null, hostSinkK, 0);
        GpuWeight valueWeight = mixer.weights().layerOrNull(mixer.layer(),
                TensorNames.weight(TensorNames.ATTN_SINK_V));
        hostSinkV = valueWeight != null
                ? hostVector(valueWeight, (long) n * heads * dV) : new float[n * heads * dV];
        if (!mixer.hostKv()) {
            sinkKBuf = floatBuffer(hostSinkK);
            sinkVBuf = floatBuffer(hostSinkV);
        }
    }

    /** Loads the per-head attention sinks into host memory for a host attention ({@link #hostSinkLogits}). */
    void bindHostLogits() {
        hostSinks = hostLogits();
    }

    /** The device copy of the normalized static sink keys, or null without sink tokens or on a host cache. */
    GpuBuffer sinkKeys() {
        return sinkKBuf;
    }

    /** The device copy of the static sink values, or null without sink tokens or on a host cache. */
    GpuBuffer sinkValues() {
        return sinkVBuf;
    }

    /** The per-head attention sinks of a host-KV layer, or null without sinks. */
    float[] hostSinkLogits() {
        return hostSinks;
    }

    /** The normalized static sink keys of a host-KV layer {@code [n_sink][n_head_kv][d_k]}, or null. */
    float[] hostKeys() {
        return hostSinkK;
    }

    /** The static sink values of a host-KV layer {@code [n_sink][n_head_kv][d_v]}, or null. */
    float[] hostValues() {
        return hostSinkV;
    }

    /** {@code attn_sinks} as F32 ({@code n_head} logits), or null without sinks. */
    GpuBuffer logits() {
        if (!mixer.spec().sinks()) {
            return null;
        }
        return mixer.vector(TensorNames.weight(TensorNames.ATTN_SINKS), mixer.spec().nHead());
    }

    /** {@code attn_sinks} in host memory for a host attention, or null without sinks. */
    float[] hostLogits() {
        if (!mixer.spec().sinks()) {
            return null;
        }
        GpuWeight w = mixer.weights().layer(mixer.layer(), TensorNames.weight(TensorNames.ATTN_SINKS));
        int n = mixer.spec().nHead();
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() != n) {
            throw new UnsupportedOperationException("layer " + mixer.layer().il() + ": " + w.name()
                    + " must be one F32 buffer of " + n + " values for a host attention, got " + w);
        }
        float[] out = new float[n];
        GpuEvent read = mixer.ctx().queue().enqueueReadBuffer(w.shard(0), w.shardRowStart(0) * w.rowBytes(),
                (long) n * Float.BYTES, out);
        if (read != null) {
            try {
                read.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while reading " + w.name()
                        + " for the host attention", e);
            } finally {
                read.close();
            }
        }
        return out;
    }

    /** The F32 values of a weight as a host array of {@code n} values, after the read has completed. */
    private float[] hostVector(GpuWeight w, long n) {
        GpuBuffer b = mixer.vector(w, n);
        float[] out = new float[(int) n];
        GpuEvent read = mixer.ctx().queue().enqueueReadBuffer(b, 0, n * Float.BYTES, out);
        if (read != null) {
            try {
                read.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while reading " + w.name()
                        + " for the static sink tokens", e);
            } finally {
                read.close();
            }
        }
        return out;
    }

    /** An F32 device buffer of {@code values}, owned by this mixer's resources. */
    private GpuBuffer floatBuffer(float[] values) {
        GpuBuffer b = mixer.resources().allocate((long) values.length * Float.BYTES);
        GpuEvent e = mixer.ctx().queue().enqueueWriteBuffer(b, 0, (long) values.length * Float.BYTES, values);
        if (e != null) {
            e.close();
        }
        return b;
    }
}
