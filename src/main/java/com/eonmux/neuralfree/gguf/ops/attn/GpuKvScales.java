package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;

import java.util.Arrays;

/**
 * The FP8 KV-cache scales of {@link GpuAttentionMixer}: the per-head K and V scale buffers the FP8 store kernels
 * divide by, the combined {@code kv_scales} buffer (K values then V values) the flash kernels multiply the
 * dequantized rows by, and the host copies a host attention stores and decodes with. No-op unless a cache dtype is
 * checkpoint-scaled (an FP8 type or NVFP4): vLLM consumes the checkpoint scales of an FP8 cache only and enforces 1.0
 * for every other (vllm kv_cache.py process_weights_after_loading); NVFP4 follows the same rule through its store
 * kernel's global scale (vllm csrc/nvfp4_kv_cache_kernels.cu). Not thread-safe.
 */
final class GpuKvScales {

    private final GpuAttentionMixer mixer;

    /** The combined scale buffer the flash kernels read (K values then V values), null without an FP8 cache. */
    private GpuBuffer kvScales;
    private GpuBuffer kScales;
    private GpuBuffer vScales;
    /** FP8 cache scales in host memory for a host attention, or null. */
    private float[] hostKScale;
    private float[] hostVScale;

    GpuKvScales(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /** Binds the device scales of the layer's cache {@code kv}, or the host scales of a host-KV layer. */
    void bind(GpuKvLayer kv, boolean hostKv) {
        if (kv != null) {
            bindDevice(kv);
        } else if (hostKv) {
            bindHost();
        }
    }

    /**
     * Loads the FP8 KV-cache scales of the layer's cache: the per-head K and V scale buffers the FP8 store kernels
     * divide by and the combined {@code kv_scales} buffer the flash kernels multiply the dequantized rows by. The
     * checkpoint's {@code vendor_side} buffer of the cache is filled with the same values when it exists.
     */
    private void bindDevice(GpuKvLayer kv) {
        KvLayout layout = kv.layout();
        float[] k = values(TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE, layout.dtypeK());
        float[] v = values(TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE, layout.dtypeV());
        if (k == null && v == null) {
            return;
        }
        int heads = mixer.spec().nHeadKv();
        float[] both = new float[2 * heads];
        for (int h = 0; h < heads; h++) {
            both[h] = k != null ? k[h] : 1.0f;
            both[heads + h] = v != null ? v[h] : 1.0f;
        }
        kvScales = floatBuffer(both);
        if (kv.vendorSide() != null) {
            GpuEvent e = mixer.ctx().queue().enqueueWriteBuffer(kv.vendorSide(), 0, (long) both.length * Float.BYTES,
                    both);
            if (e != null) {
                e.close();
            }
        }
        kScales = k != null ? floatBuffer(k) : null;
        vScales = v != null ? floatBuffer(v) : null;
    }

    /**
     * The FP8 cache scales of a host-KV layer's planned cache: the host attention stores and decodes with the same
     * per-head values the device kernels would.
     */
    private void bindHost() {
        KvLayout layout = mixer.ctx().memory().plan().layer(mixer.layer().il()).kv();
        if (layout == null) {
            // a shared-cache reader stores no rows and its scale tensors are absent, as on the device path: ones
            return;
        }
        hostKScale = values(TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE, layout.dtypeK());
        hostVScale = values(TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE, layout.dtypeV());
    }

    /** The combined {@code kv_scales} buffer of the flash kernels, or null without an FP8 cache. */
    GpuBuffer device() {
        return kvScales;
    }

    /** The per-head K scale buffer the store kernel divides by, or null for an unscaled cache. */
    GpuBuffer kDevice() {
        return kScales;
    }

    /** The per-head V scale buffer the store kernel divides by, or null for an unscaled cache. */
    GpuBuffer vDevice() {
        return vScales;
    }

    /** The per-KV-head FP8 K cache scales of a host-KV layer, or null for an unscaled cache. */
    float[] hostK() {
        return hostKScale;
    }

    /** The per-KV-head FP8 V cache scales of a host-KV layer, or null for an unscaled cache. */
    float[] hostV() {
        return hostVScale;
    }

    /**
     * The checkpoint KV-cache scale of {@code suffix} as one value per KV head (a {@code [1]} scale repeated), or
     * null when {@code type} is not an FP8 cache or the checkpoint carries none (the store and attention then scale
     * by 1.0, as vLLM does for an FP8 cache without {@code k_scale}/{@code v_scale}).
     *
     * @throws IllegalArgumentException when the tensor is not a plain F32 vector of 1 or {@code n_head_kv} values or
     *                                 holds a non-finite or non-positive scale
     */
    private float[] values(String suffix, KvDType type) {
        if (!type.checkpointScaled()) {
            return null;
        }
        GpuWeight w = mixer.weights().layerOrNull(mixer.layer(), suffix);
        int heads = mixer.spec().nHeadKv();
        if (w == null) {
            float[] ones = new float[heads];
            Arrays.fill(ones, 1.0f);
            return ones;
        }
        long total = w.rows() * w.cols();
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.hasScale() || w.inverseHadamard() != null
                || (total != 1 && total != heads)) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": " + w.name()
                    + " must be a plain F32 tensor of 1 or " + heads + " values (one KV-cache scale), holds " + total
                    + " values as " + w.type());
        }
        float[] v = new float[(int) total];
        GpuEvent e = mixer.ctx().queue().enqueueReadBuffer(w.shard(0), 0, total * Float.BYTES, v);
        if (e != null) {
            e.close();
        }
        for (float value : v) {
            if (!Float.isFinite(value) || value <= 0f) {
                throw new IllegalArgumentException("layer " + mixer.layer().il() + ": " + w.name() + " holds " + value
                        + "; KV-cache scales must be finite and positive");
            }
        }
        if (total == heads) {
            return v;
        }
        float[] out = new float[heads];
        Arrays.fill(out, v[0]);
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
