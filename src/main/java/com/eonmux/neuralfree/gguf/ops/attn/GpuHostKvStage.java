package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.util.Objects;

/**
 * The host-KV halves of {@link GpuAttentionMixer} ({@code -nkvo}): {@code projectKvForHostAttention} runs the cache
 * writes alone and copies the K/V rows to the host cache, {@code runWithHostAttention} runs the projections, norms
 * and RoPE, hands the Q/K/V rows to the caller's host attention and finishes the step over the returned attention
 * rows. The device flash attention, KV store and cache rotation do not run; the host stage applies the rotation and
 * store. Not thread-safe; driven by the owning mixer's step thread.
 */
final class GpuHostKvStage {

    private final GpuAttentionMixer mixer;

    GpuHostKvStage(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /**
     * The K and V rows of a step on the device, for the host cache to store: the K/V projections, their biases,
     * clamps, norms and RoPE. The Q projection, the attention, the cache store and the output projection do not run.
     * A shared-cache layer projects none and leaves the arrays untouched.
     *
     * @param kOut {@code nTokens * kDim} values, or null for a layer that stores no cache
     * @param vOut {@code nTokens * vDim} values, or null for a layer that stores no cache
     */
    void projectKvForHostAttention(int pos0, int nTokens, ScratchSlot in, float[] kOut, float[] vOut) {
        requireHostKv();
        Objects.requireNonNull(in, "in");
        checkStep(pos0, nTokens);
        if (!mixer.proj().projectsKv()) {
            return;
        }
        requireLength(kOut, (long) nTokens * mixer.spec().kDim(), "K rows");
        requireLength(vOut, (long) nTokens * mixer.spec().vDim(), "V rows");
        mixer.checkStepInputs(pos0, nTokens);
        mixer.projections().projectKv(nTokens, in);
        mixer.adapters().run(nTokens, in);
        GpuAttentionMixer.runAll(mixer.kvInputSteps(), nTokens,
                mixer.stepInputs() != null ? mixer.stepInputs().currentPagedStep() : null);
        readInto(mixer.kBuf(), kOut, (long) nTokens * mixer.spec().kDim());
        readInto(mixer.vBuf(), vOut, (long) nTokens * mixer.spec().vDim());
        mixer.ctx().queue().finish();
    }

    /**
     * One step whose attention runs on the host over the host KV cache: the device projections, norms and RoPE, the
     * caller's host attention over the copied Q/K/V rows ({@code q} is always projected; {@code k} and {@code v} are
     * null for a shared-cache layer), then the device output norm, gate, sub-norm, output projection and value scale
     * over the returned attention rows. The device cache store and rotation do not run; the host stage applies the
     * cache rotation and store.
     *
     * @param q    {@code nTokens * qDim} values
     * @param k    {@code nTokens * kDim} values, or null for a layer that stores no cache
     * @param v    {@code nTokens * vDim} values, or null for a layer that stores no cache
     * @param attn {@code nTokens * outDim} values the host attention writes
     */
    void runWithHostAttention(int pos0, int nTokens, ScratchSlot in, ScratchSlot out, float[] q, float[] k, float[] v,
                              float[] attn, GpuAttentionMixer.HostAttention host) {
        requireHostKv();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(host, "host");
        checkStep(pos0, nTokens);
        requireLength(q, (long) nTokens * mixer.spec().qDim(), "Q rows");
        requireLength(attn, (long) nTokens * mixer.spec().outDim(), "attention rows");
        if (mixer.proj().projectsKv()) {
            requireLength(k, (long) nTokens * mixer.spec().kDim(), "K rows");
            requireLength(v, (long) nTokens * mixer.spec().vDim(), "V rows");
        } else if (k != null || v != null) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + " stores no K/V rows of its own; pass"
                    + " null rows");
        }
        mixer.projections().project(nTokens, in);
        mixer.adapters().run(nTokens, in);
        GpuAttentionMixer.runAll(mixer.inputSteps(), nTokens);
        readInto(mixer.qBuf(), q, (long) nTokens * mixer.spec().qDim());
        if (mixer.proj().projectsKv()) {
            readInto(mixer.kBuf(), k, (long) nTokens * mixer.spec().kDim());
            readInto(mixer.vBuf(), v, (long) nTokens * mixer.spec().vDim());
        }
        mixer.ctx().queue().finish();
        host.attend(pos0, nTokens, q, k, v, attn);
        writeFrom(mixer.attnBuf(), attn, (long) nTokens * mixer.spec().outDim());
        GpuAttentionMixer.runAll(mixer.outputSteps(), nTokens);
        mixer.output().projectOutput(nTokens, out);
    }

    private void requireHostKv() {
        String refusal = mixer.hostAttentionRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
    }

    private void requireLength(float[] rows, long expected, String what) {
        if (rows == null || rows.length < expected) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": " + what + " need " + expected
                    + " floats, got " + (rows == null ? "null" : rows.length));
        }
    }

    private void checkStep(int pos0, int nTokens) {
        if (nTokens < 1 || nTokens > mixer.ubatch()) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": step of " + nTokens
                    + " tokens outside [1, " + mixer.ubatch() + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": negative position " + pos0);
        }
    }

    /** Reads {@code count} floats from the start of a dense device buffer into the host rows. */
    private void readInto(GpuBuffer src, float[] dst, long count) {
        GpuEvent event = mixer.ctx().queue().enqueueReadBuffer(src, 0L, count * Float.BYTES, dst);
        if (event != null) {
            event.close();
        }
    }

    /** Writes host rows to the start of a dense device buffer. */
    private void writeFrom(GpuBuffer dst, float[] src, long count) {
        GpuEvent event = mixer.ctx().queue().enqueueWriteBuffer(dst, 0L, count * Float.BYTES, src);
        if (event != null) {
            event.close();
        }
    }
}
