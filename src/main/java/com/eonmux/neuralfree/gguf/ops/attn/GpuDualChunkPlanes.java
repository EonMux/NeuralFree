package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * The query-plane positions and the original-context scaling of a dual-chunk attention layer (Qwen2.5-1M) in
 * {@link GpuAttentionMixer}: the five query rotation planes rotate at their derived positions and the keys at
 * {@code position % chunk_length} ({@link DualChunkRope}), and the K rows and the three used query planes are scaled
 * by the factor of the sequence's length at this step (vLLM multiplies the new K rows before caching and the query
 * too). The positions are uploaded per step before the bound RoPE steps run. Not thread-safe; driven by the owning
 * mixer's step thread.
 */
final class GpuDualChunkPlanes {

    private final GpuAttentionMixer mixer;
    private GpuDualChunkAttention attention;
    /** The five query-plane position buffers, the key position buffer and the chunk positions the stages read. */
    private final List<GpuBuffer> planePos = new ArrayList<>(DualChunkRope.VARIANTS);
    private GpuBuffer keyPos;
    private GpuBuffer origPos;
    private int[] hostPlanes;
    private int[] hostKeys;

    GpuDualChunkPlanes(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /** The dual-chunk attention whose original positions the plane uploads read. */
    void attach(GpuDualChunkAttention attention) {
        this.attention = attention;
    }

    /** Allocates the position buffers of the five query rotations, the key rotation and the stage launches. */
    void bindBuffers() {
        if (mixer.kvLayer() == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + " runs dual-chunk attention without a KV"
                    + " cache");
        }
        hostPlanes = new int[DualChunkRope.VARIANTS * mixer.ubatch()];
        hostKeys = new int[mixer.ubatch()];
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            planePos.add(mixer.resources().allocate((long) mixer.ubatch() * Integer.BYTES));
        }
        keyPos = mixer.resources().allocate((long) mixer.ubatch() * Integer.BYTES);
        origPos = mixer.resources().allocate((long) mixer.ubatch() * Integer.BYTES);
    }

    /** Floats between consecutive query rotation planes ({@code [5][ubatch][qDim]} in the Q slot). */
    int planePitch() {
        return mixer.ubatch() * mixer.spec().qDim();
    }

    /** The position buffer of query rotation plane {@code plane}. */
    GpuBuffer planePos(int plane) {
        return planePos.get(plane);
    }

    /** The key rotation's position buffer. */
    GpuBuffer keyPos() {
        return keyPos;
    }

    /** The original positions of the step's rows, the stage launches' base positions. */
    GpuBuffer origPos() {
        return origPos;
    }

    /**
     * Uploads the step positions of the rotation planes and of the key rotation from the original positions
     * {@code pos0 + t} of a contiguous step ({@link DualChunkRope}).
     */
    void uploadPositions(int pos0, int n) {
        int[] host = attention.hostPositions();
        for (int t = 0; t < n; t++) {
            host[t] = pos0 + t;
        }
        uploadPlanes(n);
    }

    /** Uploads the step positions of a paged step from its segments' first positions. */
    void uploadPositions(PagedStep step, int n) {
        int[] host = attention.hostPositions();
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int start = step.segStart(seg);
            int pos0 = step.segPos0(seg);
            for (int t = 0; t < step.segLen(seg); t++) {
                host[start + t] = pos0 + t;
            }
        }
        uploadPlanes(n);
    }

    /** The five query-plane position arrays and the key positions of the first {@code n} host rows. */
    private void uploadPlanes(int n) {
        int[] host = attention.hostPositions();
        DualChunkRope.queryPositions(mixer.dualChunk(), host, 0, n, hostPlanes);
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            writeIntSlice(planePos.get(plane), hostPlanes, plane * n, n);
        }
        DualChunkRope.keyPositions(mixer.dualChunk(), host, 0, n, hostKeys);
        writeIntSlice(keyPos, hostKeys, 0, n);
    }

    /** The key positions of the first {@code n} rows of a contiguous step (the cache-write-only pass). */
    void uploadKeyPositions(int pos0, int n) {
        int[] host = attention.hostPositions();
        for (int t = 0; t < n; t++) {
            host[t] = pos0 + t;
        }
        uploadKeyPositions(n);
    }

    /** The key positions of a paged step's rows (the cache-write-only pass). */
    void uploadKeyPositions(PagedStep step, int n) {
        int[] host = attention.hostPositions();
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int start = step.segStart(seg);
            int pos0 = step.segPos0(seg);
            for (int t = 0; t < step.segLen(seg); t++) {
                host[start + t] = pos0 + t;
            }
        }
        uploadKeyPositions(n);
    }

    private void uploadKeyPositions(int n) {
        DualChunkRope.keyPositions(mixer.dualChunk(), attention.hostPositions(), 0, n, hostKeys);
        writeIntSlice(keyPos, hostKeys, 0, n);
    }

    /**
     * Applies the original-context logit scaling of a step (vLLM multiplies the new K rows before caching and the
     * query too): rows {@code [rowStart, rowStart + n)} of K and of the three used query planes, by the factor of
     * the sequence's length at this step.
     */
    void scale(int rowStart, int n, float factor) {
        if (factor == 1.0f) {
            return;
        }
        mixer.ops().scale(mixer.kBuf(), rowStart * mixer.spec().kDim(), n * mixer.spec().kDim(), factor);
        for (int plane = 0; plane < 3; plane++) {
            mixer.ops().scale(mixer.qBuf(), plane * planePitch() + rowStart * mixer.spec().qDim(),
                    n * mixer.spec().qDim(), factor);
        }
    }

    /** The K half of {@link #scale} for the cache-write-only pass. */
    void scaleK(int rowStart, int n, float factor) {
        if (factor != 1.0f) {
            mixer.ops().scale(mixer.kBuf(), rowStart * mixer.spec().kDim(), n * mixer.spec().kDim(), factor);
        }
    }

    /** {@link #scale} over a paged step, each segment by its sequence's slot count. */
    void scale(PagedStep step) {
        int[] slots = step.seqSlots(mixer.kvLayer().group());
        for (int seg = 0; seg < step.nSeg(); seg++) {
            scale(step.segStart(seg), step.segLen(seg),
                    mixer.dualChunk().scalingFactor(slots[step.segSeq(seg)]));
        }
    }

    /** {@link #scaleK} over a paged step, each segment by its sequence's slot count. */
    void scaleK(PagedStep step) {
        int[] slots = step.seqSlots(mixer.kvLayer().group());
        for (int seg = 0; seg < step.nSeg(); seg++) {
            scaleK(step.segStart(seg), step.segLen(seg),
                    mixer.dualChunk().scalingFactor(slots[step.segSeq(seg)]));
        }
    }

    /** Writes {@code count} ints of {@code src} from {@code off} into {@code dst}. */
    private void writeIntSlice(GpuBuffer dst, int[] src, int off, int count) {
        ByteBuffer bb = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int t = 0; t < count; t++) {
            bb.putInt(src[off + t]);
        }
        GpuEvent e = mixer.ctx().queue().enqueueWriteBuffer(dst, 0, (long) count * Integer.BYTES, bb.array());
        if (e != null) {
            e.close();
        }
    }
}
