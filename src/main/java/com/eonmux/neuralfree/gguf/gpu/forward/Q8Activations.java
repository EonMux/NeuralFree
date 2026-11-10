package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * The Q8_1 activation blocks of the integer-dot matvec lane ({@value GpuOps#QUANTIZE_Q8_1}'s layout: one 36-byte
 * block per 32 elements, row {@code r} at byte {@code r * width / 32 * 36}) and where they come from.
 *
 * <p><b>Rows a norm already quantized.</b> On a backend whose kernels reduce over 32-lane warps in registers
 * ({@link com.eonmux.neuralfree.gguf.kernels.LlmKernels#warpReductions}: the subgroup shuffle of CUDA and HIP, the
 * Vulkan subgroup modules on a fixed 32-lane subgroup), a row norm whose
 * output is the input of the following projections writes the Q8_1 blocks of its rows in the same launch
 * ({@code LLM_RowNormQ8}, {@code LLM_ResidualMixNormQ8}, through {@link GpuOps#norm} and
 * {@link GpuOps#residualMixNorm}) into the shared Q8_1 staging slot and records them here. The next Q8_1 request
 * for exactly those rows ({@link #take}: a {@link GpuOps.SharedQ8} scope opening over them, a projection's own
 * quantization, a multi-row group of the counted lane) reads the recorded blocks instead of launching the
 * quantizer, which the norm kernel's segment reductions make bit for bit the quantizer's output. The record is
 * valid only until the next queue operation of its {@link GpuOps}: every kernel launch path
 * ({@link GpuOps#enqueue}, its internal launch helpers, {@link PreparedDispatch}, the subgroup-row matvecs and every
 * launch of the fast GEMM lanes, their activation staging included) and its fills and writes drop it first
 * ({@link #invalidate}), so a write to the normed rows or to the shared slot between the norm and the request can
 * never be read as stale blocks, and the first request takes or drops it. A norm does not emit while a
 * {@link GpuOps.SharedQ8} scope is open, as the scope's blocks occupy the slot. The int8 MMQ GEMM reads its own
 * record layout, never these blocks: a scope over the rows keeps the blocks it took and stages the GEMM's records
 * separately ({@link GpuOps.SharedQ8#records}).</p>
 *
 * <p><b>Only blocks that are read.</b> Whether the rows' consumer reads Q8_1 blocks depends on its weight (the
 * int8 lane serves a projection only where it measured faster, {@code GpuProjections}), which the norm does not
 * see. A norm site whose recorded blocks were dropped unread therefore stops emitting for the rest of this
 * {@link GpuOps}' life and runs the plain kernel: the blocks would only cost the norm its quantization. A site is
 * the norm's output buffer, its weight buffer and its row count and width, so one norm's unread blocks (a sandwich
 * layer's post-attention norm on the branch, an output norm before a head off the int8 lane, an FFN norm read
 * first by an F32 MoE router) never stop another norm of the same shape whose consumer does read them. A
 * {@link GpuOps.SharedQ8} scope that opens over the rows takes the record for its projections ({@link #claim}); when
 * none of them reads the blocks (each runs off the int8 lane), the scope reports the site unread as it closes
 * ({@link #unused}). Taking or dropping a record never changes a result, as the blocks equal the quantizer's bit for
 * bit.</p>
 *
 * <p><b>The quantizer launch.</b> {@value GpuOps#QUANTIZE_Q8_1} quantizes one block per 32-lane subgroup on the
 * shuffle backends and one block per work-item elsewhere; {@link #quantizeWorkItems} sizes its grid.</p>
 *
 * <p>Owned by one {@link GpuOps}; not thread-safe.</p>
 */
final class Q8Activations {

    /** Largest row group a norm writes the Q8_1 blocks of: the counted multi-column lane's widest group. */
    static final int EMIT_MAX_ROWS = LlmKernelConstants.MATVEC_N_COUNTED_MAX;

    private final GpuScratch scratch;
    private final boolean shuffle;
    private final boolean warpNorms;
    private final boolean lanePossible;
    /** The norm sites whose blocks were dropped unread. */
    private final Set<Site> unread = new HashSet<>();
    private GpuBuffer rows;
    private int nRows;
    private int width;
    private GpuBuffer blocks;
    /** The site of the recorded blocks. */
    private Site site;

    /**
     * @param warpNorms whether the row norms reduce their Q8_1 segments over 32-lane warps in registers, the
     *                  condition for a norm to write the blocks
     */
    Q8Activations(GpuScratch scratch, DeviceCapabilities caps, Int8MatvecMode int8Mode, boolean warpNorms) {
        this.scratch = Objects.requireNonNull(scratch, "scratch");
        Objects.requireNonNull(caps, "caps");
        Objects.requireNonNull(int8Mode, "int8Mode");
        this.shuffle = caps.hasFeature(DeviceFeature.SUBGROUP_SHUFFLE);
        this.warpNorms = warpNorms;
        this.lanePossible = int8Mode.forcesLane()
                || (int8Mode.permitsLane() && caps.hasFeature(DeviceFeature.INTEGER_DOT_PRODUCT));
    }

    /**
     * Whether a row norm of {@code nRows} rows of {@code width} written at element {@code outOff} of {@code out}, with
     * {@code wRows} rows of the weight {@code weight} (null without one), writes their Q8_1 blocks: the device runs
     * the norm's segment reductions in registers and may run the integer-dot lane, the rows start at element 0 (where
     * the lane's quantization reads them), the width is a whole number of blocks, one weight row serves every row (a
     * hidden-state norm, not a per-head one), the group is no wider than the counted lane serves and no earlier
     * blocks of this norm site were dropped unread. The caller adds the kernel's availability and the fusion and
     * batch-invariance modes.
     */
    boolean emits(GpuBuffer out, GpuBuffer weight, int nRows, int width, int outOff, int wRows) {
        return warpNorms && lanePossible && outOff == 0 && wRows == 1 && width % GpuOps.Q8_1_BLOCK_SIZE == 0
                && nRows >= 1 && nRows <= EMIT_MAX_ROWS && !unread.contains(new Site(out, weight, nRows, width));
    }

    /** The shared Q8_1 staging slot sized for {@code nRows} rows of {@code width}, where a norm writes the blocks. */
    GpuBuffer target(int nRows, int width) {
        long floats = (long) nRows * (width / GpuOps.Q8_1_BLOCK_SIZE) * GpuOps.Q8_1_FLOATS;
        return scratch.staging(GpuScratch.Staging.SHARED_Q8_1, floats).buffer();
    }

    /**
     * Records that the norm launch just enqueued, weighted by {@code weight} (null without one), wrote the Q8_1 blocks
     * of {@code rows[0 .. nRows * width)} to {@code blocks}.
     */
    void emitted(GpuBuffer rows, GpuBuffer weight, int nRows, int width, GpuBuffer blocks) {
        this.rows = Objects.requireNonNull(rows, "rows");
        this.nRows = nRows;
        this.width = width;
        this.blocks = Objects.requireNonNull(blocks, "blocks");
        this.site = new Site(rows, weight, nRows, width);
    }

    /**
     * The recorded blocks when they are those of {@code x[0 .. nRows * width)}, else null; either way the record is
     * dropped, so the blocks serve the first request after the norm only.
     */
    GpuBuffer take(GpuBuffer x, int nRows, int width) {
        if (blocks != null && x == rows && nRows == this.nRows && width == this.width) {
            GpuBuffer hit = blocks;
            rows = null;
            blocks = null;
            site = null;
            return hit;
        }
        invalidate();
        return null;
    }

    /**
     * {@link #take} for a {@link GpuOps.SharedQ8} scope, which holds the blocks for projections that may not read them:
     * the taken blocks with their norm's site, which {@link #unused} reports unread, else null.
     */
    Claim claim(GpuBuffer x, int nRows, int width) {
        Site taken = site;
        GpuBuffer hit = take(x, nRows, width);
        return hit != null ? new Claim(hit, taken) : null;
    }

    /** Notes the site of claimed blocks that no projection read as unread, so its norm stops emitting. */
    void unused(Claim claim) {
        unread.add(claim.site);
    }

    /** Drops the record, noting its site as unread; every kernel dispatch calls this before it enqueues. */
    void invalidate() {
        if (blocks != null) {
            unread.add(site);
        }
        rows = null;
        blocks = null;
        site = null;
    }

    /** Work-items of a {@value GpuOps#QUANTIZE_Q8_1} launch over {@code blocks} blocks. */
    long quantizeWorkItems(long blocks) {
        return shuffle ? blocks * GpuOps.Q8_1_BLOCK_SIZE : blocks;
    }

    /** Blocks a {@link GpuOps.SharedQ8} scope took from a norm, with the norm's site. Immutable. */
    static final class Claim {
        final GpuBuffer blocks;
        private final Site site;

        private Claim(GpuBuffer blocks, Site site) {
            this.blocks = blocks;
            this.site = site;
        }
    }

    /** One norm site: its output and weight buffers by identity, and its row count and width. Immutable. */
    private static final class Site {
        private final GpuBuffer out;
        private final GpuBuffer weight;
        private final int nRows;
        private final int width;

        Site(GpuBuffer out, GpuBuffer weight, int nRows, int width) {
            this.out = out;
            this.weight = weight;
            this.nRows = nRows;
            this.width = width;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Site)) {
                return false;
            }
            Site other = (Site) o;
            return out == other.out && weight == other.weight && nRows == other.nRows && width == other.width;
        }

        @Override
        public int hashCode() {
            int h = System.identityHashCode(out);
            h = 31 * h + System.identityHashCode(weight);
            h = 31 * h + nRows;
            return 31 * h + width;
        }
    }
}
