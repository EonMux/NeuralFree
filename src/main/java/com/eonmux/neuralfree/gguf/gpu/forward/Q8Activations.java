package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Objects;

/**
 * The Q8_1 activation blocks of the integer-dot matvec lane ({@value GpuOps#QUANTIZE_Q8_1}'s layout: one 36-byte
 * block per 32 elements, row {@code r} at byte {@code r * width / 32 * 36}) and where they come from.
 *
 * <p><b>Rows a norm already quantized.</b> On a backend with the subgroup shuffle (CUDA, HIP), a row norm whose
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
    private final boolean lanePossible;
    private GpuBuffer rows;
    private int nRows;
    private int width;
    private GpuBuffer blocks;

    Q8Activations(GpuScratch scratch, DeviceCapabilities caps, Int8MatvecMode int8Mode) {
        this.scratch = Objects.requireNonNull(scratch, "scratch");
        Objects.requireNonNull(caps, "caps");
        Objects.requireNonNull(int8Mode, "int8Mode");
        this.shuffle = caps.hasFeature(DeviceFeature.SUBGROUP_SHUFFLE);
        this.lanePossible = int8Mode.forcesLane()
                || (int8Mode.permitsLane() && caps.hasFeature(DeviceFeature.INTEGER_DOT_PRODUCT));
    }

    /**
     * Whether a row norm of {@code nRows} rows of {@code width} written at element {@code outOff} of its buffer, with
     * {@code wRows} weight rows, writes their Q8_1 blocks: the device runs the norm's segment reductions in registers
     * and may run the integer-dot lane, the rows start at element 0 (where the lane's quantization reads them), the
     * width is a whole number of blocks, one weight row serves every row (a hidden-state norm, not a per-head one)
     * and the group is no wider than the counted lane serves. The caller adds the kernel's availability and the
     * fusion and batch-invariance modes.
     */
    boolean emits(int nRows, int width, int outOff, int wRows) {
        return shuffle && lanePossible && outOff == 0 && wRows == 1 && width % GpuOps.Q8_1_BLOCK_SIZE == 0
                && nRows >= 1 && nRows <= EMIT_MAX_ROWS;
    }

    /** The shared Q8_1 staging slot sized for {@code nRows} rows of {@code width}, where a norm writes the blocks. */
    GpuBuffer target(int nRows, int width) {
        long floats = (long) nRows * (width / GpuOps.Q8_1_BLOCK_SIZE) * GpuOps.Q8_1_FLOATS;
        return scratch.staging(GpuScratch.Staging.SHARED_Q8_1, floats).buffer();
    }

    /** Records that the launch just enqueued wrote the Q8_1 blocks of {@code rows[0 .. nRows * width)} to blocks. */
    void emitted(GpuBuffer rows, int nRows, int width, GpuBuffer blocks) {
        this.rows = Objects.requireNonNull(rows, "rows");
        this.nRows = nRows;
        this.width = width;
        this.blocks = Objects.requireNonNull(blocks, "blocks");
    }

    /**
     * The recorded blocks when they are those of {@code x[0 .. nRows * width)}, else null; either way the record is
     * dropped, so the blocks serve the first request after the norm only.
     */
    GpuBuffer take(GpuBuffer x, int nRows, int width) {
        GpuBuffer hit = blocks != null && x == rows && nRows == this.nRows && width == this.width ? blocks : null;
        invalidate();
        return hit;
    }

    /** Drops the record; every kernel dispatch calls this before it enqueues. */
    void invalidate() {
        rows = null;
        blocks = null;
    }

    /** Work-items of a {@value GpuOps#QUANTIZE_Q8_1} launch over {@code blocks} blocks. */
    long quantizeWorkItems(long blocks) {
        return shuffle ? blocks * GpuOps.Q8_1_BLOCK_SIZE : blocks;
    }
}
