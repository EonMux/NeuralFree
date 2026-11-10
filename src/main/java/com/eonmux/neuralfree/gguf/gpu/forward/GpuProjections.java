package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.StreamKGemmPlan;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * Dense projections of one {@link GpuOps}: the weight-family matvec lanes ({@code {T}_MatVec},
 * {@code {T}_MatVec2}, {@code {T}_MatVecAdd}, {@code {T}_BatchMatMul[_2D]}), the tiled GEMM, the fast GEMM lanes
 * of {@link FastGemmLanes}, the fused gate+up form and the integer-dot (Q8_1 activation) lane; the projection
 * input staging of the weight's input transform, and the W12 row kernels over a weight's codes or its BF16
 * mirror. The public entries are {@link GpuOps#matvec}, {@link GpuOps#matmul}, {@link GpuOps#matvecAdd},
 * {@link GpuOps#gateUpGlu}, {@link GpuOps#matvecFamilyServes} and {@link GpuOps#prewarmWeightKernels}.
 * Not thread-safe; owned by its GpuOps.
 */
final class GpuProjections {

    /** Work-group size of the tiled GEMM ({@code NF_GEMM_WG}). */
    static final int TILED_WG = 256;
    /** Weight rows per tiled GEMM work-group ({@code NF_GEMM_TM}). */
    static final int TILED_ROWS = 64;
    /** Tokens per tiled GEMM work-group ({@code NF_GEMM_TN}). */
    static final int TILED_TOKENS = 32;
    /** Largest cooperating-thread count of any matvec family ({@code *_TPS}); the matvec local size is a multiple. */
    static final int MATVEC_MAX_TPS = 32;
    /**
     * Smallest number of live matvec threads for the two-row form: with fewer, halving the workgroups
     * starves the device more than the halved activation re-reads pay back.
     */
    static final int TWO_ROW_MIN_THREADS = 32768;
    /**
     * Largest token count the W12 matvec kernels serve: a wider prefill of a W12 weight decodes the weight into
     * its BF16 mirror and re-enters the projection on the mirror, so the BF16 family's GEMM lanes serve it.
     */
    static final int W12_MATVEC_MAX_TOKENS = 64;
    /** The whole-tensor BF16 decode behind a W12 weight's wide prefill. */
    private static final String W12_DECODE_BF16 = "W12_DecodeBF16";

    private final GpuOps ops;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final ComputeBackend backend;
    private final DeviceKind deviceKind;
    private final GpuScratch scratch;
    private final int matvecLocal;
    private final long[] maxGroups;
    private final long maxWorkGroupSize;
    private final Int8MatvecMode int8Mode;
    private final FusedMode fusedMode;
    /** Whether this backend holds the subgroup-row forms ({@link #subgroupRowKernel}). */
    private final boolean subgroupRows;

    /** Binds the projections to the queue, kernels and scratch of the owning GpuOps. */
    GpuProjections(GpuOps ops, ComputeQueue queue, LlmKernels kernels, GpuScratch scratch) {
        this.ops = ops;
        this.queue = queue;
        this.kernels = kernels;
        this.backend = kernels.backend();
        this.deviceKind = DeviceKind.of(backend.backendType(), kernels.caps());
        this.scratch = scratch;
        this.matvecLocal = kernels.matvecLocal();
        this.maxGroups = kernels.caps().getMaxWorkGroupCount();
        this.maxWorkGroupSize = kernels.caps().getMaxWorkGroupSize();
        this.int8Mode = ops.int8Mode;
        this.fusedMode = ops.fusedMode;
        this.subgroupRows = SubgroupRowForms.served(kernels);
    }

    void matvec(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, false, true);
    }

    void matmul(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, true, true);
    }

    boolean gateUpGluUsable(GpuWeight gate, GpuWeight up, GluOp op, ActParams params) {
        ops.ensureOpen();
        return fusedGateUpKernel(gate, up, op, params) != null;
    }

    void gateUpGlu(GpuWeight gate, GpuWeight up, GluOp op, ActParams params, GpuBuffer in, GpuBuffer out) {
        ops.ensureOpen();
        String kernel = fusedGateUpKernel(gate, up, op, params);
        if (kernel == null) {
            throw new UnsupportedOperationException("no fused gate+up kernel serves " + gate + " and " + up
                    + " with " + op + " on the " + backend.backendType() + " backend");
        }
        int cols = GpuOps.cols(gate);
        int rows = GpuOps.rows(gate);
        GpuOps.checkView("fused gate+up input", in, 0, cols);
        GpuOps.checkView("fused gate+up output", out, 0, rows);
        if (GpuOps.overlaps(in, 0, cols, out, 0, rows)) {
            throw new IllegalArgumentException("fused gate+up: input and output views overlap");
        }
        requireMatvecLocal(gate);
        GpuMatmulTap.dense(queue, gate, in, 0, 1, cols);
        GpuMatmulTap.dense(queue, up, in, 0, 1, cols);
        if (kernel.endsWith("_Q8")) {
            GpuBuffer xq = quantizedRow(new GpuOps.View(in, 0), cols);
            launchInt8GateUp(kernel, gate.type(), rows, cols, gate.shard(0), up.shard(0), xq, out);
        } else if (twoRowUsable(kernel, rows)) {
            ops.launchMatvec(twoRowName(kernel), (long) GpuOps.ceilDiv(rows, 2) * matvecLocal, rows, cols,
                    gate.shard(0), up.shard(0), in, out);
        } else {
            ops.launchMatvec(kernel, (long) rows * matvecLocal, rows, cols, gate.shard(0), up.shard(0), in, out);
        }
    }

    /**
     * The fused gate+up kernel for these weights, or null. Structural objections (shapes, types, scale, transform,
     * op) are not a fusion the caller asked for, so they return null even in {@link FusedMode#ON}; a structurally
     * eligible layer on a backend without the kernel is refused there.
     */
    private String fusedGateUpKernel(GpuWeight gate, GpuWeight up, GluOp op, ActParams params) {
        if (BatchInvariance.enabled()) {
            /* A fused form runs only the single-token path; batch-invariant steps take the separate products at
             * every row count. Requiring the fusion and requiring batch invariance cannot both hold. */
            if (fusedMode.requiresFusion()) {
                throw new IllegalArgumentException("-D" + FusedMode.PROPERTY + "=on requires the fused"
                        + " gate+up kernels, which -D" + BatchInvariance.PROPERTY + " turns off");
            }
            return null;
        }
        if (gate == null || up == null || op != GluOp.SWIGLU
                || (params != null && params != ActParams.NONE)) {
            return null;
        }
        if (gate.type() != up.type() || !gate.isSingleShard() || !up.isSingleShard()
                || gate.rows() != up.rows() || gate.cols() != up.cols()) {
            return null;
        }
        if (gate.hasScale() || up.hasScale() || gate.hasInputTransform() || up.hasInputTransform()
                || gate.hadamard() != null || up.hadamard() != null) {
            return null;
        }
        if (gate.type() == GGMLType.W12 && ops.activeW12Q != null) {
            /* A W12QView scope routes every W12 projection through the 8-bit Q view, whose row set has no
             * fused form; the separate gate and up matvecs take it. */
            return null;
        }
        if (!fusedMode.permitsFusion()) {
            return null;
        }
        WeightKernels k = ops.family(gate);
        String fused = fusedGateUpName(k.matvec);
        String q8 = k.matvecQ8 != null ? fusedGateUpName(k.matvecQ8) : null;
        if (q8 != null && GpuOps.cols(gate) % gate.type().getBlockSize() == 0
                && (int8LaneSelected(gate.type(), (long) gate.rows() * GpuOps.cols(gate), 1)
                    || subgroupRowKernel(q8, gate.type(), GpuOps.cols(gate)) != null)
                && ops.optionalReady(q8, matvecLocal)) {
            return q8;
        }
        if (ops.optionalReady(fused, matvecLocal)) {
            return fused;
        }
        if (fusedMode.requiresFusion()) {
            throw new UnsupportedKernelException(fused
                    + " is required by -D" + FusedMode.PROPERTY + "=on but the " + backend.backendType()
                    + " backend does not provide it");
        }
        return null;
    }

    /** {@code {T}_MatVec} to {@code {T}_MatVecFusedGateUp} and {@code {T}_MatVec_Q8} to its fused form. */
    private static String fusedGateUpName(String matvec) {
        return matvec.replace("MatVec", "MatVecFusedGateUp");
    }

    void matvecAdd(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer accum, int accOff, int nTokens) {
        ops.ensureOpen();
        int cols = GpuOps.cols(w);
        int rows = GpuOps.rows(w);
        GpuOps.checkTokens(nTokens);
        long nIn = (long) nTokens * cols;
        long nOut = (long) nTokens * rows;
        GpuOps.checkView("input of " + w.name(), in, inOff, nIn);
        GpuOps.checkView("accumulator of " + w.name(), accum, accOff, nOut);
        WeightKernels k = projectionFamily(w);
        boolean invariant = BatchInvariance.enabled();
        boolean q8Add = !invariant && nTokens == 1 && int8LaneUsable(k.matvecAddQ8, w, 1);
        if (!invariant && nTokens == 1 && w.isSingleShard() && accOff == 0 && !w.hasScale()
                && !GpuOps.overlaps(in, inOff, nIn, accum, accOff, nOut)
                && (q8Add || ops.optionalReady(k.matvecAdd, matvecLocal))) {
            requireMatvecLocal(w);
            GpuOps.View x = offsetFree(projectionInput(w, in, inOff, nTokens, cols, false), nTokens, cols);
            GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
            if (q8Add) {
                GpuBuffer xq = quantizedRow(x, cols);
                String sg = subgroupRowKernel(k.matvecAddQ8, w.type(), cols);
                if (sg != null) {
                    launchSubgroupRows(sg, rows, cols, w.shard(0), xq, accum);
                } else {
                    ops.launchMatvec(k.matvecAddQ8, (long) rows * matvecLocal, rows, cols, w.shard(0), xq, accum);
                }
            } else {
                ops.launchMatvec(k.matvecAdd, (long) rows * matvecLocal, rows, cols, w.shard(0), x.buf, accum);
            }
            return;
        }
        ScratchSlot result = scratch.staging(GpuScratch.Staging.RESULT, nOut);
        project(w, in, inOff, result.buffer(), result.offset(), nTokens, true, false);
        ops.axpby((int) nOut, w.scale(), result.buffer(), result.offset(), 1.0f, accum, accOff, accum, accOff);
    }

    private void project(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens,
                         boolean allowTiled, boolean applyScale) {
        ops.ensureOpen();
        int cols = GpuOps.cols(w);
        int rows = GpuOps.rows(w);
        GpuOps.checkTokens(nTokens);
        long nIn = (long) nTokens * cols;
        long nOut = (long) nTokens * rows;
        GpuOps.checkView("input of " + w.name(), in, inOff, nIn);
        GpuOps.checkView("output of " + w.name(), out, outOff, nOut);
        GpuOps.checkInt(outOff + nOut, "output end of " + w.name());
        if (GpuOps.overlaps(in, inOff, nIn, out, outOff, nOut)) {
            throw new IllegalArgumentException("projection " + w.name() + ": input and output views overlap");
        }
        if (w.type() == GGMLType.W12) {
            projectW12(w, in, inOff, out, outOff, nTokens, allowTiled, applyScale);
            return;
        }
        WeightKernels k = ops.family(w);
        /* Batch invariance pins the projection lane independently of the row count. A weight type with a fast
         * GEMM lane takes it for one row and for a full batch alike (its tiles cover the reduction dimension in a
         * fixed order); the split-K twins, which reduce a row in several slices, are never taken there. A type
         * without one keeps the float matvec family for every row count, because the exact tiled GEMM's reduction
         * of an output element does depend on the token tile. */
        boolean invariant = BatchInvariance.enabled();
        /* The multi-column lanes serve small token groups with one weight read per row for the whole
         * group, which beats the GEMM lanes and the tiled GEMM there; their per-row dots outgrow a fast
         * GEMM lane whose cost is flat in the row count. Where the lanes cross over is measured per
         * device kind, weight type and projection size (SmallBatchCrossover): the float multi-column
         * matvec serves [2, floatMatvecNMax] (preempting the tiled GEMM inside that range), and the
         * integer-dot multi-column matvec serves [2, int8Max], both measured answers. A weight whose
         * type has no measured entry keeps the global fast-lane crossover and the integer-dot cap of
         * the previous dispatch. The lanes are pinned off under batch invariance exactly like the
         * single-token lane. */
        String lane = allowTiled ? ops.lanes.fastGemm(w, cols) : null;
        long elements = (long) rows * cols;
        int fastMin = SmallBatchCrossover.fastGemmMinTokens(deviceKind, w.type(), elements);
        int floatNMax = k.matvecN != null
                ? SmallBatchCrossover.floatMatvecNMax(deviceKind, w.type(), elements) : 0;
        int int8Max = SmallBatchCrossover.int8MatvecNMax(deviceKind, w.type(), elements, lane != null);
        boolean q8n = !invariant && int8MultiColumnServes(k, w, cols, nTokens, int8Max);
        boolean floatN = !invariant && nTokens >= 2 && nTokens <= floatNMax
                && ops.optionalReady(k.matvecN, matvecLocal);
        String fast = lane != null && !q8n && !floatN && (invariant || nTokens >= fastMin)
                ? lane : null;
        int fastTileTokens = fast == null ? 0 : ops.lanes.fastTileTokens(fast);
        int fastTileRows = fast == null ? 0 : ops.lanes.fastTileRows(fast);
        int fastWorkGroup = fast == null ? 0 : ops.lanes.fastWorkGroup(fast);
        boolean fastUsable = fast != null
                && ops.lanes.fastUsable(fast, fastTileRows, fastTileTokens, fastWorkGroup, w, nTokens);
        /* A staged launch of fewer tokens than the device tile runs the twin tile of its ladder row
         * (smallFast, 16 tokens up to 16 — on a small-K reduction up to 48 — and 64 above) when the
         * device has it, so the weight stream fills whole output tiles; its split-K runs the twin
         * tile's _SKSplit / _SKSplit64 (SplitKGemmPlan.ofSmallTiles). */
        String smallFast = fastUsable ? ops.lanes.smallTileKernel(fast, w, nTokens) : null;
        int launchTileTokens = smallFast == null ? fastTileTokens : ops.lanes.fastTileTokens(smallFast);
        int launchTileRows = smallFast == null ? fastTileRows : ops.lanes.fastTileRows(smallFast);
        int launchWorkGroup = smallFast == null ? fastWorkGroup : ops.lanes.fastWorkGroup(smallFast);
        Int8ActStaging int8Staging = fastUsable && fast.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)
                ? ops.lanes.int8Staging(w, cols) : null;
        /* A plain MMQ launch on a backend with the subgroup forms (SubgroupRowForms.served: OpenCL or Intel Vulkan)
         * runs the subgroup-row GEMM instead (SubgroupMmqGemm, same arguments and staging); its split-K launches keep
         * the portable KSplit. */
        String sgFast = fastUsable && smallFast == null && !invariant
                ? ops.lanes.subgroupMmqKernel(fast, w, cols, nTokens, int8Staging) : null;
        Fp8ActStaging fp8Staging = fastUsable && FastGemmLanes.fp8Fast(w, fast)
                ? ops.lanes.fp8Staging(w, cols) : null;
        GpuOps.View x = projectionInput(w, in, inOff, nTokens, cols, int8Staging != null || fp8Staging != null);
        GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
        if (fastUsable) {
            boolean staged = FastGemmLanes.stagedFast(fast);
            boolean halfRows = FastGemmLanes.cm2Fast(fast);
            GpuOps.View activations = x;
            if (halfRows) {
                ScratchSlot f16 = ops.lanes.stageF16(w.type(), x.buf, x.off, nTokens, cols);
                activations = new GpuOps.View(f16.buffer(), 0);
            } else if (staged) {
                ScratchSlot q8;
                if (fp8Staging != null) {
                    q8 = ops.lanes.stageFp8(fp8Staging, x.buf, x.off, nTokens, cols);
                } else if (int8Staging != null) {
                    q8 = ops.lanes.stageInt8(int8Staging, x.buf, x.off, nTokens, cols);
                } else {
                    q8 = ops.lanes.stageQ8(x.buf, x.off, nTokens, cols);
                }
                activations = new GpuOps.View(q8.buffer(), q8.offset());
            }
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                int yOff = outOff + (int) w.shardRowStart(s);
                /* A plain-tile int8 launch whose tile grid would leave the last wave part-empty runs as
                 * stream-k (an even share of the grid's chunks per resident block); the twins and a
                 * batch-invariant step keep their grids. */
                StreamKGemmPlan streamK = invariant || smallFast != null ? null
                        : ops.lanes.streamK(fast, w, m, cols, nTokens, yOff, activations.off, rows);
                SplitKGemmPlan split = invariant || streamK != null ? null
                        : ops.lanes.splitK(fast, w, m, cols, nTokens, yOff, activations.off, rows,
                                smallFast != null);
                if (streamK != null) {
                    ops.lanes.streamKLaunch(streamK, w.shard(s), activations.buf, out);
                } else if (split != null) {
                    ops.lanes.splitKLaunch(split, w.shard(s), activations.buf, out, yOff, rows);
                } else {
                    int qStride = staged ? ops.lanes.stagedRowRecords(cols) : cols;
                    int qOff = staged || halfRows ? activations.off : x.off;
                    if (sgFast != null) {
                        int sgGroup = SubgroupMmqGemm.workGroup(kernels, w.type());
                        ops.launch2d(sgFast, SubgroupMmqGemm.globalRows(m, sgGroup),
                                SubgroupMmqGemm.tokenGroups(nTokens), sgGroup, m, cols, nTokens, qStride, qOff, rows,
                                yOff, w.shard(s), activations.buf, out);
                    } else {
                        ops.launch2d(smallFast == null ? fast : smallFast,
                                GpuOps.ceilDiv(m, launchTileRows) * launchWorkGroup,
                                GpuOps.ceilDiv(nTokens, launchTileTokens), launchWorkGroup, m, cols, nTokens, qStride,
                                qOff, rows, yOff, w.shard(s), activations.buf, out);
                    }
                }
            }
        } else if (!invariant && allowTiled && !q8n && !floatN && nTokens >= LlmKernelConstants.TILED_MIN_TOKENS
                && tiledUsable(k, w, nTokens)) {
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                int yOff = outOff + (int) w.shardRowStart(s);
                ops.launch2d(k.tiled, GpuOps.ceilDiv(m, TILED_ROWS) * TILED_WG, GpuOps.ceilDiv(nTokens, TILED_TOKENS), TILED_WG,
                        m, cols, nTokens, cols, x.off, rows, yOff, w.shard(s), x.buf, out);
            }
        } else {
            requireMatvecLocal(w);
            x = offsetFree(x, nTokens, cols);
            boolean direct = w.isSingleShard() && outOff == 0;
            boolean q8 = !invariant && nTokens == 1 && int8LaneUsable(k.matvecQ8, w, 1);
            GpuBuffer xq = q8 ? quantizedRow(x, cols) : q8n ? quantizedRows(x, nTokens, cols) : null;
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                if (direct) {
                    if (q8) {
                        launchInt8Matvec(k.matvecQ8, w.type(), m, cols, w.shard(s), xq, out);
                    } else if (q8n) {
                        launchInt8MatvecN(w.type(), elements, k.matvecNQ8, m, cols, nTokens, w.shard(s),
                                xq, out);
                    } else if (floatN) {
                        launchFloatMatvecN(k.matvecN, m, cols, nTokens, w.shard(s), x.buf, out);
                    } else {
                        matvecShard(k, w.shard(s), m, cols, x.buf, out, nTokens);
                    }
                } else {
                    ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, (long) nTokens * m);
                    if (q8) {
                        launchInt8Matvec(k.matvecQ8, w.type(), m, cols, w.shard(s), xq, stage.buffer());
                    } else if (q8n) {
                        launchInt8MatvecN(w.type(), elements, k.matvecNQ8, m, cols, nTokens, w.shard(s),
                                xq, stage.buffer());
                    } else if (floatN) {
                        launchFloatMatvecN(k.matvecN, m, cols, nTokens, w.shard(s), x.buf, stage.buffer());
                    } else {
                        matvecShard(k, w.shard(s), m, cols, x.buf, stage.buffer(), nTokens);
                    }
                    ops.copy2dUnchecked(nTokens, m, stage.buffer(), stage.offset(), m,
                            out, outOff + (int) w.shardRowStart(s), rows);
                }
            }
        }
        if (applyScale && w.hasScale()) {
            ops.axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
        }
    }

    boolean matvecFamilyServes(GpuWeight w, int nTokens) {
        ops.ensureOpen();
        Objects.requireNonNull(w, "weight");
        GpuOps.checkTokens(nTokens);
        int cols = GpuOps.cols(w);
        WeightKernels k = ops.family(w);
        boolean invariant = BatchInvariance.enabled();
        String lane = ops.lanes.fastGemm(w, cols);
        long elements = (long) GpuOps.rows(w) * cols;
        int fastMin = SmallBatchCrossover.fastGemmMinTokens(deviceKind, w.type(), elements);
        int floatNMax = k.matvecN != null
                ? SmallBatchCrossover.floatMatvecNMax(deviceKind, w.type(), elements) : 0;
        int int8Max = SmallBatchCrossover.int8MatvecNMax(deviceKind, w.type(), elements, lane != null);
        if (!invariant && int8MultiColumnServes(k, w, cols, nTokens, int8Max)) {
            return false;
        }
        if (!invariant && nTokens >= 2 && nTokens <= floatNMax
                && ops.optionalReady(k.matvecN, matvecLocal)) {
            return false;
        }
        String fast = invariant || nTokens >= fastMin ? lane : null;
        if (fast != null && ops.lanes.fastUsable(fast, ops.lanes.fastTileRows(fast), ops.lanes.fastTileTokens(fast),
                ops.lanes.fastWorkGroup(fast), w, nTokens)) {
            return false;
        }
        if (!invariant && nTokens >= LlmKernelConstants.TILED_MIN_TOKENS && tiledUsable(k, w, nTokens)) {
            return false;
        }
        if (k.matvecQ8 != null && !invariant && nTokens == 1 && int8LaneUsable(k.matvecQ8, w, 1)) {
            return false;
        }
        return matvecLocal % MATVEC_MAX_TPS == 0;
    }

    /**
     * The projection input: the caller's view; or the weight's PRISM rotation in staging; or the full
     * {@link InputTransform} chain in staging &mdash; the PRE_SCALE part multiplying each column into the input
     * staging, the HADAMARD part rotating it into the other staging region (a copy of the input when there is no
     * rotation), ACT_QUANT fake-quantizing in place and PERMUTE gathering out of place into whichever region the
     * preceding step did not read from.
     *
     * <p>{@code actQuantStaged} says the int8 MMQ lane stages the activation codes itself
     * ({@link FastGemmLanes#stageInt8}), so the ACT_QUANT part is left to that pass: the input then keeps its
     * float values (scaled, rotated and copied, but not fake-quantized) and the scheme is applied once, as int8
     * codes.</p>
     *
     * @throws IllegalArgumentException when the weight carries a per-expert activation pre-scale (the dense
     *                                  projection has no expert index) or lacks its device pre-scale buffer
     */
    private GpuOps.View projectionInput(GpuWeight w, GpuBuffer in, int inOff, int nTokens, int cols,
                                 boolean actQuantStaged) {
        InputTransform t = w.inputTransform();
        InputTransform.ActQuant a = t != null ? t.actQuant() : null;
        HadamardSpec h = w.hadamard();
        float[] pre = t != null ? t.preScale() : null;
        if (pre != null && pre.length != cols) {
            throw new IllegalArgumentException("projection " + w.name() + ": an activation pre-scale of "
                    + pre.length + " values for " + cols + " columns is per expert, which the dense projection"
                    + " does not apply");
        }
        boolean quantizeHere = a != null && !actQuantStaged;
        boolean needsStaging = pre != null || h != null || quantizeHere || (t != null && t.hasPerm());
        if (!needsStaging) {
            return new GpuOps.View(in, inOff);
        }
        ScratchSlot first = scratch.staging(GpuScratch.Staging.INPUT, (long) nTokens * cols);
        GpuBuffer xb = first.buffer();
        int xo = first.offset();
        if (pre != null) {
            if (w.inputPreScale() == null) {
                throw new IllegalArgumentException("projection " + w.name() + ": its activation pre-scale has no"
                        + " device buffer");
            }
            ops.scaleCols(w.inputPreScale(), in, inOff, xb, xo, nTokens, cols);
        } else if (h != null) {
            ops.hadamard(h, in, inOff, xb, xo, nTokens, cols);
        } else {
            ops.copy2dUnchecked(nTokens, cols, in, inOff, cols, xb, xo, cols);
        }
        if (pre != null && h != null) {
            ScratchSlot rotated = scratch.staging(GpuScratch.Staging.PERMUTED, (long) nTokens * cols);
            ops.hadamard(h, xb, xo, rotated.buffer(), rotated.offset(), nTokens, cols);
            xb = rotated.buffer();
            xo = rotated.offset();
        }
        if (quantizeHere) {
            ops.actFakeQuant(a.scheme(), a.scale(0), a.zeroPoint(0), a.globalScale(0), xb, xo, xb, xo, nTokens, cols);
        }
        if (t != null && t.hasPerm()) {
            // The gather must not overlap its source: PERMUTE_COLS reads out of place. The rotated result sits in
            // the permuted region, so it gathers back into the input region the rotation consumed.
            ScratchSlot p = xb == first.buffer() && xo == first.offset()
                    ? scratch.staging(GpuScratch.Staging.PERMUTED, (long) nTokens * cols)
                    : first;
            ops.permuteCols(w.inputPerm(), xb, xo, p.buffer(), p.offset(), nTokens, cols);
            return new GpuOps.View(p.buffer(), p.offset());
        }
        return new GpuOps.View(xb, xo);
    }

    /** The input at element 0 of its buffer, copying it to staging when it starts elsewhere. */
    private GpuOps.View offsetFree(GpuOps.View x, int nTokens, int cols) {
        if (x.off == 0) {
            return x;
        }
        ScratchSlot s = scratch.staging(GpuScratch.Staging.INPUT, (long) nTokens * cols);
        ops.copy2dUnchecked(nTokens, cols, x.buf, x.off, cols, s.buffer(), s.offset(), cols);
        return new GpuOps.View(s.buffer(), s.offset());
    }

    /**
     * The float matvec lanes of one shard. The single-token path runs {@code {T}_MatVec} (or its two-rows-per-
     * workgroup form) and the multi-token path {@code {T}_BatchMatMul_2D}; both reduce every output row over the
     * weight blocks in the same order, so a row's value does not depend on the token count. Batch invariance skips
     * the two-row form: it is a single-token-only lane, and the batch kernel must serve every row count instead.
     */
    private void matvecShard(WeightKernels k, GpuBuffer shard, int m, int cols, GpuBuffer x, GpuBuffer dst,
                             int nTokens) {
        long global = (long) m * matvecLocal;
        boolean invariant = BatchInvariance.enabled();
        if (!invariant && nTokens == 1 && twoRowUsable(k.matvec, m)) {
            ops.launchMatvec(twoRowName(k.matvec), (long) GpuOps.ceilDiv(m, 2) * matvecLocal, m, cols, shard, x, dst);
        } else if (nTokens == 1) {
            ops.launchMatvec(k.matvec, global, m, cols, shard, x, dst);
        } else if (m <= maxGroups[0] && nTokens <= maxGroups[1] && ops.optionalReady(k.batch2d, matvecLocal, 1)) {
            ops.launch2d(k.batch2d, global, nTokens, matvecLocal, m, cols, nTokens, shard, x, dst);
        } else {
            ops.launchMatvec(k.batch, global, m, cols, nTokens, shard, x, dst);
        }
    }

    /** {@code {T}_MatVec} to {@code {T}_MatVec2}, the two-rows-per-workgroup form. */
    private static String twoRowName(String matvec) {
        return matvec.replace("MatVec", "MatVec2");
    }

    /**
     * The W12 projection: up to {@value #W12_MATVEC_MAX_TOKENS} tokens run the {@code W12_*} row kernels, which
     * decode every row's plane codes in registers and apply its window scale and escape corrections; a wider
     * prefill decodes the weight into its BF16 mirror ({@link #w12Bf16Mirror}) and re-enters {@link #project} on
     * the mirror, so the BF16 family's fast GEMM lanes and tiled GEMM serve it. An open {@link GpuOps.W12QView} scope
     * routes the row kernels over the nested 8-bit view of the code bytes ({@code W12Q_*}); the fused gate/up
     * form has no Q-view twin, so it is not taken while a scope is open ({@link #fusedGateUpKernel}).
     */
    private void projectW12(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens,
                            boolean allowTiled, boolean applyScale) {
        long nOut = (long) nTokens * GpuOps.rows(w);
        if (nTokens > W12_MATVEC_MAX_TOKENS) {
            GpuWeight mirror = w12Bf16Mirror(w);
            project(mirror, in, inOff, out, outOff, nTokens, allowTiled, false);
            if (applyScale && w.hasScale()) {
                ops.axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
            }
            return;
        }
        int cols = GpuOps.cols(w);
        requireMatvecLocal(w);
        WeightKernels k = projectionFamily(w);
        GpuOps.View x = projectionInput(w, in, inOff, nTokens, cols, false);
        GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
        x = offsetFree(x, nTokens, cols);
        boolean direct = w.isSingleShard() && outOff == 0;
        for (int s = 0; s < w.shardCount(); s++) {
            int m = (int) w.shardRows(s);
            if (direct) {
                matvecShard(k, w.shard(s), m, cols, x.buf, out, nTokens);
            } else {
                ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, (long) nTokens * m);
                matvecShard(k, w.shard(s), m, cols, x.buf, stage.buffer(), nTokens);
                ops.copy2dUnchecked(nTokens, m, stage.buffer(), stage.offset(), m,
                        out, outOff + (int) w.shardRowStart(s), GpuOps.rows(w));
            }
        }
        if (applyScale && w.hasScale()) {
            ops.axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
        }
    }

    /**
     * Decodes every row of the W12 weight into its BF16 mirror in the W12 staging region (one
     * {@value #W12_DECODE_BF16} launch per row shard, each shard placed at its row start) and returns the mirror
     * as a BF16 weight of the same shape. The mirror lives in staging, so it is valid for this projection only;
     * the in-order queue reads it before the next projection reuses the region.
     */
    private GpuWeight w12Bf16Mirror(GpuWeight w) {
        int cols = GpuOps.cols(w);
        long elements = Math.multiplyExact(w.rows(), cols);
        ScratchSlot mirror = scratch.staging(GpuScratch.Staging.W12_BF16, elements / 2);
        for (int s = 0; s < w.shardCount(); s++) {
            ops.launchMatvec(W12_DECODE_BF16, w.shardRows(s) * matvecLocal, (int) w.shardRows(s), cols,
                    (int) w.shardRowStart(s), w.shard(s), mirror.buffer());
        }
        return GpuWeight.of(w.name(), GGMLType.BF16, w.ne(), mirror.buffer());
    }

    /** {@link GpuOps#family} for the weight, or the W12 Q-view row set while a {@link W12QView} scope is open. */
    private WeightKernels projectionFamily(GpuWeight w) {
        WeightKernels k = ops.family(w);
        if (w.type() == GGMLType.W12 && ops.activeW12Q != null) {
            k = GpuWeightFamilies.W12Q_FAMILY;
        }
        return k;
    }

    /**
     * Launches the integer-dot matvec {@code kernel} over {@code m} rows of {@code weight} against the Q8_1
     * row {@code xq}: the paired {@code {T}_MatVec2_Q8} form at {@code ceil(m / 2) * L} when
     * {@link #twoRowUsable} selects it and the type measured faster paired on this device kind
     * ({@link Int8MatvecTuning#pairedIsFaster}), so the activation row is read once per two weight rows,
     * otherwise the single-row form at {@code m * L}.
     */
    private void launchInt8Matvec(String kernel, GGMLType type, int m, int cols, GpuBuffer weight, GpuBuffer q8,
                                  GpuBuffer out) {
        String sg = subgroupRowKernel(kernel, type, cols);
        if (sg != null) {
            launchSubgroupRows(sg, m, cols, weight, q8, out);
        } else if (pairInt8Rows(type) && twoRowUsable(kernel, m)) {
            ops.launchMatvec(twoRowName(kernel), (long) GpuOps.ceilDiv(m, 2) * matvecLocal, m, cols, weight, q8, out);
        } else {
            ops.launchMatvec(kernel, (long) m * matvecLocal, m, cols, weight, q8, out);
        }
    }

    /**
     * Launches the multi-column integer-dot matvec {@code kernel} over {@code m} rows of {@code weight}
     * against the {@code nTokens} consecutive Q8_1 rows of {@code xq} ({@code nTokens} in
     * [1, {@link LlmKernelConstants#MATVEC_N_MAX_TOKENS}]). The exact-token-count twin
     * ({@code {T}_MatVecN<n>_Q8}, the same kernel with the token loop folded to {@code nTokens}) serves
     * the launch when the type provides it, and the counted two-row twin ({@code {T}_MatVec2N<n>_Q8},
     * one work-group per row pair so the Q8_1 rows are read once per two weight rows) serves it when
     * {@link Int8MatvecTuning#pairedCountedIsFaster} selects the pairing at {@code elements},
     * {@code nTokens} and {@code m}. Ahead of both, the tensor-core multi-column matvec
     * ({@code {T}_MatVecMma{8,16}_Q8}, {@link MatVecMmaPlan}) serves the launch where
     * {@link MatVecMmaTuning} measured it faster on this device kind; it reads the same Q8_1 rows.
     */
    private void launchInt8MatvecN(GGMLType type, long elements, String kernel, int m, int cols, int nTokens,
                                   GpuBuffer weight, GpuBuffer xq, GpuBuffer out) {
        MatVecMmaPlan mma = tensorCoreMatvecN(type, m, cols, nTokens);
        if (mma != null) {
            ops.launch2d(mma.kernel(), mma.globalWorkSize()[0], 1, (int) mma.localWorkSize()[0], m, cols, nTokens,
                    weight, xq, out);
            return;
        }
        String counted = countedMatvecN(kernel, nTokens);
        if (counted != kernel) {
            String paired = pairedCountedMatvecN(type, elements, counted, nTokens, m);
            if (paired != null) {
                ops.launchMatvec(paired, (long) GpuOps.ceilDiv(m, 2) * matvecLocal, m, cols, weight, xq, out);
            } else {
                ops.launchMatvec(counted, (long) m * matvecLocal, m, cols, weight, xq, out);
            }
        } else {
            ops.launchMatvec(counted, (long) m * matvecLocal, m, cols, nTokens, weight, xq, out);
        }
    }

    /**
     * Whether the integer-dot multi-column lane takes {@code nTokens} (two or more) rows through the weight
     * {@code w} of {@code cols} columns: through the tensor-core form wherever {@link MatVecMmaTuning} measured it
     * faster than every lane the projection would otherwise take, on every shard (its cost is flat in the row
     * count inside its token tile, so this holds above the counted cap and on projections whose counted forms
     * lose), else through the counted forms up to the measured cap {@code int8Max}
     * ({@link SmallBatchCrossover#int8MatvecNMax}) behind their own evidence ({@link #int8LaneUsable}).
     */
    private boolean int8MultiColumnServes(WeightKernels k, GpuWeight w, int cols, int nTokens, int int8Max) {
        if (nTokens < 2) {
            return false;
        }
        if (int8Mode.permitsLane() && cols % w.type().getBlockSize() == 0
                && tensorCoreServesShards(w, cols, nTokens)) {
            return true;
        }
        return nTokens <= int8Max && int8LaneUsable(k.matvecNQ8, w, nTokens);
    }

    /** Whether the tensor-core multi-column matvec serves {@code nTokens} rows through every shard of {@code w}. */
    private boolean tensorCoreServesShards(GpuWeight w, int cols, int nTokens) {
        for (int s = 0; s < w.shardCount(); s++) {
            if (tensorCoreMatvecN(w.type(), (int) w.shardRows(s), cols, nTokens) == null) {
                return false;
            }
        }
        return true;
    }

    /**
     * The tensor-core multi-column matvec plan of {@code nTokens} rows through {@code m x cols} rows of
     * {@code type}, or null when {@link MatVecMmaTuning} keeps the counted forms here or the kernel is not in the
     * library (or cannot run its widest work-group on this device).
     */
    private MatVecMmaPlan tensorCoreMatvecN(GGMLType type, int m, int cols, int nTokens) {
        int warps = MatVecMmaTuning.warps(deviceKind, type, m, cols, nTokens);
        if (warps == 0) {
            return null;
        }
        MatVecMmaPlan plan = MatVecMmaPlan.of(type, m, cols, nTokens, warps);
        return ops.kernels().kernelNames().contains(plan.kernel())
                && ops.optionalReady(plan.kernel(), 32L * MatVecMmaPlan.MAX_WARPS) ? plan : null;
    }

    /**
     * The counted two-row twin of the counted one-row matvec: {@code MatVecN} replaced by
     * {@code MatVec2N} in {@code counted} ({@code {T}_MatVecN4_Q8} gives {@code {T}_MatVec2N4_Q8}),
     * or null when the pairing measured slower here ({@link Int8MatvecTuning#pairedCountedIsFaster})
     * or the twin is not in the library (or cannot run the matvec local size on this device).
     */
    private String pairedCountedMatvecN(GGMLType type, long elements, String counted, int nTokens, int rows) {
        if (!Int8MatvecTuning.pairedCountedIsFaster(deviceKind, type, elements, nTokens, rows)) {
            return null;
        }
        String paired = pairedTwinName(counted);
        return ops.kernels().kernelNames().contains(paired) && ops.optionalReady(paired, matvecLocal)
                ? paired : null;
    }

    /** {@code MatVecN} replaced by {@code MatVec2N} in a counted multi-column matvec name. */
    private static String pairedTwinName(String counted) {
        int at = counted.indexOf("MatVecN");
        return at < 0 ? counted
                : counted.substring(0, at) + "MatVec2N" + counted.substring(at + "MatVecN".length());
    }

    /**
     * The exact-token-count twin of a multi-column matvec kernel
     * ({@link LlmKernelConstants#countedMatvecNName}), or {@code kernel} when the twin is not in the
     * library (or cannot run the matvec local size on this device).
     */
    private String countedMatvecN(String kernel, int nTokens) {
        String counted = LlmKernelConstants.countedMatvecNName(kernel, nTokens);
        return counted.equals(kernel) ? kernel
                : ops.kernels().kernelNames().contains(counted)
                        && ops.optionalReady(counted, matvecLocal) ? counted : kernel;
    }

    /**
     * Launches the float-input multi-column matvec {@code kernel} over {@code m} rows of {@code weight}
     * against the {@code nTokens} consecutive float rows of {@code x} ({@code nTokens} in
     * [2, {@link SmallBatchCrossover#floatMatvecNMax}]): the one-workgroup-per-row geometry of the
     * single-row form, with the weight row read once for the whole token group and every row's
     * reduction the exact single-row reduction.
     */
    private void launchFloatMatvecN(String kernel, int m, int cols, int nTokens, GpuBuffer weight,
                                    GpuBuffer x, GpuBuffer out) {
        ops.launchMatvec(kernel, (long) m * matvecLocal, m, cols, nTokens, weight, x, out);
    }

    /**
     * Launches the fused gate/up integer-dot matvec {@code kernel} over {@code m} rows of
     * {@code gateWeight} and {@code upWeight} against the Q8_1 row {@code xq}, pairing the rows in the
     * {@code {T}_MatVec2FusedGateUp_Q8} form above the two-row floor when the type measured faster paired.
     */
    private void launchInt8GateUp(String kernel, GGMLType type, int m, int cols, GpuBuffer gateWeight,
                                  GpuBuffer upWeight, GpuBuffer q8, GpuBuffer out) {
        String sg = subgroupRowKernel(kernel, type, cols);
        if (sg != null) {
            launchSubgroupRows(sg, m, cols, gateWeight, upWeight, q8, out);
        } else if (pairInt8Rows(type) && twoRowUsable(kernel, m)) {
            ops.launchMatvec(twoRowName(kernel), (long) GpuOps.ceilDiv(m, 2) * matvecLocal, m, cols, gateWeight, upWeight, q8,
                    out);
        } else {
            ops.launchMatvec(kernel, (long) m * matvecLocal, m, cols, gateWeight, upWeight, q8, out);
        }
    }

    /** Whether the paired integer-dot form measured faster than the single-row form for {@code type} here. */
    private boolean pairInt8Rows(GGMLType type) {
        return Int8MatvecTuning.pairedIsFaster(deviceKind, type);
    }

    /**
     * The subgroup-row twin ({@link SubgroupRowMatvec}) of the integer-dot single-token kernel {@code kernel} for
     * rows of {@code cols} elements of {@code type}, or null when it does not serve: the configured
     * {@link Int8MatvecMode} keeps the lane, batch invariance is off (the twin's float summation order differs
     * from the multi-column forms'), the type and shape meet the twin's preconditions, this backend holds the
     * pinned subgroup forms ({@link SubgroupRowForms#served}: an OpenCL program built with them, or an Intel Vulkan
     * device that pins their 16-lane subgroups) and the kernel is prepared. Where it serves it replaces the
     * single-row and paired forms, which it outruns on every measured shape.
     */
    private String subgroupRowKernel(String kernel, GGMLType type, int cols) {
        if (!subgroupRows || !int8Mode.permitsLane() || BatchInvariance.enabled()
                || !SubgroupRowMatvec.shapeServes(type, cols)) {
            return null;
        }
        String twin = SubgroupRowMatvec.twinName(kernel, type);
        return twin != null && ops.optionalReady(twin, SubgroupRowMatvec.WORK_GROUP, 1) ? twin : null;
    }

    /**
     * Launches a subgroup-row form over {@code rows} output rows with the form's own work-group, one-dimensional
     * like the family it replaces, so a head of more rows than {@code maxComputeWorkGroupCount[0]} work-groups
     * hold folds instead of failing.
     */
    private void launchSubgroupRows(String kernel, int rows, int cols, Object... buffers) {
        Object[] args = new Object[buffers.length + 2];
        args[0] = rows;
        args[1] = cols;
        System.arraycopy(buffers, 0, args, 2, buffers.length);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{SubgroupRowMatvec.globalSize(rows)},
                new long[]{SubgroupRowMatvec.WORK_GROUP}, args);
    }

    /**
     * Whether the two-row form of {@code kernel} serves {@code m} rows: at least two rows, enough live
     * threads ({@link #TWO_ROW_MIN_THREADS}) that pairing the rows does not starve the device more
     * than the halved activation re-reads pay back, and the kernel prepared at the matvec work-group
     * size (a weight type without the family form, such as F16 or F32, takes the single-row kernel).
     * The integer-dot lane's {@code {T}_MatVec_Q8} and {@code {T}_MatVecFusedGateUp_Q8} pair through
     * {@code {T}_MatVec2_Q8} and {@code {T}_MatVec2FusedGateUp_Q8} under the same floor.
     */
    private boolean twoRowUsable(String kernel, int m) {
        if (m < 2 || (long) GpuOps.ceilDiv(m, 2) * matvecLocal < TWO_ROW_MIN_THREADS) {
            return false;
        }
        return ops.optionalReady(twoRowName(kernel), matvecLocal);
    }

    /**
     * Whether the integer-dot (Q8_1 activation) matvec lane can serve a projection of
     * {@code w} through {@code kernel} (the single-token and multi-column lanes share the gate): the
     * weight type has the kernel, its rows are a whole number of
     * weight blocks (the int8 kernels read complete 32/256-element groups), the configured
     * {@link Int8MatvecMode} and device permit it (packed dot products reported, or the mode forces the
     * lane), {@link Int8MatvecTuning} has no decode evidence against the type on this device kind, and
     * the kernel is prepared at the matvec work-group size. {@code nTokens} is 1 for the single-row
     * lane and the launch's row count for the counted multi-column lane. A single-token kernel whose subgroup-row
     * twin serves ({@link #subgroupRowKernel}) takes the lane whatever that evidence says: the evidence measured
     * the family's forms, which the twin replaces.
     */
    private boolean int8LaneUsable(String kernel, GpuWeight w, int nTokens) {
        if (kernel == null || GpuOps.cols(w) % w.type().getBlockSize() != 0) {
            return false;
        }
        boolean selected = int8LaneSelected(w.type(), (long) GpuOps.rows(w) * GpuOps.cols(w), nTokens)
                || (nTokens == 1 && subgroupRowKernel(kernel, w.type(), GpuOps.cols(w)) != null);
        return selected && ops.optionalReady(kernel, matvecLocal);
    }

    /**
     * Whether the configured {@link Int8MatvecMode} selects the lane for a projection of {@code type}
     * with {@code elements} weight elements on this device kind: the mode is not {@link Int8MatvecMode#OFF},
     * and either it forces the lane or the type measured faster with the lane here at that projection
     * size behind the lane's feature gate — the single-token lane's own evidence
     * ({@link Int8MatvecTuning#laneIsFaster}) behind the device's packed dot product for the
     * single-row form ({@code nTokens == 1}), the counted multi-column lane's
     * ({@link Int8MatvecTuning#countedLaneIsFaster(DeviceKind, GGMLType, long, int)} at {@code nTokens} rows)
     * behind the packed dot product or the measured portable-arithmetic exception
     * ({@link Int8MatvecTuning#countedLaneNeedsPackedDots}) for the counted forms.
     */
    private boolean int8LaneSelected(GGMLType type, long elements, int nTokens) {
        if (!int8Mode.permitsLane()) {
            return false;
        }
        if (int8Mode.forcesLane()) {
            return true;
        }
        if (nTokens > 1) {
            return Int8MatvecTuning.countedLaneIsFaster(deviceKind, type, elements, nTokens)
                    && (kernels.caps().hasFeature(DeviceFeature.INTEGER_DOT_PRODUCT)
                        || !Int8MatvecTuning.countedLaneNeedsPackedDots(deviceKind, type));
        }
        return kernels.caps().hasFeature(DeviceFeature.INTEGER_DOT_PRODUCT)
                && Int8MatvecTuning.laneIsFaster(deviceKind, type, elements);
    }

    /**
     * The Q8_1 blocks of a single-token row at element 0 of its buffer: the open {@link GpuOps.SharedQ8}'s when it
     * covers the view, the blocks the row's norm wrote ({@link Q8Activations#take}), else {@link #quantizeQ8}.
     */
    private GpuBuffer quantizedRow(GpuOps.View x, int cols) {
        GpuOps.SharedQ8 share = ops.activeShare;
        if (share != null && share.matches(x, cols, 1)) {
            return share.blocks();
        }
        GpuBuffer normed = ops.q8.take(x.buf, 1, cols);
        return normed != null ? normed : quantizeQ8(x.buf, cols);
    }

    /** {@link #quantizeQ8N} through the open {@link GpuOps.SharedQ8} when it covers these rows. */
    private GpuBuffer quantizedRows(GpuOps.View x, int nTokens, int cols) {
        GpuOps.SharedQ8 share = ops.activeShare;
        if (share != null && share.matches(x, cols, nTokens)) {
            return share.blocks();
        }
        return quantizeQ8N(x.buf, nTokens, cols);
    }

    /**
     * Quantizes one F32 activation row of {@code cols} elements (a multiple of {@link #GpuOps.Q8_1_BLOCK_SIZE})
     * to Q8_1 blocks in the Q8_1 staging slot ({@value #GpuOps.QUANTIZE_Q8_1}) for the integer-dot matvec lane.
     * The row must start at element 0 of {@code x}.
     */
    private GpuBuffer quantizeQ8(GpuBuffer x, int cols) {
        int blocks = cols / GpuOps.Q8_1_BLOCK_SIZE;
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1, (long) blocks * GpuOps.Q8_1_FLOATS);
        ops.quantizeQ8_1(x, cols, slot.buffer());
        return slot.buffer();
    }

    /**
     * The Q8_1 blocks of {@code nTokens} consecutive F32 activation rows of {@code cols} elements (each a multiple
     * of {@link #GpuOps.Q8_1_BLOCK_SIZE}) for the multi-column integer-dot matvec lane: the blocks the rows' norm
     * wrote ({@link Q8Activations#take}), else the rows quantized to the Q8_1 staging slot
     * ({@value #GpuOps.QUANTIZE_Q8_1}). The rows are contiguous in the input and in the blocks, so the quantizer's
     * linear block walk serves the whole group in one launch; the group must start at element 0 of {@code x}.
     */
    private GpuBuffer quantizeQ8N(GpuBuffer x, int nTokens, int cols) {
        GpuBuffer normed = ops.q8.take(x, nTokens, cols);
        if (normed != null) {
            return normed;
        }
        int blocks = cols / GpuOps.Q8_1_BLOCK_SIZE;
        int size = Math.multiplyExact(nTokens, cols);
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1, (long) blocks * nTokens * GpuOps.Q8_1_FLOATS);
        ops.quantizeQ8_1(x, size, slot.buffer());
        return slot.buffer();
    }

    private boolean tiledUsable(WeightKernels k, GpuWeight w, int nTokens) {
        if (maxWorkGroupSize < TILED_WG || GpuOps.ceilDiv(nTokens, TILED_TOKENS) > maxGroups[1]) {
            return false;
        }
        for (int s = 0; s < w.shardCount(); s++) {
            if (GpuOps.ceilDiv(w.shardRows(s), TILED_ROWS) > maxGroups[0]) {
                return false;
            }
        }
        return ops.optionalReady(k.tiled, TILED_WG, 1);
    }

    /** Runs the preparation described on {@link GpuOps#prewarmWeightKernels}. */
    void prewarmWeightKernels(GGMLType type, boolean matmul, boolean lookup) {
        ops.ensureOpen();
        Objects.requireNonNull(type, "type");
        WeightKernels k = ops.family(type);
        long[] matvec = {matvecLocal};
        if (matmul && matvecLocal % MATVEC_MAX_TPS == 0) {
            prewarmIfPresent(k.matvec, matvec);
            prewarmIfPresent(k.batch, matvec);
            ops.optionalReady(k.matvecAdd, matvecLocal);
            ops.optionalReady(k.batch2d, matvecLocal, 1);
            ops.optionalReady(twoRowName(k.matvec), matvecLocal);
            if (k.matvecN != null && SmallBatchCrossover.hasFloatMatvecN(deviceKind, type)) {
                ops.optionalReady(k.matvecN, matvecLocal);
            }
            if (k.matvecQ8 != null && (int8LaneSelected(type, Long.MAX_VALUE, 1)
                    || int8LaneSelected(type, Long.MAX_VALUE, LlmKernelConstants.MATVEC_N_COUNTED_MAX))) {
                ops.optionalReady(k.matvecQ8, matvecLocal);
                ops.optionalReady(k.matvecAddQ8, matvecLocal);
                ops.optionalReady(k.matvecNQ8, matvecLocal);
                if (k.matvecNQ8 != null) {
                    // The widest exact count the type instantiates (LlmKernelConstants.countedMaxTokens):
                    // the dispatched range (MATVEC_N_DISPATCH_MAX) for most family types, the measured
                    // entries' range above it for the types whose kernels instantiate to 16. The smaller
                    // counts' readiness the dispatch checks on its first launch of each.
                    String counted = countedMatvecN(k.matvecNQ8, LlmKernelConstants.countedMaxTokens(type));
                    ops.optionalReady(pairedTwinName(counted), matvecLocal);
                }
                if (pairInt8Rows(type)) {
                    ops.optionalReady(twoRowName(k.matvecQ8), matvecLocal);
                    ops.optionalReady(twoRowName(fusedGateUpName(k.matvecQ8)), matvecLocal);
                }
            }
            if (type == GGMLType.W12) {
                // The whole-tensor BF16 decode behind a wide prefill, and the Q-view row set of the
                // neuralfree.weights.w12 verification pass; the fused gate/up has no Q-view twin.
                prewarmIfPresent(W12_DECODE_BF16, matvec);
                ops.optionalReady(GpuWeightFamilies.W12Q_FAMILY.matvec, matvecLocal);
                ops.optionalReady(GpuWeightFamilies.W12Q_FAMILY.matvecAdd, matvecLocal);
                ops.optionalReady(GpuWeightFamilies.W12Q_FAMILY.batch, matvec);
                ops.optionalReady(GpuWeightFamilies.W12Q_FAMILY.batch2d, matvecLocal, 1);
            }
        }
        if (matmul && k.tiled != null && maxWorkGroupSize >= TILED_WG) {
            ops.optionalReady(k.tiled, TILED_WG, 1);
            ops.lanes.prewarm(type);
        }
        if (lookup) {
            prewarmIfPresent(k.embed, matvec);
            prewarmIfPresent(k.batchEmbed, matvec);
        }
    }

    /** Prepares a provided kernel that has no fallback; one that cannot run {@code localSize} fails the bind. */
    private void prewarmIfPresent(String kernel, long[] localSize) {
        if (kernels.has(kernel)) {
            kernels.prewarm(kernel, localSize);
        }
    }

    private void requireMatvecLocal(GpuWeight w) {
        if (matvecLocal % MATVEC_MAX_TPS != 0) {
            throw new UnsupportedOperationException("matvec of " + w.name() + " needs a work-group size that is a"
                    + " multiple of " + MATVEC_MAX_TPS + "; this device allows " + matvecLocal);
        }
    }
}
