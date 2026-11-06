package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.TiledGemmPlan;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The expert-sorted rows of one set of batched expert products on the GPU, and the kernel choice of the grouped GEMM
 * that consumes them.
 *
 * <p>Per step of {@code n} tokens ({@code n * k} pairs, ids in the bound {@code ids} buffer) {@link #dispatch()}
 * enqueues the stable counting sort by expert: {@code LLM_MoE_Bucketize} (one work-group) for at most
 * {@link MoeKernels#BUCKET_CHUNK} pairs, else {@code LLM_MoE_BucketCount}, {@code LLM_MoE_BucketScan} and
 * {@code LLM_MoE_BucketScatter} over {@code ceil(pairs / BUCKET_CHUNK)} work-groups, which write the same counts,
 * offsets and sorted rows. When built aligned, {@code LLM_MoE_AlignBlocks} then lays the rows out in whole tiles per
 * expert over a multi-work-group grid ({@link MoeKernels#alignGlobal}). Every output is deterministic: integer sums
 * and in-chunk ranks only, no order-dependent atomics.</p>
 *
 * <p>The buffers are the {@code <prefix>.*} {@link GpuScratch} temporaries sized for the bind-time token bound, so
 * they carry data only within one block call. Not thread-safe.</p>
 */
public final class GpuGroupedRows implements AutoCloseable {

    private final int k;
    private final int nExpert;
    private final int local;
    private final int nTiles;
    private final GpuBuffer counts;
    private final GpuBuffer offsets;
    private final GpuBuffer rowToken;
    private final GpuBuffer rowSlot;
    private final GpuBuffer gemmToken;
    private final GpuBuffer gemmSlot;
    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private PreparedDispatch bucketize;
    private PreparedDispatch count;
    private PreparedDispatch scan;
    private PreparedDispatch scatter;
    private PreparedDispatch align;
    private int lastTokens = 1;

    /**
     * Prepares the sort (and the alignment when {@code aligned}) of up to {@code maxTokens * k} pairs.
     *
     * @param prefix    name prefix of the scratch temporaries ({@code <prefix>.counts}, {@code <prefix>.row_token}, ...)
     * @param maxTokens largest token count of a step (the scratch {@code ubatch})
     * @param ids       {@code int} expert ids of the pairs, {@code p = t * k + j}, from element 0
     * @param aligned   whether the step also writes the aligned rows of the grouped GEMM; check {@link #alignUsable}
     * @throws IllegalArgumentException for non-positive sizes or more than {@link MoeKernels#MAX_EXPERTS} experts
     */
    public GpuGroupedRows(GpuOps ops, GpuScratch scratch, LlmKernels kernels, String prefix, int maxTokens, int k,
                          int nExpert, GpuBuffer ids, boolean aligned) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(scratch, "scratch");
        Objects.requireNonNull(kernels, "kernels");
        Objects.requireNonNull(prefix, "prefix");
        Objects.requireNonNull(ids, "ids");
        int maxRows = pairs(maxTokens, k);
        this.k = k;
        this.nExpert = nExpert;
        this.local = Integer.highestOneBit(Math.max(1, Math.min(ops.local(), MoeKernels.MAX_LOCAL)));
        this.nTiles = aligned ? MoeKernels.groupedTiles(maxRows, nExpert) : 0;
        this.counts = scratch.temp(prefix + ".counts", nExpert).buffer();
        this.offsets = scratch.temp(prefix + ".offsets", nExpert).buffer();
        this.rowToken = scratch.temp(prefix + ".row_token", maxRows).buffer();
        this.rowSlot = scratch.temp(prefix + ".row_slot", maxRows).buffer();
        long positions = aligned ? MoeKernels.alignedPositions(maxRows, nExpert) : 0;
        this.gemmToken = aligned ? scratch.temp(prefix + ".gemm_token", positions).buffer() : null;
        this.gemmSlot = aligned ? scratch.temp(prefix + ".gemm_slot", positions).buffer() : null;
        try {
            long[] group = {local};
            bucketize = keep(ops.prepare(MoeKernels.BUCKETIZE, group, group,
                    MoeKernels.bucketizeArgs(1, k, nExpert, ids, counts, offsets, rowToken, rowSlot)));
            if (maxRows > MoeKernels.BUCKET_CHUNK && multiGroupSort(kernels)) {
                prepareMultiGroup(ops, scratch, prefix, maxRows, ids);
            }
            if (aligned) {
                align = keep(ops.prepare(MoeKernels.ALIGN_BLOCKS, MoeKernels.alignGlobal(nTiles, local), group,
                        MoeKernels.alignBlocksArgs(1, k, nExpert, nTiles, counts, offsets, rowToken, rowSlot,
                                gemmToken, gemmSlot)));
            }
        } catch (RuntimeException e) {
            throw Closeables.close(this, e);
        }
    }

    private void prepareMultiGroup(GpuOps ops, GpuScratch scratch, String prefix, int maxRows, GpuBuffer ids) {
        int chunk = MoeKernels.BUCKET_CHUNK;
        GpuBuffer hist = scratch.temp(prefix + ".bucket_hist", MoeKernels.bucketHistInts(maxRows, nExpert, chunk))
                .buffer();
        long[] group = {local};
        long[] groups = {(long) MoeKernels.bucketGroups(maxRows, chunk) * local};
        count = keep(ops.prepare(MoeKernels.BUCKET_COUNT, groups, group,
                MoeKernels.bucketCountArgs(1, k, nExpert, chunk, ids, hist)));
        scan = keep(ops.prepare(MoeKernels.BUCKET_SCAN, group, group,
                MoeKernels.bucketScanArgs(1, k, nExpert, chunk, hist, counts, offsets)));
        scatter = keep(ops.prepare(MoeKernels.BUCKET_SCATTER, groups, group,
                MoeKernels.bucketScatterArgs(1, k, nExpert, chunk, ids, hist, rowToken, rowSlot)));
    }

    private static boolean multiGroupSort(LlmKernels kernels) {
        return kernels.has(MoeKernels.BUCKET_COUNT) && kernels.has(MoeKernels.BUCKET_SCAN)
                && kernels.has(MoeKernels.BUCKET_SCATTER);
    }

    private static int pairs(int tokens, int k) {
        if (tokens < 1 || k < 1) {
            throw new IllegalArgumentException("expert rows need tokens >= 1 and k >= 1, got " + tokens + ", " + k);
        }
        return Math.toIntExact((long) tokens * k);
    }

    private PreparedDispatch keep(PreparedDispatch d) {
        prepared.add(d);
        return d;
    }

    /** Sets the token count of the next {@link #dispatch()}s; a no-op when unchanged. */
    public void update(int nTokens) {
        if (nTokens == lastTokens) {
            return;
        }
        int rows = pairs(nTokens, k);
        bucketize.setInt(MoeKernels.ARG_N_TOKENS, nTokens);
        if (count != null) {
            long[] groups = {(long) MoeKernels.bucketGroups(rows, MoeKernels.BUCKET_CHUNK) * local};
            count.setInt(MoeKernels.ARG_N_TOKENS, nTokens);
            count.setGlobal(groups);
            scan.setInt(MoeKernels.ARG_N_TOKENS, nTokens);
            scatter.setInt(MoeKernels.ARG_N_TOKENS, nTokens);
            scatter.setGlobal(groups);
        }
        if (align != null) {
            align.setInt(MoeKernels.ARG_N_TOKENS, nTokens);
        }
        lastTokens = nTokens;
    }

    /** Enqueues the sort, then the alignment when built aligned, for the token count of the last {@link #update}. */
    public void dispatch() {
        if (count != null && lastTokens * k > MoeKernels.BUCKET_CHUNK) {
            count.dispatch();
            scan.dispatch();
            scatter.dispatch();
        } else {
            bucketize.dispatch();
        }
        if (align != null) {
            align.dispatch();
        }
    }

    /** Whether the steps write the aligned rows ({@link #gemmToken()}, {@link #gemmSlot()}). */
    public boolean aligned() {
        return align != null;
    }

    /**
     * Whether steps of more than {@link MoeKernels#BUCKET_CHUNK} pairs sort on the multi-work-group path (the
     * capacity exceeds one chunk and the device provides its three kernels).
     */
    public boolean multiGroupSort() {
        return count != null;
    }

    /** Bind-time tile bound of the aligned rows ({@link MoeKernels#groupedTiles}); 0 when not aligned. */
    public int nTiles() {
        return nTiles;
    }

    public GpuBuffer counts() {
        return counts;
    }

    public GpuBuffer offsets() {
        return offsets;
    }

    /** Pair index of each sorted row. */
    public GpuBuffer rowToken() {
        return rowToken;
    }

    /** Expert of each sorted row, -1 for an out-of-range id. */
    public GpuBuffer rowSlot() {
        return rowSlot;
    }

    /** Pair of each aligned position, -1 for a filler; null when not aligned. */
    public GpuBuffer gemmToken() {
        return gemmToken;
    }

    /** Expert of each aligned position, -1 for a filler or an out-of-range id; null when not aligned. */
    public GpuBuffer gemmSlot() {
        return gemmSlot;
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            throw failure;
        }
    }

    // ================================================================== kernel choice

    /**
     * Whether the backend can lay {@code maxRows} pairs over {@code nExpert} experts out for the grouped GEMM: the
     * align kernel exists and the grouped GEMM's grid ({@code ceil(maxOutRows / 64)} x the tile bound) fits the
     * device's work-group counts.
     */
    public static boolean alignUsable(LlmKernels kernels, int maxRows, int nExpert, long maxOutRows) {
        if (!kernels.has(MoeKernels.ALIGN_BLOCKS) || nExpert > MoeKernels.MAX_EXPERTS) {
            return false;
        }
        long[] maxGroups = kernels.caps().getMaxWorkGroupCount();
        if (maxGroups.length >= 2 && maxGroups[1] > 0 && MoeKernels.groupedTiles(maxRows, nExpert) > maxGroups[1]) {
            return false;
        }
        long rowTiles = (maxOutRows + TiledGemmPlan.TILE_ROWS - 1) / TiledGemmPlan.TILE_ROWS;
        return maxGroups.length < 1 || maxGroups[0] <= 0 || rowTiles <= maxGroups[0];
    }

    /**
     * Whether an expert tensor of {@code type} with reduction length {@code inDim} can run the grouped GEMM: the
     * type has the tiled and grouped kernels, the device runs the tiled work-group, and {@code inDim} is a multiple
     * of the type's block size.
     */
    public static boolean groupedUsable(LlmKernels kernels, GGMLType type, int inDim) {
        return groupedUsable(kernels, type, inDim, false);
    }

    /**
     * {@link #groupedUsable(LlmKernels, GGMLType, int)} for the direct form ({@code mapped} false,
     * {@code {T}_GroupedGemm}) or the expert-cache form over a bank and its map ({@code mapped} true,
     * {@code {T}_GroupedGemmInd}).
     */
    public static boolean groupedUsable(LlmKernels kernels, GGMLType type, int inDim, boolean mapped) {
        if (!MoeKernels.supports(type) || !GemmKernels.hasTiled(type) || inDim % type.getBlockSize() != 0) {
            return false;
        }
        DeviceCapabilities caps = kernels.caps();
        return TiledGemmPlan.deviceSupport(type, caps) == null && kernels.has(scalarKernel(type, mapped));
    }

    /**
     * The grouped GEMM kernel of an expert type ({@link #groupedUsable} must hold for the same {@code mapped}):
     * the cooperative-matrix {@code {T}_GroupedGemmMma} ({@code {T}_GroupedGemmMmaInd} when {@code mapped}) on a
     * CUDA or Vulkan device with tensor cores ({@link DeviceFeature#COOPERATIVE_MATRIX}: CUDA sm_80+, or Vulkan's
     * fp16 x fp16 -> fp32 matrices with full subgroups) whose backend provides it and prepares it for the
     * {@link MoeKernels#groupedGemmLocal} work-group ({@link GpuOps#optionalReady}), else the scalar
     * {@code {T}_GroupedGemm} ({@code {T}_GroupedGemmInd}). Both forms take the same arguments and grid.
     */
    public static String groupedKernel(GpuOps ops, GGMLType type, boolean mapped) {
        Objects.requireNonNull(ops, "ops");
        if (tensorCores(ops.kernels()) && MoeKernels.hasGroupedGemmMma(type)) {
            String mma = mapped ? MoeKernels.groupedGemmMmaInd(type) : MoeKernels.groupedGemmMma(type);
            if (ops.kernels().has(mma) && ops.optionalReady(mma, MoeKernels.groupedGemmLocal())) {
                return mma;
            }
        }
        return scalarKernel(type, mapped);
    }

    private static String scalarKernel(GGMLType type, boolean mapped) {
        return mapped ? MoeKernels.groupedGemmInd(type) : MoeKernels.groupedGemm(type);
    }

    private static boolean tensorCores(LlmKernels kernels) {
        ComputeBackend.BackendType backend = kernels.backend().backendType();
        return (backend == ComputeBackend.BackendType.CUDA || backend == ComputeBackend.BackendType.VULKAN
                || backend == ComputeBackend.BackendType.HIP)
                && kernels.caps().hasFeature(DeviceFeature.COOPERATIVE_MATRIX);
    }

    @Override
    public String toString() {
        return "GpuGroupedRows{k=" + k + ", " + nExpert + " experts, tiles=" + nTiles
                + (count != null ? ", multi-group sort" : "") + '}';
    }
}
