package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.hip.HipGfxTarget;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Per-architecture tile config of the int8 MMQ GEMM (the counterpart of llama.cpp's
 * {@code mmq-config} tables): the stage depth, the row tile, the token tile and the weight rows one
 * work-item covers, which the {@code <T>_MatMul_MMQ} kernel is compiled with on a device
 * ({@code NF_MMQ_TK}, {@code NF_MMQ_TM}, {@code NF_MMQ_TN} and {@code NF_MMQ_RM} in
 * kernels/llm/common/matmul_mmq_config.h). The host plans the launch grid with the config of the
 * device the kernel runs on, so both sides agree on the tile.
 *
 * <p>The configs follow llama.cpp's {@code ggml-cuda/mmq-config-*.cuh} tables. A workgroup of
 * {@link #workGroupSize()} work-items computes a {@code tileRows} x {@code tileTokens} tile: every
 * work-item holds {@link #rowsPerWorkItem()} weight rows for one of {@link #TOKEN_LANES} token
 * lanes, and the tile tokens are covered in groups of {@link #TOKEN_LANES}. The work-group size is
 * therefore {@code tileRows / rowsPerWorkItem() * TOKEN_LANES}. A 64-token tile keeps the four rows
 * per work-item of the original kernel (256 work-items); the CDNA 128-row tile takes eight rows per
 * work-item so its work-group is also 256 work-items.</p>
 *
 * <p>Instances are immutable values; {@link #forDevice} and {@link #forHipTarget} select the tile of
 * a device and {@link #of} builds one from validated tile sizes.</p>
 */
public final class MmqConfig {

    /**
     * dp4a (CUDA sm_61+) or scalar int dots, 64 x 16 tile, 16-deep stages: CUDA below sm_80, OpenCL,
     * Vulkan and the HIP targets without a ported MMQ table (llama.cpp's packed-dot families).
     */
    public static final MmqConfig PACKED_DOT = new MmqConfig("packed-dot", 16, 64, 16, 4);
    /** int8 tensor cores ({@code mma.m16n8k32} / {@code m16n8k16} s8), 64 x 64 tile, 32-deep stages: CUDA sm_80+. */
    public static final MmqConfig INT8_TENSOR = new MmqConfig("int8-tensor", 32, 64, 64, 4);
    /**
     * RDNA3 (gfx1100&ndash;gfx1103): llama.cpp's {@code mmq-config-rdna3.cuh} widest tile, 64 rows x
     * 128 tokens over 16-deep stages. It is the table's non-fallback J=128 row of every int8 MMQ type
     * except Q2_K, whose widest row is 128 rows; the family tile keeps one shape for all types.
     */
    public static final MmqConfig RDNA3 = new MmqConfig("rdna3", 16, 64, 128, 4);
    /**
     * CDNA (gfx908, gfx90a, gfx942, gfx950, gfx1250): llama.cpp's {@code mmq-config-cdna.cuh} tile,
     * 128 rows x 64 tokens over 16-deep stages (I 128, J 64), the same row for every MMQ weight type;
     * eight rows per work-item keep its work-group at 256 work-items.
     */
    public static final MmqConfig CDNA = new MmqConfig("cdna", 16, 128, 64, 8);

    /** Smallest CUDA compute capability of the int8 tensor path (Ampere). */
    public static final int TENSOR_MIN_COMPUTE_CAPABILITY = 80;
    /** Largest token tile over the configs ({@code NF_MMQ_TN}); bounds the int tile indices of N. */
    public static final int MAX_TILE_TOKENS = 128;
    /** Largest row tile over the configs ({@code NF_MMQ_TM}); bounds the int tile indices of M. */
    public static final int MAX_TILE_ROWS = 128;
    /** Largest stage depth over the configs ({@code NF_MMQ_TK}); bounds the int tile indices of K. */
    public static final int MAX_STAGE_DEPTH = 32;
    /** Default weight rows one work-item covers ({@code NF_MMQ_RM}) on the 64-row tiles. */
    public static final int ROWS_PER_WORK_ITEM = 4;
    /** Largest weight rows one work-item covers ({@code NF_MMQ_RM}). */
    public static final int MAX_ROWS_PER_WORK_ITEM = 8;
    /** Token lanes of a tile, the work-items that share a row-tile column ({@code NF_MMQ_TN / NF_MMQ_RN}). */
    public static final int TOKEN_LANES = 16;
    /** System property that forces one of the named tiles on the runtime-compiled OpenCL kernels. */
    public static final String TILE_PROPERTY = "neuralfree.mmq.tile";

    private static final List<MmqConfig> NAMED = Collections.unmodifiableList(
            Arrays.asList(PACKED_DOT, INT8_TENSOR, RDNA3, CDNA));

    private final String name;
    private final int stageDepth;
    private final int tileRows;
    private final int tileTokens;
    private final int rowsPerWorkItem;

    private MmqConfig(String name, int stageDepth, int tileRows, int tileTokens, int rowsPerWorkItem) {
        this.name = name;
        this.stageDepth = stageDepth;
        this.tileRows = tileRows;
        this.tileTokens = tileTokens;
        this.rowsPerWorkItem = rowsPerWorkItem;
    }

    /**
     * The config the MMQ kernels of a device are built with: {@link #INT8_TENSOR} on CUDA devices of
     * compute capability {@value #TENSOR_MIN_COMPUTE_CAPABILITY} and above, the family tile of a HIP
     * target ({@link #forHipTarget}), and {@link #PACKED_DOT} on OpenCL, Vulkan, CUDA below that
     * capability and the HIP families without a ported table.
     *
     * <p>OpenCL compiles its kernels from source at run time, so the OpenCL tile can be forced with
     * the {@value #TILE_PROPERTY} system property (one of {@code packed-dot}, {@code int8-tensor},
     * {@code rdna3}, {@code cdna}); the property lets the RDNA3 and CDNA tiles run on an OpenCL
     * device. CUDA and HIP tiles are fixed at build time and ignore it.</p>
     *
     * @param computeCapability NVIDIA {@code major * 10 + minor} or the AMD {@code 0x1000000 | gfx}
     *                          value of the device; ignored on OpenCL and Vulkan
     * @throws IllegalArgumentException when {@code backend} is HIP and {@code computeCapability} is
     *                                  not an AMD value, or when the OpenCL override names no tile
     */
    public static MmqConfig forDevice(ComputeBackend.BackendType backend, int computeCapability) {
        Objects.requireNonNull(backend, "backend");
        if (backend == ComputeBackend.BackendType.OPENCL) {
            MmqConfig forced = tileOverride();
            return forced != null ? forced : PACKED_DOT;
        }
        if (backend == ComputeBackend.BackendType.CUDA) {
            return computeCapability >= TENSOR_MIN_COMPUTE_CAPABILITY ? INT8_TENSOR : PACKED_DOT;
        }
        if (backend == ComputeBackend.BackendType.HIP) {
            return forHipFamily(HipGfxTarget.familyOfComputeCapability(computeCapability));
        }
        return PACKED_DOT;
    }

    /**
     * The MMQ tile of a HIP processor family: {@link #CDNA} on Instinct accelerators and
     * {@link #RDNA3} on the RDNA3 family (gfx1100&ndash;gfx1103), {@link #PACKED_DOT} on the other
     * families. RDNA3.5, RDNA4-M and RDNA4 have their own llama.cpp tables
     * ({@code mmq-config-rdna3-5.cuh}, {@code mmq-config-rdna4.cuh}) and keep the packed-dot tile
     * until those are ported.
     *
     * @throws IllegalArgumentException when {@code target} is null
     */
    public static MmqConfig forHipTarget(HipGfxTarget target) {
        Objects.requireNonNull(target, "target");
        return forHipFamily(target.family());
    }

    /** {@link #forHipTarget} for a family, the form the kernel's {@code __gfxNNNN__} blocks mirror. */
    public static MmqConfig forHipFamily(HipGfxTarget.Family family) {
        Objects.requireNonNull(family, "family");
        if (family == HipGfxTarget.Family.CDNA) {
            return CDNA;
        }
        if (family == HipGfxTarget.Family.RDNA3) {
            return RDNA3;
        }
        return PACKED_DOT;
    }

    /**
     * A config of the given tile sizes with {@link #ROWS_PER_WORK_ITEM} rows per work-item,
     * validated against the kernel layout.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static MmqConfig of(int stageDepth, int tileRows, int tileTokens) {
        return of(stageDepth, tileRows, tileTokens, ROWS_PER_WORK_ITEM);
    }

    /**
     * A config of the given tile sizes, validated against the kernel layout: the row tile is a
     * multiple of {@code rowsPerWorkItem}, the token tile a multiple of {@link #TOKEN_LANES} and at
     * most {@link #MAX_TILE_TOKENS}, the stage depth 16 or 32, the work-group size at most 1024
     * work-items and {@code rowsPerWorkItem} in [1, {@link #MAX_ROWS_PER_WORK_ITEM}].
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static MmqConfig of(int stageDepth, int tileRows, int tileTokens, int rowsPerWorkItem) {
        if (stageDepth != 16 && stageDepth != 32) {
            throw new IllegalArgumentException("MMQ stage depth must be 16 or 32, got " + stageDepth);
        }
        if (rowsPerWorkItem < 1 || rowsPerWorkItem > MAX_ROWS_PER_WORK_ITEM) {
            throw new IllegalArgumentException("MMQ rows per work-item must be in [1, "
                    + MAX_ROWS_PER_WORK_ITEM + "], got " + rowsPerWorkItem);
        }
        if (tileRows < rowsPerWorkItem || tileRows > MAX_TILE_ROWS || tileRows % rowsPerWorkItem != 0) {
            throw new IllegalArgumentException("MMQ row tile must be a positive multiple of "
                    + rowsPerWorkItem + " rows per work-item at most " + MAX_TILE_ROWS + ", got " + tileRows);
        }
        if (tileTokens < TOKEN_LANES || tileTokens > MAX_TILE_TOKENS || tileTokens % TOKEN_LANES != 0) {
            throw new IllegalArgumentException("MMQ token tile must be a positive multiple of "
                    + TOKEN_LANES + " at most " + MAX_TILE_TOKENS + ", got " + tileTokens);
        }
        int workGroup = tileRows / rowsPerWorkItem * TOKEN_LANES;
        if (workGroup > 1024) {
            throw new IllegalArgumentException("MMQ tile " + tileRows + " x " + tileTokens + " needs "
                    + workGroup + " work-items per workgroup, more than the 1024 a device may run");
        }
        return new MmqConfig("tile-" + tileRows + "x" + tileTokens + "x" + stageDepth, stageDepth, tileRows,
                tileTokens, rowsPerWorkItem);
    }

    /**
     * The tile of the small-token-tile GEMM twins ({@code <T>_MatMul_MMQ_S}): the device config's
     * stage depth, row tile and rows per work-item with a 16-token tile, the tile of the packed-dot
     * path and what llama.cpp picks per launch for small batches (ggml_cuda_mmq_get_J_best). Only a
     * device tile wider than 16 tokens has a twin; {@link MmqGemmPlan} launches the twin with this
     * tile's geometry.
     *
     * @throws IllegalArgumentException when {@code device} is null
     */
    public static MmqConfig smallTile(MmqConfig device) {
        Objects.requireNonNull(device, "device");
        if (device.tileTokens <= TOKEN_LANES) {
            throw new IllegalArgumentException("MMQ tile " + device.name + " is already the "
                    + TOKEN_LANES + "-token tile; it has no small-token-tile twin");
        }
        return of(device.stageDepth, device.tileRows, TOKEN_LANES, device.rowsPerWorkItem);
    }

    /**
     * The tile forced by the {@value #TILE_PROPERTY} system property, or null when it is unset.
     *
     * @throws IllegalArgumentException when the property names no config
     */
    public static MmqConfig tileOverride() {
        String value = System.getProperty(TILE_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return null;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        for (MmqConfig config : NAMED) {
            if (config.name.equals(normalized)) {
                return config;
            }
        }
        throw new IllegalArgumentException("unsupported " + TILE_PROPERTY + " '" + value + "'; expected one of "
                + names());
    }

    /** The names of the configs the override accepts, comma-separated. */
    public static String names() {
        StringBuilder sb = new StringBuilder();
        for (MmqConfig config : NAMED) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(config.name);
        }
        return sb.toString();
    }

    /** The name of the config, for diagnostics and the OpenCL program key. */
    public String name() {
        return name;
    }

    /** Reduction depth of one stage, and the activation chunk of one staged 8-bit scale ({@code NF_MMQ_TK}). */
    public int stageDepth() {
        return stageDepth;
    }

    /** Weight rows per workgroup tile ({@code NF_MMQ_TM}). */
    public int tileRows() {
        return tileRows;
    }

    /** Tokens per workgroup tile ({@code NF_MMQ_TN}), sized per architecture as llama.cpp picks its token tiles. */
    public int tileTokens() {
        return tileTokens;
    }

    /** Weight rows one work-item covers ({@code NF_MMQ_RM}). */
    public int rowsPerWorkItem() {
        return rowsPerWorkItem;
    }

    /** Work-items per workgroup ({@code NF_MMQ_WG}): {@code tileRows / rowsPerWorkItem() * TOKEN_LANES}. */
    public int workGroupSize() {
        return tileRows / rowsPerWorkItem * TOKEN_LANES;
    }

    /**
     * The preprocessor defines that compile the MMQ kernels with this tile, or the empty string when
     * they already are the tile of the dialect's default config: {@code NF_MMQ_TM}, {@code NF_MMQ_TN},
     * {@code NF_MMQ_TK} and {@code NF_MMQ_RM} (when not 4), which
     * kernels/llm/common/matmul_mmq_config.h takes when defined.
     *
     * @param tensorPass whether the pass compiles the int8 tensor path (CUDA sm_80+), whose default
     *                   tile is {@link #INT8_TENSOR}; the packed-dot default is {@link #PACKED_DOT}
     * @throws IllegalArgumentException when a tensor pass is asked for a tile whose rows per
     *                                  work-item is not {@link #ROWS_PER_WORK_ITEM}: the mma
     *                                  fragments map the tile rows to warps, not work-items
     */
    public String kernelDefines(boolean tensorPass) {
        if (tensorPass && rowsPerWorkItem != ROWS_PER_WORK_ITEM) {
            throw new IllegalArgumentException("MMQ tile " + name + " covers " + rowsPerWorkItem
                    + " rows per work-item; the int8 tensor path needs " + ROWS_PER_WORK_ITEM);
        }
        int defaultDepth = tensorPass ? INT8_TENSOR.stageDepth : PACKED_DOT.stageDepth;
        int defaultTokens = tensorPass ? INT8_TENSOR.tileTokens : PACKED_DOT.tileTokens;
        if (stageDepth == defaultDepth && tileRows == PACKED_DOT.tileRows && tileTokens == defaultTokens
                && rowsPerWorkItem == ROWS_PER_WORK_ITEM) {
            return "";
        }
        StringBuilder defines = new StringBuilder()
                .append("-DNF_MMQ_TM=").append(tileRows)
                .append(" -DNF_MMQ_TN=").append(tileTokens)
                .append(" -DNF_MMQ_TK=").append(stageDepth);
        if (rowsPerWorkItem != ROWS_PER_WORK_ITEM) {
            defines.append(" -DNF_MMQ_RM=").append(rowsPerWorkItem);
        }
        return defines.toString();
    }

    /**
     * The program-key suffix of this tile: the empty string when {@link #kernelDefines} is empty,
     * otherwise the config name after {@code @}, so programs of different tiles never share a cache entry.
     */
    public String kernelKey(boolean tensorPass) {
        return kernelDefines(tensorPass).isEmpty() ? "" : "@" + name;
    }

    /** Chunk records of one staged activation row of {@code k} elements: {@code k / stageDepth}. */
    public int rowRecords(int k) {
        return k / stageDepth;
    }

    /**
     * 32-bit units of one staged chunk record ({@code NF_MMQ_RC}): {@code stageDepth / 4} code ints,
     * the chunk scale {@code d}, the {@code stageDepth / 16} code sums of the 16-element halves and
     * the asymmetric zero point (0 for every symmetric form).
     */
    public int recordUnits() {
        return stageDepth / 4 + 2 + stageDepth / 16;
    }

    /**
     * Local memory of one workgroup, as a float-sized upper bound over the dialects (the GLSL twin
     * stages codes as one int per code): the pipelined stage holds two copies of the weight-side
     * tiles, one per in-flight chunk &mdash; {@code 2 * TM * TK} staged weight codes, {@code 4 * TM *
     * TK/16} scale floats and {@code 2 * TM * TK/16} per-half weight code sums of the asymmetric
     * int8 form &mdash; and one copy of the tokens' {@code TN} chunk scales, {@code TN * TK/16} code
     * sums and {@code TN} zero points gathered from the staged records; the activations themselves
     * are read from the staging buffer.
     */
    public long sharedBytes() {
        long halves = stageDepth / 16;
        return 4L * (2L * tileRows * stageDepth + 6L * tileRows * halves
                + 2L * tileTokens + (long) tileTokens * halves);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MmqConfig)) {
            return false;
        }
        MmqConfig that = (MmqConfig) o;
        return stageDepth == that.stageDepth && tileRows == that.tileRows && tileTokens == that.tileTokens
                && rowsPerWorkItem == that.rowsPerWorkItem;
    }

    @Override
    public int hashCode() {
        return Objects.hash(stageDepth, tileRows, tileTokens, rowsPerWorkItem);
    }

    @Override
    public String toString() {
        return "MmqConfig{" + name + ", TK=" + stageDepth + ", TM=" + tileRows + ", TN=" + tileTokens
                + ", RM=" + rowsPerWorkItem + ", WG=" + workGroupSize() + '}';
    }
}
