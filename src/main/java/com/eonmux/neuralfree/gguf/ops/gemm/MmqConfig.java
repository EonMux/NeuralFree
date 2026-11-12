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
 * therefore {@code tileRows / rowsPerWorkItem() * TOKEN_LANES}. The packed-dot and RDNA3 tiles keep
 * the four rows per work-item of the original kernel (256 work-items); the CDNA 128-row tile and the
 * CUDA 128 x 128 tensor tile take eight rows per work-item, so their work-groups are also 256
 * work-items (on the tensor tile eight warps of two m16 row minitiles by 64 tokens, llama.cpp's
 * 256-thread block). The small-token twins of a tile take {@link #twinRowsPerWorkItem()} rows per
 * work-item: four on the tensor tile, whose twins keep the packed-dot family's 64-row tile.</p>
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
    /** int8 tensor cores ({@code mma.m16n8k32} / {@code m16n8k16} s8), 128 x 128 tile over 32-deep
     *  stages (llama.cpp's widest I/J table row) at eight rows per work-item (256 work-items per
     *  workgroup), with four-row twins: CUDA sm_80+. */
    public static final MmqConfig INT8_TENSOR = new MmqConfig("int8-tensor", 32, 128, 128, 8, 4);
    /**
     * The small-token tensor tile of CUDA sm_80+: the {@link #INT8_TENSOR} tile's 128 rows, 32-deep stages,
     * eight rows per work-item and 256 work-items over 16 tokens, llama.cpp's {@code mmq_x = 16} row at
     * {@code mmq_y = 128}. Its only kernels are the stream-k form {@code <T>_MatMul_MMQ_StreamK_S} and its fixup
     * ({@link StreamKGemmPlan#ofSmall}), the decode batches' MMQ form; it is no device tile.
     */
    public static final MmqConfig INT8_TENSOR_SMALL = new MmqConfig("int8-tensor-small", 32, 128, 16, 8, 4);
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

    /** Chunks the CUDA tensor-core tile stages per barrier ({@code NF_MMQ_TC_DEPTH}). */
    public static final int TENSOR_CHUNKS_PER_STAGE = 2;
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
    /**
     * Work-items per workgroup of the small-token-tile twins: small batches launch few blocks, and
     * the measured RTX 5080 Laptop 9-token steps regressed 12-38% when the 128-row tensor tile's
     * twin widened its workgroup to 512 &mdash; the 512-item blocks cap at one per SM (the kernel's
     * 896-thread occupancy limit) and the wave loses its second block. The twins keep the
     * 256-item workgroups of the packed-dot family.
     */
    public static final int TWIN_WORK_GROUP = 256;
    /**
     * Widest token tile of the small-token-tile twins ({@code NF_MMQ_TN} of the S64 forms): the
     * measured RTX 5080 Laptop mid batches. Up to {@link #TOKEN_LANES} tokens the 16-token twin
     * wins (one weight pass over one token tile, the band the counted matvecs surround); over
     * 17..64 tokens the 64-token twin wins the head and the FFN-out shapes by 29-42% over the
     * 128-token device tile (n=48: 2063 vs 2923 us on the 151936x2560 head, 168.8 vs 261.0 on
     * 9728x2560) because the 128-wide tile pads up to 80 dead token columns and its 512-item
     * workgroups cap at one block per SM; above 64 the device tile has no dead columns left worth
     * a twin (n=96: the 128-wide tile wins the head back, 3191.6 vs 4124.5 us).
     */
    public static final int TWIN_MAX_TOKENS = 64;
    /**
     * Largest launch that keeps the 16-token twin on a small-K reduction (K below {@link
     * SplitKGemmPlan#MIN_SPLIT_K}): at or below it the 64-token tile pads a quarter of its token
     * columns or more, on a grid too small to hide the waste; from 49 tokens the tile is more
     * than three-quarters full and its single weight pass wins (see {@link #twinTokens(int, int)}).
     */
    public static final int SMALL_K_TWIN_MAX_TOKENS = TWIN_MAX_TOKENS * 3 / 4;
    /** System property that forces one of the named tiles on the runtime-compiled OpenCL kernels. */
    public static final String TILE_PROPERTY = "neuralfree.mmq.tile";

    private static final List<MmqConfig> NAMED = Collections.unmodifiableList(
            Arrays.asList(PACKED_DOT, INT8_TENSOR, RDNA3, CDNA));

    private final String name;
    private final int stageDepth;
    private final int tileRows;
    private final int tileTokens;
    private final int rowsPerWorkItem;
    private final int twinRowsPerWorkItem;

    private MmqConfig(String name, int stageDepth, int tileRows, int tileTokens, int rowsPerWorkItem) {
        this(name, stageDepth, tileRows, tileTokens, rowsPerWorkItem, rowsPerWorkItem);
    }

    private MmqConfig(String name, int stageDepth, int tileRows, int tileTokens, int rowsPerWorkItem,
                      int twinRowsPerWorkItem) {
        this.name = name;
        this.stageDepth = stageDepth;
        this.tileRows = tileRows;
        this.tileTokens = tileTokens;
        this.rowsPerWorkItem = rowsPerWorkItem;
        this.twinRowsPerWorkItem = twinRowsPerWorkItem;
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
     * stage depth and twin rows per work-item ({@link #twinRowsPerWorkItem()}) with a 16-token tile
     * and a row tile that keeps the twin's workgroup at {@link #TWIN_WORK_GROUP} work-items (the
     * tensor tile's twins and the packed-dot family's 64 rows), the tile
     * of the packed-dot path and what llama.cpp picks per launch for small batches
     * (ggml_cuda_mmq_get_J_best). Only a device tile wider than 16 tokens has a twin;
     * {@link MmqGemmPlan} launches the twin with this tile's geometry.
     *
     * @throws IllegalArgumentException when {@code device} is null
     */
    public static MmqConfig smallTile(MmqConfig device) {
        return smallTile(device, TOKEN_LANES);
    }

    /**
     * The twin tile of the given token width: as {@link #smallTile(MmqConfig)} over
     * {@code tokens} tokens ({@code <T>_MatMul_MMQ_S} at 16, the S64 forms at {@link
     * #TWIN_MAX_TOKENS}). Only a device tile wider than {@code tokens} has that twin, so a
     * 64-token device tile serves the 17..64 band with its plain tile and no twin.
     *
     * @throws IllegalArgumentException when {@code device} is null, {@code tokens} is not a twin
     *                                  width (16 or 64) or the device tile is not wider than it
     */
    public static MmqConfig smallTile(MmqConfig device, int tokens) {
        Objects.requireNonNull(device, "device");
        if (tokens != TOKEN_LANES && tokens != TWIN_MAX_TOKENS) {
            throw new IllegalArgumentException("MMQ twin tiles come in " + TOKEN_LANES + " and "
                    + TWIN_MAX_TOKENS + " tokens, got " + tokens);
        }
        if (device.tileTokens <= tokens) {
            throw new IllegalArgumentException("MMQ tile " + device.name + " is at most " + tokens
                    + " tokens wide; it has no " + tokens + "-token twin");
        }
        int twinRows = Math.min(device.tileRows, TWIN_WORK_GROUP / TOKEN_LANES * device.twinRowsPerWorkItem);
        return of(device.stageDepth, twinRows, tokens, device.twinRowsPerWorkItem);
    }

    /**
     * The token width of the twin tile that serves a launch of {@code tokens} tokens over a
     * reduction of {@code k}: the 16-token tile at up to 16 tokens (the counted-matvec band) and
     * the 64-token tile above, llama.cpp's per-launch J ladder (ggml_cuda_mmq_get_J_best) — with
     * one measured exception: on a small-K reduction (K below {@link SplitKGemmPlan#MIN_SPLIT_K},
     * the weights small enough to sit in L2 and no split-K to fill the grid) launches up to
     * {@link #SMALL_K_TWIN_MAX_TOKENS} keep the 16-token twin, whose filled grid beats the
     * 64-token tile's dead columns (the Qwen3 0.6B Q8_0 layers, all K 1024/2048, measured 2701 and
     * 3653 t/s at pp24 and pp32 on the 16-token twin against 2171 and 2875 on the 64-token one,
     * while the K-2560 layers of the 4B model prefer the 64-token tile everywhere in the band).
     * The caller builds the twin with {@link #smallTile(MmqConfig, int)} and falls back to the
     * plain tile when the device tile is not wider than this.
     */
    public static int twinTokens(int tokens, int k) {
        if (tokens <= TOKEN_LANES
                || (k < SplitKGemmPlan.MIN_SPLIT_K && tokens <= SMALL_K_TWIN_MAX_TOKENS)) {
            return TOKEN_LANES;
        }
        return TWIN_MAX_TOKENS;
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

    /**
     * Weight rows one work-item of the small-token twins covers ({@code NF_MMQ_TWIN_RM}, or
     * {@code NF_MMQ_RM} when the tile defines none): four on {@link #INT8_TENSOR}, whose twins keep the
     * one-minitile-per-warp mma layout of the 64-row tile, {@link #rowsPerWorkItem()} elsewhere.
     */
    public int twinRowsPerWorkItem() {
        return twinRowsPerWorkItem;
    }

    /** Work-items per workgroup ({@code NF_MMQ_WG}): {@code tileRows / rowsPerWorkItem() * TOKEN_LANES}. */
    public int workGroupSize() {
        return tileRows / rowsPerWorkItem * TOKEN_LANES;
    }

    /**
     * The preprocessor defines that compile the MMQ kernels with this tile, or the empty string when
     * they already are the tile of the dialect's default config: {@code NF_MMQ_TM}, {@code NF_MMQ_TN},
     * {@code NF_MMQ_TK}, {@code NF_MMQ_RM} (when not the default's) and {@code NF_MMQ_TWIN_RM} (when
     * the twins take other rows per work-item than the tile), which
     * kernels/llm/common/matmul_mmq_config.h takes when defined.
     *
     * @param tensorPass whether the pass compiles the int8 tensor path (CUDA sm_80+), whose default
     *                   tile is {@link #INT8_TENSOR}; the packed-dot default is {@link #PACKED_DOT}
     * @throws IllegalArgumentException when a tensor pass is asked for a tile other than
     *                                  {@link #INT8_TENSOR}'s 32-deep stages, multiple-of-four rows
     *                                  per work-item and four-row twins: the mma fragments map the
     *                                  tile rows to warps of {@code rowsPerWorkItem / 4} m16
     *                                  minitiles, and the twins' to one minitile per warp
     */
    public String kernelDefines(boolean tensorPass) {
        if (tensorPass && (stageDepth != INT8_TENSOR.stageDepth || rowsPerWorkItem % 4 != 0
                || twinRowsPerWorkItem != INT8_TENSOR.twinRowsPerWorkItem)) {
            throw new IllegalArgumentException("MMQ tile " + name + " stages " + stageDepth + " elements over "
                    + rowsPerWorkItem + " rows per work-item (" + twinRowsPerWorkItem + " on its twins); the"
                    + " int8 tensor path needs 32-deep stages, a multiple of 4 rows per work-item and "
                    + INT8_TENSOR.twinRowsPerWorkItem + "-row twins");
        }
        MmqConfig base = tensorPass ? INT8_TENSOR : PACKED_DOT;
        if (stageDepth == base.stageDepth && tileRows == base.tileRows && tileTokens == base.tileTokens
                && rowsPerWorkItem == base.rowsPerWorkItem && twinRowsPerWorkItem == base.twinRowsPerWorkItem) {
            return "";
        }
        StringBuilder defines = new StringBuilder()
                .append("-DNF_MMQ_TM=").append(tileRows)
                .append(" -DNF_MMQ_TN=").append(tileTokens)
                .append(" -DNF_MMQ_TK=").append(stageDepth);
        if (rowsPerWorkItem != base.rowsPerWorkItem) {
            defines.append(" -DNF_MMQ_RM=").append(rowsPerWorkItem);
        }
        if (twinRowsPerWorkItem != rowsPerWorkItem) {
            defines.append(" -DNF_MMQ_TWIN_RM=").append(twinRowsPerWorkItem);
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
     * Local memory of one workgroup, as an upper bound over the dialects that compile this tile:
     * the pipelined stage holds three copies of the weight-side tiles, one per in-flight chunk
     * ({@code NF_MMQ_PIPE_BUF}) &mdash; the staged weight codes ({@code 3 * TM * TK}, four codes
     * per int on the 32-deep tensor path whose kernels only the CUDA dialect builds, one code per
     * int on the packed dialects), three {@code TM * TK/16} tiles of scale floats, offset floats
     * and per-half weight code sums of the asymmetric int8 form &mdash; and one copy of the tokens'
     * {@code TN} staged records ({@code TK/4 + 2 + TK/16} units: the code ints the B fragments
     * read, the chunk scales, the code sums and the zero points); the activations themselves are
     * read from the staging buffer. A 32-deep tile also bounds the tensor-core plain kernels of the
     * CUDA dialect (common/matmul_mmq_family.h, NF_MMQ_TC_KERNEL_BODY), which stage
     * {@value #TENSOR_CHUNKS_PER_STAGE} chunks per barrier in two alternating groups: four copies of the weight
     * codes ({@code TM * TK / 4} ints) and, at most, of the scale and offset tiles ({@code TM * TK / 16}
     * floats each), and four of the record tile ({@code TN} records), whose loads run one group ahead of the
     * combine. The bound takes both forms at every 32-deep geometry, so a twin tile's, whose kernels run only
     * the pipelined stage, can exceed what they use.
     */
    public long sharedBytes() {
        long halves = stageDepth / 16;
        long recordTile = (long) tileTokens * (stageDepth / 4 + 2 + halves);
        boolean tensor = stageDepth > PACKED_DOT.stageDepth;
        long codes = 3L * tileRows * stageDepth / (tensor ? 4 : 1);
        long pipelined = codes + 9L * tileRows * halves + recordTile;
        if (!tensor) {
            return 4L * pipelined;
        }
        long copies = 2L * TENSOR_CHUNKS_PER_STAGE;
        long tensorTile = copies * tileRows * stageDepth / 4 + 2 * copies * tileRows * halves + copies * recordTile;
        return 4L * Math.max(pipelined, tensorTile);
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
                && rowsPerWorkItem == that.rowsPerWorkItem && twinRowsPerWorkItem == that.twinRowsPerWorkItem;
    }

    @Override
    public int hashCode() {
        return Objects.hash(stageDepth, tileRows, tileTokens, rowsPerWorkItem, twinRowsPerWorkItem);
    }

    @Override
    public String toString() {
        return "MmqConfig{" + name + ", TK=" + stageDepth + ", TM=" + tileRows + ", TN=" + tileTokens
                + ", RM=" + rowsPerWorkItem + ", twin RM=" + twinRowsPerWorkItem + ", WG=" + workGroupSize() + '}';
    }
}
