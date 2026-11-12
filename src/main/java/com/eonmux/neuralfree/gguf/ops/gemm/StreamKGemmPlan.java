package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Objects;

/**
 * The stream-k launch of a staged int8 MMQ GEMM over the CUDA tensor tile ({@link MmqConfig#INT8_TENSOR}):
 * {@code <T>_MatMul_MMQ_StreamK} followed, when a tile is split, by {@code LLM_MMQ_StreamK_Fixup}
 * ({@code kernels/llm/matmul_mmq_stream.cl}), llama.cpp's stream-k decomposition of its MMQ grid
 * ({@code ggml-cuda/mmq.cuh}, {@code mul_mat_q} with {@code tmp_fixup} and {@code mul_mat_q_stream_k_fixup}).
 *
 * <p>The tensor tile keeps one 256-work-item block resident per compute unit (its kernels hold 166-248
 * registers per work-item, {@link #TENSOR_TILE_BLOCKS_PER_UNIT}), so a grid of {@code T} tiles runs in
 * {@code ceil(T / units)} waves and its last wave idles {@code units - T % units} compute units for a whole
 * tile time: the 4B layers' 80-tile projections (2560 rows over 512 tokens) run two waves for 1.33 waves of
 * work, the 304-tile ones six for 5.07. The stream-k kernel lays the grid's {@code T * K / 32} (tile, chunk)
 * steps out as one sequence, tile-major with the token tile fastest, and gives each of {@code units} blocks
 * an equal contiguous run of it; a block stores the tiles it runs whole and the tile its first segment
 * finishes, and writes the partial sums of its last segment, when that stops inside a tile, into its own
 * {@code TM x TN} plane of the fixup buffer, which the fixup adds into the tile afterwards. The fixup is a
 * separate launch, as in llama.cpp: summing the planes inside the stream-k kernel would add their loads to the
 * tile's register budget and slow its main loop by more than the extra launch costs.</p>
 *
 * <p>The plan follows llama.cpp's choice: a grid whose waves are already at least
 * {@link #PLAIN_GRID_EFFICIENCY_PERCENT} percent full keeps the plain kernel (the stream-k blocks would
 * gain less than the fixup costs), and so does a device that does not report its compute units. A tile
 * one block runs whole equals the plain kernel's output bit for bit; a split tile differs by the fp32
 * rounding of adding its partial sums, and its partition depends on the token count, so a batch-invariant
 * run keeps the plain kernel (the caller's decision, as for {@link SplitKGemmPlan}).</p>
 *
 * <p>The gate/up form ({@link #ofGateUp}, {@code <T>_MatMul_MMQ_StreamK_GateUp}) runs a dense FFN's gate and up
 * projections of one type and shape as one launch that writes {@code silu(gate . x) * (up . x)}, llama.cpp's
 * fused gate/up MMQ: each output tile of {@code TM / 2} rows runs the {@code TM}-row tensor tile over the gate
 * rows and the up rows of the same indices. It always runs stream-k over the resident blocks (it has no plain
 * grid), and replaces two stream-k launches and the {@code LLM_GLU} pass, which writes and reads back the
 * {@code 2 * n * n_ff} products. The activation
 * cannot be added onto, so a block keeps both the segment that finishes a split tile and the one that stops inside
 * a tile in planes of its own ({@link #planesPerBlock}), and {@code LLM_MMQ_StreamK_Fixup_GateUp} activates their
 * sums.</p>
 *
 * <p>The small-token form ({@link #ofSmall}, {@code <T>_MatMul_MMQ_StreamK_S} and {@code LLM_MMQ_StreamK_Fixup_S})
 * runs the same decomposition over the {@link MmqConfig#INT8_TENSOR_SMALL} tile of 128 rows by 16 tokens, the form
 * of the decode batches of up to 16 tokens (llama.cpp's stream-k {@code mul_mat_q} at {@code mmq_x = 16}), over
 * {@link #SMALL_TILE_BLOCKS_PER_UNIT} block per compute unit. The tile has no plain kernel: a grid whose waves are
 * full enough runs the stream-k kernel with one block per tile, each running its tile whole, and no fixup.</p>
 *
 * <p>Geometry: stream-k {@code global = {blocks * WG, 1}}, {@code local = {WG, 1}}; the fixup
 * {@code global = {blocks * WG, outputs / (WG * }{@link #FIXUP_ELEMENTS_PER_WORK_ITEM}{@code )}} over the same
 * block count, one work-group per slice of the {@code TM x TN} (gate/up: {@code TM / 2 x TN}) outputs of the
 * tile a block finishes.</p>
 */
public final class StreamKGemmPlan {

    /** Wave fill (percent of the last wave's compute units busy, llama.cpp's tiles_efficiency_percent) from
     *  which the plain grid is kept: llama.cpp runs its stream-k kernel as plain tiles there on NVIDIA. */
    public static final int PLAIN_GRID_EFFICIENCY_PERCENT = 90;
    /** Resident blocks per compute unit of the tensor tile's kernels: 256 work-items at 166-248 registers
     *  (sm_120 build) leave room for one block of 65536 registers. */
    public static final int TENSOR_TILE_BLOCKS_PER_UNIT = 1;
    /**
     * Resident blocks per compute unit the small-token tile's stream-k launch plans with ({@link #ofSmall}): one,
     * as for the device tile. Its kernels hold their working set in 123-152 registers per work-item on sm_120 and
     * 110-170 on sm_80 and sm_86, past the 128 a second block of 256 work-items leaves for most types, and capping
     * them at 128 spills the IQ1_S and IQ3_S kernels on sm_120 and the IQ2_XXS, IQ3_S, Q5_0 or Q5_1 kernels on
     * sm_80 and sm_86.
     */
    public static final int SMALL_TILE_BLOCKS_PER_UNIT = 1;
    /**
     * Compute units per output tile below which the small-token tile keeps the 16-token twin ({@link #ofSmall}): a
     * grid of fewer tiles than a quarter of the compute units (at most 14 tiles, 1792 rows, on 60 units) splits each
     * tile over several blocks, and its fixup pass then costs about what it saves; at 8 tiles (1024 rows) the
     * twin's split-K measured as fast as the stream-k form on the RTX 5080 Laptop, and from 16 tiles (2048 rows) up
     * the stream-k form measured 1.1-1.8x faster than the twin.
     */
    public static final int SMALL_TILE_MAX_UNITS_PER_TILE = 4;

    /** Tile outputs one fixup work-item adds ({@code NF_MMQ_SK_FIX_E}): a tile's outputs spread over
     *  {@code outputs / (WG * 4)} work-groups, so each split tile's planes are read with four loads in flight per
     *  work-item rather than walked by one work-group as a long dependent chain. */
    public static final int FIXUP_ELEMENTS_PER_WORK_ITEM = 4;
    /** {@code <T>_MatMul_MMQ_StreamK}: the fixup planes, {@link #planesPerBlock} {@code TM x TN} planes per
     *  block. */
    public static final int ARG_FIXUP = GemmKernels.MATMUL_TILED_ARG_COUNT;
    /** Number of {@code <T>_MatMul_MMQ_StreamK} arguments. */
    public static final int ARG_COUNT = ARG_FIXUP + 1;
    /** {@code <T>_MatMul_MMQ_StreamK_GateUp}: the up weight (the base launch's weight is the gate). */
    public static final int ARG_WEIGHT_UP = ARG_COUNT;
    /** Number of {@code <T>_MatMul_MMQ_StreamK_GateUp} arguments. */
    public static final int GATE_UP_ARG_COUNT = ARG_WEIGHT_UP + 1;
    /** {@code LLM_MMQ_StreamK_Fixup} and its gate/up form: weight rows. */
    public static final int FIXUP_ARG_M = 0;
    /** The fixups: reduction length (fixes the chunk count per tile). */
    public static final int FIXUP_ARG_K = 1;
    /** The fixups: tokens. */
    public static final int FIXUP_ARG_N = 2;
    /** The fixups: elements between output rows. */
    public static final int FIXUP_ARG_Y_STRIDE = 3;
    /** The fixups: element offset of output row 0. */
    public static final int FIXUP_ARG_Y_OFF = 4;
    /** The fixups: the planes the stream-k kernel wrote. */
    public static final int FIXUP_ARG_FIXUP = 5;
    /** The fixups: the output. */
    public static final int FIXUP_ARG_Y = 6;
    /** Number of fixup arguments. */
    public static final int FIXUP_ARG_COUNT = 7;

    private final MmqGemmPlan base;
    private final String kernelName;
    private final int blocks;
    private final boolean gateUp;

    private StreamKGemmPlan(MmqGemmPlan base, int blocks, boolean gateUp) {
        this.base = base;
        this.kernelName = gateUp ? GemmKernels.matmulMmqStreamKGateUp(base.type())
                : small(base) ? GemmKernels.matmulMmqStreamKSmall(base.type())
                : GemmKernels.matmulMmqStreamK(base.type());
        this.blocks = blocks;
        this.gateUp = gateUp;
    }

    /**
     * The stream-k plan of a plain staged launch, or null when the launch keeps the plain grid: a tile other
     * than {@link MmqConfig#INT8_TENSOR}, a weight type without the int8 datapath, a device that does not
     * report its compute units, or a grid whose waves are at least {@link #PLAIN_GRID_EFFICIENCY_PERCENT}
     * percent full.
     *
     * @param base         the plain launch ({@link MmqGemmPlan#of}) the stream-k launch replaces
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @return the plan, or null when the launch keeps the plain grid
     * @throws IllegalArgumentException when {@code computeUnits} is negative
     */
    public static StreamKGemmPlan of(MmqGemmPlan base, int computeUnits) {
        Objects.requireNonNull(base, "base");
        if (small(base) || !resident(base, computeUnits)) {
            return null;
        }
        long resident = (long) computeUnits * TENSOR_TILE_BLOCKS_PER_UNIT;
        if (waveFillPercent((long) base.rowTiles() * base.tokenTiles(), resident) >= PLAIN_GRID_EFFICIENCY_PERCENT) {
            return null;
        }
        return overResident(base, resident, false);
    }

    /**
     * The stream-k plan of a launch over the small-token tensor tile ({@link MmqGemmPlan#tensorSmallOf}): the
     * tile's steps over {@link #SMALL_TILE_BLOCKS_PER_UNIT} resident blocks per compute unit, or one block per tile
     * when the tile grid's waves are at least {@link #PLAIN_GRID_EFFICIENCY_PERCENT} percent full (each block then
     * runs its tile whole and no fixup follows). Null when the device does not report its compute units, and when the
     * grid has fewer tiles than a {@link #SMALL_TILE_MAX_UNITS_PER_TILE}th of them.
     *
     * @throws IllegalArgumentException when {@code base} is not a small-token tensor tile launch, or
     *                                  {@code computeUnits} is negative
     */
    public static StreamKGemmPlan ofSmall(MmqGemmPlan base, int computeUnits) {
        Objects.requireNonNull(base, "base");
        if (!small(base)) {
            throw new IllegalArgumentException("the small-token stream-k plan runs the " + MmqConfig.INT8_TENSOR_SMALL
                    + " tile, not " + base.config());
        }
        if (!resident(base, computeUnits)) {
            return null;
        }
        long resident = (long) computeUnits * SMALL_TILE_BLOCKS_PER_UNIT;
        long tiles = (long) base.rowTiles() * base.tokenTiles();
        if (tiles * SMALL_TILE_MAX_UNITS_PER_TILE < computeUnits) {
            return null;
        }
        if (waveFillPercent(tiles, resident) >= PLAIN_GRID_EFFICIENCY_PERCENT) {
            return tiles > Integer.MAX_VALUE ? null : new StreamKGemmPlan(base, (int) tiles, false);
        }
        return overResident(base, resident, false);
    }

    /**
     * The fused gate/up SwiGLU launch over {@code base}'s shape, whose weight is the gate and whose output rows are
     * the activated rows, or null when the device or tile has no stream-k form (as {@link #of}, without the
     * wave-fill rule: the form has no plain grid to keep).
     *
     * @throws IllegalArgumentException when {@code computeUnits} is negative
     */
    public static StreamKGemmPlan ofGateUp(MmqGemmPlan base, int computeUnits) {
        Objects.requireNonNull(base, "base");
        if (small(base) || !resident(base, computeUnits)) {
            return null;
        }
        return overResident(base, (long) computeUnits * TENSOR_TILE_BLOCKS_PER_UNIT, true);
    }

    /** Whether the device reports its compute units and the launch runs the int8 tensor tile. */
    private static boolean resident(MmqGemmPlan base, int computeUnits) {
        if (computeUnits < DeviceCapabilities.UNKNOWN_COMPUTE_UNITS) {
            throw new IllegalArgumentException("stream-k needs a non-negative compute-unit count ("
                    + DeviceCapabilities.UNKNOWN_COMPUTE_UNITS + " when the device does not report one), got "
                    + computeUnits);
        }
        return computeUnits != DeviceCapabilities.UNKNOWN_COMPUTE_UNITS
                && (MmqConfig.INT8_TENSOR.equals(base.config()) || small(base)) && GemmKernels.isInt8Mmq(base.type());
    }

    /** Whether {@code base} runs the small-token tensor tile ({@link #ofSmall}). */
    private static boolean small(MmqGemmPlan base) {
        return MmqConfig.INT8_TENSOR_SMALL.equals(base.config());
    }

    /** The launch over {@code min(resident, steps)} blocks, or null when its steps overflow an int. */
    private static StreamKGemmPlan overResident(MmqGemmPlan base, long resident, boolean gateUp) {
        long steps = tiles(base, gateUp) * (base.k() / base.config().stageDepth());
        if (steps > Integer.MAX_VALUE) {
            return null;
        }
        return new StreamKGemmPlan(base, (int) Math.min(resident, steps), gateUp);
    }

    /**
     * The stream-k launch of {@code base} over exactly {@code blocks} blocks, whatever the wave fill (the
     * tests' and benchmarks' fixed partitions; {@link #of} picks the device's resident block count).
     *
     * @throws IllegalArgumentException when the tile or type has no stream-k kernel, or {@code blocks} is
     *                                  outside [1, tiles * chunks per tile]
     */
    public static StreamKGemmPlan ofBlocks(MmqGemmPlan base, int blocks) {
        return ofBlocks(base, blocks, false);
    }

    /** The gate/up form ({@link #ofGateUp}) over exactly {@code blocks} blocks, as {@link #ofBlocks}. */
    public static StreamKGemmPlan ofGateUpBlocks(MmqGemmPlan base, int blocks) {
        return ofBlocks(base, blocks, true);
    }

    private static StreamKGemmPlan ofBlocks(MmqGemmPlan base, int blocks, boolean gateUp) {
        Objects.requireNonNull(base, "base");
        if (!(MmqConfig.INT8_TENSOR.equals(base.config()) || (small(base) && !gateUp))
                || !GemmKernels.isInt8Mmq(base.type())) {
            throw new IllegalArgumentException((gateUp ? "the gate/up stream-k form runs the int8 tensor device tile"
                    : "stream-k runs the int8 tensor tiles") + ", not " + base.config() + " over " + base.type());
        }
        long steps = tiles(base, gateUp) * (base.k() / base.config().stageDepth());
        if (blocks < 1 || blocks > steps || steps > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("stream-k needs 1 <= blocks <= " + steps + " (tiles x chunks), got "
                    + blocks);
        }
        return new StreamKGemmPlan(base, blocks, gateUp);
    }

    /** Output tiles of the launch: row tiles of {@code TM} rows, or of {@code TM / 2} in the gate/up form, times
     *  the token tiles. */
    private static long tiles(MmqGemmPlan base, boolean gateUp) {
        long rows = gateUp ? base.config().tileRows() / 2 : base.config().tileRows();
        return (base.m() + rows - 1) / rows * base.tokenTiles();
    }

    /**
     * The busy share of the plain grid's waves in percent: {@code 100 * tiles / (resident * waves)} with
     * {@code waves = ceil(tiles / resident)}, llama.cpp's tiles_efficiency_percent.
     *
     * @throws IllegalArgumentException when {@code tiles} or {@code resident} is not positive
     */
    public static int waveFillPercent(long tiles, long resident) {
        if (tiles < 1 || resident < 1) {
            throw new IllegalArgumentException("wave fill needs positive tiles and resident blocks, got " + tiles
                    + " / " + resident);
        }
        long waves = (tiles + resident - 1) / resident;
        return (int) (100 * tiles / (resident * waves));
    }

    /** The plain launch the plan replaces (shape, views and tile). */
    public MmqGemmPlan base() {
        return base;
    }

    /** {@code <T>_MatMul_MMQ_StreamK}, {@code <T>_MatMul_MMQ_StreamK_GateUp} for the gate/up form, or
     *  {@code <T>_MatMul_MMQ_StreamK_S} over the small-token tile. */
    public String kernelName() {
        return kernelName;
    }

    /** Whether this is the fused gate/up SwiGLU form ({@link #ofGateUp}). */
    public boolean gateUp() {
        return gateUp;
    }

    /** Blocks of the stream-k launch. */
    public int blocks() {
        return blocks;
    }

    /** Output tiles of the launch: the plain grid's {@code rowTiles * tokenTiles}, or twice the row tiles of
     *  half the rows in the gate/up form. */
    public long tiles() {
        return tiles(base, gateUp);
    }

    /** Stage-depth chunks of one tile: {@code K / stageDepth}. */
    public int chunksPerTile() {
        return base.k() / base.config().stageDepth();
    }

    /**
     * The first (tile, chunk) step of block {@code b}: {@code b * tiles * chunks / blocks}, the kernel's
     * {@code nf_mmq_streamk_start}; block {@code blocks} gives the end of the sequence.
     *
     * @throws IllegalArgumentException when {@code b} is outside [0, blocks]
     */
    public long blockStart(int b) {
        if (b < 0 || b > blocks) {
            throw new IllegalArgumentException("stream-k block " + b + " outside [0, " + blocks + "]");
        }
        return (long) b * (tiles() * chunksPerTile()) / blocks;
    }

    /**
     * The block whose run holds (tile, chunk) step {@code step}: the largest {@code b} with
     * {@code blockStart(b) <= step}, {@code ((step + 1) * blocks - 1) / (tiles * chunks)}, the kernel's
     * {@code nf_mmq_streamk_owner}.
     *
     * @throws IllegalArgumentException when {@code step} is outside the sequence
     */
    public int ownerOf(long step) {
        long total = tiles() * chunksPerTile();
        if (step < 0 || step >= total) {
            throw new IllegalArgumentException("stream-k step " + step + " outside [0, " + total + ")");
        }
        return (int) (((step + 1) * blocks - 1) / total);
    }

    /**
     * Blocks that run some chunk of output tile {@code tile}: one for a tile run whole, more for a split tile,
     * whose fixup adds the planes of all but the last onto the last one's partial.
     *
     * @throws IllegalArgumentException when {@code tile} is outside the launch's tiles
     */
    public int contributors(long tile) {
        if (tile < 0 || tile >= tiles()) {
            throw new IllegalArgumentException("stream-k tile " + tile + " outside [0, " + tiles() + ")");
        }
        long first = tile * chunksPerTile();
        return ownerOf(first + chunksPerTile() - 1) - ownerOf(first) + 1;
    }

    /** Whether some tile is split across blocks, so the fixup must follow the launch: some block starts inside a
     *  tile. */
    public boolean splitsTiles() {
        for (int b = 1; b < blocks; b++) {
            if (blockStart(b) % chunksPerTile() != 0) {
                return true;
            }
        }
        return false;
    }

    /** Fixup planes per block: the plain form's one plane of a last segment that stops inside a tile, or the
     *  gate/up form's two, which also keep the first segment when it finishes a tile another block started. */
    public int planesPerBlock() {
        return gateUp ? 2 : 1;
    }

    /** Floats of the fixup buffer: {@link #planesPerBlock} {@code TM x TN} planes per block. */
    public long fixupElements() {
        return (long) blocks * planesPerBlock() * base.config().tileRows() * base.config().tileTokens();
    }

    /** {@code LLM_MMQ_StreamK_Fixup}, {@code LLM_MMQ_StreamK_Fixup_GateUp} for the gate/up form, or
     *  {@code LLM_MMQ_StreamK_Fixup_S} over the small-token tile. */
    public String fixupKernelName() {
        return gateUp ? GemmKernels.STREAM_K_FIXUP_GATE_UP
                : small(base) ? GemmKernels.STREAM_K_FIXUP_SMALL : GemmKernels.STREAM_K_FIXUP;
    }

    /** {@code {blocks * WG, 1}}: the global work size of the stream-k launch. */
    public long[] globalWorkSize() {
        return new long[]{(long) blocks * base.config().workGroupSize(), 1};
    }

    /** {@code {WG, 1}}: the local work size of both launches. */
    public long[] localWorkSize() {
        return new long[]{base.config().workGroupSize(), 1};
    }

    /** {@code {blocks * WG, outputs / (WG * FIXUP_ELEMENTS_PER_WORK_ITEM)}}: the global work size of the fixup, the
     *  outputs those of one output tile ({@code TM x TN}, or {@code TM / 2 x TN} in the gate/up form). */
    public long[] fixupGlobalWorkSize() {
        MmqConfig c = base.config();
        long rows = gateUp ? c.tileRows() / 2 : c.tileRows();
        long slices = rows * c.tileTokens() / ((long) c.workGroupSize() * FIXUP_ELEMENTS_PER_WORK_ITEM);
        return new long[]{(long) blocks * c.workGroupSize(), slices};
    }

    /**
     * Whether the device can run the launches: the tile's device support, the block count in dimension 0 and the
     * fixup's slices in dimension 1.
     *
     * @return null when it can, otherwise the reason
     */
    public String checkDevice(DeviceCapabilities caps) {
        Objects.requireNonNull(caps, "caps");
        String reason = MmqGemmPlan.deviceSupport(base.config(), base.type(), caps);
        if (reason != null) {
            return reason;
        }
        long[] maxGroups = caps.getMaxWorkGroupCount();
        long slices = fixupGlobalWorkSize()[1];
        if (blocks > maxGroups[0] || slices > maxGroups[1]) {
            return kernelName + " needs " + blocks + " x " + slices + " work-groups; device " + caps.getDeviceName()
                    + " allows " + maxGroups[0] + " x " + maxGroups[1];
        }
        return null;
    }

    /**
     * Stream-k kernel arguments in declared order: the plain kernel's ({@link MmqGemmPlan#args}, which checks
     * the weight, record and output buffers) followed by the fixup planes.
     *
     * @throws IllegalStateException    for the gate/up form, which takes {@link #argsGateUp}
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs, or the fixup buffer is
     *                                  another argument's buffer
     */
    public Object[] args(GpuBuffer weight, GpuBuffer q8, GpuBuffer y, GpuBuffer fixup) {
        if (gateUp) {
            throw new IllegalStateException(kernelName + " takes the up weight too: use argsGateUp");
        }
        return streamKArgs(weight, q8, y, fixup, ARG_COUNT);
    }

    /**
     * Gate/up kernel arguments in declared order: {@link #args}'s over the gate weight, then the up weight, which
     * must be a buffer of its own at least the gate's size (the two weights share type and shape).
     *
     * @throws IllegalStateException    when this is not the gate/up form
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException as {@link #args}, or for an up weight smaller than the gate or shared with
     *                                  another argument
     */
    public Object[] argsGateUp(GpuBuffer gate, GpuBuffer up, GpuBuffer q8, GpuBuffer y, GpuBuffer fixup) {
        if (!gateUp) {
            throw new IllegalStateException(kernelName + " has no up weight: use args");
        }
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(gate, "gate");
        if (up == gate || up == q8 || up == y || up == fixup) {
            throw new IllegalArgumentException(kernelName + ": the up weight needs its own buffer");
        }
        if (up.sizeInBytes() < gate.sizeInBytes()) {
            throw new IllegalArgumentException(kernelName + ": up weight holds " + up.sizeInBytes()
                    + " bytes, less than the gate's " + gate.sizeInBytes());
        }
        Object[] args = streamKArgs(gate, q8, y, fixup, GATE_UP_ARG_COUNT);
        args[ARG_WEIGHT_UP] = up;
        return args;
    }

    private Object[] streamKArgs(GpuBuffer weight, GpuBuffer q8, GpuBuffer y, GpuBuffer fixup, int count) {
        Object[] plain = base.args(weight, q8, y);
        requireFixup(fixup, y, q8, weight);
        Object[] args = new Object[count];
        System.arraycopy(plain, 0, args, 0, plain.length);
        args[ARG_FIXUP] = fixup;
        return args;
    }

    /**
     * Fixup arguments in declared order ({@link #fixupKernelName}): the launch's shape and output view, the
     * planes the stream-k kernel wrote and the output it wrote.
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when either buffer is smaller than the launch needs, or they are one buffer
     */
    public Object[] fixupArgs(GpuBuffer fixup, GpuBuffer y) {
        Objects.requireNonNull(y, "y");
        requireFixup(fixup, y, null, null);
        if (y.sizeInBytes() < base.yElementsRequired() * Float.BYTES) {
            throw new IllegalArgumentException(fixupKernelName() + ": output buffer holds " + y.sizeInBytes()
                    + " bytes but the launch needs " + base.yElementsRequired() * Float.BYTES);
        }
        Object[] args = new Object[FIXUP_ARG_COUNT];
        args[FIXUP_ARG_M] = base.m();
        args[FIXUP_ARG_K] = base.k();
        args[FIXUP_ARG_N] = base.n();
        args[FIXUP_ARG_Y_STRIDE] = base.yStride();
        args[FIXUP_ARG_Y_OFF] = base.yOff();
        args[FIXUP_ARG_FIXUP] = fixup;
        args[FIXUP_ARG_Y] = y;
        return args;
    }

    private void requireFixup(GpuBuffer fixup, GpuBuffer y, GpuBuffer q8, GpuBuffer weight) {
        Objects.requireNonNull(fixup, "fixup");
        if (fixup == y || fixup == q8 || fixup == weight) {
            throw new IllegalArgumentException(kernelName + ": the fixup planes need their own buffer");
        }
        if (fixup.sizeInBytes() < fixupElements() * Float.BYTES) {
            throw new IllegalArgumentException(kernelName + ": fixup buffer holds " + fixup.sizeInBytes()
                    + " bytes but " + blocks + " blocks need " + fixupElements() * Float.BYTES);
        }
    }

    @Override
    public String toString() {
        return "StreamKGemmPlan{" + kernelName + ", M=" + base.m() + ", K=" + base.k() + ", N=" + base.n()
                + ", tiles=" + tiles() + ", blocks=" + blocks + '}';
    }
}
