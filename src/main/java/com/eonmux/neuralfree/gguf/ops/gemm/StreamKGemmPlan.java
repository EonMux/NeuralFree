package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Objects;

/**
 * The stream-k launch of a staged int8 MMQ GEMM over the CUDA tensor tile ({@link MmqConfig#INT8_TENSOR}):
 * {@code <T>_MatMul_MMQ_StreamK} followed, when a tile is split, by {@code LLM_MMQ_StreamK_Fixup}
 * ({@code kernels/llm/matmul_mmq_stream.cl}), llama.cpp's stream-k decomposition of its MMQ grid
 * ({@code ggml-cuda/mmq.cuh}, {@code mul_mat_q} with {@code tmp_fixup}).
 *
 * <p>The tensor tile keeps one 256-work-item block resident per compute unit (its kernels hold 166-248
 * registers per work-item, {@link #TENSOR_TILE_BLOCKS_PER_UNIT}), so a grid of {@code T} tiles runs in
 * {@code ceil(T / units)} waves and its last wave idles {@code units - T % units} compute units for a whole
 * tile time: the 4B layers' 80-tile projections (2560 rows over 512 tokens) run two waves for 1.33 waves of
 * work, the 304-tile ones six for 5.07. The stream-k kernel lays the grid's {@code T * K / 32} (tile, chunk)
 * steps out as one sequence, tile-major with the token tile fastest, and gives each of {@code units} blocks
 * an equal contiguous run of it; a block stores the tiles whose last chunk it runs, and writes the
 * partial sums of a run that stops inside a tile into its own {@code TM x TN} plane of the fixup buffer,
 * which the fixup kernel adds into the tile afterwards (at most one plane per block).</p>
 *
 * <p>The plan follows llama.cpp's choice: a grid whose waves are already at least
 * {@link #PLAIN_GRID_EFFICIENCY_PERCENT} percent full keeps the plain kernel (the stream-k blocks would
 * gain less than the fixup costs), and so does a device that does not report its compute units. A tile
 * one block runs whole equals the plain kernel's output bit for bit; a split tile differs by the fp32
 * rounding of adding its partial sums, and its partition depends on the token count, so a batch-invariant
 * run keeps the plain kernel (the caller's decision, as for {@link SplitKGemmPlan}).</p>
 *
 * <p>Geometry: stream-k {@code global = {blocks * WG, 1}}, {@code local = {WG, 1}}; the fixup
 * {@code global = {blocks * WG, TM * TN / (WG * }{@link #FIXUP_ELEMENTS_PER_WORK_ITEM}{@code )}} over the same
 * block count, one work-group per slice of a tile's outputs.</p>
 */
public final class StreamKGemmPlan {

    /** Wave fill (percent of the last wave's compute units busy, llama.cpp's tiles_efficiency_percent) from
     *  which the plain grid is kept: llama.cpp runs its stream-k kernel as plain tiles there on NVIDIA. */
    public static final int PLAIN_GRID_EFFICIENCY_PERCENT = 90;
    /** Resident blocks per compute unit of the tensor tile's kernels: 256 work-items at 166-248 registers
     *  (sm_120 build) leave room for one block of 65536 registers. */
    public static final int TENSOR_TILE_BLOCKS_PER_UNIT = 1;

    /** Tile outputs one fixup work-item adds ({@code NF_MMQ_SK_FIX_E}): a tile's {@code TM x TN} outputs
     *  spread over {@code TM * TN / (WG * 4)} work-groups, so each split tile's planes are read with four
     *  loads in flight per work-item rather than walked by one work-group (a 64-load chain per work-item
     *  measured 12-14 us per fixup launch at 60 blocks). */
    public static final int FIXUP_ELEMENTS_PER_WORK_ITEM = 4;
    /** {@code <T>_MatMul_MMQ_StreamK}: the fixup planes, {@code blocks * TM * TN} floats. */
    public static final int ARG_FIXUP = GemmKernels.MATMUL_TILED_ARG_COUNT;
    /** Number of {@code <T>_MatMul_MMQ_StreamK} arguments. */
    public static final int ARG_COUNT = ARG_FIXUP + 1;
    /** {@code LLM_MMQ_StreamK_Fixup}: weight rows. */
    public static final int FIXUP_ARG_M = 0;
    /** {@code LLM_MMQ_StreamK_Fixup}: reduction length (fixes the chunk count per tile). */
    public static final int FIXUP_ARG_K = 1;
    /** {@code LLM_MMQ_StreamK_Fixup}: tokens. */
    public static final int FIXUP_ARG_N = 2;
    /** {@code LLM_MMQ_StreamK_Fixup}: elements between output rows. */
    public static final int FIXUP_ARG_Y_STRIDE = 3;
    /** {@code LLM_MMQ_StreamK_Fixup}: element offset of output row 0. */
    public static final int FIXUP_ARG_Y_OFF = 4;
    /** {@code LLM_MMQ_StreamK_Fixup}: the fixup planes the stream-k kernel wrote. */
    public static final int FIXUP_ARG_FIXUP = 5;
    /** {@code LLM_MMQ_StreamK_Fixup}: the output. */
    public static final int FIXUP_ARG_Y = 6;
    /** Number of {@code LLM_MMQ_StreamK_Fixup} arguments. */
    public static final int FIXUP_ARG_COUNT = 7;

    private final MmqGemmPlan base;
    private final String kernelName;
    private final int blocks;

    private StreamKGemmPlan(MmqGemmPlan base, int blocks) {
        this.base = base;
        this.kernelName = GemmKernels.matmulMmqStreamK(base.type());
        this.blocks = blocks;
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
        if (computeUnits < DeviceCapabilities.UNKNOWN_COMPUTE_UNITS) {
            throw new IllegalArgumentException("stream-k needs a non-negative compute-unit count ("
                    + DeviceCapabilities.UNKNOWN_COMPUTE_UNITS + " when the device does not report one), got "
                    + computeUnits);
        }
        if (computeUnits == DeviceCapabilities.UNKNOWN_COMPUTE_UNITS || !MmqConfig.INT8_TENSOR.equals(base.config())
                || !GemmKernels.isInt8Mmq(base.type())) {
            return null;
        }
        long resident = (long) computeUnits * TENSOR_TILE_BLOCKS_PER_UNIT;
        long tiles = (long) base.rowTiles() * base.tokenTiles();
        if (waveFillPercent(tiles, resident) >= PLAIN_GRID_EFFICIENCY_PERCENT) {
            return null;
        }
        long steps = tiles * (base.k() / base.config().stageDepth());
        if (steps > Integer.MAX_VALUE) {
            return null;
        }
        return new StreamKGemmPlan(base, (int) Math.min(resident, steps));
    }

    /**
     * The stream-k launch of {@code base} over exactly {@code blocks} blocks, whatever the wave fill (the
     * tests' and benchmarks' fixed partitions; {@link #of} picks the device's resident block count).
     *
     * @throws IllegalArgumentException when the tile or type has no stream-k kernel, or {@code blocks} is
     *                                  outside [1, tiles * chunks per tile]
     */
    public static StreamKGemmPlan ofBlocks(MmqGemmPlan base, int blocks) {
        Objects.requireNonNull(base, "base");
        if (!MmqConfig.INT8_TENSOR.equals(base.config()) || !GemmKernels.isInt8Mmq(base.type())) {
            throw new IllegalArgumentException("stream-k runs the int8 tensor tile, not " + base.config() + " over "
                    + base.type());
        }
        long steps = (long) base.rowTiles() * base.tokenTiles() * (base.k() / base.config().stageDepth());
        if (blocks < 1 || blocks > steps || steps > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("stream-k needs 1 <= blocks <= " + steps + " (tiles x chunks), got "
                    + blocks);
        }
        return new StreamKGemmPlan(base, blocks);
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

    /** {@code <T>_MatMul_MMQ_StreamK}. */
    public String kernelName() {
        return kernelName;
    }

    /** Blocks of the stream-k and fixup launches. */
    public int blocks() {
        return blocks;
    }

    /** Output tiles of the launch: {@code rowTiles * tokenTiles} of the plain grid. */
    public long tiles() {
        return (long) base.rowTiles() * base.tokenTiles();
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

    /** Whether some tile is split across blocks, so the fixup kernel must run: some block starts inside a tile. */
    public boolean fixupNeeded() {
        for (int b = 1; b < blocks; b++) {
            if (blockStart(b) % chunksPerTile() != 0) {
                return true;
            }
        }
        return false;
    }

    /** Floats of the fixup buffer: one {@code TM x TN} plane per block. */
    public long fixupElements() {
        return (long) blocks * base.config().tileRows() * base.config().tileTokens();
    }

    /** {@code {blocks * WG, 1}}: the global work size of the stream-k launch. */
    public long[] globalWorkSize() {
        return new long[]{(long) blocks * base.config().workGroupSize(), 1};
    }

    /** {@code {WG, 1}}: the local work size of both launches. */
    public long[] localWorkSize() {
        return new long[]{base.config().workGroupSize(), 1};
    }

    /** {@code {blocks * WG, TM * TN / (WG * FIXUP_ELEMENTS_PER_WORK_ITEM)}}: the global work size of the fixup. */
    public long[] fixupGlobalWorkSize() {
        MmqConfig c = base.config();
        long slices = (long) c.tileRows() * c.tileTokens() / ((long) c.workGroupSize() * FIXUP_ELEMENTS_PER_WORK_ITEM);
        return new long[]{(long) blocks * c.workGroupSize(), slices};
    }

    /**
     * Whether the device can run the launches: the tile's device support, the block count in dimension 0
     * and the fixup's slices in dimension 1.
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
     * the weight, record and output buffers) followed by the fixup buffer.
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs, or the fixup buffer is
     *                                  the output or the records
     */
    public Object[] args(GpuBuffer weight, GpuBuffer q8, GpuBuffer y, GpuBuffer fixup) {
        Object[] plain = base.args(weight, q8, y);
        requireFixup(fixup, y, q8);
        Object[] args = new Object[ARG_COUNT];
        System.arraycopy(plain, 0, args, 0, plain.length);
        args[ARG_FIXUP] = fixup;
        return args;
    }

    /**
     * Fixup kernel arguments in declared order ({@link #FIXUP_ARG_M} .. {@link #FIXUP_ARG_Y}).
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when the fixup buffer is too small or is the output
     */
    public Object[] fixupArgs(GpuBuffer fixup, GpuBuffer y) {
        Objects.requireNonNull(y, "y");
        requireFixup(fixup, y, null);
        if (y.sizeInBytes() < base.yElementsRequired() * Float.BYTES) {
            throw new IllegalArgumentException(GemmKernels.STREAM_K_FIXUP + ": output buffer holds " + y.sizeInBytes()
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

    private void requireFixup(GpuBuffer fixup, GpuBuffer y, GpuBuffer q8) {
        Objects.requireNonNull(fixup, "fixup");
        if (fixup == y || fixup == q8) {
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
