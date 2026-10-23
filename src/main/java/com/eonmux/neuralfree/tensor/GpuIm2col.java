package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.GpuConvOps.ConvGeometry;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

/**
 * The im2col + GEMM path of the convolutions of {@link GpuConvOps} and {@link GpuConvGradOps} (the formulation of
 * Caffe, PyTorch's THCUNN and ggml): per chunk of images, {@code TR_Im2col_F32} lays out the column matrix
 * {@code [C*KH*KW, OH*OW]} of each image in a reusable scratch buffer, and the strided GEMM {@code TR_Gemm_F32}
 * multiplies it with the weight; the transposed direction multiplies the transposed weight with the input into
 * columns and sums them back with {@code TR_Col2im_F32}. Every operand is a dense FLOAT32 view on one backend; the
 * geometry is a {@link ConvGeometry} in the roles of a convolution (input {@code [B, C, H, W]}, output
 * {@code [B, OC, OH, OW]}, weight {@code [OC, C/groups, KH, KW]}), except for {@link #transposed}, which takes the
 * transposed convolution's own geometry. The CPU references are those of the direct kernels ({@code CpuConvPool}
 * {@code conv2d}, {@code convTranspose2d} and {@code conv2dWeightGrad}, the loops of {@code nn.layers.Conv2d} and
 * {@code ConvTranspose2d}).
 *
 * <p><b>Selection.</b> {@link Strategy#AUTO} takes this path when the serial tap loop of the direct kernel,
 * {@code (C/groups)*KH*KW}, has at least {@value #MIN_TAPS} taps, the columns of one image fit the scratch budget of
 * {@value #COLUMN_BUDGET_BYTES} bytes, and each GEMM needs at most {@link GpuMatmul#MAX_GROUPS} tiles per dimension;
 * otherwise the direct kernels run. A chunk holds {@code min(B, budget / columnBytesPerImage)} images. The choice
 * and the chunk size depend only on the shapes.</p>
 *
 * <p><b>Accumulation.</b> Forward: each output sums its {@code (C/groups)*KH*KW} products in ascending
 * {@code (ic, kh, kw)} order with {@code fma} from {@code +0}, then adds the bias (the output is pre-filled with the
 * bias and the GEMM runs with {@code beta = 1}). Transposed: the GEMM sums each tap's products over the group's input
 * channels in ascending order, then {@code TR_Col2im_F32} adds the valid taps in {@code (kh, kw)} order onto the
 * bias. Weight gradient: each image's {@code OH*OW} products are summed with {@code fma} in ascending position order,
 * and the images are added in ascending order ({@code beta = 1} after the first). All three are in the reduction
 * class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract} with the same term counts as the direct
 * kernels. A padding tap enters a GEMM as a product with an exact {@code +0} column entry, so it changes no finite sum;
 * a non-finite weight (forward) or gradient (weight gradient) at a tap that reads only padding gives NaN here where the
 * direct kernel skips the tap, as in the im2col implementations named above.</p>
 *
 * <p><b>Locking.</b> Every sequence runs inside {@link TrainGpu#exclusive}, because the column buffer is the
 * backend's scratch slot {@value #SLOT_COLUMNS}.</p>
 */
final class GpuIm2col {

    /** Upper bound of the column scratch buffer of one chunk, in bytes. */
    static final long COLUMN_BUDGET_BYTES = 64L << 20;
    /** Fewest taps per output of the direct kernel's serial loop for which {@link Strategy#AUTO} takes this path. */
    static final int MIN_TAPS = 16;
    /** Scratch slot of the column buffer, distinct from the slots of the other dispatchers. */
    private static final int SLOT_COLUMNS = 6;

    private GpuIm2col() {
    }

    /** How a convolution chooses between the direct kernels and this path. Immutable. */
    static final class Strategy {
        /** The selection rule of the class javadoc with the default budget. */
        static final Strategy AUTO = new Strategy(Mode.AUTO, COLUMN_BUDGET_BYTES);
        /** Always the direct kernels. */
        static final Strategy DIRECT = new Strategy(Mode.DIRECT, COLUMN_BUDGET_BYTES);

        private final Mode mode;
        private final long budgetBytes;

        private Strategy(Mode mode, long budgetBytes) {
            this.mode = mode;
            this.budgetBytes = budgetBytes;
        }

        /**
         * Always this path, with chunks bounded by {@code budgetBytes} of columns; a convolution whose columns of one
         * image exceed the budget, or whose GEMM exceeds the tile limit, throws {@link IllegalArgumentException}.
         *
         * @throws IllegalArgumentException unless {@code 4 <= budgetBytes <= 4 * (2^31-1)}
         */
        static Strategy im2col(long budgetBytes) {
            if (budgetBytes < Float.BYTES || budgetBytes / Float.BYTES > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("column budget must hold between 1 and " + Integer.MAX_VALUE
                        + " floats, got " + budgetBytes + " bytes");
            }
            return new Strategy(Mode.IM2COL, budgetBytes);
        }

        @Override
        public String toString() {
            return mode == Mode.IM2COL ? "IM2COL(" + budgetBytes + " bytes)" : mode.name();
        }
    }

    private enum Mode {
        AUTO, DIRECT, IM2COL
    }

    /** The chunking of one convolution on this path. Immutable. */
    static final class Plan {
        /** Images per chunk, at least 1. */
        final int imagesPerChunk;
        /** Column-buffer elements per image. */
        final int columnsPerImage;

        private Plan(int imagesPerChunk, int columnsPerImage) {
            this.imagesPerChunk = imagesPerChunk;
            this.columnsPerImage = columnsPerImage;
        }
    }

    // ------------------------------------------------------------------ planning

    /**
     * The plan of a convolution forward over {@code geo}: columns {@code [C*KH*KW, OH*OW]} per image and per group
     * the GEMM {@code [OC/groups, (C/groups)*KH*KW] x [(C/groups)*KH*KW, OH*OW]}; null for the direct kernel.
     */
    static Plan planForward(ConvGeometry geo, Strategy s) {
        long taps = (long) (geo.c / geo.groups) * geo.kh * geo.kw;
        return plan(geo.b, (long) geo.c * geo.kh * geo.kw, (long) geo.oh * geo.ow, taps, geo.oc / geo.groups,
                (long) geo.oh * geo.ow, s, "conv2d");
    }

    /**
     * The plan of a convolution weight gradient over {@code geo}: the forward's columns and per image and group the
     * GEMM {@code [OC/groups, OH*OW] x [OH*OW, (C/groups)*KH*KW]}; null for the direct kernel.
     */
    static Plan planWeightGrad(ConvGeometry geo, Strategy s) {
        long taps = (long) (geo.c / geo.groups) * geo.kh * geo.kw;
        return plan(geo.b, (long) geo.c * geo.kh * geo.kw, (long) geo.oh * geo.ow, taps, geo.oc / geo.groups, taps,
                s, "conv2d weight gradient");
    }

    /**
     * The plan of a transposed convolution over its own geometry {@code geo} (input {@code [B, C, H, W]}, output
     * {@code [B, OC, OH, OW]}, weight {@code [C, OC/groups, KH, KW]}): columns {@code [OC*KH*KW, H*W]} per image and
     * per group the GEMM {@code [(OC/groups)*KH*KW, C/groups] x [C/groups, H*W]}; null for the direct kernel.
     */
    static Plan planTransposed(ConvGeometry geo, Strategy s) {
        long taps = (long) (geo.c / geo.groups) * geo.kh * geo.kw;
        long rowsPerGroup = (long) (geo.oc / geo.groups) * geo.kh * geo.kw;
        return plan(geo.b, rowsPerGroup * geo.groups, (long) geo.h * geo.w, taps, rowsPerGroup,
                (long) geo.h * geo.w, s, "convTranspose2d");
    }

    private static Plan plan(int images, long rows, long cols, long taps, long gemmM, long gemmN, Strategy s,
                             String op) {
        if (s.mode == Mode.DIRECT) {
            return null;
        }
        long budget = s.budgetBytes / Float.BYTES;
        long perImage = rows * cols;
        boolean fits = perImage >= 1 && perImage <= budget && tiles(gemmM) <= GpuMatmul.MAX_GROUPS
                && tiles(gemmN) <= GpuMatmul.MAX_GROUPS;
        if (s.mode == Mode.AUTO) {
            return fits && taps >= MIN_TAPS ? new Plan((int) Math.min(images, budget / perImage), (int) perImage)
                    : null;
        }
        if (!fits) {
            throw new IllegalArgumentException(op + " cannot run as im2col + GEMM: " + perImage + " column elements"
                    + " per image against a budget of " + budget + ", GEMM " + gemmM + " x " + gemmN + " against "
                    + GpuMatmul.MAX_GROUPS + " tiles of " + TrainOps.GEMM_TILE + " per dimension");
        }
        return new Plan((int) Math.min(images, budget / perImage), (int) perImage);
    }

    // ------------------------------------------------------------------ sequences

    /**
     * {@code out = conv(in, w)} plus the bias already in {@code out} when {@code accumulate}; {@code out} is the dense
     * {@code [B, OC, OH, OW]} result at element offset 0.
     */
    static void forward(TrainGpu g, Plan plan, ConvGeometry geo, StridedView in, StridedView w, boolean accumulate,
                        GpuBuffer out) {
        int cg = geo.c / geo.groups;
        int ocg = geo.oc / geo.groups;
        int taps = cg * geo.kh * geo.kw;
        int positions = geo.oh * geo.ow;
        long imageIn = (long) geo.c * geo.h * geo.w;
        g.exclusive(() -> {
            GpuBuffer cols = columns(g, plan);
            for (int b0 = 0; b0 < geo.b; b0 += plan.imagesPerChunk) {
                int count = Math.min(plan.imagesPerChunk, geo.b - b0);
                im2col(g, geo, count, in.buffer(), in.offset() + b0 * imageIn, cols);
                gemm(g, ocg, positions, taps,
                        new Side(w.buffer(), w.offset(), taps, 1, 0, (long) ocg * taps),
                        new Side(cols, 0, positions, 1, plan.columnsPerImage, (long) taps * positions),
                        new Side(out, (long) b0 * geo.oc * positions, positions, 1, (long) geo.oc * positions,
                                (long) ocg * positions),
                        count, geo.groups, accumulate ? 1.0f : 0.0f);
            }
            return null;
        });
    }

    /**
     * The transposed convolution {@code out = convTranspose(in, w) + bias} over its own geometry: per chunk and group
     * the GEMM {@code W_g^T x in_g} into columns, then {@code TR_Col2im_F32} into the dense {@code [B, OC, OH, OW]}
     * result at element offset 0. {@code bias} is null without bias.
     */
    static void transposed(TrainGpu g, Plan plan, ConvGeometry geo, StridedView in, StridedView w, StridedView bias,
                           GpuBuffer out) {
        int cg = geo.c / geo.groups;
        int rowsPerGroup = (geo.oc / geo.groups) * geo.kh * geo.kw;
        int positions = geo.h * geo.w;
        long imageOut = (long) geo.oc * geo.oh * geo.ow;
        g.exclusive(() -> {
            GpuBuffer cols = columns(g, plan);
            for (int b0 = 0; b0 < geo.b; b0 += plan.imagesPerChunk) {
                int count = Math.min(plan.imagesPerChunk, geo.b - b0);
                gemm(g, rowsPerGroup, positions, cg,
                        new Side(w.buffer(), w.offset(), 1, rowsPerGroup, 0, (long) cg * rowsPerGroup),
                        new Side(in.buffer(), in.offset() + (long) b0 * geo.c * positions, positions, 1,
                                (long) geo.c * positions, (long) cg * positions),
                        new Side(cols, 0, positions, 1, plan.columnsPerImage, (long) rowsPerGroup * positions),
                        count, geo.groups, 0.0f);
                g.launch1d(Kernels.COL2IM_F32, count * imageOut,
                        count, geo.oc, geo.oh, geo.ow, geo.h, geo.w, geo.kh, geo.kw, geo.sh, geo.sw, geo.ph, geo.pw,
                        geo.dh, geo.dw, bias == null ? 0 : 1, 0, bias == null ? 0 : bias.offset(),
                        toInt(b0 * imageOut, "col2im output offset"),
                        cols, bias == null ? g.placeholder() : bias.buffer(), out);
            }
            return null;
        });
    }

    /**
     * The weight gradient {@code gw[OC, C/groups, KH, KW] = sum over images of gy_b x cols_b^T} per group, into the
     * dense {@code gw} at element offset 0 ({@code x} is the convolution input, {@code gy} its output gradient).
     */
    static void weightGrad(TrainGpu g, Plan plan, ConvGeometry geo, StridedView x, StridedView gy, GpuBuffer gw) {
        int cg = geo.c / geo.groups;
        int ocg = geo.oc / geo.groups;
        int taps = cg * geo.kh * geo.kw;
        int positions = geo.oh * geo.ow;
        long imageIn = (long) geo.c * geo.h * geo.w;
        long imageGy = (long) geo.oc * positions;
        g.exclusive(() -> {
            GpuBuffer cols = columns(g, plan);
            for (int b0 = 0; b0 < geo.b; b0 += plan.imagesPerChunk) {
                int count = Math.min(plan.imagesPerChunk, geo.b - b0);
                im2col(g, geo, count, x.buffer(), x.offset() + b0 * imageIn, cols);
                for (int i = 0; i < count; i++) {
                    gemm(g, ocg, taps, positions,
                            new Side(gy.buffer(), gy.offset() + (b0 + i) * imageGy, positions, 1, 0,
                                    (long) ocg * positions),
                            new Side(cols, (long) i * plan.columnsPerImage, 1, positions, 0, (long) taps * positions),
                            new Side(gw, 0, taps, 1, 0, (long) ocg * taps),
                            1, geo.groups, b0 + i == 0 ? 0.0f : 1.0f);
                }
            }
            return null;
        });
    }

    private static GpuBuffer columns(TrainGpu g, Plan plan) {
        return g.scratch(SLOT_COLUMNS, (long) plan.imagesPerChunk * plan.columnsPerImage * Float.BYTES);
    }

    private static void im2col(TrainGpu g, ConvGeometry geo, int count, GpuBuffer in, long inOffset, GpuBuffer cols) {
        long items = (long) count * geo.c * geo.kh * geo.kw * geo.oh * geo.ow;
        g.launch1d(Kernels.IM2COL_F32, items,
                count, geo.c, geo.h, geo.w, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw, geo.ph, geo.pw,
                geo.dh, geo.dw, toInt(inOffset, "im2col input offset"), 0, in, cols);
    }

    // ------------------------------------------------------------------ GEMM over images and groups

    /**
     * One GEMM operand: element offset, row and column strides, and the offset steps per image and per group. The
     * result operand's column stride is 1. Immutable.
     */
    private static final class Side {
        final GpuBuffer buffer;
        final long offset;
        final int rowStride;
        final int colStride;
        final long perImage;
        final long perGroup;

        Side(GpuBuffer buffer, long offset, int rowStride, int colStride, long perImage, long perGroup) {
            this.buffer = buffer;
            this.offset = offset;
            this.rowStride = rowStride;
            this.colStride = colStride;
            this.perImage = perImage;
            this.perGroup = perGroup;
        }

        int offset(boolean batchImages, int loop, int start) {
            long base = batchImages ? offset + loop * perGroup + start * perImage
                    : offset + loop * perImage + start * perGroup;
            return toInt(base, "GEMM operand offset");
        }

        int batchStride(boolean batchImages) {
            return toInt(batchImages ? perImage : perGroup, "GEMM batch stride");
        }
    }

    /**
     * {@code C = A x B + beta * C} for every (image, group) pair of {@code images x groups}, the products differing
     * only in their offsets. The larger of the two counts is the GEMM batch (chunked to
     * {@link GpuMatmul#MAX_GROUPS} entries per launch) and the other is looped. Distinct (image, group) pairs must
     * address distinct result elements, except that a result whose per-image step is 0 (the weight gradient, which
     * accumulates the images) is called with {@code images == 1}. Every product computes the same sums whatever the
     * batching.
     */
    private static void gemm(TrainGpu g, int m, int n, int k, Side a, Side b, Side c, int images, int groups,
                             float beta) {
        boolean batchImages = images >= groups;
        int batch = batchImages ? images : groups;
        int loops = batchImages ? groups : images;
        long[] local = TrainOps.localSize(Kernels.GEMM_F32);
        long gx = tiles(n) * local[0];
        long gy = tiles(m) * local[1];
        for (int loop = 0; loop < loops; loop++) {
            for (int start = 0; start < batch; start += GpuMatmul.MAX_GROUPS) {
                int count = Math.min(GpuMatmul.MAX_GROUPS, batch - start);
                g.launch(Kernels.GEMM_F32, new long[]{gx, gy, count}, local,
                        m, n, k, count,
                        a.offset(batchImages, loop, start), a.rowStride, a.colStride, a.batchStride(batchImages),
                        b.offset(batchImages, loop, start), b.rowStride, b.colStride, b.batchStride(batchImages),
                        c.offset(batchImages, loop, start), c.rowStride, c.batchStride(batchImages),
                        1.0f, beta,
                        a.buffer, b.buffer, c.buffer);
            }
        }
    }

    private static long tiles(long extent) {
        return (extent + TrainOps.GEMM_TILE - 1) / TrainOps.GEMM_TILE;
    }

    private static int toInt(long value, String what) {
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + value + " is outside the 32-bit kernel range");
        }
        return (int) value;
    }
}
