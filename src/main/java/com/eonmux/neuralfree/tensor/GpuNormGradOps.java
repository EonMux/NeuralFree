package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.GpuNormOps.ChannelGeometry;
import com.eonmux.neuralfree.tensor.GpuNormOps.Staging;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Backward passes of layer, RMS, batch and group normalization on the train kernels (the backward kernels of
 * {@code kernels/train/norm.cl}, with {@code TR_ReduceCols_F32}, {@code TR_Copy_F32} and the forward's statistics
 * kernels). The results are new contiguous FLOAT32 tensors on the input's backend, returned as
 * {@link NormOps.Gradients}; nothing is read back to the host.
 *
 * <p><b>Dispatcher contract.</b> Every method returns {@code null} exactly when the train library is unavailable on
 * the input's backend ({@link TrainGpu#forBackend}) or an operand is not FLOAT32; that is the caller's CPU path
 * ({@code CpuNorm}), and nothing has been enqueued then. Invalid arguments throw {@link IllegalArgumentException}
 * whether or not the library is available. Once the library is available, kernel and backend failures propagate and
 * the gradients made so far are closed.</p>
 *
 * <p><b>Statistics.</b> The normalized value {@code xhat} the forward produced is recomputed, never stored: the row
 * backward kernels recompute mean, variance and {@code sdev} with the forward kernel's own lane helpers and tree;
 * BatchNorm in training mode recomputes the batch statistics with {@code TR_ChannelStats_F32} exactly as
 * {@link GpuNormOps#batchNormTrain} launches it (never through {@code batchNormTrain} itself, which would update the
 * running statistics a second time); GroupNorm recomputes them with the forward's {@link GpuNormOps#reduceRows} SUM
 * and SQDEV calls. Each statistic is therefore a deterministic function of the same values, shape and launch as the
 * forward's, and {@code xhat} is bit-identical to the forward's. BatchNorm in evaluation mode reads the statistics it
 * is given (the copies of the running statistics the forward normalized with).</p>
 *
 * <p><b>Tolerance</b> ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): every gradient is in the
 * reduction class, with K the longest sum it depends on (the normalized row or group, the batch of a channel, or the
 * rows of an affine column). The affine-parameter gradients of LayerNorm and RMSNorm sum their columns in ascending row
 * order per chunk ({@code TR_ReduceCols_F32} partition and chunk rule), so the gamma gradient and the beta gradient
 * each have one fixed order.</p>
 */
public final class GpuNormGradOps {

    /** Affine column sums are chunked above this many rows ({@code TR_ReduceCols_F32} shape rule). */
    private static final int COL_CHUNK_ROWS_ABOVE = 4096;
    private static final int COL_MAX_CHUNKS = 256;
    private static final int COL_CHUNK_ROWS = 1024;

    private GpuNormGradOps() {
    }

    // ------------------------------------------------------------------ row normalizations

    /**
     * LayerNorm backward over rows of {@code normalizedSize} elements ({@link NormOps#layerNormBackward}):
     * {@code TR_LayerNormRowsBackward_F32} gives {@code gx} and the per-row {@code (mean, sdev)} that
     * {@code TR_NormAffineGradCols_F32} reads for the gamma gradient (with the beta gradient as its second plane when
     * both are wanted); a beta gradient without a gamma gradient is the column sum of {@code gy}, whose partition and
     * order equal the fused kernel's beta plane.
     *
     * @return the gradients, or null (dispatcher contract)
     * @throws IllegalArgumentException for the argument errors of {@link NormOps#layerNormBackward}, or operands on
     *                                  different backends
     */
    public static NormOps.Gradients layerNormBackward(Tensor gy, Tensor x, int normalizedSize, Tensor gamma,
                                                      float eps, boolean needInput, boolean needGamma,
                                                      Shape betaShape) {
        checkLayerNorm(gy, x, normalizedSize, gamma, needGamma, betaShape);
        requireSameBackend(x, gy, "gradient");
        requireSameBackend(x, gamma, "gamma");
        if (!allFloat32(gy, x, gamma)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        Results res = new Results(g, x.backend());
        try (Staging s = new Staging(g)) {
            StridedView gyv = s.dense(gy);
            int cols = normalizedSize;
            int rows = (int) (x.numElements() / cols);
            Tensor gx = needInput ? res.allocate(x.shape()) : null;
            Tensor gw = needGamma ? res.allocate(gamma.shape()) : null;
            Tensor gb = betaShape != null ? res.allocate(betaShape) : null;
            boolean fusedBeta = false;
            if (gx != null || gw != null) {
                StridedView xv = s.dense(x);
                StridedView gv = s.denseOrNull(gamma);
                GpuBuffer gxBuf = gx != null ? gx.storage().gpuBuffer() : s.temp(xv.numElements());
                GpuBuffer stats = s.temp(2 * rows);
                g.launchRows(Kernels.LAYER_NORM_ROWS_BACKWARD_F32, rows,
                        rows, cols, eps, flag(gv), xv.offset(), offset(gv), gyv.offset(), 0,
                        xv.buffer(), buffer(g, gv), gyv.buffer(), gxBuf, stats);
                if (gw != null) {
                    fusedBeta = gb != null && fitsTwoPlanes(rows, cols);
                    affineGrads(g, s, TrainOps.TR_N_AFF_LN, rows, cols, xv, gyv, stats, gw, fusedBeta ? gb : null);
                }
            }
            if (gb != null && !fusedBeta) {
                columnSums(g, s, rows, cols, gyv, gb);
            }
            return new NormOps.Gradients(gx, gw, gb);
        } catch (RuntimeException e) {
            throw res.closeAfter(e);
        }
    }

    /**
     * RMSNorm backward over the last dim ({@link NormOps#rmsNormBackward}): {@code TR_RMSNormRowsBackward_F32} gives
     * {@code gx} and the per-row {@code rms} that {@code TR_NormAffineGradCols_F32} reads for the weight gradient.
     *
     * @return the gradients, or null (dispatcher contract)
     * @throws IllegalArgumentException for the argument errors of {@link NormOps#rmsNormBackward}, or operands on
     *                                  different backends
     */
    public static NormOps.Gradients rmsNormBackward(Tensor gy, Tensor x, Tensor weight, float eps, boolean needInput,
                                                    boolean needWeight) {
        checkRmsNorm(gy, x, weight, needWeight);
        requireSameBackend(x, gy, "gradient");
        requireSameBackend(x, weight, "weight");
        if (!allFloat32(gy, x, weight)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        Results res = new Results(g, x.backend());
        try (Staging s = new Staging(g)) {
            StridedView xv = s.dense(x);
            StridedView gyv = s.dense(gy);
            StridedView wv = s.denseOrNull(weight);
            int cols = x.shape().dim(x.shape().rank() - 1);
            int rows = cols == 0 ? 0 : xv.numElements() / cols;
            Tensor gx = needInput ? res.allocate(x.shape()) : null;
            Tensor gw = needWeight ? res.allocate(weight.shape()) : null;
            if (gx != null || gw != null) {
                GpuBuffer gxBuf = gx != null ? gx.storage().gpuBuffer() : s.temp(xv.numElements());
                GpuBuffer stats = s.temp(rows);
                g.launchRows(Kernels.RMS_NORM_ROWS_BACKWARD_F32, rows,
                        rows, cols, eps, flag(wv), xv.offset(), offset(wv), gyv.offset(), 0,
                        xv.buffer(), buffer(g, wv), gyv.buffer(), gxBuf, stats);
                if (gw != null) {
                    affineGrads(g, s, TrainOps.TR_N_AFF_RMS, rows, cols, xv, gyv, stats, gw, null);
                }
            }
            return new NormOps.Gradients(gx, gw, null);
        } catch (RuntimeException e) {
            throw res.closeAfter(e);
        }
    }

    // ------------------------------------------------------------------ channel normalizations

    /**
     * BatchNorm backward for an input {@code [B, C, *]} ({@link NormOps#batchNormBackward}). Training mode: the batch
     * statistics are recomputed with {@code TR_ChannelStats_F32} (the forward's launch), {@code TR_ChannelGradSums_F32}
     * writes {@code [sum(gy) | sum(gy * xhat)]} per channel into one planar {@code [2C]} scratch, which the
     * {@code TR_N_CB_BATCH} mode of {@code TR_ChannelNormBackwardApply_F32} reads for {@code gx}; the beta and gamma
     * gradients are copies of its two halves. Evaluation mode: {@code gx} from the {@code TR_N_CB_EVAL} mode with the
     * given statistics, and the affine gradients from {@code TR_ChannelGradSums_F32} over those statistics.
     *
     * @return the gradients, or null (dispatcher contract)
     * @throws IllegalArgumentException for the argument errors of {@link NormOps#batchNormBackward}, or operands on
     *                                  different backends
     */
    public static NormOps.Gradients batchNormBackward(Tensor gy, Tensor x, Tensor gamma, Tensor runningMean,
                                                      Tensor runningVar, boolean training, float eps,
                                                      boolean needInput, boolean needGamma, Shape betaShape) {
        ChannelGeometry geo = checkBatchNorm(gy, x, gamma, runningMean, runningVar, training, needGamma, betaShape);
        requireSameBackend(x, gy, "gradient");
        requireSameBackend(x, gamma, "gamma");
        if (!training) {
            requireSameBackend(x, runningMean, "runningMean");
            requireSameBackend(x, runningVar, "runningVar");
        }
        if (!allFloat32(gy, x, gamma) || (!training && !allFloat32(runningMean, runningVar))) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        Results res = new Results(g, x.backend());
        try (Staging s = new Staging(g)) {
            Tensor gx = needInput ? res.allocate(x.shape()) : null;
            Tensor gw = needGamma ? res.allocate(gamma.shape()) : null;
            Tensor gb = betaShape != null ? res.allocate(betaShape) : null;
            int c = geo.channels;
            if (x.numElements() == 0) {
                zeroAffine(g, gw, gb);
                return new NormOps.Gradients(gx, gw, gb);
            }
            StridedView gyv = s.dense(gy);
            StridedView gv = s.atOffsetZeroOrNull(gamma);
            int n = (int) x.numElements();
            if (training) {
                StridedView xv = s.dense(x);
                GpuBuffer mean = s.temp(c);
                GpuBuffer var = s.temp(c);
                GpuBuffer sums = s.temp(2 * c);
                g.launchRows(Kernels.CHANNEL_STATS_F32, c,
                        geo.batch, c, geo.spatial, xv.offset(), xv.buffer(), mean, var);
                channelGradSums(g, geo, 0, eps, xv, gyv, mean, var, sums);
                if (gx != null) {
                    g.launch1d(Kernels.CHANNEL_NORM_BACKWARD_APPLY_F32, n,
                            TrainOps.TR_N_CB_BATCH, geo.batch, c, geo.spatial, 0, geo.batch * geo.spatial, eps,
                            flag(gv), xv.offset(), gyv.offset(), 0,
                            xv.buffer(), mean, var, buffer(g, gv), gyv.buffer(), sums, gx.storage().gpuBuffer());
                }
                copyHalves(g, sums, c, gw, gb);
                return new NormOps.Gradients(gx, gw, gb);
            }
            GpuBuffer mean = s.atOffsetZero(runningMean).buffer();
            GpuBuffer var = s.atOffsetZero(runningVar).buffer();
            if (gx != null) {
                g.launch1d(Kernels.CHANNEL_NORM_BACKWARD_APPLY_F32, n,
                        TrainOps.TR_N_CB_EVAL, geo.batch, c, geo.spatial, 0, geo.batch * geo.spatial, eps,
                        flag(gv), 0, gyv.offset(), 0,
                        g.placeholder(), mean, var, buffer(g, gv), gyv.buffer(), g.placeholder(),
                        gx.storage().gpuBuffer());
            }
            if (gw != null || gb != null) {
                StridedView xv = s.dense(x);
                GpuBuffer sums = s.temp(2 * c);
                channelGradSums(g, geo, 0, eps, xv, gyv, mean, var, sums);
                copyHalves(g, sums, c, gw, gb);
            }
            return new NormOps.Gradients(gx, gw, gb);
        } catch (RuntimeException e) {
            throw res.closeAfter(e);
        }
    }

    /**
     * GroupNorm backward for an input {@code [B, C, *]} with {@code groups} groups ({@link NormOps#groupNormBackward}):
     * the statistics of the {@code B * groups} rows are recomputed with the forward's {@code reduceRows} SUM and SQDEV
     * calls; the affine gradients come from {@code TR_ChannelGradSums_F32} with the group statistics, {@code gx} from
     * {@code TR_GroupGradSums_F32} (planar {@code [sum(d) | sum(d * xhat)]} per row) and the {@code TR_N_CB_GROUP}
     * mode of {@code TR_ChannelNormBackwardApply_F32}.
     *
     * @return the gradients, or null (dispatcher contract)
     * @throws IllegalArgumentException for the argument errors of {@link NormOps#groupNormBackward}, or operands on
     *                                  different backends
     */
    public static NormOps.Gradients groupNormBackward(Tensor gy, Tensor x, int groups, Tensor gamma, float eps,
                                                      boolean needInput, boolean needGamma, Shape betaShape) {
        ChannelGeometry geo = checkGroupNorm(gy, x, groups, gamma, needGamma, betaShape);
        requireSameBackend(x, gy, "gradient");
        requireSameBackend(x, gamma, "gamma");
        if (!allFloat32(gy, x, gamma)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        Results res = new Results(g, x.backend());
        try (Staging s = new Staging(g)) {
            Tensor gx = needInput ? res.allocate(x.shape()) : null;
            Tensor gw = needGamma ? res.allocate(gamma.shape()) : null;
            Tensor gb = betaShape != null ? res.allocate(betaShape) : null;
            if (x.numElements() == 0) {
                zeroAffine(g, gw, gb);
                return new NormOps.Gradients(gx, gw, gb);
            }
            StridedView xv = s.dense(x);
            StridedView gyv = s.dense(gy);
            StridedView gv = s.atOffsetZeroOrNull(gamma);
            int rows = geo.batch * groups;
            int count = (geo.channels / groups) * geo.spatial;
            GpuBuffer mean = s.temp(rows);
            GpuBuffer var = s.temp(rows);
            GpuNormOps.reduceRows(g, s, TrainOps.TR_R_SUM, rows, count, xv, g.placeholder(), mean);
            GpuNormOps.reduceRows(g, s, TrainOps.TR_R_SQDEV, rows, count, xv, mean, var);
            if (gw != null || gb != null) {
                GpuBuffer sums = s.temp(2 * geo.channels);
                channelGradSums(g, geo, groups, eps, xv, gyv, mean, var, sums);
                copyHalves(g, sums, geo.channels, gw, gb);
            }
            if (gx != null) {
                GpuBuffer rsums = s.temp(2 * rows);
                g.launchRows(Kernels.GROUP_GRAD_SUMS_F32, rows,
                        geo.batch, geo.channels, geo.spatial, groups, eps, flag(gv), xv.offset(), gyv.offset(),
                        xv.buffer(), mean, var, buffer(g, gv), gyv.buffer(), rsums);
                g.launch1d(Kernels.CHANNEL_NORM_BACKWARD_APPLY_F32, xv.numElements(),
                        TrainOps.TR_N_CB_GROUP, geo.batch, geo.channels, geo.spatial, groups, count, eps,
                        flag(gv), xv.offset(), gyv.offset(), 0,
                        xv.buffer(), mean, var, buffer(g, gv), gyv.buffer(), rsums, gx.storage().gpuBuffer());
            }
            return new NormOps.Gradients(gx, gw, gb);
        } catch (RuntimeException e) {
            throw res.closeAfter(e);
        }
    }

    // ------------------------------------------------------------------ launches

    /**
     * {@code TR_ChannelGradSums_F32} over C channel groups into the planar {@code [2C]} {@code sums} at offset 0:
     * {@code sums[c] = sum(gy)}, {@code sums[C + c] = sum(gy * xhat)}, statistics keyed per channel
     * ({@code groups == 0}) or per (batch, group).
     */
    private static void channelGradSums(TrainGpu g, ChannelGeometry geo, int groups, float eps, StridedView xv,
                                        StridedView gyv, GpuBuffer mean, GpuBuffer var, GpuBuffer sums) {
        g.launchRows(Kernels.CHANNEL_GRAD_SUMS_F32, geo.channels,
                geo.batch, geo.channels, geo.spatial, groups, eps, xv.offset(), gyv.offset(), 0,
                xv.buffer(), mean, var, gyv.buffer(), sums);
    }

    /** The beta gradient from {@code sums[0, C)} and the gamma gradient from {@code sums[C, 2C)}, exact copies. */
    private static void copyHalves(TrainGpu g, GpuBuffer sums, int channels, Tensor gw, Tensor gb) {
        if (gb != null) {
            g.launch1d(Kernels.COPY_F32, channels, channels, 0, 0, sums, gb.storage().gpuBuffer());
        }
        if (gw != null) {
            g.launch1d(Kernels.COPY_F32, channels, channels, channels, 0, sums, gw.storage().gpuBuffer());
        }
    }

    /** Zero affine gradients: the sums over an empty input. */
    private static void zeroAffine(TrainGpu g, Tensor gw, Tensor gb) {
        for (Tensor t : new Tensor[]{gw, gb}) {
            if (t != null && t.numElements() > 0) {
                g.fill(t.storage().gpuBuffer(), 0, t.numElements(), 0.0f);
            }
        }
    }

    /**
     * Affine column gradients of a {@code [rows, cols]} block with {@code TR_NormAffineGradCols_F32}: the gamma (or
     * RMS weight) gradient into {@code gw}, and the beta gradient into {@code gb} as the kernel's second plane when
     * {@code gb} is given. The kernel writes {@code chunks} partial rows per plane into scratch; each plane is then
     * summed by {@code TR_ReduceCols_F32} ({@code rows = chunks}), or copied exactly when {@code chunks == 1}.
     */
    private static void affineGrads(TrainGpu g, Staging s, int mode, int rows, int cols, StridedView xv,
                                    StridedView gyv, GpuBuffer stats, Tensor gw, Tensor gb) {
        if (cols == 0) {
            return;
        }
        int chunks = columnChunks(rows);
        int plane = chunks * cols;
        GpuBuffer partials = s.temp(gb != null ? 2 * plane : plane);
        g.launch1d(Kernels.NORM_AFFINE_GRAD_COLS_F32, plane,
                mode, rows, cols, chunks, flag(gb), xv.offset(), gyv.offset(), 0,
                xv.buffer(), gyv.buffer(), stats, partials);
        reducePlane(g, partials, 0, chunks, cols, gw);
        if (gb != null) {
            reducePlane(g, partials, plane, chunks, cols, gb);
        }
    }

    /** {@code out[col] = sum over the chunks rows of the plane at planeOffset}; an exact copy for one chunk. */
    private static void reducePlane(TrainGpu g, GpuBuffer partials, int planeOffset, int chunks, int cols,
                                    Tensor out) {
        GpuBuffer dst = out.storage().gpuBuffer();
        if (chunks == 1) {
            g.launch1d(Kernels.COPY_F32, cols, cols, planeOffset, 0, partials, dst);
        } else {
            g.launch1d(Kernels.REDUCE_COLS_F32, cols, chunks, cols, 1, planeOffset, 0, partials, dst);
        }
    }

    /**
     * Column sums of the {@code [rows, cols]} gradient block into {@code out} with {@code TR_ReduceCols_F32} and the
     * chunk rule of {@link #affineGrads}, so the result equals the kernel's beta plane bit for bit.
     */
    private static void columnSums(TrainGpu g, Staging s, int rows, int cols, StridedView gyv, Tensor out) {
        if (cols == 0) {
            return;
        }
        GpuBuffer dst = out.storage().gpuBuffer();
        int chunks = columnChunks(rows);
        if (chunks == 1) {
            g.launch1d(Kernels.REDUCE_COLS_F32, cols, rows, cols, 1, gyv.offset(), 0, gyv.buffer(), dst);
            return;
        }
        GpuBuffer partials = s.temp(chunks * cols);
        g.launch1d(Kernels.REDUCE_COLS_F32, (long) chunks * cols, rows, cols, chunks, gyv.offset(), 0,
                gyv.buffer(), partials);
        g.launch1d(Kernels.REDUCE_COLS_F32, cols, chunks, cols, 1, 0, 0, partials, dst);
    }

    /** Chunk count of the affine column sums: {@code rows > 4096 ? min(256, ceil(rows / 1024)) : 1}. */
    private static int columnChunks(int rows) {
        return rows > COL_CHUNK_ROWS_ABOVE ? Math.min(COL_MAX_CHUNKS, ceilDiv(rows, COL_CHUNK_ROWS)) : 1;
    }

    /** Whether the two planes of {@code TR_NormAffineGradCols_F32} fit the kernel's {@code int} indexing. */
    private static boolean fitsTwoPlanes(int rows, int cols) {
        return 2L * columnChunks(rows) * cols <= Integer.MAX_VALUE;
    }

    // ------------------------------------------------------------------ validation (shared with NormOps)

    /** The argument checks of {@link NormOps#layerNormBackward}. */
    static void checkLayerNorm(Tensor gy, Tensor x, int normalizedSize, Tensor gamma, boolean needGamma,
                               Shape betaShape) {
        requireGradient(gy, x);
        if (normalizedSize <= 0) {
            throw new IllegalArgumentException("normalizedSize must be positive, got " + normalizedSize);
        }
        if (x.numElements() % normalizedSize != 0) {
            throw new IllegalArgumentException("input " + x.shape() + " does not split into rows of "
                    + normalizedSize + " elements");
        }
        requireKernelSize(x.shape());
        requirePlanar(x.numElements() / normalizedSize, "row statistics");
        requireElements(gamma, normalizedSize, "gamma");
        requireAffineGradients(gamma, needGamma, betaShape, normalizedSize);
    }

    /** The argument checks of {@link NormOps#rmsNormBackward}. */
    static void checkRmsNorm(Tensor gy, Tensor x, Tensor weight, boolean needWeight) {
        requireGradient(gy, x);
        if (x.shape().rank() < 1) {
            throw new IllegalArgumentException("rmsNorm needs an input of rank at least 1, got " + x.shape());
        }
        requireKernelSize(x.shape());
        requireElements(weight, x.shape().dim(x.shape().rank() - 1), "weight");
        if (needWeight && weight == null) {
            throw new IllegalArgumentException("a weight gradient needs the weight");
        }
    }

    /** The argument checks of {@link NormOps#batchNormBackward}; returns the input's geometry. */
    static ChannelGeometry checkBatchNorm(Tensor gy, Tensor x, Tensor gamma, Tensor runningMean, Tensor runningVar,
                                          boolean training, boolean needGamma, Shape betaShape) {
        requireGradient(gy, x);
        ChannelGeometry geo = ChannelGeometry.of(x, 0);
        requirePlanar(geo.channels, "channel sums");
        requireElements(gamma, geo.channels, "gamma");
        requireAffineGradients(gamma, needGamma, betaShape, geo.channels);
        if (!training) {
            Objects.requireNonNull(runningMean, "runningMean must not be null in evaluation mode");
            Objects.requireNonNull(runningVar, "runningVar must not be null in evaluation mode");
            requireElements(runningMean, geo.channels, "runningMean");
            requireElements(runningVar, geo.channels, "runningVar");
        }
        return geo;
    }

    /** The argument checks of {@link NormOps#groupNormBackward}; returns the input's geometry. */
    static ChannelGeometry checkGroupNorm(Tensor gy, Tensor x, int groups, Tensor gamma, boolean needGamma,
                                          Shape betaShape) {
        requireGradient(gy, x);
        if (groups <= 0) {
            throw new IllegalArgumentException("groups must be positive, got " + groups);
        }
        ChannelGeometry geo = ChannelGeometry.of(x, groups);
        requirePlanar(geo.channels, "channel sums");
        requirePlanar((long) geo.batch * groups, "group sums");
        requireElements(gamma, geo.channels, "gamma");
        requireAffineGradients(gamma, needGamma, betaShape, geo.channels);
        return geo;
    }

    private static void requireGradient(Tensor gy, Tensor x) {
        Objects.requireNonNull(gy, "gradient must not be null");
        Objects.requireNonNull(x, "input must not be null");
        if (!gy.shape().equals(x.shape())) {
            throw new IllegalArgumentException("gradient " + gy.shape() + " does not match the input " + x.shape());
        }
    }

    private static void requireAffineGradients(Tensor gamma, boolean needGamma, Shape betaShape, long elements) {
        if (needGamma && gamma == null) {
            throw new IllegalArgumentException("a gamma gradient needs gamma");
        }
        if (betaShape != null && betaShape.numElements() != elements) {
            throw new IllegalArgumentException("beta must have " + elements + " elements, got shape " + betaShape);
        }
    }

    private static void requireElements(Tensor p, long elements, String name) {
        if (p != null && p.numElements() != elements) {
            throw new IllegalArgumentException(name + " must have " + elements + " elements, got shape " + p.shape());
        }
    }

    /** The train kernel ABI and the host arrays address at most {@code 2^31-1} elements. */
    private static void requireKernelSize(Shape shape) {
        if (shape.numElements() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("input " + shape + " exceeds the 2^31-1 elements of the train kernel "
                    + "ABI");
        }
    }

    /** A planar pair of {@code count} values ({@code [s1 | s2]}) is addressed with {@code int} indices. */
    private static void requirePlanar(long count, String what) {
        if (2 * count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the " + count + " " + what + " of the norm backward exceed the "
                    + "2^31-1 elements of the train kernel ABI as a planar pair");
        }
    }

    private static void requireSameBackend(Tensor x, Tensor operand, String name) {
        if (operand != null && operand.backend() != x.backend()) {
            throw new IllegalArgumentException(name + " lives on a different backend than the input");
        }
    }

    // ------------------------------------------------------------------ helpers

    /** True when every non-null tensor is FLOAT32. */
    private static boolean allFloat32(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null && t.dtype() != DType.FLOAT32) {
                return false;
            }
        }
        return true;
    }

    private static int flag(Object operand) {
        return operand == null ? 0 : 1;
    }

    private static int offset(StridedView v) {
        return v == null ? 0 : v.offset();
    }

    private static GpuBuffer buffer(TrainGpu g, StridedView v) {
        return v == null ? g.placeholder() : v.buffer();
    }

    private static int ceilDiv(int a, int b) {
        return a / b + (a % b == 0 ? 0 : 1);
    }

    /** The gradient tensors of one dispatch, closed together when a later launch fails. */
    private static final class Results {
        private final TrainGpu g;
        private final ComputeBackend backend;
        private final List<Tensor> made = new ArrayList<>();

        Results(TrainGpu g, ComputeBackend backend) {
            this.g = g;
            this.backend = backend;
        }

        /** A new contiguous FLOAT32 tensor of {@code shape}, owned by this result set until it is returned. */
        Tensor allocate(Shape shape) {
            Tensor t = Tensor.fromStorage(backend, shape, DType.FLOAT32,
                    g.allocate(shape.numElements(), DType.FLOAT32));
            made.add(t);
            return t;
        }

        /** Closes every tensor made so far, attaching close failures to {@code failure}, and returns it. */
        RuntimeException closeAfter(RuntimeException failure) {
            for (Tensor t : made) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    failure.addSuppressed(e);
                }
            }
            made.clear();
            return failure;
        }
    }
}
