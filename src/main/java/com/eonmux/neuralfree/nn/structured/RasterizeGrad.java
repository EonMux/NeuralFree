package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The host rasterizer of {@link GaussianSplatting} and its gradient, one autograd node whose input is the recorded
 * {@code [numGaussians, 9]} splat tensor.
 *
 * <p><b>Forward.</b> Splat row {@code g} is {@code [px, py, varX, covXY, varY, opacity, r, g, b]}. The Gaussians are
 * composited per pixel in Gaussian order over their three-sigma box: {@code det = max(varX varY - covXY^2, 1e-8)},
 * {@code m = (varY dx^2 - 2 covXY dx dy + varX dy^2) / det} with {@code dx = x - px}, {@code dy = y - py},
 * {@code w = exp(-m/2) * opacity}, then {@code a = w (1 - A)}, {@code C += c a} and {@code A += a}, where {@code A} is
 * the pixel's accumulated alpha (the image's fourth channel). The transmittance in front of a splat is
 * {@code T = 1 - A}.</p>
 *
 * <p><b>Backward.</b> With the pixel gradient {@code G = (G_r, G_g, G_b, G_A)} and the alpha channel read as a fourth
 * colour equal to 1, the per-pixel lists of the forward are rebuilt on the host in the same Gaussian order, the
 * forward's {@code w} and {@code T} are recomputed bit for bit, and the list is walked back to front with the colour
 * behind each splat {@code R_{i-1} = c_i w_i + (1 - w_i) R_i} ({@code R} of the last splat is 0):
 * {@code dL/dc_i = G w_i T_i}, {@code dL/dw_i = T_i * sum_k G_k (c_ik - R_ik)}. The weight is chained through
 * {@code w = exp(-m/2) * opacity}: {@code dL/dopacity = dL/dw * exp(-m/2)}, {@code dL/dm = -w/2 * dL/dw}, and through
 * {@code m}: {@code dm/dpx = -2 (varY dx - covXY dy) / det}, {@code dm/dpy = -2 (varX dy - covXY dx) / det},
 * {@code dm/dvarX = dy^2/det - m varY/det}, {@code dm/dvarY = dx^2/det - m varX/det},
 * {@code dm/dcovXY = -2 dx dy/det + 2 m covXY/det}; the {@code det} terms are dropped where {@code det} was floored.
 * The box bounds are piecewise constant and contribute nothing. The result is uploaded as the {@code [numGaussians, 9]}
 * gradient of the splat tensor on the splat tensor's backend.</p>
 *
 * <p>Saved: a host copy of the splat rows (no tensor), dropped by {@link #clearSavedTensors()}. Tolerance: host
 * computation in a fixed sequential order with double accumulation, deterministic; it has no device counterpart.
 * {@link #supportsCreateGraph()} is false: the backward is a host computation, not a recorded composition, so a
 * create-graph pass through it throws.</p>
 */
public final class RasterizeGrad implements GradientFunction {
    /** Columns of a splat row. */
    static final int FIELDS = 9;
    private static final int PX = 0;
    private static final int PY = 1;
    private static final int VAR_X = 2;
    private static final int COV_XY = 3;
    private static final int VAR_Y = 4;
    private static final int OPACITY = 5;
    private static final int RED = 6;
    private static final int CHANNELS = 4;
    private static final float MIN_DET = 1e-8f;
    /** The largest Java array the per-pixel lists may use. */
    private static final long MAX_LIST_ENTRIES = Integer.MAX_VALUE - 8L;

    private final int numGaussians;
    private final int width;
    private final int height;
    private final ComputeBackend backend;
    private float[] splats;

    /**
     * The gradient of {@link #rasterize} at {@code splats} (taken over, not copied), for a splat tensor on
     * {@code backend}.
     *
     * @throws IllegalArgumentException when {@code splats} does not hold {@code numGaussians} rows or a size is not
     *                                  positive
     */
    RasterizeGrad(float[] splats, int numGaussians, int width, int height, ComputeBackend backend) {
        requireSplats(splats, numGaussians, width, height);
        this.splats = splats;
        this.numGaussians = numGaussians;
        this.width = width;
        this.height = height;
        this.backend = Objects.requireNonNull(backend, "backend cannot be null");
    }

    /**
     * The {@code [height, width, 4]} RGBA image of the splat rows, row-major (see the class description).
     *
     * @throws IllegalArgumentException when {@code splats} does not hold {@code numGaussians} rows or a size is not
     *                                  positive
     */
    static float[] rasterize(float[] splats, int numGaussians, int width, int height) {
        requireSplats(splats, numGaussians, width, height);
        float[] image = new float[pixelCount(width, height) * CHANNELS];
        for (int g = 0; g < numGaussians; g++) {
            Footprint f = Footprint.of(splats, g, width, height);
            float r = splats[g * FIELDS + RED];
            float gc = splats[g * FIELDS + RED + 1];
            float b = splats[g * FIELDS + RED + 2];
            for (int y = f.minY; y <= f.maxY; y++) {
                for (int x = f.minX; x <= f.maxX; x++) {
                    float weight = f.falloff(f.mahalanobis(x, y)) * f.opacity;
                    int pixelIdx = (y * width + x) * CHANNELS;
                    float existingAlpha = image[pixelIdx + 3];
                    float newAlpha = weight * (1.0f - existingAlpha);
                    image[pixelIdx] += r * newAlpha;
                    image[pixelIdx + 1] += gc * newAlpha;
                    image[pixelIdx + 2] += b * newAlpha;
                    image[pixelIdx + 3] += newAlpha;
                }
            }
        }
        return image;
    }

    /**
     * The gradient of the splat tensor as a new {@code [numGaussians, 9]} FLOAT32 tensor on the splat tensor's
     * backend; {@code gradOutput} is read once to the host and left open.
     *
     * @throws IllegalArgumentException when {@code gradOutput} is not a FLOAT32 {@code [height, width, 4]} tensor
     * @throws IllegalStateException    when the node was released
     */
    @Override
    public List<Tensor> backward(Tensor gradOutput) {
        float[] rows = splats;
        if (rows == null) {
            throw new IllegalStateException("The rasterizer node was released; its splats are no longer available");
        }
        Objects.requireNonNull(gradOutput, "gradOutput cannot be null");
        Shape imageShape = Shape.of(height, width, CHANNELS);
        if (!gradOutput.shape().equals(imageShape) || gradOutput.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("rasterizer gradient must be FLOAT32 " + imageShape + ", got "
                + gradOutput.dtype() + " " + gradOutput.shape());
        }
        float[] gy = gradOutput.toFloatArray();
        double[] g = splatGradient(rows, gy);
        float[] out = new float[g.length];
        for (int i = 0; i < g.length; i++) {
            out[i] = (float) g[i];
        }
        try (Tensor flat = Tensor.fromArray(backend, out)) {
            return Collections.singletonList(flat.reshape(Shape.matrix(numGaussians, FIELDS)));
        }
    }

    /** The splat-row gradient in double precision, pixel by pixel in row-major order. */
    private double[] splatGradient(float[] rows, float[] gy) {
        Footprint[] footprints = new Footprint[numGaussians];
        for (int g = 0; g < numGaussians; g++) {
            footprints[g] = Footprint.of(rows, g, width, height);
        }
        PixelLists lists = PixelLists.build(footprints, width, height);
        Scratch scratch = new Scratch(lists.longest());
        double[] grad = new double[numGaussians * FIELDS];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = y * width + x;
                int start = lists.offsets[pixel];
                int end = lists.offsets[pixel + 1];
                int at = pixel * CHANNELS;
                boolean silent = gy[at] == 0.0f && gy[at + 1] == 0.0f && gy[at + 2] == 0.0f && gy[at + 3] == 0.0f;
                if (start < end && !silent) {
                    accumulatePixel(x, y, lists.ids, start, end, footprints, rows, gy, at, scratch, grad);
                }
            }
        }
        return grad;
    }

    /** Adds one pixel's contribution to {@code grad}: the forward composite recomputed, then walked back to front. */
    private static void accumulatePixel(int x, int y, int[] ids, int start, int end, Footprint[] footprints,
                                        float[] rows, float[] gy, int at, Scratch s, double[] grad) {
        int n = end - start;
        float alpha = 0.0f;
        for (int j = 0; j < n; j++) {
            Footprint f = footprints[ids[start + j]];
            float mahal = f.mahalanobis(x, y);
            float falloff = f.falloff(mahal);
            float weight = falloff * f.opacity;
            s.mahal[j] = mahal;
            s.falloff[j] = falloff;
            s.weight[j] = weight;
            s.transmittance[j] = 1.0f - alpha;
            float newAlpha = weight * (1.0f - alpha);
            alpha += newAlpha;
        }
        double gr = gy[at];
        double gg = gy[at + 1];
        double gb = gy[at + 2];
        double ga = gy[at + 3];
        double behindR = 0.0;
        double behindG = 0.0;
        double behindB = 0.0;
        double behindA = 0.0;
        for (int j = n - 1; j >= 0; j--) {
            int gi = ids[start + j];
            int base = gi * FIELDS;
            double w = s.weight[j];
            double t = s.transmittance[j];
            double cr = rows[base + RED];
            double cg = rows[base + RED + 1];
            double cb = rows[base + RED + 2];
            double dLdw = t * (gr * (cr - behindR) + gg * (cg - behindG) + gb * (cb - behindB)
                + ga * (1.0 - behindA));
            double contribution = w * t;
            grad[base + RED] += gr * contribution;
            grad[base + RED + 1] += gg * contribution;
            grad[base + RED + 2] += gb * contribution;
            behindR = cr * w + (1.0 - w) * behindR;
            behindG = cg * w + (1.0 - w) * behindG;
            behindB = cb * w + (1.0 - w) * behindB;
            behindA = w + (1.0 - w) * behindA;
            footprints[gi].chain(x, y, dLdw, s.falloff[j], s.weight[j], s.mahal[j], grad, base);
        }
    }

    @Override
    public List<Tensor> savedTensors() {
        return Collections.emptyList();
    }

    @Override
    public void clearSavedTensors() {
        splats = null;
    }

    @Override
    public boolean supportsCreateGraph() {
        return false;
    }

    private static void requireSplats(float[] splats, int numGaussians, int width, int height) {
        Objects.requireNonNull(splats, "splats cannot be null");
        if (numGaussians <= 0 || width <= 0 || height <= 0) {
            throw new IllegalArgumentException("numGaussians, width and height must be positive: " + numGaussians
                + ", " + width + ", " + height);
        }
        if ((long) numGaussians * FIELDS != splats.length) {
            throw new IllegalArgumentException("expected " + numGaussians + " splat rows of " + FIELDS
                + " values, got " + splats.length + " values");
        }
        pixelCount(width, height);
    }

    private static int pixelCount(int width, int height) {
        long pixels = (long) width * height;
        if (pixels * CHANNELS > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("image " + width + "x" + height + " is too large for the host "
                + "rasterizer");
        }
        return (int) pixels;
    }

    /** One splat's screen-space constants and its clipped three-sigma pixel box. */
    private static final class Footprint {
        final float px;
        final float py;
        final float varX;
        final float covXY;
        final float varY;
        final float opacity;
        final float invDet;
        final boolean detFloored;
        final int minX;
        final int maxX;
        final int minY;
        final int maxY;

        private Footprint(float[] s, int g, int width, int height) {
            int base = g * FIELDS;
            px = s[base + PX];
            py = s[base + PY];
            varX = s[base + VAR_X];
            covXY = s[base + COV_XY];
            varY = s[base + VAR_Y];
            opacity = s[base + OPACITY];
            float det = varX * varY - covXY * covXY;
            detFloored = det < MIN_DET;
            if (detFloored) {
                det = MIN_DET;
            }
            invDet = 1.0f / det;
            int radius = (int) Math.ceil(3.0f * Math.sqrt(Math.max(varX, varY)));
            minX = Math.max(0, (int) (px - radius));
            maxX = Math.min(width - 1, (int) (px + radius));
            minY = Math.max(0, (int) (py - radius));
            maxY = Math.min(height - 1, (int) (py + radius));
        }

        static Footprint of(float[] s, int g, int width, int height) {
            return new Footprint(s, g, width, height);
        }

        /** The box's pixel count, 0 when it is empty. */
        long area() {
            if (maxX < minX || maxY < minY) {
                return 0L;
            }
            return (long) (maxX - minX + 1) * (maxY - minY + 1);
        }

        /** {@code m} at pixel {@code (x, y)}, in the forward's float evaluation order. */
        float mahalanobis(int x, int y) {
            float dx = x - px;
            float dy = y - py;
            return invDet * (varY * dx * dx - 2 * covXY * dx * dy + varX * dy * dy);
        }

        /** {@code exp(-m/2)}, in the forward's float evaluation. */
        float falloff(float mahal) {
            return (float) Math.exp(-0.5f * mahal);
        }

        /** Adds {@code dL/dw} at pixel {@code (x, y)} chained to this splat's row of {@code grad} at {@code base}. */
        void chain(int x, int y, double dLdw, float falloff, float weight, float mahal, double[] grad, int base) {
            double dx = x - px;
            double dy = y - py;
            double inv = invDet;
            double dLdm = -0.5 * dLdw * weight;
            double dmdVarX = inv * dy * dy;
            double dmdVarY = inv * dx * dx;
            double dmdCov = -2.0 * inv * dx * dy;
            if (!detFloored) {
                double m = mahal;
                dmdVarX -= m * inv * varY;
                dmdVarY -= m * inv * varX;
                dmdCov += 2.0 * m * inv * covXY;
            }
            grad[base + PX] += dLdm * inv * -2.0 * (varY * dx - covXY * dy);
            grad[base + PY] += dLdm * inv * -2.0 * (varX * dy - covXY * dx);
            grad[base + VAR_X] += dLdm * dmdVarX;
            grad[base + COV_XY] += dLdm * dmdCov;
            grad[base + VAR_Y] += dLdm * dmdVarY;
            grad[base + OPACITY] += dLdw * falloff;
        }
    }

    /** The splats covering each pixel, in Gaussian order, as a CSR over the row-major pixels. */
    private static final class PixelLists {
        final int[] offsets;
        final int[] ids;

        private PixelLists(int[] offsets, int[] ids) {
            this.offsets = offsets;
            this.ids = ids;
        }

        static PixelLists build(Footprint[] footprints, int width, int height) {
            int pixels = pixelCount(width, height);
            int[] offsets = new int[pixels + 1];
            long total = 0L;
            for (Footprint f : footprints) {
                total += f.area();
                if (total > MAX_LIST_ENTRIES) {
                    throw new IllegalStateException("the splats cover more than " + MAX_LIST_ENTRIES
                        + " pixels in total, too many for the host rasterizer backward");
                }
                for (int y = f.minY; y <= f.maxY; y++) {
                    for (int x = f.minX; x <= f.maxX; x++) {
                        offsets[y * width + x + 1]++;
                    }
                }
            }
            for (int p = 0; p < pixels; p++) {
                offsets[p + 1] += offsets[p];
            }
            int[] cursor = new int[pixels];
            System.arraycopy(offsets, 0, cursor, 0, pixels);
            int[] ids = new int[(int) total];
            for (int g = 0; g < footprints.length; g++) {
                Footprint f = footprints[g];
                for (int y = f.minY; y <= f.maxY; y++) {
                    for (int x = f.minX; x <= f.maxX; x++) {
                        ids[cursor[y * width + x]++] = g;
                    }
                }
            }
            return new PixelLists(offsets, ids);
        }

        /** The length of the longest pixel list. */
        int longest() {
            int longest = 0;
            for (int p = 0; p + 1 < offsets.length; p++) {
                longest = Math.max(longest, offsets[p + 1] - offsets[p]);
            }
            return longest;
        }
    }

    /** Per-entry forward values of one pixel list. */
    private static final class Scratch {
        final float[] mahal;
        final float[] falloff;
        final float[] weight;
        final float[] transmittance;

        Scratch(int capacity) {
            mahal = new float[capacity];
            falloff = new float[capacity];
            weight = new float[capacity];
            transmittance = new float[capacity];
        }
    }
}
