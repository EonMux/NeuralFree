package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;
import java.util.Random;

/**
 * 3D Gaussian splatting renderer: per-Gaussian means, scales, rotation quaternions {@code (w, x, y, z)}, opacities
 * and spherical-harmonic colour coefficients (the first three, through a sigmoid, give the base colour).
 *
 * <p>The per-Gaussian projection runs as tensor ops on the parameters' backend, on the device when the train kernels
 * are available there: the view transform {@code [m, 1] @ pose^T} (a GEMM), the perspective divide, the screen
 * mapping, the leading 2x2 block of the 3D covariance {@code (R S)(R S)^T} (a GEMM for the rotation rows and a
 * batched matmul), the variance floor {@code max(c, 0.001)} and the opacity and colour sigmoids. The sums are in
 * the GEMM tolerance class and the sigmoids in the transcendental class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. The rasterizer is sequential per-pixel compositing
 * and stays on the host.</p>
 *
 * <p><b>Gradients.</b> The projection is a recorded composition, and the rasterizer is one recorded node,
 * {@link RasterizeGrad}, whose input is the projected {@code [numGaussians, 9]} splat tensor: its backward rebuilds the
 * per-pixel composites on the host and returns the splat gradient, which the projection's nodes carry to the means,
 * scales, rotations, opacities, the first three spherical-harmonic coefficients and a pose that requires grad. The
 * three-sigma box bounds are constants. The rasterizer node does not support a create-graph pass.</p>
 */
public final class GaussianSplatting extends Module {
    private static final int SH_DEGREE = 3;
    private static final int SH_COEFFS_PER_GAUSSIAN = 3 * ((SH_DEGREE + 1) * (SH_DEGREE + 1));
    private static final float MIN_SCALE = 0.001f;
    private static final float MIN_VARIANCE = 0.001f;

    /**
     * Coefficients of {@code q_a q_b} (row {@code 4a + b}, quaternion order w, x, y, z) in the rotation entries
     * {@code R00, R01, R02, R10, R11, R12} (columns): {@code R00 = 1 - 2yy - 2zz}, {@code R01 = 2xy - 2zw},
     * {@code R02 = 2xz + 2yw}, {@code R10 = 2xy + 2zw}, {@code R11 = 1 - 2xx - 2zz}, {@code R12 = 2yz - 2xw}.
     */
    private static final float[] ROTATION_COEFFICIENTS = rotationCoefficients();
    /** The constant parts of {@code R00 .. R12}. */
    private static final float[] ROTATION_OFFSETS = {1, 0, 0, 0, 1, 0};

    private final int numGaussians;
    private final int imageWidth;
    private final int imageHeight;

    private Parameter means;
    private Parameter scales;
    private Parameter rotations;
    private Parameter opacities;
    private Parameter shCoeffs;

    private boolean initialized = false;

    public GaussianSplatting(int numGaussians, int imageWidth, int imageHeight) {
        if (numGaussians <= 0) {
            throw new IllegalArgumentException("numGaussians must be positive: " + numGaussians);
        }
        if (imageWidth <= 0) {
            throw new IllegalArgumentException("imageWidth must be positive: " + imageWidth);
        }
        if (imageHeight <= 0) {
            throw new IllegalArgumentException("imageHeight must be positive: " + imageHeight);
        }

        this.numGaussians = numGaussians;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        float[] meansData = new float[numGaussians * 3];
        for (int i = 0; i < numGaussians * 3; i++) {
            meansData[i] = (random.nextFloat() - 0.5f) * 2.0f;
        }
        this.means = registerParameter("means", parameter(backend, Shape.matrix(numGaussians, 3), meansData, "means"));

        float[] scalesData = new float[numGaussians * 3];
        for (int i = 0; i < numGaussians * 3; i++) {
            scalesData[i] = 0.01f + random.nextFloat() * 0.1f;
        }
        this.scales = registerParameter("scales", parameter(backend, Shape.matrix(numGaussians, 3), scalesData,
            "scales"));

        float[] rotationsData = new float[numGaussians * 4];
        for (int i = 0; i < numGaussians; i++) {
            rotationsData[i * 4] = 1.0f;
        }
        this.rotations = registerParameter("rotations", parameter(backend, Shape.matrix(numGaussians, 4),
            rotationsData, "rotations"));

        float[] opacitiesData = new float[numGaussians];
        Arrays.fill(opacitiesData, 0.5f);
        this.opacities = registerParameter("opacities", parameter(backend, Shape.matrix(numGaussians, 1),
            opacitiesData, "opacities"));

        float[] shCoeffsData = new float[numGaussians * SH_COEFFS_PER_GAUSSIAN];
        for (int i = 0; i < shCoeffsData.length; i++) {
            shCoeffsData[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }
        this.shCoeffs = registerParameter("shCoeffs", parameter(backend,
            Shape.matrix(numGaussians, SH_COEFFS_PER_GAUSSIAN), shCoeffsData, "shCoeffs"));

        initialized = true;
    }

    @Override
    public Tensor forward(Tensor cameraPose) {
        return render(cameraPose);
    }

    /**
     * Renders the Gaussians seen through {@code cameraPose} ({@code [4, 4]}, row-major) into a new
     * {@code [imageHeight, imageWidth, 4]} RGBA image on the pose's backend.
     *
     * <p>The per-Gaussian projection runs on the parameters' backend as tensor ops (see the class description) and is
     * read back once as a {@code [numGaussians, 9]} block; the rasterizer, a sequential per-pixel alpha composite in
     * Gaussian order, runs on the host ({@link RasterizeGrad}). When the projection requires grad
     * in grad mode the image is the output of a recorded {@link RasterizeGrad} node over it. A pose on another backend
     * is copied to the parameters' backend first, which cuts the pose's own gradient.</p>
     *
     * @throws IllegalArgumentException when the pose is not {@code [4, 4]}
     */
    public Tensor render(Tensor cameraPose) {
        ensureInitialized();
        Objects.requireNonNull(cameraPose, "cameraPose cannot be null");
        Shape poseShape = cameraPose.shape();
        if (poseShape.rank() != 2 || poseShape.dim(0) != 4 || poseShape.dim(1) != 4) {
            throw new IllegalArgumentException("cameraPose must be [4, 4], got: " + poseShape);
        }

        ComputeBackend backend = means.tensor().backend();
        if (cameraPose.backend() == backend) {
            return rasterized(cameraPose, cameraPose.backend());
        }
        try (Tensor pose = upload(backend, Shape.matrix(4, 4), cameraPose.toFloatArray())) {
            return rasterized(pose, cameraPose.backend());
        }
    }

    /**
     * The image of the splats projected through {@code pose}, uploaded to {@code imageBackend}: the projection is read
     * back once, rasterized on the host, and recorded as a {@link RasterizeGrad} node over the projection when it
     * requires grad.
     */
    private Tensor rasterized(Tensor pose, ComputeBackend imageBackend) {
        try (Tensor packed = projectSplats(pose)) {
            float[] splats = packed.toFloatArray();
            float[] image = RasterizeGrad.rasterize(splats, numGaussians, imageWidth, imageHeight);
            Tensor raw = upload(imageBackend, Shape.of(imageHeight, imageWidth, 4), image);
            if (!Autograd.needsGrad(packed)) {
                return raw;
            }
            ComputeBackend splatBackend = packed.backend();
            return Autograd.record(raw, Collections.singletonList(packed),
                out -> new RasterizeGrad(splats, numGaussians, imageWidth, imageHeight, splatBackend));
        }
    }

    /**
     * The screen-space splats as a new {@code [numGaussians, 9]} tensor of rows
     * {@code [px, py, varX, covXY, varY, opacity, r, g, b]} on the parameters' backend, recorded when a parameter or
     * the pose requires grad.
     */
    private Tensor projectSplats(Tensor pose) {
        try (Tensor screen = screenPositions(pose);
             Tensor covariance = screenCovariance();
             Tensor varX = clampedColumn(covariance, 0);
             Tensor covXY = covariance.slice(1, 1, 2);
             Tensor varY = clampedColumn(covariance, 3);
             Tensor opacity = TensorOps.sigmoid(opacities.tensor());
             Tensor baseColor = shCoeffs.tensor().slice(1, 0, 3);
             Tensor color = TensorOps.sigmoid(baseColor)) {
            return TensorOps.cat(Arrays.asList(screen, varX, covXY, varY, opacity, color), 1);
        }
    }

    /**
     * {@code [numGaussians, 2]} pixel positions: {@code v = pose @ [m, 1]}, divided by {@code v.w} where it is
     * non-zero, then {@code px = (vx + 1) * imageWidth / 2} and {@code py = (1 - vy) * imageHeight / 2}. The divisor
     * is 1 where {@code v.w} is zero, so the unselected quotient there is finite and its zero gradient stays zero.
     */
    private Tensor screenPositions(Tensor pose) {
        ComputeBackend backend = pose.backend();
        Shape plane = Shape.matrix(numGaussians, 2);
        try (Tensor ones = Tensor.ones(backend, Shape.matrix(numGaussians, 1));
             Tensor homogeneous = TensorOps.cat(Arrays.asList(means.tensor(), ones), 1);
             Tensor poseT = pose.transpose();
             Tensor view = TensorOps.matmul(homogeneous, poseT);
             Tensor vxy = view.slice(1, 0, 2);
             Tensor vw = view.slice(1, 3, 4);
             Tensor zero = Tensor.scalar(backend, 0.0f);
             Tensor w = vw.expand(plane);
             Tensor atInfinity = TensorOps.eq(w, zero);
             Tensor unit = Tensor.ones(backend, plane);
             Tensor divisor = TensorOps.where(atInfinity, unit, w);
             Tensor ratio = TensorOps.div(vxy, divisor);
             Tensor ndc = TensorOps.where(atInfinity, vxy, ratio);
             Tensor signs = upload(backend, Shape.vector(2), new float[]{1.0f, -1.0f});
             Tensor flipped = TensorOps.mul(ndc, signs);
             Tensor shifted = TensorOps.add(flipped, 1.0f);
             Tensor extent = upload(backend, Shape.vector(2), new float[]{imageWidth, imageHeight});
             Tensor scaled = TensorOps.mul(shifted, extent)) {
            return TensorOps.mul(scaled, 0.5f);
        }
    }

    /**
     * {@code [numGaussians, 4]} rows {@code [c00, c01, c10, c11]} of the leading 2x2 block of {@code (R S)(R S)^T},
     * with {@code R} the rotation of the quaternion {@code (w, x, y, z)} and {@code S = diag(max(scale, 0.001))}.
     * The first two rows of {@code R} are the products {@code q_a q_b} times a constant coefficient matrix plus the
     * identity's entries.
     */
    private Tensor screenCovariance() {
        Tensor q = rotations.tensor();
        ComputeBackend backend = q.backend();
        try (Tensor qColumn = q.reshape(Shape.of(numGaussians, 4, 1));
             Tensor qRow = q.reshape(Shape.of(numGaussians, 1, 4));
             Tensor products = TensorOps.mul(qColumn, qRow);
             Tensor flatProducts = products.reshape(Shape.matrix(numGaussians, 16));
             Tensor coefficients = upload(backend, Shape.matrix(16, 6), ROTATION_COEFFICIENTS);
             Tensor linear = TensorOps.matmul(flatProducts, coefficients);
             Tensor identity = upload(backend, Shape.vector(6), ROTATION_OFFSETS);
             Tensor rotationRows = TensorOps.add(linear, identity);
             Tensor r = rotationRows.reshape(Shape.of(numGaussians, 2, 3));
             Tensor clampedScales = TensorOps.clamp(scales.tensor(), MIN_SCALE, Float.POSITIVE_INFINITY);
             Tensor s = clampedScales.reshape(Shape.of(numGaussians, 1, 3));
             Tensor rs = TensorOps.mul(r, s);
             Tensor rsT = rs.transpose(0, 2, 1);
             Tensor covariance = TensorOps.matmul(rs, rsT)) {
            return covariance.reshape(Shape.matrix(numGaussians, 4));
        }
    }

    private static Tensor clampedColumn(Tensor covariance, int column) {
        try (Tensor entry = covariance.slice(1, column, column + 1)) {
            return TensorOps.clamp(entry, MIN_VARIANCE, Float.POSITIVE_INFINITY);
        }
    }

    /** A trainable parameter that takes over the uploaded {@code data} (the parameter makes the leaf). */
    private static Parameter parameter(ComputeBackend backend, Shape shape, float[] data, String name) {
        return new Parameter(upload(backend, shape, data), name);
    }

    private static float[] rotationCoefficients() {
        final int w = 0;
        final int x = 1;
        final int y = 2;
        final int z = 3;
        float[] c = new float[16 * 6];
        setCoefficient(c, y, y, 0, -2);
        setCoefficient(c, z, z, 0, -2);
        setCoefficient(c, x, y, 1, 2);
        setCoefficient(c, z, w, 1, -2);
        setCoefficient(c, x, z, 2, 2);
        setCoefficient(c, y, w, 2, 2);
        setCoefficient(c, x, y, 3, 2);
        setCoefficient(c, z, w, 3, 2);
        setCoefficient(c, x, x, 4, -2);
        setCoefficient(c, z, z, 4, -2);
        setCoefficient(c, y, z, 5, 2);
        setCoefficient(c, x, w, 5, -2);
        return c;
    }

    private static void setCoefficient(float[] c, int a, int b, int entry, float value) {
        c[(4 * a + b) * 6 + entry] = value;
    }

    private static Tensor upload(ComputeBackend backend, Shape shape, float[] data) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    public Tensor getMeans() {
        ensureInitialized();
        return means.tensor();
    }

    public Tensor getScales() {
        ensureInitialized();
        return scales.tensor();
    }

    public Tensor getRotations() {
        ensureInitialized();
        return rotations.tensor();
    }

    public Tensor getOpacities() {
        ensureInitialized();
        return opacities.tensor();
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("GaussianSplatting not initialized. Call initialize(backend) first.");
        }
    }

    public int numGaussians() { return numGaussians; }
    public int imageWidth() { return imageWidth; }
    public int imageHeight() { return imageHeight; }
    public Parameter meansParam() { ensureInitialized(); return means; }
    public Parameter scalesParam() { ensureInitialized(); return scales; }
    public Parameter rotationsParam() { ensureInitialized(); return rotations; }
    public Parameter opacitiesParam() { ensureInitialized(); return opacities; }
    public Parameter shCoeffsParam() { ensureInitialized(); return shCoeffs; }

    @Override
    public String toString() {
        return "GaussianSplatting{" +
            "numGaussians=" + numGaussians +
            ", imageWidth=" + imageWidth +
            ", imageHeight=" + imageHeight +
            ", initialized=" + initialized +
            '}';
    }
}
