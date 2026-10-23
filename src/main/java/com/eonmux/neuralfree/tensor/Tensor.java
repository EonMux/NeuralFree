package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.autograd.ViewGradFunctions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

/**
 * An n-dimensional array on a {@link ComputeBackend}: a shape, a dtype and strides over a reference-counted
 * {@link TensorStorage}, with the autograd state of a leaf or a recorded node output.
 *
 * <p><b>Views record.</b> {@link #reshape}, {@link #transpose}, {@link #permute}, {@link #slice}, {@link #squeeze},
 * {@link #unsqueeze}, {@link #expand}, {@link #contiguous} and the floating-point casts of {@link #to(DType)} are
 * differentiable: when grad mode is on and this tensor requires grad ({@link Autograd#needsGrad(Tensor)}), the result
 * is a new handle that requires grad and is recorded as the output of a node whose input is this tensor, so a gradient
 * reaching a view of a parameter flows back to the parameter. Otherwise the result does not require grad. Every
 * result is a new handle the caller owns, except where {@link #contiguous} and {@link #to(DType)} return this tensor
 * because there is nothing to compute. {@link #detach()} never records and {@link #withRequiresGrad(boolean)} is the
 * leaf constructor.</p>
 */
public final class Tensor implements AutoCloseable {
    private final Shape shape;
    private final DType dtype;
    private final Strides strides;
    private final TensorStorage storage;
    private final ComputeBackend backend;

    private final boolean requiresGrad;
    private Tensor grad;
    private Object gradFn;

    private volatile boolean closed = false;

    private Tensor(Shape shape, DType dtype, Strides strides, TensorStorage storage, ComputeBackend backend, boolean requiresGrad) {
        this.shape = Objects.requireNonNull(shape, "Shape cannot be null");
        this.dtype = Objects.requireNonNull(dtype, "DType cannot be null");
        this.strides = Objects.requireNonNull(strides, "Strides cannot be null");
        this.storage = Objects.requireNonNull(storage, "Storage cannot be null");
        this.backend = Objects.requireNonNull(backend, "Backend cannot be null");
        this.requiresGrad = requiresGrad;
        this.grad = null;
        this.gradFn = null;
    }

    public static Tensor zeros(ComputeBackend backend, Shape shape) {
        return zeros(backend, shape, DType.FLOAT32);
    }

    /**
     * A zero-filled tensor (every byte 0). FLOAT32, INT32 and BOOL tensors are filled on the device when the train
     * kernels are available ({@link GpuCoreOps#zeros}); other dtypes, and backends without them, upload zeros.
     */
    public static Tensor zeros(ComputeBackend backend, Shape shape, DType dtype) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");

        Tensor device = GpuCoreOps.zeros(backend, shape, dtype);
        if (device != null) {
            return device;
        }
        TensorStorage storage = TensorStorage.allocate(backend, shape.numElements(), dtype);
        try {
            byte[] zeroData = new byte[hostArrayLength(storage.lengthBytes(), "zeros() bytes")];
            closeEvent(storage.copyFromHost(ByteBuffer.wrap(zeroData), backend.transferQueue()));
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
        return new Tensor(shape, dtype, Strides.contiguous(shape), storage, backend, false);
    }

    public static Tensor ones(ComputeBackend backend, Shape shape) {
        return ones(backend, shape, DType.FLOAT32);
    }

    public static Tensor ones(ComputeBackend backend, Shape shape, DType dtype) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");

        if (!dtype.isFloatingPoint()) {
            throw new IllegalArgumentException("ones() is only supported for floating-point dtypes, got: " + dtype);
        }
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("ones() supports FLOAT32 only, got dtype: " + dtype);
        }
        float[] onesData = new float[hostArrayLength(shape.numElements(), "ones() elements")];
        Arrays.fill(onesData, 1.0f);
        return hostFloats(backend, shape, onesData);
    }

    public static Tensor rand(ComputeBackend backend, Shape shape) {
        return rand(backend, shape, DType.FLOAT32, new Random());
    }

    public static Tensor rand(ComputeBackend backend, Shape shape, long seed) {
        return rand(backend, shape, DType.FLOAT32, new Random(seed));
    }

    public static Tensor rand(ComputeBackend backend, Shape shape, DType dtype, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        if (!dtype.isFloatingPoint()) {
            throw new IllegalArgumentException("rand() is only supported for floating-point dtypes, got: " + dtype);
        }
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("rand() supports FLOAT32 only, got dtype: " + dtype);
        }
        float[] randData = new float[hostArrayLength(shape.numElements(), "rand() elements")];
        for (int i = 0; i < randData.length; i++) {
            randData[i] = random.nextFloat();
        }
        return hostFloats(backend, shape, randData);
    }

    public static Tensor randn(ComputeBackend backend, Shape shape) {
        return randn(backend, shape, DType.FLOAT32, new Random());
    }

    public static Tensor randn(ComputeBackend backend, Shape shape, long seed) {
        return randn(backend, shape, DType.FLOAT32, new Random(seed));
    }

    public static Tensor randn(ComputeBackend backend, Shape shape, DType dtype, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        if (!dtype.isFloatingPoint()) {
            throw new IllegalArgumentException("randn() is only supported for floating-point dtypes, got: " + dtype);
        }
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("randn() supports FLOAT32 only, got dtype: " + dtype);
        }
        float[] randnData = new float[hostArrayLength(shape.numElements(), "randn() elements")];
        for (int i = 0; i < randnData.length; i++) {
            randnData[i] = (float) random.nextGaussian();
        }
        return hostFloats(backend, shape, randnData);
    }

    public static Tensor fromArray(ComputeBackend backend, float[] data) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(data, "Data cannot be null");

        Shape shape = Shape.vector(data.length);
        TensorStorage storage = TensorStorage.fromHostFloat(backend, data);
        Strides strides = Strides.contiguous(shape);

        return new Tensor(shape, DType.FLOAT32, strides, storage, backend, false);
    }

    public static Tensor fromArray(ComputeBackend backend, float[][] data) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(data, "Data cannot be null");

        if (data.length == 0) {
            throw new IllegalArgumentException("Data array cannot be empty");
        }

        int rows = data.length;
        int cols = data[0].length;

        for (float[] row : data) {
            if (row.length != cols) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
        }

        Shape shape = Shape.matrix(rows, cols);
        float[] flatData = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flatData, i * cols, cols);
        }

        TensorStorage storage = TensorStorage.fromHostFloat(backend, flatData);
        Strides strides = Strides.contiguous(shape);

        return new Tensor(shape, DType.FLOAT32, strides, storage, backend, false);
    }

    public static Tensor fromArray(ComputeBackend backend, int[] data) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(data, "Data cannot be null");

        Shape shape = Shape.vector(data.length);

        ByteBuffer buffer = ByteBuffer.allocate(data.length * DType.INT32.byteSize())
            .order(ByteOrder.nativeOrder());
        for (int val : data) {
            buffer.putInt(val);
        }
        buffer.flip();

        TensorStorage storage = TensorStorage.fromHost(backend, buffer, DType.INT32);
        Strides strides = Strides.contiguous(shape);

        return new Tensor(shape, DType.INT32, strides, storage, backend, false);
    }

    public static Tensor fromArray(ComputeBackend backend, byte[] data) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(data, "Data cannot be null");

        Shape shape = Shape.vector(data.length);
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder());

        TensorStorage storage = TensorStorage.fromHost(backend, buffer, DType.INT8);
        Strides strides = Strides.contiguous(shape);

        return new Tensor(shape, DType.INT8, strides, storage, backend, false);
    }

    public static Tensor eye(ComputeBackend backend, int n) {
        return eye(backend, n, n, DType.FLOAT32);
    }

    public static Tensor eye(ComputeBackend backend, int rows, int cols) {
        return eye(backend, rows, cols, DType.FLOAT32);
    }

    public static Tensor eye(ComputeBackend backend, int rows, int cols, DType dtype) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Dimensions must be non-negative: rows=" + rows + ", cols=" + cols);
        }
        Objects.requireNonNull(dtype, "DType cannot be null");

        if (!dtype.isFloatingPoint()) {
            throw new IllegalArgumentException("eye() is only supported for floating-point dtypes, got: " + dtype);
        }
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("eye() supports FLOAT32 only, got dtype: " + dtype);
        }
        Shape shape = Shape.matrix(rows, cols);
        float[] eyeData = new float[hostArrayLength(shape.numElements(), "eye() elements")];
        int minDim = Math.min(rows, cols);
        for (int i = 0; i < minDim; i++) {
            eyeData[i * cols + i] = 1.0f;
        }
        return hostFloats(backend, shape, eyeData);
    }

    /**
     * A contiguous tensor over {@code storage}, which it takes ownership of (closing the tensor closes the storage).
     *
     * @throws IllegalArgumentException when the storage belongs to another backend, holds another dtype, or has
     *                                  fewer elements than the shape
     */
    public static Tensor fromStorage(ComputeBackend backend, Shape shape, DType dtype, TensorStorage storage) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");
        Objects.requireNonNull(storage, "Storage cannot be null");
        if (storage.backend() != backend) {
            throw new IllegalArgumentException("Storage belongs to another backend");
        }
        if (storage.dtype() != dtype) {
            throw new IllegalArgumentException("Storage dtype " + storage.dtype() + " does not match " + dtype);
        }
        if (storage.numElements() < shape.numElements()) {
            throw new IllegalArgumentException("Storage of " + storage.numElements() + " elements is smaller than shape "
                + shape);
        }
        return new Tensor(shape, dtype, Strides.contiguous(shape), storage, backend, false);
    }

    /** A tensor over the given view metadata; takes ownership of {@code storage}. */
    static Tensor wrap(Shape shape, DType dtype, Strides strides, TensorStorage storage, ComputeBackend backend,
                       boolean requiresGrad) {
        return new Tensor(shape, dtype, strides, storage, backend, requiresGrad);
    }

    private static Tensor hostFloats(ComputeBackend backend, Shape shape, float[] data) {
        TensorStorage storage = TensorStorage.fromHostFloat(backend, data);
        return new Tensor(shape, DType.FLOAT32, Strides.contiguous(shape), storage, backend, false);
    }

    public static Tensor scalar(ComputeBackend backend, float value) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        return hostFloats(backend, Shape.scalar(), new float[]{value});
    }

    // ------------------------------------------------------------------ views (recorded; see the class description)

    /**
     * A tensor of the same elements in row-major order with a new shape: a view sharing the storage when this
     * tensor is dense ({@link #isDense()}), otherwise a view of a new contiguous copy. Recorded with
     * {@link GradFunctions.ReshapeGrad}.
     */
    public Tensor reshape(Shape newShape) {
        ensureOpen();
        Objects.requireNonNull(newShape, "New shape cannot be null");

        Shape resolvedShape = this.shape.reshape(newShape.dims());
        Shape original = this.shape;
        return recorded(reshapeRaw(resolvedShape), () -> new GradFunctions.ReshapeGrad(original));
    }

    public Tensor reshape(int... dims) {
        return reshape(this.shape.reshape(dims));
    }

    /** The view with every axis reversed ({@code transpose(rank-1, ..., 0)}); the rank must be at least 2. */
    public Tensor transpose() {
        ensureOpen();

        if (shape.rank() < 2) {
            throw new UnsupportedOperationException("Cannot transpose tensor with rank < 2");
        }

        int[] axes = new int[shape.rank()];
        for (int i = 0; i < shape.rank(); i++) {
            axes[i] = shape.rank() - 1 - i;
        }

        return transpose(axes);
    }

    /**
     * The view whose axis {@code i} is this tensor's axis {@code axes[i]}; {@code axes} is a permutation of
     * {@code [0, rank)}. Recorded with {@link GradFunctions.TransposeGrad}.
     */
    public Tensor transpose(int... axes) {
        ensureOpen();
        Objects.requireNonNull(axes, "Axes cannot be null");

        if (axes.length != shape.rank()) {
            throw new IllegalArgumentException("Number of axes (" + axes.length +
                ") must match rank (" + shape.rank() + ")");
        }

        int[] order = axes.clone();
        Tensor raw = view(shape.transpose(order), strides.transpose(order));
        return recorded(raw, () -> new GradFunctions.TransposeGrad(order));
    }

    public Tensor permute(int... axes) {
        return transpose(axes);
    }

    /**
     * A view of {@code [start, end)} along {@code axis} (negative bounds count from the end); {@code start == end}
     * gives an empty view. It shares the storage unchanged; the start is carried only in the strides' element
     * offset. Recorded with {@link GradFunctions.SliceGrad}.
     */
    public Tensor slice(int axis, long start, long end) {
        ensureOpen();

        if (axis < 0 || axis >= shape.rank()) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + shape.rank());
        }
        if (start < 0) {
            start = shape.dim(axis) + start;
        }
        if (end < 0) {
            end = shape.dim(axis) + end;
        }
        if (start < 0 || end > shape.dim(axis) || start > end) {
            throw new IllegalArgumentException("Invalid slice bounds: start=" + start + ", end=" + end +
                ", dim=" + shape.dim(axis));
        }

        int[] newDims = shape.dims();
        newDims[axis] = (int) (end - start);

        Tensor raw = view(Shape.of(newDims), strides.slice(axis, start, end));
        Shape original = this.shape;
        int first = (int) start;
        int stop = (int) end;
        return recorded(raw, () -> new GradFunctions.SliceGrad(original, axis, first, stop));
    }

    /** The view without the axes of size 1. Recorded with {@link GradFunctions.ReshapeGrad}. */
    public Tensor squeeze() {
        ensureOpen();

        Shape newShape = shape.squeeze();
        long[] source = strides.strides();
        long[] kept = new long[newShape.rank()];
        int next = 0;
        for (int axis = 0; axis < shape.rank(); axis++) {
            if (shape.dim(axis) != 1) {
                kept[next++] = source[axis];
            }
        }
        Strides newStrides = Strides.of(kept, strides.offset());

        Shape original = this.shape;
        return recorded(view(newShape, newStrides), () -> new GradFunctions.ReshapeGrad(original));
    }

    /** The view without axis {@code axis} (negative counts from the end), which must have size 1. */
    public Tensor squeeze(int axis) {
        ensureOpen();

        if (axis < 0) {
            axis = shape.rank() + axis;
        }
        if (axis < 0 || axis >= shape.rank()) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + shape.rank());
        }

        Shape newShape = shape.squeeze(axis);
        Strides newStrides = strides.squeeze(axis);

        Shape original = this.shape;
        return recorded(view(newShape, newStrides), () -> new GradFunctions.ReshapeGrad(original));
    }

    /** The view with a new axis of size 1 at {@code axis} (in {@code [-rank-1, rank]}). */
    public Tensor unsqueeze(int axis) {
        ensureOpen();

        if (axis < 0) {
            axis = shape.rank() + axis + 1;
        }
        if (axis < 0 || axis > shape.rank()) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for unsqueeze (valid: 0 to " + shape.rank() + ")");
        }

        Shape newShape = shape.unsqueeze(axis);
        Strides newStrides = strides.unsqueeze(axis);

        Shape original = this.shape;
        return recorded(view(newShape, newStrides), () -> new GradFunctions.ReshapeGrad(original));
    }

    /**
     * A view broadcast to {@code targetShape}: prepended dims and dims of size 1 that grow get stride 0, so every
     * broadcast index reads the same element ({@link BroadcastEngine#broadcastStrides}). Recorded with
     * {@link ViewGradFunctions.ExpandGrad}.
     *
     * @throws IllegalArgumentException when the shape does not broadcast to {@code targetShape}
     */
    public Tensor expand(Shape targetShape) {
        ensureOpen();
        Objects.requireNonNull(targetShape, "Target shape cannot be null");

        if (!BroadcastEngine.isBroadcastableTo(this.shape, targetShape)) {
            throw new IllegalArgumentException("Cannot expand shape " + shape + " to " + targetShape);
        }
        long[] expanded = BroadcastEngine.broadcastStrides(this.shape, targetShape, strides);
        Tensor raw = view(targetShape, Strides.of(expanded, strides.offset()));
        Shape original = this.shape;
        return recorded(raw, () -> new ViewGradFunctions.ExpandGrad(original));
    }

    public Tensor expand(int... dims) {
        return expand(Shape.of(dims));
    }

    /**
     * This tensor itself when it is contiguous ({@link #isContiguous()}), with nothing recorded; otherwise a new
     * row-major copy, made on the device by {@link GpuCoreOps#contiguous} when possible, else gathered on the host and
     * uploaded, and recorded with {@link ViewGradFunctions.IdentityGrad}.
     */
    public Tensor contiguous() {
        ensureOpen();

        if (strides.isContiguous(shape)) {
            return this;
        }
        return recorded(contiguousCopy(), ViewGradFunctions.IdentityGrad::new);
    }

    /**
     * {@code raw}, a new result computed from this tensor that does not require grad, returned as it is when an
     * operation on this tensor does not record ({@link Autograd#needsGrad(Tensor)}); otherwise handed to
     * {@link Autograd#record} with this tensor as the only input and the gradient function {@code gradFn} builds.
     * The raw helpers run no recording operation, so they need no {@link Autograd#untracked} scope.
     */
    private Tensor recorded(Tensor raw, Supplier<GradientFunction> gradFn) {
        if (!Autograd.needsGrad(this)) {
            return raw;
        }
        return Autograd.record(raw, Collections.singletonList(this), out -> gradFn.get());
    }

    /** A view over this tensor's storage (a new reference) that does not require grad. */
    private Tensor view(Shape viewShape, Strides viewStrides) {
        return new Tensor(viewShape, dtype, viewStrides, storage.share(), backend, false);
    }

    /** The raw {@link #reshape}: {@code resolvedShape} has this tensor's element count. */
    private Tensor reshapeRaw(Shape resolvedShape) {
        if (isDense()) {
            return view(resolvedShape, Strides.contiguous(resolvedShape).withOffset(strides.offset()));
        }
        Tensor dense = contiguousCopy();
        return new Tensor(resolvedShape, dtype, Strides.contiguous(resolvedShape), dense.storage, backend, false);
    }

    /** A new row-major copy that does not require grad: on the device when possible, else a host gather. */
    private Tensor contiguousCopy() {
        Tensor device = GpuCoreOps.contiguous(this);
        if (device != null) {
            return device;
        }
        TensorStorage newStorage = TensorStorage.fromHost(backend, ByteBuffer.wrap(denseBytes()), dtype);
        return new Tensor(shape, dtype, Strides.contiguous(shape), newStorage, backend, false);
    }

    /** The FLOAT32 elements in row-major order. A strided view is copied contiguously on the device first when possible. */
    public float[] toFloatArray() {
        ensureOpen();

        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("toFloatArray() only supported for FLOAT32 tensors, got: " + dtype);
        }
        long n = numElements();
        hostArrayLength(n, "toFloatArray() elements");
        if (n == 0) {
            return new float[0];
        }
        ComputeQueue queue = backend.transferQueue();
        if (isDense()) {
            return storage.copyToHostFloat(strides.offset(), n, queue);
        }
        Tensor dense = GpuCoreOps.contiguous(this);
        if (dense != null) {
            try {
                return dense.storage.copyToHostFloat(0, n, queue);
            } finally {
                dense.close();
            }
        }
        ByteBuffer bytes = ByteBuffer.wrap(gatherBytes(queue)).order(ByteOrder.nativeOrder());
        float[] result = new float[(int) n];
        bytes.asFloatBuffer().get(result);
        return result;
    }

    /** The elements' bytes in row-major order, in native byte order, positioned at 0. */
    public ByteBuffer toByteBuffer() {
        ensureOpen();
        return ByteBuffer.wrap(denseBytes()).order(ByteOrder.nativeOrder());
    }

    /** The INT32 elements, or the INT64 elements narrowed by Java {@code (int)}, in row-major order. */
    public int[] toIntArray() {
        ensureOpen();

        if (dtype != DType.INT32 && dtype != DType.INT64) {
            throw new UnsupportedOperationException("toIntArray() only supported for INT32 or INT64 tensors, got: " + dtype);
        }

        ByteBuffer buffer = toByteBuffer();
        int[] result = new int[(int) shape.numElements()];
        if (dtype == DType.INT32) {
            buffer.asIntBuffer().get(result);
        } else {
            for (int i = 0; i < result.length; i++) {
                result[i] = (int) buffer.getLong(i * Long.BYTES);
            }
        }
        return result;
    }

    /**
     * This tensor converted to {@code targetDType}; this tensor itself, with nothing recorded, when the dtype already
     * matches. Supported pairs: FLOAT32 to and from FLOAT64 (host), INT32, BOOL, FLOAT16 and BFLOAT16, and INT64 to
     * INT32. Every pair except FLOAT64 converts on the device when the train kernels are available
     * ({@link GpuCoreOps#cast}); the host path computes the same bits (Java casts, {@link HalfFloats}). Casts among
     * FLOAT32, FLOAT16 and BFLOAT16 are recorded with {@link ViewGradFunctions.CastGrad} (the gradient reaching the
     * source is FLOAT32); every other cast is not differentiable and its result never requires grad.
     *
     * @throws UnsupportedOperationException for any other pair
     */
    public Tensor to(DType targetDType) {
        ensureOpen();
        Objects.requireNonNull(targetDType, "Target DType cannot be null");

        if (this.dtype == targetDType) {
            return this;
        }
        if (!isSupportedConversion(this.dtype, targetDType)) {
            throw new UnsupportedOperationException("DType conversion from " + dtype + " to " + targetDType
                + " is not supported; supported conversions: FLOAT32 <-> FLOAT64, FLOAT32 <-> INT32, FLOAT32 <-> BOOL,"
                + " FLOAT32 <-> FLOAT16, FLOAT32 <-> BFLOAT16, INT64 -> INT32");
        }
        Tensor raw = convertRaw(targetDType);
        if (!isDifferentiableFloat(this.dtype) || !isDifferentiableFloat(targetDType)) {
            return raw;
        }
        DType source = this.dtype;
        return recorded(raw, () -> new ViewGradFunctions.CastGrad(source, targetDType));
    }

    /** The raw {@link #to}: a new tensor of a supported pair that does not require grad, on the device when possible. */
    private Tensor convertRaw(DType target) {
        if (GpuCoreOps.castKernel(this.dtype, target) != null) {
            Tensor device = GpuCoreOps.cast(this, target);
            if (device != null) {
                return device;
            }
        }
        return hostConvert(target);
    }

    /** The dtypes whose casts among each other are differentiable: FLOAT32, FLOAT16 and BFLOAT16. */
    private static boolean isDifferentiableFloat(DType d) {
        return d == DType.FLOAT32 || d == DType.FLOAT16 || d == DType.BFLOAT16;
    }

    private static boolean isSupportedConversion(DType from, DType to) {
        if (from == DType.INT64 && to == DType.INT32) {
            return true;
        }
        DType other = from == DType.FLOAT32 ? to : (to == DType.FLOAT32 ? from : null);
        return other == DType.FLOAT64 || other == DType.INT32 || other == DType.BOOL || other == DType.FLOAT16
            || other == DType.BFLOAT16;
    }

    /** The host path of {@link #to}: element-wise Java conversion of the row-major elements, then one upload. */
    private Tensor hostConvert(DType target) {
        ByteBuffer in = toByteBuffer();
        int n = (int) numElements();
        ByteBuffer out = ByteBuffer.allocate(hostArrayLength((long) n * target.byteSize(), "converted bytes"))
            .order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            if (dtype == DType.FLOAT32) {
                int bits = in.getInt(i * 4);
                float x = Float.intBitsToFloat(bits);
                switch (target) {
                    case FLOAT64: out.putDouble(i * 8, x); break;
                    case INT32: out.putInt(i * 4, (int) x); break;
                    case BOOL: out.put(i, (byte) (x != 0.0f ? 1 : 0)); break;
                    case FLOAT16: out.putShort(i * 2, (short) HalfFloats.floatBitsToHalf(bits)); break;
                    case BFLOAT16: out.putShort(i * 2, (short) HalfFloats.floatBitsToBFloat16(bits)); break;
                    default: throw new IllegalStateException("unreachable conversion to " + target);
                }
            } else {
                switch (dtype) {
                    case FLOAT64: out.putFloat(i * 4, (float) in.getDouble(i * 8)); break;
                    case INT32: out.putFloat(i * 4, (float) in.getInt(i * 4)); break;
                    case BOOL: out.putFloat(i * 4, in.get(i) != 0 ? 1.0f : 0.0f); break;
                    case FLOAT16: out.putInt(i * 4, HalfFloats.halfToFloatBits(in.getShort(i * 2) & 0xffff)); break;
                    case BFLOAT16:
                        out.putInt(i * 4, HalfFloats.bfloat16ToFloatBits(in.getShort(i * 2) & 0xffff));
                        break;
                    case INT64: out.putInt(i * 4, (int) in.getLong(i * 8)); break;
                    default: throw new IllegalStateException("unreachable conversion from " + dtype);
                }
            }
        }
        TensorStorage newStorage = TensorStorage.fromHost(backend, out, target);
        return new Tensor(shape, target, Strides.contiguous(shape), newStorage, backend, false);
    }

    /**
     * The elements' bytes in row-major order: one read of a dense view, else one read of a device contiguous copy
     * (the 4- and 8-byte dtypes {@link GpuCoreOps#contiguous} copies), else a host gather.
     */
    private byte[] denseBytes() {
        long n = numElements();
        int elementBytes = dtype.byteSize();
        int total = hostArrayLength(n * elementBytes, "tensor bytes");
        if (n == 0) {
            return new byte[0];
        }
        ComputeQueue queue = backend.transferQueue();
        if (isDense()) {
            return storage.copyToHostBytes(strides.offset() * elementBytes, total, queue);
        }
        Tensor dense = GpuCoreOps.contiguous(this);
        if (dense != null) {
            try {
                return dense.storage.copyToHostBytes(0, total, queue);
            } finally {
                dense.close();
            }
        }
        return gatherBytes(queue);
    }

    /**
     * Reads the storage range the view reaches (from its offset to its highest element) and gathers the elements
     * in row-major order; strides index from that offset, so stride-0 and transposed views read correctly.
     */
    private byte[] gatherBytes(ComputeQueue queue) {
        int rank = shape.rank();
        long n = numElements();
        int elementBytes = dtype.byteSize();
        long[] st = strides.strides();
        int[] dims = shape.dims();
        long span = 0;
        for (int k = 0; k < rank; k++) {
            span += (long) (dims[k] - 1) * st[k];
        }
        byte[] raw = storage.copyToHostBytes(strides.offset() * elementBytes,
            hostArrayLength((span + 1) * elementBytes, "strided view span bytes"), queue);
        byte[] out = new byte[hostArrayLength(n * elementBytes, "tensor bytes")];
        int[] idx = new int[rank];
        long src = 0;
        for (int i = 0; i < n; i++) {
            System.arraycopy(raw, (int) (src * elementBytes), out, i * elementBytes, elementBytes);
            for (int k = rank - 1; k >= 0; k--) {
                src += st[k];
                if (++idx[k] < dims[k]) {
                    break;
                }
                src -= (long) dims[k] * st[k];
                idx[k] = 0;
            }
        }
        return out;
    }

    /** A host array length, checked against the Java array limit. */
    private static int hostArrayLength(long n, String what) {
        if (n < 0 || n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(what + " (" + n + ") exceed the Java array limit");
        }
        return (int) n;
    }

    private static void closeEvent(GpuEvent event) {
        if (event != null) {
            event.close();
        }
    }

    public Shape shape() {
        return shape;
    }

    public DType dtype() {
        return dtype;
    }

    public Strides strides() {
        return strides;
    }

    public TensorStorage storage() {
        ensureOpen();
        return storage;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public long numElements() {
        return shape.numElements();
    }

    public long byteSize() {
        return shape.byteSize(dtype);
    }

    /** Row-major dense with element offset 0 from the storage start. */
    public boolean isContiguous() {
        return strides.isContiguous(shape);
    }

    /** Row-major dense, possibly starting at a non-zero element offset ({@code strides().offset()}). */
    public boolean isDense() {
        if (strides.rank() != shape.rank()) {
            return false;
        }
        long expected = 1;
        for (int k = shape.rank() - 1; k >= 0; k--) {
            int dim = shape.dim(k);
            if (dim == 0) {
                return true;
            }
            if (dim != 1 && strides.stride(k) != expected) {
                return false;
            }
            expected *= dim;
        }
        return true;
    }

    public boolean isScalar() {
        return shape.isScalar();
    }

    public boolean isVector() {
        return shape.isVector();
    }

    public boolean isMatrix() {
        return shape.isMatrix();
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /**
     * This tensor when the flag already matches, otherwise a new handle over the same storage with the flag set and
     * no gradient function: with {@code true}, applied to a tensor without a gradient function, this is how a leaf
     * is created ({@code detach()} a recorded tensor first). Never recorded.
     */
    public Tensor withRequiresGrad(boolean requiresGrad) {
        ensureOpen();

        if (this.requiresGrad == requiresGrad) {
            return this;
        }

        return new Tensor(shape, dtype, strides, storage.share(), backend, requiresGrad);
    }

    public Tensor grad() {
        return grad;
    }

    public void setGrad(Tensor grad) {
        if (!requiresGrad) {
            throw new IllegalStateException("Cannot set gradient on tensor that does not require grad");
        }

        if (grad != null && !grad.shape.equals(this.shape)) {
            throw new IllegalArgumentException("Gradient shape " + grad.shape + " does not match tensor shape " + this.shape);
        }

        this.grad = grad;
    }

    public Object gradFn() {
        return gradFn;
    }

    public void setGradFn(Object gradFn) {
        this.gradFn = gradFn;
    }

    /** A new handle over the same storage that does not require grad and has no gradient function; never recorded. */
    public Tensor detach() {
        ensureOpen();
        return new Tensor(shape, dtype, strides, storage.share(), backend, false);
    }

    public void backward() {
        if (!requiresGrad) {
            throw new IllegalStateException("Cannot call backward() on tensor that does not require grad");
        }
        ComputationGraph.current().backward(this);
    }

    public void backward(Tensor gradOutput) {
        if (!requiresGrad) {
            throw new IllegalStateException("Cannot call backward() on tensor that does not require grad");
        }
        Objects.requireNonNull(gradOutput, "Gradient output cannot be null");
        ComputationGraph.current().backward(this, gradOutput);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (storage != null) {
                storage.close();
            }
        }
    }

    @Override
    public String toString() {
        return "Tensor{" +
            "shape=" + shape +
            ", dtype=" + dtype +
            ", strides=" + strides +
            ", requiresGrad=" + requiresGrad +
            ", closed=" + closed +
            '}';
    }
}
