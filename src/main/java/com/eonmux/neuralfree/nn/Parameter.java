package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A trainable tensor of a module, with its gradient slot.
 *
 * <p><b>Leaf.</b> The tensor of a parameter is always an autograd leaf: it has no gradient function, and it requires
 * grad exactly when the parameter is trainable ({@link #requiresGrad()}, {@link #setRequiresGrad(boolean)}). Every
 * path that replaces the tensor (construction, {@link #to(ComputeBackend)}, {@link #to(DType)}, moves, conversions,
 * loads, {@link #detach()}, {@link #copy()}, {@link #copyFrom(Tensor)}) rebuilds such a leaf, so a frozen parameter
 * stays frozen through moves, loads and copies. A tensor handed over with a gradient function (the output of a
 * recorded operation) is detached first. Conversions and copies run under {@link ComputationGraph#noGrad()} and
 * record nothing; loads ({@link Module#loadStateDict}) and {@link #copyFrom(Tensor)} are assignments, like
 * PyTorch's {@code copy_} under {@code no_grad}.</p>
 *
 * <p><b>Ownership.</b> Device buffers have no GC-based release, so a parameter owns its tensor and its gradient. The
 * constructors take over the tensor they are given: the parameter closes it in {@link #close()}, and when the leaf
 * needs a new handle the parameter releases the given reference at once, so the caller uses {@link #tensor()}
 * afterwards and never the tensor it passed. The gradient slot owns its gradient: {@link #setGrad} takes over the
 * tensor it is given and closes the gradient it replaces, {@link #zeroGrad()} closes the gradient it clears, and
 * {@link #close()} closes the gradient too. Autograd gives every gradient slot its own tensor object, and closing is
 * idempotent, so a gradient tensor is never shared between slots unless a caller hands one object to several of
 * them.</p>
 *
 * <p><b>Substitution.</b> {@link #substitute(Tensor)} lets a forward pass read another tensor in place of the owned
 * one (spectral normalization, second-order meta-learning, functional evaluation with other weights); the gradient
 * slot keeps addressing the owned tensor. Scopes nest, so a module that substitutes its own weight (spectral
 * normalization) runs inside an outer substitution of the same parameter (MAML's adapted weights).</p>
 */
public final class Parameter implements AutoCloseable {
    private Tensor tensor;
    private final String name;
    private boolean trainable;
    private volatile ActiveSubstitution substitution;
    private volatile boolean closed = false;

    /** An unnamed trainable parameter that takes over {@code tensor} (see the class description). */
    public Parameter(Tensor tensor) {
        this(tensor, null);
    }

    /**
     * A trainable parameter named {@code name} that takes over {@code tensor} (see the class description).
     *
     * @throws NullPointerException  when {@code tensor} is null
     * @throws IllegalStateException when {@code tensor} is closed
     */
    public Parameter(Tensor tensor, String name) {
        this(tensor, name, true);
    }

    private Parameter(Tensor tensor, String name, boolean trainable) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        this.tensor = asLeaf(tensor, trainable);
        this.name = name;
        this.trainable = trainable;
    }

    /**
     * The tensor forward passes read: the substitute while a {@link #substitute(Tensor)} scope is open, else the
     * owned leaf.
     */
    public Tensor tensor() {
        ensureOpen();
        ActiveSubstitution active = substitution;
        return active != null ? active.value : tensor;
    }

    /** The owned leaf, also while a substitution is open (state dicts and other persistence read it). */
    Tensor ownedTensor() {
        ensureOpen();
        return tensor;
    }

    public String name() {
        return name;
    }

    public Shape shape() {
        ensureOpen();
        return tensor.shape();
    }

    public DType dtype() {
        ensureOpen();
        return tensor.dtype();
    }

    public ComputeBackend backend() {
        ensureOpen();
        return tensor.backend();
    }

    public long numElements() {
        ensureOpen();
        return tensor.numElements();
    }

    public long byteSize() {
        ensureOpen();
        return tensor.byteSize();
    }

    /** The gradient of the owned tensor (also while a substitution is open); null for none and when frozen. */
    public Tensor grad() {
        ensureOpen();
        return tensor.grad();
    }

    /**
     * Makes {@code grad} (null for none) the owned tensor's gradient, taking it over, and closes the gradient it
     * replaces unless that is {@code grad} itself. Clearing the gradient of a frozen parameter does nothing.
     *
     * @throws IllegalArgumentException when {@code grad} does not have the parameter's shape
     * @throws IllegalStateException    when {@code grad} is not null and the parameter is frozen
     */
    public void setGrad(Tensor grad) {
        ensureOpen();
        if (!trainable) {
            if (grad != null) {
                throw new IllegalStateException("Parameter '" + name + "' is frozen (requiresGrad=false) and holds"
                    + " no gradient; call setRequiresGrad(true) first");
            }
            return;
        }
        Tensor previous = tensor.grad();
        tensor.setGrad(grad);
        if (previous != null && previous != grad) {
            previous.close();
        }
    }

    /** Clears the gradient and closes it. */
    public void zeroGrad() {
        ensureOpen();
        setGrad(null);
    }

    /** Whether the parameter is trainable: its tensor is a leaf that requires grad. */
    public boolean requiresGrad() {
        return trainable;
    }

    /**
     * Makes the parameter trainable ({@code true}) or frozen ({@code false}). The owned tensor is swapped for a leaf
     * over the same storage with the new flag; freezing closes the gradient. Forward passes that read only frozen
     * parameters and inputs that do not require grad produce results that do not require grad. Does nothing when the
     * flag is unchanged. Change the flag between training steps, never between a forward pass and its backward pass:
     * a graph recorded before the change still references the replaced tensor, not the one the parameter now holds.
     */
    public void setRequiresGrad(boolean requiresGrad) {
        ensureOpen();
        if (trainable == requiresGrad) {
            return;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor grad = tensor.grad();
            tensor = asLeaf(tensor, requiresGrad);
            trainable = requiresGrad;
            if (grad != null) {
                grad.close();
            }
        }
    }

    /**
     * Opens a scope in which {@link #tensor()} returns {@code value} instead of the owned tensor. {@code value} stays
     * the caller's (it is not closed) and may be the output of a recorded operation, so gradients of what the forward
     * pass computes from it flow to whatever {@code value} was computed from. {@link #grad()}, {@link #setGrad} and
     * {@link #zeroGrad()} keep addressing the owned tensor. Closing the scope restores the owned tensor; closing it
     * again does nothing. The scope is meant for forward passes: optimizers update {@link #tensor()}, so they step
     * after it closes. {@link #copyFrom(Tensor)} and {@link #setRequiresGrad(boolean)} address the owned tensor, and
     * moving or converting the parameter is refused while the scope is open.
     *
     * <p><b>Nesting.</b> A scope opened while another substitution of this parameter is open shadows it:
     * {@link #tensor()} returns the innermost open scope's value, which a forward pass typically computes from the
     * shadowed value it read through {@link #tensor()} (an outer adaptation, then an inner normalization of it).
     * Closing a scope removes only its own value, so after the innermost scope closes {@link #tensor()} returns the
     * value it shadowed. Scopes are meant to close in reverse order of opening; one closed out of order is removed
     * without changing the others. Scopes belong to the thread that opened them and are not shared between
     * threads.</p>
     *
     * @throws NullPointerException     when {@code value} is null
     * @throws IllegalArgumentException when {@code value} differs from the parameter in shape, dtype or backend
     */
    public Substitution substitute(Tensor value) {
        Substitution scope = new Substitution();
        beginSubstitution(value, scope);
        scope.add(this);
        return scope;
    }

    /** Starts the substitution of {@code value} on behalf of {@code scope} (see {@link #substitute(Tensor)}). */
    synchronized void beginSubstitution(Tensor value, Substitution scope) {
        ensureOpen();
        Objects.requireNonNull(value, "Substitute tensor for parameter '" + name + "' cannot be null");
        if (!value.shape().equals(tensor.shape()) || value.dtype() != tensor.dtype()) {
            throw new IllegalArgumentException("Parameter '" + name + "' of shape " + tensor.shape() + " and dtype "
                + tensor.dtype() + " cannot be substituted by a tensor of shape " + value.shape() + " and dtype "
                + value.dtype());
        }
        if (value.backend() != tensor.backend()) {
            throw new IllegalArgumentException("Parameter '" + name + "' cannot be substituted by a tensor on another"
                + " backend");
        }
        substitution = new ActiveSubstitution(scope, value, substitution);
    }

    /** Removes the substitution {@code scope} opened from the stack of open substitutions, if it is still there. */
    synchronized void endSubstitution(Substitution scope) {
        substitution = without(substitution, scope);
    }

    /** The stack {@code top} without the topmost entry of {@code scope} (the same stack when it has none). */
    private static ActiveSubstitution without(ActiveSubstitution top, Substitution scope) {
        if (top == null) {
            return null;
        }
        if (top.scope == scope) {
            return top.previous;
        }
        ActiveSubstitution rest = without(top.previous, scope);
        return rest == top.previous ? top : new ActiveSubstitution(top.scope, top.value, rest);
    }

    public Parameter to(ComputeBackend backend) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");

        if (tensor.backend() == backend) {
            return this;
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return new Parameter(copyToBackend(tensor, backend), name, trainable);
        }
    }

    public Parameter to(DType dtype) {
        ensureOpen();
        Objects.requireNonNull(dtype, "DType cannot be null");

        if (tensor.dtype() == dtype) {
            return this;
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return new Parameter(tensor.to(dtype), name, trainable);
        }
    }

    /**
     * Moves the values to {@code backend} in place, keeping this parameter object, so modules and optimizers that
     * hold it see the moved tensor. The bytes cross through the host; the old tensor and the
     * gradient, which lives on the old backend, are closed.
     *
     * @throws IllegalStateException while a substitution is open
     */
    void moveTo(ComputeBackend backend) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");
        ensureNotSubstituted("moved to another backend");
        if (tensor.backend() != backend) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                replaceDroppingGrad(asLeaf(copyToBackend(tensor, backend), trainable));
            }
        }
    }

    /**
     * Converts the values to {@code dtype} in place ({@link Tensor#to(DType)}), keeping this parameter object; the
     * old tensor and the gradient are closed.
     *
     * @throws IllegalStateException while a substitution is open
     */
    void convertTo(DType dtype) {
        ensureOpen();
        Objects.requireNonNull(dtype, "DType cannot be null");
        ensureNotSubstituted("converted to another dtype");
        if (tensor.dtype() != dtype) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                replaceDroppingGrad(asLeaf(tensor.to(dtype), trainable));
            }
        }
    }

    /**
     * Overwrites the values in place with {@code values} in row-major order, keeping this parameter object, its
     * tensor and its gradient. A FLOAT32 parameter is written into its storage ({@link GpuOptimizerOps#storeFloats});
     * any other dtype gets a new tensor converted from the FLOAT32 values, and the old one is closed.
     *
     * @throws IllegalArgumentException when the number of values differs from the element count
     */
    void load(float[] values) {
        ensureOpen();
        Objects.requireNonNull(values, "Values cannot be null");
        if (values.length != tensor.numElements()) {
            throw new IllegalArgumentException("Parameter '" + name + "' of shape " + tensor.shape() + " holds "
                + tensor.numElements() + " values, got " + values.length);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (tensor.dtype() == DType.FLOAT32) {
                GpuOptimizerOps.storeFloats(tensor, values);
                return;
            }
            ComputeBackend backend = tensor.backend();
            Tensor floats = Tensor.fromStorage(backend, tensor.shape(), DType.FLOAT32,
                TensorStorage.fromHostFloat(backend, values));
            Tensor converted;
            try {
                converted = floats.to(tensor.dtype());
            } finally {
                floats.close();
            }
            replaceKeepingGrad(asLeaf(converted, trainable));
        }
    }

    /**
     * A contiguous copy of {@code source} on {@code backend} with the same shape, dtype and bytes, through the host;
     * it does not require grad.
     */
    static Tensor copyToBackend(Tensor source, ComputeBackend backend) {
        TensorStorage storage = TensorStorage.fromHost(backend, source.toByteBuffer(), source.dtype());
        return Tensor.fromStorage(backend, source.shape(), source.dtype(), storage);
    }

    /**
     * {@code raw} as a leaf that requires grad exactly when {@code trainable}. A tensor with a gradient function is
     * detached first. When the leaf is a new handle, {@code raw}'s reference (and that of the intermediate detached
     * handle) is released; a {@code raw} that already is such a leaf is returned unchanged.
     */
    private static Tensor asLeaf(Tensor raw, boolean trainable) {
        Tensor base = raw;
        if (raw.gradFn() != null) {
            base = raw.detach();
            raw.close();
        }
        Tensor leaf;
        try {
            leaf = base.withRequiresGrad(trainable);
        } catch (RuntimeException e) {
            if (base != raw) {
                base.close();
            }
            throw e;
        }
        if (leaf != base) {
            base.close();
        }
        return leaf;
    }

    private void replace(Tensor next) {
        Tensor previous = tensor;
        tensor = next;
        if (previous != next) {
            previous.close();
        }
    }

    /** {@link #replace} for a new tensor that starts without a gradient; the old tensor's gradient is closed. */
    private void replaceDroppingGrad(Tensor next) {
        Tensor grad = tensor.grad();
        replace(next);
        if (grad != null) {
            grad.close();
        }
    }

    /** {@link #replace} that moves the old tensor's gradient (null when frozen) to the new one. */
    private void replaceKeepingGrad(Tensor next) {
        Tensor grad = tensor.grad();
        replace(next);
        if (grad != null) {
            tensor.setGrad(grad);
        }
    }

    /** A parameter over the same storage as this one (no copy) with no gradient, trainable as this one is. */
    public Parameter detach() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return new Parameter(tensor.detach(), name, trainable);
        }
    }

    /**
     * A parameter of the same name over a new copy of the values on the same backend, with no gradient, trainable as
     * this one is ({@link #copyTensor()}).
     */
    public Parameter copy() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return new Parameter(copyTensor(), name, trainable);
        }
    }

    /**
     * A new contiguous copy of the owned values with the same shape, dtype and backend that does not require grad;
     * the caller owns it. The copy is made on the device by {@link GpuCoreOps#contiguous} when the train kernels are
     * available for the dtype, otherwise through the host. Nothing is recorded.
     */
    public Tensor copyTensor() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor device = GpuCoreOps.contiguous(tensor);
            return device != null ? device : copyToBackend(tensor, tensor.backend());
        }
    }

    /**
     * Overwrites the values in place with those of {@code source}, keeping this parameter object, its tensor and its
     * gradient: an assignment that records nothing, whatever {@code source} was computed from. FLOAT32 and INT32
     * values are copied on the device by {@link GpuCoreOps#copyInto} when the train kernels are available; otherwise
     * FLOAT32 values are written through the host ({@link GpuOptimizerOps#storeFloats}), and any other dtype gets a
     * new leaf holding a copy of {@code source} (the old tensor is closed, the gradient kept).
     *
     * @throws IllegalArgumentException when {@code source} lives on another backend or differs in shape or dtype
     */
    public void copyFrom(Tensor source) {
        ensureOpen();
        Objects.requireNonNull(source, "Source tensor cannot be null");
        if (source.backend() != tensor.backend()) {
            throw new IllegalArgumentException("Parameter '" + name + "' lives on another backend than the source");
        }
        if (!source.shape().equals(tensor.shape()) || source.dtype() != tensor.dtype()) {
            throw new IllegalArgumentException("Parameter '" + name + "' of shape " + tensor.shape() + " and dtype "
                + tensor.dtype() + " cannot take values of shape " + source.shape() + " and dtype " + source.dtype());
        }
        if (source == tensor) {
            return;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            DType dtype = tensor.dtype();
            boolean wordType = dtype == DType.FLOAT32 || dtype == DType.INT32;
            if (wordType && TrainGpu.forBackend(tensor.backend()) != null) {
                GpuCoreOps.copyInto(source, tensor);
            } else if (dtype == DType.FLOAT32) {
                GpuOptimizerOps.storeFloats(tensor, source.toFloatArray());
            } else {
                replaceKeepingGrad(asLeaf(copyToBackend(source, tensor.backend()), trainable));
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Parameter has been closed");
        }
    }

    /**
     * @throws IllegalStateException while a substitution is open; {@code action} completes the message ("cannot be
     *                               ...")
     */
    void ensureNotSubstituted(String action) {
        if (substitution != null) {
            throw new IllegalStateException("Parameter '" + name + "' cannot be " + action
                + " while a substitution is open");
        }
    }

    /** Closes the tensor and the gradient and ends an open substitution; idempotent. */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            substitution = null;
            if (tensor != null) {
                Tensor grad = tensor.grad();
                tensor.close();
                tensor = null;
                if (grad != null) {
                    grad.close();
                }
            }
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "Parameter{closed=true}";
        }
        return "Parameter{" +
            "name='" + name + '\'' +
            ", shape=" + tensor.shape() +
            ", dtype=" + tensor.dtype() +
            ", requiresGrad=" + trainable +
            '}';
    }

    /**
     * An open substitution of one or more parameters ({@link #substitute(Tensor)},
     * {@link Module#substitute(java.util.Map)}). {@link #close()} restores every parameter it substituted, in reverse
     * order, and is idempotent; it never closes the substitute tensors.
     */
    public static final class Substitution implements AutoCloseable {
        private final List<Parameter> substituted = new ArrayList<>();
        private boolean closed;

        Substitution() {
        }

        /** Records that {@code parameter} was substituted on behalf of this scope. */
        void add(Parameter parameter) {
            substituted.add(parameter);
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            for (int i = substituted.size() - 1; i >= 0; i--) {
                substituted.get(i).endSubstitution(this);
            }
            substituted.clear();
        }
    }

    /**
     * One open substitution: the substitute {@link #tensor()} returns, the scope that installed it, and the open
     * substitution it shadows (null for the owned tensor). Immutable, so the stack is replaced, never edited.
     */
    private static final class ActiveSubstitution {
        private final Substitution scope;
        private final Tensor value;
        private final ActiveSubstitution previous;

        ActiveSubstitution(Substitution scope, Tensor value, ActiveSubstitution previous) {
            this.scope = scope;
            this.value = value;
            this.previous = previous;
        }
    }
}
