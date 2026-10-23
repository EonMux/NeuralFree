package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuMatmul;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuReduceOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The device side of {@link PipelineScheduler}: a workspace sized for one network shape and batch, and the operation
 * of every {@link ExecutionNode.Operation} on it, built from the train library's dense kernels and the tensor
 * dispatchers ({@link GpuMatmul}, {@link GpuReduceOps}, {@link GpuOptimizerOps}, {@link GpuCoreOps}).
 *
 * <p><b>Workspace.</b> Six FLOAT32 buffers, laid out so no GEMM output shares a buffer with its operands:
 * {@code state = [W_0..W_{L-1} | b_0..b_{L-1} | dW_0.. | db_0..]} (weights and gradients in the
 * {@link NeuralNetworkData#flattenWeights} and {@link NeuralNetworkData#flattenBiases} layouts, so one SGD launch
 * covers every parameter and one read returns them all), {@code io = [x | t]} (input and target rows),
 * {@code z} (every layer's pre-activations), {@code act} (every layer's activations, the last one being the output)
 * and two gradient buffers that hold {@code dY} and, in place, {@code dZ} of alternate layers. Layer {@code l}'s
 * weights are the row-major {@code [out, in]} view; {@code x * W^T} and {@code dZ^T * x} use transposed views.</p>
 *
 * <p><b>Numerics.</b> The CPU reference of a pass is: {@code z = sum_k x_k * W_k} in ascending k, then
 * {@code z + b} and {@code a = act(z)} ({@link NeuralNetworkData.ActivationFunctionType}); on the output
 * {@code dY = s * a + (-s) * t} with {@code s = 1f / rows}; {@code dZ = dY * act'(z, a)}; {@code dW} and
 * {@code dY(l - 1)} the ascending sums over rows and outputs; {@code db} the ascending column sums;
 * {@code p = p - lr * g}. The GEMMs accumulate with {@code fma} and the activations built on transcendentals follow
 * the tolerance class of {@code kernels/train/dense.cl}; every other step rounds as the reference does.</p>
 *
 * <p>Not thread-safe; the scheduler that owns it serializes its use.</p>
 */
final class DenseNetworkGpu implements AutoCloseable {

    /** The inputs and results of one execution of a graph. */
    static final class Pass {
        final TrainGpu gpu;
        final NeuralNetworkData network;
        final float[] input;
        final float[] target;
        final int rows;
        final int[] activations;
        final float learningRate;
        private float[] output;
        private float[] state;

        Pass(TrainGpu gpu, NeuralNetworkData network, float[] input, float[] target, int rows) {
            this.gpu = Objects.requireNonNull(gpu, "gpu must not be null");
            this.network = Objects.requireNonNull(network, "network must not be null");
            this.input = Objects.requireNonNull(input, "input must not be null");
            this.target = target;
            this.rows = rows;
            this.activations = network.getActivationTypes();
            this.learningRate = network.getLearningRate();
        }

        /** The output rows read by {@link ExecutionNode.Operation#READ_OUTPUT}. */
        float[] output() {
            return require(output, "output");
        }

        /** Weights, biases, then their gradients, as read by {@link ExecutionNode.Operation#READ_RESULTS}. */
        float[] state() {
            return require(state, "parameters and gradients");
        }

        private static float[] require(float[] values, String what) {
            if (values == null) {
                throw new IllegalStateException("the executed graph did not read the " + what + " back");
            }
            return values;
        }
    }

    private final ComputeBackend backend;
    private Workspace workspace;

    DenseNetworkGpu(ComputeBackend backend) {
        this.backend = Objects.requireNonNull(backend, "backend must not be null");
    }

    /**
     * Makes the workspace match the layer widths and batch rows, reallocating it when they changed.
     *
     * @throws IllegalArgumentException when a buffer would exceed the {@code 2^31 - 1} element limit of the train
     *                                  kernels
     */
    void prepare(TrainGpu gpu, int[] sizes, int rows) {
        if (workspace != null && workspace.matches(sizes, rows)) {
            return;
        }
        releaseWorkspace();
        workspace = new Workspace(gpu, sizes, rows);
    }

    /** Number of weights in the current workspace (the gradient-weight count too). */
    int weightCount() {
        return requireWorkspace().weightCount;
    }

    /** Number of biases in the current workspace. */
    int biasCount() {
        return requireWorkspace().biasCount;
    }

    /**
     * Runs one node on the prepared workspace.
     *
     * @throws IllegalArgumentException      when an uploaded array does not match the network's shape
     * @throws UnsupportedKernelException    when the train library stopped serving a dispatcher
     * @throws IllegalStateException         when the node has no operation or the workspace is not prepared
     */
    void execute(ExecutionNode node, Pass pass) {
        Workspace ws = requireWorkspace();
        ExecutionNode.Operation op = node.operation();
        if (op == null) {
            throw new IllegalStateException("execution node " + node.nodeId() + " has no operation");
        }
        if (pass.rows != ws.rows) {
            throw new IllegalStateException("the workspace is prepared for " + ws.rows + " rows, the pass has "
                    + pass.rows);
        }
        int layer = node.layerIndex();
        switch (op) {
            case UPLOAD_WEIGHTS:
                upload(ws.weights[layer], pass.network.getWeights()[layer], "weights of layer " + layer);
                break;
            case UPLOAD_BIASES:
                upload(ws.bias[layer], pass.network.getBiases()[layer], "biases of layer " + layer);
                break;
            case UPLOAD_INPUT:
                upload(ws.x[0], pass.input, "input rows");
                break;
            case UPLOAD_TARGET:
                upload(ws.target, Objects.requireNonNull(pass.target, "the pass has no target rows"), "target rows");
                break;
            case LINEAR:
                gemm(ws.x[layer], ws.weightsT[layer], ws.z[layer], "z = x * W^T of layer " + layer);
                break;
            case BIAS_ACTIVATION:
                biasActivation(pass.gpu, ws, layer, pass.activations[layer]);
                break;
            case LOSS_GRADIENT:
                lossGradient(pass.gpu, ws);
                break;
            case ACTIVATION_BACKWARD:
                activationBackward(pass.gpu, ws, layer, pass.activations[layer]);
                break;
            case WEIGHT_GRADIENT:
                gemm(ws.dyT[layer], ws.x[layer], ws.gradW[layer], "dW = dZ^T * x of layer " + layer);
                break;
            case BIAS_GRADIENT:
                biasGradient(ws, layer);
                break;
            case INPUT_GRADIENT:
                gemm(ws.dy[layer], ws.weights[layer], ws.dy[layer - 1], "dY = dZ * W of layer " + layer);
                break;
            case SGD_UPDATE:
                sgdUpdate(ws, pass.learningRate);
                break;
            case READ_OUTPUT:
                pass.output = download(ws.a[ws.layers - 1]);
                break;
            case READ_RESULTS:
                pass.state = download(ws.allState);
                break;
            default:
                throw new IllegalStateException("unhandled operation " + op + " of node " + node.nodeId());
        }
    }

    /** Releases the workspace; the instance can prepare a new one afterwards. */
    @Override
    public void close() {
        releaseWorkspace();
    }

    // ------------------------------------------------------------------ operations

    private void upload(Tensor view, float[] values, String what) {
        if (values == null || values.length != view.numElements()) {
            throw new IllegalArgumentException(what + ": expected " + view.numElements() + " values, got "
                    + (values == null ? "none" : String.valueOf(values.length)));
        }
        closeEvent(view.storage().copyFromHostFloat(values, view.strides().offset(), backend.transferQueue()));
    }

    private float[] download(Tensor view) {
        return view.storage().copyToHostFloat(view.strides().offset(), view.numElements(), backend.transferQueue());
    }

    private static void gemm(Tensor a, Tensor b, Tensor c, String what) {
        if (!GpuMatmul.gemm(a, b, 1.0f, 0.0f, c)) {
            throw unavailable(c.backend(), what);
        }
    }

    /** TR_DenseBiasAct_F32: {@code z[r, c] += b[c]; a[r, c] = act(z[r, c])}. */
    private static void biasActivation(TrainGpu gpu, Workspace ws, int layer, int act) {
        int cols = ws.sizes[layer + 1];
        StridedView z = StridedView.of(ws.z[layer]);
        StridedView b = StridedView.of(ws.bias[layer]);
        StridedView a = StridedView.of(ws.a[layer]);
        gpu.launch1d(Kernels.DENSE_BIAS_ACT_F32, (long) ws.rows * cols, act, ws.rows, cols, z.offset(), b.offset(),
                a.offset(), z.buffer(), b.buffer(), a.buffer());
    }

    /**
     * {@code dY = s * a + (-s) * t} with {@code s = 1f / rows}, the gradient of {@code sum((a - t)^2) / (2 * rows)}:
     * TR_Axpby_F32 writes {@code s * a} ({@code beta = 0} does not read {@code dY}), then adds {@code -s * t}
     * ({@code beta = 1} keeps {@code s * a} exactly).
     */
    private static void lossGradient(TrainGpu gpu, Workspace ws) {
        int last = ws.layers - 1;
        int n = Math.toIntExact((long) ws.rows * ws.sizes[ws.layers]);
        float s = 1.0f / ws.rows;
        StridedView a = StridedView.of(ws.a[last]);
        StridedView t = StridedView.of(ws.target);
        StridedView dy = StridedView.of(ws.dy[last]);
        gpu.launch1d(Kernels.AXPBY_F32, n, n, s, 0.0f, a.offset(), dy.offset(), a.buffer(), dy.buffer());
        gpu.launch1d(Kernels.AXPBY_F32, n, n, -s, 1.0f, t.offset(), dy.offset(), t.buffer(), dy.buffer());
    }

    /** TR_DenseActBackward_F32 in place: {@code dZ = dY * act'(z, a)}. */
    private static void activationBackward(TrainGpu gpu, Workspace ws, int layer, int act) {
        int n = Math.toIntExact((long) ws.rows * ws.sizes[layer + 1]);
        StridedView z = StridedView.of(ws.z[layer]);
        StridedView a = StridedView.of(ws.a[layer]);
        StridedView dy = StridedView.of(ws.dy[layer]);
        gpu.launch1d(Kernels.DENSE_ACT_BACKWARD_F32, n, act, n, z.offset(), a.offset(), dy.offset(), dy.offset(),
                z.buffer(), a.buffer(), dy.buffer(), dy.buffer());
    }

    /** {@code db = sum over rows of dZ} ({@link GpuReduceOps#sumColumns}), copied into its slot of the state. */
    private static void biasGradient(Workspace ws, int layer) {
        Tensor sums = GpuReduceOps.sumColumns(ws.dy[layer]);
        if (sums == null) {
            throw unavailable(ws.state.backend(), "db = column sums of dZ of layer " + layer);
        }
        try {
            GpuCoreOps.copyInto(sums, ws.gradB[layer]);
        } finally {
            sums.close();
        }
    }

    /** {@code p = p - lr * g} over every weight and bias (TR_SgdStep_F32, no momentum, no decay). */
    private static void sgdUpdate(Workspace ws, float learningRate) {
        GpuOptimizerOps.SgdParams sgd = new GpuOptimizerOps.SgdParams(learningRate, 0.0f, 0.0f, false);
        if (!GpuOptimizerOps.sgdStep(ws.params, ws.grads, null, sgd)) {
            throw unavailable(ws.state.backend(), "SGD update");
        }
    }

    private static UnsupportedKernelException unavailable(ComputeBackend backend, String what) {
        return new UnsupportedKernelException("train kernels unavailable on " + backend.backendType()
                + " during " + what + ": " + TrainGpu.availability(backend).reason());
    }

    private static void closeEvent(GpuEvent event) {
        if (event != null) {
            event.close();
        }
    }

    private Workspace requireWorkspace() {
        if (workspace == null) {
            throw new IllegalStateException("no workspace prepared");
        }
        return workspace;
    }

    private void releaseWorkspace() {
        Workspace ws = workspace;
        workspace = null;
        if (ws != null) {
            ws.close();
        }
    }

    // ------------------------------------------------------------------ workspace

    /** The device buffers of one network shape and batch, and the per-layer views into them. */
    private static final class Workspace implements AutoCloseable {
        final int[] sizes;
        final int rows;
        final int layers;
        final int weightCount;
        final int biasCount;

        final Tensor state;
        final Tensor io;
        final Tensor preActivations;
        final Tensor activations;
        final Tensor[] deltas = new Tensor[2];

        final Tensor allState;
        final Tensor params;
        final Tensor grads;
        final Tensor target;
        final Tensor[] weights;
        final Tensor[] weightsT;
        final Tensor[] bias;
        final Tensor[] gradW;
        final Tensor[] gradB;
        final Tensor[] z;
        final Tensor[] a;
        /** Layer inputs: the input rows, then the previous layer's activations. */
        final Tensor[] x;
        final Tensor[] dy;
        final Tensor[] dyT;

        private final List<Tensor> owned = new ArrayList<>();

        Workspace(TrainGpu gpu, int[] sizes, int rows) {
            this.sizes = sizes.clone();
            this.rows = rows;
            this.layers = sizes.length - 1;
            long weightTotal = 0;
            long biasTotal = 0;
            long rowsTotal = 0;
            long widest = 0;
            for (int l = 0; l < layers; l++) {
                weightTotal += (long) sizes[l] * sizes[l + 1];
                biasTotal += sizes[l + 1];
                rowsTotal += (long) rows * sizes[l + 1];
                widest = Math.max(widest, (long) rows * sizes[l + 1]);
            }
            this.weightCount = elements(weightTotal, "weights");
            this.biasCount = elements(biasTotal, "biases");
            int paramCount = elements(weightTotal + biasTotal, "parameters");
            long inputs = (long) rows * sizes[0];
            long outputs = (long) rows * sizes[layers];

            this.weights = new Tensor[layers];
            this.weightsT = new Tensor[layers];
            this.bias = new Tensor[layers];
            this.gradW = new Tensor[layers];
            this.gradB = new Tensor[layers];
            this.z = new Tensor[layers];
            this.a = new Tensor[layers];
            this.x = new Tensor[layers];
            this.dy = new Tensor[layers];
            this.dyT = new Tensor[layers];
            try {
                this.state = allocate(gpu, 2L * paramCount, "parameter and gradient state");
                this.io = allocate(gpu, inputs + outputs, "input and target rows");
                this.preActivations = allocate(gpu, rowsTotal, "pre-activations");
                this.activations = allocate(gpu, rowsTotal, "activations");
                this.deltas[0] = allocate(gpu, widest, "gradients");
                this.deltas[1] = allocate(gpu, widest, "gradients");

                this.allState = view(state, 0, 2 * paramCount);
                this.params = view(state, 0, paramCount);
                this.grads = view(state, paramCount, paramCount);
                this.target = view(io, inputs, rows, sizes[layers]);
                long w = 0;
                long b = weightTotal;
                long r = 0;
                for (int l = 0; l < layers; l++) {
                    int in = sizes[l];
                    int out = sizes[l + 1];
                    weights[l] = view(state, w, out, in);
                    weightsT[l] = own(weights[l].transpose());
                    bias[l] = view(state, b, out);
                    gradW[l] = view(state, paramCount + w, out, in);
                    gradB[l] = view(state, paramCount + b, out);
                    z[l] = view(preActivations, r, rows, out);
                    a[l] = view(activations, r, rows, out);
                    x[l] = l == 0 ? view(io, 0, rows, in) : a[l - 1];
                    dy[l] = view(deltas[l % 2], 0, rows, out);
                    dyT[l] = own(dy[l].transpose());
                    w += (long) in * out;
                    b += out;
                    r += (long) rows * out;
                }
            } catch (RuntimeException e) {
                closeAll(e);
                throw e;
            }
        }

        boolean matches(int[] otherSizes, int otherRows) {
            return rows == otherRows && Arrays.equals(sizes, otherSizes);
        }

        private Tensor allocate(TrainGpu gpu, long count, String what) {
            int n = elements(count, what);
            return own(Tensor.fromStorage(gpu.backend(), Shape.vector(n), DType.FLOAT32,
                    gpu.allocate(n, DType.FLOAT32)));
        }

        /** A dense view of {@code dims} starting {@code offset} elements into the 1-D {@code base}. */
        private Tensor view(Tensor base, long offset, int... dims) {
            long n = 1;
            for (int d : dims) {
                n *= d;
            }
            Tensor slice = base.slice(0, offset, offset + n);
            try {
                return own(slice.reshape(dims));
            } finally {
                slice.close();
            }
        }

        private Tensor own(Tensor t) {
            owned.add(t);
            return t;
        }

        private static int elements(long count, String what) {
            if (count > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("the " + what + " of the network need " + count
                        + " elements, above the 2^31-1 limit of the train kernels");
            }
            return (int) count;
        }

        @Override
        public void close() {
            closeAll(null);
        }

        private void closeAll(RuntimeException primary) {
            RuntimeException failure = primary;
            for (int i = owned.size() - 1; i >= 0; i--) {
                try {
                    owned.get(i).close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            owned.clear();
            if (primary == null && failure != null) {
                throw failure;
            }
        }
    }
}
