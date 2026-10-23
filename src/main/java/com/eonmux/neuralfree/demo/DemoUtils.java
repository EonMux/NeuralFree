package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

final class DemoUtils {

    /**
     * The APIs that run the training framework's GPU kernels. The framework's tensor ops dispatch to the
     * {@code train} kernel library, which is built for OpenCL (C source), CUDA (PTX) and Vulkan (SPIR-V) with the
     * same kernels and semantics, so AUTO considers all three.
     */
    static final Set<BackendType> FRAMEWORK_APIS = Collections.unmodifiableSet(
            EnumSet.of(BackendType.OPENCL, BackendType.CUDA, BackendType.VULKAN));

    private DemoUtils() {}

    /**
     * Enumerates the devices {@code mode} can use: every API of {@link #FRAMEWORK_APIS} for AUTO, otherwise only the
     * API it names. The caller shuts the instance down when the demos finish, which also closes every backend it
     * created.
     *
     * @param mode the API, or AUTO; null means AUTO
     */
    static HWAcceleration createAcceleration(ACCELERATION_MODE mode) {
        System.out.println("=== NeuralFree Demo Suite ===\n");
        return new HWAcceleration(apisFor(mode));
    }

    /**
     * Creates the backend the framework demos share ({@link NeuralFree#createTrainingBackend}) and prints it, with
     * whether the train kernels run there. {@code hw.shutdown()} closes it.
     *
     * @param mode the API, or AUTO; null means AUTO
     * @throws NoBackendAvailableException when no backend of the requested API (or of any API, for AUTO) can be
     *                                     created
     */
    static ComputeBackend initializeBackend(HWAcceleration hw, ACCELERATION_MODE mode) {
        ComputeBackend backend = NeuralFree.createTrainingBackend(hw, mode);
        DeviceCapabilities caps = backend.capabilities();

        System.out.println("GPU Backend: " + backend.backendType());
        System.out.println("Device: " + caps.getDeviceName());
        System.out.println("Memory: " + (caps.getTotalVramBytes() / 1024 / 1024) + " MB");
        System.out.println("Compute Units: " + caps.getComputeUnits());
        System.out.println("FP16 Support: " + caps.supportsFloat16());
        System.out.println("Train kernels: " + NeuralFree.describeTrainKernels(backend));
        System.out.println();

        return backend;
    }

    private static Set<BackendType> apisFor(ACCELERATION_MODE mode) {
        if (mode == null || mode == ACCELERATION_MODE.AUTO) {
            return FRAMEWORK_APIS;
        }
        switch (mode) {
            case OPENCL:
                return EnumSet.of(BackendType.OPENCL);
            case CUDA:
                return EnumSet.of(BackendType.CUDA);
            case VULKAN:
                return EnumSet.of(BackendType.VULKAN);
            default:
                throw new IllegalArgumentException("Unknown acceleration mode: " + mode);
        }
    }

    // ------------------------------------------------------------------ tensor lifetime

    /**
     * A FLOAT32 tensor holding {@code data} in the given shape. Device buffers are not released by the garbage
     * collector, so the caller closes it.
     */
    static Tensor tensor(ComputeBackend backend, float[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(dims);
        }
    }

    /** An INT32 tensor holding {@code data} in the given shape; the caller closes it. */
    static Tensor intTensor(ComputeBackend backend, int[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(dims);
        }
    }

    /**
     * Applies {@code op} to an intermediate tensor the caller owns and closes the intermediate, unless {@code op}
     * returned that same tensor. The intermediate is closed also when {@code op} throws.
     */
    static Tensor consume(Tensor intermediate, UnaryOperator<Tensor> op) {
        Tensor result = null;
        try {
            result = op.apply(intermediate);
            return result;
        } finally {
            if (result != intermediate) {
                intermediate.close();
            }
        }
    }

    /**
     * Closes every non-null tensor, all of them even when one fails; the first failure is rethrown with the others
     * suppressed.
     */
    static void closeAll(Tensor... tensors) {
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            if (t == null) {
                continue;
            }
            try {
                t.close();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * Clears and closes the gradient that {@code backward} left on each tensor, then closes the tensor itself, as
     * {@link #closeAll} does. Each gradient is its own tensor object (views share refcounted storage), so closing one
     * never closes another tensor's gradient; clearing the slot lets the thread's graph forget the leaf.
     */
    static void closeWithGrads(Tensor... tensors) {
        Tensor[] all = new Tensor[tensors.length * 2];
        for (int i = 0; i < tensors.length; i++) {
            Tensor t = tensors[i];
            if (t != null && t.grad() != null) {
                all[2 * i] = t.grad();
                t.setGrad(null);
            }
            all[2 * i + 1] = t;
        }
        closeAll(all);
    }

    /**
     * The softmax probabilities of {@code logits} (one sample, {@code numClasses} logits), read back to the host.
     * {@code logits} stays open.
     */
    static float[] probabilities(Tensor logits, int numClasses) {
        try (Tensor flat = logits.reshape(numClasses);
             Tensor probs = TensorOps.softmax(flat, -1)) {
            return probs.toFloatArray();
        }
    }

    /** The scalar value of a loss tensor, read back to the host. */
    static float scalar(Tensor loss) {
        return loss.toFloatArray()[0];
    }

    // ------------------------------------------------------------------ training steps

    /**
     * One training step of a hand-written loop, inside its own {@link ComputationGraph.StepScope}: zero the
     * gradients, {@code forward}, {@code lossOf} the output, backward, then {@code optimizer.step()}. With
     * {@code reportGradients} the global gradient norm ({@link #printGradientNorm}) is printed between the backward
     * and the step. The backward releases the nodes it traverses and the scope discards any other node the step
     * recorded, so the thread's graph is back at its size before the step; the output and the loss are closed.
     *
     * @return the loss, read back for logging
     */
    static float trainStep(Module model, Optimizer optimizer, Supplier<Tensor> forward, UnaryOperator<Tensor> lossOf,
                           boolean reportGradients) {
        try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
            optimizer.zeroGrad();
            try (Tensor output = forward.get();
                 Tensor loss = lossOf.apply(output)) {
                float value = scalar(loss);
                loss.backward();
                if (reportGradients) {
                    printGradientNorm("First step", model.parameters());
                }
                optimizer.step();
                return value;
            }
        }
    }

    // ------------------------------------------------------------------ gradients

    /**
     * Prints the global L2 norm of the gradients of {@code params} (the total norm of PyTorch's
     * {@code clip_grad_norm_}) and how many of the trainable parameters received a gradient, so a step whose loss
     * reaches no parameter is visible. A parameter listed more than once counts once. The norm is computed on the
     * gradients' backend and read back as one scalar (a logged metric).
     *
     * @param label names the step, for example {@code "First step"}
     */
    static void printGradientNorm(String label, List<Parameter> params) {
        Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Tensor> grads = new ArrayList<>();
        int trainable = 0;
        int withGradient = 0;
        for (Parameter p : params) {
            if (!seen.add(p)) {
                continue;
            }
            if (p.requiresGrad()) {
                trainable++;
            }
            Tensor g = p.grad();
            if (g != null) {
                grads.add(g);
                if (p.requiresGrad()) {
                    withGradient++;
                }
            }
        }
        System.out.printf("%s gradient norm: %.6f (%d of %d trainable parameters received a gradient)%n",
                label, l2Norm(grads), withGradient, trainable);
    }

    /**
     * {@code sqrt(sum over the tensors of sum(t * t))}, computed without recording on the tensors' backend (they must
     * share one) and read back as one scalar; 0 for an empty list. The tensors stay open.
     */
    static float l2Norm(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            return 0.0f;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor total = null;
            try {
                for (Tensor t : tensors) {
                    Tensor part = consume(TensorOps.mul(t, t), squares -> TensorOps.sum(squares));
                    if (total == null) {
                        total = part;
                    } else {
                        Tensor previous = total;
                        total = null;
                        try {
                            total = TensorOps.add(previous, part);
                        } finally {
                            closeAll(previous, part);
                        }
                    }
                }
                try (Tensor norm = TensorOps.sqrt(total)) {
                    return scalar(norm);
                }
            } finally {
                if (total != null) {
                    total.close();
                }
            }
        }
    }

    // ------------------------------------------------------------------ data and printing

    static float[] generateRandomData(int size, Random random) {
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) (random.nextGaussian() * 0.1);
        }
        return data;
    }

    /** Index of the first largest value. */
    static int argmax(float[] values) {
        int maxIdx = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /** The values with two decimals, as {@code [a, b, ...]}. */
    static String formatProbs(float[] probs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < probs.length; i++) {
            sb.append(String.format("%.2f", probs[i]));
            if (i < probs.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    static void printSeparator() {
        System.out.println("\n" + "=".repeat(60) + "\n");
    }

    static void printHeader(String title) {
        System.out.println("=== " + title + " ===");
        System.out.println("-".repeat(title.length() + 8));
    }
}
