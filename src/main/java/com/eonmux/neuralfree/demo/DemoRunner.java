package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;

import java.util.Locale;

/**
 * Runs the training-framework demos on one shared backend.
 *
 * <p>{@code --backend} chooses the API: {@code auto}, one of the built-in APIs ({@code opencl}, {@code cuda},
 * {@code vulkan}, {@code hip}, {@code metal}) or an accelerator API served by an installed backend provider.
 * Without it, {@code -Dneuralfree.backend} (forwarded from {@code -Pneuralfree.backend} by the {@code runDemos}
 * task) does, and AUTO is the default. AUTO selects a backend that runs the train kernels, as
 * {@link NeuralFree#createTrainingBackend} documents.</p>
 */
public final class DemoRunner {

    /** The usage text {@code --help} prints; the backend names are {@link NeuralFree#parseBackend}'s. */
    static final String USAGE = String.join(System.lineSeparator(),
            "Usage: DemoRunner [--backend " + backendNames("|") + "]",
            "  --backend <api>   API the demos run on: " + backendNames(", ")
                    + " (default: -D" + NeuralFree.BACKEND_PROPERTY + ", else auto)",
            "  --help, -h        Print this help");

    private DemoRunner() {}

    public static void main(String[] args) {
        ACCELERATION_MODE mode;
        try {
            mode = parseBackend(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.println(USAGE);
            System.exit(2);
            return;
        }
        if (mode == null) {
            System.out.println(USAGE);
            return;
        }

        boolean failed = false;
        HWAcceleration hw = null;
        try {
            hw = DemoUtils.createAcceleration(mode);
            ComputeBackend backend = DemoUtils.initializeBackend(hw, mode);

            MLPDemo.run(backend);
            DemoUtils.printSeparator();

            CNNDemo.run(backend);
            DemoUtils.printSeparator();

            GpuMatmulDemo.run(backend);
            DemoUtils.printSeparator();

            TransformerDemo.run(backend);
            DemoUtils.printSeparator();

            GNNDemo.run(backend);
            DemoUtils.printSeparator();

            AutogradDemo.run(backend);
            DemoUtils.printSeparator();

            InferenceDemo.run(backend);

            System.out.println("\n=== All Demos Completed Successfully ===");

        } catch (RuntimeException e) {
            failed = true;
            System.err.println("Demo failed with error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (hw != null) {
                // Closes the backend created above, then releases the devices.
                hw.shutdown();
            }
        }
        if (failed) {
            System.exit(1);
        }
    }

    /**
     * The API names {@link NeuralFree#parseBackend} accepts, lowercase in declaration order, joined by
     * {@code separator}.
     */
    private static String backendNames(String separator) {
        StringBuilder s = new StringBuilder();
        for (ACCELERATION_MODE mode : ACCELERATION_MODE.values()) {
            if (s.length() > 0) {
                s.append(separator);
            }
            s.append(mode.name().toLowerCase(Locale.ROOT));
        }
        return s.toString();
    }

    /**
     * The API the arguments choose: {@code --backend <api>}, else {@code -Dneuralfree.backend}, else AUTO; null
     * when help was requested.
     *
     * @throws IllegalArgumentException for an unknown argument, a missing value or an unknown API
     */
    static ACCELERATION_MODE parseBackend(String[] args) {
        ACCELERATION_MODE fromArgs = null;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    return null;
                case "--backend":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--backend needs a value: auto, "
                                + BackendType.acceptedApiNames());
                    }
                    String value = args[++i];
                    try {
                        fromArgs = NeuralFree.parseBackend(value);
                    } catch (IllegalArgumentException e) {
                        throw new IllegalArgumentException("--backend: " + e.getMessage(), e);
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Unknown argument: " + a);
            }
        }
        return fromArgs != null ? fromArgs : NeuralFree.backendFromProperty();
    }

    public static void runMLP(ComputeBackend backend) {
        MLPDemo.run(backend);
    }

    public static void runCNN(ComputeBackend backend) {
        CNNDemo.run(backend);
    }

    public static void runGpuMatmul(ComputeBackend backend) {
        GpuMatmulDemo.run(backend);
    }

    public static void runTransformer(ComputeBackend backend) {
        TransformerDemo.run(backend);
    }

    public static void runGNN(ComputeBackend backend) {
        GNNDemo.run(backend);
    }

    public static void runAutograd(ComputeBackend backend) {
        AutogradDemo.run(backend);
    }

    public static void runInference(ComputeBackend backend) {
        InferenceDemo.run(backend);
    }
}
