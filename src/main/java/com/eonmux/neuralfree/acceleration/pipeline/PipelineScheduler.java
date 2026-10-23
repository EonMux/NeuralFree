package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.ScheduleEntry;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.TransferSchedule;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData.ActivationFunctionType;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Objects;

/**
 * Runs passes of a {@link NeuralNetworkData} dense network on the train kernel library of one backend (OpenCL, CUDA
 * or Vulkan): it builds the pass's {@link ExecutionGraph}, orders it with a {@link TransferSchedule}, and executes
 * every node on a device workspace that it keeps between calls of the same network shape and batch.
 *
 * <p><b>Batches.</b> The input array holds whole rows of the network's input width; the number of rows is
 * {@code inputData.length / layerSizes[0]}. Targets hold the same number of rows of the output width. The loss is
 * {@code sum((a - t)^2) / (2 * rows)}, so gradients are averaged over the rows; the update is plain SGD with the
 * network's learning rate. Each call uploads the network's current weights and biases before its first kernel and
 * reads its results back at its end, the only point where the call waits for device work.</p>
 *
 * <p><b>Availability.</b> There is no CPU pipeline: the constructor throws {@link UnsupportedKernelException} with
 * the reason when the train library is unavailable on the backend. The scheduler never closes kernel handles, which
 * the backend owns.</p>
 */
@NotThreadSafe("Keeps one device workspace and a RingBufferManager without synchronization. "
        + "Use one PipelineScheduler per thread, or an external mutex.")
public final class PipelineScheduler implements AutoCloseable {
    private static final Logger LOGGER = LoggingUtil.getLogger(PipelineScheduler.class);

    private final ComputeBackend backend;
    private final PipelineConfig config;
    private final DenseNetworkGpu network;
    private RingBufferManager ringBufferManager;
    private int currentPrefetchDepth;
    private boolean oomFallbackTriggered;

    private volatile boolean closed;

    /**
     * A scheduler on {@code backend}.
     *
     * @throws NullPointerException       when {@code backend} or {@code config} is null
     * @throws UnsupportedKernelException when the train kernel library is unavailable on the backend; the message
     *                                    carries {@link TrainKernels#availability}'s reason
     */
    public PipelineScheduler(ComputeBackend backend, PipelineConfig config) {
        this.backend = Objects.requireNonNull(backend, "backend cannot be null");
        this.config = Objects.requireNonNull(config, "config cannot be null");
        Availability availability = TrainKernels.availability(backend);
        if (!availability.available()) {
            throw new UnsupportedKernelException("PipelineScheduler needs the train kernel library, which is "
                    + "unavailable on " + backend.backendType() + ": " + availability.reason());
        }
        this.currentPrefetchDepth = config.enablePrefetch() ? config.prefetchDepth() : 0;
        this.oomFallbackTriggered = false;
        this.ringBufferManager = new RingBufferManager(backend, createConfigWithDepth(currentPrefetchDepth));
        this.network = new DenseNetworkGpu(backend);
        this.closed = false;

        DeviceCapabilities caps = backend.capabilities();
        LOGGER.info("PipelineScheduler initialized on {} '{}' - prefetch: {}, depth: {}, kernelCache: {}, "
                        + "memoryPool: {}, VRAM: {} MB",
                backend.backendType(), caps.getDeviceName(), config.enablePrefetch(), currentPrefetchDepth,
                config.enableKernelCache(), config.enableMemoryPool(), caps.getTotalVramBytes() / (1024 * 1024));
    }

    /**
     * Checks that an activation ordinal is one the dense kernels implement.
     *
     * @throws IllegalArgumentException when it is not an {@link ActivationFunctionType} ordinal
     */
    public void validateActivationFunction(int activationTypeOrdinal) {
        ActivationFunctionType.fromOrdinal(activationTypeOrdinal);
    }

    /**
     * Checks every layer's activation ordinal, and warns when the weights exceed the device's VRAM.
     *
     * @throws IllegalArgumentException when an activation ordinal is invalid
     */
    public void validateNetworkCapabilities(NeuralNetworkData networkData) {
        for (int activationType : networkData.getActivationTypes()) {
            validateActivationFunction(activationType);
        }

        DeviceCapabilities caps = backend.capabilities();
        long totalWeights = 0;
        for (float[] weights : networkData.getWeights()) {
            totalWeights += (long) weights.length * Float.BYTES;
        }
        if (caps.getTotalVramBytes() > 0 && totalWeights > caps.getTotalVramBytes()) {
            LOGGER.warn("Network weights ({} MB) exceed device VRAM ({} MB) on '{}'. "
                            + "Operation may fail with out-of-memory error.",
                    totalWeights / (1024 * 1024),
                    caps.getTotalVramBytes() / (1024 * 1024),
                    caps.getDeviceName());
        }
    }

    public void warnIfFeatureNotSupported(String featureName, boolean supported) {
        if (!supported) {
            LOGGER.warn("Feature '{}' is not supported by backend '{}'. "
                            + "Operation may produce incorrect results or fail.",
                    featureName,
                    backend.capabilities().getDeviceName());
        }
    }

    /**
     * The network's output rows for the input rows.
     *
     * @throws IllegalStateException    when the scheduler is closed
     * @throws IllegalArgumentException when the input is not a positive whole number of rows, or an activation
     *                                  ordinal or the weight shapes are invalid
     */
    public float[] executeForwardPass(NeuralNetworkData networkData, float[] inputData) {
        ensureOpen();
        Objects.requireNonNull(networkData, "networkData cannot be null");
        Objects.requireNonNull(inputData, "inputData cannot be null");
        validateNetworkCapabilities(networkData);
        int rows = rowsOf(networkData, inputData);

        LOGGER.debug("Executing forward pass - rows: {}, numLayers: {}", rows, networkData.getNumLayers());
        DenseNetworkGpu.Pass pass = run(ExecutionGraph.buildForInference(networkData, rows), networkData, inputData,
                null, rows);
        return pass.output();
    }

    /**
     * The loss gradient of every weight ({@link NeuralNetworkData#flattenWeights} layout) for the input and target
     * rows, without changing the network. The forward activations are recomputed on the device, so
     * {@code forwardOutput} is not read; when given, it must hold one output row per input row.
     *
     * @throws IllegalStateException    when the scheduler is closed
     * @throws IllegalArgumentException when the input, target or forward output lengths do not match the rows
     */
    public float[] executeBackwardPass(NeuralNetworkData networkData, float[] inputData,
                                       float[] targetData, float[] forwardOutput) {
        ensureOpen();
        Objects.requireNonNull(networkData, "networkData cannot be null");
        Objects.requireNonNull(inputData, "inputData cannot be null");
        Objects.requireNonNull(targetData, "targetData cannot be null");
        validateNetworkCapabilities(networkData);
        int rows = rowsOf(networkData, inputData);
        requireOutputRows(networkData, targetData, rows, "targetData");
        if (forwardOutput != null) {
            requireOutputRows(networkData, forwardOutput, rows, "forwardOutput");
        }

        LOGGER.debug("Executing backward pass - rows: {}, numLayers: {}", rows, networkData.getNumLayers());
        DenseNetworkGpu.Pass pass = run(ExecutionGraph.buildForGradients(networkData, rows), networkData, inputData,
                targetData, rows);
        int weights = network.weightCount();
        int params = weights + network.biasCount();
        return Arrays.copyOfRange(pass.state(), params, params + weights);
    }

    /**
     * One SGD step: the forward output (before the update), the gradients, and the updated weights and biases
     * ({@link NeuralNetworkData#flattenWeights} and {@link NeuralNetworkData#flattenBiases} layouts). The network
     * itself is not modified; {@link NeuralNetworkData#setParameters} applies the update.
     *
     * @throws IllegalStateException    when the scheduler is closed
     * @throws IllegalArgumentException when the input or target lengths do not match the rows
     */
    public TrainingResult executeTrainingIteration(NeuralNetworkData networkData,
                                                   float[] inputData, float[] targetData) {
        ensureOpen();
        Objects.requireNonNull(networkData, "networkData cannot be null");
        Objects.requireNonNull(inputData, "inputData cannot be null");
        Objects.requireNonNull(targetData, "targetData cannot be null");
        validateNetworkCapabilities(networkData);
        int rows = rowsOf(networkData, inputData);
        requireOutputRows(networkData, targetData, rows, "targetData");

        LOGGER.debug("Executing training iteration - rows: {}, numLayers: {}", rows, networkData.getNumLayers());
        long startTime = System.nanoTime();
        DenseNetworkGpu.Pass pass = run(ExecutionGraph.buildForTraining(networkData, rows), networkData, inputData,
                targetData, rows);
        float[] state = pass.state();
        int weights = network.weightCount();
        int params = weights + network.biasCount();
        long endTime = System.nanoTime();

        return TrainingResult.builder()
                .forwardOutput(pass.output())
                .updatedWeights(Arrays.copyOfRange(state, 0, weights))
                .updatedBiases(Arrays.copyOfRange(state, weights, params))
                .gradientWeights(Arrays.copyOfRange(state, params, params + weights))
                .gradientBiases(Arrays.copyOfRange(state, params + weights, 2 * params))
                .executionTimeNanos(endTime - startTime)
                .build();
    }

    /** Executes every node of {@code graph} in schedule order on a workspace prepared for the network and rows. */
    private DenseNetworkGpu.Pass run(ExecutionGraph graph, NeuralNetworkData networkData, float[] inputData,
                                     float[] targetData, int rows) {
        TrainGpu gpu = TrainGpu.forBackend(backend);
        if (gpu == null) {
            throw new UnsupportedKernelException("train kernel library unavailable on " + backend.backendType()
                    + ": " + TrainGpu.availability(backend).reason());
        }
        prepareWorkspace(gpu, networkData.getLayerSizes(), rows);
        DenseNetworkGpu.Pass pass = new DenseNetworkGpu.Pass(gpu, networkData, inputData, targetData, rows);
        TransferSchedule schedule = new TransferSchedule(graph, config);
        for (ScheduleEntry entry : schedule.entries()) {
            ExecutionNode node = graph.getNode(entry.nodeId());
            if (node == null) {
                throw new IllegalStateException("schedule entry " + entry.nodeId() + " has no graph node");
            }
            network.execute(node, pass);
        }
        return pass;
    }

    /**
     * Prepares the device workspace; on the first out-of-memory failure with prefetching enabled, lowers the
     * prefetch depth (releasing the ring buffer's slots) and retries once.
     */
    private void prepareWorkspace(TrainGpu gpu, int[] sizes, int rows) {
        try {
            network.prepare(gpu, sizes, rows);
        } catch (GpuOutOfMemoryException e) {
            if (!reducePrefetchDepth(e)) {
                throw e;
            }
            network.prepare(gpu, sizes, rows);
        }
    }

    private boolean reducePrefetchDepth(GpuOutOfMemoryException cause) {
        if (currentPrefetchDepth <= 0 || oomFallbackTriggered) {
            return false;
        }
        oomFallbackTriggered = true;
        int newDepth = currentPrefetchDepth - 1;
        LOGGER.warn("GPU OOM during workspace allocation, reducing prefetch depth from {} to {}: {}",
                currentPrefetchDepth, newDepth, cause.getMessage());
        currentPrefetchDepth = newDepth;
        ringBufferManager.close();
        ringBufferManager = new RingBufferManager(backend, createConfigWithDepth(newDepth));
        return true;
    }

    private static int rowsOf(NeuralNetworkData networkData, float[] inputData) {
        int width = networkData.getLayerSizes()[0];
        if (inputData.length == 0 || inputData.length % width != 0) {
            throw new IllegalArgumentException("inputData holds " + inputData.length
                    + " values, not a positive whole number of rows of the input width " + width);
        }
        return inputData.length / width;
    }

    private static void requireOutputRows(NeuralNetworkData networkData, float[] values, int rows, String what) {
        long expected = (long) rows * networkData.getLayerSizes()[networkData.getNumLayers()];
        if (values.length != expected) {
            throw new IllegalArgumentException(what + " holds " + values.length + " values, expected " + expected
                    + " (" + rows + " rows of the output width)");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("PipelineScheduler is closed");
        }
    }

    public PipelineConfig config() {
        return config;
    }

    public RingBufferManager ringBufferManager() {
        return ringBufferManager;
    }

    public int getCurrentPrefetchDepth() {
        return currentPrefetchDepth;
    }

    public boolean isOomFallbackTriggered() {
        return oomFallbackTriggered;
    }

    private PipelineConfig createConfigWithDepth(int depth) {
        if (depth <= 0) {
            return PipelineConfig.builder()
                    .enablePrefetch(false)
                    .prefetchDepth(1)
                    .enableKernelCache(config.enableKernelCache())
                    .enableMemoryPool(config.enableMemoryPool())
                    .build();
        }
        return PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(depth)
                .enableKernelCache(config.enableKernelCache())
                .enableMemoryPool(config.enableMemoryPool())
                .build();
    }

    /** Releases the device workspace and the ring buffer. Idempotent; kernel handles stay with the backend. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        RuntimeException failure = null;
        try {
            network.close();
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            ringBufferManager.close();
        } catch (RuntimeException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
