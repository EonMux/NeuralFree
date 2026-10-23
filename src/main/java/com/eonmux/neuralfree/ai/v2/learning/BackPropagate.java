package com.eonmux.neuralfree.ai.v2.learning;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.pipeline.PipelineScheduler;
import com.eonmux.neuralfree.acceleration.pipeline.TrainingResult;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.utils.GeneralUtil;

import java.util.Objects;

/**
 * Backpropagation training of a {@link NeuralNetworkData} network on a compute backend's train kernels, through a
 * {@link PipelineScheduler}. The training map's keys are the input rows and its values the target rows, both
 * flattened in the map's iteration order: {@code keys.size() / layerSizes[0]} rows, and the values must hold as many
 * rows of the output width.
 *
 * <p>Construction runs one training step. Each step runs the forward pass, the gradients of
 * {@code sum((a - t)^2) / (2 * rows)} and an SGD update with the network's learning rate on the device, and copies
 * the updated weights and biases into the network.</p>
 */
public class BackPropagate implements AutoCloseable {
    private final NeuralNetworkData currentNetwork;
    private PipelineScheduler pipelineScheduler;
    private TrainingResult lastResult;

    /**
     * Opens a scheduler on {@code backend} and runs one training step.
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the train library is
     *         unavailable on the backend
     * @throws IllegalArgumentException when the training map does not hold whole input and target rows
     */
    public BackPropagate(NeuralNetworkData network, ComputeBackend backend) {
        this.currentNetwork = Objects.requireNonNull(network, "network must not be null");
        Objects.requireNonNull(backend, "backend must not be null");
        this.pipelineScheduler = new PipelineScheduler(backend, PipelineConfig.defaultConfig());
        try {
            step();
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * One training step over the training map: applies the updated parameters to the network and returns the step's
     * results.
     *
     * @throws IllegalStateException when this instance is closed
     */
    public TrainingResult step() {
        if (pipelineScheduler == null) {
            throw new IllegalStateException("BackPropagate is closed");
        }
        float[] inputs = GeneralUtil.convertCollectionToFloatArray(currentNetwork.getTrainingMap().keySet());
        float[] targets = GeneralUtil.convertCollectionToFloatArray(currentNetwork.getTrainingMap().values());
        TrainingResult result = pipelineScheduler.executeTrainingIteration(currentNetwork, inputs, targets);
        currentNetwork.setParameters(result.updatedWeights(), result.updatedBiases());
        lastResult = result;
        return result;
    }

    /** The results of the latest step. */
    public TrainingResult getLastResult() {
        return lastResult;
    }

    @Override
    public void close() {
        if (pipelineScheduler != null) {
            PipelineScheduler scheduler = pipelineScheduler;
            pipelineScheduler = null;
            scheduler.close();
        }
    }

    /** The scheduler, or null once closed. */
    public PipelineScheduler getPipelineScheduler() {
        return pipelineScheduler;
    }
}
