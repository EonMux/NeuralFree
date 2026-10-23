package com.eonmux.neuralfree.tensor;

/**
 * CPU reference of {@link NormOps}: the host loops of {@code nn/norm/LayerNorm}, {@code RMSNorm},
 * {@code BatchNorm1d}/{@code BatchNorm2d} and {@code GroupNorm}, in the modules' loop order and Java float
 * arithmetic: two-pass mean and biased variance (sequential sums, one division by the count),
 * {@code sdev = (float) Math.sqrt(var + eps)}, {@code (x - mean) / sdev}, and the affine step as a separate multiply
 * and add. Operands are read in row-major order ({@link Tensor#toFloatArray()}, so they must be FLOAT32) and each
 * result is uploaded as a new contiguous FLOAT32 tensor on the input's backend with {@code requiresGrad} false.
 * Arguments are validated by {@link NormOps} before these run.
 *
 * <p>The backward references recompute the forward's statistics with the forward loops above, so {@code xhat} is
 * the forward's value bit for bit, and then apply the textbook gradients with the operation sequence of the backward
 * kernels ({@code kernels/train/norm.cl} header): {@code d = gy * gamma} (or {@code gy} without an affine scale),
 * sums in the forward's iteration order with a separate multiply and add, and every product, difference and division
 * of the final formula as its own float operation.</p>
 */
final class CpuNorm {

    private CpuNorm() {
    }

    /**
     * {@code nn/norm/LayerNorm} forward: row {@code r} is the {@code normSize} elements from {@code r * normSize};
     * {@code gamma} and {@code beta} (each optional) are indexed by the position within the row.
     */
    static Tensor layerNorm(Tensor input, int normSize, Tensor gamma, Tensor beta, float epsilon) {
        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];
        float[] weightData = gamma == null ? null : gamma.toFloatArray();
        float[] biasData = beta == null ? null : beta.toFloatArray();
        int rowSize = normSize;
        int rows = inputData.length == 0 ? 0 : inputData.length / rowSize;

        for (int r = 0; r < rows; r++) {
            int base = r * rowSize;

            float sum = 0.0f;
            for (int n = 0; n < rowSize; n++) {
                sum += inputData[base + n];
            }
            float mean = sum / normSize;

            float varSum = 0.0f;
            for (int n = 0; n < rowSize; n++) {
                float diff = inputData[base + n] - mean;
                varSum += diff * diff;
            }
            float variance = varSum / normSize;
            float std = (float) Math.sqrt(variance + epsilon);

            for (int n = 0; n < rowSize; n++) {
                float normalized = (inputData[base + n] - mean) / std;
                if (weightData != null) {
                    normalized = weightData[n] * normalized;
                }
                if (biasData != null) {
                    normalized = normalized + biasData[n];
                }
                outputData[base + n] = normalized;
            }
        }

        return CpuShapeOps.floats(input.backend(), input.shape(), outputData, false);
    }

    /**
     * {@code nn/norm/RMSNorm} forward over the last dim: {@code rms = sqrt(sumSq / n + eps)}, {@code x / rms},
     * then times {@code weight} (optional).
     */
    static Tensor rmsNorm(Tensor input, Tensor weight, float epsilon) {
        int normalizedShape = input.shape().dim(input.shape().rank() - 1);
        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];
        float[] weightData = weight == null ? null : weight.toFloatArray();
        int rows = normalizedShape == 0 ? 0 : inputData.length / normalizedShape;

        for (int r = 0; r < rows; r++) {
            int offset = r * normalizedShape;

            float sumSq = 0.0f;
            for (int i = 0; i < normalizedShape; i++) {
                float val = inputData[offset + i];
                sumSq += val * val;
            }

            float rms = (float) Math.sqrt(sumSq / normalizedShape + epsilon);

            for (int i = 0; i < normalizedShape; i++) {
                float normalized = inputData[offset + i] / rms;
                if (weightData != null) {
                    normalized = normalized * weightData[i];
                }
                outputData[offset + i] = normalized;
            }
        }

        return CpuShapeOps.floats(input.backend(), input.shape(), outputData, false);
    }

    /**
     * {@code nn/norm/BatchNorm1d}/{@code BatchNorm2d} forward for an input {@code [B, C, *]}: element
     * {@code (b, c, s)} is {@code x[(b * C + c) * S + s]}; sums run in ascending {@code b}, then spatial order. In
     * training mode the batch mean and biased batch variance normalize the input and, when running statistics are
     * given, they are updated as PyTorch's {@code _BatchNorm} does, {@code running_mean = (1 - momentum) *
     * running_mean + momentum * mean} and {@code running_var = (1 - momentum) * running_var + momentum * var * n /
     * (n - 1)} (the unbiased batch variance, {@code n = B * prod(*)}, see {@link #runningVarWeight}), and written
     * back with {@link GpuOptimizerOps#storeFloats}; in evaluation mode the running statistics normalize it.
     * {@code gamma} and {@code beta} are given together or not at all.
     *
     * @throws IllegalArgumentException in training mode when a channel has a single value ({@link
     *                                  #requireBatchStatistics})
     */
    static Tensor batchNorm(Tensor input, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                            boolean training, float momentum, float epsilon) {
        Shape inputShape = input.shape();
        int batchSize = inputShape.dim(0);
        int numFeatures = inputShape.dim(1);
        int spatialSize = 1;
        for (int i = 2; i < inputShape.rank(); i++) {
            spatialSize *= inputShape.dim(i);
        }
        if (training) {
            requireBatchStatistics(inputShape, batchSize, spatialSize);
        }
        boolean affine = gamma != null;

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];
        float[] meanData = runningMean == null ? null : runningMean.toFloatArray();
        float[] varData = runningVar == null ? null : runningVar.toFloatArray();
        float[] weightData = affine ? gamma.toFloatArray() : null;
        float[] biasData = affine ? beta.toFloatArray() : null;
        int count = batchSize * spatialSize;
        float varWeight = runningVarWeight(momentum, count);

        for (int c = 0; c < numFeatures; c++) {
            float mean;
            float variance;
            if (training) {
                float sum = 0.0f;
                for (int b = 0; b < batchSize; b++) {
                    int base = (b * numFeatures + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        sum += inputData[base + s];
                    }
                }
                mean = sum / count;

                float varSum = 0.0f;
                for (int b = 0; b < batchSize; b++) {
                    int base = (b * numFeatures + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float diff = inputData[base + s] - mean;
                        varSum += diff * diff;
                    }
                }
                variance = varSum / count;

                if (meanData != null) {
                    meanData[c] = (1 - momentum) * meanData[c] + momentum * mean;
                    varData[c] = (1 - momentum) * varData[c] + varWeight * variance;
                }
            } else {
                mean = meanData[c];
                variance = varData[c];
            }

            float std = (float) Math.sqrt(variance + epsilon);
            for (int b = 0; b < batchSize; b++) {
                int base = (b * numFeatures + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    float normalized = (inputData[base + s] - mean) / std;
                    if (affine) {
                        normalized = weightData[c] * normalized + biasData[c];
                    }
                    outputData[base + s] = normalized;
                }
            }
        }

        if (training && meanData != null) {
            GpuOptimizerOps.storeFloats(runningMean, meanData);
            GpuOptimizerOps.storeFloats(runningVar, varData);
        }
        return CpuShapeOps.floats(input.backend(), input.shape(), outputData, false);
    }

    /**
     * PyTorch's training-mode batch-size rule for batch normalization: every channel needs more than one value
     * ({@code n = B * prod(*) > 1}), since its unbiased variance {@code var * n / (n - 1)} feeds the running update.
     * Shared by the device dispatch and this CPU reference.
     *
     * @throws IllegalArgumentException when {@code batch * spatial == 1}
     */
    static void requireBatchStatistics(Shape shape, long batch, long spatial) {
        if (batch * spatial == 1) {
            throw new IllegalArgumentException("Expected more than 1 value per channel when training, got input size "
                    + shape);
        }
    }

    /**
     * Weight of the biased batch variance in the running-variance update, {@code momentum * n / (n - 1)} for
     * {@code n = count} values per channel: {@code running_var = (1 - momentum) * running_var + weight * var} blends in
     * the unbiased batch variance as PyTorch's {@code _BatchNorm} does. The device update and this CPU reference both
     * use it, so they agree to rounding.
     */
    static float runningVarWeight(float momentum, long count) {
        return (float) (momentum * ((double) count / (count - 1)));
    }

    /**
     * {@code nn/norm/GroupNorm} forward for an input {@code [B, C, *]} with {@code numGroups} groups of
     * {@code C / numGroups} channels; each group's sums run in ascending channel, then spatial order. {@code gamma}
     * and {@code beta} are per channel, given together or not at all.
     */
    static Tensor groupNorm(Tensor input, int numGroups, Tensor gamma, Tensor beta, float epsilon) {
        Shape inputShape = input.shape();
        int batchSize = inputShape.dim(0);
        int numChannels = inputShape.dim(1);
        boolean affine = gamma != null;
        int channelsPerGroup = numChannels / numGroups;
        int spatialSize = 1;
        for (int i = 2; i < inputShape.rank(); i++) {
            spatialSize *= inputShape.dim(i);
        }
        int groupSize = channelsPerGroup * spatialSize;

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];
        float[] weightData = affine ? gamma.toFloatArray() : null;
        float[] biasData = affine ? beta.toFloatArray() : null;

        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < numGroups; g++) {
                int firstChannel = g * channelsPerGroup;
                int lastChannel = firstChannel + channelsPerGroup;

                float sum = 0.0f;
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        sum += inputData[base + s];
                    }
                }
                float mean = sum / groupSize;

                float varSum = 0.0f;
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float diff = inputData[base + s] - mean;
                        varSum += diff * diff;
                    }
                }
                float variance = varSum / groupSize;
                float std = (float) Math.sqrt(variance + epsilon);

                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float normalized = (inputData[base + s] - mean) / std;
                        if (affine) {
                            normalized = weightData[c] * normalized + biasData[c];
                        }
                        outputData[base + s] = normalized;
                    }
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), input.shape(), outputData, false);
    }

    // ------------------------------------------------------------------ backward

    /**
     * LayerNorm backward: per row the forward's mean, variance and {@code std}; {@code xhat = (x - mean) / std},
     * {@code a = sum(d) / n}, {@code q = sum(d * xhat) / n}, {@code gx = ((d - a) - xhat * q) / std}. The gamma
     * gradient {@code sum_r gy * xhat} and the beta gradient {@code sum_r gy} accumulate per column, rows ascending.
     */
    static NormOps.Gradients layerNormBackward(Tensor gy, Tensor input, int normSize, Tensor gamma, float epsilon,
                                               boolean needInput, boolean needGamma, Shape betaShape) {
        float[] inputData = input.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] weightData = gamma == null ? null : gamma.toFloatArray();
        int rows = inputData.length / normSize;
        float[] gx = needInput ? new float[inputData.length] : null;
        float[] gw = needGamma ? new float[normSize] : null;
        float[] gb = betaShape != null ? new float[normSize] : null;

        for (int r = 0; r < rows; r++) {
            int base = r * normSize;

            float sum = 0.0f;
            for (int n = 0; n < normSize; n++) {
                sum += inputData[base + n];
            }
            float mean = sum / normSize;

            float varSum = 0.0f;
            for (int n = 0; n < normSize; n++) {
                float diff = inputData[base + n] - mean;
                varSum += diff * diff;
            }
            float variance = varSum / normSize;
            float std = (float) Math.sqrt(variance + epsilon);

            float sumD = 0.0f;
            float sumDX = 0.0f;
            for (int n = 0; n < normSize; n++) {
                float xhat = (inputData[base + n] - mean) / std;
                float g = gradData[base + n];
                float d = weightData != null ? g * weightData[n] : g;
                sumD += d;
                sumDX += d * xhat;
                if (gw != null) {
                    gw[n] += g * xhat;
                }
                if (gb != null) {
                    gb[n] += g;
                }
            }
            if (gx != null) {
                float a = sumD / normSize;
                float q = sumDX / normSize;
                for (int n = 0; n < normSize; n++) {
                    float xhat = (inputData[base + n] - mean) / std;
                    float g = gradData[base + n];
                    float d = weightData != null ? g * weightData[n] : g;
                    gx[base + n] = ((d - a) - xhat * q) / std;
                }
            }
        }
        return gradients(input, gx, needGamma ? gamma.shape() : null, gw, betaShape, gb);
    }

    /**
     * RMSNorm backward over the last dim: per row the forward's {@code rms}; {@code xhat = x / rms},
     * {@code q = sum(d * xhat) / n}, {@code gx = (d - xhat * q) / rms}; the weight gradient {@code sum_r gy * xhat}
     * per column, rows ascending.
     */
    static NormOps.Gradients rmsNormBackward(Tensor gy, Tensor input, Tensor weight, float epsilon,
                                             boolean needInput, boolean needWeight) {
        int normalizedShape = input.shape().dim(input.shape().rank() - 1);
        float[] inputData = input.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] weightData = weight == null ? null : weight.toFloatArray();
        int rows = normalizedShape == 0 ? 0 : inputData.length / normalizedShape;
        float[] gx = needInput ? new float[inputData.length] : null;
        float[] gw = needWeight ? new float[normalizedShape] : null;

        for (int r = 0; r < rows; r++) {
            int offset = r * normalizedShape;

            float sumSq = 0.0f;
            for (int i = 0; i < normalizedShape; i++) {
                float val = inputData[offset + i];
                sumSq += val * val;
            }
            float rms = (float) Math.sqrt(sumSq / normalizedShape + epsilon);

            float sumDX = 0.0f;
            for (int i = 0; i < normalizedShape; i++) {
                float xhat = inputData[offset + i] / rms;
                float g = gradData[offset + i];
                float d = weightData != null ? g * weightData[i] : g;
                sumDX += d * xhat;
                if (gw != null) {
                    gw[i] += g * xhat;
                }
            }
            if (gx != null) {
                float q = sumDX / normalizedShape;
                for (int i = 0; i < normalizedShape; i++) {
                    float xhat = inputData[offset + i] / rms;
                    float g = gradData[offset + i];
                    float d = weightData != null ? g * weightData[i] : g;
                    gx[offset + i] = (d - xhat * q) / rms;
                }
            }
        }
        return gradients(input, gx, needWeight ? weight.shape() : null, gw, null, null);
    }

    /**
     * BatchNorm backward for an input {@code [B, C, *]}, channels outermost and sums in ascending {@code b}, then
     * spatial order. Training mode recomputes the forward's batch statistics, then
     * {@code s1 = sum(gy)}, {@code s2 = sum(gy * xhat)} and
     * {@code gx = gamma * ((gy - s1 / count) - xhat * (s2 / count)) / std} (without the gamma factor when there is
     * no affine step). Evaluation mode uses the given running statistics: {@code gx = (gy * gamma) / std}. In both
     * modes the beta gradient is {@code s1} and the gamma gradient {@code s2}.
     */
    static NormOps.Gradients batchNormBackward(Tensor gy, Tensor input, Tensor gamma, Tensor runningMean,
                                               Tensor runningVar, boolean training, float epsilon,
                                               boolean needInput, boolean needGamma, Shape betaShape) {
        Shape inputShape = input.shape();
        int batchSize = inputShape.dim(0);
        int numFeatures = inputShape.dim(1);
        int spatialSize = 1;
        for (int i = 2; i < inputShape.rank(); i++) {
            spatialSize *= inputShape.dim(i);
        }
        boolean affine = gamma != null;

        float[] inputData = input.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] meanData = training ? null : runningMean.toFloatArray();
        float[] varData = training ? null : runningVar.toFloatArray();
        float[] weightData = affine ? gamma.toFloatArray() : null;
        int count = batchSize * spatialSize;
        float[] gx = needInput ? new float[inputData.length] : null;
        float[] gw = needGamma ? new float[numFeatures] : null;
        float[] gb = betaShape != null ? new float[numFeatures] : null;

        for (int c = 0; c < numFeatures; c++) {
            float mean;
            float variance;
            if (training) {
                float sum = 0.0f;
                for (int b = 0; b < batchSize; b++) {
                    int base = (b * numFeatures + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        sum += inputData[base + s];
                    }
                }
                mean = sum / count;

                float varSum = 0.0f;
                for (int b = 0; b < batchSize; b++) {
                    int base = (b * numFeatures + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float diff = inputData[base + s] - mean;
                        varSum += diff * diff;
                    }
                }
                variance = varSum / count;
            } else {
                mean = meanData[c];
                variance = varData[c];
            }
            float std = (float) Math.sqrt(variance + epsilon);

            float sumGy = 0.0f;
            float sumGyX = 0.0f;
            for (int b = 0; b < batchSize; b++) {
                int base = (b * numFeatures + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    float xhat = (inputData[base + s] - mean) / std;
                    float g = gradData[base + s];
                    sumGy += g;
                    sumGyX += g * xhat;
                }
            }
            if (gb != null) {
                gb[c] = sumGy;
            }
            if (gw != null) {
                gw[c] = sumGyX;
            }
            if (gx == null) {
                continue;
            }
            for (int b = 0; b < batchSize; b++) {
                int base = (b * numFeatures + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    float g = gradData[base + s];
                    if (training) {
                        float xhat = (inputData[base + s] - mean) / std;
                        float t = (g - sumGy / count) - xhat * (sumGyX / count);
                        if (affine) {
                            t = weightData[c] * t;
                        }
                        gx[base + s] = t / std;
                    } else {
                        float d = affine ? g * weightData[c] : g;
                        gx[base + s] = d / std;
                    }
                }
            }
        }
        return gradients(input, gx, needGamma ? gamma.shape() : null, gw, betaShape, gb);
    }

    /**
     * GroupNorm backward for an input {@code [B, C, *]} with {@code numGroups} groups: per (batch, group) the
     * forward's statistics, {@code d = gy * gamma[c]}, {@code s1 = sum(d)}, {@code s2 = sum(d * xhat)} and
     * {@code gx = ((d - s1 / n) - xhat * (s2 / n)) / std} with {@code n} the group's element count. The gamma
     * gradient {@code sum gy * xhat} and the beta gradient {@code sum gy} accumulate per channel in ascending
     * {@code b}, then spatial order.
     */
    static NormOps.Gradients groupNormBackward(Tensor gy, Tensor input, int numGroups, Tensor gamma, float epsilon,
                                               boolean needInput, boolean needGamma, Shape betaShape) {
        Shape inputShape = input.shape();
        int batchSize = inputShape.dim(0);
        int numChannels = inputShape.dim(1);
        boolean affine = gamma != null;
        int channelsPerGroup = numChannels / numGroups;
        int spatialSize = 1;
        for (int i = 2; i < inputShape.rank(); i++) {
            spatialSize *= inputShape.dim(i);
        }
        int groupSize = channelsPerGroup * spatialSize;

        float[] inputData = input.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] weightData = affine ? gamma.toFloatArray() : null;
        float[] gx = needInput ? new float[inputData.length] : null;
        float[] gw = needGamma ? new float[numChannels] : null;
        float[] gb = betaShape != null ? new float[numChannels] : null;

        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < numGroups; g++) {
                int firstChannel = g * channelsPerGroup;
                int lastChannel = firstChannel + channelsPerGroup;

                float sum = 0.0f;
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        sum += inputData[base + s];
                    }
                }
                float mean = sum / groupSize;

                float varSum = 0.0f;
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float diff = inputData[base + s] - mean;
                        varSum += diff * diff;
                    }
                }
                float variance = varSum / groupSize;
                float std = (float) Math.sqrt(variance + epsilon);

                float sumD = 0.0f;
                float sumDX = 0.0f;
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float xhat = (inputData[base + s] - mean) / std;
                        float gv = gradData[base + s];
                        float d = affine ? gv * weightData[c] : gv;
                        sumD += d;
                        sumDX += d * xhat;
                        if (gw != null) {
                            gw[c] += gv * xhat;
                        }
                        if (gb != null) {
                            gb[c] += gv;
                        }
                    }
                }
                if (gx == null) {
                    continue;
                }
                for (int c = firstChannel; c < lastChannel; c++) {
                    int base = (b * numChannels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float xhat = (inputData[base + s] - mean) / std;
                        float gv = gradData[base + s];
                        float d = affine ? gv * weightData[c] : gv;
                        gx[base + s] = ((d - sumD / groupSize) - xhat * (sumDX / groupSize)) / std;
                    }
                }
            }
        }
        return gradients(input, gx, needGamma ? gamma.shape() : null, gw, betaShape, gb);
    }

    /**
     * Uploads the computed gradients (null arrays stay null): {@code gx} in the input's shape, the others in the
     * given shapes. A failed upload closes the tensors uploaded before it.
     */
    private static NormOps.Gradients gradients(Tensor input, float[] gx, Shape weightShape, float[] gw,
                                               Shape biasShape, float[] gb) {
        Tensor inputGrad = null;
        Tensor weightGrad = null;
        try {
            inputGrad = gx == null ? null : CpuShapeOps.floats(input.backend(), input.shape(), gx, false);
            weightGrad = gw == null ? null : CpuShapeOps.floats(input.backend(), weightShape, gw, false);
            Tensor biasGrad = gb == null ? null : CpuShapeOps.floats(input.backend(), biasShape, gb, false);
            return new NormOps.Gradients(inputGrad, weightGrad, biasGrad);
        } catch (RuntimeException e) {
            for (Tensor t : new Tensor[]{inputGrad, weightGrad}) {
                if (t != null) {
                    try {
                        t.close();
                    } catch (RuntimeException closing) {
                        e.addSuppressed(closing);
                    }
                }
            }
            throw e;
        }
    }
}
