package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;
import java.util.Random;

public final class WeightInit {
    private final ComputeBackend backend;
    private final Random random;

    public WeightInit(ComputeBackend backend) {
        this(backend, new Random());
    }

    public WeightInit(ComputeBackend backend, long seed) {
        this(backend, new Random(seed));
    }

    public WeightInit(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        this.backend = backend;
        this.random = random;
    }

    public Tensor xavier(Shape shape) {
        return xavier(shape, XavierMode.DEFAULT);
    }

    public Tensor xavier(Shape shape, XavierMode mode) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(mode, "Mode cannot be null");

        long numElements = shape.numElements();
        if (numElements == 0) {
            throw new IllegalArgumentException("Shape must have at least one element");
        }

        int fanIn = computeFanIn(shape);
        int fanOut = computeFanOut(shape);

        double std;
        switch (mode) {
            case UNIFORM:
                double limit = Math.sqrt(6.0 / (fanIn + fanOut));
                return uniform(shape, -limit, limit);
            case DEFAULT:
            default:
                std = Math.sqrt(2.0 / (fanIn + fanOut));
                return normal(shape, 0.0, std);
        }
    }

    public Tensor kaiming(Shape shape) {
        return kaiming(shape, KaimingMode.FAN_IN, Nonlinearity.RELU);
    }

    public Tensor kaiming(Shape shape, KaimingMode mode, Nonlinearity nonlinearity) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(mode, "Mode cannot be null");
        Objects.requireNonNull(nonlinearity, "Nonlinearity cannot be null");

        long numElements = shape.numElements();
        if (numElements == 0) {
            throw new IllegalArgumentException("Shape must have at least one element");
        }

        int fan = mode == KaimingMode.FAN_IN ? computeFanIn(shape) : computeFanOut(shape);
        
        double gain = nonlinearityGain(nonlinearity);
        double std = gain / Math.sqrt(fan);
        
        return normal(shape, 0.0, std);
    }

    public Tensor normal(Shape shape, double mean, double std) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (std <= 0) {
            throw new IllegalArgumentException("Standard deviation must be positive, got: " + std);
        }

        Tensor randn = Tensor.randn(backend, shape, random.nextLong());
        if (mean != 0.0 || std != 1.0) {
            Tensor meanTensor = Tensor.scalar(backend, (float) mean);
            Tensor stdTensor = Tensor.scalar(backend, (float) std);
            randn = TensorOps.add(TensorOps.mul(randn, stdTensor), meanTensor);
        }
        return randn;
    }

    public Tensor normal(Shape shape) {
        return normal(shape, 0.0, 1.0);
    }

    public Tensor uniform(Shape shape, double low, double high) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (low >= high) {
            throw new IllegalArgumentException("Low must be less than high: low=" + low + ", high=" + high);
        }

        Tensor rand = Tensor.rand(backend, shape, random.nextLong());
        Tensor lowTensor = Tensor.scalar(backend, (float) low);
        Tensor rangeTensor = Tensor.scalar(backend, (float) (high - low));
        
        return TensorOps.add(TensorOps.mul(rand, rangeTensor), lowTensor);
    }

    public Tensor uniform(Shape shape) {
        return uniform(shape, 0.0, 1.0);
    }

    public Tensor constant(Shape shape, double value) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        Tensor result = Tensor.zeros(backend, shape);
        if (value != 0.0) {
            Tensor valueTensor = Tensor.scalar(backend, (float) value);
            result = TensorOps.add(result, valueTensor);
        }
        return result;
    }

    public Tensor zeros(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        return Tensor.zeros(backend, shape);
    }

    public Tensor ones(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        return Tensor.ones(backend, shape);
    }

    public Tensor orthogonal(Shape shape) {
        return orthogonal(shape, 1.0);
    }

    public Tensor orthogonal(Shape shape, double gain) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (shape.rank() < 2) {
            return normal(shape, 0.0, gain);
        }

        int rows = (int) shape.dim(0);
        int cols = (int) shape.numElements() / rows;
        
        int n = Math.max(rows, cols);
        int m = Math.min(rows, cols);
        
        float[][] matrix = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = (float) random.nextGaussian();
            }
        }
        
        for (int col = 0; col < m; col++) {
            for (int row = 0; row < n; row++) {
                for (int prevCol = 0; prevCol < col; prevCol++) {
                    float dot = 0;
                    for (int i = 0; i < n; i++) {
                        dot += matrix[i][prevCol] * matrix[i][col];
                    }
                    for (int i = 0; i < n; i++) {
                        matrix[i][col] -= dot * matrix[i][prevCol];
                    }
                }
                
                float norm = 0;
                for (int i = 0; i < n; i++) {
                    norm += matrix[i][col] * matrix[i][col];
                }
                norm = (float) Math.sqrt(norm);
                
                if (norm > 1e-6f) {
                    for (int i = 0; i < n; i++) {
                        matrix[i][col] /= norm;
                    }
                }
            }
        }
        
        float[] flatData = new float[(int) shape.numElements()];
        int idx = 0;
        if (rows >= cols) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    flatData[idx++] = matrix[i][j] * (float) gain;
                }
            }
        } else {
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    flatData[idx++] = matrix[i][j] * (float) gain;
                }
            }
        }
        
        return Tensor.fromArray(backend, flatData).reshape(shape);
    }

    public Tensor truncatedNormal(Shape shape, double mean, double std, double min, double max) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (std <= 0) {
            throw new IllegalArgumentException("Standard deviation must be positive, got: " + std);
        }
        if (min >= max) {
            throw new IllegalArgumentException("Min must be less than max: min=" + min + ", max=" + max);
        }

        int numElements = (int) shape.numElements();
        float[] data = new float[numElements];
        
        for (int i = 0; i < numElements; i++) {
            double value;
            do {
                value = random.nextGaussian() * std + mean;
            } while (value < min || value > max);
            data[i] = (float) value;
        }
        
        return Tensor.fromArray(backend, data).reshape(shape);
    }

    public Tensor truncatedNormal(Shape shape) {
        return truncatedNormal(shape, 0.0, 1.0, -2.0, 2.0);
    }

    public Tensor eye(int n) {
        return eye(n, n);
    }

    public Tensor eye(int rows, int cols) {
        return Tensor.eye(backend, rows, cols);
    }

    public Tensor sparse(Shape shape, double sparsity, double std) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (sparsity < 0 || sparsity > 1) {
            throw new IllegalArgumentException("Sparsity must be in [0, 1], got: " + sparsity);
        }
        if (std <= 0) {
            throw new IllegalArgumentException("Standard deviation must be positive, got: " + std);
        }

        int numElements = (int) shape.numElements();
        float[] data = new float[numElements];
        
        int numNonZero = (int) (numElements * (1 - sparsity));
        int[] indices = new int[numElements];
        for (int i = 0; i < numElements; i++) {
            indices[i] = i;
        }
        
        for (int i = 0; i < numNonZero; i++) {
            int j = i + random.nextInt(numElements - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
            data[indices[i]] = (float) (random.nextGaussian() * std);
        }
        
        return Tensor.fromArray(backend, data).reshape(shape);
    }

    private int computeFanIn(Shape shape) {
        if (shape.rank() < 2) {
            return (int) shape.numElements();
        }
        
        int fanIn = (int) shape.dim(shape.rank() - 1);
        
        if (shape.rank() > 2) {
            int receptiveField = 1;
            for (int i = 2; i < shape.rank(); i++) {
                receptiveField *= (int) shape.dim(i);
            }
            fanIn *= receptiveField;
        }
        
        return fanIn;
    }

    private int computeFanOut(Shape shape) {
        if (shape.rank() < 2) {
            return (int) shape.numElements();
        }
        
        int fanOut = (int) shape.dim(0);
        
        if (shape.rank() > 2) {
            int receptiveField = 1;
            for (int i = 2; i < shape.rank(); i++) {
                receptiveField *= (int) shape.dim(i);
            }
            fanOut *= receptiveField;
        }
        
        return fanOut;
    }

    private double nonlinearityGain(Nonlinearity nonlinearity) {
        switch (nonlinearity) {
            case RELU:
            case LEAKY_RELU:
                return Math.sqrt(2.0);
            case TANH:
                return 5.0 / 3.0;
            case SIGMOID:
            case LINEAR:
            case IDENTITY:
            default:
                return 1.0;
        }
    }

    public ComputeBackend backend() {
        return backend;
    }

    public enum XavierMode {
        DEFAULT,
        UNIFORM
    }

    public enum KaimingMode {
        FAN_IN,
        FAN_OUT
    }

    public enum Nonlinearity {
        LINEAR,
        IDENTITY,
        SIGMOID,
        TANH,
        RELU,
        LEAKY_RELU
    }
}
