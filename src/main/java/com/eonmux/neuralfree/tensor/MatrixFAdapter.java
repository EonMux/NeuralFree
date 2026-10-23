package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.serialization.LinearMatricesF;
import com.eonmux.neuralfree.datatype.structures.MatrixF;
import com.eonmux.neuralfree.datatype.structures.ObjectSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class MatrixFAdapter {
    
    private final ComputeBackend backend;
    
    public MatrixFAdapter(ComputeBackend backend) {
        this.backend = Objects.requireNonNull(backend, "Backend cannot be null");
    }
    
    public Tensor toTensor(MatrixF matrix) {
        Objects.requireNonNull(matrix, "Matrix cannot be null");
        
        int rows = matrix.getNumRow();
        int cols = matrix.getNumCol();
        float[] data = matrix.getData();
        
        if (matrix.getLayout() == MatrixF.Layout.COL_MAJOR) {
            float[] rowMajorData = new float[data.length];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    rowMajorData[r * cols + c] = data[c * rows + r];
                }
            }
            data = rowMajorData;
        }
        
        float[][] data2D = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data, r * cols, data2D[r], 0, cols);
        }
        
        return Tensor.fromArray(backend, data2D);
    }
    
    public MatrixF fromTensor(Tensor tensor) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        
        if (!tensor.isMatrix()) {
            throw new IllegalArgumentException("Tensor must be 2D (matrix), got rank: " + tensor.shape().rank());
        }
        
        int rows = tensor.shape().dim(0);
        int cols = tensor.shape().dim(1);
        
        float[] data = tensor.toFloatArray();
        
        return new MatrixF(rows, cols, data, MatrixF.Layout.ROW_MAJOR);
    }
    
    public List<Tensor> toTensorList(LinearMatricesF linearMatrices) {
        Objects.requireNonNull(linearMatrices, "LinearMatrices cannot be null");
        
        ObjectSet<MatrixF> matrices = linearMatrices.getPlanar();
        List<Tensor> tensors = new ArrayList<>(matrices.size());
        
        for (int i = 0; i < matrices.size(); i++) {
            MatrixF matrix = matrices.get(i);
            tensors.add(toTensor(matrix));
        }
        
        return tensors;
    }
    
    public LinearMatricesF fromTensors(List<Tensor> tensors) {
        Objects.requireNonNull(tensors, "Tensors list cannot be null");
        
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Tensors list cannot be empty");
        }
        
        ObjectSet<MatrixF> matrices = new ObjectSet<>();
        
        for (Tensor tensor : tensors) {
            matrices.add(fromTensor(tensor));
        }
        
        return new LinearMatricesF(matrices);
    }
    
    public ComputeBackend backend() {
        return backend;
    }
}
