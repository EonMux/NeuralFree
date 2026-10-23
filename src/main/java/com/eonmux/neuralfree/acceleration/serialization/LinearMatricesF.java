package com.eonmux.neuralfree.acceleration.serialization;

import com.eonmux.neuralfree.datatype.structures.MatrixF;
import com.eonmux.neuralfree.datatype.structures.ObjectSet;
import com.landawn.abacus.util.stream.Stream;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * A linear matrix is a matrix that has been collapsed to 1 dimension
 * and is usually grouped with other collapsed matrices with an offset
 * indicating the position of this matrix from the start of its containing array.
 * <p>
 * Note that since the data storage for matrices is already linear, this just adds an offset indicating the position
 * of every matrix in the array.
 * <p>
 * The LinearMatricesF class handles translation between Planar and Linear types.
 */
public class LinearMatricesF {

    /**
     * Position of each matrix in the linearized array (inclusive)
     * EG: Position index 5 of length 4 is 5,6,7,8 in linearMatricesData
     */
    int[] linearMatricesPosition;
    float[] linearMatricesData;

    int[] matrixRowSizes;
    int[] matrixColSizes;

    MatrixF.Layout layout;

    public LinearMatricesF(ObjectSet<MatrixF> matrices) {
        int numMatrices = matrices.size();

        this.linearMatricesPosition = new int[numMatrices];
        this.matrixRowSizes = new int[numMatrices];
        this.matrixColSizes = new int[numMatrices];

        ObjectSet<Float> linearizedMatrices = new ObjectSet<>();

        int offset = 0;

        for (int i = 0; i < numMatrices; i++) {
            MatrixF curMatrix = matrices.get(i);
            float[] matrixData = curMatrix.getData();
            Float[] matrixBoxedData = IntStream.range(0, matrixData.length).mapToObj(j -> matrixData[j]).toArray(Float[]::new);

            //Ensure all matrices are of the same layout
            //Don't mix row-major and col-major
            if(i == 0) {
                this.layout = curMatrix.getLayout();
            } else {
                assert (this.layout == curMatrix.getLayout());
            }

            linearizedMatrices.addArray(matrixBoxedData, 0, matrixBoxedData.length);
            if (offset > 0) {
                this.linearMatricesPosition[i] = offset;
            } else {
                this.linearMatricesPosition[i] = 0;
            }
            offset += matrixData.length;

            this.matrixRowSizes[i] = curMatrix.getNumRow();
            this.matrixColSizes[i] = curMatrix.getNumCol();
        }

        this.linearMatricesData = Stream.of(linearizedMatrices).mapToFloat(Float::floatValue).toArray();
    }

    public LinearMatricesF(int[] linearMatricesPosition, float[] linearMatricesData, int[] matrixRowSizes, int[] matrixColSizes) {
        this.linearMatricesPosition = linearMatricesPosition;
        this.linearMatricesData = linearMatricesData;
        this.matrixRowSizes = matrixRowSizes;
        this.matrixColSizes = matrixColSizes;
    }

    public LinearMatricesF(int[] matrixRowSizes, int[] matrixColSizes) {
         this.matrixRowSizes = matrixRowSizes;
         this.matrixColSizes = matrixColSizes;

         allocateData();
    }

    private void allocateData() {
        this.linearMatricesPosition = new int[this.matrixRowSizes.length];

        int total = 0;

        for(int i = 0; i < this.matrixRowSizes.length; i++) {
            this.linearMatricesPosition[i] = total;
            total += this.matrixRowSizes[i] * this.matrixColSizes[i];
        }

        this.linearMatricesData = new float[total];
    }

    public ObjectSet<MatrixF> getPlanar() throws NullPointerException {

        Objects.requireNonNull(this.linearMatricesData);
        Objects.requireNonNull(this.linearMatricesPosition);
        Objects.requireNonNull(this.matrixRowSizes);
        Objects.requireNonNull(this.matrixColSizes);

        ObjectSet<MatrixF> planarMatrices = new ObjectSet<>();

        for (int i = 0; i < this.linearMatricesPosition.length; i++) {
            int matrixSize = this.matrixRowSizes[i] * this.matrixColSizes[i];
            int offset = 0;
            if (i > 0) {
                offset = i * (this.matrixRowSizes[i-1] * this.matrixColSizes[i-1]);
            }

            float[] slicedData = Arrays.copyOfRange(linearMatricesData, offset, offset + matrixSize);

            MatrixF nMatrix = new MatrixF(this.matrixRowSizes[i], this.matrixColSizes[i], slicedData, this.layout);
            planarMatrices.add(nMatrix);
        }

        return planarMatrices;
    }

    public int[] getLinearMatricesPosition() {
        return linearMatricesPosition;
    }

    public void setLinearMatricesPosition(int[] linearMatricesPosition) {
        this.linearMatricesPosition = linearMatricesPosition;
    }

    public float[] getLinearMatricesData() {
        return linearMatricesData;
    }

    public void setLinearMatricesData(float[] linearMatricesData) {
        this.linearMatricesData = linearMatricesData;
    }

    public int[] getMatrixRowSizes() {
        return matrixRowSizes;
    }

    public void setMatrixRowSizes(int[] matrixRowSizes) {
        this.matrixRowSizes = matrixRowSizes;
    }

    public int[] getMatrixColSizes() {
        return matrixColSizes;
    }

    public void setMatrixColSizes(int[] matrixColSizes) {
        this.matrixColSizes = matrixColSizes;
    }

}
