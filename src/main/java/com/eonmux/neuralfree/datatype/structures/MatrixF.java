package com.eonmux.neuralfree.datatype.structures;

public class MatrixF {

    public enum Layout {
        ROW_MAJOR,
        COL_MAJOR
    }

    float[] data;
    int numCol, numRow;
    Layout layout;

    public MatrixF(int numRow, int numCol) {
        this.numRow = numRow;
        this.numCol = numCol;
        data = new float[numRow * numCol];
        for(int m = 0; m < numRow; m++) {
            for(int n = 0; n < numCol; n++) {
                data[m*numCol + n] = 0f;
            }
        }
        this.layout = Layout.ROW_MAJOR;
    }

    public MatrixF(int numRow, int numCol, float[] data, Layout layout) {
        this.numRow = numRow;
        this.numCol = numCol;
        this.data = data;
        this.layout = layout;
    }

    /**
     * Sets the value of the matrix at the row column position
     * Note that the first row and column are 0,0
     *
     * @param row
     * @param col
     * @param value
     */
    public void setValue(int row, int col, float value) {
        data[col + (row * numCol)] = value;
    }

    /**
     * Sets the value of the matrix at the row column position
     * Note that the first row and column are 0,0
     *
     * @param row
     * @param rowValues
     */
    public void setRowValues(int row, FloatDataSet rowValues) {
        for(int index = 0; index < rowValues.size(); index++) {
            data[index + (row*numCol)] = rowValues.get(index);
        }
    }

    /**
     * Sets the value of the matrix at the row column position
     * Note that the first row and column are 0,0
     *
     * @param row
     * @param rowValues
     */
    public void setRowValues(int row, float[] rowValues) {
        for(int index = 0; index < rowValues.length; index++) {
            data[index + (row*numCol)] = rowValues[index];
        }
    }

    /**
     * Sets the value of the matrix at the row column position
     * Note that the first row and column are 0,0
     *
     * @param row
     * @param col
     */
    public float getValue(int row, int col) {
        return data[col + (row * numCol)];
    }

    public float[] getData() {
        return data;
    }

    public int getNumCol() {
        return numCol;
    }

    public int getNumRow() {
        return numRow;
    }

    /**
     * The size of a matrix is the number of its elements
     * @return numRow * numCol
     */
    public int getSize() {
        return numRow * numCol;
    }

    public Layout getLayout() {
        return layout;
    }

    public void setLayout(Layout layout) {
        this.layout = layout;
    }

    @Override
    public String toString() {
        StringBuilder matrix = new StringBuilder();
        matrix.append("[\n");
        for (int i = 0; i < this.numRow; i++) {
            for (int j = 0; j < this.numCol; j++) {
                matrix.append(this.data[i * this.numCol + j]).append(" ");
            }
            matrix.append("\n");
        }
        matrix.append("]\n");
        return matrix.toString();
    }
}
