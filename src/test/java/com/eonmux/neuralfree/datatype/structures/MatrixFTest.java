package com.eonmux.neuralfree.datatype.structures;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Construction, element access, size and layout of {@link MatrixF}. */
public class MatrixFTest {

    @Test
    public void constructionKeepsDimensionsAndData() {
        float[] data = {1, 2, 3, 4};
        MatrixF matrix = new MatrixF(2, 2, data, MatrixF.Layout.ROW_MAJOR);

        assertEquals(2, matrix.getNumRow());
        assertEquals(2, matrix.getNumCol());
        assertArrayEquals(data, matrix.getData(), 0f);
    }

    @Test
    public void setValueThenGetValueRoundTrips() {
        MatrixF matrix = new MatrixF(2, 2, new float[4], MatrixF.Layout.ROW_MAJOR);

        matrix.setValue(0, 0, 1.0f);
        matrix.setValue(0, 1, 2.0f);
        matrix.setValue(1, 0, 3.0f);
        matrix.setValue(1, 1, 4.0f);

        assertEquals(1.0f, matrix.getValue(0, 0), 0f);
        assertEquals(2.0f, matrix.getValue(0, 1), 0f);
        assertEquals(3.0f, matrix.getValue(1, 0), 0f);
        assertEquals(4.0f, matrix.getValue(1, 1), 0f);
    }

    @Test
    public void sizeIsRowsTimesColumns() {
        MatrixF matrix = new MatrixF(3, 4, new float[12], MatrixF.Layout.ROW_MAJOR);

        assertEquals(12, matrix.getSize());
    }

    @Test
    public void layoutIsKept() {
        MatrixF rowMajor = new MatrixF(2, 2, new float[4], MatrixF.Layout.ROW_MAJOR);
        MatrixF colMajor = new MatrixF(2, 2, new float[4], MatrixF.Layout.COL_MAJOR);

        assertEquals(MatrixF.Layout.ROW_MAJOR, rowMajor.getLayout());
        assertEquals(MatrixF.Layout.COL_MAJOR, colMajor.getLayout());
    }
}
