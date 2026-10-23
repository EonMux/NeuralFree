package com.eonmux.neuralfree.data;

import org.junit.Test;

import static org.junit.Assert.*;

public class TensorDatasetTest {

    static class SimpleSample {
        final float input;
        final float target;
        SimpleSample(float input, float target) {
            this.input = input;
            this.target = target;
        }
    }

    @Test(expected = NullPointerException.class)
    public void testSampleNullInput() {
        new TensorDataset.Sample(null, null);
    }
}
