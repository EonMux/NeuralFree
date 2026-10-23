package com.eonmux.neuralfree.datatype.structures;

import org.magicwerk.brownies.collections.primitive.FloatGapList;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A growable list of primitive floats ({@link FloatGapList}) that can be filled from an array or with random values,
 * with value-based equality.
 */
public class FloatDataSet extends FloatGapList {
    public FloatDataSet() {
        super();
    }

    /**
     * Initializes this dataset with a set of random values of a given length and min/max bound
     *
     * @param min
     * @param max
     * @param length
     */
    public void randomInit(float min, float max, int length) {
        ThreadLocalRandom currentThread = ThreadLocalRandom.current();
        for (int i = 0; i < length; i++) {
            this.add((float) currentThread.nextDouble(min, max));
        }
    }

    /**
     * Add an array of floats to this data set
     * @param floats the float[]
     */
    public void addAll(float[] floats) {
        for(float f : floats) {
            this.add(f);
        }
    }

    @Override
    public boolean equals(Object obj) {
        // Check if object references are the same
        if (this == obj) {
            return true;
        }

        // Check the instance type
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }

        // Cast to our Type
        FloatDataSet dataSet = (FloatDataSet) obj;

        // Check if our keys and values are equal
        return (Arrays.equals(dataSet.toArray(), this.toArray()));
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(this.toArray());
    }
}
