package com.eonmux.neuralfree.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class NeuralUtil {
    private static final Random random = new Random();

    public static Map<Float, Float> randomFloatMap(int numValues) {
        HashMap<Float, Float> map = new HashMap<>();
        for (int i = 0; i < numValues; i++) {
            map.put((float) Math.random(), (float) Math.random());
        }
        return map;
    }

    public static float generateRandomFloat(float lowerBound, float upperBound) {
        if (lowerBound > upperBound) {
            throw new IllegalArgumentException("Lower bound must be smaller than or equal to the upper bound.");
        }
        return lowerBound + random.nextFloat() * (upperBound - lowerBound);
    }
}
