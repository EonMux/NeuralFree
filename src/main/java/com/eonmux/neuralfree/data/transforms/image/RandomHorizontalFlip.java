package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;
import java.util.Random;

public class RandomHorizontalFlip implements Transform<Transform.ImageData> {
    private final double probability;
    private final Random random;
    private boolean lastFlipped;
    
    public RandomHorizontalFlip() {
        this(0.5);
    }
    
    public RandomHorizontalFlip(double probability) {
        this(probability, new Random());
    }
    
    public RandomHorizontalFlip(double probability, long seed) {
        this(probability, new Random(seed));
    }
    
    public RandomHorizontalFlip(double probability, Random random) {
        if (probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("Probability must be between 0 and 1: " + probability);
        }
        this.probability = probability;
        this.random = random != null ? random : new Random();
    }
    
    public static RandomHorizontalFlip withProbability(double probability) {
        return new RandomHorizontalFlip(probability);
    }
    
    public static RandomHorizontalFlip withSeed(long seed) {
        return new RandomHorizontalFlip(0.5, seed);
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        boolean shouldFlip = random.nextDouble() < probability;
        lastFlipped = shouldFlip;
        
        if (!shouldFlip) {
            return input;
        }
        
        return flipHorizontal(input);
    }
    
    public Transform.ImageData applyForced(Transform.ImageData input, boolean flip) {
        lastFlipped = flip;
        
        if (!flip) {
            return input;
        }
        
        return flipHorizontal(input);
    }
    
    private Transform.ImageData flipHorizontal(Transform.ImageData input) {
        float[] output = new float[input.data.length];
        
        for (int c = 0; c < input.channels; c++) {
            for (int y = 0; y < input.height; y++) {
                for (int x = 0; x < input.width; x++) {
                    int srcIdx = c * input.height * input.width + y * input.width + x;
                    int dstIdx = c * input.height * input.width + y * input.width + (input.width - 1 - x);
                    output[dstIdx] = input.data[srcIdx];
                }
            }
        }
        
        return input.withData(output);
    }
    
    public boolean lastFlipped() { return lastFlipped; }
    public double probability() { return probability; }
}
