package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;

public class Normalize implements Transform<Transform.ImageData> {
    private final float[] mean;
    private final float[] std;
    
    public Normalize(float[] mean, float[] std) {
        if (mean == null || std == null) {
            throw new IllegalArgumentException("Mean and std cannot be null");
        }
        if (mean.length != std.length) {
            throw new IllegalArgumentException("Mean and std must have same length: " + 
                mean.length + " vs " + std.length);
        }
        this.mean = mean.clone();
        this.std = std.clone();
        
        for (float s : std) {
            if (s == 0) {
                throw new IllegalArgumentException("Standard deviation cannot be zero");
            }
        }
    }
    
    public static Normalize of(float[] mean, float[] std) {
        return new Normalize(mean, std);
    }
    
    public static Normalize imagenet() {
        return new Normalize(
            new float[]{0.485f, 0.456f, 0.406f},
            new float[]{0.229f, 0.224f, 0.225f}
        );
    }
    
    public static Normalize cifar10() {
        return new Normalize(
            new float[]{0.4914f, 0.4822f, 0.4465f},
            new float[]{0.2470f, 0.2435f, 0.2616f}
        );
    }
    
    public static Normalize mnist() {
        return new Normalize(
            new float[]{0.1307f},
            new float[]{0.3081f}
        );
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        if (input.channels != mean.length) {
            throw new IllegalArgumentException("Number of channels (" + input.channels + 
                ") does not match mean/std length (" + mean.length + ")");
        }
        
        float[] output = new float[input.data.length];
        
        for (int c = 0; c < input.channels; c++) {
            float channelMean = mean[c];
            float channelStd = std[c];
            
            for (int y = 0; y < input.height; y++) {
                for (int x = 0; x < input.width; x++) {
                    int idx = c * input.height * input.width + y * input.width + x;
                    output[idx] = (input.data[idx] - channelMean) / channelStd;
                }
            }
        }
        
        return input.withData(output);
    }
    
    public Transform.ImageData denormalize(Transform.ImageData normalized) {
        if (normalized.channels != mean.length) {
            throw new IllegalArgumentException("Number of channels does not match");
        }
        
        float[] output = new float[normalized.data.length];
        
        for (int c = 0; c < normalized.channels; c++) {
            float channelMean = mean[c];
            float channelStd = std[c];
            
            for (int y = 0; y < normalized.height; y++) {
                for (int x = 0; x < normalized.width; x++) {
                    int idx = c * normalized.height * normalized.width + y * normalized.width + x;
                    output[idx] = normalized.data[idx] * channelStd + channelMean;
                }
            }
        }
        
        return normalized.withData(output);
    }
    
    public float[] mean() { return mean.clone(); }
    public float[] std() { return std.clone(); }
}
