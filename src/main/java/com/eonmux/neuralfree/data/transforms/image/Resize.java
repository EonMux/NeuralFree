package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;
import com.eonmux.neuralfree.tensor.Tensor;

public class Resize implements Transform<Transform.ImageData> {
    private final int targetHeight;
    private final int targetWidth;
    private final InterpolationMode mode;
    private final com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend;
    
    public enum InterpolationMode {
        NEAREST,
        BILINEAR
    }
    
    public Resize(int targetHeight, int targetWidth, InterpolationMode mode) {
        this(targetHeight, targetWidth, mode, null);
    }
    
    public Resize(int targetHeight, int targetWidth, InterpolationMode mode, 
                  com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
        if (targetHeight <= 0 || targetWidth <= 0) {
            throw new IllegalArgumentException("Target dimensions must be positive: " + 
                targetHeight + "x" + targetWidth);
        }
        this.targetHeight = targetHeight;
        this.targetWidth = targetWidth;
        this.mode = mode != null ? mode : InterpolationMode.BILINEAR;
        this.backend = backend;
    }
    
    public static Resize bilinear(int targetHeight, int targetWidth) {
        return new Resize(targetHeight, targetWidth, InterpolationMode.BILINEAR);
    }
    
    public static Resize bilinear(int targetHeight, int targetWidth, 
                                   com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
        return new Resize(targetHeight, targetWidth, InterpolationMode.BILINEAR, backend);
    }
    
    public static Resize nearest(int targetHeight, int targetWidth) {
        return new Resize(targetHeight, targetWidth, InterpolationMode.NEAREST);
    }
    
    public static Resize nearest(int targetHeight, int targetWidth,
                                  com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
        return new Resize(targetHeight, targetWidth, InterpolationMode.NEAREST, backend);
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        if (input.height == targetHeight && input.width == targetWidth) {
            return input;
        }
        
        if (backend != null) {
            return applyGpu(input);
        }
        return applyCpu(input);
    }
    
    private Transform.ImageData applyCpu(Transform.ImageData input) {
        float[] output = new float[targetHeight * targetWidth * input.channels];
        
        float scaleY = (float) input.height / targetHeight;
        float scaleX = (float) input.width / targetWidth;
        
        for (int c = 0; c < input.channels; c++) {
            for (int y = 0; y < targetHeight; y++) {
                for (int x = 0; x < targetWidth; x++) {
                    int outIdx = c * targetHeight * targetWidth + y * targetWidth + x;
                    
                    if (mode == InterpolationMode.NEAREST) {
                        int srcY = Math.min((int) (y * scaleY), input.height - 1);
                        int srcX = Math.min((int) (x * scaleX), input.width - 1);
                        int inIdx = c * input.height * input.width + srcY * input.width + srcX;
                        output[outIdx] = input.data[inIdx];
                    } else {
                        float srcY = y * scaleY;
                        float srcX = x * scaleX;
                        output[outIdx] = bilinearInterpolate(input, c, srcY, srcX);
                    }
                }
            }
        }
        
        return input.withDataAndDimensions(output, targetHeight, targetWidth);
    }
    
    private float bilinearInterpolate(Transform.ImageData input, int c, float y, float x) {
        int y0 = (int) y;
        int x0 = (int) x;
        int y1 = Math.min(y0 + 1, input.height - 1);
        int x1 = Math.min(x0 + 1, input.width - 1);
        
        float fy = y - y0;
        float fx = x - x0;
        
        int baseIdx = c * input.height * input.width;
        float v00 = input.data[baseIdx + y0 * input.width + x0];
        float v01 = input.data[baseIdx + y0 * input.width + x1];
        float v10 = input.data[baseIdx + y1 * input.width + x0];
        float v11 = input.data[baseIdx + y1 * input.width + x1];
        
        return v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
               v10 * (1 - fx) * fy + v11 * fx * fy;
    }
    
    private Transform.ImageData applyGpu(Transform.ImageData input) {
        return applyCpu(input);
    }
    
    @Override
    public boolean isGpuAccelerated() {
        return backend != null;
    }
    
    public int targetHeight() { return targetHeight; }
    public int targetWidth() { return targetWidth; }
    public InterpolationMode mode() { return mode; }
}
