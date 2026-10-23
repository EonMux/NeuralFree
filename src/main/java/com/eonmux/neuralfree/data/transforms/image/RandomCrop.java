package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;
import java.util.Random;

public class RandomCrop implements Transform<Transform.ImageData> {
    private final int cropHeight;
    private final int cropWidth;
    private final Random random;
    private final boolean deterministic;
    private int lastTop;
    private int lastLeft;
    
    public RandomCrop(int cropHeight, int cropWidth) {
        this(cropHeight, cropWidth, new Random());
    }
    
    public RandomCrop(int cropHeight, int cropWidth, long seed) {
        this(cropHeight, cropWidth, new Random(seed));
    }
    
    public RandomCrop(int cropHeight, int cropWidth, Random random) {
        if (cropHeight <= 0 || cropWidth <= 0) {
            throw new IllegalArgumentException("Crop dimensions must be positive: " + 
                cropHeight + "x" + cropWidth);
        }
        this.cropHeight = cropHeight;
        this.cropWidth = cropWidth;
        this.random = random != null ? random : new Random();
        this.deterministic = false;
    }
    
    public static RandomCrop of(int cropHeight, int cropWidth) {
        return new RandomCrop(cropHeight, cropWidth);
    }
    
    public static RandomCrop withSeed(int cropHeight, int cropWidth, long seed) {
        return new RandomCrop(cropHeight, cropWidth, seed);
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        if (cropHeight > input.height || cropWidth > input.width) {
            throw new IllegalArgumentException("Crop size (" + cropHeight + "x" + cropWidth + 
                ") exceeds image dimensions (" + input.height + "x" + input.width + ")");
        }
        
        int maxTop = input.height - cropHeight;
        int maxLeft = input.width - cropWidth;
        
        int top = maxTop > 0 ? random.nextInt(maxTop + 1) : 0;
        int left = maxLeft > 0 ? random.nextInt(maxLeft + 1) : 0;
        
        lastTop = top;
        lastLeft = left;
        
        return crop(input, top, left);
    }
    
    public Transform.ImageData applyAt(Transform.ImageData input, int top, int left) {
        if (top < 0 || left < 0 || top + cropHeight > input.height || left + cropWidth > input.width) {
            throw new IllegalArgumentException("Invalid crop position: (" + left + ", " + top + ")");
        }
        
        lastTop = top;
        lastLeft = left;
        
        return crop(input, top, left);
    }
    
    private Transform.ImageData crop(Transform.ImageData input, int top, int left) {
        float[] output = new float[cropHeight * cropWidth * input.channels];
        
        for (int c = 0; c < input.channels; c++) {
            for (int y = 0; y < cropHeight; y++) {
                for (int x = 0; x < cropWidth; x++) {
                    int srcY = top + y;
                    int srcX = left + x;
                    int srcIdx = c * input.height * input.width + srcY * input.width + srcX;
                    int dstIdx = c * cropHeight * cropWidth + y * cropWidth + x;
                    output[dstIdx] = input.data[srcIdx];
                }
            }
        }
        
        return input.withDataAndDimensions(output, cropHeight, cropWidth);
    }
    
    public int lastTop() { return lastTop; }
    public int lastLeft() { return lastLeft; }
    public int cropHeight() { return cropHeight; }
    public int cropWidth() { return cropWidth; }
}
