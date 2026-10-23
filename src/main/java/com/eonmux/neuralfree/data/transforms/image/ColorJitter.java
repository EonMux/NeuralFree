package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;
import java.util.Random;

public class ColorJitter implements Transform<Transform.ImageData> {
    private final float brightness;
    private final float contrast;
    private final float saturation;
    private final float hue;
    private final Random random;
    
    public ColorJitter() {
        this(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    public ColorJitter(float brightness, float contrast, float saturation, float hue) {
        this(brightness, contrast, saturation, hue, new Random());
    }
    
    public ColorJitter(float brightness, float contrast, float saturation, float hue, long seed) {
        this(brightness, contrast, saturation, hue, new Random(seed));
    }
    
    public ColorJitter(float brightness, float contrast, float saturation, float hue, Random random) {
        if (brightness < 0 || contrast < 0 || saturation < 0) {
            throw new IllegalArgumentException("Brightness, contrast, and saturation must be non-negative");
        }
        if (hue < 0 || hue > 0.5) {
            throw new IllegalArgumentException("Hue must be between 0 and 0.5: " + hue);
        }
        
        this.brightness = brightness;
        this.contrast = contrast;
        this.saturation = saturation;
        this.hue = hue;
        this.random = random != null ? random : new Random();
    }
    
    public static ColorJitter of(float brightness, float contrast, float saturation, float hue) {
        return new ColorJitter(brightness, contrast, saturation, hue);
    }
    
    public static ColorJitter brightnessOnly(float factor) {
        return new ColorJitter(factor, 0, 0, 0);
    }
    
    public static ColorJitter contrastOnly(float factor) {
        return new ColorJitter(0, factor, 0, 0);
    }
    
    public static ColorJitter defaultTransform() {
        return new ColorJitter(0.4f, 0.4f, 0.4f, 0.1f);
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        if (input.channels < 3) {
            return input;
        }
        
        float[] data = input.data.clone();
        
        if (brightness > 0) {
            applyBrightness(data, input.height, input.width, input.channels);
        }
        
        if (contrast > 0) {
            applyContrast(data, input.height, input.width, input.channels);
        }
        
        if (saturation > 0) {
            applySaturation(data, input.height, input.width, input.channels);
        }
        
        if (hue > 0) {
            applyHue(data, input.height, input.width, input.channels);
        }
        
        return input.withData(data);
    }
    
    private void applyBrightness(float[] data, int height, int width, int channels) {
        float factor = 1.0f + (random.nextFloat() * 2 - 1) * brightness;
        factor = Math.max(0, factor);
        
        for (int i = 0; i < data.length; i++) {
            data[i] = clamp(data[i] * factor);
        }
    }
    
    private void applyContrast(float[] data, int height, int width, int channels) {
        float factor = 1.0f + (random.nextFloat() * 2 - 1) * contrast;
        factor = Math.max(0, factor);
        
        float mean = 0;
        int count = 0;
        for (int c = 0; c < Math.min(3, channels); c++) {
            for (int i = 0; i < height * width; i++) {
                mean += data[c * height * width + i];
                count++;
            }
        }
        mean /= count;
        
        for (int i = 0; i < data.length; i++) {
            data[i] = clamp((data[i] - mean) * factor + mean);
        }
    }
    
    private void applySaturation(float[] data, int height, int width, int channels) {
        float factor = 1.0f + (random.nextFloat() * 2 - 1) * saturation;
        factor = Math.max(0, factor);
        
        for (int i = 0; i < height * width; i++) {
            float r = data[0 * height * width + i];
            float g = data[1 * height * width + i];
            float b = data[2 * height * width + i];
            
            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
            
            data[0 * height * width + i] = clamp(gray + factor * (r - gray));
            data[1 * height * width + i] = clamp(gray + factor * (g - gray));
            data[2 * height * width + i] = clamp(gray + factor * (b - gray));
        }
    }
    
    private void applyHue(float[] data, int height, int width, int channels) {
        float factor = (random.nextFloat() * 2 - 1) * hue;
        
        for (int i = 0; i < height * width; i++) {
            float r = data[0 * height * width + i];
            float g = data[1 * height * width + i];
            float b = data[2 * height * width + i];
            
            float[] hsv = rgbToHsv(r, g, b);
            hsv[0] = (hsv[0] + factor + 1) % 1;
            
            float[] rgb = hsvToRgb(hsv[0], hsv[1], hsv[2]);
            data[0 * height * width + i] = rgb[0];
            data[1 * height * width + i] = rgb[1];
            data[2 * height * width + i] = rgb[2];
        }
    }
    
    private float[] rgbToHsv(float r, float g, float b) {
        float max = Math.max(r, Math.max(g, b));
        float min = Math.min(r, Math.min(g, b));
        float diff = max - min;
        
        float h = 0, s = 0, v = max;
        
        if (max > 0) {
            s = diff / max;
        }
        
        if (diff > 0) {
            if (max == r) {
                h = (g - b) / diff + (g < b ? 6 : 0);
            } else if (max == g) {
                h = (b - r) / diff + 2;
            } else {
                h = (r - g) / diff + 4;
            }
            h /= 6;
        }
        
        return new float[]{h, s, v};
    }
    
    private float[] hsvToRgb(float h, float s, float v) {
        float r, g, b;
        
        int i = (int) (h * 6);
        float f = h * 6 - i;
        float p = v * (1 - s);
        float q = v * (1 - f * s);
        float t = v * (1 - (1 - f) * s);
        
        i = i % 6;
        
        switch (i) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
        
        return new float[]{clamp(r), clamp(g), clamp(b)};
    }
    
    private float clamp(float value) {
        return Math.max(0, Math.min(1, value));
    }
    
    public float brightness() { return brightness; }
    public float contrast() { return contrast; }
    public float saturation() { return saturation; }
    public float hue() { return hue; }
}
