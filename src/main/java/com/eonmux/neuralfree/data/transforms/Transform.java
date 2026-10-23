package com.eonmux.neuralfree.data.transforms;

import com.eonmux.neuralfree.tensor.Tensor;

public interface Transform<T> {
    T apply(T input);
    
    default boolean isGpuAccelerated() {
        return false;
    }
    
    static class ImageData {
        public final float[] data;
        public final int height;
        public final int width;
        public final int channels;
        
        public ImageData(float[] data, int height, int width, int channels) {
            this.data = data;
            this.height = height;
            this.width = width;
            this.channels = channels;
        }
        
        public int totalPixels() {
            return height * width * channels;
        }
        
        public ImageData withData(float[] newData) {
            return new ImageData(newData, height, width, channels);
        }
        
        public ImageData withDimensions(int newHeight, int newWidth) {
            return new ImageData(data, newHeight, newWidth, channels);
        }
        
        public ImageData withDataAndDimensions(float[] newData, int newHeight, int newWidth) {
            return new ImageData(newData, newHeight, newWidth, channels);
        }
        
        public Tensor toTensor(com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
            com.eonmux.neuralfree.tensor.Shape shape = com.eonmux.neuralfree.tensor.Shape.of(channels, height, width);
            return Tensor.fromArray(backend, data).reshape(shape);
        }
    }
    
    static class TextData {
        public final String text;
        public final int[] tokens;
        
        public TextData(String text) {
            this.text = text;
            this.tokens = null;
        }
        
        public TextData(String text, int[] tokens) {
            this.text = text;
            this.tokens = tokens;
        }
        
        public TextData withTokens(int[] newTokens) {
            return new TextData(text, newTokens);
        }
        
        public boolean hasTokens() {
            return tokens != null;
        }
        
        public Tensor toTensor(com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
            if (tokens == null) {
                throw new IllegalStateException("Cannot convert to tensor: no tokens available");
            }
            return Tensor.fromArray(backend, tokens);
        }
    }
}
