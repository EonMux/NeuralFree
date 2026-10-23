package com.eonmux.neuralfree.data.transforms.image;

import com.eonmux.neuralfree.data.transforms.Transform;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

public class ToTensor implements Transform<Transform.ImageData> {
    public enum Layout {
        CHW,
        HWC
    }
    
    private final ComputeBackend backend;
    private final Layout layout;
    private final DType dtype;
    private final boolean scaleTo01;
    
    public ToTensor(ComputeBackend backend) {
        this(backend, Layout.CHW, DType.FLOAT32, true);
    }
    
    public ToTensor(ComputeBackend backend, Layout layout, DType dtype, boolean scaleTo01) {
        this.backend = backend;
        this.layout = layout != null ? layout : Layout.CHW;
        this.dtype = dtype != null ? dtype : DType.FLOAT32;
        this.scaleTo01 = scaleTo01;
    }
    
    public static ToTensor chw(ComputeBackend backend) {
        return new ToTensor(backend, Layout.CHW, DType.FLOAT32, true);
    }
    
    public static ToTensor hwc(ComputeBackend backend) {
        return new ToTensor(backend, Layout.HWC, DType.FLOAT32, true);
    }
    
    public static ToTensor noScale(ComputeBackend backend) {
        return new ToTensor(backend, Layout.CHW, DType.FLOAT32, false);
    }
    
    public static ToTensor uint8(ComputeBackend backend) {
        return new ToTensor(backend, Layout.CHW, DType.INT8, false);
    }
    
    @Override
    public Transform.ImageData apply(Transform.ImageData input) {
        float[] data = input.data;
        
        if (scaleTo01) {
            data = scaleData(data);
        }
        
        return input.withData(data);
    }
    
    public Tensor toTensor(Transform.ImageData input) {
        float[] data = input.data;
        
        if (scaleTo01) {
            data = scaleData(data);
        }
        
        Shape shape;
        if (layout == Layout.CHW) {
            shape = Shape.of(input.channels, input.height, input.width);
        } else {
            shape = Shape.of(input.height, input.width, input.channels);
        }
        
        float[] rearranged = rearrangeData(data, input.height, input.width, input.channels);
        
        return Tensor.fromArray(backend, rearranged).reshape(shape);
    }
    
    private float[] scaleData(float[] data) {
        float[] scaled = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            if (data[i] >= 0 && data[i] <= 255) {
                scaled[i] = data[i] / 255.0f;
            } else {
                scaled[i] = data[i];
            }
        }
        return scaled;
    }
    
    private float[] rearrangeData(float[] data, int height, int width, int channels) {
        float[] rearranged = new float[data.length];
        
        if (layout == Layout.CHW) {
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int srcIdx = c * height * width + y * width + x;
                        int dstIdx = c * height * width + y * width + x;
                        rearranged[dstIdx] = data[srcIdx];
                    }
                }
            }
        } else {
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int srcIdx = c * height * width + y * width + x;
                        int dstIdx = y * width * channels + x * channels + c;
                        rearranged[dstIdx] = data[srcIdx];
                    }
                }
            }
        }
        
        return rearranged;
    }
    
    @Override
    public boolean isGpuAccelerated() {
        return backend != null;
    }
    
    public Layout layout() { return layout; }
    public DType dtype() { return dtype; }
    public boolean scaleTo01() { return scaleTo01; }
}
