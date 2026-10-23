package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class QuantizationConfig {
    public enum QuantizationType {
        INT8_SYMMETRIC,
        INT8_ASYMMETRIC,
        INT4_SYMMETRIC,
        FP8_E4M3,
        FP8_E5M2
    }
    
    private final QuantizationType quantizationType;
    private final boolean perChannel;
    private final boolean dynamic;
    private final float scale;
    private final int zeroPoint;
    
    public QuantizationConfig(QuantizationType quantizationType, boolean perChannel, 
                              boolean dynamic, float scale, int zeroPoint) {
        Objects.requireNonNull(quantizationType, "QuantizationType cannot be null");
        
        this.quantizationType = quantizationType;
        this.perChannel = perChannel;
        this.dynamic = dynamic;
        this.scale = scale;
        this.zeroPoint = zeroPoint;
    }
    
    public static QuantizationConfig int8Symmetric() {
        return new QuantizationConfig(QuantizationType.INT8_SYMMETRIC, true, false, 1.0f, 0);
    }
    
    public static QuantizationConfig int8Asymmetric() {
        return new QuantizationConfig(QuantizationType.INT8_ASYMMETRIC, true, false, 1.0f, 128);
    }
    
    public static QuantizationConfig int4Symmetric() {
        return new QuantizationConfig(QuantizationType.INT4_SYMMETRIC, true, false, 1.0f, 0);
    }
    
    public static QuantizationConfig fp8E4M3() {
        return new QuantizationConfig(QuantizationType.FP8_E4M3, false, false, 1.0f, 0);
    }
    
    public static QuantizationConfig fp8E5M2() {
        return new QuantizationConfig(QuantizationType.FP8_E5M2, false, false, 1.0f, 0);
    }
    
    public static QuantizationConfig dynamic(QuantizationType type) {
        return new QuantizationConfig(type, true, true, 1.0f, 0);
    }
    
    public QuantizationType quantizationType() {
        return quantizationType;
    }
    
    public boolean perChannel() {
        return perChannel;
    }
    
    public boolean dynamic() {
        return dynamic;
    }
    
    public float scale() {
        return scale;
    }
    
    public int zeroPoint() {
        return zeroPoint;
    }
    
    public int bits() {
        switch (quantizationType) {
            case INT8_SYMMETRIC:
            case INT8_ASYMMETRIC:
            case FP8_E4M3:
            case FP8_E5M2:
                return 8;
            case INT4_SYMMETRIC:
                return 4;
            default:
                return 8;
        }
    }
    
    public QuantizationConfig withScale(float newScale) {
        return new QuantizationConfig(quantizationType, perChannel, dynamic, newScale, zeroPoint);
    }
    
    public QuantizationConfig withZeroPoint(int newZeroPoint) {
        return new QuantizationConfig(quantizationType, perChannel, dynamic, scale, newZeroPoint);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        QuantizationConfig that = (QuantizationConfig) o;
        return perChannel == that.perChannel &&
               dynamic == that.dynamic &&
               Float.compare(that.scale, scale) == 0 &&
               zeroPoint == that.zeroPoint &&
               quantizationType == that.quantizationType;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(quantizationType, perChannel, dynamic, scale, zeroPoint);
    }
    
    @Override
    public String toString() {
        return "QuantizationConfig{" +
            "type=" + quantizationType +
            ", perChannel=" + perChannel +
            ", dynamic=" + dynamic +
            ", scale=" + scale +
            ", zeroPoint=" + zeroPoint +
            '}';
    }
}
