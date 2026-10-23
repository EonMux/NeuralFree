package com.eonmux.neuralfree.tensor;

public enum DType {
    FLOAT16(2),
    FLOAT32(4),
    FLOAT64(8),
    INT8(1),
    INT16(2),
    INT32(4),
    INT64(8),
    BFLOAT16(2),
    BOOL(1);

    private final int byteSize;

    DType(int byteSize) {
        this.byteSize = byteSize;
    }

    public int byteSize() {
        return byteSize;
    }

    public boolean isFloatingPoint() {
        return this == FLOAT16 || this == FLOAT32 || this == FLOAT64 || this == BFLOAT16;
    }

    public boolean isInteger() {
        return this == INT8 || this == INT16 || this == INT32 || this == INT64;
    }

    public boolean isBoolean() {
        return this == BOOL;
    }
}
