package com.eonmux.neuralfree.gguf;

public final class GGUFTensorInfo {
    private final String name;
    private final long[] dims;
    private final GGMLType type;
    private final long offset;
    
    public GGUFTensorInfo(String name, long[] dims, GGMLType type, long offset) {
        this.name = name;
        this.dims = dims;
        this.type = type;
        this.offset = offset;
    }
    
    public String name() { return name; }
    public long[] dims() { return dims; }
    public GGMLType type() { return type; }
    public long offset() { return offset; }
    
    public long numElements() {
        long n = 1;
        for (long d : dims) n *= d;
        return n;
    }
    
    public int rank() { return dims.length; }
    
    public long dim(int i) { 
        return i >= 0 && i < dims.length ? dims[i] : 1; 
    }
    
    public long sizeInBytes() {
        return type.getBytesForElements(numElements());
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(name);
        sb.append(" [");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        sb.append("] type=").append(type).append(" offset=").append(offset);
        return sb.toString();
    }
}
