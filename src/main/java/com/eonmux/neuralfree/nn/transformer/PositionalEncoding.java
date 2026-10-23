package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.Tensor;

public interface PositionalEncoding {
    Tensor apply(Tensor input, int offset);
    
    default Tensor apply(Tensor input) {
        return apply(input, 0);
    }
    
    int maxSequenceLength();
    
    int dModel();
    
    String name();
}
