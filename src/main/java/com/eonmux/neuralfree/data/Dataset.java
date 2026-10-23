package com.eonmux.neuralfree.data;

public interface Dataset<T> {
    int size();
    T get(int index);
}
