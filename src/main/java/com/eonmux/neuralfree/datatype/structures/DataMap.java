package com.eonmux.neuralfree.datatype.structures;

import java.util.HashMap;

/**
 * Extends HashMap
 *
 * @param <K>
 * @param <V>
 */
public class DataMap<K, V> extends HashMap<K, V> {
    public DataMap() {
        super();
    }

    @Override
    public boolean equals(Object obj) {
        // Check if object references are the same
        if (this == obj) {
            return true;
        }

        // Check the instance type
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }

        // Cast to our Type
        DataMap<?, ?> dataMap = (DataMap<?, ?>) obj;

        // Check if our keys and values are equal
        return (dataMap.keySet().equals(this.keySet()) && dataMap.values().equals(this.values()));
    }

    @Override
    public int hashCode() {
        int h = 0;
        for (Entry<K, V> entry : entrySet()) {
            h += entry.hashCode();
        }
        return h;
    }
}
