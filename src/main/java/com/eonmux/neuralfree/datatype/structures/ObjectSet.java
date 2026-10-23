package com.eonmux.neuralfree.datatype.structures;

import org.magicwerk.brownies.collections.GapList;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A growable list of objects ({@link GapList}) with value-based equality and a copy-returning {@link #reversed()}.
 */
public class ObjectSet<E> extends GapList<E> {
    public ObjectSet() {
        super();
    }

    public static <E> ObjectSet<E> create(Collection<? extends E> coll) {
        ObjectSet<E> set = new ObjectSet<>();
        set.addAll(coll);
        return set;
    }

    /**
     * Returns a new set holding this set's elements in reverse order; this set is not modified.
     */
    public ObjectSet<E> reversed() {
        ObjectSet<E> result = new ObjectSet<>();
        for (int i = size() - 1; i >= 0; i--) {
            result.add(doGet(i));
        }
        return result;
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
        ObjectSet<E> objectSet = (ObjectSet<E>) obj;

        // Check if our keys and values are equal
        return objectSet.size() == this.size() && this.containsAll(objectSet);
    }

    @Override
    public int hashCode() {
        int hashCode = 1;
        int size = size();
        for (int i = 0; i < size; i++) {
            E elem = doGet(i);
            hashCode = 31 * hashCode + Objects.requireNonNullElse(elem.hashCode(), 0);
        }
        return hashCode;
    }

    @Override
    public String toString() {
        StringBuilder eString = new StringBuilder();
        eString.append("{\n");
        for(E elem : this) {
            eString.append(elem != null ? elem.toString() : "null").append(",\n");
        }
        eString.deleteCharAt(eString.length()-1).deleteCharAt(eString.length()-1).append("}\n");
        return eString.toString();
    }

}
