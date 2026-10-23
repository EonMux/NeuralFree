package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The tensors a gradient function saves during the forward pass. The context only holds the references it is given:
 * {@link #clear()} drops them without closing any tensor. A gradient function that saved references of its own
 * closes those in {@link GradientFunction#clearSavedTensors()}.
 */
public final class GradientContext {
    private List<Tensor> savedTensors;
    private boolean saved = false;

    public GradientContext() {
        this.savedTensors = null;
    }

    public void saveForBackward(Tensor... tensors) {
        if (saved) {
            throw new IllegalStateException("saveForBackward() can only be called once during forward pass");
        }
        Objects.requireNonNull(tensors, "Tensors array cannot be null");
        
        savedTensors = new ArrayList<>(tensors.length);
        for (Tensor t : tensors) {
            savedTensors.add(t);
        }
        saved = true;
    }

    public List<Tensor> getSavedTensors() {
        if (savedTensors == null) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableList(savedTensors);
    }

    public boolean hasSavedTensors() {
        return savedTensors != null && !savedTensors.isEmpty();
    }

    public int numSavedTensors() {
        return savedTensors == null ? 0 : savedTensors.size();
    }

    public void clear() {
        if (savedTensors != null) {
            savedTensors.clear();
            savedTensors = null;
        }
        saved = false;
    }
}
