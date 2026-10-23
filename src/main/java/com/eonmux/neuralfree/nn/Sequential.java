package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class Sequential extends Module {
    private final List<Module> layers = new ArrayList<>();

    public Sequential() {
    }

    public Sequential(Module... modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        for (int i = 0; i < modules.length; i++) {
            add(modules[i], i);
        }
    }

    public Sequential(List<Module> modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        for (int i = 0; i < modules.size(); i++) {
            add(modules.get(i), i);
        }
    }

    public Sequential add(Module module) {
        return add(module, layers.size());
    }

    private Sequential add(Module module, int index) {
        Objects.requireNonNull(module, "Module cannot be null");
        
        layers.add(module);
        registerModule(String.valueOf(index), module);
        
        return this;
    }

    public Module get(int index) {
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + layers.size());
        }
        return layers.get(index);
    }

    public int size() {
        return layers.size();
    }

    public boolean isEmpty() {
        return layers.isEmpty();
    }

    public List<Module> layers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * Applies the layers in order. Each result in between is closed once the next layer has consumed it
     * ({@link #forwardStep}); {@code input} stays open and the result belongs to the caller.
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module layer : layers) {
            output = forwardStep(output, input, layer::forward);
        }
        return output;
    }

    @Override
    public String moduleName() {
        return "Sequential";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Sequential {\n");
        
        for (int i = 0; i < layers.size(); i++) {
            sb.append("  (").append(i).append("): ").append(layers.get(i)).append("\n");
        }
        
        sb.append("}");
        return sb.toString();
    }
}
