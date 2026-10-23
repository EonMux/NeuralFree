package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;

public class ModuleDict extends Module implements Map<String, Module> {
    private final Map<String, Module> modules = new LinkedHashMap<>();

    public ModuleDict() {
    }

    public ModuleDict(Map<String, Module> modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        for (Map.Entry<String, Module> entry : modules.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
            "ModuleDict.forward() cannot be called directly. " +
            "Access individual modules by key and call forward() on them."
        );
    }

    public Tensor forward(String key, Tensor input) {
        Module module = get(key);
        if (module == null) {
            throw new IllegalArgumentException("No module found with key: " + key);
        }
        return module.forward(input);
    }

    @Override
    public int size() {
        return modules.size();
    }

    @Override
    public boolean isEmpty() {
        return modules.isEmpty();
    }

    @Override
    public boolean containsKey(Object key) {
        return modules.containsKey(key);
    }

    @Override
    public boolean containsValue(Object value) {
        return modules.containsValue(value);
    }

    @Override
    public Module get(Object key) {
        return modules.get(key);
    }

    /**
     * Makes {@code module} the child {@code key}, as PyTorch's {@code ModuleDict.__setitem__} does: an existing key keeps
     * its position and its child is replaced, a new key is appended. The replaced child is returned unclosed to the
     * caller (null for a new key); this dict takes {@code module} over and closes it on {@link #close()}.
     *
     * @throws IllegalArgumentException when the key is empty or contains '.', or {@code module} is this dict or has it
     *                                  as a descendant; the dict is then left unchanged
     */
    @Override
    public Module put(String key, Module module) {
        Objects.requireNonNull(key, "Key cannot be null");
        Objects.requireNonNull(module, "Module cannot be null");

        validateKey(key);

        Module previous = replaceModule(key, module);
        modules.put(key, module);
        return previous;
    }

    @Override
    public Module remove(Object key) {
        throw new UnsupportedOperationException("ModuleDict does not support remove operations");
    }

    @Override
    public void putAll(Map<? extends String, ? extends Module> m) {
        Objects.requireNonNull(m, "Map cannot be null");
        for (Map.Entry<? extends String, ? extends Module> entry : m.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("ModuleDict does not support clear operations");
    }

    @Override
    public Set<String> keySet() {
        return Collections.unmodifiableSet(modules.keySet());
    }

    @Override
    public Collection<Module> values() {
        return Collections.unmodifiableCollection(modules.values());
    }

    /**
     * A read-only view of the entries in registration order. {@link Map.Entry#setValue} throws
     * {@link UnsupportedOperationException}: a child is replaced only through {@link #put} or {@link #replaceAll}, which
     * keep the registered children in step with this dict.
     */
    @Override
    public Set<Entry<String, Module>> entrySet() {
        return Collections.unmodifiableMap(modules).entrySet();
    }

    /**
     * Replaces every child with {@code function.apply(key, child)}, in registration order, each through {@link #put}: the
     * new children are registered and owned by this dict, and the replaced ones are unregistered without being closed.
     * Every replacement is computed before any is applied, and a replacement that {@link #put} rejects restores the
     * children replaced before it, so a failed call leaves the dict unchanged.
     *
     * @throws NullPointerException     when {@code function} is null or returns null
     * @throws IllegalArgumentException when a returned module is this dict or has it as a descendant
     * @throws IllegalStateException    when this dict or a returned module is closed
     */
    @Override
    public void replaceAll(BiFunction<? super String, ? super Module, ? extends Module> function) {
        Objects.requireNonNull(function, "Function cannot be null");
        List<Map.Entry<String, Module>> replacements = new ArrayList<>(modules.size());
        for (Map.Entry<String, Module> entry : modules.entrySet()) {
            Module replacement = function.apply(entry.getKey(), entry.getValue());
            Objects.requireNonNull(replacement, "Replacement for key '" + entry.getKey() + "' cannot be null");
            replacements.add(new AbstractMap.SimpleImmutableEntry<>(entry.getKey(), replacement));
        }
        List<Map.Entry<String, Module>> applied = new ArrayList<>(replacements.size());
        try {
            for (Map.Entry<String, Module> replacement : replacements) {
                Module previous = put(replacement.getKey(), replacement.getValue());
                applied.add(new AbstractMap.SimpleImmutableEntry<>(replacement.getKey(), previous));
            }
        } catch (RuntimeException e) {
            for (int i = applied.size() - 1; i >= 0; i--) {
                put(applied.get(i).getKey(), applied.get(i).getValue());
            }
            throw e;
        }
    }

    public Module update(Map<String, Module> modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        for (Map.Entry<String, Module> entry : modules.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
        return null;
    }

    public ModuleDict keys(Set<String> keys) {
        Objects.requireNonNull(keys, "Keys cannot be null");
        ModuleDict result = new ModuleDict();
        for (String key : keys) {
            Module module = get(key);
            if (module != null) {
                result.put(key, module);
            }
        }
        return result;
    }

    private void validateKey(String key) {
        if (key.isEmpty()) {
            throw new IllegalArgumentException("Key cannot be empty");
        }
        if (key.contains(".")) {
            throw new IllegalArgumentException("Key cannot contain '.' character: " + key);
        }
    }

    @Override
    public String moduleName() {
        return "ModuleDict";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ModuleDict {\n");
        
        for (Map.Entry<String, Module> entry : modules.entrySet()) {
            sb.append("  '").append(entry.getKey()).append("': ").append(entry.getValue()).append("\n");
        }
        
        sb.append("}");
        return sb.toString();
    }
}
