package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;

public class ModuleList extends Module implements List<Module> {
    private final List<Module> modules = new ArrayList<>();

    public ModuleList() {
    }

    public ModuleList(int initialCapacity) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Initial capacity cannot be negative: " + initialCapacity);
        }
    }

    public ModuleList(Module... modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        for (int i = 0; i < modules.length; i++) {
            add(modules[i]);
        }
    }

    public ModuleList(Collection<Module> modules) {
        Objects.requireNonNull(modules, "Modules cannot be null");
        int index = 0;
        for (Module module : modules) {
            add(index, module);
            index++;
        }
    }

    /**
     * Applies the modules in order. Each result in between is closed once the next module has consumed it
     * ({@link #forwardStep}); {@code input} stays open and the result belongs to the caller.
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module module : modules) {
            output = forwardStep(output, input, module::forward);
        }
        return output;
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
    public boolean contains(Object o) {
        return modules.contains(o);
    }

    @Override
    public Iterator<Module> iterator() {
        return Collections.unmodifiableList(modules).iterator();
    }

    @Override
    public Object[] toArray() {
        return modules.toArray();
    }

    @Override
    public <T> T[] toArray(T[] a) {
        return modules.toArray(a);
    }

    /**
     * Appends {@code module} as the child named by its index; this list takes it over and closes it on
     * {@link #close()}.
     *
     * @throws IllegalArgumentException when {@code module} is this list or has it as a descendant; the list is then
     *                                  left unchanged
     */
    @Override
    public boolean add(Module module) {
        Objects.requireNonNull(module, "Module cannot be null");
        replaceModule(String.valueOf(modules.size()), module);
        modules.add(module);
        return true;
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException("ModuleList does not support remove operations");
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        return modules.containsAll(c);
    }

    @Override
    public boolean addAll(Collection<? extends Module> c) {
        Objects.requireNonNull(c, "Collection cannot be null");
        for (Module module : c) {
            add(module);
        }
        return !c.isEmpty();
    }

    @Override
    public boolean addAll(int index, Collection<? extends Module> c) {
        throw new UnsupportedOperationException("ModuleList does not support addAll at index");
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("ModuleList does not support removeAll operations");
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("ModuleList does not support retainAll operations");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("ModuleList does not support clear operations");
    }

    @Override
    public Module get(int index) {
        return modules.get(index);
    }

    @Override
    public Module set(int index, Module element) {
        throw new UnsupportedOperationException("ModuleList does not support set operations");
    }

    /**
     * Inserts {@code module} before the child at {@code index}, as PyTorch's {@code ModuleList.insert} does: the
     * children from {@code index} on move up one index and are registered under their new index names, and
     * {@code module} is registered as child {@code index}. Registration order stays index order.
     *
     * @throws IndexOutOfBoundsException when {@code index} is outside {@code [0, size()]}
     * @throws IllegalArgumentException  when {@code module} is this list or has it as a descendant; the list is then
     *                                   left unchanged
     */
    @Override
    public void add(int index, Module module) {
        Objects.requireNonNull(module, "Module cannot be null");
        int size = modules.size();
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + size);
        }

        replaceModule(String.valueOf(index), module);
        for (int i = size; i > index; i--) {
            replaceModule(String.valueOf(i), modules.get(i - 1));
        }
        modules.add(index, module);
    }

    @Override
    public Module remove(int index) {
        throw new UnsupportedOperationException("ModuleList does not support remove operations");
    }

    @Override
    public int indexOf(Object o) {
        return modules.indexOf(o);
    }

    @Override
    public int lastIndexOf(Object o) {
        return modules.lastIndexOf(o);
    }

    @Override
    public ListIterator<Module> listIterator() {
        return Collections.unmodifiableList(modules).listIterator();
    }

    @Override
    public ListIterator<Module> listIterator(int index) {
        return Collections.unmodifiableList(modules).listIterator(index);
    }

    @Override
    public List<Module> subList(int fromIndex, int toIndex) {
        return Collections.unmodifiableList(modules.subList(fromIndex, toIndex));
    }

    @Override
    public String moduleName() {
        return "ModuleList";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ModuleList [\n");
        
        for (int i = 0; i < modules.size(); i++) {
            sb.append("  ").append(i).append(": ").append(modules.get(i)).append("\n");
        }
        
        sb.append("]");
        return sb.toString();
    }
}
