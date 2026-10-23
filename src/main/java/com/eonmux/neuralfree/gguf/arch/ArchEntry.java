package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One row of an architecture table: the architecture's static traits, its hparams hook and its tensor
 * list. Immutable.
 */
public final class ArchEntry {

    private final ArchTraits traits;
    private final ArchHparamsHook hook;
    private final List<TensorRequirement> tensors;

    private ArchEntry(ArchTraits traits, ArchHparamsHook hook, List<TensorRequirement> tensors) {
        this.traits = traits;
        this.hook = hook;
        this.tensors = tensors;
    }

    /** Entry with traits and a hook; the tensor list is empty. */
    public static ArchEntry of(ArchTraits traits, ArchHparamsHook hook) {
        return of(traits, hook, Collections.<TensorRequirement>emptyList());
    }

    /** Entry with traits, hook and tensor list. */
    public static ArchEntry of(ArchTraits traits, ArchHparamsHook hook, List<TensorRequirement> tensors) {
        Objects.requireNonNull(traits, "traits");
        Objects.requireNonNull(hook, "hook");
        Objects.requireNonNull(tensors, "tensors");
        return new ArchEntry(traits, hook, Collections.unmodifiableList(new ArrayList<>(tensors)));
    }

    public Arch arch() {
        return traits.arch();
    }

    public ArchTraits traits() {
        return traits;
    }

    public ArchHparamsHook hook() {
        return hook;
    }

    /** Required and optional tensors of the architecture. */
    public List<TensorRequirement> tensors() {
        return tensors;
    }

    @Override
    public String toString() {
        return "ArchEntry{" + traits.arch().ggufName() + "}";
    }
}
