package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * One llama.cpp {@code -ot} / {@code --override-tensor} entry: a tensor-name pattern and the side that runs its
 * tensors. The pattern is a regular expression matched with {@code find()} against a stored tensor name
 * ({@code blk.5.attn_k.weight}), as llama.cpp matches with {@code std::regex_search}
 * (src/llama-model-loader.cpp:1250-1252); the first matching entry wins. The target is {@code cpu} (the tensors
 * are kept on the host and their layers compute there) or a device target ({@code gpu} or a backend name), which
 * keeps them on the device. The device targets are the GPU backends of {@link EngineConfig.BackendMode}, so a
 * backend added there is a target here. Immutable.
 */
public final class TensorOverride {

    /** Every target of {@link #parse} ({@code cpu}, {@code gpu}, then {@link #deviceTargets()}). */
    private static final List<String> TARGETS = computeTargets();

    private final Pattern pattern;
    private final boolean host;
    private final String target;

    private TensorOverride(Pattern pattern, boolean host, String target) {
        this.pattern = pattern;
        this.host = host;
        this.target = target;
    }

    /**
     * The device targets of {@link #parse}: the GPU backends of {@link EngineConfig.BackendMode} without auto and
     * cpu, lowercase in declaration order ({@code opencl, cuda, vulkan, hip, metal}, then the accelerator APIs);
     * unmodifiable.
     */
    public static List<String> deviceTargets() {
        return Collections.unmodifiableList(TARGETS.subList(2, TARGETS.size()));
    }

    /**
     * Every target of {@link #parse}, lowercase in that order ({@code cpu, gpu, opencl..metal}, then the
     * accelerator APIs); unmodifiable.
     */
    public static List<String> targets() {
        return TARGETS;
    }

    private static List<String> computeTargets() {
        List<String> names = new ArrayList<>(2 + EngineConfig.BackendMode.values().length);
        names.add("cpu");
        names.add("gpu");
        for (EngineConfig.BackendMode mode : EngineConfig.BackendMode.values()) {
            if (mode != EngineConfig.BackendMode.AUTO && mode.usesGpu()) {
                names.add(mode.name().toLowerCase(Locale.ROOT));
            }
        }
        return Collections.unmodifiableList(names);
    }

    /**
     * @param patternText a regular expression over stored tensor names
     * @param target      {@code cpu} for the host, or {@code gpu} / a backend name ({@code opencl}, {@code cuda},
     *                    {@code vulkan}, {@code hip}, {@code metal} or an accelerator API) for the device,
     *                    case-insensitive
     * @throws IllegalArgumentException for a blank pattern, an invalid regular expression or an unknown target
     */
    public static TensorOverride parse(String patternText, String target) {
        if (patternText == null || patternText.trim().isEmpty()) {
            throw new IllegalArgumentException("tensor override pattern is empty");
        }
        Pattern pattern;
        try {
            pattern = Pattern.compile(patternText.trim());
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("tensor override pattern '" + patternText + "' is not a valid"
                    + " regular expression: " + e.getMessage(), e);
        }
        String t = Objects.requireNonNull(target, "target").trim().toLowerCase(Locale.ROOT);
        boolean host;
        if (t.equals("cpu")) {
            host = true;
        } else if (TARGETS.contains(t)) {
            host = false;
        } else {
            throw new IllegalArgumentException("tensor override target '" + target + "' is invalid; use "
                    + String.join(", ", TARGETS));
        }
        return new TensorOverride(pattern, host, t);
    }

    /** The pattern source of the override. */
    public String pattern() {
        return pattern.pattern();
    }

    /** Whether the pattern matches a stored tensor name ({@code regex_search} semantics). */
    public boolean matches(String storedName) {
        return pattern.matcher(Objects.requireNonNull(storedName, "storedName")).find();
    }

    /** True when matched tensors are kept on the host. */
    public boolean host() {
        return host;
    }

    /** The configured target name. */
    public String target() {
        return target;
    }

    @Override
    public String toString() {
        return pattern.pattern() + "=" + target;
    }
}
