package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One llama-bench {@code -ot}/{@code --override-tensor} sweep member: the tensor-name patterns of one
 * comma-separated group, each with the placement its llama.cpp buffer type stands for. A group's overrides are
 * separated by {@code ;}; the whole {@code -ot} value is split by {@code ,} into one member per sweep, as the
 * reference expands {@code params.tensor_buft_overrides}. Buffer types map onto the engine's placement targets
 * ({@link TensorOverride}): {@code CPU} is the host, and {@code CUDA0}/{@code Vulkan0}/{@code OpenCL0}/
 * {@code HIP0}/{@code Metal0} and their other device indices are that backend (the engine places by backend, not
 * by device index); llama.cpp's HIP build names its buffers {@code ROCm<n>} and its Metal buffers {@code MTL<n>}
 * (formerly {@code Metal<n>}), which map onto {@code hip} and {@code metal}; the accelerator APIs
 * ({@code SYCL0}, {@code XPU0}, {@code MUSA0}, ...) map onto their API target, which a load without the API's
 * installed backend provider refuses by name. A buffer type with no engine counterpart is refused by name.
 * Immutable.
 */
final class BenchTensorOverrides {

    /** The engine's device targets a buffer type maps onto, derived from {@link TensorOverride#deviceTargets()}. */
    private static final Pattern DEVICE_TARGET = Pattern.compile("(?i)("
            + String.join("|", TensorOverride.deviceTargets()) + ")\\d*");

    /** The llama.cpp buffer names that are one of the engine's backends under another name, with a device index. */
    private static final Pattern DEVICE_ALIAS = Pattern.compile("(?i)(rocm|mtl)\\d*");

    /** The engine target of each {@link #DEVICE_ALIAS} buffer name. */
    private static final Map<String, String> ALIAS_TARGET;

    static {
        Map<String, String> aliases = new HashMap<>();
        aliases.put("rocm", "hip");
        aliases.put("mtl", "metal");
        ALIAS_TARGET = Collections.unmodifiableMap(aliases);
    }

    /** The empty member llama-bench prints as {@code none} and applies as no overrides. */
    static final BenchTensorOverrides NONE =
            new BenchTensorOverrides(Collections.<TensorOverride>emptyList(), "none");

    private final List<TensorOverride> overrides;
    private final String text;

    private BenchTensorOverrides(List<TensorOverride> overrides, String text) {
        this.overrides = Collections.unmodifiableList(new ArrayList<>(overrides));
        this.text = text;
    }

    /** The parsed overrides, empty for no override. */
    List<TensorOverride> overrides() {
        return overrides;
    }

    /** The member as llama-bench prints it ({@code pattern=buffer type} joined by {@code ;}, or {@code none}). */
    String text() {
        return text;
    }

    /** Whether the member overrides anything. */
    boolean isEmpty() {
        return overrides.isEmpty();
    }

    /**
     * The engine flag value of this member ({@code pattern=target} entries joined by {@code ,}), or null when it
     * overrides nothing.
     */
    String engineArg() {
        if (overrides.isEmpty()) {
            return null;
        }
        StringBuilder s = new StringBuilder();
        for (TensorOverride o : overrides) {
            if (s.length() > 0) {
                s.append(',');
            }
            s.append(o.pattern()).append('=').append(o.target());
        }
        return s.toString();
    }

    /**
     * Parses one {@code -ot} value into its comma-separated sweep members.
     *
     * @param flag name of the flag being parsed, for the error messages
     * @throws BenchCommand.UsageException for a missing {@code =}, an invalid regular expression, or a buffer type
     *                                     with no engine placement
     */
    static List<BenchTensorOverrides> parse(String flag, String value) throws BenchCommand.UsageException {
        List<BenchTensorOverrides> members = new ArrayList<>();
        for (String group : value.split(",", -1)) {
            List<TensorOverride> overrides = new ArrayList<>();
            List<String> printed = new ArrayList<>();
            for (String rawEntry : group.split(";", -1)) {
                String entry = rawEntry.trim();
                if (entry.isEmpty()) {
                    continue;
                }
                int eq = entry.indexOf('=');
                if (eq <= 0 || eq == entry.length() - 1) {
                    throw new BenchCommand.UsageException(flag + " entry '" + entry
                            + "' is <tensor name pattern>=<buffer type>");
                }
                String pattern = entry.substring(0, eq);
                String bufferType = entry.substring(eq + 1);
                String target = target(flag, entry, bufferType);
                try {
                    overrides.add(TensorOverride.parse(pattern, target));
                } catch (IllegalArgumentException e) {
                    throw new BenchCommand.UsageException(flag + " entry '" + entry + "': " + e.getMessage());
                }
                printed.add(pattern + "=" + bufferType);
            }
            members.add(overrides.isEmpty() ? NONE
                    : new BenchTensorOverrides(overrides, String.join(";", printed)));
        }
        return members;
    }

    /** The engine placement of a llama.cpp buffer type, or a refusal naming it. */
    private static String target(String flag, String entry, String bufferType) throws BenchCommand.UsageException {
        String name = bufferType.trim().toLowerCase(Locale.ROOT);
        if (name.equals("cpu")) {
            return "cpu";
        }
        if (name.equals("gpu")) {
            return "gpu";
        }
        String trimmed = bufferType.trim();
        Matcher device = DEVICE_TARGET.matcher(trimmed);
        if (device.matches()) {
            return device.group(1).toLowerCase(Locale.ROOT);
        }
        Matcher alias = DEVICE_ALIAS.matcher(trimmed);
        if (alias.matches()) {
            return ALIAS_TARGET.get(alias.group(1).toLowerCase(Locale.ROOT));
        }
        throw new BenchCommand.UsageException(flag + " entry '" + entry + "': buffer type '" + bufferType
                + "' has no NeuralFree placement; use CPU, GPU, CUDA, ROCm, HIP, Metal, MTL, Vulkan, OpenCL"
                + " or an accelerator API buffer (" + acceleratorBufferNames()
                + "; a device index after the backend name is accepted)");
    }

    /** The accelerator APIs' buffer names ({@code SYCL, XPU, ...}), for the refusal and the usage text. */
    static String acceleratorBufferNames() {
        List<String> names = new ArrayList<>();
        for (ComputeBackend.BackendType api : ComputeBackend.BackendType.values()) {
            if (api.isAccelerator()) {
                names.add(api.label());
            }
        }
        return String.join(", ", names);
    }

    @Override
    public String toString() {
        return text;
    }
}
