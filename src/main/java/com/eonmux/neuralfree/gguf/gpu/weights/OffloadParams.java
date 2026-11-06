package com.eonmux.neuralfree.gguf.gpu.weights;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A set of parameter name segments selecting which tensors an offload tier takes (vLLM's
 * {@code cpu_offload_params} / prefetch {@code offload_params}): a segment matches whole dotted parts
 * of the stored tensor name, so {@code experts} and {@code experts.w2_weight} match
 * {@code mlp.experts.w2_weight} while {@code expert} and {@code w2} do not. That keeps
 * {@code ffn_gate_exps} from also matching {@code ffn_gate_exps_scale}.
 *
 * <p>An empty set selects every tensor (a non-selective offload). Immutable.</p>
 */
public final class OffloadParams {

    private static final OffloadParams NONE = new OffloadParams(Collections.<String>emptyList());

    private final List<String> segments;

    private OffloadParams(List<String> segments) {
        this.segments = Collections.unmodifiableList(segments);
    }

    /** The empty set: matches every tensor. */
    public static OffloadParams none() {
        return NONE;
    }

    /**
     * The set of {@code segments}: comma-free, non-blank parameter name segments.
     *
     * @param property property name the errors name, for example {@code neuralfree.stream.params}
     * @throws IllegalArgumentException when a segment is blank or has an empty dotted part
     */
    public static OffloadParams of(String property, List<String> segments) {
        Objects.requireNonNull(segments, "segments");
        Objects.requireNonNull(property, "property");
        if (segments.isEmpty()) {
            return NONE;
        }
        List<String> copy = new ArrayList<>(segments.size());
        for (String segment : segments) {
            copy.add(checkSegment(property, segment));
        }
        return new OffloadParams(copy);
    }

    private static String checkSegment(String property, String segment) {
        if (segment == null || segment.trim().isEmpty()) {
            throw new IllegalArgumentException(property + " has a blank parameter segment");
        }
        String s = segment.trim();
        for (String part : s.split("\\.", -1)) {
            if (part.isEmpty()) {
                throw new IllegalArgumentException(property + " segment '" + s + "' has an empty dotted part;"
                        + " use whole name parts, for example experts or ffn_gate_exps");
            }
        }
        return s;
    }

    /** True when the set is empty and therefore matches every tensor. */
    public boolean isEmpty() {
        return segments.isEmpty();
    }

    /** The segments in order (unmodifiable). */
    public List<String> segments() {
        return segments;
    }

    /** True when {@code tensorName} contains one of the segments as whole dotted parts, or the set is empty. */
    public boolean matches(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        if (segments.isEmpty()) {
            return true;
        }
        String padded = "." + tensorName + ".";
        for (String segment : segments) {
            if (padded.contains("." + segment + ".")) {
                return true;
            }
        }
        return false;
    }

    /**
     * The segments that match no name of {@code tensorNames}, in order; empty when every segment matches
     * something or the set is empty.
     */
    public List<String> unmatched(Collection<String> tensorNames) {
        Objects.requireNonNull(tensorNames, "tensorNames");
        if (segments.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (String segment : segments) {
            boolean found = false;
            for (String name : tensorNames) {
                if (("." + name + ".").contains("." + segment + ".")) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                out.add(segment);
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return segments.isEmpty() ? "OffloadParams{all}" : "OffloadParams" + segments;
    }
}
