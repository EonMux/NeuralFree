package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet.Entry;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The tensors one GPU load keeps in host-mapped memory the device reads in place (the UVA zero-copy
 * tier of vLLM's {@code UVAOffloader}), chosen from {@code neuralfree.cpu.offload.mb} and
 * {@code neuralfree.cpu.offload.params}.
 *
 * <p>Every trunk weight layer the load places on the device contributes its matching tensors, taken
 * in weight-layer order and file order until the next tensor would exceed the budget; the ones taken
 * are neither uploaded nor streamed. An empty parameter set takes tensors of any kind, and a set of
 * name segments restricts the offload the way vLLM's {@code cpu_offload_params} does. A plan that
 * takes no tensor, or a parameter segment that matches no device tensor, is refused by naming the
 * property that caused it. Immutable.</p>
 */
public final class HostMappedPlan {

    private static final HostMappedPlan EMPTY = new HostMappedPlan(Collections.<LayerWeightSet>emptyList());

    private final List<LayerWeightSet> sets;
    private final Set<String> names;
    private final long bytes;
    private final int tensorCount;

    private HostMappedPlan(List<LayerWeightSet> sets) {
        this.sets = Collections.unmodifiableList(new ArrayList<>(sets));
        Set<String> all = new LinkedHashSet<>();
        long total = 0L;
        int tensors = 0;
        for (LayerWeightSet set : this.sets) {
            for (Entry e : set.entries()) {
                all.add(e.name());
                total = Math.addExact(total, e.uploadBytes());
                tensors++;
            }
        }
        this.names = Collections.unmodifiableSet(all);
        this.bytes = total;
        this.tensorCount = tensors;
    }

    /** The empty plan. */
    public static HostMappedPlan empty() {
        return EMPTY;
    }

    /**
     * The host-mapped tensors of {@code spec} for {@code cfg}.
     *
     * @param options  upload planning of the tensors (device types, shard limits, input transforms)
     * @param onDevice which stored tensor names the load places on the device; tensors it keeps in RAM
     *                 already, or never places, are not candidates
     * @throws IllegalArgumentException when a parameter segment matches no device tensor, when no trunk
     *                                  tensor is left to offload, or when the budget is below the first
     *                                  selected tensor
     */
    public static HostMappedPlan plan(EngineConfig cfg, ModelSpec spec, LayerWeightSet.Options options,
                                      Predicate<String> onDevice) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(onDevice, "onDevice");
        long budget = cfg.cpuOffloadBytes();
        if (budget == 0) {
            return EMPTY;
        }
        OffloadParams selection = OffloadParams.of(EngineConfig.PROP_CPU_OFFLOAD_PARAMS, cfg.cpuOffloadParams());
        checkSelection(spec, selection, onDevice);
        LayerWeightSet.Options selected = LayerWeightSet.Options.builder(options.policy())
                .filter(ref -> onDevice.test(ref.name()) && selection.matches(ref.name()))
                .maxShardBytes(options.maxShardBytes())
                .transforms(options.transforms())
                .build();
        List<LayerWeightSet> candidates = LayerWeightSet.forTrunk(spec, selected);
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_CPU_OFFLOAD_MB + "=" + cfg.cpuOffloadMb()
                    + " MiB has no trunk tensor to offload to host-mapped memory; the model keeps its trunk"
                    + " tensors in RAM already" + (selection.isEmpty() ? "" : " or "
                    + EngineConfig.PROP_CPU_OFFLOAD_PARAMS + " " + selection.segments() + " excludes them all"));
        }
        return new HostMappedPlan(withinBudget(cfg, candidates, budget));
    }

    /**
     * Refuses a parameter segment that names no tensor the load places on the device.
     *
     * @throws IllegalArgumentException naming {@link EngineConfig#PROP_CPU_OFFLOAD_PARAMS}
     */
    private static void checkSelection(ModelSpec spec, OffloadParams selection, Predicate<String> onDevice) {
        if (selection.isEmpty()) {
            return;
        }
        List<String> deviceNames = new ArrayList<>();
        for (String name : spec.tensors().names()) {
            if (onDevice.test(name)) {
                deviceNames.add(name);
            }
        }
        List<String> unmatched = selection.unmatched(deviceNames);
        if (!unmatched.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_CPU_OFFLOAD_PARAMS + " " + unmatched
                    + " match no tensor the load places on the device; name whole dotted parts of a stored tensor"
                    + " (for example experts or ffn_gate_exps)");
        }
    }

    /**
     * The prefix of {@code candidates} whose tensors fit {@code budget}, in weight-layer and file order, stopping at
     * the first tensor that would exceed it.
     *
     * @throws IllegalArgumentException naming {@link EngineConfig#PROP_CPU_OFFLOAD_MB} when not one tensor fits
     */
    private static List<LayerWeightSet> withinBudget(EngineConfig cfg, List<LayerWeightSet> candidates, long budget) {
        List<LayerWeightSet> chosen = new ArrayList<>(candidates.size());
        long used = 0L;
        for (LayerWeightSet set : candidates) {
            List<Entry> kept = new ArrayList<>(set.entries().size());
            boolean full = false;
            for (Entry e : set.entries()) {
                long need = e.uploadBytes();
                if (need > budget - used) {
                    full = true;
                    break;
                }
                kept.add(e);
                used += need;
            }
            if (!kept.isEmpty()) {
                chosen.add(set.retaining(kept::contains));
            }
            if (full) {
                break;
            }
        }
        if (chosen.isEmpty()) {
            long smallest = Long.MAX_VALUE;
            for (LayerWeightSet set : candidates) {
                for (Entry e : set.entries()) {
                    smallest = Math.min(smallest, e.uploadBytes());
                }
            }
            throw new IllegalArgumentException(EngineConfig.PROP_CPU_OFFLOAD_MB + "=" + cfg.cpuOffloadMb()
                    + " MiB is below the " + (smallest / (1024 * 1024)) + " MiB of the smallest host-mapped"
                    + " tensor; raise the budget");
        }
        return chosen;
    }

    /** True when the plan offloads nothing. */
    public boolean isEmpty() {
        return sets.isEmpty();
    }

    /** True when the stored tensor {@code name} is host-mapped. */
    public boolean contains(String name) {
        return names.contains(name);
    }

    /** The offloaded tensors by weight layer, in weight-layer order (unmodifiable). */
    public List<LayerWeightSet> sets() {
        return sets;
    }

    /** The stored names of the offloaded tensors (unmodifiable). */
    public Set<String> names() {
        return names;
    }

    /** Host bytes the offloaded tensors hold, in their device types. */
    public long bytes() {
        return bytes;
    }

    /** Number of offloaded tensors. */
    public int tensorCount() {
        return tensorCount;
    }

    /** The plan as a log line. */
    public String describe() {
        long mib = bytes / (1024 * 1024);
        return tensorCount + " tensor(s) of " + sets.size() + " layer(s) in host-mapped memory (" + mib + " MiB)";
    }

    @Override
    public String toString() {
        return "HostMappedPlan{" + tensorCount + " tensors, " + bytes + " B}";
    }
}
