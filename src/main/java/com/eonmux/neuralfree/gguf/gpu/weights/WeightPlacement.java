package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ops.heads.HostHeadTensors;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecValidator;

import java.util.Objects;
import java.util.function.Predicate;

/**
 * Which stored tensors of a model a GPU load places on the device, resident or streamed, and so charges to its VRAM
 * fit:
 * <ul>
 *   <li>the NextN / MTP blocks ({@code blk.N} for the weight layers of {@link ModelSpec#nextnLayers()}) only when MTP
 *       is loaded. llama.cpp marks them {@code TENSOR_SKIP} unless {@code load_mtp} (llama-model.cpp:3915-3917), drops
 *       skipped tensors without allocating them (llama-model-loader.cpp:1186-1193), sets {@code load_mtp} for an MTP
 *       speculative type (common/common.cpp:1710) and counts the NextN layers in its fit only then
 *       (common/fit.cpp:141-143). A drafter that attaches later places them with {@link #withNextnOnDevice()};</li>
 *   <li>never the tensors only the host heads read ({@link HostHeadTensors});</li>
 *   <li>never the tensors a {@link TensorPlacementOverrides} rule keeps in RAM ({@code --cpu-moe},
 *       {@code --n-cpu-moe}, {@code --n-cpu-ffn}, {@code -ot}); their products run on the host;</li>
 *   <li>never the tensors a host tensor test names (the tensors a hybrid CPU+GPU layer split keeps on the host,
 *       {@code com.eonmux.neuralfree.gguf.hybrid}).</li>
 * </ul>
 * Every other tensor is placed. Immutable.
 */
public final class WeightPlacement {

    private final HostHeadTensors hostOnly;
    private final int nextnFirst;
    private final int nextnEnd;
    private final boolean nextnOnDevice;
    private final TensorPlacementOverrides overrides;
    private final Predicate<String> hostTensors;

    private WeightPlacement(HostHeadTensors hostOnly, int nextnFirst, int nextnEnd, boolean nextnOnDevice,
                            TensorPlacementOverrides overrides, Predicate<String> hostTensors) {
        this.hostOnly = hostOnly;
        this.nextnFirst = nextnFirst;
        this.nextnEnd = nextnEnd;
        this.nextnOnDevice = nextnOnDevice;
        this.overrides = Objects.requireNonNull(overrides, "overrides");
        this.hostTensors = Objects.requireNonNull(hostTensors, "hostTensors");
    }

    /**
     * The placement of {@code spec}'s tensors.
     *
     * @param loadNextn whether the NextN / MTP blocks are placed (MTP drafting configured when the model loads)
     * @throws IllegalStateException when the NextN blocks of the spec are not consecutive weight layers
     */
    public static WeightPlacement of(ModelSpec spec, boolean loadNextn) {
        return of(spec, loadNextn, TensorPlacementOverrides.none());
    }

    /**
     * The placement of {@code spec}'s tensors with per-tensor host overrides.
     *
     * @param loadNextn whether the NextN / MTP blocks are placed (MTP drafting configured when the model loads)
     * @param overrides rules that keep matched tensors in RAM ({@link TensorPlacementOverrides#none()} for none)
     * @throws IllegalStateException when the NextN blocks of the spec are not consecutive weight layers
     */
    public static WeightPlacement of(ModelSpec spec, boolean loadNextn, TensorPlacementOverrides overrides) {
        return of(spec, loadNextn, overrides, n -> false);
    }

    /**
     * The placement of {@code spec}'s tensors with the given tensors kept on the host.
     *
     * @param hostTensors test over stored tensor names of the tensors a hybrid load keeps on the host
     * @throws IllegalStateException when the NextN blocks of the spec are not consecutive weight layers
     */
    public static WeightPlacement of(ModelSpec spec, boolean loadNextn, Predicate<String> hostTensors) {
        return of(spec, loadNextn, TensorPlacementOverrides.none(), hostTensors);
    }

    /**
     * The placement of {@code spec}'s tensors with per-tensor host overrides and additional host tensors.
     *
     * @param loadNextn   whether the NextN / MTP blocks are placed (MTP drafting configured when the model loads)
     * @param overrides   rules that keep matched tensors in RAM ({@link TensorPlacementOverrides#none()} for none)
     * @param hostTensors further test over stored tensor names of the tensors kept on the host
     * @throws IllegalStateException when the NextN blocks of the spec are not consecutive weight layers
     */
    public static WeightPlacement of(ModelSpec spec, boolean loadNextn, TensorPlacementOverrides overrides,
                                     Predicate<String> hostTensors) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(overrides, "overrides");
        Objects.requireNonNull(hostTensors, "hostTensors");
        int first = 0;
        int end = 0;
        if (!spec.nextnLayers().isEmpty()) {
            first = spec.nextnLayers().get(0).weightLayer();
            end = first;
            for (LayerSpec l : spec.nextnLayers()) {
                if (l.weightLayer() != end) {
                    throw new IllegalStateException(spec.arch().ggufName() + ": NextN block " + l.il()
                            + " reads weight layer " + l.weightLayer() + ", expected " + end);
                }
                end++;
            }
        }
        return new WeightPlacement(HostHeadTensors.of(spec), first, end, loadNextn, overrides, hostTensors);
    }

    /** True when {@code t} is placed on the device. */
    public boolean onDevice(GGUFTensorInfo t) {
        return onDeviceName(Objects.requireNonNull(t, "t").name());
    }

    /** True when the tensor stored under {@code storedName} is placed on the device. */
    public boolean onDevice(String storedName) {
        return onDeviceName(storedName);
    }

    /** True when the tensor stored under {@code storedName} is placed on the device. */
    public boolean onDeviceName(String storedName) {
        return !hostOnly.test(storedName) && !hostTensors.test(storedName)
                && (nextnOnDevice || !isNextn(storedName)) && !overrides.host(storedName);
    }

    /** True when {@code t} belongs to a NextN / MTP block. */
    public boolean isNextn(GGUFTensorInfo t) {
        return isNextn(Objects.requireNonNull(t, "t").name());
    }

    /** True when tensor {@code storedName} belongs to a NextN / MTP block. */
    public boolean isNextn(String storedName) {
        return nextnEnd > nextnFirst && ModelSpecValidator.isTrunkTensor(storedName, nextnEnd)
                && !ModelSpecValidator.isTrunkTensor(storedName, nextnFirst);
    }

    /** True when tensor {@code storedName} is read only by the host heads. */
    public boolean isHostOnly(String storedName) {
        return hostOnly.test(storedName);
    }

    /** True when tensor {@code storedName} is kept on the host by the host tensor test. */
    public boolean isHostTensor(String storedName) {
        return hostTensors.test(storedName);
    }

    /** True when a rule keeps tensor {@code storedName} in RAM. */
    public boolean isOverriddenHost(String storedName) {
        return overrides.host(storedName);
    }

    /** True when the model has NextN / MTP blocks. */
    public boolean hasNextn() {
        return nextnEnd > nextnFirst;
    }

    /** True when the NextN / MTP blocks are placed on the device. */
    public boolean nextnOnDevice() {
        return nextnOnDevice;
    }

    /** The per-tensor host overrides of this placement. */
    public TensorPlacementOverrides overrides() {
        return overrides;
    }

    /** This placement with additional stored names kept on the host. */
    public WeightPlacement withHostTensors(Predicate<String> hostTensors) {
        Objects.requireNonNull(hostTensors, "hostTensors");
        return new WeightPlacement(hostOnly, nextnFirst, nextnEnd, nextnOnDevice, overrides,
                this.hostTensors.or(hostTensors));
    }

    /** This placement with the NextN / MTP blocks on the device. */
    public WeightPlacement withNextnOnDevice() {
        return nextnOnDevice ? this
                : new WeightPlacement(hostOnly, nextnFirst, nextnEnd, true, overrides, hostTensors);
    }

    @Override
    public String toString() {
        return "WeightPlacement{nextn=" + (hasNextn() ? "[" + nextnFirst + ", " + nextnEnd + ") "
                + (nextnOnDevice ? "on device" : "on host") : "none") + ", hostOnly=" + hostOnly
                + (overrides.isEmpty() ? "" : ", " + overrides) + '}';
    }
}
