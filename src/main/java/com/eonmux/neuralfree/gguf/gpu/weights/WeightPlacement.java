package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ops.heads.HostHeadTensors;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecValidator;

import java.util.Objects;

/**
 * Which stored tensors of a model a GPU load places on the device, resident or streamed, and so charges to its VRAM
 * fit:
 * <ul>
 *   <li>the NextN / MTP blocks ({@code blk.N} for the weight layers of {@link ModelSpec#nextnLayers()}) only when MTP
 *       is loaded. llama.cpp marks them {@code TENSOR_SKIP} unless {@code load_mtp} (llama-model.cpp:3915-3917), drops
 *       skipped tensors without allocating them (llama-model-loader.cpp:1186-1193), sets {@code load_mtp} for an MTP
 *       speculative type (common/common.cpp:1710) and counts the NextN layers in its fit only then
 *       (common/fit.cpp:141-143). A drafter that attaches later places them with {@link #withNextnOnDevice()};</li>
 *   <li>never the tensors only the host heads read ({@link HostHeadTensors}).</li>
 * </ul>
 * Every other tensor is placed. Immutable.
 */
public final class WeightPlacement {

    private final HostHeadTensors hostOnly;
    private final int nextnFirst;
    private final int nextnEnd;
    private final boolean nextnOnDevice;

    private WeightPlacement(HostHeadTensors hostOnly, int nextnFirst, int nextnEnd, boolean nextnOnDevice) {
        this.hostOnly = hostOnly;
        this.nextnFirst = nextnFirst;
        this.nextnEnd = nextnEnd;
        this.nextnOnDevice = nextnOnDevice;
    }

    /**
     * The placement of {@code spec}'s tensors.
     *
     * @param loadNextn whether the NextN / MTP blocks are placed (MTP drafting configured when the model loads)
     * @throws IllegalStateException when the NextN blocks of the spec are not consecutive weight layers
     */
    public static WeightPlacement of(ModelSpec spec, boolean loadNextn) {
        Objects.requireNonNull(spec, "spec");
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
        return new WeightPlacement(HostHeadTensors.of(spec), first, end, loadNextn);
    }

    /** True when {@code t} is placed on the device. */
    public boolean onDevice(GGUFTensorInfo t) {
        String name = Objects.requireNonNull(t, "t").name();
        return !hostOnly.test(name) && (nextnOnDevice || !isNextn(name));
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

    /** True when the model has NextN / MTP blocks. */
    public boolean hasNextn() {
        return nextnEnd > nextnFirst;
    }

    /** True when the NextN / MTP blocks are placed on the device. */
    public boolean nextnOnDevice() {
        return nextnOnDevice;
    }

    /** This placement with the NextN / MTP blocks on the device. */
    public WeightPlacement withNextnOnDevice() {
        return nextnOnDevice ? this : new WeightPlacement(hostOnly, nextnFirst, nextnEnd, true);
    }

    @Override
    public String toString() {
        return "WeightPlacement{nextn=" + (hasNextn() ? "[" + nextnFirst + ", " + nextnEnd + ") "
                + (nextnOnDevice ? "on device" : "on host") : "none") + ", hostOnly=" + hostOnly + '}';
    }
}
