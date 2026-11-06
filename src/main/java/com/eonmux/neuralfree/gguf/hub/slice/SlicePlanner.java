package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.io.safetensors.SafetensorsHeader;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Turns the safetensors headers of a repository and its parsed {@code config.json} into a {@link SlicePlan}: it
 * finds the numeric slot families of the tensor names, decides which are layers, experts or neither, keeps the
 * requested layers and experts with the indices renumbered to {@code 0..n-1}, and row-slices stacked expert and
 * vocab tensors whose first dimension it can cut contiguously. The heuristics and their limits are the reference
 * skill's: a family is a layer family only when its index count is a layer count the config declares (or, without a
 * config, when its container is a layer container and it has at least four indices), and an expert row slice needs
 * the first dimension to equal a declared expert count and the name to mention experts, routers, gates or scores.
 */
final class SlicePlanner {

    private static final Set<String> LAYER_CONTAINERS =
            set("layers", "layer", "blocks", "block", "h");
    private static final String[] EXPERT_ROW_KEYWORDS = {"expert", "router", "gate", "score"};

    private SlicePlanner() {
    }

    /**
     * Plans a slice over the shards {@code weightFiles} and their parsed headers.
     *
     * @throws IllegalArgumentException when an explicit keep-layers index lies outside a layer family, naming it
     */
    static SlicePlan plan(List<String> weightFiles, Map<String, SafetensorsHeader> headers, Object configJson,
                          SliceOptions options) {
        ConfigAnalysis config = ConfigAnalysis.of(configJson);
        Map<String, TensorFamily> families = buildFamilies(weightFiles, headers);
        markLayerFamilies(families, config);
        planFamilies(families, config, options);
        List<TensorSlice> tensors = new ArrayList<>();
        int totalTensors = 0;
        for (String file : weightFiles) {
            SafetensorsHeader header = headers.get(file);
            if (header == null) {
                throw new IllegalArgumentException("no header was read for the shard '" + file + "'");
            }
            long dataStart = header.dataStart();
            for (StTensor tensor : header.tensors()) {
                totalTensors++;
                String outputName = outputName(tensor.name(), families);
                if (outputName == null) {
                    continue;
                }
                tensors.add(slice(tensor, outputName, file, dataStart, config, options));
            }
        }
        return new SlicePlan(weightFiles, families, tensors, totalTensors, config);
    }

    /** Every numeric slot of every tensor name, as a family of the indices seen under it. */
    private static Map<String, TensorFamily> buildFamilies(List<String> weightFiles,
                                                           Map<String, SafetensorsHeader> headers) {
        Map<String, TensorFamily> families = new LinkedHashMap<>();
        for (String file : weightFiles) {
            SafetensorsHeader header = headers.get(file);
            if (header == null) {
                throw new IllegalArgumentException("no header was read for the shard '" + file + "'");
            }
            for (StTensor tensor : header.tensors()) {
                String[] tokens = tensor.name().split("\\.", -1);
                for (TensorFamily.Slot slot : TensorFamily.slots(tensor.name())) {
                    String container = slot.position() > 0 ? tokens[slot.position() - 1] : "";
                    TensorFamily family = families.get(slot.prefix());
                    if (family == null) {
                        family = new TensorFamily(slot.prefix(),
                                container.contains("expert")
                                        ? TensorFamily.Kind.EXPERT : TensorFamily.Kind.OTHER);
                        families.put(slot.prefix(), family);
                    }
                    family.addIndex(slot.index());
                }
            }
        }
        return families;
    }

    /** Marks the families whose index count matches a declared layer count (plus any MTP layers). */
    private static void markLayerFamilies(Map<String, TensorFamily> families, ConfigAnalysis config) {
        Map<Long, Long> acceptable = new HashMap<>();
        for (Long layers : config.layerCounts()) {
            acceptable.put(layers, layers);
            for (Long nextn : config.nextnCounts()) {
                try {
                    acceptable.put(Math.addExact(layers.longValue(), nextn.longValue()), layers);
                } catch (ArithmeticException e) {
                    // a layer count that cannot be added to is no acceptable card either
                }
            }
        }
        for (TensorFamily family : families.values()) {
            if (family.kind() == TensorFamily.Kind.EXPERT) {
                continue;
            }
            if (config.config() != null) {
                Long regular = acceptable.get((long) family.card());
                if (regular != null) {
                    family.makeLayer(regular.longValue());
                }
            } else if (family.card() >= 4
                    && LAYER_CONTAINERS.contains(TensorFamily.container(family.prefix()))) {
                family.makeLayer(family.card());
            }
        }
    }

    /** Chooses the kept indices of every family: the requested layers (with the MTP tail), the first experts. */
    private static void planFamilies(Map<String, TensorFamily> families, ConfigAnalysis config,
                                     SliceOptions options) {
        for (TensorFamily family : families.values()) {
            if (family.kind() == TensorFamily.Kind.LAYER) {
                List<Long> kept = new ArrayList<>();
                if (options.keepLayers() != null) {
                    for (Integer layer : options.keepLayers()) {
                        if (layer.longValue() >= family.regularTotal()) {
                            throw new IllegalArgumentException("keep-layers index " + layer + " is outside the "
                                    + family.regularTotal() + " regular layers of '" + family.prefix() + "'");
                        }
                        kept.add(layer.longValue());
                    }
                } else {
                    long base = Math.min(options.numLayers(), family.regularTotal());
                    for (long i = 0; i < base; i++) {
                        kept.add(i);
                    }
                    long firstDense = config.firstKDense();
                    if (firstDense > 0L && firstDense < family.regularTotal() && firstDense >= base && base >= 1L) {
                        List<Long> withMoe = new ArrayList<>();
                        for (long i = 0; i < base - 1; i++) {
                            withMoe.add(i);
                        }
                        withMoe.add(firstDense);
                        kept = withMoe;
                    }
                }
                family.setNewRegular(kept.size());
                for (Long index : family.indices()) {
                    if (index.longValue() >= family.regularTotal()) {
                        kept.add(index);
                    }
                }
                family.keep(kept);
            } else if (family.kind() == TensorFamily.Kind.EXPERT) {
                List<Long> kept = new ArrayList<>();
                for (Long index : family.indices()) {
                    if (kept.size() >= options.numExperts()) {
                        break;
                    }
                    kept.add(index);
                }
                family.setNewRegular(kept.size());
                family.keep(kept);
            } else {
                family.keepAll();
            }
        }
    }

    /** The output name of one tensor, or null when a family drops it. */
    private static String outputName(String name, Map<String, TensorFamily> families) {
        String[] tokens = name.split("\\.", -1);
        for (TensorFamily.Slot slot : TensorFamily.slots(name)) {
            TensorFamily family = families.get(slot.prefix());
            if (family == null || family.kept() == null) {
                continue;
            }
            Long renumbered = family.renumber(slot.index());
            if (renumbered == null) {
                return null;
            }
            tokens[slot.position()] = Long.toString(renumbered.longValue());
        }
        return String.join(".", tokens);
    }

    /** One tensor's plan, row-slicing a stacked expert or vocab tensor when its first dimension can be cut. */
    private static TensorSlice slice(StTensor tensor, String outputName, String file, long dataStart,
                                     ConfigAnalysis config, SliceOptions options) {
        long[] shape = tensor.shape();
        long[] outputShape = shape;
        long byteLength = tensor.byteLength();
        Long newRows = null;
        if (options.vocabSize() > 0 && shape.length > 0 && shape[0] > options.vocabSize()
                && matchesVocab(shape[0], config.vocabSizes())) {
            newRows = (long) options.vocabSize();
        } else if (shape.length > 0 && config.expertCounts().contains(shape[0])
                && shape[0] > options.numExperts() && containsAny(tensor.name(), EXPERT_ROW_KEYWORDS)) {
            newRows = (long) options.numExperts();
        }
        if (newRows != null && shape[0] > newRows.longValue() && tensor.byteLength() % shape[0] == 0) {
            long rowBytes = tensor.byteLength() / shape[0];
            outputShape = new long[shape.length];
            outputShape[0] = newRows.longValue();
            System.arraycopy(shape, 1, outputShape, 1, shape.length - 1);
            byteLength = newRows.longValue() * rowBytes;
        }
        return new TensorSlice(tensor.name(), outputName, tensor.dtype(), outputShape, shape,
                dataStart + tensor.begin(), byteLength, file);
    }

    /** Whether {@code rows} is a declared vocab size up to 1024 rows of padding above it. */
    private static boolean matchesVocab(long rows, Set<Long> vocabSizes) {
        for (Long vocab : vocabSizes) {
            if (rows >= vocab.longValue() && rows - vocab.longValue() <= 1024L) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsAny(String name, String[] keywords) {
        for (String keyword : keywords) {
            if (name.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private static Set<String> set(String... values) {
        Set<String> set = new HashSet<>();
        Collections.addAll(set, values);
        return Collections.unmodifiableSet(set);
    }
}
