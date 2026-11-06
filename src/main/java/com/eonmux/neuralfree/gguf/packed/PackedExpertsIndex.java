package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSource;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The packed-expert layers of one {@link ModelSource}: the {@code blk.N.ffn_exps_packed} tensors
 * matched with their {@code avalon.experts_packed.blk.N} metadata arrays, each parsed and checked
 * into a {@link PackedExpertEntry}. {@link #of} returns null for a source with neither the tensors
 * nor the keys, so an unpacked model opens exactly as before; a source with one side alone is
 * refused with the layer or key named. A layer whose file also holds one of the three standard
 * expert tensors is refused as duplicated, as Avalon's loader refuses it. Immutable.
 */
public final class PackedExpertsIndex {

    /** {@code blk.<layer>.<suffix>} of a packed tensor. */
    private static final Pattern PACKED_TENSOR = Pattern.compile("blk\\.(\\d+)\\." + PackedExpertEntry.TENSOR_SUFFIX);

    /** {@code blk.<layer>} suffix of a metadata key. */
    private static final Pattern PACKED_KEY = Pattern.compile("blk\\.(\\d+)");

    private final Map<Integer, PackedExpertEntry> entries;
    private final Map<Integer, GGUFTensorInfo> packedTensors;

    private PackedExpertsIndex(Map<Integer, PackedExpertEntry> entries, Map<Integer, GGUFTensorInfo> packedTensors) {
        this.entries = entries;
        this.packedTensors = packedTensors;
    }

    /** Whether {@code name} is a packed-expert tensor name ({@code blk.N.ffn_exps_packed}). */
    public static boolean isPackedTensorName(String name) {
        return PACKED_TENSOR.matcher(name).matches();
    }

    /** Whether {@code key} is a packed-expert metadata key ({@code avalon.experts_packed.blk.N}). */
    public static boolean isMetadataKey(String key) {
        return key.startsWith(PackedExpertEntry.KEY_PREFIX)
                && PACKED_KEY.matcher(key.substring(PackedExpertEntry.KEY_PREFIX.length())).matches();
    }

    /**
     * The packed-expert layers of {@code source}, or null when it holds neither the packed tensors
     * nor their metadata keys.
     *
     * @throws IOException with the layer, key or tensor named when a side is missing, a tensor or
     *                     key carries an unknown layer, the metadata array is malformed, or a
     *                     standard expert tensor of a packed layer is still in the file
     */
    public static PackedExpertsIndex of(ModelSource source) throws IOException {
        Map<Integer, GGUFTensorInfo> tensors = new TreeMap<>();
        for (GGUFTensorInfo t : source.getTensors()) {
            Matcher m = PACKED_TENSOR.matcher(t.name());
            if (m.matches()) {
                tensors.put(Integer.parseInt(m.group(1)), t);
            }
        }
        if (tensors.isEmpty() && !hasMetadataKeys(source)) {
            return null;
        }
        Map<Integer, long[]> values = new TreeMap<>();
        for (Map.Entry<String, Object> e : source.getMetadata().entrySet()) {
            String key = e.getKey();
            if (!isMetadataKey(key)) {
                continue;
            }
            int layer = layerOfKey(key);
            GGUFValueType type = source.getValueType(key);
            GGUFValueType element = source.getArrayElementType(key);
            if (type != GGUFValueType.ARRAY || element != GGUFValueType.UINT64) {
                throw new IOException("packed expert layer " + layer + ": key " + key + " is a "
                        + (type == GGUFValueType.ARRAY ? element + " array" : type) + ", expected a UINT64 array");
            }
            values.put(layer, GGUFValueType.uint64Array(key, (List<?>) e.getValue(), element));
        }
        for (Map.Entry<Integer, GGUFTensorInfo> e : tensors.entrySet()) {
            if (!values.containsKey(e.getKey())) {
                throw new IOException("packed expert layer " + e.getKey() + ": tensor " + e.getValue().name()
                        + " has no " + PackedExpertEntry.KEY_PREFIX + "blk." + e.getKey() + " metadata");
            }
        }
        for (Integer layer : values.keySet()) {
            if (!tensors.containsKey(layer)) {
                throw new IOException("packed expert layer " + layer + ": key " + PackedExpertEntry.KEY_PREFIX
                        + "blk." + layer + " describes no blk." + layer + "." + PackedExpertEntry.TENSOR_SUFFIX
                        + " tensor of " + source.getFilePath());
            }
        }
        Map<Integer, PackedExpertEntry> entries = new LinkedHashMap<>();
        for (Map.Entry<Integer, long[]> e : values.entrySet()) {
            int layer = e.getKey();
            entries.put(layer, PackedExpertEntry.fromMetadata(layer, e.getValue(), tensors.get(layer)));
        }
        checkStandardAbsent(source, entries);
        return new PackedExpertsIndex(entries, Collections.unmodifiableMap(tensors));
    }

    private static boolean hasMetadataKeys(ModelSource source) {
        for (String key : source.getMetadata().keySet()) {
            if (isMetadataKey(key)) {
                return true;
            }
        }
        return false;
    }

    private static int layerOfKey(String key) throws IOException {
        Matcher m = PACKED_KEY.matcher(key.substring(PackedExpertEntry.KEY_PREFIX.length()));
        if (!m.matches()) {
            throw new IOException("malformed packed-expert metadata key '" + key + "'");
        }
        return Integer.parseInt(m.group(1));
    }

    /** Refuses a packed layer whose standard expert tensor is also in the file, as Avalon's loader does. */
    private static void checkStandardAbsent(ModelSource source, Map<Integer, PackedExpertEntry> entries)
            throws IOException {
        List<GGUFTensorInfo> tensors = source.getTensors();
        for (PackedExpertEntry entry : entries.values()) {
            for (PackedExpertEntry.Role role : entry.roles()) {
                String name = role.standardTensorName(entry.layer());
                for (GGUFTensorInfo t : tensors) {
                    if (t.name().equals(name)) {
                        throw new IOException("packed expert layer " + entry.layer() + ": tensor " + name
                                + " is in the file beside the packed tensor " + entry.packedTensorName()
                                + "; the standard and packed layouts cannot both be present");
                    }
                }
            }
        }
    }

    /** The entries by layer, in increasing layer order; unmodifiable. */
    public Map<Integer, PackedExpertEntry> entries() {
        return entries;
    }

    /** The packed tensor description of a layer, or null when the layer is not packed. */
    public GGUFTensorInfo packedTensor(int layer) {
        return packedTensors.get(layer);
    }

    /** The layers with a packed tensor, in increasing order. */
    public List<Integer> layers() {
        return List.copyOf(entries.keySet());
    }

    /** Whether {@code name} is the standard expert tensor of a packed layer. */
    public boolean packsStandardName(String name) {
        for (PackedExpertEntry entry : entries.values()) {
            for (PackedExpertEntry.Role role : entry.roles()) {
                if (role.standardTensorName(entry.layer()).equals(name)) {
                    return true;
                }
            }
        }
        return false;
    }
}
