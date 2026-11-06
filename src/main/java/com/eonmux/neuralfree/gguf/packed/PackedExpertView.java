package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A {@link ModelSource} that presents an ordinary MoE model as its experts-packed form, the layout
 * Avalon's {@code pack-experts.py} writes: every layer with the three tensors
 * {@code blk.N.ffn_{up,gate,down}_exps.weight} collapses into one I8 tensor
 * {@code blk.N.ffn_exps_packed} of shape {@code [stride, n_expert]}, described by the
 * {@code avalon.experts_packed.blk.N} metadata array of its {@link PackedExpertEntry}. The packed
 * tensor's bytes are served by {@link PackedExpertStore} and {@link PackedExpertRegions}, assembled
 * from the delegate's expert tensors expert by expert, with every role's pad filled with the next
 * expert's first bytes of the same role (the last expert repeats its own), exactly as the pack
 * script writes them. The delegate stays owned by the caller of {@code ModelSources}, which closes
 * it. Immutable.
 */
public final class PackedExpertView implements ModelSource {

    /** {@code blk.<layer>.<suffix>} of a standard routed-expert tensor ({@code ffn_up_exps.weight}). */
    private static final Pattern EXPERT_TENSOR = Pattern.compile("blk\\.(\\d+)\\.ffn_(up|gate|down)_exps\\.weight");

    private final ModelSource delegate;
    private final Map<Integer, PackedExpertEntry> entries;
    private final List<GGUFTensorInfo> viewTensors;
    private final Map<String, GGUFTensorInfo> expertByName;
    private final List<Packed> packedByLayer;
    private final Map<String, long[]> metadataArrays;

    /**
     * The experts-packed view of {@code delegate}, or the delegate itself when it has no routed
     * experts to pack.
     *
     * @throws IOException with the layer or tensor named when a layer holds some of the three
     *                     expert tensors but not all, or an expert tensor is not a 3-D
     *                     {@code [ne0, ne1, n_expert]} tensor
     */
    public static ModelSource packedViewIfMoe(ModelSource delegate) throws IOException {
        Objects.requireNonNull(delegate, "delegate");
        Map<Integer, List<GGUFTensorInfo>> byLayer = byRole(expertTensors(delegate.getTensors()));
        if (byLayer.isEmpty()) {
            return delegate;
        }
        Map<Integer, PackedExpertEntry> entries = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<GGUFTensorInfo>> e : byLayer.entrySet()) {
            entries.put(e.getKey(), PackedExpertEntry.fromStandardTensors(e.getKey(), e.getValue()));
        }
        return new PackedExpertView(delegate, entries, byLayer);
    }

    /** The routed-expert tensors of the table by exact name. */
    private static Map<String, GGUFTensorInfo> expertTensors(List<GGUFTensorInfo> tensors) {
        Map<String, GGUFTensorInfo> experts = new LinkedHashMap<>();
        for (GGUFTensorInfo t : tensors) {
            if (EXPERT_TENSOR.matcher(t.name()).matches()) {
                experts.put(t.name(), t);
            }
        }
        return experts;
    }

    /** The tensors by layer, each layer's three roles in storage order; empty when none. */
    private static Map<Integer, List<GGUFTensorInfo>> byRole(Map<String, GGUFTensorInfo> experts)
            throws IOException {
        Map<Integer, GGUFTensorInfo[]> byLayer = new LinkedHashMap<>();
        for (Map.Entry<String, GGUFTensorInfo> e : experts.entrySet()) {
            Matcher m = EXPERT_TENSOR.matcher(e.getKey());
            if (!m.matches()) {
                continue;
            }
            int layer = Integer.parseInt(m.group(1));
            int role = roleIndex(m.group(2));
            GGUFTensorInfo[] roles = byLayer.computeIfAbsent(layer,
                    k -> new GGUFTensorInfo[PackedExpertEntry.ROLE_SUFFIXES.length]);
            if (roles[role] != null) {
                throw new IOException("packed expert layer " + layer + ": tensor " + e.getKey() + " is listed twice");
            }
            roles[role] = e.getValue();
        }
        Map<Integer, List<GGUFTensorInfo>> result = new LinkedHashMap<>();
        for (Map.Entry<Integer, GGUFTensorInfo[]> e : byLayer.entrySet()) {
            for (int r = 0; r < e.getValue().length; r++) {
                if (e.getValue()[r] == null) {
                    throw new IOException("packed expert layer " + e.getKey() + ": tensor blk."
                            + e.getKey() + ".ffn_" + PackedExpertEntry.ROLE_SUFFIXES[r]
                            + "_exps.weight is missing; packing needs all three expert tensors");
                }
            }
            result.put(e.getKey(), Collections.unmodifiableList(Arrays.asList(e.getValue())));
        }
        return result;
    }

    private static int roleIndex(String role) {
        for (int r = 0; r < PackedExpertEntry.ROLE_SUFFIXES.length; r++) {
            if (PackedExpertEntry.ROLE_SUFFIXES[r].equals(role)) {
                return r;
            }
        }
        throw new IllegalArgumentException("not an expert role: " + role);
    }

    private PackedExpertView(ModelSource delegate, Map<Integer, PackedExpertEntry> entries,
                             Map<Integer, List<GGUFTensorInfo>> byLayer) {
        this.delegate = delegate;
        this.entries = entries;
        long base = 0;
        this.expertByName = new LinkedHashMap<>();
        for (List<GGUFTensorInfo> layerExperts : byLayer.values()) {
            for (GGUFTensorInfo t : layerExperts) {
                expertByName.put(t.name(), t);
            }
        }
        // the packed tensors live beyond every delegate tensor, so no virtual range overlaps one
        for (GGUFTensorInfo t : delegate.getTensors()) {
            base = Math.max(base, t.offset() + t.sizeInBytes());
        }
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        List<Packed> packed = new ArrayList<>();
        long offset = alignUp(base, PackedExpertEntry.ALIGN);
        for (GGUFTensorInfo t : delegate.getTensors()) {
            int layer = packedLayerOf(t.name());
            if (layer < 0) {
                tensors.add(t);
                continue;
            }
            if (!packed.isEmpty() && packed.get(packed.size() - 1).entry.layer() == layer) {
                continue;
            }
            PackedExpertEntry entry = entries.get(layer);
            GGUFTensorInfo p = new GGUFTensorInfo(entry.packedTensorName(),
                    new long[]{entry.stride(), entry.nExpert()}, GGMLType.I8, offset);
            tensors.add(p);
            packed.add(new Packed(p, entry, roleTensors(entry)));
            offset = alignUp(offset + p.sizeInBytes(), PackedExpertEntry.ALIGN);
        }
        this.packedByLayer = Collections.unmodifiableList(packed);
        this.viewTensors = Collections.unmodifiableList(tensors);
        this.metadataArrays = metadataArrays(entries);
    }

    /** The expert tensors of an entry's roles in storage order, as the delegate stores them. */
    private GGUFTensorInfo[] roleTensors(PackedExpertEntry entry) {
        GGUFTensorInfo[] tensors = new GGUFTensorInfo[entry.roles().size()];
        for (int r = 0; r < tensors.length; r++) {
            String name = entry.roles().get(r).standardTensorName(entry.layer());
            GGUFTensorInfo t = expertByName.get(name);
            if (t == null) {
                throw new IllegalStateException("expert tensor " + name + " of the packed view is not in the"
                        + " delegate table");
            }
            tensors[r] = t;
        }
        return tensors;
    }

    /**
     * One packed tensor of the view: its description, its entry, and the mapping from tensor bytes
     * to the delegate's expert tensors. Inside expert {@code e}'s entry, role {@code r}'s stretch
     * runs from its offset to the next role's offset (the stride for the last role) and holds, as
     * {@code pack-experts.py} writes it: the role's data of expert {@code e}; then its pad, the first
     * bytes of expert {@code e + 1}'s role data (the last expert repeats its own), at most one
     * expert's role bytes of it; then zeros up to the stretch's end (the 256-byte alignment slack).
     */
    static final class Packed {
        /** {@link #source} of a byte the pack writes as zero. */
        static final long ZERO = -1L;

        private final GGUFTensorInfo info;
        private final PackedExpertEntry entry;
        private final GGUFTensorInfo[] roleTensors;

        Packed(GGUFTensorInfo info, PackedExpertEntry entry, GGUFTensorInfo[] roleTensors) {
            this.info = info;
            this.entry = entry;
            this.roleTensors = roleTensors;
        }

        GGUFTensorInfo info() {
            return info;
        }

        long sizeBytes() {
            return info.sizeInBytes();
        }

        /** Bytes of the run beginning at tensor byte {@code x} that one source (or the zero fill) serves. */
        long runLength(long x) {
            long local = x % entry.stride();
            int r = roleAt(local);
            PackedExpertEntry.Role role = entry.roles().get(r);
            long dataEnd = role.offset() + role.entryBytes();
            if (local < dataEnd) {
                return dataEnd - local;
            }
            long padEnd = dataEnd + padFill(role);
            return local < padEnd ? padEnd - local : stretchEnd(r) - local;
        }

        /**
         * Delegate virtual offset of the source bytes of tensor byte {@code x}, or {@link #ZERO} when
         * the byte lies in an alignment slack.
         */
        long source(long x) {
            long e = x / entry.stride();
            long local = x - e * entry.stride();
            int r = roleAt(local);
            PackedExpertEntry.Role role = entry.roles().get(r);
            GGUFTensorInfo t = roleTensors[r];
            long dataEnd = role.offset() + role.entryBytes();
            if (local < dataEnd) {
                return t.offset() + e * role.entryBytes() + (local - role.offset());
            }
            if (local < dataEnd + padFill(role)) {
                long padExpert = Math.min(e + 1, entry.nExpert() - 1);
                return t.offset() + padExpert * role.entryBytes() + (local - dataEnd);
            }
            return ZERO;
        }

        /** The delegate's expert tensor that {@link #source} reads tensor byte {@code x} from. */
        GGUFTensorInfo sourceTensor(long x) {
            return roleTensors[roleAt(x % entry.stride())];
        }

        /** Pad bytes filled from the next expert: the pad, at most one expert's role bytes. */
        private static long padFill(PackedExpertEntry.Role role) {
            return Math.min(role.pad(), role.entryBytes());
        }

        /** Entry offset at which role {@code r}'s stretch ends: the next role's offset, or the stride. */
        private long stretchEnd(int r) {
            return r + 1 < entry.roles().size() ? entry.roles().get(r + 1).offset() : entry.stride();
        }

        /** Role whose stretch spans entry offset {@code local}. */
        private int roleAt(long local) {
            for (int r = 0; r < entry.roles().size(); r++) {
                if (local < stretchEnd(r)) {
                    return r;
                }
            }
            throw new IllegalArgumentException("entry offset " + local + " of " + info.name()
                    + " lies outside the stride " + entry.stride());
        }
    }

    /** The packed tensor of a stored name, or null when the name is not one. */
    Packed packed(String name) {
        for (Packed p : packedByLayer) {
            if (p.info().name().equals(name)) {
                return p;
            }
        }
        return null;
    }

    /**
     * The packed tensor whose span holds the byte range {@code [tensorOffset, +size)}, or null when
     * the range lies outside the packed tensors.
     */
    Packed packedAt(long tensorOffset, long size) {
        for (Packed p : packedByLayer) {
            long start = p.info().offset();
            if (tensorOffset >= start && size >= 0 && size <= p.sizeBytes() - (tensorOffset - start)) {
                return p;
            }
        }
        return null;
    }

    /** The layer a standard expert tensor belongs to, or -1 when the name is not one. */
    private int packedLayerOf(String name) {
        if (!expertByName.containsKey(name)) {
            return -1;
        }
        Matcher m = EXPERT_TENSOR.matcher(name);
        if (!m.matches()) {
            return -1;
        }
        int layer = Integer.parseInt(m.group(1));
        return entries.containsKey(layer) ? layer : -1;
    }

    /** The {@code avalon.experts_packed.blk.N} arrays of the entries, in increasing layer order. */
    private static Map<String, long[]> metadataArrays(Map<Integer, PackedExpertEntry> entries) {
        Map<String, long[]> arrays = new LinkedHashMap<>();
        for (PackedExpertEntry entry : entries.values()) {
            arrays.put(entry.metadataKey(), entry.metadataValues());
        }
        return Collections.unmodifiableMap(arrays);
    }

    private static long alignUp(long v, long alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    /** The packed-expert entries by layer. */
    public Map<Integer, PackedExpertEntry> entries() {
        return entries;
    }

    @Override
    public ModelFormat format() {
        return delegate.format();
    }

    @Override
    public String getFilePath() {
        return delegate.getFilePath();
    }

    /** The delegate's metadata plus the {@code avalon.experts_packed.blk.N} array of every packed layer. */
    @Override
    public Map<String, Object> getMetadata() {
        Map<String, Object> m = new LinkedHashMap<>(delegate.getMetadata());
        for (Map.Entry<String, long[]> e : metadataArrays.entrySet()) {
            List<Object> values = new ArrayList<>(e.getValue().length);
            for (long v : e.getValue()) {
                values.add(v);
            }
            m.put(e.getKey(), values);
        }
        return Collections.unmodifiableMap(m);
    }

    @Override
    public GGUFValueType getValueType(String key) {
        if (metadataArrays.containsKey(key)) {
            return GGUFValueType.ARRAY;
        }
        return delegate.getValueType(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        if (metadataArrays.containsKey(key)) {
            return GGUFValueType.UINT64;
        }
        return delegate.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        if (metadataArrays.containsKey(key)) {
            return null;
        }
        return delegate.getString(key);
    }

    /** The delegate's tensor table with every packed layer's experts replaced by its packed tensor. */
    @Override
    public List<GGUFTensorInfo> getTensors() {
        return viewTensors;
    }

    @Override
    public long getTensorCount() {
        return viewTensors.size();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        Objects.requireNonNull(use, "use");
        return new PackedExpertStore(delegate.createMappedStore(use), this);
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        Objects.requireNonNull(mode, "mode");
        return new PackedExpertRegions(delegate.openRegions(mode), this);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return delegate + " with " + entries.size() + " layers packed";
    }
}
