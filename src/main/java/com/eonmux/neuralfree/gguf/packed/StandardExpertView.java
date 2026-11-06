package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link ModelSource} that presents an experts-packed model in the standard layout: the
 * {@code blk.N.ffn_exps_packed} tensors and {@code avalon.experts_packed.*} keys of the delegate
 * are hidden, and every packed layer's three standard tensors
 * {@code blk.N.ffn_{up,gate,down}_exps.weight} are synthesized from its
 * {@link PackedExpertsIndex} entry, at virtual offsets beyond the delegate's data. Tensor bytes are
 * served by {@link StandardExpertRegions} and {@link StandardExpertStore}, which map (or read)
 * every expert's role data from its place in the packed tensor — the views of Avalon's loader
 * (llama-model-loader.cpp:718-767) — so nothing is repacked and nothing but the request's own
 * ranges is touched. Every other tensor, key and byte is the delegate's, so catalogs, hparams,
 * tensor requirements, weight stores and tokenizers see an unpacked MoE model. The delegate stays
 * owned by the caller of {@code ModelSources}, which closes it; the regions and stores this view
 * opens are owned by their callers. Immutable.
 */
public final class StandardExpertView implements ModelSource {

    /** One synthesized standard expert tensor: its description, entry, role and packed tensor. */
    static final class Synth {
        final GGUFTensorInfo info;
        final PackedExpertEntry entry;
        final PackedExpertEntry.Role role;
        final GGUFTensorInfo packedTensor;

        Synth(GGUFTensorInfo info, PackedExpertEntry entry, PackedExpertEntry.Role role,
              GGUFTensorInfo packedTensor) {
            this.info = info;
            this.entry = entry;
            this.role = role;
            this.packedTensor = packedTensor;
        }

        /** Tensor bytes of the whole tensor ({@code n_expert} entries of the role). */
        long sizeBytes() {
            return entry.nExpert() * role.entryBytes();
        }
    }

    private final ModelSource delegate;
    private final PackedExpertsIndex index;
    private final List<GGUFTensorInfo> viewTensors;
    private final Map<String, Synth> synthByName;
    private final List<Synth> synthsByOffset;
    private final long synthBase;

    private StandardExpertView(ModelSource delegate, PackedExpertsIndex index) {
        this.delegate = delegate;
        this.index = index;
        // the synthesized tensors live beyond every delegate tensor, so no virtual range overlaps one
        long base = 0;
        Map<String, PackedExpertEntry> packedByName = new LinkedHashMap<>();
        for (PackedExpertEntry entry : index.entries().values()) {
            packedByName.put(entry.packedTensorName(), entry);
        }
        for (GGUFTensorInfo t : delegate.getTensors()) {
            base = Math.max(base, t.offset() + t.sizeInBytes());
        }
        this.synthBase = base;
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        Map<String, Synth> byName = new LinkedHashMap<>();
        List<Synth> byOffset = new ArrayList<>();
        // one aligned slot per synthesized tensor, beyond every delegate tensor and disjoint, so a
        // virtual offset names exactly one tensor
        long slot = base;
        for (GGUFTensorInfo t : delegate.getTensors()) {
            PackedExpertEntry entry = packedByName.get(t.name());
            if (entry == null) {
                tensors.add(t);
                continue;
            }
            for (PackedExpertEntry.Role role : entry.roles()) {
                Synth synth = new Synth(synthInfo(entry, role, slot), entry, role,
                        index.packedTensor(entry.layer()));
                tensors.add(synth.info);
                byName.put(synth.info.name(), synth);
                byOffset.add(synth);
                slot += alignUp(synth.sizeBytes(), PackedExpertEntry.ALIGN);
            }
        }
        this.viewTensors = Collections.unmodifiableList(tensors);
        this.synthByName = Collections.unmodifiableMap(byName);
        this.synthsByOffset = Collections.unmodifiableList(byOffset);
    }

    private static long alignUp(long v, long alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    /**
     * The source in the standard layout when it is experts-packed, else the source itself.
     *
     * @throws IOException as {@link PackedExpertsIndex#of} reports a malformed packed model
     */
    public static ModelSource standardViewIfPacked(ModelSource source) throws IOException {
        Objects.requireNonNull(source, "source");
        PackedExpertsIndex index = PackedExpertsIndex.of(source);
        return index == null ? source : new StandardExpertView(source, index);
    }

    private GGUFTensorInfo synthInfo(PackedExpertEntry entry, PackedExpertEntry.Role role, long slot) {
        String name = TensorNames.layerName(entry.layer(), TensorNames.weight("ffn_" + role.name() + "_exps"));
        return new GGUFTensorInfo(name, new long[]{role.ne0(), role.ne1(), entry.nExpert()}, role.type(), slot);
    }

    /** The packed-expert layout the view was built from. */
    public PackedExpertsIndex index() {
        return index;
    }

    /** The synthesized expert tensor of a stored name, or null when the name is not one. */
    Synth synth(String storedName) {
        return synthByName.get(storedName);
    }

    /**
     * The synthesized expert tensor whose span holds the byte range {@code [virtualOffset,
     * +length)}, or null when the range lies outside the synthesized tensors.
     */
    Synth synthAt(long virtualOffset, long length) {
        for (Synth s : synthsByOffset) {
            long start = s.info.offset();
            if (virtualOffset >= start && length >= 0 && length <= s.sizeBytes() - (virtualOffset - start)) {
                return s;
            }
        }
        return null;
    }

    /** First virtual offset beyond the delegate's data; the synthesized tensors start here. */
    long synthBase() {
        return synthBase;
    }

    @Override
    public ModelFormat format() {
        return delegate.format();
    }

    @Override
    public String getFilePath() {
        return delegate.getFilePath();
    }

    /** The delegate's metadata without the {@code avalon.experts_packed.*} keys. */
    @Override
    public Map<String, Object> getMetadata() {
        Map<String, Object> m = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : delegate.getMetadata().entrySet()) {
            if (!PackedExpertsIndex.isMetadataKey(e.getKey())) {
                m.put(e.getKey(), e.getValue());
            }
        }
        return Collections.unmodifiableMap(m);
    }

    @Override
    public GGUFValueType getValueType(String key) {
        if (key != null && PackedExpertsIndex.isMetadataKey(key)) {
            return null;
        }
        return delegate.getValueType(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        if (key != null && PackedExpertsIndex.isMetadataKey(key)) {
            return null;
        }
        return delegate.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        if (key != null && PackedExpertsIndex.isMetadataKey(key)) {
            return null;
        }
        return delegate.getString(key);
    }

    /** The delegate's tensor table with the packed tensors replaced by the three standard tensors. */
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
        return new StandardExpertStore(delegate.createMappedStore(use), this);
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        Objects.requireNonNull(mode, "mode");
        return new StandardExpertRegions(delegate.openRegions(mode), this);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return delegate + " with " + index.entries().size() + " experts-packed layers presented unpacked";
    }
}
