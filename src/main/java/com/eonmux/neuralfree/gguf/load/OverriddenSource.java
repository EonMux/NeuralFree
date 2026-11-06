package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A {@link ModelSource} whose metadata is the delegate's with the {@code --override-kv}
 * ({@link MetadataOverrides}) entries of {@link ModelSourceOptions#metadataOverrides()} and of the rope options
 * ({@code ModelSourceOptions#ropeOverrides()}) applied on top; every other call, the tensor bytes included, is the
 * delegate's. The view is applied by {@code ModelSources} after it builds the source and before anything reads its
 * metadata (hparams, tensor catalogs, tokenizers), and is the same for GGUF and Hugging Face sources: the overrides
 * are post-hoc on the format-neutral GGUF view, unlike the pre-conversion {@code neuralfree.hf.overrides} JSON.
 *
 * <p>Later entries of the list win over earlier ones and over the delegate. A scalar override keeps the decoded
 * representations of its {@linkplain MetadataOverrides.Override#type() GGUF type}; an ARRAY override (the
 * {@code --metadata} file of the converters, {@link MetadataOverrides#fromJsonObject}) carries an element type,
 * so the writer stores its list as a GGUF array. Overridden values are indistinguishable from metadata that came
 * from the file. Removed keys (the indexed {@code general.base_model.*} keys a {@code --metadata} list replaces) are
 * absent from the view unless an override sets them again. Immutable once built; the delegate stays owned by the caller of
 * {@code ModelSources}, which closes it.
 */
public final class OverriddenSource implements ModelSource {

    private final ModelSource delegate;
    private final Map<String, Object> metadata;
    private final Map<String, MetadataOverrides.Override> overrides;
    private final Set<String> removed;
    private final String overridesText;

    /**
     * @param overrides the overrides in list order; later entries win. Empty lists return the plain view, which
     *                  callers may shortcut. A scalar override replaces a scalar key; an ARRAY override (of
     *                  {@link MetadataOverrides.Override#elementType()}) replaces or introduces an array key.
     */
    public OverriddenSource(ModelSource delegate, List<MetadataOverrides.Override> overrides) {
        this(delegate, overrides, Collections.emptySet());
    }

    /**
     * @param overrides   the overrides in list order; later entries win
     * @param removedKeys delegate keys the view drops; an override of the same key sets it again
     */
    public OverriddenSource(ModelSource delegate, List<MetadataOverrides.Override> overrides,
                            Set<String> removedKeys) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        Objects.requireNonNull(overrides, "overrides");
        Objects.requireNonNull(removedKeys, "removedKeys");
        Map<String, MetadataOverrides.Override> byKey = new LinkedHashMap<>();
        for (MetadataOverrides.Override o : overrides) {
            Objects.requireNonNull(o, "override");
            byKey.put(o.key(), o);
        }
        this.overrides = Collections.unmodifiableMap(byKey);
        Set<String> gone = new LinkedHashSet<>(removedKeys);
        gone.removeAll(byKey.keySet());
        this.removed = Collections.unmodifiableSet(gone);
        this.overridesText = MetadataOverrides.toString(overrides)
                + (gone.isEmpty() ? "" : " removing " + String.join(",", gone));
        Map<String, Object> merged = new LinkedHashMap<>();
        merged.putAll(delegate.getMetadata());
        merged.keySet().removeAll(gone);
        for (MetadataOverrides.Override o : byKey.values()) {
            merged.put(o.key(), o.value());
        }
        this.metadata = Collections.unmodifiableMap(merged);
    }

    /** Whether the view carries at least one override. */
    public boolean hasOverrides() {
        return !overrides.isEmpty();
    }

    /** The overrides by key, in list order (the last entry of a key wins). */
    public Map<String, MetadataOverrides.Override> overrides() {
        return overrides;
    }

    @Override
    public ModelFormat format() {
        return delegate.format();
    }

    @Override
    public String getFilePath() {
        return delegate.getFilePath();
    }

    /** The delegate's metadata with every override applied, in the GGUF value model. */
    @Override
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /** The GGUF type of the override when {@code key} is overridden, null when it is removed, else the delegate's. */
    @Override
    public GGUFValueType getValueType(String key) {
        MetadataOverrides.Override o = key == null ? null : overrides.get(key);
        if (o != null) {
            return o.type();
        }
        return removed.contains(key) ? null : delegate.getValueType(key);
    }

    /** The element type of an overridden array, null for a removed key, else the delegate's. */
    @Override
    public GGUFValueType getArrayElementType(String key) {
        MetadataOverrides.Override o = key == null ? null : overrides.get(key);
        if (o != null) {
            return o.elementType();
        }
        return removed.contains(key) ? null : delegate.getArrayElementType(key);
    }

    /** The overridden string when {@code key} is overridden as STRING, null for a removed key, else the delegate's. */
    @Override
    public String getString(String key) {
        MetadataOverrides.Override o = key == null ? null : overrides.get(key);
        if (o != null) {
            return o.type() == GGUFValueType.STRING ? (String) o.value() : null;
        }
        return removed.contains(key) ? null : delegate.getString(key);
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return delegate.getTensors();
    }

    @Override
    public long getTensorCount() {
        return delegate.getTensorCount();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        return delegate.createMappedStore(use);
    }

    @Override
    public WeightStore createMappedStore() throws IOException {
        return delegate.createMappedStore();
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        return delegate.openRegions(mode);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    /** The parsed overrides, for messages. */
    @Override
    public String toString() {
        return delegate + " with metadata overrides " + overridesText;
    }
}
