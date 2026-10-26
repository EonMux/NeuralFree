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
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link ModelSource} whose metadata is the delegate's with the {@code --override-kv}
 * ({@link MetadataOverrides}) entries of {@link ModelSourceOptions#metadataOverrides()} and of the rope options
 * ({@code ModelSourceOptions#ropeOverrides()}) applied on top; every other call, the tensor bytes included, is the
 * delegate's. The view is applied by {@code ModelSources} after it builds the source and before anything reads its
 * metadata (hparams, tensor catalogs, tokenizers), and is the same for GGUF and Hugging Face sources: the overrides
 * are post-hoc on the format-neutral GGUF view, unlike the pre-conversion {@code neuralfree.hf.overrides} JSON.
 *
 * <p>Later entries of the list win over earlier ones and over the delegate. Overridden values keep the decoded
 * representations of their {@linkplain MetadataOverrides.Override#type() GGUF type}, so they are indistinguishable
 * from metadata that came from the file. Immutable once built; the delegate stays owned by the caller of
 * {@code ModelSources}, which closes it.
 */
public final class OverriddenSource implements ModelSource {

    private final ModelSource delegate;
    private final Map<String, Object> metadata;
    private final Map<String, MetadataOverrides.Override> overrides;
    private final String overridesText;

    /**
     * @param overrides the overrides in list order; later entries win. Empty lists return the plain view, which
     *                  callers may shortcut.
     */
    public OverriddenSource(ModelSource delegate, List<MetadataOverrides.Override> overrides) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        Objects.requireNonNull(overrides, "overrides");
        Map<String, MetadataOverrides.Override> byKey = new LinkedHashMap<>();
        for (MetadataOverrides.Override o : overrides) {
            Objects.requireNonNull(o, "override");
            if (o.type() == GGUFValueType.ARRAY) {
                throw new IllegalArgumentException("override of key '" + o.key() + "': metadata overrides carry"
                        + " str, bool, int or float values, not ARRAY");
            }
            byKey.put(o.key(), o);
        }
        this.overrides = Collections.unmodifiableMap(byKey);
        this.overridesText = MetadataOverrides.toString(overrides);
        Map<String, Object> merged = new LinkedHashMap<>();
        merged.putAll(delegate.getMetadata());
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

    /** The GGUF type of the override when {@code key} is overridden, else the delegate's. */
    @Override
    public GGUFValueType getValueType(String key) {
        MetadataOverrides.Override o = key == null ? null : overrides.get(key);
        return o != null ? o.type() : delegate.getValueType(key);
    }

    /** Null when {@code key} is overridden (overrides are never arrays), else the delegate's. */
    @Override
    public GGUFValueType getArrayElementType(String key) {
        if (key != null && overrides.containsKey(key)) {
            return null;
        }
        return delegate.getArrayElementType(key);
    }

    /** The overridden string when {@code key} is overridden as STRING, else the delegate's. */
    @Override
    public String getString(String key) {
        MetadataOverrides.Override o = key == null ? null : overrides.get(key);
        if (o != null) {
            return o.type() == GGUFValueType.STRING ? (String) o.value() : null;
        }
        return delegate.getString(key);
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
