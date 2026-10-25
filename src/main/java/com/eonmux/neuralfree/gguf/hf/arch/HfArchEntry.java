package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * How one llama.cpp converter class is reproduced at load time: the Hugging Face {@code architectures[0]} names it
 * registers and the hooks that port its methods, run in the order of the {@code HfSourceAssembler} pipeline. Built
 * with {@link #of} and the withers; every wither returns a copy. Immutable.
 *
 * <p>A rejected entry ({@link #rejected}) names checkpoints the converter lists but NeuralFree refuses with a stated
 * reason; resolving it throws {@link GgufFormatException} with that reason.</p>
 */
public final class HfArchEntry {

    /** Which converter map a name belongs to: {@code TEXT_MODEL_MAP} or {@code MMPROJ_MODEL_MAP}. */
    public enum Kind {
        /** A language model (llama.cpp's {@code TextModel} subclasses). */
        TEXT,
        /** A vision or audio projector (llama.cpp's {@code MmprojModel} subclasses). */
        MMPROJ
    }

    private final List<String> hfNames;
    private final Kind kind;
    private final HfArchResolver arch;
    private final HfMetadataHook meta;
    private final HfTensorHook tensors;
    private final HfVocabHook vocab;
    private final HfExtraTensors extra;
    private final HfIndexHook index;
    private final Function<HfLoadContext, HfNameFilter> nameFilter;
    private final ToIntFunction<HfConfig> blockCount;
    private final ToIntFunction<HfLoadContext> indexedBlockCount;
    private final HfConfigFixup configFixup;
    private final HfForceType forceType;
    private final boolean undoPermute;
    private final String rejection;

    private HfArchEntry(Builder b) {
        this.hfNames = b.hfNames;
        this.kind = b.kind;
        this.arch = b.arch;
        this.meta = b.meta;
        this.tensors = b.tensors;
        this.vocab = b.vocab;
        this.extra = b.extra;
        this.index = b.index;
        this.nameFilter = b.nameFilter;
        this.blockCount = b.blockCount;
        this.indexedBlockCount = b.indexedBlockCount;
        this.configFixup = b.configFixup;
        this.forceType = b.forceType;
        this.undoPermute = b.undoPermute;
        this.rejection = b.rejection;
    }

    /**
     * An entry for {@code hfNames} with the given hooks, no extra tensors, no index hook, no class name filter, the
     * configuration's block count, no configuration fixup, no forced types, {@code undo_permute} on, kind {@link Kind#TEXT}.
     *
     * @throws IllegalArgumentException when {@code hfNames} is empty or holds an empty or repeated name
     */
    public static HfArchEntry of(List<String> hfNames, HfArchResolver arch, HfMetadataHook meta,
                                 HfTensorHook tensors, HfVocabHook vocab) {
        Builder b = new Builder(names(hfNames));
        b.arch = Objects.requireNonNull(arch, "arch");
        b.meta = Objects.requireNonNull(meta, "meta");
        b.tensors = Objects.requireNonNull(tensors, "tensors");
        b.vocab = Objects.requireNonNull(vocab, "vocab");
        return new HfArchEntry(b);
    }

    /**
     * An entry that refuses {@code hfNames} with {@code message}.
     *
     * @throws IllegalArgumentException when {@code hfNames} is empty or holds an empty or repeated name, or the
     *                                  message is empty
     */
    public static HfArchEntry rejected(List<String> hfNames, String message) {
        Objects.requireNonNull(message, "message");
        if (message.trim().isEmpty()) {
            throw new IllegalArgumentException("a rejected entry needs a reason");
        }
        Builder b = new Builder(names(hfNames));
        b.rejection = message;
        b.arch = (cfg, idx) -> {
            throw new GgufFormatException(message);
        };
        b.meta = (ctx, kv, plan) -> {
            throw new GgufFormatException(message);
        };
        b.tensors = (ctx, name, src, out) -> {
            throw new GgufFormatException(message);
        };
        b.vocab = (ctx, pre) -> {
            throw new GgufFormatException(message);
        };
        return new HfArchEntry(b);
    }

    private static List<String> names(List<String> hfNames) {
        Objects.requireNonNull(hfNames, "hfNames");
        if (hfNames.isEmpty()) {
            throw new IllegalArgumentException("an entry needs at least one Hugging Face architecture name");
        }
        List<String> out = new ArrayList<>(hfNames.size());
        for (String n : hfNames) {
            if (n == null || n.isEmpty()) {
                throw new IllegalArgumentException("empty Hugging Face architecture name in " + hfNames);
            }
            if (out.contains(n)) {
                throw new IllegalArgumentException("Hugging Face architecture name '" + n + "' is listed twice");
            }
            out.add(n);
        }
        return Collections.unmodifiableList(out);
    }

    // ------------------------------------------------------------------ withers

    /** A copy with {@code generate_extra_tensors}. */
    public HfArchEntry withExtra(HfExtraTensors extra) {
        Builder b = new Builder(this);
        b.extra = Objects.requireNonNull(extra, "extra");
        return new HfArchEntry(b);
    }

    /** A copy with the class's index hook. */
    public HfArchEntry withIndex(HfIndexHook index) {
        Builder b = new Builder(this);
        b.index = Objects.requireNonNull(index, "index");
        return new HfArchEntry(b);
    }

    /** A copy with the class's {@code filter_tensors} name filter, the same for every load. */
    public HfArchEntry withNameFilter(HfNameFilter nameFilter) {
        Objects.requireNonNull(nameFilter, "nameFilter");
        return withLoadNameFilter(ctx -> nameFilter);
    }

    /**
     * A copy whose {@code filter_tensors} name filter is built per load from the load context, for a converter class
     * whose filter reads instance state (the original block count, {@code no_mtp}). The factory must return a
     * non-null filter.
     */
    public HfArchEntry withLoadNameFilter(Function<HfLoadContext, HfNameFilter> nameFilter) {
        Builder b = new Builder(this);
        b.nameFilter = Objects.requireNonNull(nameFilter, "nameFilter");
        return new HfArchEntry(b);
    }

    /**
     * A copy whose block count comes from {@code blockCount} instead of {@link HfConfig#blockCount()}. An indexed
     * block count set by {@link #withIndexedBlockCount} still takes precedence once the index is known.
     */
    public HfArchEntry withBlockCount(ToIntFunction<HfConfig> blockCount) {
        Builder b = new Builder(this);
        b.blockCount = Objects.requireNonNull(blockCount, "blockCount");
        return new HfArchEntry(b);
    }

    /**
     * A copy whose block count, once the checkpoint index is known, comes from {@code blockCount} applied to the load
     * context (for a count that depends on the checkpoint's tensor names, such as appended MTP layers). Before the
     * index step the configuration rule of {@link #withBlockCount} applies.
     */
    public HfArchEntry withIndexedBlockCount(ToIntFunction<HfLoadContext> blockCount) {
        Builder b = new Builder(this);
        b.indexedBlockCount = Objects.requireNonNull(blockCount, "blockCount");
        return new HfArchEntry(b);
    }

    /** A copy with the class's {@code __init__} hyperparameter changes. */
    public HfArchEntry withConfigFixup(HfConfigFixup configFixup) {
        Builder b = new Builder(this);
        b.configFixup = Objects.requireNonNull(configFixup, "configFixup");
        return new HfArchEntry(b);
    }

    /** A copy with the class's {@code tensor_force_quant}. */
    public HfArchEntry withForceType(HfForceType forceType) {
        Builder b = new Builder(this);
        b.forceType = Objects.requireNonNull(forceType, "forceType");
        return new HfArchEntry(b);
    }

    /** A copy with the class's {@code undo_permute} flag (whether Q/K weights are un-permuted). */
    public HfArchEntry withUndoPermute(boolean undoPermute) {
        Builder b = new Builder(this);
        b.undoPermute = undoPermute;
        return new HfArchEntry(b);
    }

    /** A copy registered under {@code kind}. */
    public HfArchEntry withKind(Kind kind) {
        Builder b = new Builder(this);
        b.kind = Objects.requireNonNull(kind, "kind");
        return new HfArchEntry(b);
    }

    // ------------------------------------------------------------------ getters

    /** The Hugging Face architecture names (unmodifiable, at least one). */
    public List<String> hfNames() { return hfNames; }

    public Kind kind() { return kind; }

    public HfArchResolver arch() { return arch; }

    public HfMetadataHook meta() { return meta; }

    public HfTensorHook tensors() { return tensors; }

    public HfVocabHook vocab() { return vocab; }

    public HfExtraTensors extra() { return extra; }

    public HfIndexHook index() { return index; }

    /**
     * The class's name filter bound to the load {@code ctx}.
     *
     * @throws IllegalStateException when a load-scoped filter factory returns null
     */
    public HfNameFilter nameFilter(HfLoadContext ctx) {
        HfNameFilter f = nameFilter.apply(Objects.requireNonNull(ctx, "ctx"));
        if (f == null) {
            throw new IllegalStateException(this + ": the load name filter factory returned null");
        }
        return f;
    }

    public HfConfigFixup configFixup() { return configFixup; }

    public HfForceType forceType() { return forceType; }

    public boolean undoPermute() { return undoPermute; }

    /**
     * The block count of {@code cfg}.
     *
     * @throws GgufFormatException when the configuration has none or it is outside {@code [0, HfConfig.MAX_LAYERS]}
     */
    public int blockCount(HfConfig cfg) {
        int n = blockCount.applyAsInt(Objects.requireNonNull(cfg, "cfg"));
        if (n < 0 || n > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": block count " + n + " is outside [0, "
                    + HfConfig.MAX_LAYERS + "]");
        }
        return n;
    }

    /**
     * The block count of the load {@code ctx}: the indexed rule of {@link #withIndexedBlockCount} when one is set,
     * else {@link #blockCount(HfConfig)} of {@code ctx.config()}.
     *
     * @throws GgufFormatException when the count is outside {@code [0, HfConfig.MAX_LAYERS]}
     */
    public int blockCount(HfLoadContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        if (indexedBlockCount == null) {
            return blockCount(ctx.config());
        }
        int n = indexedBlockCount.applyAsInt(ctx);
        if (n < 0 || n > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(ctx.config().source() + ": block count " + n + " is outside [0, "
                    + HfConfig.MAX_LAYERS + "]");
        }
        return n;
    }

    /** True for an entry made by {@link #rejected}. */
    public boolean isRejected() { return rejection != null; }

    /** The reason of a rejected entry, or null. */
    public String rejection() { return rejection; }

    @Override
    public String toString() {
        return "HfArchEntry{" + kind + " " + hfNames + (rejection != null ? ", rejected: " + rejection : "") + "}";
    }

    private static final class Builder {
        final List<String> hfNames;
        Kind kind = Kind.TEXT;
        HfArchResolver arch;
        HfMetadataHook meta;
        HfTensorHook tensors;
        HfVocabHook vocab;
        HfExtraTensors extra = HfExtraTensors.NONE;
        HfIndexHook index = HfIndexHook.NONE;
        Function<HfLoadContext, HfNameFilter> nameFilter = ctx -> HfNameFilter.INHERITED;
        ToIntFunction<HfConfig> blockCount = HfConfig::blockCount;
        ToIntFunction<HfLoadContext> indexedBlockCount;
        HfConfigFixup configFixup = HfConfigFixup.NONE;
        HfForceType forceType = HfForceType.NONE;
        boolean undoPermute = true;
        String rejection;

        Builder(List<String> hfNames) {
            this.hfNames = hfNames;
        }

        Builder(HfArchEntry e) {
            this.hfNames = e.hfNames;
            this.kind = e.kind;
            this.arch = e.arch;
            this.meta = e.meta;
            this.tensors = e.tensors;
            this.vocab = e.vocab;
            this.extra = e.extra;
            this.index = e.index;
            this.nameFilter = e.nameFilter;
            this.blockCount = e.blockCount;
            this.indexedBlockCount = e.indexedBlockCount;
            this.configFixup = e.configFixup;
            this.forceType = e.forceType;
            this.undoPermute = e.undoPermute;
            this.rejection = e.rejection;
        }
    }
}
