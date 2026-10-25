package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * What every hook of a Hugging Face checkpoint sees: the fixed configuration, the load options, the open checkpoint
 * and its current tensor index, the tokenizer files, the resolved entry and architecture, the block count and its
 * tensor name map, and the load's shared resources. The assembler derives a new context for each pipeline step with
 * the withers. Immutable; the lazily opened draft target and the per-load hook state ({@link #state}) are shared by
 * every context derived from the same load.
 */
public final class HfLoadContext {

    private static final Logger LOG = LogManager.getLogger(HfLoadContext.class);

    /** The most digits a block number may have in {@link #bid}, so that it fits an {@code int}. */
    private static final int MAX_BID_DIGITS = 9;

    private final HfConfig config;
    private final ModelSourceOptions options;
    private final CheckpointContainer checkpoint;
    private final HfTokenizerFiles tokenizerFiles;
    private final HfTensorIndex index;
    private final HfArchEntry entry;
    private final Arch arch;
    private final int blockCount;
    private final OptionalInt vocabTokenCount;
    private final Map<String, Boolean> precA4;
    private final HfResources resources;
    private final Target target;
    private final ConcurrentHashMap<String, Object> state;

    private HfLoadContext(Builder b) {
        this.config = b.config;
        this.options = b.options;
        this.checkpoint = b.checkpoint;
        this.tokenizerFiles = b.tokenizerFiles;
        this.index = b.index;
        this.entry = b.entry;
        this.arch = b.arch;
        this.blockCount = b.blockCount;
        this.vocabTokenCount = b.vocabTokenCount;
        this.precA4 = b.precA4;
        this.resources = b.resources;
        this.target = b.target;
        this.state = b.state;
    }

    /**
     * A builder for the context of a checkpoint whose fixed configuration is {@code config}, resolved to
     * {@code entry} and {@code arch}; the block count is {@code entry.blockCount(config)}.
     *
     * @throws IllegalArgumentException for {@link Arch#UNKNOWN} or a rejected entry
     */
    public static Builder builder(HfConfig config, HfArchEntry entry, Arch arch) {
        return new Builder(config, entry, arch);
    }

    // ------------------------------------------------------------------ accessors

    /** The configuration after the entry's fixup. */
    public HfConfig config() { return config; }

    public ModelSourceOptions options() { return options; }

    /**
     * The open checkpoint.
     *
     * @throws IllegalStateException when the context was built without one (configuration-only contexts)
     */
    public CheckpointContainer checkpoint() {
        if (checkpoint == null) {
            throw new IllegalStateException("this context of " + config.architecture() + " has no open checkpoint");
        }
        return checkpoint;
    }

    public HfTokenizerFiles tokenizerFiles() { return tokenizerFiles; }

    /** The tensor index of the current pipeline step. */
    public HfTensorIndex index() { return index; }

    public HfArchEntry entry() { return entry; }

    public Arch arch() { return arch; }

    public int blockCount() { return blockCount; }

    /** The tensor name map of {@link #arch()} with {@link #blockCount()} blocks. */
    public HfTensorNameMap names() {
        return HfTensorNameMap.forArch(arch, blockCount);
    }

    /** The token count of the converted vocabulary, set once the vocabulary step ran. */
    public OptionalInt vocabTokenCount() { return vocabTokenCount; }

    /**
     * GGUF tensor name to its {@code prec_a4} flag, for the layers the quantization rewrite marks (ModelOpt
     * W4A16_NVFP4 layers are {@code false}, LC/conversion/base.py:665-675); empty when none (unmodifiable).
     */
    public Map<String, Boolean> precA4() { return precA4; }

    /** The resources the load keeps open until the model source closes. */
    public HfResources resources() { return resources; }

    /**
     * The model a draft checkpoint serves, opened from the target path on first use and kept in
     * {@link #resources()}.
     *
     * @throws GgufFormatException when the load has no target path
     * @throws IOException         when the target cannot be opened
     */
    public ModelSource target() throws IOException {
        return target.get(this);
    }

    /**
     * The path of the model a draft checkpoint serves, as given to the load, for hooks that read the target's own
     * files (the converter's {@code --target-model-dir}) instead of its GGUF view; empty when the load has none.
     */
    public Optional<Path> targetPath() {
        return Optional.ofNullable(target.path);
    }

    /**
     * The block number of the Hugging Face tensor {@code hfName}: its first dot-separated part made only of digits,
     * as {@code prepare_tensors} computes {@code bid} (LC/conversion/base.py:1232-1236), or -1 when it has none.
     *
     * @throws GgufFormatException when that part does not fit an {@code int}
     */
    public int bid(String hfName) {
        return bidOf(hfName);
    }

    /**
     * {@link #bid}, for code that has no context at hand.
     *
     * @throws GgufFormatException when the block part does not fit an {@code int}
     */
    public static int bidOf(String hfName) {
        Objects.requireNonNull(hfName, "hfName");
        for (String part : hfName.split("\\.", -1)) {
            if (!part.isEmpty() && isAsciiDigits(part)) {
                if (part.length() > MAX_BID_DIGITS) {
                    throw new GgufFormatException("tensor '" + hfName + "': block number " + part + " is too large");
                }
                return Integer.parseInt(part);
            }
        }
        return -1;
    }

    /**
     * The per-load state stored under {@code key}, created by {@code init} on first use: the counterpart of the
     * instance fields converter classes keep across {@code modify_tensors} calls (expert buffers, for example). Shared
     * by every context derived from the same load; thread-safe.
     *
     * @throws IllegalStateException when {@code key} holds a value of another type
     */
    public <T> T state(String key, Class<T> type, Supplier<? extends T> init) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(init, "init");
        Object v = state.computeIfAbsent(key, k -> Objects.requireNonNull(init.get(), "initial state of " + k));
        if (!type.isInstance(v)) {
            throw new IllegalStateException("load state '" + key + "' is a " + v.getClass().getName() + ", not a "
                    + type.getName());
        }
        return type.cast(v);
    }

    /** Logs {@code message} at INFO, prefixed with the Hugging Face architecture. */
    public void log(String message) {
        LOG.info("{}: {}", config.architecture(), message);
    }

    // ------------------------------------------------------------------ withers

    /** A copy with {@code index} as the current tensor index. */
    public HfLoadContext withIndex(HfTensorIndex index) {
        Builder b = new Builder(this);
        b.index = Objects.requireNonNull(index, "index");
        return new HfLoadContext(b);
    }

    /**
     * A copy with {@code blockCount} blocks (the converter's {@code --no-mtp} drops trailing MTP blocks).
     *
     * @throws IllegalArgumentException when the count is outside {@code [0, HfConfig.MAX_LAYERS]}
     */
    public HfLoadContext withBlockCount(int blockCount) {
        Builder b = new Builder(this);
        b.blockCount = checkBlockCount(blockCount);
        return new HfLoadContext(b);
    }

    /**
     * A copy recording the vocabulary's token count.
     *
     * @throws IllegalArgumentException when the count is negative
     */
    public HfLoadContext withVocabTokenCount(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("vocabulary token count " + count + " is negative");
        }
        Builder b = new Builder(this);
        b.vocabTokenCount = OptionalInt.of(count);
        return new HfLoadContext(b);
    }

    /** A copy with the {@code prec_a4} flags of {@code precA4} (GGUF tensor name to flag). */
    public HfLoadContext withPrecA4(Map<String, Boolean> precA4) {
        Builder b = new Builder(this);
        b.precA4 = copyPrecA4(precA4);
        return new HfLoadContext(b);
    }

    @Override
    public String toString() {
        return "HfLoadContext{" + config.architecture() + " -> " + arch.ggufName() + ", " + blockCount + " blocks, "
                + index + "}";
    }

    private static boolean isAsciiDigits(String s) {
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    private static int checkBlockCount(int n) {
        if (n < 0 || n > HfConfig.MAX_LAYERS) {
            throw new IllegalArgumentException("block count " + n + " is outside [0, " + HfConfig.MAX_LAYERS + "]");
        }
        return n;
    }

    private static Map<String, Boolean> copyPrecA4(Map<String, Boolean> precA4) {
        Objects.requireNonNull(precA4, "precA4");
        Map<String, Boolean> out = new LinkedHashMap<>();
        for (Map.Entry<String, Boolean> e : precA4.entrySet()) {
            out.put(Objects.requireNonNull(e.getKey(), "tensor name"),
                    Objects.requireNonNull(e.getValue(), () -> "prec_a4 of " + e.getKey()));
        }
        return Collections.unmodifiableMap(out);
    }

    /** The draft target, opened once and shared by every context of one load. */
    private static final class Target {
        private final Path path;
        private ModelSource source;

        Target(Path path) {
            this.path = path;
        }

        synchronized ModelSource get(HfLoadContext ctx) throws IOException {
            if (source == null) {
                if (path == null) {
                    throw new GgufFormatException(ctx.config.architecture()
                            + " is a draft model; open it as a draft of its target model");
                }
                source = ctx.resources.register(ModelSources.open(path.toString(), ctx.options));
            }
            return source;
        }
    }

    /** Builds the first context of a load. Mutable. */
    public static final class Builder {
        private HfConfig config;
        private ModelSourceOptions options = ModelSourceOptions.defaults();
        private CheckpointContainer checkpoint;
        private HfTokenizerFiles tokenizerFiles = HfTokenizerFiles.empty();
        private HfTensorIndex index = HfTensorIndex.of(Collections.emptyMap());
        private HfArchEntry entry;
        private Arch arch;
        private int blockCount;
        private OptionalInt vocabTokenCount = OptionalInt.empty();
        private Map<String, Boolean> precA4 = Collections.emptyMap();
        private HfResources resources = new HfResources();
        private Target target = new Target(null);
        private ConcurrentHashMap<String, Object> state = new ConcurrentHashMap<>();

        private Builder(HfConfig config, HfArchEntry entry, Arch arch) {
            this.config = Objects.requireNonNull(config, "config");
            this.entry = Objects.requireNonNull(entry, "entry");
            this.arch = Objects.requireNonNull(arch, "arch");
            if (arch == Arch.UNKNOWN) {
                throw new IllegalArgumentException("a load context needs a known architecture");
            }
            if (entry.isRejected()) {
                throw new IllegalArgumentException("entry " + entry + " is rejected");
            }
            this.blockCount = entry.blockCount(config);
        }

        private Builder(HfLoadContext c) {
            this.config = c.config;
            this.options = c.options;
            this.checkpoint = c.checkpoint;
            this.tokenizerFiles = c.tokenizerFiles;
            this.index = c.index;
            this.entry = c.entry;
            this.arch = c.arch;
            this.blockCount = c.blockCount;
            this.vocabTokenCount = c.vocabTokenCount;
            this.precA4 = c.precA4;
            this.resources = c.resources;
            this.target = c.target;
            this.state = c.state;
        }

        public Builder options(ModelSourceOptions v) {
            this.options = Objects.requireNonNull(v, "options");
            return this;
        }

        public Builder checkpoint(CheckpointContainer v) {
            this.checkpoint = Objects.requireNonNull(v, "checkpoint");
            return this;
        }

        public Builder tokenizerFiles(HfTokenizerFiles v) {
            this.tokenizerFiles = Objects.requireNonNull(v, "tokenizerFiles");
            return this;
        }

        public Builder index(HfTensorIndex v) { this.index = Objects.requireNonNull(v, "index"); return this; }

        public Builder resources(HfResources v) {
            this.resources = Objects.requireNonNull(v, "resources");
            return this;
        }

        /** The model a draft checkpoint serves; null when the checkpoint is not opened as a draft. */
        public Builder targetPath(Path v) { this.target = new Target(v); return this; }

        public HfLoadContext build() {
            return new HfLoadContext(this);
        }
    }
}
