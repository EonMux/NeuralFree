package com.eonmux.neuralfree.gguf.hf.lora;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.SourceOrigin;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import com.eonmux.neuralfree.gguf.hf.HfSourceAssembler;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfExtraTensors;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters;
import com.eonmux.neuralfree.gguf.hf.quant.HfQuantRewriter;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.HfTensorRegions;
import com.eonmux.neuralfree.gguf.hf.store.HfWeightStore;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy;
import com.eonmux.neuralfree.gguf.hf.store.ProducedSegments;
import com.eonmux.neuralfree.gguf.hf.store.SidecarDirectory;
import com.eonmux.neuralfree.gguf.hf.store.VirtualLayout;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.LoraFactorSource;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A PEFT LoRA adapter directory seen as the GGUF adapter {@code convert_lora_to_gguf.py} writes: the metadata of
 * {@code LoraModel.set_type} / {@code set_gguf_parameters} and the {@code <gguf name>.lora_a} / {@code .lora_b}
 * factor tensors of {@link com.eonmux.neuralfree.gguf.ModelSources#openAdapter}, with the base model's architecture
 * mapping the module names through the package-public {@link HfSourceAssembler} steps {@code filter} and {@code plan}
 * (LC/convert_lora_to_gguf.py:410-529, the {@code LoraModel}).
 *
 * <p>The pipeline: {@link PeftAdapterConfig#read} validates {@code adapter_config.json}; the adapter
 * weights open through the checkpoint containers; the base model's configuration resolves — a safetensors base's own
 * configuration, {@link ModelSourceOptions#loraBaseConfig()}, a local {@code base_model_name_or_path} directory, else
 * the named error of rule 4; the base architecture's {@code HfArchEntry} fixes the configuration and the block count;
 * {@link HfSourceAssembler#filter} then applies the base arch's name filters and {@link HfSourceAssembler#plan} runs
 * the base arch's tensor hooks over {@link LoraFactorSource}s, so q/k permutes, QKV splits and expert stacking act on
 * the factors as they do on the base model's weights.</p>
 *
 * <p>Tensors: names lose their {@code base_model.model.} prefix and the {@code lora_A} / {@code lora_B} /
 * {@code lora_embedding_A/B} markers (LC/convert_lora_to_gguf.py:272-279); {@code .base_layer.weight} tensors are
 * skipped (:474-476); {@code _norm} / {@code .norm} tensors of mergekit-extracted adapters pass through as plain
 * sources (:478-480, kept by :513-517); a name the base arch's filter keeps that is neither factor nor norm is the
 * converter's error (:482-486); a pair is planned as its two factors {@code <name>.lora_a} ({@code ne {in, r}}) and
 * {@code <name>.lora_b} ({@code ne {r, out}}) by
 * {@link com.eonmux.neuralfree.gguf.hf.tensor.TensorPlanBuilder#add(String, LoraFactorSource)}, which transposes
 * {@code token_embd}'s {@code A} (:521-529). Extra tensors are never generated, as {@code LoraModel}'s
 * {@code generate_extra_tensors} returns nothing (:456-459).</p>
 *
 * <p>Metadata: {@code general.type = adapter}, the base architecture, {@code adapter.type = lora} and
 * {@code adapter.lora.alpha} (:425-431); for an aLoRA adapter also
 * {@code adapter.alora.invocation_tokens} — from {@code alora_invocation_tokens}, or from {@code invocation_string}
 * tokenized with the base model's tokenizer with the first id dropped (:428-451), the HF-directory route through the
 * tokenizer metadata of the TK-3 vocab writer and the GGUF base route through {@link GGUFTokenizer}. The stores
 * reuse the HS-1 classes ({@link VirtualLayout}, {@link HfWeightStore}/{@link HfTensorRegions},
 * {@link MaterializePolicy}, {@link SidecarDirectory}) over the adapter's files. Thread-safe after {@link #open}.</p>
 */
public final class PeftAdapterSource implements ModelSource {

    private static final Logger LOG = LogManager.getLogger(PeftAdapterSource.class);

    /** The module prefix of every PEFT LoRA tensor name (LC/convert_lora_to_gguf.py:273). */
    private static final String BASE_MODEL_PREFIX = "base_model.model.";

    /** {@code general.type} of an adapter view (LC/convert_lora_to_gguf.py:426, GGUFType.ADAPTER). */
    private static final String TYPE_ADAPTER = "adapter";
    /** {@code adapter.type} of a LoRA adapter (LC/convert_lora_to_gguf.py:427). */
    private static final String ADAPTER_TYPE_LORA = "lora";

    /** The adapter weights file of the safetensors format (LC/convert_lora_to_gguf.py:358). */
    private static final String ADAPTER_MODEL_SAFETENSORS = "adapter_model.safetensors";
    /** The adapter weights file of the pickle format (LC/convert_lora_to_gguf.py:372). */
    private static final String ADAPTER_MODEL_BIN = "adapter_model.bin";

    /** The output layer whose deltas an arch may not map (LC/convert_lora_to_gguf.py:506-511). */
    private static final String LM_HEAD = "lm_head.weight";

    private final String filePath;
    private final ModelSourceOptions options;
    private final HfLoadContext context;
    private final TensorPlan plan;
    private final VirtualLayout layout;
    private final Map<String, Object> metadata;
    private final Map<String, GGUFValueType> valueTypes;
    private final Map<String, GGUFValueType> elementTypes;
    private final List<GGUFTensorInfo> tensors;
    private final CheckpointContainer checkpoint;
    private final HfResources resources;
    private final SidecarDirectory sidecars;
    private final List<AutoCloseable> readers = new ArrayList<>();
    private boolean closed;

    private PeftAdapterSource(String filePath, ModelSourceOptions options, HfLoadContext context, TensorPlan plan,
                              HfMetadata md, VirtualLayout layout, CheckpointContainer checkpoint,
                              HfResources resources) {
        this.filePath = filePath;
        this.options = options;
        this.context = context;
        this.plan = plan;
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(md.values()));
        this.valueTypes = Collections.unmodifiableMap(new LinkedHashMap<>(md.types()));
        this.elementTypes = Collections.unmodifiableMap(new LinkedHashMap<>(md.elementTypes()));
        this.tensors = layout.tensors();
        this.checkpoint = checkpoint;
        this.resources = resources;
        this.layout = layout;
        this.sidecars = new SidecarDirectory(options.cacheDir());
    }

    /**
     * Opens the PEFT adapter directory {@code dir} ({@code adapter_config.json} with the adapter weights) for the
     * base model at {@code base}.
     *
     * @throws FileNotFoundException          when {@code dir} is no PEFT adapter directory
     * @throws IOException                    when the adapter, its base configuration or the invocation tokenizer
     *                                        cannot be read
     * @throws GgufFormatException            for an adapter the conversion refuses, or for rule 4 of the base
     *                                        configuration
     * @throws UnsupportedArchException when the base architecture is registered as not supported
     */
    public static PeftAdapterSource open(Path dir, SourceOrigin base, ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(o, "o");
        if (!Files.isDirectory(dir)) {
            throw new FileNotFoundException("PEFT adapter not found: '" + dir + "' is not a directory");
        }
        if (!Files.isReadable(dir)) {
            throw new FileNotFoundException("PEFT adapter not readable: '" + dir + "'");
        }
        PeftAdapterConfig config = PeftAdapterConfig.read(dir);
        CheckpointContainer ckpt = adapterCheckpoint(dir);
        HfConfig baseCfg = baseConfig(dir, base, o, config);
        LoraTensors lora = LoraTensors.of(ckpt);
        lora.requirePaired(dir);
        HfArchEntry.Kind kind = HfArchEntry.Kind.TEXT;
        HfArchEntry entry = HfArchRegistry.resolve(baseCfg, kind);
        HfResources resources = new HfResources();
        try {
            HfTokenizerFiles tokenizer = HfTokenizerFiles.locate(baseCfg.dir())
                    .withTokenizerDir(o.tokenizerDir());
            HfTensorIndex moduleIndex = lora.moduleIndex(dir);
            Arch arch = entry.arch().resolve(baseCfg, moduleIndex);
            requireSupported(baseCfg, arch);
            HfConfig fixed = entry.configFixup().apply(baseCfg);
            HfLoadContext ctx = HfLoadContext.builder(fixed, entry.withExtra(HfExtraTensors.NONE), arch)
                    .options(o).checkpoint(ckpt).tokenizerFiles(tokenizer)
                    .index(moduleIndex).resources(resources).build().withBlockCount(entry.blockCount(
                            HfLoadContext.builder(fixed, entry, arch).build()));
            lora.requireNamed(dir, ctx);
            HfQuantConfig quant = HfQuantConfig.of(baseCfg);
            // The adapter's factors are plain float sources the base model's quantization does not rewrite: the
            // converter's dequant_model leaves an adapter's LoraTorchTensor tensors alone, so the rewrite runs
            // unquantized and only the arch's filters and tensor hooks act on the factors.
            HfQuantRewriter.Result pass = HfQuantRewriter.rewrite(HfQuantConfig.none(), ctx, moduleIndex, o);
            HfSourceAssembler.Filtered kept = HfSourceAssembler.filter(ctx, pass, kind);
            HfSourceAssembler.Planned p = HfSourceAssembler.plan(kept.ctx(), kept, pass.moduleArgs());
            lora.requireMappedLmHead(dir, kept, p.plan());
            TensorPlan plan = p.plan();
            // the vision-tower and connector modules the text route's filter drop run through the base
            // architecture's projector conversion, which plans them as VISION pairs (vLLM's tower/connector LoRA)
            Set<String> towerModules = towerModules(lora, ctx);
            VisionRoute vision = towerModules.isEmpty() ? null
                    : visionRoute(dir, baseCfg, ckpt, resources, o, tokenizer, lora, towerModules);
            if (vision != null) {
                plan = TensorPlan.merge(plan, vision.plan);
            }
            HfMetadata md = metadata(config, arch, plan, kept.ctx(), base);
            validate(dir, plan, md, arch);
            VirtualLayout layout = VirtualLayout.of(ckpt, plan);
            PeftAdapterSource s = new PeftAdapterSource(dir.toString(), o, kept.ctx(), plan, md, layout, ckpt,
                    resources);
            LOG.info("Opened PEFT adapter {} as '{}': {} tensors ({} produced), alpha {}, {} metadata keys{}", dir,
                    arch.ggufName(), layout.produced().size(), layout.tensors().size() - layout.produced().size(),
                    config.alpha(), s.metadata.size(), vision == null ? ""
                    : " (" + vision.plan.tensors().size() + " vision-tower tensors)");
            return s;
        } catch (IOException | RuntimeException | Error e) {
            try {
                resources.close();
            } catch (Exception failed) {
                e.addSuppressed(failed);
            }
            try {
                ckpt.close();
            } catch (IOException failed) {
                e.addSuppressed(failed);
            }
            throw e;
        }
    }

    // ------------------------------------------------------------------ getters

    @Override
    public ModelFormat format() {
        return ModelFormat.SAFETENSORS;
    }

    @Override
    public String getFilePath() {
        return filePath;
    }

    @Override
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return valueTypes.get(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return elementTypes.get(key);
    }

    @Override
    public String getString(String key) {
        Object v = metadata.get(key);
        return v instanceof String ? (String) v : null;
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return tensors;
    }

    @Override
    public long getTensorCount() {
        return tensors.size();
    }

    /** The load context of the base architecture the view was built with. */
    public HfLoadContext context() {
        return context;
    }

    /** The tensor plan behind {@link #getTensors()}. */
    public TensorPlan plan() {
        return plan;
    }

    /** The virtual tensor-data layout. */
    public VirtualLayout layout() {
        return layout;
    }

    /** The private directory of the derived tensors kept on disk. */
    public SidecarDirectory sidecars() {
        return sidecars;
    }

    /**
     * A new weight store over the adapter's files and the produced tensors. The source closes it when it closes.
     *
     * @throws IOException when the source is closed or an adapter file cannot be mapped
     */
    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        Objects.requireNonNull(use, "use");
        synchronized (readers) {
            ensureOpen();
            HfWeightStore store = new HfWeightStore(layout(), view(MaterializePolicy.readerOf(use)));
            readers.add(store);
            return store;
        }
    }

    /**
     * New tensor regions for the CPU weights over the adapter's files and the produced tensors. The source closes
     * them when it closes.
     *
     * @throws IOException when the source is closed or an adapter file cannot be opened
     */
    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        Objects.requireNonNull(mode, "mode");
        synchronized (readers) {
            ensureOpen();
            HfTensorRegions regions = new HfTensorRegions(layout(), view(MaterializePolicy.Reader.CPU_REGIONS), mode,
                    filePath);
            readers.add(regions);
            return regions;
        }
    }

    private ProducedSegments.View view(MaterializePolicy.Reader reader) {
        ProducedSegments produced = layout().produced();
        return produced.view(MaterializePolicy.of(reader, options, produced.tensors()), sidecars);
    }

    private void ensureOpen() throws IOException {
        if (closed) {
            throw new IOException("PEFT adapter source " + filePath + " is closed");
        }
    }

    /**
     * Closes, in reverse open order, every weight store and region set, the sidecar directory and the adapter
     * checkpoint. The first failure is thrown with later ones suppressed; later calls do nothing.
     */
    @Override
    public void close() throws IOException {
        List<AutoCloseable> toClose;
        synchronized (readers) {
            if (closed) {
                return;
            }
            closed = true;
            toClose = new ArrayList<>(readers);
            readers.clear();
        }
        Collections.reverse(toClose);
        toClose.add(sidecars);
        toClose.add(checkpoint);
        IOException first = null;
        for (AutoCloseable c : toClose) {
            try {
                c.close();
            } catch (Exception e) {
                IOException io = e instanceof IOException ? (IOException) e
                        : new IOException("closing " + c + " of " + filePath + " failed", e);
                if (first == null) {
                    first = io;
                } else {
                    first.addSuppressed(io);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "PeftAdapterSource{" + filePath + ", " + context.arch().ggufName() + ", " + tensors.size()
                + " tensors}";
    }

    // ------------------------------------------------------------------ the vision tower's modules

    /**
     * One vision-tower planning route: the projector conversion's load context, its plan and the GGUF tensors each
     * module planned. Immutable.
     */
    private static final class VisionRoute {
        final HfLoadContext ctx;
        final TensorPlan plan;
        final Map<String, List<String>> added;

        VisionRoute(HfLoadContext ctx, HfSourceAssembler.Planned planned) {
            this.ctx = ctx;
            this.plan = planned.plan();
            this.added = planned.added();
        }
    }

    /**
     * The adapter modules the text route's filter drops: the vision-tower and connector candidates, which the
     * projector conversion plans. The paired modules and the kept side tensors, plus the raw names of neither kind
     * under their module weight name (those are refused through the vision route when its filter keeps them).
     */
    private static Set<String> towerModules(LoraTensors lora, HfLoadContext ctx) {
        Set<String> out = new LinkedHashSet<>();
        for (String module : lora.moduleNames()) {
            if (HfTensorFilters.filterName(ctx, module) == null) {
                out.add(module);
            }
        }
        for (String raw : lora.unmappedNames()) {
            String base = LoraTensors.baseName(raw);
            if (HfTensorFilters.filterName(ctx, base) == null) {
                out.add(base);
            }
        }
        return out;
    }

    /**
     * The projector conversion of the tower and connector modules: the base architecture's MMPROJ entry resolves the
     * checkpoint's projector class, whose name filters and tensor hooks plan the modules over a vision-only module
     * index exactly as an mmproj export does, and every routed module must land in the plan or be refused by name.
     *
     * @throws GgufFormatException when the base architecture has no projector conversion, a routed module maps to
     *                             no tensor of the projector conversion, or a module name the projector filter
     *                             keeps is neither a LoRA factor pair nor a norm
     */
    private static VisionRoute visionRoute(Path dir, HfConfig baseCfg, CheckpointContainer ckpt,
                                           HfResources resources, ModelSourceOptions o, HfTokenizerFiles tokenizer,
                                           LoraTensors lora, Set<String> modules) throws IOException {
        HfArchEntry projEntry;
        try {
            projEntry = HfArchRegistry.resolve(baseCfg, HfArchEntry.Kind.MMPROJ);
        } catch (GgufFormatException e) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': it adapts the vision tower or the"
                    + " connector (" + modules.iterator().next() + "), which needs the base architecture's"
                    + " projector conversion: " + e.getMessage(), e);
        }
        HfTensorIndex visionIndex = lora.moduleIndex(dir, modules);
        Arch arch = projEntry.arch().resolve(baseCfg, visionIndex);
        requirePlannableArch(baseCfg, arch);
        HfConfig fixed = projEntry.configFixup().apply(baseCfg);
        HfLoadContext ctx = HfLoadContext.builder(fixed, projEntry.withExtra(HfExtraTensors.NONE), arch)
                .options(o).checkpoint(ckpt).tokenizerFiles(tokenizer)
                .index(visionIndex).resources(resources).build().withBlockCount(projEntry.blockCount(
                        HfLoadContext.builder(fixed, projEntry, arch).build()));
        lora.requireNamed(dir, ctx);
        // the factors are plain float sources, as in the text route: the projector filters and tensor hooks act on
        // the module names as on an mmproj export's tensors
        HfQuantRewriter.Result pass = HfQuantRewriter.rewrite(HfQuantConfig.none(), ctx, visionIndex, o);
        HfSourceAssembler.Filtered kept = HfSourceAssembler.filter(ctx, pass, HfArchEntry.Kind.MMPROJ);
        HfSourceAssembler.Planned planned = HfSourceAssembler.plan(kept.ctx(), kept, pass.moduleArgs());
        for (String module : modules) {
            String filtered = HfTensorFilters.filterName(ctx, module);
            if (filtered == null) {
                continue; // the projector filter drops the name, as the converter skips filter-dropped names
            }
            if (planned.added().getOrDefault(filtered, List.of()).isEmpty()
                    && !planned.plan().dropped().containsKey(filtered)) {
                throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': the vision-tower module '"
                        + module + "' maps to no tensor of the projector conversion (arch '" + arch.ggufName()
                        + "')");
            }
        }
        return new VisionRoute(kept.ctx(), planned);
    }

    // ------------------------------------------------------------------ pieces

    /**
     * The adapter weights of {@code dir}, stored or its declared container format: the safetensors file the
     * lookahead detects, else {@code adapter_model.bin} when a registered container accepts it,
     * which none does by default.
     *
     * @throws FileNotFoundException         when the directory holds no adapter weights
     * @throws IOException                   when the weights cannot be read or no container accepts the format
     */
    private static CheckpointContainer adapterCheckpoint(Path dir) throws IOException {
        Path safetensors = dir.resolve(ADAPTER_MODEL_SAFETENSORS);
        if (Files.isRegularFile(safetensors)) {
            // The single adapter file itself, not the directory: the model discovery would exclude
            // adapter_model.safetensors by name (SafetensorsCheckpoint.byName:262) and a PEFT directory with no
            // model*.safetensors throws "it holds no model *.safetensors files".
            return CheckpointContainers.open(safetensors, dir, false);
        }
        Path bin = dir.resolve(ADAPTER_MODEL_BIN);
        if (Files.isRegularFile(bin)) {
            CheckpointContainers.Factory f = CheckpointContainers.forFile(bin);
            if (f != null) {
                return f.open(bin, dir, false);
            }
            throw new IOException("Invalid PEFT adapter '" + dir + "': " + ADAPTER_MODEL_BIN + " is a PyTorch"
                    + " pickle checkpoint; no registered container reads it and" + " " + ADAPTER_MODEL_SAFETENSORS
                    + " is missing");
        }
        throw new FileNotFoundException("Invalid PEFT adapter '" + dir + "': no adapter weights in " + dir
                + " (looked for " + ADAPTER_MODEL_SAFETENSORS + " and " + ADAPTER_MODEL_BIN + ")");
    }

    /**
     * The base model's configuration, in the order {@code convert_lora_to_gguf.py} follows: the safetensors base's own {@code HfConfig};
     * {@link ModelSourceOptions#loraBaseConfig()}, a directory or a {@code config.json}; {@code
     * base_model_name_or_path} when it names an existing local directory (absolute, or relative to the adapter
     * directory); else rule 4's error, as llama.cpp needs {@code --base} for the same reason.
     *
     * @throws IOException        when a configuration file cannot be read
     * @throws GgufFormatException for rule 4 of the resolution
     */
    static HfConfig baseConfig(Path dir, SourceOrigin base, ModelSourceOptions o, PeftAdapterConfig config)
            throws IOException {
        if (base.format() == ModelFormat.SAFETENSORS) {
            return HfConfigLoader.load(baseDir(base), o);
        }
        Path given = o.loraBaseConfig();
        if (given != null) {
            return HfConfigLoader.load(configDirOf(given), o);
        }
        String named = config.baseModelNameOrPath();
        if (named != null) {
            Path guessed = dir.resolve(named).normalize();
            if (Files.isDirectory(guessed)) {
                return HfConfigLoader.load(guessed, o);
            }
        }
        throw new GgufFormatException("a PEFT adapter on a GGUF base needs the base model's Hugging Face"
                + " config.json: pass --lora-base-config <dir>");
    }

    /** The directory of {@code base}: the path itself for a directory, else the file's directory. */
    private static Path baseDir(SourceOrigin base) {
        Path p = base.path().toAbsolutePath().normalize();
        return Files.isDirectory(p) ? p : p.getParent();
    }

    /** {@code given}: a {@code config.json} file's directory, or the directory itself. */
    private static Path configDirOf(Path given) {
        Path p = given.toAbsolutePath().normalize();
        return Files.isRegularFile(p) ? p.getParent() : p;
    }

    /** The arch registration of {@code cfg}: an unknown architecture is refused before the plan starts. */
    private static void requireSupported(HfConfig cfg, Arch arch) {
        if (arch == null || arch == Arch.UNKNOWN) {
            throw new GgufFormatException("Hugging Face architecture '" + cfg.architecture()
                    + "' resolves to no NeuralFree architecture for this checkpoint");
        }
        if (ArchRegistry.has(arch)) {
            ArchTraits traits = ArchRegistry.traits(arch);
            if (traits.support() == SupportClass.NA) {
                throw new UnsupportedArchException(arch, traits.unsupportedReason());
            }
        }
    }

    /**
     * The planning arch of the projector conversion: its MMPROJ entry resolves to a projector arch (SupportClass
     * NA), which never opens standalone but here only names the planned tensors, so only an unknown architecture
     * is refused; the base architecture's own support was checked on the text route.
     */
    private static void requirePlannableArch(HfConfig cfg, Arch arch) {
        if (arch == null || arch == Arch.UNKNOWN) {
            throw new GgufFormatException("Hugging Face architecture '" + cfg.architecture()
                    + "' resolves to no NeuralFree architecture for this checkpoint");
        }
    }

    /**
     * The adapter's metadata: {@code general.architecture}, {@code general.type = adapter},
     * {@code adapter.type = lora}, {@code adapter.lora.alpha} and the aLoRA invocation tokens of the two
     * configuration forms; first the keys the tensor hooks recorded in {@code plan.kv()}.
     */
    private static HfMetadata metadata(PeftAdapterConfig config, Arch arch, TensorPlan plan, HfLoadContext ctx,
                                       SourceOrigin base) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(arch.ggufName());
        kv.set(GgufKeys.GENERAL_ARCHITECTURE, arch.ggufName());
        for (TensorPlan.KvEntry e : plan.kv()) {
            kv.setTyped(e.key(), e.type(), e.elementType(), e.value());
        }
        kv.set(GgufKeys.GENERAL_TYPE, TYPE_ADAPTER);
        kv.set(GgufKeys.ADAPTER_TYPE, ADAPTER_TYPE_LORA);
        kv.setTyped(GgufKeys.ADAPTER_LORA_ALPHA, GGUFValueType.FLOAT32, config.alpha());
        List<Integer> tokens = invocationTokens(config, ctx, base);
        if (tokens != null) {
            kv.setTyped(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS, GGUFValueType.ARRAY, GGUFValueType.UINT32, tokens);
        }
        return kv.build();
    }

    /**
     * {@code adapter.alora.invocation_tokens}: the configuration's {@code alora_invocation_tokens}, or the
     * {@code invocation_string} tokenized with the base model's tokenizer with the first id dropped
     * (LC/convert_lora_to_gguf.py:428-451); null when neither is set or the tokenized invocation is empty.
     *
     * @throws IOException when the base tokenizer cannot be opened or read
     */
    private static List<Integer> invocationTokens(PeftAdapterConfig config, HfLoadContext ctx, SourceOrigin base)
            throws IOException {
        if (config.aloraInvocationTokens() != null) {
            return config.aloraInvocationTokens();
        }
        String invocation = config.invocationString();
        if (invocation == null) {
            return null;
        }
        int[] all = tokenizer(ctx, base).encode(invocation);
        if (all.length <= 1) {
            return null;
        }
        List<Integer> out = new ArrayList<>(all.length - 1);
        for (int i = 1; i < all.length; i++) {
            out.add(all[i]);
        }
        return out;
    }

    /**
     * The tokenizer of the base model at {@code base}: a GGUF base's {@code tokenizer.*} metadata through
     * {@link GGUFTokenizer}, an HF-directory base's signals through the vocabulary written by the base arch's
     * vocab hook (the {@code AutoTokenizer.from_pretrained} of the converter).
     *
     * @throws IOException when the base tokenizer cannot be read
     */
    private static GGUFTokenizer tokenizer(HfLoadContext ctx, SourceOrigin base) throws IOException {
        if (base.format() == ModelFormat.GGUF) {
            try (GGUFFileParser parser = ModelSources.parseGguf(base.path().toString())) {
                return new GGUFTokenizer(parser);
            }
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(HfSourceAssembler.vocab(ctx).metadata()));
    }

    /**
     * The view's invariants: at least one tensor, unique planned names, every tensor valid as a GGUF tensor info, and
     * {@code general.architecture} the resolved base architecture.
     *
     * @throws IOException         for an invalid tensor info
     * @throws GgufFormatException for a violated invariant
     */
    private static void validate(Path dir, TensorPlan plan, HfMetadata md, Arch arch) throws IOException {
        String prefix = "Invalid PEFT adapter '" + dir + "'";
        if (plan.tensors().isEmpty()) {
            throw new GgufFormatException(prefix + ": the conversion plans no tensors");
        }
        Set<String> names = new HashSet<>();
        for (PlannedTensor t : plan.tensors()) {
            if (!names.add(t.ggufName())) {
                throw new GgufFormatException(prefix + ": tensor '" + t.ggufName() + "' is planned twice");
            }
            GGUFTensorInfo.validate(prefix, new GGUFTensorInfo(t.ggufName(), t.ne(), t.type(), 0));
        }
        Object archName = md.get(GgufKeys.GENERAL_ARCHITECTURE);
        if (!arch.ggufName().equals(archName)) {
            throw new GgufFormatException(prefix + ": " + GgufKeys.GENERAL_ARCHITECTURE + " is '" + archName
                    + "', not the resolved architecture '" + arch.ggufName() + "'");
        }
    }

    // ------------------------------------------------------------------ the adapter's tensors

    /**
     * The tensors of one adapter directory collected from the checkpoint's names (LC/convert_lora_to_gguf.py:460-502):
     * the factor halves of each module, the kept {@code _norm} / {@code .norm} side tensors, and the raw names of
     * neither kind, which are each refused when they survive the base arch's filter. Immutable once collected;
     * the index's order is the checkpoint's.
     */
    private static final class LoraTensors {

        /** The two stored tensor names of one module: {@code lora_A}/{@code lora_embedding_A} and the B half. */
        private static final class Half {
            String a;
            String b;
        }

        /** The checkpoint the sources come from. */
        private final CheckpointContainer checkpoint;
        /** One entry per module whose weight name a factor marks, in discovery order. */
        private final LinkedHashMap<String, Half> halves = new LinkedHashMap<>();
        /** The norm tensors kept, module weight name to source. */
        private final LinkedHashMap<String, RowSource> plain = new LinkedHashMap<>();
        /** The raw names that are neither factor halves nor known side tensors. */
        private final List<String> unmapped = new ArrayList<>();

        private LoraTensors(CheckpointContainer checkpoint) {
            this.checkpoint = checkpoint;
        }

        static LoraTensors of(CheckpointContainer ckpt) {
            LoraTensors t = new LoraTensors(ckpt);
            for (StTensor st : ckpt.tensors()) {
                String name = st.name();
                boolean isA = name.contains(".lora_A.weight") || name.contains(".lora_embedding_A");
                boolean isB = name.contains(".lora_B.weight") || name.contains(".lora_embedding_B");
                if (isA || isB) {
                    Half h = t.halves.computeIfAbsent(baseName(name), k -> new Half());
                    if (isA) {
                        h.a = name;
                    } else {
                        h.b = name;
                    }
                    continue;
                }
                if (name.contains(".base_layer.weight")) {
                    continue;
                }
                if (name.contains("_layernorm") || name.contains(".norm")) {
                    t.plain.put(baseName(name), ckpt.source(name));
                    continue;
                }
                t.unmapped.add(name);
            }
            return t;
        }

        /**
         * Refuses the name pairings the conversion cannot plan: a module with only one factor half
         * (LC/convert_lora_to_gguf.py:499-501).
         */
        void requirePaired(Path dir) {
            for (Map.Entry<String, Half> e : halves.entrySet()) {
                Half h = e.getValue();
                if (h.a == null || h.b == null) {
                    throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': LoRA tensor '" + e.getKey()
                            + "' has no " + (h.a == null ? "lora_A" : "lora_B") + " half"
                            + " (convert_lora_to_gguf.py:499-501)");
                }
            }
        }

        /**
         * The output layer's delta that the base arch ignores: kept by the filter but dropped by its tensor hook
         * (LC/convert_lora_to_gguf.py:506-511).
         */
        void requireMappedLmHead(Path dir, HfSourceAssembler.Filtered kept, TensorPlan plan) {
            if (!halves.containsKey(LM_HEAD) || !kept.ctx().index().has(LM_HEAD)) {
                return;
            }
            if (plan.dropped().containsKey(LM_HEAD)) {
                throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': lm_head is present in adapter, but"
                        + " is ignored in base model (convert_lora_to_gguf.py:506-511)");
            }
        }

        /**
         * The module weight name of {@code loraName}, with the {@code base_model.model.} prefix and the marker
         * stripped (LC/convert_lora_to_gguf.py:272-278), so the keys are the base model's module names (the
         * {@code model.layers.0....} / {@code model.embed_tokens.weight} / {@code lm_head.weight} the base arch's
         * hooks map).
         */
        private static String baseName(String loraName) {
            String base = loraName.startsWith(BASE_MODEL_PREFIX)
                    ? loraName.substring(BASE_MODEL_PREFIX.length()) : loraName;
            return base.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
                    .replace(".lora_embedding_A", ".weight").replace(".lora_embedding_B", ".weight");
        }

        /** The module weight names of the paired modules and the kept side tensors, in index order. */
        List<String> moduleNames() {
            List<String> out = new ArrayList<>(halves.keySet());
            out.addAll(plain.keySet());
            return out;
        }

        /** The raw checkpoint names of neither kind, in discovery order. */
        List<String> unmappedNames() {
            return Collections.unmodifiableList(new ArrayList<>(unmapped));
        }

        /** The raw checkpoint name behind the module name {@code module}, or null when it is of a known kind. */
        String rawNameOf(String module) {
            for (String raw : unmapped) {
                if (baseName(raw).equals(module)) {
                    return raw;
                }
            }
            return null;
        }

        /**
         * The module index the base arch's hooks run over: one {@link LoraFactorSource} per paired module under its
         * weight name, the kept side tensors as stored sources.
         *
         * @throws GgufFormatException when a module has only one half (LC/convert_lora_to_gguf.py:499-501)
         */
        HfTensorIndex moduleIndex(Path dir) {
            LinkedHashMap<String, RowSource> index = new LinkedHashMap<>();
            for (Map.Entry<String, Half> e : halves.entrySet()) {
                Half h = e.getValue();
                if (h.a == null || h.b == null) {
                    throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': LoRA tensor '" + e.getKey()
                            + "' has no " + (h.a == null ? "lora_A" : "lora_B") + " half"
                            + " (convert_lora_to_gguf.py:499-501)");
                }
                index.put(e.getKey(), new LoraFactorSource(e.getKey(), ckptSource(h.a), ckptSource(h.b)));
            }
            index.putAll(plain);
            return HfTensorIndex.of(index);
        }

        /**
         * {@link #moduleIndex(Path)} over the modules {@code modules} name only: their paired factors, then their
         * kept side tensors ({@link #requirePaired} has run, so every half pair is complete). A module of neither
         * kind is in no index: {@link #requireNamed} refuses it by name or both filters drop it.
         */
        HfTensorIndex moduleIndex(Path dir, Set<String> modules) {
            LinkedHashMap<String, RowSource> index = new LinkedHashMap<>();
            for (Map.Entry<String, Half> e : halves.entrySet()) {
                if (!modules.contains(e.getKey())) {
                    continue;
                }
                Half h = e.getValue();
                index.put(e.getKey(), new LoraFactorSource(e.getKey(), ckptSource(h.a), ckptSource(h.b)));
            }
            for (Map.Entry<String, RowSource> e : plain.entrySet()) {
                if (modules.contains(e.getKey())) {
                    index.put(e.getKey(), e.getValue());
                }
            }
            return HfTensorIndex.of(index);
        }

        private RowSource ckptSource(String name) {
            return checkpoint.source(name);
        }

        /**
         * Refuses the names the converter refuses once the base arch's filter is known: an unmapped name the filter
         * keeps (:482-486); names the filter drops are skipped as the converter's filter does (:469-471).
         *
         * @throws GgufFormatException for a kept name that is not a LoRA tensor or a norm
         */
        void requireNamed(Path dir, HfLoadContext ctx) {
            for (String raw : unmapped) {
                if (HfTensorFilters.filterName(ctx, baseName(raw)) != null) {
                    String hint = raw.contains(".embed_tokens.weight") || raw.contains(".lm_head.weight")
                            ? "; an embedding in the adapter is not a LoRA delta (a fine tune that added tokens?"
                            + " see the llama.cpp PR 9948 discussion)"
                            : "";
                    throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': Unexpected name '" + raw
                            + "': Not a lora_A or lora_B tensor" + hint);
                }
            }
        }
    }
}
