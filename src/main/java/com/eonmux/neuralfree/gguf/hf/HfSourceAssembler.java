package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantMethod;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.meta.HfHadamardMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.quant.HfQuantRewriter;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.VirtualLayout;
import com.eonmux.neuralfree.gguf.hf.tensor.DtypePolicy;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlanBuilder;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.OnlineQuantizers;
import com.eonmux.neuralfree.gguf.quant.RequantPolicy;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfGenerationDefaults;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * The pipeline that turns a Hugging Face checkpoint into the GGUF view of a {@link HfModelSource}, one public method
 * per step, in the converter's order (LC/conversion/base.py:1400-1406):
 * <ol>
 *   <li>{@link #config}: configuration, quantization configuration and tokenizer files;</li>
 *   <li>{@link #checkpoint}: the checkpoint container;</li>
 *   <li>{@link #resolve}: the architecture entry, the NeuralFree architecture, the configuration fixup and the load
 *       context;</li>
 *   <li>{@link #index}: the entry's index hook and the block count that depends on the checkpoint's names;</li>
 *   <li>{@link #quant}: the quantization rewrite;</li>
 *   <li>{@link #filter}: the converter's name filters and {@code --no-mtp};</li>
 *   <li>{@link #vocab}: the tokenizer metadata;</li>
 *   <li>{@link #plan}: the tensor plan and the {@code prec_a4} flags, requantized by the online-quantization seam
 *       ({@link #requant}, QZ-1);</li>
 *   <li>{@link #metadata}: every metadata key;</li>
 *   <li>{@link #validate}: the view's invariants;</li>
 *   <li>{@link #layout}: the virtual tensor-data layout.</li>
 * </ol>
 * {@link #assemble} runs them all.
 */
public final class HfSourceAssembler {

    /** {@code neuralfree.source.format} of every view built here. */
    public static final String SOURCE_FORMAT = "safetensors";
    /** {@code general.quantization_version}, always written (LC/conversion/base.py:1384-1386). */
    public static final int QUANTIZATION_VERSION = 2;
    /** {@code general.type} of a model (GGUFType.MODEL). */
    public static final String TYPE_MODEL = "model";
    /** {@code general.type} of a projector view (GGUFType.MMPROJ, LC/conversion/base.py:2923). */
    public static final String TYPE_MMPROJ = "mmproj";

    private HfSourceAssembler() {
    }

    /** Step 1: the model directory, its configuration, quantization configuration and tokenizer files. Immutable. */
    public static final class Config {
        private final Path dir;
        private final HfConfig config;
        private final HfQuantConfig quant;
        private final HfTokenizerFiles tokenizer;

        Config(Path dir, HfConfig config, HfQuantConfig quant, HfTokenizerFiles tokenizer) {
            this.dir = dir;
            this.config = config;
            this.quant = quant;
            this.tokenizer = tokenizer;
        }

        public Path dir() { return dir; }

        public HfConfig config() { return config; }

        public HfQuantConfig quant() { return quant; }

        public HfTokenizerFiles tokenizer() { return tokenizer; }
    }

    /** Step 6: the filtered load, the index the output type is guessed from, the dropped prefixes. Immutable. */
    public static final class Filtered {
        private final HfLoadContext ctx;
        private final HfTensorIndex guessIndex;
        private final List<String> droppedPrefixes;
        private final List<KvScaleCollector.Scale> kvScales;

        Filtered(HfLoadContext ctx, HfTensorIndex guessIndex, List<String> droppedPrefixes,
                 List<KvScaleCollector.Scale> kvScales) {
            this.ctx = ctx;
            this.guessIndex = guessIndex;
            this.droppedPrefixes = droppedPrefixes;
            this.kvScales = kvScales;
        }

        /** The load with the filtered index and the final block count. */
        public HfLoadContext ctx() { return ctx; }

        /** The stored tensors under converter names, before the quantization rewrite (for {@code --outtype auto}). */
        public HfTensorIndex guessIndex() { return guessIndex; }

        public List<String> droppedPrefixes() { return droppedPrefixes; }

        /** The KV-cache scales of the kept layers. */
        public List<KvScaleCollector.Scale> kvScales() { return kvScales; }
    }

    /** Step 7: the load with the vocabulary's token count, and the tokenizer metadata. Immutable. */
    public static final class Vocab {
        private final HfLoadContext ctx;
        private final Map<String, Object> metadata;

        Vocab(HfLoadContext ctx, Map<String, Object> metadata) {
            this.ctx = ctx;
            this.metadata = metadata;
        }

        public HfLoadContext ctx() { return ctx; }

        /** The {@code tokenizer.*} keys (concrete names). Unmodifiable. */
        public Map<String, Object> metadata() { return metadata; }
    }

    /** Step 8: the load with its {@code prec_a4} flags, and the plan. Immutable. */
    public static final class Planned {
        private final HfLoadContext ctx;
        private final TensorPlan plan;

        Planned(HfLoadContext ctx, TensorPlan plan) {
            this.ctx = ctx;
            this.plan = plan;
        }

        public HfLoadContext ctx() { return ctx; }

        public TensorPlan plan() { return plan; }
    }

    /** Every step's product. The checkpoint and the resources stay open until the model source closes them. */
    public static final class Assembly {
        private final HfLoadContext ctx;
        private final CheckpointContainer checkpoint;
        private final HfResources resources;
        private final TensorPlan plan;
        private final HfMetadata metadata;
        private final VirtualLayout layout;

        Assembly(HfLoadContext ctx, CheckpointContainer checkpoint, HfResources resources, TensorPlan plan,
                 HfMetadata metadata, VirtualLayout layout) {
            this.ctx = ctx;
            this.checkpoint = checkpoint;
            this.resources = resources;
            this.plan = plan;
            this.metadata = metadata;
            this.layout = layout;
        }

        public HfLoadContext ctx() { return ctx; }

        public CheckpointContainer checkpoint() { return checkpoint; }

        public HfResources resources() { return resources; }

        public TensorPlan plan() { return plan; }

        public HfMetadata metadata() { return metadata; }

        public VirtualLayout layout() { return layout; }
    }

    /**
     * Runs every step for the checkpoint at {@code path} (a directory or one {@code .safetensors} file) opened as
     * {@code kind}. On failure everything opened so far is closed.
     *
     * @param target the model a draft checkpoint serves, or null
     * @param mode   how the engine treats the checkpoint's activation quantization
     */
    public static Assembly assemble(Path path, ModelSourceOptions o, Path target, ActivationQuant mode,
                                    HfArchEntry.Kind kind) throws IOException {
        Config c = config(path, o);
        CheckpointContainer ckpt = checkpoint(path, c, o);
        HfResources resources = new HfResources();
        try {
            HfLoadContext ctx = index(resolve(c, ckpt, kind, o, resources, target));
            HfQuantRewriter.Result q = quant(ctx, c.quant(), mode);
            Filtered f = filter(ctx, q, kind);
            Vocab v = vocab(f.ctx());
            Planned p = plan(v.ctx(), f, q.moduleArgs());
            HfMetadata md = metadata(p.ctx(), p.plan(), q.hadamard().orElse(null), v.metadata(),
                    f.droppedPrefixes(), c.quant());
            validate(p.ctx(), p.plan(), md);
            VirtualLayout layout = layout(ckpt, p.plan());
            return new Assembly(p.ctx(), ckpt, resources, p.plan(), md, layout);
        } catch (IOException | RuntimeException | Error e) {
            closeAfterFailure(resources, ckpt, e);
            throw e;
        }
    }

    /**
     * Step 1. The configuration directory is {@code path} itself or, for a single file, its directory; the tokenizer
     * files come from it unless {@link ModelSourceOptions#tokenizerDir()} names another source.
     *
     * @throws IOException         when the configuration or tokenizer files cannot be read
     * @throws GgufFormatException when the configuration or its quantization method is invalid or refused
     */
    public static Config config(Path path, ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(o, "o");
        Path p = path.toAbsolutePath().normalize();
        Path dir = Files.isDirectory(p) ? p : p.getParent();
        HfConfig cfg = HfConfigLoader.load(dir, o);
        HfQuantConfig q = HfQuantConfig.of(cfg);
        HfTokenizerFiles tok = HfTokenizerFiles.locate(dir).withTokenizerDir(o.tokenizerDir());
        return new Config(dir, cfg, q, tok);
    }

    /**
     * Step 2: the checkpoint container. With {@link ModelSourceOptions#stDenseModules()} false, the
     * sentence-transformers {@code N_Dense} side files read as absent.
     */
    public static CheckpointContainer checkpoint(Path path, Config c, ModelSourceOptions o) throws IOException {
        CheckpointContainer ckpt = CheckpointContainers.open(path, c.dir(), c.config().mistralNative());
        return o.stDenseModules() ? ckpt : new HfAssemblerParts.WithoutDenseModules(ckpt);
    }

    /**
     * Step 3: the entry of {@code kind}, the NeuralFree architecture it resolves the checkpoint to, the fixed
     * configuration and the load context over the checkpoint's index.
     *
     * @throws GgufFormatException      when no entry converts the checkpoint or no architecture fits it
     * @throws UnsupportedArchException when the architecture is registered as not supported
     */
    public static HfLoadContext resolve(Config c, CheckpointContainer ckpt, HfArchEntry.Kind kind,
                                        ModelSourceOptions o, HfResources resources, Path target) {
        HfTensorIndex index = HfTensorIndex.of(ckpt);
        HfArchEntry entry = kind == HfArchEntry.Kind.MMPROJ ? projectorEntry(c.config())
                : HfArchRegistry.resolve(c.config(), kind);
        Arch arch = entry.arch().resolve(c.config(), index);
        requireSupported(c.config(), arch);
        HfConfig fixed = entry.configFixup().apply(c.config());
        return HfLoadContext.builder(fixed, entry, arch).options(o).checkpoint(ckpt).tokenizerFiles(c.tokenizer())
                .index(index).resources(resources).targetPath(target).build();
    }

    /** Step 4: the entry's index hook, then the block count that may depend on the checkpoint's names. */
    public static HfLoadContext index(HfLoadContext ctx) throws IOException {
        HfTensorIndex index = ctx.entry().index().apply(ctx, ctx.index());
        HfLoadContext next = ctx.withIndex(index);
        return next.withBlockCount(next.entry().blockCount(next));
    }

    /** Step 5: the quantization rewrite of the step-4 index. */
    public static HfQuantRewriter.Result quant(HfLoadContext ctx, HfQuantConfig q, ActivationQuant mode)
            throws IOException {
        return HfQuantRewriter.rewrite(q, ctx, ctx.index(), ctx.options(), mode);
    }

    /**
     * Step 6: the converter's name filters over the rewritten index and over the step-4 index (which the output type
     * is guessed from). With {@link ModelSourceOptions#includeMtp()} false, the layers past the configuration's block
     * count and the {@code mtp}/{@code nextn} tensors are dropped and the block count is reduced, as {@code --no-mtp}
     * does. The KV-cache scales follow their tensors through the same filters and take the block of the filtered
     * name.
     *
     * @throws GgufFormatException when two kept KV-cache scales become one GGUF tensor
     */
    public static Filtered filter(HfLoadContext ctx, HfQuantRewriter.Result q, HfArchEntry.Kind kind) {
        HfTensorFilters.Result kept = HfTensorFilters.apply(ctx, q.index(), kind);
        HfTensorIndex guess = HfTensorFilters.apply(ctx, ctx.index(), kind).index();
        HfTensorIndex index = kept.index();
        HfLoadContext next = ctx;
        int base = Math.min(ctx.blockCount(), ctx.config().blockCount());
        boolean dropMtp = !ctx.options().includeMtp();
        if (dropMtp) {
            index = HfAssemblerParts.MtpFilter.drop(index, base);
            guess = HfAssemblerParts.MtpFilter.drop(guess, base);
            if (base < ctx.blockCount()) {
                ctx.log("MTP layers excluded: block count " + ctx.blockCount() + " -> " + base);
                next = ctx.withBlockCount(base);
            }
        }
        Map<String, KvScaleCollector.Scale> scales = new LinkedHashMap<>();
        for (KvScaleCollector.Scale s : q.kvScales()) {
            String filtered = HfTensorFilters.filterName(ctx, s.hfName());
            if (filtered == null || (dropMtp && HfAssemblerParts.MtpFilter.isMtp(filtered, base))) {
                continue;
            }
            int bid = HfLoadContext.bidOf(filtered);
            if (bid < 0 || bid >= next.blockCount()) {
                continue;
            }
            KvScaleCollector.Scale at = s.atLayer(bid);
            KvScaleCollector.Scale prev = scales.putIfAbsent(at.ggufName(), at);
            if (prev != null) {
                throw new GgufFormatException("KV-cache scale " + at.ggufName() + " is given by both "
                        + prev.hfName() + " and " + at.hfName());
            }
        }
        return new Filtered(next.withIndex(index), guess, kept.droppedPrefixes(),
                Collections.unmodifiableList(new ArrayList<>(scales.values())));
    }

    /** Step 7: the entry's vocabulary, and the load with its token count. */
    public static Vocab vocab(HfLoadContext ctx) throws IOException {
        Map<String, Object> tok = ctx.entry().vocab().build(ctx, new HfPreTokenizerResolver());
        Map<String, Object> copy = Collections.unmodifiableMap(new LinkedHashMap<>(tok));
        Object tokens = copy.get(GgufKeys.TOKENIZER_LIST);
        HfLoadContext next = tokens instanceof List ? ctx.withVocabTokenCount(((List<?>) tokens).size()) : ctx;
        return new Vocab(next, copy);
    }

    /**
     * Step 8: every tensor of the filtered index through the entry's tensor hook (in index order, skipping consumed
     * names), the KV-cache scales, the entry's extra tensors; then the {@code prec_a4} flags of the GGUF weights the
     * hooks planned from W4A16_NVFP4 modules (LC/conversion/base.py:665-675).
     */
    public static Planned plan(HfLoadContext ctx, Filtered f, Map<String, QuantArgs> moduleArgs) throws IOException {
        DtypePolicy policy = new DtypePolicy(HfNameTables.forcedF32(), ctx.options().outType())
                .resolveAuto(f.guessIndex());
        TensorPlanBuilder b = new TensorPlanBuilder(policy, ctx.entry().forceType());
        HfAssemblerParts.Requant requant = requant(ctx, policy);
        HfAssemblerParts.RecordingSink sink = new HfAssemblerParts.RecordingSink(b, requant);
        HfTensorIndex index = ctx.index();
        for (String name : index.names()) {
            if (!b.isConsumed(name)) {
                sink.begin(name);
                ctx.entry().tensors().apply(ctx, name, index.get(name), sink);
            }
        }
        for (KvScaleCollector.Scale s : f.kvScales()) {
            b.add(s.ggufName(), s.source());
        }
        ctx.entry().extra().generate(ctx, b);
        TensorPlan plan = b.build();
        Map<String, Boolean> prec = HfAssemblerParts.PrecA4.flags(ctx, moduleArgs, sink.added(), plan);
        return new Planned(prec.isEmpty() ? ctx : ctx.withPrecA4(prec), plan);
    }

    /**
     * Step 8a (QZ-1): the plan's online-quantization seam. {@code --quant-online} requantizes float matmul weights
     * while planning, after the plan's tensors are decided and before {@link #metadata} types them: every tensor the
     * hooks plan ({@link RequantPolicy#eligible(String, GGMLType, long[], GGMLType)}, llama-quant.cpp:295-396) from
     * a plain float source is planned from a producer that requantizes its stored floats into the option's
     * quantizer, so the planned type, the served bytes and the recomputed {@code general.file_type} are the
     * quantized ones; an already-quantized checkpoint's tensors and {@code NONE} requantize nothing. The decision
     * runs on the plan's add-time sink because the planned type it reads is the policy's; PAR QN-4 extends the
     * quantizers behind {@link OnlineQuantizers}.
     *
     * @return the requantization the adds of {@code ctx}'s planning pass apply, null for {@code NONE}
     */
    static HfAssemblerParts.Requant requant(HfLoadContext ctx, DtypePolicy policy) {
        if (ctx.options().onlineQuant() == ModelSourceOptions.OnlineQuant.NONE) {
            return null;
        }
        return new HfAssemblerParts.Requant(ctx.entry().forceType(), policy, ctx.options().onlineQuant());
    }

    /**
     * Step 9: {@code general.architecture}, the entry's metadata hook, the Hadamard and {@code prec_a4} keys, the
     * keys the tensor hooks recorded, the remaining {@code general.*} keys, the {@code neuralfree.*} keys, the
     * tokenizer keys, and the generation defaults no earlier step wrote.
     *
     * @param online the online Hadamard input transform in checkpoint names, or null
     */
    public static HfMetadata metadata(HfLoadContext ctx, TensorPlan plan, HadamardPlan online,
                                      Map<String, Object> vocab, List<String> droppedPrefixes, HfQuantConfig q)
            throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        kv.set(GgufKeys.GENERAL_ARCHITECTURE, ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, plan);
        HfHadamardMetadata.apply(ctx, kv, plan, online);
        for (TensorPlan.KvEntry e : plan.kv()) {
            kv.setTyped(e.key(), e.type(), e.elementType(), e.value());
        }
        if (!kv.has(GgufKeys.GENERAL_NAME)) {
            kv.set(GgufKeys.GENERAL_NAME, generalName(ctx.config()));
        }
        kv.set(GgufKeys.GENERAL_TYPE, ctx.entry().kind() == HfArchEntry.Kind.MMPROJ ? TYPE_MMPROJ : TYPE_MODEL);
        kv.set(GgufKeys.GENERAL_FILE_TYPE, HfFileType.guess(plan));
        kv.set(GgufKeys.GENERAL_QUANTIZATION_VERSION, QUANTIZATION_VERSION);
        neuralfreeKeys(kv, plan, droppedPrefixes, q);
        kv.putAll(vocab);
        for (Map.Entry<String, Object> e : HfGenerationDefaults.of(ctx.config(), ctx.vocabTokenCount()).entrySet()) {
            if (!kv.has(e.getKey())) {
                kv.putAll(Collections.singletonMap(e.getKey(), e.getValue()));
            }
        }
        return kv.build();
    }

    /**
     * Step 10: at least one tensor, unique names, every tensor valid as a GGUF tensor info, the extension-type flag
     * consistent with the planned types, and {@code general.architecture} the resolved architecture.
     *
     * @throws IOException         for an invalid tensor info
     * @throws GgufFormatException for a violated invariant
     */
    public static void validate(HfLoadContext ctx, TensorPlan plan, HfMetadata md) throws IOException {
        String prefix = "Invalid Hugging Face checkpoint '" + ctx.config().dir() + "'";
        if (plan.tensors().isEmpty()) {
            throw new GgufFormatException(prefix + ": the conversion plans no tensors");
        }
        Set<String> names = new HashSet<>();
        boolean extension = false;
        for (PlannedTensor t : plan.tensors()) {
            if (!names.add(t.ggufName())) {
                throw new GgufFormatException(prefix + ": tensor '" + t.ggufName() + "' is planned twice");
            }
            GGUFTensorInfo.validate(prefix, new GGUFTensorInfo(t.ggufName(), t.ne(), t.type(), 0));
            extension |= t.type().isExtension();
        }
        boolean declared = Boolean.TRUE.equals(md.get(GgufKeys.NF_EXTENSION_TYPES));
        if (declared != extension) {
            throw new GgufFormatException(prefix + ": " + GgufKeys.NF_EXTENSION_TYPES + " is "
                    + md.get(GgufKeys.NF_EXTENSION_TYPES) + " but " + (extension ? "a" : "no")
                    + " tensor uses a NeuralFree extension type");
        }
        Object archName = md.get(GgufKeys.GENERAL_ARCHITECTURE);
        if (!ctx.arch().ggufName().equals(archName)) {
            throw new GgufFormatException(prefix + ": " + GgufKeys.GENERAL_ARCHITECTURE + " is '" + archName
                    + "', not the resolved architecture '" + ctx.arch().ggufName() + "'");
        }
    }

    /** Step 11: the virtual layout of {@code plan} over the checkpoint's files. */
    public static VirtualLayout layout(CheckpointContainer ckpt, TensorPlan plan) {
        return VirtualLayout.of(ckpt, plan);
    }

    // ------------------------------------------------------------------ helpers

    /**
     * The projector entry of {@code cfg}; a registered entry that is rejected throws its own reason.
     *
     * @throws GgufFormatException when no projector entry is registered for the checkpoint
     */
    static HfArchEntry projectorEntry(HfConfig cfg) {
        try {
            return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
        } catch (GgufFormatException e) {
            if (HfArchRegistry.has(cfg.architecture(), HfArchEntry.Kind.MMPROJ)) {
                throw e;
            }
            throw new GgufFormatException("'" + cfg.architecture() + "' has no projector conversion yet"
                    + " (vision/audio towers are delivered by the multimodal work)", e);
        }
    }

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

    /** {@code general.name}: the last path segment of {@code _name_or_path}, else the model directory's name. */
    static String generalName(HfConfig cfg) {
        Object v = cfg.flat().get("_name_or_path");
        if (v instanceof String) {
            String[] parts = ((String) v).split("[/\\\\]");
            for (int i = parts.length - 1; i >= 0; i--) {
                if (!parts[i].trim().isEmpty()) {
                    return parts[i].trim();
                }
            }
        }
        Path name = cfg.dir() == null ? null : cfg.dir().getFileName();
        return name == null ? cfg.architecture() : name.toString();
    }

    private static void neuralfreeKeys(HfMetadataBuilder kv, TensorPlan plan, List<String> dropped,
                                       HfQuantConfig q) {
        kv.set(GgufKeys.NF_SOURCE_FORMAT, SOURCE_FORMAT);
        kv.set(GgufKeys.NF_SOURCE_QUANT_METHOD, q.method().name().toLowerCase(Locale.ROOT));
        // NeuralFree bookkeeping, not a converter key: an empty list means the source loaded with no drops,
        // and absence must stay distinguishable from it, so the empty array is written (HfMetadataBuilder#setTyped)
        kv.setTyped(GgufKeys.NF_HF_DROPPED_PREFIXES, GGUFValueType.ARRAY, GGUFValueType.STRING, dropped);
        for (PlannedTensor t : plan.tensors()) {
            if (t.type().isExtension()) {
                kv.set(GgufKeys.NF_EXTENSION_TYPES, true);
                break;
            }
        }
        Map<String, ActQuantScheme> act = new TreeMap<>(plan.actQuantEntries());
        scaleUpperBound(kv, q, act);
        if (!act.isEmpty()) {
            List<Integer> codes = new ArrayList<>(act.size());
            for (ActQuantScheme s : act.values()) {
                codes.add(s.code());
            }
            kv.set(GgufKeys.NF_ACT_QUANT_VERSION, 1);
            kv.set(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, new ArrayList<>(act.keySet()));
            kv.set(GgufKeys.NF_ACT_QUANT_SCHEMES, codes);
        }
    }

    /**
     * {@code neuralfree.act_quant.scale_ub} (QZ-1, §1.4/§1.7): the fbgemm_fp8 {@code activation_scale_ub}
     * (vllm model_executor/layers/quantization/fbgemm_fp8.py:76, :142-146) written as the upper bound on
     * {@code max|x|} of the dynamic FP8 per-token scheme when the checkpoint is fbgemm_fp8 and any weight carries
     * {@link ActQuantScheme#FP8_TOKEN}; a two-entry bound keeps its upper entry (the transformers form). Nothing is
     * written when the configuration carries no bound, and a bound that is not finite and &gt; 0 is a
     * {@link GgufFormatException} naming the key.
     */
    private static void scaleUpperBound(HfMetadataBuilder kv, HfQuantConfig q, Map<String, ActQuantScheme> act) {
        if (q.method() != HfQuantMethod.FBGEMM_FP8 || !act.containsValue(ActQuantScheme.FP8_TOKEN)) {
            return;
        }
        Object configured = q.view().raw("activation_scale_ub");
        Object number = configured;
        if (configured instanceof List) {
            List<?> entries = (List<?>) configured;
            for (int i = entries.size() - 1; i >= 0 && !(number instanceof Number); i--) {
                number = entries.get(i);
            }
        }
        if (!(number instanceof Number)) {
            throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_SCALE_UB + ": activation_scale_ub is "
                    + (number == null ? "null" : number.getClass().getSimpleName()));
        }
        float ub = ((Number) number).floatValue();
        if (!(ub > 0.0f && ub < Float.POSITIVE_INFINITY)) {
            throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_SCALE_UB + " is " + ub
                    + "; expected a finite positive bound");
        }
        kv.set(GgufKeys.NF_ACT_QUANT_SCALE_UB, ub);
    }

    private static void closeAfterFailure(HfResources resources, CheckpointContainer ckpt, Throwable failure) {
        try {
            resources.close();
        } catch (Exception e) {
            failure.addSuppressed(e);
        }
        try {
            ckpt.close();
        } catch (IOException e) {
            failure.addSuppressed(e);
        }
    }

    /** The {@code --no-mtp} drop and the QZ-1 requantization decisions, the recording sink, the {@code prec_a4}
     *  flags and the dense-module wrapper live in {@link HfAssemblerParts}. */
}
