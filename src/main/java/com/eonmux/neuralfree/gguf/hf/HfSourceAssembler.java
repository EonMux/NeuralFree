package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
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
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.VllmWeightsMapper;
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
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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
 *       ({@link #requant});</li>
 *   <li>{@link #metadata}: every metadata key;</li>
 *   <li>{@link #validate}: the view's invariants;</li>
 *   <li>{@link #layout}: the virtual tensor-data layout.</li>
 * </ol>
 * {@link #assemble} runs them all.
 */
public final class HfSourceAssembler {

    private static final Logger LOG = LogManager.getLogger(HfSourceAssembler.class);

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
        private final Map<String, String> checkpointNames;

        Filtered(HfLoadContext ctx, HfTensorIndex guessIndex, List<String> droppedPrefixes,
                 List<KvScaleCollector.Scale> kvScales, Map<String, String> checkpointNames) {
            this.ctx = ctx;
            this.guessIndex = guessIndex;
            this.droppedPrefixes = droppedPrefixes;
            this.kvScales = kvScales;
            this.checkpointNames = checkpointNames;
        }

        /** The load with the filtered index and the final block count. */
        public HfLoadContext ctx() { return ctx; }

        /** The stored tensors under converter names, before the quantization rewrite (for {@code --outtype auto}). */
        public HfTensorIndex guessIndex() { return guessIndex; }

        public List<String> droppedPrefixes() { return droppedPrefixes; }

        /** The KV-cache scales of the kept layers. */
        public List<KvScaleCollector.Scale> kvScales() { return kvScales; }

        /** The checkpoint name of each kept tensor, by its converter name (unmodifiable). */
        public Map<String, String> checkpointNames() { return checkpointNames; }
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
        private final Map<String, List<String>> added;

        Planned(HfLoadContext ctx, TensorPlan plan, Map<String, List<String>> added) {
            this.ctx = ctx;
            this.plan = plan;
            Map<String, List<String>> copy = new LinkedHashMap<>();
            for (Map.Entry<String, List<String>> e : added.entrySet()) {
                copy.put(e.getKey(), Collections.unmodifiableList(new ArrayList<>(e.getValue())));
            }
            this.added = Collections.unmodifiableMap(copy);
        }

        public HfLoadContext ctx() { return ctx; }

        public TensorPlan plan() { return plan; }

        /** The GGUF tensors each checkpoint tensor planned, by its filtered name (unmodifiable). */
        public Map<String, List<String>> added() { return added; }
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
        ModelSourceOptions effective = onlineDefault(c.quant(), o);
        CheckpointContainer ckpt = checkpoint(path, c, effective);
        HfResources resources = new HfResources();
        try {
            HfLoadContext ctx = index(resolve(c, ckpt, kind, effective, resources, target));
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
     * The options of the load, with its online quantization resolved against the checkpoint's quantization
     * config, so the requantization seam, the file type and every step after see the quantized load:
     * <ul>
     *   <li>{@code --quant-online mxfp4} or {@code mxfp8} on a checkpoint stored in that method names the
     *       checkpoint's own quantization and adds none, as vLLM's deferred shorthands, unless a
     *       {@code --quantization-config} is given;</li>
     *   <li>without {@code --quant-online}, a checkpoint whose quant_method is one of vLLM's online shorthands
     *       ({@link HfQuantConfig#onlineTarget()}) quantizes with it, and one whose quant_method is
     *       {@code online} with its own linear/moe/ignore/targets ({@link HfQuantConfig#onlineConfig()});</li>
     *   <li>{@code --quant-online online} quantizes with {@code --quantization-config}, else the checkpoint's online
     *       config; a {@code --quantization-config} without {@code --quant-online} is the {@code online} method
     *       (over the checkpoint's shorthand or online config, when it has one).</li>
     * </ul>
     *
     * @throws GgufFormatException when the result is no usable configuration ({@link OnlineQuantConfig#effective})
     */
    static ModelSourceOptions onlineDefault(HfQuantConfig q, ModelSourceOptions o) {
        OnlineQuant flag = o.onlineQuant();
        OnlineQuantConfig user = o.quantConfig();
        HfQuantConfig ckpt = q == null ? HfQuantConfig.none() : q;
        if (user == null && (flag == OnlineQuant.MXFP4 && ckpt.method() == HfQuantMethod.MXFP4
                || flag == OnlineQuant.MXFP8 && ckpt.method() == HfQuantMethod.MXFP8)) {
            LOG.info("--quant-online {} names the checkpoint's own quant_method '{}'; no online quantization is"
                    + " added", flag.name().toLowerCase(Locale.ROOT), ckpt.methodName());
            return o.toBuilder().onlineQuant(OnlineQuant.NONE).build();
        }
        OnlineQuant resolvedFlag = flag;
        OnlineQuantConfig resolvedConfig = user;
        try {
            if (flag == OnlineQuant.NONE && ckpt.onlineTarget() != null) {
                resolvedFlag = ckpt.onlineTarget();
                LOG.info("quant_method '{}' quantizes at load as {}", ckpt.methodName(),
                        resolvedFlag.name().toLowerCase(Locale.ROOT));
            } else if ((flag == OnlineQuant.NONE || flag == OnlineQuant.ONLINE) && ckpt.onlineConfig() != null) {
                resolvedFlag = OnlineQuant.ONLINE;
                resolvedConfig = user == null ? ckpt.onlineConfig() : user.over(ckpt.onlineConfig());
                LOG.info("quant_method 'online' quantizes at load as its quantization_config selects");
            } else if (flag == OnlineQuant.NONE && user != null) {
                resolvedFlag = OnlineQuant.ONLINE;
            }
            if (resolvedFlag == OnlineQuant.NONE) {
                return o;
            }
            OnlineQuantConfig.effective(resolvedFlag, resolvedConfig);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(ModelSourceOptions.PROP_ONLINE_QUANT + "="
                    + resolvedFlag.name().toLowerCase(Locale.ROOT) + (resolvedConfig == null ? ""
                    : " with " + ModelSourceOptions.PROP_QUANT_CONFIG) + ": " + e.getMessage(), e);
        }
        return o.toBuilder().onlineQuant(resolvedFlag).quantConfig(resolvedConfig).build();
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
        CheckpointContainer ckpt = CheckpointContainers.open(path, c.dir(), c.config().mistralNative(),
                c.config());
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
        // a projector checkpoint has no text blocks: --no-nextn leaves it alone, the two draft exports do not apply
        if (kind == HfArchEntry.Kind.MMPROJ && (o.mtpOnly() || o.dspark())) {
            throw new GgufFormatException((o.dspark() ? "--dspark" : "--mtp")
                    + " selects a text model's tensors; a projector export has none");
        }
        HfArchEntry entry = kind == HfArchEntry.Kind.MMPROJ ? projectorEntry(c.config())
                : textEntry(c.config(), o);
        Arch arch = entry.arch().resolve(c.config(), index);
        requireSupported(c.config(), entry, arch);
        HfConfig fixed = entry.configFixup().apply(c.config());
        return HfLoadContext.builder(fixed, entry, arch).options(o).checkpoint(ckpt).tokenizerFiles(c.tokenizer())
                .index(index).resources(resources).targetPath(target).build();
    }

    /**
     * The text entry of {@code cfg}: the DSpark draft the target's architecture selects under {@code --dspark}, else
     * the target's own entry. {@code --mtp} and {@code --no-nextn} are refused for a class whose converter does not
     * support the MTP export ({@code supports_mtp_export}), and {@code --mtp} for a checkpoint without the MTP head its
     * class requires.
     *
     * @throws GgufFormatException with the converter's message for an unsupported target
     */
    private static HfArchEntry textEntry(HfConfig cfg, ModelSourceOptions o) {
        if (o.dspark()) {
            return HfArchRegistry.resolveDspark(cfg);
        }
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        if (o.mtpOnly() || o.draftMode() == ModelSourceOptions.DraftMode.NO_NEXTN) {
            if (!entry.supportsMtpExport()) {
                throw new GgufFormatException("--mtp / --no-nextn are not supported for " + cfg.architecture());
            }
            if (o.mtpOnly() && entry.mtpOnlyGuard() != null && !entry.mtpOnlyGuard().test(cfg)) {
                throw new GgufFormatException("--mtp was requested, but this model does not contain a supported"
                        + " MTP head");
            }
        }
        return entry;
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
        int base = ctx.blockCount();
        // --mtp keeps only the NextN / MTP tensors and the shared root tensors the class names; the block count
        // stays the extended one, as the converter's mtp_only classes keep it
        boolean mtpOnly = kind == HfArchEntry.Kind.TEXT && ctx.options().mtpOnly();
        if (mtpOnly) {
            int main = ctx.config().blockCount();
            Set<String> rootKeep = ctx.entry().mtpOnlyRootKeep();
            index = HfAssemblerParts.MtpFilter.only(index, rootKeep, main);
            guess = HfAssemblerParts.MtpFilter.only(guess, rootKeep, main);
        }
        // --no-mtp drops text-model MTP layers; a projector checkpoint has no text blocks to drop past
        boolean dropMtp = kind == HfArchEntry.Kind.TEXT && !ctx.options().includeMtp();
        if (dropMtp) {
            base = Math.min(ctx.blockCount(), ctx.config().blockCount());
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
            if (filtered == null || (dropMtp && HfAssemblerParts.MtpFilter.isMtp(filtered, base))
                    || (mtpOnly && !HfAssemblerParts.MtpFilter.keepForMtpOnly(filtered,
                    ctx.entry().mtpOnlyRootKeep(), ctx.config().blockCount()))) {
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
                Collections.unmodifiableList(new ArrayList<>(scales.values())), kept.origins());
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
        HfOnlineRequant requant = requant(ctx, policy, f.checkpointNames());
        HfAssemblerParts.RecordingSink sink = new HfAssemblerParts.RecordingSink(b, requant);
        HfTensorIndex index = ctx.index();
        for (String name : index.names()) {
            if (!b.isConsumed(name)) {
                sink.begin(name);
                ctx.entry().tensors().apply(ctx, name, index.get(name), sink);
            }
        }
        sink.flush();
        for (KvScaleCollector.Scale s : f.kvScales()) {
            b.add(s.ggufName(), s.source());
        }
        ctx.entry().extra().generate(ctx, b);
        TensorPlan plan = b.build();
        Map<String, Boolean> prec = HfAssemblerParts.PrecA4.flags(ctx, moduleArgs, sink.added(), plan);
        return new Planned(prec.isEmpty() ? ctx : ctx.withPrecA4(prec), plan, sink.added());
    }

    /**
     * Step 8a: the plan's online-quantization seam. {@code --quant-online} and the checkpoint's own online
     * quantization configuration (the {@link #onlineDefault} resolution) requantize float matmul weights while
     * planning, after the plan's tensors are decided and before {@link #metadata} types them: every tensor the
     * hooks plan ({@link RequantPolicy#eligible(String, Arch, boolean, GGMLType, long[], GGMLType)},
     * llama-quant.cpp:302-421)
     * from a plain float source is planned from a producer that requantizes its stored floats into the option's
     * quantizer, so the planned type, the served bytes and the recomputed {@code general.file_type} are the
     * quantized ones; an already-quantized checkpoint's tensors and {@code NONE} requantize nothing. vLLM's
     * methods select layers by their quantization config ({@link OnlineQuantConfig}) and quantize each fused
     * projection with shared scales when the pass ends. The decision runs on the plan's add-time sink because the
     * planned type it reads is the policy's; {@link OnlineQuantizers} is the extension point of the quantizers.
     *
     * vLLM matches its configuration's patterns against its own layer names: the checkpoint names
     * ({@code checkpointNames}, by converter name) through the architecture's {@link VllmWeightsMapper}.
     *
     * @return the requantization the adds of {@code ctx}'s planning pass apply, null for none
     * @throws GgufFormatException when the options carry no usable online quantization config
     */
    static HfOnlineRequant requant(HfLoadContext ctx, DtypePolicy policy, Map<String, String> checkpointNames) {
        ModelSourceOptions o = ctx.options();
        if (o.onlineQuant() == OnlineQuant.NONE && o.quantConfig() == null) {
            return null;
        }
        OnlineQuantConfig config;
        try {
            config = OnlineQuantConfig.effective(o.onlineQuant(), o.quantConfig());
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(ModelSourceOptions.PROP_ONLINE_QUANT + "="
                    + o.onlineQuant().name().toLowerCase(Locale.ROOT) + ": " + e.getMessage(), e);
        }
        VllmWeightsMapper mapper = VllmWeightsMapper.of(vllmArchitecture(ctx.config()));
        Set<String> stored = new HashSet<>(checkpointNames.values());
        return new HfOnlineRequant(ctx.entry().forceType(), policy, ctx.arch(), o.quantizeOutputTensor(),
                o.onlineQuant(), config, stored::contains, name -> checkpointNames.getOrDefault(name, name), mapper);
    }

    /**
     * The architecture vLLM serves the checkpoint as: the root {@code architectures[0]} of {@code config.json}
     * (vLLM's {@code ModelConfig.architectures}), else the converter's architecture.
     */
    static String vllmArchitecture(HfConfig cfg) {
        Object arches = cfg.raw().raw("architectures");
        if (arches instanceof List && !((List<?>) arches).isEmpty() && ((List<?>) arches).get(0) instanceof String) {
            return (String) ((List<?>) arches).get(0);
        }
        return cfg.architecture();
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
        kv.set(GgufKeys.GENERAL_FILE_TYPE, ctx.options().onlineQuant() == OnlineQuant.NONE
                ? HfFileType.forOutType(ctx.options().outType(), plan) : HfFileType.guess(plan));
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
     * The projector entry of {@code cfg}.
     *
     * @throws GgufFormatException when no projector entry is registered for the checkpoint
     */
    static HfArchEntry projectorEntry(HfConfig cfg) {
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
    }

    /**
     * Refuses an architecture NeuralFree does not run. {@link Arch#CLIP} is never a text model: a {@link
     * HfArchEntry.Kind#TEXT} entry is a dual encoder ({@code CLIPModel}, {@code SiglipModel}), whose two-tower file
     * the CLIP dual encoder serves, and a {@link HfArchEntry.Kind#MMPROJ} entry is a projector view, whose file the
     * projector loaders serve; the language-model path refuses the architecture when it opens the file.
     */
    private static void requireSupported(HfConfig cfg, HfArchEntry entry, Arch arch) {
        if (arch == null || arch == Arch.UNKNOWN) {
            throw new GgufFormatException("Hugging Face architecture '" + cfg.architecture()
                    + "' resolves to no NeuralFree architecture for this checkpoint");
        }
        if (arch == Arch.CLIP) {
            return;
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
     * {@code neuralfree.act_quant.scale_ub}: the fbgemm_fp8 {@code activation_scale_ub}
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

    /** The {@code --no-mtp} drop and the online requantization decisions, the recording sink, the {@code prec_a4}
     *  flags and the dense-module wrapper live in {@link HfAssemblerParts}. */
}
