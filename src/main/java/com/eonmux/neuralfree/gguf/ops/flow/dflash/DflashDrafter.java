package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.TreeDraftModel;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.watermark.DraftWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Block-diffusion speculative drafter for dflash draft models (DFlash, DFlash2, DSpark, LiLiCorr and the
 * DeepSeek-V4 DSpark backbone), following llama.cpp's {@code common_speculative_impl_draft_dflash}
 * (common/speculative.cpp:981-1429) and models/dflash.cpp.
 *
 * <p>Protocol:
 * <ul>
 *   <li>{@link #bindTarget} installs a {@link HiddenTap} on the target that gathers the inputs of the
 *       {@code target_layers}. After every target forward call those rows are fused by the encoder
 *       ({@code rms(fc . features) * enc.output_norm}, dflash.cpp:856-858) and injected into the draft KV cache at
 *       the positions the call appended, {@code nPast - nRows ..} (speculative.cpp process()).</li>
 *   <li>{@link #draft} runs the noise block {@code [lastToken, mask, ...]} at positions {@code nPast ..} (the
 *       position the sampled token will take) and reads the draft: greedy top tokens with the {@code p_min} stop
 *       (DFlash), confidence-truncated Markov-biased tokens (DSpark), or the lattice walk (DFlash2, LiLiCorr);
 *       drafts shorter than {@code n_min} are dropped (speculative.cpp draft()). The block's cache rows are removed
 *       again, as the server's {@code seq_rm} after drafting.</li>
 *   <li>Rows of rejected tokens are dropped by position: every injection and every draft first forgets the cached
 *       positions at or after its start, which equals llama.cpp's {@code seq_rm(pos, -1)} after acceptance, so
 *       {@link #accept} keeps no state (speculative.cpp:1424-1426). A generation over a target's reused prompt
 *       prefix ({@link #beginGeneration(int)}) keeps the cache rows of the shared positions and drops the rest, so
 *       the draft attends over the same features it was built from.</li>
 * </ul>
 * The drafter runs on the CPU over memory-mapped weights, or with a {@link DflashDevice} on the target's GPU
 * ({@link GpuDflashBackbone}: the encoder, the stack and its KV cache on the device, as llama.cpp's draft context on
 * the target's backend), for every draft {@link DflashDevice#refusal} admits. On the device every readout reads only
 * what it consumes ({@code --spec-draft-backend-sampling}): a plain DFlash draft the greedy id of every block row,
 * with {@code p_min} and a LiLiCorr draft the top candidates, softmax denominators and {@code t_embd} rows
 * ({@code LLM_Row_TopK}), a DSpark draft the chain of Markov-biased argmax ids with the {@code t_embd} rows its
 * confidence head reads, and a DFlash2 draft the selector's chosen ids (the lattice scored on the device, its
 * probabilities read back only under a {@code p_min}); a non-ascending {@code d2t} keeps the block's full rows and
 * the host readout. The target's feature rows arrive through a device hidden tap
 * ({@link DflashDeviceFeed}) when the target streams them on the drafter's device, else through the host tap.
 * <p>With the tree option ({@link Options#withTree}, {@code neuralfree.draft.tree}) the drafter also implements
 * {@link TreeDraftModel}: one block forward serves a best-first draft tree over every slot's top candidates
 * ({@link DflashTreeBuilder}, SpecPilot's {@code ddtree}), which the target verifies in one forward over its
 * page-shared sequences; the drafter's single selector walk remains the one-branch case. The tree option requires
 * the target's memory to be paged with several sequences ({@link #branches()}), and a watermarked request refuses
 * it: the tree cannot carry a watermark key.</p>
 *
 * Not thread-safe: the target's forward thread drives the tap and must also call {@link #draft}.
 */
public final class DflashDrafter implements DraftModel, DraftProbs, TreeDraftModel {

    private static final Logger LOGGER = LogManager.getLogger(DflashDrafter.class);

    /** The draft-side watermark of the current generation, or null for the ordinary greedy readout. */
    private DraftWatermarker draftWatermark;
    private SplittableRandom draftRng;
    private float draftTemperature = 1f;
    /** The proposal distributions of the last draft's emitted positions, by position. */
    private final List<float[]> draftProposals = new ArrayList<>();
    /** The tokens the last draft emitted. */
    private int draftCount;
    /** Suspends the feature injection for one target forward that computes no committed rows. */
    private boolean tapSuspended;

    /** Rows per encoder pass of an injection, llama.cpp's default draft {@code n_ubatch}. */
    static final int INJECT_CHUNK = 512;

    /**
     * Drafting options ({@code common_params_speculative_draft} n_min / p_min), the draft KV type, whether a
     * quantized draft cache uses the Hadamard rotation ({@code neuralfree.kv.rotation}, llama.cpp's
     * {@code attn_rot}) and how the draft's activation quantization is applied ({@link ActivationQuant}), plus the
     * single-pass draft-tree option and its score temperature. Immutable.
     */
    public static final class Options {
        private static final Options DEFAULTS = new Options(0, 0.0f, KvDType.F16, true, ActivationQuant.WEIGHT_ONLY,
                false, 1.0);

        private final int nMin;
        private final float pMin;
        private final KvDType kvType;
        private final boolean kvRotation;
        private final ActivationQuant activationQuant;
        private final boolean treeVerify;
        private final double treeTemp;

        private Options(int nMin, float pMin, KvDType kvType, boolean kvRotation, ActivationQuant activationQuant,
                        boolean treeVerify, double treeTemp) {
            this.nMin = nMin;
            this.pMin = pMin;
            this.kvType = kvType;
            this.kvRotation = kvRotation;
            this.activationQuant = activationQuant;
            this.treeVerify = treeVerify;
            this.treeTemp = treeTemp;
        }

        /**
         * {@code n_min = 0}, {@code p_min = 0}, F16 cache, rotation enabled (llama.cpp defaults), weight-only
         * activations.
         */
        public static Options defaults() {
            return DEFAULTS;
        }

        /**
         * Options with the KV rotation enabled.
         *
         * @throws IllegalArgumentException for a negative {@code nMin} or a {@code pMin} outside {@code [0, 1]}
         */
        public static Options of(int nMin, float pMin, KvDType kvType) {
            return of(nMin, pMin, kvType, true);
        }

        /**
         * @param kvRotation rotate the rows of a quantized draft cache (no effect on F16 / F32 caches)
         * @throws IllegalArgumentException for a negative {@code nMin} or a {@code pMin} outside {@code [0, 1]}
         */
        public static Options of(int nMin, float pMin, KvDType kvType, boolean kvRotation) {
            if (nMin < 0) {
                throw new IllegalArgumentException("draft n_min must be non-negative, got " + nMin);
            }
            if (!(pMin >= 0.0f && pMin <= 1.0f)) {
                throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
            }
            return new Options(nMin, pMin, Objects.requireNonNull(kvType, "kvType"), kvRotation,
                    ActivationQuant.WEIGHT_ONLY, false, 1.0);
        }

        /** These options with the draft's activation quantization applied as {@code mode}. */
        public Options withActivationQuant(ActivationQuant mode) {
            return new Options(nMin, pMin, kvType, kvRotation, Objects.requireNonNull(mode, "mode"),
                    treeVerify, treeTemp);
        }

        /**
         * These options with the single-pass draft-tree option: the drafter proposes a best-first draft tree over
         * the block's per-slot top candidates (SpecPilot's {@code ddtree}, arXiv 2611.02501) that the target
         * verifies in one forward over its page-shared sequences, instead of its single selector walk. The
         * temperature {@code treeTemp} scales the builder's path-conditional scores (SpecPilot's
         * {@code TREE_TEMP}); the plain walk stays the one-branch case.
         *
         * @throws IllegalArgumentException for a non-positive or non-finite {@code treeTemp}
         */
        public Options withTree(boolean treeVerify, double treeTemp) {
            if (!(treeTemp > 0.0) || !Double.isFinite(treeTemp)) {
                throw new IllegalArgumentException("draft tree temperature must be finite and positive, got "
                        + treeTemp);
            }
            return new Options(nMin, pMin, kvType, kvRotation, activationQuant, treeVerify, treeTemp);
        }

        public int nMin() { return nMin; }
        public float pMin() { return pMin; }
        public KvDType kvType() { return kvType; }
        public boolean kvRotation() { return kvRotation; }
        /** How the draft's activation quantization is applied ({@link ActivationQuant#WEIGHT_ONLY} by default). */
        public ActivationQuant activationQuant() { return activationQuant; }
        /** Whether the drafter proposes a single-pass draft tree ({@link #withTree}, off by default). */
        public boolean treeVerify() { return treeVerify; }
        /** The tree builder's score temperature ({@link #withTree}, 1.0 by default). */
        public double treeTemp() { return treeTemp; }

        @Override
        public String toString() {
            return "DflashDrafter.Options{nMin=" + nMin + ", pMin=" + pMin + ", kv=" + kvType
                    + (kvRotation ? "" : ", no rotation")
                    + (activationQuant == ActivationQuant.WEIGHT_ONLY ? "" : ", actQuant="
                            + activationQuant.propertyValue())
                    + (treeVerify ? ", tree(temp=" + treeTemp + ")" : "") + '}';
        }
    }

    private final DflashSpec spec;
    private final ModelHparams hp;
    private final CpuWeights draft;
    private final boolean ownsDraft;
    private final Options options;
    private final DflashMatrix tokEmbd;
    private final DflashMatrix output;
    private final DflashMatrix fc;
    private final float[] encNorm;
    private final int targetEmbd;
    private final int[] d2t;
    private final DflashMarkovHead markov;
    private final DflashMatrix selectorPrev;
    private final DflashMatrix selectorNext;
    private final DflashMatrix selectorHidden;
    private final DflashLilicorr lilicorr;
    private final float[] lilicorrRows;
    private final int[] lilicorrPos;
    private final DflashDevice device;
    private final boolean d2tAscending;

    private ForwardBackend backend;
    private final List<ForwardBackend> viewBackends = new ArrayList<>();
    private DflashCache cache;
    private DflashStack stack;
    private GpuDflashBackbone deviceStack;
    private GpuDflashKvCache deviceCache;
    private DflashTargetFeatures features;
    private DflashDeviceFeed deviceFeed;
    /** The target rows of the suspended verify forward: on the host, or in {@code captureBuffer} on the device. */
    private float[] capturedRows;
    private GpuBuffer captureBuffer;
    private int capturedCount;

    /**
     * Drafter over an opened draft model.
     *
     * @param draft     the draft model's weights
     * @param ownsDraft close {@code draft} with this drafter
     * @param target    the target model's weights, for the shared {@code token_embd} / {@code output}; may be
     *                  null when the draft carries both
     * @param device    the target's GPU to draft on, or null to draft on the host
     * @throws IllegalArgumentException      when the draft is not a supported dflash model or needs a missing tensor
     * @throws UnsupportedOperationException when a device is given for a draft {@link DflashDevice#refusal} refuses
     */
    public DflashDrafter(ModelHparams hp, GgufKv kv, CpuWeights draft, boolean ownsDraft, CpuWeights target,
                         Options options, DflashDevice device) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.ownsDraft = ownsDraft;
        this.options = Objects.requireNonNull(options, "options");
        this.spec = DflashSpec.of(hp, kv, draft);
        final int nEmbd = spec.nEmbd();

        this.tokEmbd = firstOf(TensorNames.weight(TensorNames.TOKEN_EMBD), draft, target, null);
        if (tokEmbd.cols() != nEmbd || tokEmbd.rows() < spec.nVocab()) {
            throw new IllegalArgumentException("dflash drafter: token embeddings " + tokEmbd + " do not cover "
                    + spec.nVocab() + " tokens of width " + nEmbd);
        }
        this.output = firstOf(TensorNames.weight(TensorNames.OUTPUT), draft, target,
                TensorNames.weight(TensorNames.TOKEN_EMBD)).expect(spec.nVocabDraft(), nEmbd);
        this.fc = DflashMatrix.require(draft, TensorNames.weight(TensorNames.FC));
        if (fc.rows() != nEmbd || fc.cols() % spec.nTargets() != 0) {
            throw new IllegalArgumentException("dflash drafter: " + fc + " does not map " + spec.nTargets()
                    + " target layers to width " + nEmbd);
        }
        this.targetEmbd = fc.cols() / spec.nTargets();
        this.encNorm = DflashBackbone.vector(draft.floats(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)), nEmbd,
                TensorNames.ENC_OUTPUT_NORM);
        this.d2t = spec.hasD2t() ? readD2t(draft, spec.nVocab()) : null;
        this.d2tAscending = d2t == null || ascending(d2t);
        String refusal = device == null ? null : DflashDevice.refusal(spec, options.kvType());
        if (refusal != null) {
            throw new UnsupportedOperationException("the dflash drafter cannot run on the device: " + refusal);
        }
        this.device = device;

        if (spec.dspark() && options.pMin() > 0.0f && !spec.confidenceHeadMeta()) {
            throw new IllegalArgumentException("DSpark draft has no confidence head: please set the draft p_min to 0");
        }
        this.markov = spec.dspark() ? new DflashMarkovHead(spec, draft, d2t) : null;
        if (markov != null && options.pMin() > 0.0f && !markov.hasConfidence()) {
            throw new IllegalArgumentException("DSpark draft declares a confidence head but has no "
                    + TensorNames.weight(TensorNames.DSPARK_CONF_PROJ) + ": please set the draft p_min to 0");
        }
        if (spec.dflash2()) {
            final int rank = spec.selectorRank();
            this.selectorPrev = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_PREV))
                    .expect(spec.nVocab(), rank);
            this.selectorNext = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_NEXT))
                    .expect(spec.nVocab(), rank);
            this.selectorHidden = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_HIDDEN))
                    .expect(rank, nEmbd);
        } else {
            this.selectorPrev = null;
            this.selectorNext = null;
            this.selectorHidden = null;
        }
        if (spec.lilicorr()) {
            this.lilicorr = new DflashLilicorr(spec, draft, tokEmbd);
            this.lilicorrRows = new float[(spec.blockSize() + 1) * nEmbd];
            this.lilicorrPos = new int[spec.blockSize() + 1];
            Arrays.fill(lilicorrPos, -1);
        } else {
            this.lilicorr = null;
            this.lilicorrRows = null;
            this.lilicorrPos = null;
        }
    }

    /**
     * Opens a dflash draft model drafting on the host; the drafter owns and closes its weights.
     *
     * @param target the target model's weights for the shared tensors, or null when the draft carries them
     * @throws IOException              when the draft's tensor data cannot be mapped
     * @throws IllegalArgumentException when the file is not a supported dflash draft
     */
    public static DflashDrafter open(ModelSource draft, CpuWeights target, Options options) throws IOException {
        return open(draft, target, options, null);
    }

    /**
     * Opens a dflash draft model; the drafter owns and closes its host weights.
     *
     * @param target the target model's weights for the shared tensors, or null when the draft carries them
     * @param device the target's GPU to draft on, or null to draft on the host
     * @throws IOException                   when the draft's tensor data cannot be mapped
     * @throws IllegalArgumentException      when the file is not a supported dflash draft
     * @throws UnsupportedOperationException when a device is given for a draft {@link DflashDevice#refusal} refuses
     */
    public static DflashDrafter open(ModelSource draft, CpuWeights target, Options options, DflashDevice device)
            throws IOException {
        Objects.requireNonNull(draft, "draft");
        Objects.requireNonNull(options, "options");
        GgufKv kv = GgufKv.of(draft);
        ModelHparams hp = HparamsLoader.load(kv);
        TensorCatalog catalog = TensorCatalog.of(draft);
        CpuWeights weights = CpuWeights.open(draft, catalog,
                InputTransformLoader.load(draft, kv, catalog, options.activationQuant()));
        try {
            return new DflashDrafter(hp, kv, weights, true, target, options, device);
        } catch (RuntimeException e) {
            try {
                weights.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /** The draft model's description. */
    public DflashSpec spec() {
        return spec;
    }

    /** Whether the drafter drafts on the target's GPU. */
    public boolean onDevice() {
        return device != null;
    }

    /**
     * The spec of a dflash draft model, read without binding a drafter; the draft's tensors are mapped only while it
     * is read.
     *
     * @throws IOException              when the draft's tensor data cannot be mapped
     * @throws IllegalArgumentException when the file is not a supported dflash draft
     */
    public static DflashSpec readSpec(ModelSource draft) throws IOException {
        Objects.requireNonNull(draft, "draft");
        GgufKv kv = GgufKv.of(draft);
        ModelHparams hp = HparamsLoader.load(kv);
        try (CpuWeights weights = CpuWeights.open(draft)) {
            return DflashSpec.of(hp, kv, weights);
        }
    }

    /**
     * Bytes of the host KV cache a drafter of {@code spec} allocates when it binds to a target with a context of
     * {@code nCtx} tokens and stores its rows as {@code kvType}: the context plus one block per full-attention
     * layer, a ring per sliding-window layer.
     *
     * @throws IllegalArgumentException when the context is not positive
     */
    public static long cacheBytes(DflashSpec spec, int nCtx, KvDType kvType) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(kvType, "kvType");
        if (nCtx <= 0) {
            throw new IllegalArgumentException("dflash cache of a context of " + nCtx + " tokens");
        }
        return DflashKvCache.bytes(cacheCapacity(spec, nCtx), spec.blockSize(), spec.nSwa(), swaLayers(spec),
                spec.dsv4(), kvHeads(spec), spec.headDim(), kvType);
    }

    /** Positions a full-attention layer of the cache holds: the target context plus one noise block. */
    private static int cacheCapacity(DflashSpec spec, int nCtx) {
        return Math.addExact(nCtx, spec.blockSize());
    }

    private static boolean[] swaLayers(DflashSpec spec) {
        boolean[] swa = new boolean[spec.nLayer()];
        for (int il = 0; il < swa.length; il++) {
            swa[il] = spec.isSwa(il);
        }
        return swa;
    }

    private static int[] kvHeads(DflashSpec spec) {
        int[] heads = new int[spec.nLayer()];
        for (int il = 0; il < heads.length; il++) {
            heads[il] = spec.nHeadKv(il);
        }
        return heads;
    }

    // ------------------------------------------------------------------ DraftModel

    @Override
    public void bindTarget(EngineModel target) {
        Objects.requireNonNull(target, "target");
        if (backend != null) {
            throw new IllegalStateException("dflash drafter is already bound to a target");
        }
        ModelHparams thp = target.spec().hparams();
        if (thp.nEmbd() != targetEmbd) {
            throw new IllegalArgumentException("dflash draft fc expects target width " + targetEmbd + ", target "
                    + thp.arch().ggufName() + " has " + thp.nEmbd());
        }
        if (thp.nVocab() != spec.nVocab()) {
            throw new IllegalArgumentException("dflash draft vocabulary " + spec.nVocab() + " differs from the target's "
                    + thp.nVocab());
        }
        for (int layer : spec.targetLayers()) {
            if (layer < 0 || layer > thp.nLayer()) {
                throw new IllegalArgumentException("dflash target layer " + layer + " is outside the target's tap points"
                        + " [0, " + thp.nLayer() + "]");
            }
        }
        ForwardBackend fb = Objects.requireNonNull(target.forwardBackend(), "target forward backend");
        final int nCtx = fb.memory().sequences().sequence(0).nCtx();
        if (device != null) {
            bindDevice(nCtx);
        } else {
            DflashKvCache kvCache = new DflashKvCache(cacheCapacity(spec, nCtx), spec.blockSize(), spec.nSwa(),
                    swaLayers(spec), spec.causal(), spec.dsv4(), kvHeads(spec), spec.headDim(), options.kvType(),
                    options.kvRotation());
            this.stack = spec.dsv4()
                    ? new DflashDsv4Backbone(spec, hp.ropeType(), draft, tokEmbd, output, kvCache)
                    : new DflashBackbone(spec, hp, draft, tokEmbd, output, kvCache, nCtx);
            this.cache = kvCache;
        }
        DflashDeviceFeed feed = deviceTap(fb, target);
        if (feed != null) {
            this.deviceFeed = feed;
            fb.setHiddenTap(feed);
            LOGGER.info("dflash drafter reads the target's rows of " + spec.targetLayers().length + " layers on"
                    + " the target's device");
        } else {
            this.features = new DflashTargetFeatures(spec.targetLayers(), targetEmbd, this::inject);
            fb.setHiddenTap(features);
            if (deviceStack != null) {
                LOGGER.info("dflash drafter reads the target's rows on the host: " + hostTapReason(fb, target));
            }
        }
        this.backend = fb;
    }

    /**
     * Installs the target tap on a second backend over the same memory as the target — the quantized view of
     * two-level verification ({@code neuralfree.draft.tlv}): the chain replays and the tree rounds it runs must
     * feed the drafter's cache exactly as the target's passes do, from the same features or device feed. The
     * first bound backend keeps its tap; {@link #close} clears both.
     *
     * @throws IllegalStateException when the view runs on a different device than the drafter's device stack
     */
    public void bindSecondBackend(ForwardBackend viewBackend) {
        Objects.requireNonNull(viewBackend, "viewBackend");
        requireBound();
        if (viewBackend == backend) {
            return;
        }
        if (deviceFeed != null) {
            if (!(viewBackend instanceof GpuForwardBackend)
                    || ((GpuForwardBackend) viewBackend).context().backend() != device.context().backend()) {
                throw new IllegalStateException("the dflash drafter reads the target's rows on the device, which"
                        + " a quantized view on another device cannot feed; run two-level verification with the"
                        + " view beside the target");
            }
            viewBackend.setHiddenTap(deviceFeed);
        } else {
            viewBackend.setHiddenTap(features);
        }
        viewBackends.add(viewBackend);
    }

    /**
     * The device feed of a device stack over {@code target}, or null when the call's rows reach the drafter on the
     * host: the target's rows carry several hyper-connection streams the host tap collapses, or the target's
     * forward backend reads its tap points on the host.
     */
    private DflashDeviceFeed deviceTap(ForwardBackend fb, EngineModel target) {
        if (deviceStack == null || streamWidth(target) != targetEmbd
                || !(fb instanceof GpuForwardBackend)
                || ((GpuForwardBackend) fb).context() != device.context()) {
            return null;
        }
        return new DflashDeviceFeed(spec.targetLayers(), targetEmbd, fb.memory().sequences().sequence(0),
                device.context().ops(), device.context().backend(), this::injectDevice);
    }

    /** Why a device stack's drafter reads the target's rows on the host, for the fallback log. */
    private static String hostTapReason(ForwardBackend fb, EngineModel target) {
        int nEmbd = target.spec().hparams().nEmbd();
        int width = streamWidth(target);
        if (width != nEmbd) {
            return "the target's rows carry " + (width / nEmbd) + " hyper-connection streams the host tap collapses";
        }
        return fb.getClass().getSimpleName() + " reads its tap points on the host";
    }

    /** Width of the target's residual stream at its last layer. */
    private static int streamWidth(EngineModel target) {
        List<LayerSpec> layers = target.spec().layers();
        return layers.get(layers.size() - 1).residual().streamWidth();
    }

    /**
     * The target's host record of its committed sequence 0: the paged memory's own record on a paged target (whose
     * paged steps extend it), the memory's contiguous record otherwise. The injection and the block runs align the
     * draft cache to this record's position.
     */
    private SequenceState targetSequence() {
        return backend.memory().sequences().sequence(0);
    }

    /** The device features of one injection chunk: the encoder and the K/V injection without an upload. */
    private void injectDevice(GpuBuffer feats, int nRows, int pos0) {
        if (tapSuspended) {
            captureDeviceRows(feats, nRows);
            return;
        }
        injectRows(null, feats, nRows, pos0);
    }

    /** The device cache and stack at the target's context; a failure releases what was allocated. */
    private void bindDevice(int nCtx) {
        GpuForwardContext ctx = device.context();
        int maxRows = Math.max(INJECT_CHUNK, spec.blockSize());
        GpuDflashKvCache kvCache = new GpuDflashKvCache(ctx.backend(), ctx.ops(), cacheCapacity(spec, nCtx),
                spec.blockSize(), spec.nSwa(), swaLayers(spec), spec.causal(), kvHeads(spec), spec.headDim(),
                options.kvType(), options.kvRotation(), maxRows);
        try {
            this.deviceStack = new GpuDflashBackbone(spec, hp, ctx, device.draft(), draft, tokEmbd, device.output(),
                    kvCache, fc.cols(), nCtx, topKWidth());
        } catch (RuntimeException e) {
            Closeables.close(kvCache, e);
            throw e;
        }
        this.stack = deviceStack;
        this.cache = kvCache;
        this.deviceCache = kvCache;
    }

    /**
     * Candidates per row the device top-k readout binds: the lattice width for a LiLiCorr or DFlash2 draft, the
     * sampler's ten for a plain DFlash draft with {@code p_min}; 0 when every device readout reads only ids
     * (greedy DFlash, the DSpark chain).
     */
    private int topKWidth() {
        boolean topK = spec.lilicorr() || spec.dflash2() || options.pMin() > 0.0f;
        return topK ? GpuDflashBackbone.topKWidth(spec) : 0;
    }

    @Override
    public int[] draft(int lastToken, int n) {
        return draft(lastToken, n, 1f);
    }

    @Override
    public int[] draft(int lastToken, int n, float temperature) {
        requireBound();
        draftProposals.clear();
        draftCount = 0;
        draftTemperature = temperature;
        if (lastToken < 0 || lastToken >= spec.nVocab()) {
            throw new IllegalArgumentException("token " + lastToken + " is outside the vocabulary of "
                    + spec.nVocab());
        }
        final int nMax = Math.min(n, spec.nDraftMax());
        final int nMin = Math.min(options.nMin(), spec.nDraftMax());
        if (nMax <= 0) {
            return new int[0];
        }
        final int pos0 = targetSequence().nPast();
        final int nBlock = spec.blockTokens(nMax);
        if (nBlock <= 0 || !cache.fits(pos0, nBlock) || (lilicorr != null && pos0 < 1)) {
            return new int[0];
        }
        int[] tokens = new int[nBlock];
        Arrays.fill(tokens, spec.maskTokenId());
        tokens[0] = lastToken;

        boolean watermarked = draftWatermark != null && temperature > 0f;
        if (watermarked) {
            draftWatermark.sync(targetSequence().textTokens(), lastToken);
            draftWatermark.beginRound();
        }
        // a keyed draw needs the block's logits, which only the host readout has
        int[] result = deviceStack != null && !watermarked
                ? draftDevice(tokens, pos0, nBlock, nMax, lastToken)
                : draftHost(tokens, pos0, nBlock, nMax, lastToken);
        if (result.length < nMin) {
            draftProposals.clear();
            return new int[0];
        }
        draftCount = result.length;
        return result;
    }

    /** The host round: the block over the host stack, then the readout over its logits. */
    private int[] draftHost(int[] tokens, int pos0, int nBlock, int nMax, int lastToken) {
        cache.trim(pos0);
        DflashStack.Block block;
        try {
            block = stack.forward(tokens, pos0);
        } finally {
            cache.trim(pos0);
        }
        return readout(block, pos0, nBlock, nMax, lastToken);
    }

    /**
     * The device round: the readout the draft runs on the device — the greedy ids, the top-k candidates, or the
     * DSpark chain — or, when the readout's readback would not equal the host's (a non-ascending {@code d2t}), the
     * block's full rows read back and the host readout over them.
     */
    private int[] draftDevice(int[] tokens, int pos0, int nBlock, int nMax, int lastToken) {
        if (deviceGreedy()) {
            int[] result = readDeviceGreedy(tokens, pos0, nBlock);
            return result;
        }
        if (!d2tAscending && !spec.dflash2()) {
            return draftHost(tokens, pos0, nBlock, nMax, lastToken);
        }
        cache.trim(pos0);
        try {
            if (spec.dflash2()) {
                return deviceStack.forwardSelector(tokens, pos0, lastToken, options.pMin());
            }
            if (spec.dspark()) {
                return readDsparkDevice(tokens, pos0, nBlock, lastToken);
            }
            int k = topKWidth();
            GpuDflashBackbone.TopKRows rows = deviceStack.forwardTopK(tokens, pos0, k);
            if (lilicorr != null) {
                DflashSelector.Lattice lattice = lilicorr.score(candidates(rows), rows.tEmbd,
                        lilicorrAnchor(pos0 - 1));
                return DflashSelector.walk(lattice, nMax, options.pMin());
            }
            return readGreedyTopK(rows, nBlock);
        } finally {
            cache.trim(pos0);
        }
    }

    /** The host readout over one block: the draft's tokens, stopping at {@code p_min}. */
    private int[] readout(DflashStack.Block block, int pos0, int nBlock, int nMax, int lastToken) {
        float[] logits = targetLogits(block);
        float[] conf = markov != null ? markov.apply(logits, block.tEmbd(), nBlock, lastToken) : null;

        int[] result;
        if (spec.dflash2()) {
            DflashSelector.Lattice lattice = DflashSelector.dflash2(logits, spec.nVocab(), block.tEmbd(),
                    spec.nEmbd(), nBlock, lastToken, spec.selectorTopK(), spec.graphBlockSize(), selectorPrev,
                    selectorNext, selectorHidden);
            result = DflashSelector.walk(lattice, Math.min(nBlock, lattice.nRows()) - 1, options.pMin());
        } else if (lilicorr != null) {
            DflashSelector.Lattice lattice = lilicorr.score(logits, block.tEmbd(), lilicorrAnchor(pos0 - 1));
            result = DflashSelector.walk(lattice, nMax, options.pMin());
        } else if (spec.dspark()) {
            result = readDspark(logits, options.pMin() > 0.0f ? conf : null, nBlock);
        } else {
            result = readGreedy(logits, nBlock);
        }
        return result;
    }

    @Override
    public boolean supportsWatermarking() {
        return !spec.dflash2() && lilicorr == null;
    }

    @Override
    public void watermark(WatermarkConfig config, int promptLength, long seed) {
        this.draftWatermark = DraftWatermarker.of(config);
        if (draftWatermark != null) {
            draftWatermark.reset(promptLength);
        }
        this.draftRng = draftWatermark == null ? null : new SplittableRandom(seed);
    }

    @Override
    public float[] draftProbs(int position) {
        if (position < 0 || position >= draftCount) {
            throw new IllegalStateException("no drafted token at position " + position);
        }
        float[] row = position < draftProposals.size() ? draftProposals.get(position) : null;
        return row == null ? null : row.clone();
    }

    /**
     * Records verification of the last draft. The cache needs no update here: rows of rejected tokens lie at or
     * after the target's next position and are dropped by the next injection or draft.
     *
     * @throws IllegalArgumentException for a negative count
     */
    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must be non-negative, got " + nAccepted);
        }
    }

    // ------------------------------------------------------------------ TreeDraftModel

    /**
     * The target's sequences the draft tree verifies over, or 0 without the tree option or on a target whose
     * memory is not paged: the decoder then keeps the drafter's single selector walk.
     */
    @Override
    public int branches() {
        if (!options.treeVerify() || backend == null) {
            return 0;
        }
        PagedSequences seqs = backend.memory().pagedSequences();
        return seqs != null ? seqs.maxSequences() : 0;
    }

    /** The draft tree verifies in one target forward over page-shared sequences. */
    @Override
    public boolean singlePassVerify() {
        return options.treeVerify();
    }

    /** No tree state to realign: the draft block's rows leave the cache by position, as {@link #accept}. */
    @Override
    public void acceptTree(int nAccepted) {
        accept(nAccepted);
    }

    @Override
    public void featureTapSuspended(boolean suspended) {
        this.tapSuspended = suspended;
    }

    /**
     * The draft tree of the round: one block forward over {@code [lastToken, mask, ...]}, then the best-first
     * builder over the block's per-slot candidates — the selector lattice's conditional scores (DFlash2,
     * LiLiCorr), the Markov-biased top-k of every row under the conditioning token (DSpark), or the plain top-k of
     * the rows' logits (DFlash). The one-branch tree is the drafter's selector walk.
     *
     * <p>Always reads the block back on the host: the builder scores candidates, which the device readouts read
     * back only partially. {@code rng} is unused — the builder's ranking is deterministic at any temperature.
     *
     * @throws UnsupportedOperationException on an active draft watermark, whose key the tree cannot carry
     */
    @Override
    public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                               SplittableRandom rng) {
        requireBound();
        draftProposals.clear();
        draftCount = 0;
        if (!options.treeVerify()) {
            return new DraftTree(new ArrayList<>());
        }
        if (draftWatermark != null) {
            throw new UnsupportedOperationException("a watermarked request cannot draft a dflash tree: the tree"
                    + " cannot draw under the watermark key (verify with neuralfree.draft.verify=match, or turn off"
                    + " neuralfree.draft.tree)");
        }
        if (lastToken < 0 || lastToken >= spec.nVocab()) {
            throw new IllegalArgumentException("token " + lastToken + " is outside the vocabulary of "
                    + spec.nVocab());
        }
        final int nMax = Math.min(maxTokens, spec.nDraftMax());
        if (nMax <= 0 || branches <= 0 || maxRows <= 0) {
            return new DraftTree(new ArrayList<>());
        }
        final int pos0 = targetSequence().nPast();
        final int nBlock = spec.blockTokens(nMax);
        if (nBlock <= 0 || !cache.fits(pos0, nBlock) || (lilicorr != null && pos0 < 1)) {
            return new DraftTree(new ArrayList<>());
        }
        int[] tokens = new int[nBlock];
        Arrays.fill(tokens, spec.maskTokenId());
        tokens[0] = lastToken;
        cache.trim(pos0);
        DflashStack.Block block;
        try {
            block = stack.forward(tokens, pos0);
        } finally {
            cache.trim(pos0);
        }
        DflashTreeBuilder.Table table = table(block, pos0, nBlock, nMax, lastToken);
        DraftTree proposal = DflashTreeBuilder.build(table, lastToken, nMax, branches, maxRows,
                (float) options.treeTemp());
        final int nMin = Math.min(options.nMin(), spec.nDraftMax());
        if (proposal.size() < nMin) {
            return new DraftTree(new ArrayList<>());
        }
        draftCount = proposal.size();
        return proposal;
    }

    @Override
    public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature, SplittableRandom rng) {
        return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
    }

    /** The per-slot candidate table of one block, by draft variant. */
    private DflashTreeBuilder.Table table(DflashStack.Block block, int pos0, int nBlock, int nMax, int lastToken) {
        float[] logits = targetLogits(block);
        if (spec.dflash2()) {
            DflashSelector.Lattice lattice = DflashSelector.dflash2(logits, spec.nVocab(), block.tEmbd(),
                    spec.nEmbd(), nBlock, lastToken, spec.selectorTopK(), spec.graphBlockSize(), selectorPrev,
                    selectorNext, selectorHidden);
            return new DflashTreeBuilder.LatticeTable(lattice, 1, Math.min(nBlock, lattice.nRows()) - 1);
        }
        if (lilicorr != null) {
            DflashSelector.Lattice lattice = lilicorr.score(logits, block.tEmbd(), lilicorrAnchor(pos0 - 1));
            return new DflashTreeBuilder.LatticeTable(lattice, 1, Math.min(nMax, lattice.nRows() - 1));
        }
        if (spec.dspark()) {
            int firstDepth = spec.sampleFromAnchor() ? 0 : 1;
            return new DflashTreeBuilder.DsparkTable(markov, logits, spec.nVocab(), firstDepth, nBlock - 1,
                    DflashSpec.SAMPLER_TOP_K);
        }
        int k = Math.min(DflashSpec.SAMPLER_TOP_K, spec.nVocab());
        int rows = nBlock - 1;
        int[] ids = new int[rows * k];
        float[] scores = new float[rows * k];
        for (int r = 0; r < rows; r++) {
            final int off = (r + 1) * spec.nVocab();
            CpuOps.topK(logits, off, spec.nVocab(), k, ids, r * k);
            for (int j = 0; j < k; j++) {
                scores[r * k + j] = logits[off + ids[r * k + j]];
            }
        }
        return new DflashTreeBuilder.TopKTable(1, rows, k, ids, scores);
    }

    /**
     * Starts a generation over the target's reused prefix: the draft cache keeps the injected rows of positions
     * {@code [0, reused)} (they hold the same target features), the rows after it leave, and the LiLiCorr anchor
     * slots of dropped positions are cleared. No position is re-evaluated: the tail's injection covers every new
     * position.
     */
    @Override
    public int beginGeneration(int reused) {
        if (reused < 0) {
            throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
        }
        if (backend == null) {
            return 0;
        }
        releaseCapture();
        if (features != null) {
            features.discard();
        }
        if (deviceFeed != null) {
            deviceFeed.drop();
        }
        if (cache != null) {
            cache.trim(reused);
        }
        if (lilicorrPos != null) {
            for (int slot = 0; slot < lilicorrPos.length; slot++) {
                if (lilicorrPos[slot] >= reused) {
                    lilicorrPos[slot] = -1;
                    Arrays.fill(lilicorrRows, slot * spec.nEmbd(), (slot + 1) * spec.nEmbd(), 0.0f);
                }
            }
        }
        return 0;
    }

    @Override
    public boolean readsTargetHidden() {
        return true;
    }

    @Override
    public void reset() {
        draftProposals.clear();
        draftCount = 0;
        releaseCapture();
        if (features != null) {
            features.discard();
        }
        if (deviceFeed != null) {
            deviceFeed.drop();
        }
        if (cache != null) {
            cache.clear();
        }
        if (lilicorrPos != null) {
            Arrays.fill(lilicorrPos, -1);
            Arrays.fill(lilicorrRows, 0.0f);
        }
    }

    @Override
    public void close() {
        if (backend != null) {
            backend.setHiddenTap(HiddenTap.NONE);
            backend = null;
        }
        for (ForwardBackend view : viewBackends) {
            view.setHiddenTap(HiddenTap.NONE);
        }
        viewBackends.clear();
        releaseCapture();
        RuntimeException failure = Closeables.close(deviceStack, null);
        failure = Closeables.close(deviceCache, failure);
        if (deviceFeed != null) {
            deviceFeed.close();
            deviceFeed = null;
        }
        deviceStack = null;
        deviceCache = null;
        if (ownsDraft) {
            try {
                draft.close();
            } catch (IOException e) {
                UncheckedIOException closeFailure = new UncheckedIOException("failed to close the dflash draft"
                        + " weights", e);
                if (failure != null) {
                    closeFailure.addSuppressed(failure);
                }
                throw closeFailure;
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ injection

    /** {@link DflashTargetFeatures.Sink}: fuses and injects the rows of one target forward call. */
    private void inject(float[] feats, int nRows) {
        if (tapSuspended) {
            captureHostRows(feats, nRows);
            return;
        }
        SequenceState seq = targetSequence();
        injectRows(feats, null, nRows, seq.nPast() - nRows);
    }

    /**
     * Fuses and injects {@code nRows} feature rows at {@code pos0}, from the host rows a target forward read back
     * ({@code hostRows}, the encoder's fc matmul runs on them) or from device rows ({@code deviceRows}, the
     * {@link DflashDeviceFeed}'s zipped chunk). The rows enter the draft KV cache at {@code pos0 ..} after the
     * cache trims there, and the LiLiCorr anchor keeps the fused rows of the chunk's last positions.
     */
    private void injectRows(float[] hostRows, GpuBuffer deviceRows, int nRows, int pos0) {
        SequenceState seq = targetSequence();
        final int nEmbd = spec.nEmbd();
        final int width = fc.cols();
        if (pos0 < 0 || !cache.fits(pos0, nRows)) {
            throw new IllegalStateException("dflash drafter: " + nRows + " target rows do not end at the target's"
                    + " position " + seq.nPast() + " within the draft cache of " + cache.capacity() + " positions");
        }
        cache.trim(pos0);
        if (deviceStack != null) {
            float[] fused = lilicorr != null ? new float[nRows * nEmbd] : null;
            if (deviceRows != null) {
                deviceStack.injectFeatures(deviceRows, nRows, pos0, fused);
            } else {
                deviceStack.injectFeatures(hostRows, width, nRows, pos0, fused);
            }
            if (fused != null) {
                for (int off = 0; off < nRows; off += INJECT_CHUNK) {
                    int n = Math.min(INJECT_CHUNK, nRows - off);
                    keepLilicorrRows(Arrays.copyOfRange(fused, off * nEmbd, (off + n) * nEmbd), n, pos0 + off);
                }
            }
            return;
        }
        for (int off = 0; off < nRows; off += INJECT_CHUNK) {
            final int n = Math.min(INJECT_CHUNK, nRows - off);
            float[] g = new float[n * nEmbd];
            fc.matmul(hostRows, off * width, g, 0, n);
            CpuOps.rmsNorm(g, 0, g, 0, n, nEmbd, encNorm, spec.rmsEps());
            stack.inject(g, n, pos0 + off);
            if (lilicorr != null) {
                keepLilicorrRows(g, n, pos0 + off);
            }
        }
    }

    /**
     * Reinjects the accepted rows of a tree verify from the capture of its suspended forward: the gathered rows
     * enter the draft cache at {@code pos0 ..}, exactly where the committed path's rows sit, as a chain decode's
     * injection would have left them. The capture releases with the call.
     *
     * @throws IllegalStateException when no capture holds the rows or a row lies outside it
     */
    @Override
    public void injectVerifyRows(int[] verifyRow, int pos0) {
        requireBound();
        final int n = verifyRow.length;
        if (capturedCount < n) {
            throw new IllegalStateException("dflash drafter: the suspended verify forward captured " + capturedCount
                    + " feature rows, the accepted path needs " + n);
        }
        for (int row : verifyRow) {
            if (row < 0 || row >= capturedCount) {
                throw new IllegalStateException("dflash drafter: verify row " + row + " is outside the "
                        + capturedCount + " captured feature rows");
            }
        }
        final int width = fc.cols();
        if (captureBuffer != null) {
            // the rows ascend and each source sits above its destination: the in-place gather reads only rows no
            // earlier copy wrote, on the queue that runs the copies in order
            ComputeQueue queue = device.context().queue();
            long bytes = (long) width * Float.BYTES;
            for (int i = 0; i < n; i++) {
                if (verifyRow[i] != i) {
                    queue.enqueueCopyBuffer(captureBuffer, verifyRow[i] * bytes, captureBuffer, i * bytes, bytes);
                }
            }
            injectRows(null, captureBuffer, n, pos0);
        } else {
            float[] accepted = new float[n * width];
            for (int i = 0; i < n; i++) {
                System.arraycopy(capturedRows, verifyRow[i] * width, accepted, i * width, width);
            }
            injectRows(accepted, null, n, pos0);
        }
        releaseCapture();
    }

    /** Keeps the host rows of a suspended verify forward: a defensive copy, the tap reuses its buffer. */
    private void captureHostRows(float[] feats, int nRows) {
        capturedRows = feats.clone();
        capturedCount = nRows;
    }

    /** Copies the device rows of a suspended verify forward into the drafter's own capture buffer. */
    private void captureDeviceRows(GpuBuffer feats, int nRows) {
        long count = (long) nRows * fc.cols();
        long bytes = Math.max(count * Float.BYTES, Integer.BYTES);
        if (captureBuffer == null || captureBuffer.sizeInBytes() < bytes) {
            if (captureBuffer != null) {
                captureBuffer.close();
            }
            captureBuffer = device.context().backend().allocateBuffer(bytes);
        }
        device.context().queue().enqueueCopyBuffer(feats, 0, captureBuffer, 0, bytes);
        capturedCount = nRows;
    }

    /** Drops a verify capture: the host rows and the device buffer's lease, if one is held. */
    private void releaseCapture() {
        capturedRows = null;
        capturedCount = 0;
        if (captureBuffer != null) {
            captureBuffer.close();
            captureBuffer = null;
        }
    }

    /**
     * Keeps the fused rows LiLiCorr reads as its anchor: rows of the last {@code block_size + 1} positions of the
     * chunk, in slot {@code pos % (block_size + 1)} (speculative.cpp:1229-1245).
     */
    private void keepLilicorrRows(float[] g, int n, int pos0) {
        final int nEmbd = spec.nEmbd();
        final int slots = lilicorrPos.length;
        final int posMin = Math.max(0, pos0 + n - 1 - spec.blockSize());
        for (int t = 0; t < n; t++) {
            final int pos = pos0 + t;
            if (pos < posMin) {
                continue;
            }
            final int slot = pos % slots;
            System.arraycopy(g, t * nEmbd, lilicorrRows, slot * nEmbd, nEmbd);
            lilicorrPos[slot] = pos;
        }
    }

    /** The stored fused row of slot {@code pos % (block_size + 1)} (speculative.cpp:1257-1262). */
    private float[] lilicorrAnchor(int pos) {
        final int nEmbd = spec.nEmbd();
        final int slot = pos % lilicorrPos.length;
        return Arrays.copyOfRange(lilicorrRows, slot * nEmbd, (slot + 1) * nEmbd);
    }

    // ------------------------------------------------------------------ readout

    /**
     * Logits of every block row in the target vocabulary: for DFlash2 first {@code * f_logit_scale} and the final
     * softcap (dflash.cpp:1065-1076), then the {@code d2t} scatter with {@code -inf} elsewhere (dflash.cpp:1078-1092).
     */
    private float[] targetLogits(DflashStack.Block block) {
        float[] head = block.logits();
        if (spec.dflash2()) {
            if (spec.logitScale() != 0.0f) {
                for (int i = 0; i < head.length; i++) {
                    head[i] *= spec.logitScale();
                }
            }
            final float cap = spec.finalSoftcap();
            if (cap > 0.0f) {
                for (int i = 0; i < head.length; i++) {
                    head[i] = (float) Math.tanh(head[i] * (1.0f / cap)) * cap;
                }
            }
        }
        if (d2t == null) {
            return head;
        }
        final int nVocab = spec.nVocab();
        final int rows = block.vocab();
        float[] full = new float[block.nTokens() * nVocab];
        Arrays.fill(full, Float.NEGATIVE_INFINITY);
        for (int t = 0; t < block.nTokens(); t++) {
            for (int r = 0; r < rows; r++) {
                full[t * nVocab + d2t[r]] = head[t * rows + r];
            }
        }
        return full;
    }

    /**
     * DFlash: the top token of block positions {@code 1 ..}, stopping when its probability among the
     * {@value DflashSpec#SAMPLER_TOP_K} best candidates is below {@code p_min} (speculative.cpp:1386-1414).
     */
    private int[] readGreedy(float[] logits, int nBlock) {
        final int nVocab = spec.nVocab();
        final boolean watermarked = draftWatermark != null && draftTemperature > 0f;
        int[] out = new int[nBlock - 1];
        int n = 0;
        for (int i = 1; i < nBlock; i++) {
            final int off = i * nVocab;
            if (watermarked) {
                float[] row = Arrays.copyOfRange(logits, off, off + nVocab);
                DraftWatermarker.Draw draw = draftWatermark.draw(row, nVocab, draftTemperature,
                        draftRng::nextDouble);
                if (options.pMin() > 0.0f && draw.topProbability() < options.pMin()) {
                    break;
                }
                draftProposals.add(draw.proposal());
                out[n++] = draw.token();
                continue;
            }
            final int top = CpuOps.argmax(logits, off, nVocab);
            if (options.pMin() > 0.0f && topProbability(logits, off, nVocab) < options.pMin()) {
                break;
            }
            out[n++] = top;
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Whether this round reads only the device's greedy ids: a plain DFlash draft (no DSpark, DFlash2 or LiLiCorr
     * readout) on the device without {@code p_min}, whose {@code d2t} (if any) keeps the target ids ascending, so the
     * first maximum over the draft-head rows is the first maximum over the scattered target row.
     */
    private boolean deviceGreedy() {
        return deviceStack != null && !spec.dspark() && !spec.dflash2() && lilicorr == null
                && options.pMin() <= 0.0f && d2tAscending;
    }

    /** {@link #readGreedy} over the device's greedy ids of block positions {@code 1 ..}, mapped through {@code d2t}. */
    private int[] readDeviceGreedy(int[] tokens, int pos0, int nBlock) {
        int[] ids;
        cache.trim(pos0);
        try {
            ids = deviceStack.forwardGreedy(tokens, pos0);
        } finally {
            cache.trim(pos0);
        }
        int[] out = new int[nBlock - 1];
        for (int i = 1; i < nBlock; i++) {
            out[i - 1] = d2t == null ? ids[i] : d2t[ids[i]];
        }
        return out;
    }

    /**
     * The {@code p_min} readout over the device's per-row top candidates: the top token of block positions
     * {@code 1 ..}, mapped through {@code d2t}, stopping when its probability among the candidates is below
     * {@code p_min}. The candidates arrive in rank order, so the probability sums their logits exactly as
     * {@link #topProbability} sums the host's top-k scan.
     */
    private int[] readGreedyTopK(GpuDflashBackbone.TopKRows rows, int nBlock) {
        int[] out = new int[nBlock - 1];
        int n = 0;
        for (int i = 1; i < rows.rows && i < nBlock; i++) {
            final int top = d2t == null ? rows.id(i, 0) : d2t[rows.id(i, 0)];
            if (options.pMin() > 0.0f && candidateProbability(rows, i) < options.pMin()) {
                break;
            }
            out[n++] = top;
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Probability of a row's best candidate after the {@code top_k(10)} restricted softmax: {@code 1 / sum_k
     * exp(l_k - l_max)} over the row's candidates.
     */
    private static float candidateProbability(GpuDflashBackbone.TopKRows rows, int row) {
        final float max = rows.val(row, 0);
        float sum = 0.0f;
        for (int j = 0; j < rows.k; j++) {
            sum += (float) Math.exp(rows.val(row, j) - max);
        }
        return 1.0f / sum;
    }

    /**
     * The DSpark readout over the device's biased argmax ids: positions {@code 0 ..} (anchor-first) or {@code 1 ..},
     * stopping at the first confidence below {@code p_min}. The confidences recompute on the host from the read-back
     * {@code t_embd} rows and the chain of winners, exactly {@code DflashMarkovHead.apply} builds them.
     */
    private int[] readDsparkDevice(int[] tokens, int pos0, int nBlock, int lastToken) {
        final int iBeg = spec.sampleFromAnchor() ? 0 : 1;
        GpuDflashBackbone.DsparkRows rows = deviceStack.forwardDspark(tokens, pos0, lastToken, d2t, iBeg,
                nBlock <= spec.graphBlockSize());
        float[] conf = null;
        if (options.pMin() > 0.0f && nBlock <= spec.graphBlockSize()) {
            conf = new float[nBlock];
            if (iBeg == 1 && nBlock > 0) {
                conf[0] = ActMath.sigmoid(rows.row0Logit);
            }
            int prev = lastToken;
            for (int i = iBeg; i < nBlock; i++) {
                conf[i] = markov.confidence(prev, rows.tEmbd, i * spec.nEmbd());
                prev = d2t == null ? rows.ids[i - iBeg] : d2t[rows.ids[i - iBeg]];
            }
        }
        int[] out = new int[Math.max(0, nBlock - iBeg)];
        int n = 0;
        for (int i = iBeg; i < nBlock; i++) {
            if (conf != null && conf[i] < options.pMin()) {
                break;
            }
            out[n++] = d2t == null ? rows.ids[i - iBeg] : d2t[rows.ids[i - iBeg]];
        }
        return Arrays.copyOf(out, n);
    }

    /** The read-back candidates of a device top-k block as target-vocabulary {@link DflashCandidates}. */
    private DflashCandidates candidates(GpuDflashBackbone.TopKRows rows) {
        int[] ids = new int[rows.rows * rows.k];
        float[] probs = new float[rows.rows * rows.k];
        for (int r = 0; r < rows.rows; r++) {
            for (int j = 0; j < rows.k; j++) {
                final int head = rows.id(r, j);
                ids[r * rows.k + j] = d2t == null ? head : d2t[head];
                probs[r * rows.k + j] = (float) (Math.exp(rows.val(r, j) - rows.val(r, 0)) / rows.sumExp[r]);
            }
        }
        return new DflashCandidates(rows.rows, rows.k, ids, probs);
    }

    private static boolean ascending(int[] ids) {
        for (int i = 1; i < ids.length; i++) {
            if (ids[i] <= ids[i - 1]) {
                return false;
            }
        }
        return true;
    }

    /**
     * DSpark: the top token of positions {@code 0 ..} (anchor-first) or {@code 1 ..}, stopping at the first
     * confidence below {@code p_min} (speculative.cpp:1351-1384).
     */
    private int[] readDspark(float[] logits, float[] conf, int nBlock) {
        final int nVocab = spec.nVocab();
        final int iBeg = spec.sampleFromAnchor() ? 0 : 1;
        final boolean watermarked = draftWatermark != null && draftTemperature > 0f;
        int[] out = new int[Math.max(0, nBlock - iBeg)];
        int n = 0;
        for (int i = iBeg; i < nBlock; i++) {
            if (conf != null && conf[i] < options.pMin()) {
                break;
            }
            if (watermarked) {
                float[] row = Arrays.copyOfRange(logits, i * nVocab, (i + 1) * nVocab);
                DraftWatermarker.Draw draw = draftWatermark.draw(row, nVocab, draftTemperature,
                        draftRng::nextDouble);
                draftProposals.add(draw.proposal());
                out[n++] = draw.token();
                continue;
            }
            out[n++] = CpuOps.argmax(logits, i * nVocab, nVocab);
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Probability of the best candidate after {@code top_k(10)} and the distribution sampler's softmax:
     * {@code 1 / sum_k exp(l_k - l_max)} over the ten largest logits.
     */
    static float topProbability(float[] logits, int off, int nVocab) {
        final int k = Math.min(DflashSpec.SAMPLER_TOP_K, nVocab);
        int[] idx = new int[k];
        CpuOps.topK(logits, off, nVocab, k, idx, 0);
        final float max = logits[off + idx[0]];
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            sum += (float) Math.exp(logits[off + idx[j]] - max);
        }
        return 1.0f / sum;
    }

    // ------------------------------------------------------------------ setup helpers

    private void requireBound() {
        if (backend == null) {
            throw new IllegalStateException("dflash drafter is not bound to a target (bindTarget)");
        }
    }

    /**
     * The tensor {@code name} of the draft, else of the target, else {@code fallback} of the target.
     *
     * @throws IllegalArgumentException when none exists
     */
    private static DflashMatrix firstOf(String name, CpuWeights draft, CpuWeights target, String fallback) {
        DflashMatrix m = DflashMatrix.orNull(draft, name);
        if (m == null) {
            m = DflashMatrix.orNull(target, name);
        }
        if (m == null && fallback != null) {
            m = DflashMatrix.orNull(target, fallback);
        }
        if (m == null) {
            throw new IllegalArgumentException("dflash drafter needs '" + name + "' from the draft or the target"
                    + (target == null ? " (no target weights were given)" : ""));
        }
        return m;
    }

    /**
     * {@code d2t}: the target id of each draft-head row, an I64 vector (dflash.cpp:128-135, 1084).
     *
     * @throws IllegalArgumentException for another element type or an id outside the vocabulary
     */
    static int[] readD2t(CpuWeights draft, int nVocab) {
        CpuTensorRef t = draft.get(TensorNames.D2T);
        if (t.type() != GGMLType.I64) {
            throw new IllegalArgumentException("dflash d2t must be I64, got " + t.type());
        }
        final long count = t.rows() * t.cols();
        if (count > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("dflash d2t holds " + count + " entries");
        }
        int[] ids = new int[(int) count];
        int i = 0;
        for (long r = 0; r < t.rows(); r++) {
            ByteBuffer b = t.rowBuffer(r);
            final int base = t.rowOffset(r);
            for (long c = 0; c < t.cols(); c++) {
                final long id = b.getLong(base + (int) (8 * c));
                if (id < 0 || id >= nVocab) {
                    throw new IllegalArgumentException("dflash d2t entry " + i + " maps to token " + id
                            + " outside the vocabulary of " + nVocab);
                }
                ids[i++] = (int) id;
            }
        }
        return ids;
    }

    /** Whether the target's feature rows arrive on the drafter's device ({@link DflashDeviceFeed}). */
    public boolean readsTargetRowsOnDevice() {
        return deviceFeed != null;
    }

    @Override
    public String toString() {
        return "DflashDrafter{" + spec + ", " + options + (device != null ? ", " + device : "")
                + (backend != null ? ", bound" : "") + '}';
    }
}
