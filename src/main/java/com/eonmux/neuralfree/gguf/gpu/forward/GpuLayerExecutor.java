package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuRecurrentTreeMixer;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.ops.norm.GpuAdaNorm;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The GPU layer executor: runs a stack of {@link LayerSpec}s over the residual stream {@link ScratchSlots#X} for one
 * ubatch, in this order, every step conditional on the spec flags:
 * <ol>
 *   <li>{@code extras.beforeBlock} (hyper-connection collapse, AltUp predict, PLE inputs);</li>
 *   <li>{@code attn_norm} into {@link ScratchSlots#NORMED} unless the layer has none (POST_ONLY, BERT post-LN),
 *       in which case the mixer reads {@code X} directly; falcon-40b's attention reads {@code attn_norm_2} and its
 *       FFN {@code attn_norm}, both of the layer input (falcon.cpp:71-88, 125);</li>
 *   <li>the mixer from the registry (T5 decoder layers: self-attention through the registry's ATTENTION factory;
 *       falcon-h1: the parallel attention branch summed into the recurrent branch, falcon-h1.cpp:152-166);</li>
 *   <li>post-attention norm on the branch (SANDWICH and POST_ONLY) and grok's {@code attn_output_norm}
 *       (grok.cpp:143-148);</li>
 *   <li>{@code extras.afterMixer}; when any returns true the residual is consumed;</li>
 *   <li>the residual by {@link ResidualKind} through {@value NormKernels#RESIDUAL_MIX} (STANDARD add, SCALED
 *       {@code x + s*branch}, NORMED_INPUT_SCALED {@code branch + s*norm(x)} minimax-01.cpp:428-458, PARALLEL
 *       deferred), skipped when {@code mixerResidual} is false; BERT post-LN applies {@code attn_output_norm} to the
 *       sum, then with {@code attn_norm_2} (jina-bert-v2) adds the layer input again and applies it
 *       (bert.cpp:156-165); T5 decoder layers then run {@code cross_attn_norm}, the cross-attention mixer of the
 *       CROSS_ATTENTION_T5 factory and its residual (t5.cpp:137-236);</li>
 *   <li>the FFN input: the shared {@code attn_norm} output, or the FFN norm tensor {@link NormResolver} names
 *       ({@code ffn_norm}, {@code post_attention_norm} for PRE_FFN, {@code attn_norm_2}, dbrx's
 *       {@code attn_output_norm}), or {@code X} without one; Voxtral's adaptive {@code ffn_norm} then scales it by
 *       {@code 1 + ada} ({@link GpuAdaNorm});</li>
 *   <li>the FFN / MoE block from the registry;</li>
 *   <li>the post-FFN norm on the branch (SANDWICH, POST_ONLY; grok's {@code layer_output_norm});</li>
 *   <li>{@code extras.afterFfn};</li>
 *   <li>the residual as in step 6; PARALLEL adds both branches here; BERT post-LN applies
 *       {@code layer_output_norm};</li>
 *   <li>{@code discardLayerOutput} restores the layer input (deci.cpp:145-148);</li>
 *   <li>{@code extras.afterBlock}, then {@code layer_output_scale} ({@value NormKernels#MUL_ROW_SCALE},
 *       gemma4.cpp) and {@code blockOutputScale} (rwkv6.cpp:164-165).</li>
 * </ol>
 * Schedules: slots run in order with their own memory and the weights of {@link LayerSpec#weightLayer()}. LOOPED
 * loop norms are the {@code loop_norm} extra's; HRM passes start from {@code zH + zL} (zL from
 * {@code hrm.z_l_init}, broadcast) and end with a weightless RMS norm whose result becomes {@code zL} or {@code zH}
 * (hrm-text.cpp:168-211, build_stack 92-167).
 *
 * <p>Blocks never write their input slot (GpuMixer / GpuFfnBlock contract), so the shared-norm FFN input, the
 * normed input of NORMED_INPUT_SCALED residuals and the second falcon-h1 mixer read {@code NORMED} as the norm left
 * it. A step enqueues work only: no {@code finish()} and no read; a {@link TapSink} that wants a tap point enqueues
 * its own read. Not thread-safe.</p>
 */
public final class GpuLayerExecutor implements AutoCloseable {

    /** How mixers run a step. */
    public enum StepMode {
        /** One token against memory: {@link GpuMixer#decode}. */
        DECODE,
        /** A ubatch, or any step of a model without memory: {@link GpuMixer#prefill}. */
        PREFILL
    }

    /** Receiver of the residual stream at tap points ({@code HiddenTap} semantics: tap L is the input of layer L). */
    public interface TapSink {

        /** A sink that wants nothing. */
        TapSink NONE = new TapSink() {
            @Override
            public boolean wants(int tap) {
                return false;
            }

            @Override
            public void capture(int tap, ScratchSlot x, int nTokens, int width) {
                throw new IllegalStateException("TapSink.NONE captures nothing");
            }

            @Override
            public String toString() {
                return "TapSink.NONE";
            }
        };

        /** Whether tap point {@code tap} (0 = stack input, layer count = stack output) is wanted. */
        boolean wants(int tap);

        /**
         * Called on the forward thread when {@link #wants} is true; {@code x} holds {@code nTokens} rows of
         * {@code width} once the queue reaches this point.
         */
        void capture(int tap, ScratchSlot x, int nTokens, int width);
    }

    /** Kernels the executor launches itself, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            NormKernels.ROW_NORM, NormKernels.RESIDUAL_MIX, NormKernels.RESIDUAL_MIX_NORM,
            NormKernels.MUL_ROW_SCALE, GpuOps.COPY2D, GpuOps.AXPBY));

    /** Mixer branch held back for a PARALLEL residual. */
    static final String TEMP_PARALLEL = "exec.parallel_branch";
    /** Output of the falcon-h1 attention branch before it joins the recurrent branch. */
    static final String TEMP_SECOND_MIXER = "exec.second_mixer";
    /** Layer input kept for {@code discardLayerOutput}. */
    static final String TEMP_LAYER_INPUT = "exec.layer_input";
    /** HRM high-level stream {@code zH}. */
    static final String TEMP_HRM_HIGH = "exec.hrm.z_h";
    /** HRM low-level stream {@code zL}. */
    static final String TEMP_HRM_LOW = "exec.hrm.z_l";
    /** Repeat schedule: the stream entering the repeated block. */
    static final String TEMP_REPEAT_INPUT = "exec.repeat.input";
    /** Repeat schedule: the state carried between passes. */
    static final String TEMP_REPEAT_STATE = "exec.repeat.state";
    /** SLERP residual: the normalized stream and branch rows. */
    static final String TEMP_SLERP_A = "exec.slerp.a";
    static final String TEMP_SLERP_B = "exec.slerp.b";
    private static final int ONE_BITS = Float.floatToRawIntBits(1.0f);

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final LayerSchedule schedule;
    private final List<LayerBlocks> layers;
    private final ScratchSlot x;
    private final ScratchSlot normed;
    private final ScratchSlot branch;
    private final ScratchSlot parallel;
    private final ScratchSlot second;
    private final ScratchSlot layerInput;
    private final ScratchSlot hrmHigh;
    private final ScratchSlot hrmLow;
    private final GpuWeight hrmLowInit;
    private final int hrmWidth;
    private final ScratchSlot repeatInput;
    private final ScratchSlot repeatState;
    private final int repeatWidth;
    private final boolean repeatCombine;
    private final ScratchSlot slerpA;
    private final ScratchSlot slerpB;
    private final int ubatch;
    private final int local;
    private final GpuBuffer placeholder;
    /** Adaptive FFN-norm vectors shared by the layers with {@code NormSpec.ffnAda()}; null when none has it. */
    private final GpuAdaNorm adaNorm;
    /** A residual mix held back for fusion with the next row norm of the stream; null when none is held. */
    private PendingMix pendingMix;
    /** The running recurrent tree round, or null when no tree verify runs over a recurrent target. */
    private GpuRecurrentTreeRound round;
    private boolean closed;

    /**
     * Executor of the model's trunk ({@link ModelSpec#layers()} under {@link ModelSpec#schedule()}).
     *
     * @throws UnsupportedOperationException when the registry lacks a block the spec needs
     * @throws IllegalStateException         when a layer's extras disagree with its residual or schedule flags
     * @throws MissingTensorException        when a norm tensor the spec implies is not resident
     */
    public static GpuLayerExecutor forTrunk(GpuForwardContext ctx, GpuBlockRegistry registry) {
        Objects.requireNonNull(ctx, "ctx");
        return new GpuLayerExecutor(ctx, registry, ctx.spec().layers(), ctx.spec().schedule());
    }

    /**
     * Executor of another stack (T5 encoder or decoder layers), run in list order with a linear schedule.
     *
     * @throws UnsupportedOperationException when the registry lacks a block the stack needs
     */
    public static GpuLayerExecutor forStack(GpuForwardContext ctx, GpuBlockRegistry registry, List<LayerSpec> layers) {
        return new GpuLayerExecutor(ctx, registry, layers, null);
    }

    private GpuLayerExecutor(GpuForwardContext ctx, GpuBlockRegistry registry, List<LayerSpec> stack,
                             LayerSchedule schedule) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(stack, "layers");
        String arch = ctx.spec().arch().ggufName();
        if (stack.isEmpty()) {
            throw new IllegalArgumentException(arch + ": the executor needs at least one layer");
        }
        if (schedule != null && schedule.slots() != stack.size()) {
            throw new IllegalArgumentException(arch + ": " + stack.size() + " layers but the schedule has "
                    + schedule.slots() + " slots");
        }
        this.schedule = schedule;
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        this.local = ops.local();
        this.placeholder = ctx.placeholders().readOnlyFloats();
        int maxStream = 0;
        boolean anyParallel = false;
        boolean anySecond = false;
        boolean anyKeepInput = false;
        for (LayerSpec l : stack) {
            maxStream = Math.max(maxStream, l.residual().streamWidth());
            anyParallel |= l.residual().kind() == ResidualKind.PARALLEL;
            anySecond |= l.parallelAttention();
            anyKeepInput |= LayerBlocks.keepsLayerInput(ctx.spec(), l);
        }
        long rows = (long) ubatch * maxStream;
        GpuScratch scratch = ctx.scratch();
        this.x = scratch.slot(ScratchSlots.X).require(rows);
        this.normed = scratch.slot(ScratchSlots.NORMED).require(rows);
        this.branch = scratch.slot(ScratchSlots.BRANCH).require(rows);
        this.parallel = anyParallel ? scratch.temp(TEMP_PARALLEL, rows) : null;
        this.second = anySecond ? scratch.temp(TEMP_SECOND_MIXER, rows) : null;
        this.layerInput = anyKeepInput ? scratch.temp(TEMP_LAYER_INPUT, rows) : null;
        boolean hrm = schedule != null && schedule.kind() == LayerSchedule.Kind.HRM;
        this.hrmWidth = hrm ? stack.get(0).residual().streamWidth() : 0;
        this.hrmHigh = hrm ? scratch.temp(TEMP_HRM_HIGH, (long) ubatch * hrmWidth) : null;
        this.hrmLow = hrm ? scratch.temp(TEMP_HRM_LOW, (long) ubatch * hrmWidth) : null;
        this.hrmLowInit = hrm ? hrmInit(ctx, hrmWidth) : null;
        if (hrm && hrmLow.offset() != 0) {
            throw new IllegalStateException(TEMP_HRM_LOW + " must start at element 0 for " + NormKernels.MUL_ROW_SCALE);
        }
        this.repeatCombine = schedule != null && schedule.kind() == LayerSchedule.Kind.REPEAT
                && schedule.loopCombine() == LayerSchedule.LoopCombine.JUSTNORM;
        if (repeatCombine) {
            this.repeatWidth = stack.get(0).residual().streamWidth();
            this.repeatInput = scratch.temp(TEMP_REPEAT_INPUT, (long) ubatch * repeatWidth);
            this.repeatState = scratch.temp(TEMP_REPEAT_STATE, (long) ubatch * repeatWidth);
        } else {
            this.repeatWidth = 0;
            this.repeatInput = null;
            this.repeatState = null;
        }
        boolean anySlerp = false;
        for (LayerSpec l : stack) {
            anySlerp |= l.residual().kind() == ResidualKind.SLERP;
        }
        this.slerpA = anySlerp ? scratch.temp(TEMP_SLERP_A, rows) : null;
        this.slerpB = anySlerp ? scratch.temp(TEMP_SLERP_B, rows) : null;
        this.layers = Collections.unmodifiableList(bindAll(ctx, registry, stack, schedule));
        try {
            for (LayerBlocks l : layers) {
                if (l.outScale != null && x.offset() != 0) {
                    throw new IllegalStateException("layer_output_scale needs the X slot at element 0, it starts at "
                            + x.offset());
                }
            }
            bindResidualNormFusion(hrm);
            this.adaNorm = adaNorm(ctx, layers);
            prewarmKernels(hrm);
        } catch (RuntimeException e) {
            closeAll(layers, e);
            throw e;
        }
    }

    /**
     * Marks the layers whose FFN residual may fuse with the next layer's attention norm
     * ({@link LayerBlocks#carriesFfnResidual}): the launches between the residual and that norm are the
     * in-place post-LN norm, the discard copy, the output scales and the extras' and schedules' stream
     * accesses, so a layer with none of them and a next layer that starts from a plain attention norm of the
     * same width carries it.
     */
    private void bindResidualNormFusion(boolean hrm) {
        if (hrm || repeatCombine) {
            return;
        }
        for (int i = 0; i < layers.size() - 1; i++) {
            LayerBlocks l = layers.get(i);
            LayerBlocks next = layers.get(i + 1);
            l.carriesFfnResidual = l.postLnLayer == null && !l.discard && l.outScale == null
                    && l.blockScale == 1.0f && l.extras.isEmpty() && next.extras.isEmpty() && !next.keepInput
                    && next.mixer != null && next.attnNorm != null && l.width == l.streamWidth
                    && next.width == l.width && next.streamWidth == l.streamWidth;
        }
    }

    /**
     * Prepares the kernels this executor launches at work-group size {@code L} ({@link GpuOps#prewarm}):
     * {@value NormKernels#RESIDUAL_MIX} for every residual, {@value NormKernels#RESIDUAL_MIX_NORM} when a layer
     * fuses its residual with a row norm, and {@value NormKernels#MUL_ROW_SCALE} when a layer has
     * {@code layer_output_scale} or a weighted-input post-FFN norm, or the schedule is HRM.
     */
    private void prewarmKernels(boolean hrm) {
        long[] size = {local};
        ops.prewarm(NormKernels.RESIDUAL_MIX, size);
        boolean rowScale = hrm;
        boolean mixNorm = false;
        for (LayerBlocks l : layers) {
            rowScale |= l.outScale != null || (l.postFfn != null && l.postFfn.weightFirst);
            mixNorm |= l.carriesFfnResidual || (l.ffn != null && l.ffnNorm != null && !l.sharedFfnNorm);
        }
        if (mixNorm) {
            ops.prewarm(NormKernels.RESIDUAL_MIX_NORM, size);
        }
        if (rowScale) {
            ops.prewarm(NormKernels.MUL_ROW_SCALE, size);
        }
    }

    /** The shared ada vectors sized for every layer that conditions its FFN norm, or null when none does. */
    private static GpuAdaNorm adaNorm(GpuForwardContext ctx, List<LayerBlocks> layers) {
        LayerBlocks first = null;
        int maxHidden = 0;
        for (LayerBlocks l : layers) {
            if (l.ada == null) {
                continue;
            }
            if (first == null) {
                first = l;
            } else if (l.width != first.width || l.spec.norm().ffnAdaDelay() != first.spec.norm().ffnAdaDelay()) {
                throw new IllegalStateException("layer " + l.spec.il() + ": the ada norm width or delay differs from"
                        + " layer " + first.spec.il());
            }
            maxHidden = Math.max(maxHidden, l.ada.hidden());
        }
        return first == null ? null : new GpuAdaNorm(ctx, first.width, first.spec.norm().ffnAdaDelay(), maxHidden);
    }

    private static GpuWeight hrmInit(GpuForwardContext ctx, int width) {
        String name = TensorNames.HRM_Z_L_INIT;
        GpuWeight w = ctx.weights().getOrNull(name);
        if (w == null) {
            w = ctx.weightSource().weightsFor(0).getOrNull(name);
        }
        if (w == null) {
            throw new MissingTensorException(ctx.spec().arch().ggufName(), -1, name);
        }
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.cols() != width || w.rows() != 1) {
            throw new IllegalArgumentException(name + " must be a single-buffer F32 vector of " + width + ", got " + w);
        }
        return w;
    }

    private static List<LayerBlocks> bindAll(GpuForwardContext ctx, GpuBlockRegistry registry, List<LayerSpec> stack,
                                             LayerSchedule schedule) {
        List<LayerBlocks> bound = new ArrayList<>(stack.size());
        NormNames names = new NormNames(ctx.spec());
        try {
            for (int i = 0; i < stack.size(); i++) {
                LayerSpec l = Objects.requireNonNull(stack.get(i), "layer");
                checkExtras(ctx.spec(), l, schedule != null && schedule.normAfterSlot(i));
                bound.add(LayerBlocks.bind(ctx, registry, l, names));
            }
        } catch (RuntimeException e) {
            closeAll(bound, e);
            throw e;
        }
        return bound;
    }

    /** Built-in residual features run as extras; a layer flagged for one must list it. */
    private static void checkExtras(ModelSpec spec, LayerSpec l, boolean loopNorm) {
        ResidualSpec r = l.residual();
        require(spec, l, r.kind() != ResidualKind.HYPER_CONNECTIONS || l.hasExtra(LayerSpec.Extras.HC),
                LayerSpec.Extras.HC, "hyper-connection residual");
        require(spec, l, !r.alphaBeta() || l.hasExtra(LayerSpec.Extras.RESID_ALPHA_BETA),
                LayerSpec.Extras.RESID_ALPHA_BETA, "alpha/beta residual");
        require(spec, l, !r.embeddingSkip() || l.hasExtra(LayerSpec.Extras.EMBD_SKIP),
                LayerSpec.Extras.EMBD_SKIP, "embedding skip");
        require(spec, l, !loopNorm || l.hasExtra(LayerSpec.Extras.LOOP_NORM),
                LayerSpec.Extras.LOOP_NORM, "loop norm after this slot");
    }

    private static void require(ModelSpec spec, LayerSpec l, boolean ok, String extra, String feature) {
        if (!ok) {
            throw new IllegalStateException(spec.arch().ggufName() + " layer " + l.il() + ": the " + feature
                    + " needs the '" + extra + "' layer extra, the layer lists " + l.extras());
        }
    }

    // ================================================================== accessors

    /** The layers this executor runs, in order. */
    public List<LayerSpec> layers() {
        List<LayerSpec> out = new ArrayList<>(layers.size());
        for (LayerBlocks l : layers) {
            out.add(l.spec);
        }
        return Collections.unmodifiableList(out);
    }

    /** Floats per {@code X} row after the last layer. */
    public int outputWidth() {
        return layers.get(layers.size() - 1).streamWidth;
    }

    /** The stack's slot count. */
    public int layerCount() {
        return layers.size();
    }

    /** The spec of slot {@code i}. */
    public LayerSpec spec(int i) {
        return layers.get(i).spec;
    }

    /** The mixer of slot {@code i}, or null when the layer has none. */
    public GpuMixer mixer(int i) {
        return layers.get(i).mixer;
    }

    /**
     * Installs the running tree round (a tree verify's masked step over a recurrent target); the mixer blocks
     * route the recurrent layers' steps through it while it is set, and the caller clears it after the step.
     */
    public void recurrentTreeRound(GpuRecurrentTreeRound r) {
        this.round = r;
    }

    /**
     * Whether a one-token decode step of this stack reads the scratch {@code POS} slot: true when any layer's mixer,
     * parallel attention or cross attention reads it ({@link GpuMixer#decodeReadsPositionSlot()}).
     */
    public boolean decodeReadsPositionSlot() {
        ensureOpen();
        for (GpuMixer m : stepMixers()) {
            if (m.decodeReadsPositionSlot()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether a decode step of this stack enqueues only kernel dispatches, which the decode-step replay records.
     * False when a layer's FFN block reads step results back to the host (a hosted MoE expert pass).
     */
    public boolean capturableDecode() {
        ensureOpen();
        for (LayerBlocks l : layers) {
            if (l.ffn != null && !l.ffn.capturableDecode()) {
                return false;
            }
        }
        return true;
    }

    /**
     * The decode-replay patchers of every mixer this stack's decode step runs: each layer's mixer, its parallel
     * attention and its cross attention, in trunk order. A graph plan claims the captured dispatches of the
     * step with them and patches the per-step values the decode body would re-set. Empty when no mixer has
     * per-step derivable kernel values.
     */
    public List<GpuStepPatcher> decodeReplayPatchers() {
        ensureOpen();
        List<GpuStepPatcher> out = new ArrayList<>();
        for (GpuMixer m : stepMixers()) {
            out.addAll(m.decodeReplayPatchers());
        }
        return out;
    }

    /**
     * The multi-row step replay patchers of every mixer this stack's {@code prefill} step runs: each layer's
     * mixer, its parallel attention and its cross attention, in trunk order. A graph plan claims the captured
     * dispatches of the step with them and patches the per-step values the prefill body would re-set. Empty
     * when no mixer has per-step derivable kernel values on its multi-row path.
     */
    public List<GpuStepPatcher> stepReplayPatchers() {
        ensureOpen();
        List<GpuStepPatcher> out = new ArrayList<>();
        for (GpuMixer m : stepMixers()) {
            out.addAll(m.stepReplayPatchers());
        }
        return out;
    }

    /**
     * The paged decode-step replay patchers of every mixer this stack's plain paged decode step runs (each layer's
     * mixer, its parallel attention and its cross attention, in trunk order), or null when one of them cannot vouch
     * for its launches' per-step values ({@link GpuMixer#pagedDecodeReplayPatchers()}).
     */
    public List<GpuStepPatcher> pagedDecodeReplayPatchers() {
        ensureOpen();
        List<GpuStepPatcher> out = new ArrayList<>();
        for (GpuMixer m : stepMixers()) {
            List<GpuStepPatcher> patchers = m.pagedDecodeReplayPatchers();
            if (patchers == null) {
                return null;
            }
            out.addAll(patchers);
        }
        return out;
    }

    /** The bound mixers a step runs: each layer's mixer, its parallel attention and its cross attention, in trunk
     *  order. */
    private List<GpuMixer> stepMixers() {
        List<GpuMixer> out = new ArrayList<>();
        for (LayerBlocks l : layers) {
            for (GpuMixer m : new GpuMixer[]{l.mixer, l.parallelAttention, l.crossMixer}) {
                if (m != null) {
                    out.add(m);
                }
            }
        }
        return out;
    }

    /**
     * Starts making the first layer's weights resident, so a streamed upload overlaps the embedding stage of the
     * step; {@link #run} prefetches it again (a no-op once started) for callers that skip this.
     */
    public void prefetchFirst() {
        ensureOpen();
        ctx.weightSource().prefetch(layers.get(0).spec.weightLayer());
    }

    // ================================================================== step

    /**
     * Enqueues every layer for {@code nTokens} tokens at positions {@code pos0 ..}; {@code X} holds the stack input
     * on entry and the stack output afterwards. The queue is flushed after every layer, so the device starts on a
     * layer while the host records the next one.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}, DECODE has more than
     *                                  one token, or {@code pos0} is negative
     * @throws IllegalStateException    when a hyper-connection layer's residual was not consumed by its extra
     */
    public void run(int pos0, int nTokens, StepMode mode, TapSink taps) {
        checkStep(pos0, nTokens, mode);
        runStack(pos0, nTokens, mode, taps, null, null);
    }

    /**
     * Whether {@link #run(int, int, StepMode, TapSink, int[])} prunes a {@code nTokens}-row step whose stack output
     * is read at the step-local {@code rows} only ({@link LastLayerRows}): a PREFILL step of a stack with no looped
     * or repeated schedule, no observer and no stack-output tap, whose last layer passes
     * {@link LastLayerRows#layerPrunable}, with a {@link LastLayerRows#compactable} row set; an empty set (nothing
     * read) also needs a memory to write.
     */
    public boolean prunesOutputRows(int nTokens, StepMode mode, TapSink taps, int[] rows) {
        ensureOpen();
        return mode == StepMode.PREFILL && hrmHigh == null && !repeatCombine && !ctx.observer().enabled()
                && (taps == null || !taps.wants(layers.size())) && LastLayerRows.compactable(nTokens, rows)
                && (rows.length > 0 || ctx.hasMemory()) && LastLayerRows.layerPrunable(layers.get(layers.size() - 1));
    }

    /**
     * {@link #run(int, int, StepMode, TapSink)} for a step whose stack output is read at the ascending step-local
     * {@code rows} only: the last layer's FFN part runs over those rows, which {@code X} then holds at rows
     * {@code 0 .. rows.length - 1}; with no row read, the last layer runs its cache writes alone, as {@link #runKV}
     * does, and leaves {@code X} mid-layer.
     *
     * @throws IllegalArgumentException as {@link #run(int, int, StepMode, TapSink)}, or when
     *                                  {@link #prunesOutputRows} does not hold
     */
    public void run(int pos0, int nTokens, StepMode mode, TapSink taps, int[] rows) {
        checkStep(pos0, nTokens, mode);
        if (!prunesOutputRows(nTokens, mode, taps, rows)) {
            throw new IllegalArgumentException("the step of " + nTokens + " rows cannot prune its output to "
                    + Arrays.toString(rows));
        }
        runStack(pos0, nTokens, mode, taps, rows, null);
    }

    private void checkStep(int pos0, int nTokens, StepMode mode) {
        ensureOpen();
        Objects.requireNonNull(mode, "mode");
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("step of " + nTokens + " tokens outside [1, " + ubatch + "]");
        }
        if (mode == StepMode.DECODE && nTokens != 1) {
            throw new IllegalArgumentException("DECODE steps carry one token, got " + nTokens);
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("step position " + pos0 + " is negative");
        }
    }

    /**
     * The layer loop of the run entries: every layer over the segments of {@code batched} when it is non-null, else
     * over the contiguous rows at {@code pos0}, the last layer pruned to {@code rows} when they are non-null.
     */
    private void runStack(int pos0, int nTokens, StepMode mode, TapSink taps, int[] rows, StepLayout batched) {
        TapSink sink = taps != null ? taps : TapSink.NONE;
        LayerWeightSource ws = ctx.weightSource();
        boolean hrm = hrmHigh != null;
        if (hrm) {
            startHrm(nTokens);
        }
        ws.prefetch(layers.get(0).spec.weightLayer());
        for (int i = 0; i < layers.size(); i++) {
            LayerBlocks l = layers.get(i);
            if (i + 1 < layers.size()) {
                ws.prefetch(layers.get(i + 1).spec.weightLayer());
            }
            ws.await(l.spec.weightLayer());
            if (hrm && schedule.passStart(i)) {
                mix(MixMode.ADD, 1.0f, hrmHigh, hrmLow, nTokens * hrmWidth);
            }
            repeatBeforeSlot(i, nTokens);
            if (sink.wants(i) && pendingMix != null) {
                flushPendingMix();
            }
            if (sink.wants(i)) {
                sink.capture(i, x, nTokens, l.streamWidth);
            }
            if (batched != null) {
                runLayerBatched(l, batched);
            } else if (rows != null && i == layers.size() - 1) {
                runLayerForRows(l, pos0, nTokens, mode, rows);
            } else {
                runLayer(l, pos0, nTokens, mode);
            }
            repeatAfterSlot(i, nTokens);
            if (hrm && schedule.passEnd(i)) {
                endHrmPass(i, l, nTokens);
            }
            ctx.queue().flush();
        }
        flushPendingMix();
        int out = layers.size();
        if (sink.wants(out)) {
            sink.capture(out, x, nTokens, outputWidth());
        }
    }

    /**
     * The pruned last layer: with rows to read, the norm and attention part over every row, then the read rows
     * moved to the front of {@code X} and the FFN part over them; with none, the cache writes alone.
     */
    private void runLayerForRows(LayerBlocks l, int pos0, int n, StepMode mode, int[] rows) {
        if (rows.length == 0) {
            mixerBlockKV(l, pos0, n, mode);
            return;
        }
        ScratchSlot in = runLayerNorm(l.spec.il(), pos0, n);
        mixerBlock(l, pos0, n, mode, in);
        flushPendingMix();
        LastLayerRows.compact(ops, x, l.streamWidth, rows);
        runLayerPost(l.spec.il(), pos0, rows.length, null);
    }

    /**
     * Enqueues every layer's cache writes alone: the extras' pre-block hooks, the attention norm and the mixer's
     * cache-write-only step ({@link GpuMixer#prefillKV}). A layer whose mixer has no such step runs its whole
     * attention part instead ({@link #mixerBlock}), so its caches are written either way. A step whose stack output
     * no reader consumes thus stops at the K/V projections and the store; {@code X} is left mid-layer. The queue is
     * flushed after every layer, as {@link #run} does.
     *
     * @throws IllegalArgumentException as {@link #run(int, int, StepMode, TapSink)}
     * @throws IllegalStateException    when the stack is a looped schedule (HRM), whose passes read the stream
     *                                  between layers
     */
    public void runKV(int pos0, int nTokens, StepMode mode) {
        ensureOpen();
        Objects.requireNonNull(mode, "mode");
        if (hrmHigh != null) {
            throw new IllegalStateException("a looped-schedule stack cannot run its cache writes only");
        }
        checkStep(pos0, nTokens, mode);
        LayerWeightSource ws = ctx.weightSource();
        for (int i = 0; i < layers.size(); i++) {
            LayerBlocks l = layers.get(i);
            if (i + 1 < layers.size()) {
                ws.prefetch(layers.get(i + 1).spec.weightLayer());
            }
            ws.await(l.spec.weightLayer());
            int streamLen = nTokens * l.streamWidth;
            if (l.keepInput) {
                ops.copy(x.buffer(), x.offset(), layerInput.buffer(), layerInput.offset(), streamLen);
            }
            for (GpuLayerExtra e : l.extras) {
                e.beforeBlock(pos0, nTokens, x);
            }
            mixerBlockKV(l, pos0, nTokens, mode);
            ctx.queue().flush();
        }
        flushPendingMix();
    }

    /**
     * The cache-write-only form of steps 2-6: the attention norm and the mixer's {@link GpuMixer#prefillKV}. The
     * branch norms, the extras' mixer hooks, the residual and the post-LN steps are skipped, since no reader consumes
     * the layer output. A layer whose mixer (or parallel attention) has no cache-write-only step falls back to
     * {@link #mixerBlock}, whose attention part writes the same caches.
     */
    private void mixerBlockKV(LayerBlocks l, int pos0, int n, StepMode mode) {
        if (l.mixer == null) {
            return;
        }
        ScratchSlot in = x;
        if (l.attnNorm != null) {
            stageNorm(l.spec.il(), ExecutionObserver.Stage.ATTN_NORM, l.attnNorm, l, x, normed, n);
            in = normed;
        }
        if (!kvOnlyCapable(l)) {
            mixerBlock(l, pos0, n, mode, in);
            return;
        }
        l.mixer.prefillKV(pos0, n, in);
        if (l.parallelAttention != null) {
            l.parallelAttention.prefillKV(pos0, n, in);
        }
    }
    /**
     * Whether a layer's whole attention part can be reduced to its cache writes: the mixer and any parallel
     * attention have a cache-write-only step, and no cross attention feeds the mixer a different input.
     */
    private static boolean kvOnlyCapable(LayerBlocks l) {
        return l.mixer.kvRefusal() == null
                && (l.parallelAttention == null || l.parallelAttention.kvRefusal() == null)
                && l.crossMixer == null;
    }

    /**
     * Enqueues every layer over the rows of a {@link StepLayout}: {@link #run(int, int, StepMode, TapSink)} for a
     * contiguous un-paged step, the batched step otherwise — every mixer steps over the layout's segments through
     * the batched hooks ({@code GpuMixer.batch}, {@code GpuLayerExtra} batched forms), whose paged steps address
     * the KV rows through the memory's current {@code PagedStep}. {@code X} holds the stack input on entry and the
     * stack output afterwards. The queue is flushed after every layer.
     *
     * @throws IllegalArgumentException when the step holds more rows than the ubatch
     * @throws UnsupportedOperationException when a mixer or extra refuses the batched step
     */
    public void run(StepLayout s, TapSink taps) {
        ensureOpen();
        Objects.requireNonNull(s, "s");
        int n = s.nTokens();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("step of " + n + " tokens outside [1, " + ubatch + "]");
        }
        if (s.contiguousSingle()) {
            run(s.pos0(0), n, ctx.hasMemory() && n == 1 ? StepMode.DECODE : StepMode.PREFILL, taps);
            return;
        }
        runStack(s.pos0(0), n, StepMode.PREFILL, taps, null, s);
    }

    /** One layer over a batched step: the extras' and mixers' batched hooks instead of the contiguous ones. */
    private void runLayerBatched(LayerBlocks l, StepLayout s) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        int il = l.spec.il();
        int n = s.nTokens();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.LAYER);
        }
        int streamLen = n * l.streamWidth;
        if (l.keepInput) {
            ops.copy(x.buffer(), x.offset(), layerInput.buffer(), layerInput.offset(), streamLen);
        }
        for (GpuLayerExtra e : l.extras) {
            e.beforeBlock(s, x);
        }
        ScratchSlot deferred = mixerBlockBatched(l, s);
        if (l.crossMixer != null) {
            crossBlockBatched(l, s);
        }
        ffnBlock(l, s.pos0(0), n, deferred);
        if (l.discard) {
            ops.copy(layerInput.buffer(), layerInput.offset(), x.buffer(), x.offset(), streamLen);
        }
        for (GpuLayerExtra e : l.extras) {
            e.afterBlock(s, x);
        }
        if (l.outScale != null) {
            ops.enqueue(NormKernels.MUL_ROW_SCALE, global(streamLen), new long[]{local},
                    NormKernels.mulRowScaleArgs(n, l.streamWidth, l.outScale.shard(0), (int) l.outScale.cols(),
                            x.buffer()));
        }
        if (l.blockScale != 1.0f) {
            ops.scale(x.buffer(), x.offset(), streamLen, l.blockScale);
        }
        if (observed) {
            flushPendingMix();
            obs.onStageEnd(il, ExecutionObserver.Stage.LAYER, x, n, l.streamWidth);
        }
    }

    /** Steps 2-6 over a batched step. Returns the mixer branch of a PARALLEL residual, else null. */
    private ScratchSlot mixerBlockBatched(LayerBlocks l, StepLayout s) {
        int il = l.spec.il();
        int n = s.nTokens();
        int len = n * l.width;
        if (l.mixer == null) {
            if (l.sharedFfnNorm) {
                stageNorm(il, ExecutionObserver.Stage.ATTN_NORM, l.attnNorm, l, x, normed, n);
            }
            return null;
        }
        ScratchSlot in = x;
        if (l.attnNorm != null) {
            stageNorm(il, ExecutionObserver.Stage.ATTN_NORM, l.attnNorm, l, x, normed, n);
            in = normed;
        }
        ScratchSlot out = l.kind == ResidualKind.PARALLEL ? parallel : branch;
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER);
        }
        if (!mixerTreeStep(l, n, in, out)) {
            l.mixer.batch(s, in, out);
        }
        if (l.parallelAttention != null) {
            l.parallelAttention.batch(s, in, second);
            ops.add(out.buffer(), out.offset(), second.buffer(), second.offset(), out.buffer(), out.offset(), len);
        }
        if (l.postAttn != null) {
            norm(l.postAttn, l, out, out, n);
        }
        if (l.attnOutBranch != null) {
            norm(l.attnOutBranch, l, out, out, n);
        }
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER, out, n, l.width);
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER_RESIDUAL);
        }
        boolean consumed = false;
        for (GpuLayerExtra e : l.extras) {
            consumed |= e.afterMixer(s, x, out);
        }
        ScratchSlot deferred = null;
        if (!consumed) {
            requireNotHc(l, "afterMixer");
            if (l.mixerResidual) {
                if (l.kind == ResidualKind.PARALLEL) {
                    deferred = out;
                } else {
                    residual(l, in, out, len, false);
                }
            }
        }
        if (l.postLnAttn != null) {
            norm(l.postLnAttn, l, x, x, n);
            if (l.postLnAttn2 != null) {
                mix(MixMode.ADD, 1.0f, x, layerInput, len);
                norm(l.postLnAttn2, l, x, x, n);
            }
        }
        if (observed) {
            flushPendingMix();
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER_RESIDUAL, x, n, l.streamWidth);
        }
        return deferred;
    }

    /**
     * The recurrent tree step of the running round: captures the step's mixer-input rows for the round's replay,
     * then runs every computed row through the mixer against its stack slot (the rows of a padding tail stay
     * uncomputed — no reader consumes them). False when no round runs or this layer's mixer keeps no recurrent
     * tree step, so the normal batched step runs instead; the layer's other sub-blocks (a hybrid layer's parallel
     * attention, the norms, the residual) always run their normal step, which the tree mask makes per-path correct.
     */
    private boolean mixerTreeStep(LayerBlocks l, int n, ScratchSlot in, ScratchSlot out) {
        GpuRecurrentTreeRound r = round;
        if (r == null || !(l.mixer instanceof GpuRecurrentTreeMixer)) {
            return false;
        }
        r.captureInput(l.spec.il(), in, n);
        ((GpuRecurrentTreeMixer) l.mixer).treeStep(r, in, out);
        return true;
    }

    /** T5 decoder cross attention over a batched step (t5.cpp:178-206). */
    private void crossBlockBatched(LayerBlocks l, StepLayout s) {
        int il = l.spec.il();
        int n = s.nTokens();
        stageNorm(il, ExecutionObserver.Stage.ATTN_NORM, l.crossNorm, l, x, normed, n);
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER);
        }
        l.crossMixer.batch(s, normed, branch);
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER, branch, n, l.width);
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER_RESIDUAL);
        }
        mix(MixMode.ADD, 1.0f, x, branch, n * l.width);
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER_RESIDUAL, x, n, l.streamWidth);
        }
    }

    private void runLayer(LayerBlocks l, int pos0, int n, StepMode mode) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        int il = l.spec.il();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.LAYER);
        }
        ScratchSlot in = runLayerNorm(l.spec.il(), pos0, n);
        ScratchSlot deferred = runLayerAttention(l.spec.il(), pos0, n, mode, in);
        runLayerPost(l.spec.il(), pos0, n, deferred);
        if (observed) {
            flushPendingMix();
            obs.onStageEnd(il, ExecutionObserver.Stage.LAYER, x, n, l.streamWidth);
        }
    }

    /**
     * The first phase of layer {@code il}'s contiguous blocks: the layer-input copy, the extras' pre-block hooks and
     * the attention norm. Returns the (normed) input the mixer blocks read; pass it to
     * {@link #runLayerAttention}. A stack runs whole through {@link #run}; the phase entries exist for the prefill
     * context-parallel coordinator, which interleaves the ranks layer by layer between the phases.
     */
    public ScratchSlot runLayerNorm(int il, int pos0, int n) {
        ensureOpen();
        LayerBlocks l = layerAt(il);
        int streamLen = n * l.streamWidth;
        if (l.keepInput) {
            ops.copy(x.buffer(), x.offset(), layerInput.buffer(), layerInput.offset(), streamLen);
        }
        for (GpuLayerExtra e : l.extras) {
            e.beforeBlock(pos0, n, x);
        }
        ScratchSlot in = x;
        if ((l.mixer != null || l.sharedFfnNorm) && l.attnNorm != null) {
            stageNorm(il, ExecutionObserver.Stage.ATTN_NORM, l.attnNorm, l, x, normed, n);
            in = normed;
        }
        return in;
    }

    /**
     * The mixer phase of layer {@code il}'s contiguous blocks: the mixer (and any parallel attention) over
     * {@code in}, the branch norms, the extras' mixer hooks, the residual and the post-attention norms. Returns the
     * branch a PARALLEL residual defers to the FFN phase; pass it to {@link #runLayerPost}.
     */
    public ScratchSlot runLayerAttention(int il, int pos0, int n, StepMode mode, ScratchSlot in) {
        ensureOpen();
        return mixerBlock(layerAt(il), pos0, n, mode, in);
    }

    /**
     * The last phase of layer {@code il}'s contiguous blocks: the cross attention (when the layer has one), the FFN
     * block, the discard, the extras' after-block hooks and the output scales. {@code deferred} is the branch
     * {@link #runLayerAttention} returned.
     */
    public void runLayerPost(int il, int pos0, int n, ScratchSlot deferred) {
        ensureOpen();
        LayerBlocks l = layerAt(il);
        int streamLen = n * l.streamWidth;
        if (l.crossMixer != null) {
            crossBlock(l, pos0, n, StepMode.PREFILL);
        }
        ffnBlock(l, pos0, n, deferred);
        if (l.discard) {
            ops.copy(layerInput.buffer(), layerInput.offset(), x.buffer(), x.offset(), streamLen);
        }
        for (GpuLayerExtra e : l.extras) {
            e.afterBlock(pos0, n, x);
        }
        if (l.outScale != null) {
            ops.enqueue(NormKernels.MUL_ROW_SCALE, global(streamLen), new long[]{local},
                    NormKernels.mulRowScaleArgs(n, l.streamWidth, l.outScale.shard(0), (int) l.outScale.cols(),
                            x.buffer()));
        }
        if (l.blockScale != 1.0f) {
            ops.scale(x.buffer(), x.offset(), streamLen, l.blockScale);
        }
    }

    private LayerBlocks layerAt(int il) {
        for (LayerBlocks l : layers) {
            if (l.spec.il() == il) {
                return l;
            }
        }
        throw new IllegalArgumentException("layer " + il + " is not one of this executor's "
                + layers.size() + " layers");
    }

    /**
     * Steps 3-6 over the (normed) input {@code in} the norm phase produced. Returns the mixer branch when a
     * PARALLEL residual defers its add to step 11, else null.
     */
    private ScratchSlot mixerBlock(LayerBlocks l, int pos0, int n, StepMode mode, ScratchSlot in) {
        int il = l.spec.il();
        int len = n * l.width;
        if (l.mixer == null) {
            return null;
        }
        ScratchSlot out = l.kind == ResidualKind.PARALLEL ? parallel : branch;
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER);
        }
        runMixer(l.mixer, pos0, n, mode, in, out);
        if (l.parallelAttention != null) {
            runMixer(l.parallelAttention, pos0, n, mode, in, second);
            ops.add(out.buffer(), out.offset(), second.buffer(), second.offset(), out.buffer(), out.offset(), len);
        }
        if (l.postAttn != null) {
            norm(l.postAttn, l, out, out, n);
        }
        if (l.attnOutBranch != null) {
            norm(l.attnOutBranch, l, out, out, n);
        }
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER, out, n, l.width);
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER_RESIDUAL);
        }
        boolean consumed = false;
        for (GpuLayerExtra e : l.extras) {
            consumed |= e.afterMixer(pos0, n, x, out);
        }
        ScratchSlot deferred = null;
        if (!consumed) {
            requireNotHc(l, "afterMixer");
            if (l.mixerResidual) {
                if (l.kind == ResidualKind.PARALLEL) {
                    deferred = out;
                } else {
                    residual(l, in, out, len, false);
                }
            }
        }
        if (l.postLnAttn != null) {
            norm(l.postLnAttn, l, x, x, n);
            if (l.postLnAttn2 != null) {
                mix(MixMode.ADD, 1.0f, x, layerInput, len);
                norm(l.postLnAttn2, l, x, x, n);
            }
        }
        if (observed) {
            flushPendingMix();
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER_RESIDUAL, x, n, l.streamWidth);
        }
        return deferred;
    }

    /** T5 decoder: {@code x += cross_attn(cross_attn_norm(x))} after the self-attention residual (t5.cpp:178-206). */
    private void crossBlock(LayerBlocks l, int pos0, int n, StepMode mode) {
        int il = l.spec.il();
        stageNorm(il, ExecutionObserver.Stage.ATTN_NORM, l.crossNorm, l, x, normed, n);
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER);
        }
        runMixer(l.crossMixer, pos0, n, mode, normed, branch);
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER, branch, n, l.width);
            obs.onStageStart(il, ExecutionObserver.Stage.MIXER_RESIDUAL);
        }
        mix(MixMode.ADD, 1.0f, x, branch, n * l.width);
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.MIXER_RESIDUAL, x, n, l.streamWidth);
        }
    }

    /** Steps 7-11. */
    private void ffnBlock(LayerBlocks l, int pos0, int n, ScratchSlot deferred) {
        int il = l.spec.il();
        int len = n * l.width;
        if (l.ffn == null) {
            if (deferred != null) {
                mix(MixMode.ADD, 1.0f, x, deferred, len);
            }
            if (l.postLnLayer != null) {
                norm(l.postLnLayer, l, x, x, n);
            }
            return;
        }
        ScratchSlot in = x;
        if (l.sharedFfnNorm) {
            in = normed;
        } else if (l.ffnNorm != null) {
            stageNorm(il, ExecutionObserver.Stage.FFN_NORM, l.ffnNorm, l, x, normed, n);
            in = normed;
            if (l.ada != null) {
                adaNorm.apply(l.ada, normed.buffer(), normed.offset(), n);
            }
        }
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.FFN);
        }
        l.ffn.forward(n, in, branch);
        if (l.postFfn != null) {
            norm(l.postFfn, l, branch, branch, n);
        }
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.FFN, branch, n, l.width);
            obs.onStageStart(il, ExecutionObserver.Stage.FFN_RESIDUAL);
        }
        boolean consumed = false;
        for (GpuLayerExtra e : l.extras) {
            consumed |= e.afterFfn(pos0, n, x, branch);
        }
        if (!consumed) {
            requireNotHc(l, "afterFfn");
            if (deferred != null) {
                mix(MixMode.ADD, 1.0f, x, deferred, len);
            }
            residual(l, in, branch, len, true);
        }
        if (l.postLnLayer != null) {
            norm(l.postLnLayer, l, x, x, n);
        }
        if (observed) {
            flushPendingMix();
            obs.onStageEnd(il, ExecutionObserver.Stage.FFN_RESIDUAL, x, n, l.streamWidth);
        }
    }

    private void requireNotHc(LayerBlocks l, String hook) {
        if (l.kind == ResidualKind.HYPER_CONNECTIONS) {
            throw new IllegalStateException("layer " + l.spec.il() + ": no extra consumed the hyper-connection"
                    + " residual in " + hook);
        }
    }

    /**
     * {@code x} joins {@code branch} by the layer's residual kind; {@code in} is the sub-block's (normed) input and
     * {@code ffn} tells the FFN sub-block from the mixer. The add and scaled residuals hold back for fusion with the
     * next row norm of the stream ({@link #deferOrMix}) where the launches between them are exactly that norm.
     */
    private void residual(LayerBlocks l, ScratchSlot in, ScratchSlot branchOut, int len, boolean ffn) {
        switch (l.kind) {
            case STANDARD:
            case PARALLEL:
                deferOrMix(MixMode.ADD, 1.0f, branchOut, len, l, ffn);
                break;
            case SCALED:
                deferOrMix(MixMode.SCALED, l.residualScale, branchOut, len, l, ffn);
                break;
            case NORMED_INPUT_SCALED:
                mix(MixMode.NORMED_INPUT_SCALED, l.residualScale, in, branchOut, len);
                break;
            case DEPTH_SCALED: {
                ResidualSpec r = l.spec.residual();
                ops.scale(x.buffer(), x.offset(), len, r.streamScale(ffn));
                mix(MixMode.SCALED, r.branchScale(ffn), x, branchOut, len);
                break;
            }
            case SLERP:
                slerp(l, in, branchOut, len, ffn);
                break;
            default:
                throw new IllegalStateException("layer " + l.spec.il() + ": no executor residual for " + l.kind);
        }
    }

    /** A residual mix held back for {@link #consumePendingMix}: the add it owes {@code x}, its scale and shape. */
    private static final class PendingMix {
        final MixMode mode;
        final float s;
        final ScratchSlot addend;
        final int nRows;
        final int width;

        PendingMix(MixMode mode, float s, ScratchSlot addend, int nRows, int width) {
            this.mode = mode;
            this.s = s;
            this.addend = addend;
            this.nRows = nRows;
            this.width = width;
        }
    }

    /**
     * Launches {@code x = mix(x, addend)} ({@value NormKernels#RESIDUAL_MIX}) or, when the launches that follow
     * before {@code x} is read again are exactly one row norm of {@code x}, holds the mix back for
     * {@link NormKernels#RESIDUAL_MIX_NORM}. The mixer and FFN residuals of the add and scaled kinds fuse with the
     * layer's FFN norm ({@code nextFfnNorm} false, a sub-block residual) or, at {@code carriesFfnResidual}, with
     * the next layer's attention norm. An {@link ExecutionObserver} sees the streams at stage boundaries, so an
     * observed step applies the pending mix first and runs the two launches it would without the fusion.
     */
    private void deferOrMix(MixMode mode, float s, ScratchSlot addend, int len, LayerBlocks l, boolean nextFfnNorm) {
        int n = len / l.width;
        boolean deferrable = (mode == MixMode.ADD || mode == MixMode.SCALED) && n >= 1
                && (nextFfnNorm ? l.carriesFfnResidual : canFuseIntoFfnNorm(l));
        if (!deferrable) {
            mix(mode, s, x, addend, len);
            return;
        }
        pendingMix = new PendingMix(mode, s, addend, n, l.width);
    }

    /**
     * Whether this layer's next launch reading {@code x} is the FFN sub-block's input norm: the mixer residual
     * fuses into it when no in-place post-attention norm sits between them.
     */
    private static boolean canFuseIntoFfnNorm(LayerBlocks l) {
        return l.postLnAttn == null && l.ffn != null && l.ffnNorm != null && !l.sharedFfnNorm
                && l.width == l.streamWidth;
    }

    /** Applies a held-back residual mix as its own launch ({@value NormKernels#RESIDUAL_MIX}). */
    private void flushPendingMix() {
        if (pendingMix == null) {
            return;
        }
        PendingMix p = pendingMix;
        pendingMix = null;
        mix(p.mode, p.s, x, p.addend, p.nRows * p.width);
    }

    /**
     * Launches a held-back residual mix and the norm of {@code dst} as one {@value NormKernels#RESIDUAL_MIX_NORM}
     * dispatch when both are the plain forms the kernel provides: {@code src} is the stream, the norm is an
     * ungrouped row norm without the weighted-input form, the shapes match and the backend provides the kernel.
     *
     * @return true when the fused launch replaced the pending mix and this norm; false leaves the pending mix for
     *         the caller to flush and this norm to run as it would
     */
    private boolean consumePendingMix(Norm nw, LayerBlocks l, ScratchSlot src, ScratchSlot dst, int n, float eps) {
        PendingMix p = pendingMix;
        if (src != x || nw.groups > 1 || nw.weightFirst || dst == x || n != p.nRows || l.width != p.width) {
            return false;
        }
        if (!ops.residualMixNorm(p.mode, p.s, l.normKind, nw.weight, nw.bias, eps,
                x.buffer(), x.offset(), p.addend.buffer(), p.addend.offset(), dst.buffer(), dst.offset(),
                n, l.width)) {
            return false;
        }
        pendingMix = null;
        return true;
    }

    /** {@code X = mix(inp, cur)} over {@code n} floats ({@value NormKernels#RESIDUAL_MIX}). */
    private void mix(MixMode mode, float s, ScratchSlot inp, ScratchSlot cur, int n) {
        ops.enqueue(NormKernels.RESIDUAL_MIX, global(n), new long[]{local},
                NormKernels.residualMixArgs(mode, n, s, inp.buffer(), inp.offset(), cur.buffer(), cur.offset(),
                        placeholder, placeholder, 1, x.buffer(), x.offset(), placeholder));
    }

    /**
     * Dory's slerp residual on the GPU: {@code justNorm(justNorm(a) + lr * (justNorm(b) - justNorm(a)))} over each
     * row, with the sub-block's per-element {@code lr} and its complement ({@link MixMode#ALPHA_BETA}).
     */
    private void slerp(LayerBlocks l, ScratchSlot in, ScratchSlot branchOut, int len, boolean ffn) {
        GpuWeight lr = ffn ? l.ffnSlerpLr : l.attnSlerpLr;
        GpuWeight oneMinus = ffn ? l.ffnSlerpOneMinus : l.attnSlerpOneMinus;
        if (lr == null || slerpA == null) {
            throw new IllegalStateException("layer " + l.spec.il() + ": no slerp alpha for the "
                    + (ffn ? "FFN" : "mixer") + " sub-block");
        }
        int n = len / l.width;
        ops.norm(NormKind.L2_GDN, null, null, l.normEps, in.buffer(), in.offset(), slerpA.buffer(),
                slerpA.offset(), n, l.width);
        ops.norm(NormKind.L2_GDN, null, null, l.normEps, branchOut.buffer(), branchOut.offset(), slerpB.buffer(),
                slerpB.offset(), n, l.width);
        ops.enqueue(NormKernels.RESIDUAL_MIX, global(len), new long[]{local},
                NormKernels.residualMixArgs(MixMode.ALPHA_BETA, len, 1.0f, slerpA.buffer(), slerpA.offset(),
                        slerpB.buffer(), slerpB.offset(), oneMinus.shard(0), lr.shard(0), l.width, x.buffer(),
                        x.offset(), placeholder));
        ops.norm(NormKind.L2_GDN, null, null, l.normEps, x.buffer(), x.offset(), x.buffer(), x.offset(), n, l.width);
    }

    private void stageNorm(int il, ExecutionObserver.Stage stage, Norm nw, LayerBlocks l, ScratchSlot src,
                           ScratchSlot dst, int n) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(il, stage);
        }
        norm(nw, l, src, dst, n);
        if (observed) {
            obs.onStageEnd(il, stage, dst, n, l.width);
        }
    }

    private void norm(Norm nw, LayerBlocks l, ScratchSlot src, ScratchSlot dst, int n) {
        float eps = Float.isNaN(nw.eps) ? l.normEps : nw.eps;
        if (pendingMix != null) {
            if (consumePendingMix(nw, l, src, dst, n, eps)) {
                return;
            }
            flushPendingMix();
        }
        if (nw.groups > 1) {
            int rows = n * nw.groups;
            int slice = l.width / nw.groups;
            ops.enqueue(NormKernels.ROW_NORM, new long[]{(long) rows * local}, new long[]{local},
                    NormKernels.rowNormArgs(l.normKind, eps, rows, slice, slice, src.buffer(), src.offset(),
                            dst.buffer(), dst.offset(), nw.weight.shard(0), null, nw.groups, placeholder));
            return;
        }
        if (nw.weightFirst) {
            if (dst.offset() != 0) {
                throw new IllegalStateException("layer " + l.spec.il() + ": the weighted-input norm needs its output"
                        + " at the start of " + dst.name());
            }
            ops.copy(src.buffer(), src.offset(), dst.buffer(), dst.offset(), n * l.width);
            ops.enqueue(NormKernels.MUL_ROW_SCALE, global((long) n * l.width), new long[]{local},
                    NormKernels.mulRowScaleArgs(n, l.width, nw.weight.shard(0), l.width, dst.buffer()));
            ops.norm(l.normKind, null, null, eps, dst.buffer(), dst.offset(), dst.buffer(), dst.offset(), n, l.width);
            return;
        }
        ops.norm(l.normKind, nw.weight, nw.bias, eps, src.buffer(), src.offset(), dst.buffer(), dst.offset(), n,
                l.width);
    }

    private static void runMixer(GpuMixer m, int pos0, int n, StepMode mode, ScratchSlot in, ScratchSlot out) {
        if (mode == StepMode.DECODE) {
            m.decode(pos0, in, out);
        } else {
            m.prefill(pos0, n, in, out);
        }
    }

    // ================================================================== repeat schedules

    /**
     * {@link LayerSchedule.LoopCombine#JUSTNORM} before slot {@code i}: keep the stream entering the repeated block,
     * start each pass from {@code justNorm(state + input)} (Dory's {@code combined = justnorm(state + input_h)}).
     */
    private void repeatBeforeSlot(int i, int n) {
        if (!repeatCombine) {
            return;
        }
        int len = n * repeatWidth;
        if (i == 0 && schedule.nInputLayers() == 0) {
            ops.copy(x.buffer(), x.offset(), repeatInput.buffer(), repeatInput.offset(), len);
        }
        if (i == schedule.recurrentStart()) {
            ops.copy(repeatInput.buffer(), repeatInput.offset(), repeatState.buffer(), repeatState.offset(), len);
        }
        if (schedule.loopStart(i)) {
            mix(MixMode.ADD, 1.0f, repeatState, repeatInput, len);
            ops.norm(NormKind.L2_GDN, null, null, schedule.loopEps(), x.buffer(), x.offset(), x.buffer(), x.offset(),
                    n, repeatWidth);
        }
    }

    /**
     * {@link LayerSchedule.LoopCombine#JUSTNORM} after slot {@code i}: a pass ends with {@code state = justNorm(x)};
     * the input stream is captured after the last input layer, and the first output layer starts from the state.
     */
    private void repeatAfterSlot(int i, int n) {
        if (!repeatCombine) {
            return;
        }
        int len = n * repeatWidth;
        if (schedule.loopEnd(i)) {
            ops.norm(NormKind.L2_GDN, null, null, schedule.loopEps(), x.buffer(), x.offset(), x.buffer(), x.offset(),
                    n, repeatWidth);
            ops.copy(x.buffer(), x.offset(), repeatState.buffer(), repeatState.offset(), len);
        }
        if (i == schedule.nInputLayers() - 1) {
            ops.copy(x.buffer(), x.offset(), repeatInput.buffer(), repeatInput.offset(), len);
        }
        if (i + 1 == schedule.recurrentEnd() && schedule.nOutputLayers() > 0) {
            ops.copy(repeatState.buffer(), repeatState.offset(), x.buffer(), x.offset(), len);
        }
    }

    // ================================================================== HRM

    /** {@code zH = embeddings}, {@code zL = broadcast(hrm.z_l_init)} (hrm-text.cpp:176-186). */
    private void startHrm(int n) {
        int len = n * hrmWidth;
        ops.copy(x.buffer(), x.offset(), hrmHigh.buffer(), hrmHigh.offset(), len);
        ops.fill(hrmLow.buffer(), 0, (long) len * Float.BYTES, ONE_BITS);
        ops.enqueue(NormKernels.MUL_ROW_SCALE, global(len), new long[]{local},
                NormKernels.mulRowScaleArgs(n, hrmWidth, hrmLowInit.shard(0), hrmWidth, hrmLow.buffer()));
    }

    /** Weightless RMS norm ending a pass (hrm-text.cpp:158), written to the stream the pass updates. */
    private void endHrmPass(int slot, LayerBlocks l, int n) {
        ops.norm(NormKind.RMS, null, null, l.normEps, x.buffer(), x.offset(), x.buffer(), x.offset(), n, hrmWidth);
        ScratchSlot target = schedule.stream(slot) == LayerSchedule.Stream.LOW ? hrmLow : hrmHigh;
        ops.copy(x.buffer(), x.offset(), target.buffer(), target.offset(), n * hrmWidth);
    }

    private long[] global(long n) {
        return new long[]{(n + local - 1) / local * local};
    }

    // ================================================================== memory

    /**
     * Lets every KV-cached mixer re-rotate its keys after {@code DeviceModelMemory.shiftPositions} moved the rows
     * (llama.cpp K-shift).
     *
     * @throws UnsupportedOperationException when a mixer cannot shift
     */
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        for (LayerBlocks l : layers) {
            GpuMixer owner = l.cacheMixer();
            if (owner != null) {
                owner.shiftContext(nKeep, nDiscard);
            }
        }
    }

    /**
     * Re-rotates every cache-owning mixer by the memory's last per-row shift deltas, without a keep/discard rule:
     * the position edits ({@code seq_add}, {@code seq_div}) re-rotate by the same {@code shift_delta} buffer a
     * paged shift fills.
     *
     * @throws UnsupportedOperationException when a cache-owning mixer cannot shift its memory
     */
    public void applyShiftDeltas() {
        ensureOpen();
        for (LayerBlocks l : layers) {
            GpuMixer owner = l.cacheMixer();
            if (owner != null) {
                owner.applyShiftDeltas();
            }
        }
    }

    /**
     * The bound cross-attention mixers of the {@link MixerKind#CROSS_ATTENTION_T5} layers in layer order, so the
     * encoder-decoder backend can hand them the encoder output before the first decoder step; empty without
     * cross-attention layers.
     */
    public List<GpuMixer> crossAttentionMixers() {
        ensureOpen();
        List<GpuMixer> out = new ArrayList<>();
        for (LayerBlocks l : layers) {
            if (l.crossMixer != null) {
                out.add(l.crossMixer);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Why {@link #shiftContext} would refuse, or null: the first refusal of a cache mixer
     * ({@link GpuMixer#shiftRefusal}), asked before the memory moves any row.
     */
    public String shiftRefusal() {
        ensureOpen();
        for (LayerBlocks l : layers) {
            GpuMixer owner = l.cacheMixer();
            String refusal = owner != null ? owner.shiftRefusal() : null;
            if (refusal != null) {
                return "layer " + l.spec.il() + ": " + refusal;
            }
        }
        return null;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("layer executor is closed");
        }
    }

    /** Closes every block this executor bound. The queue must be idle first. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        if (adaNorm != null) {
            try {
                adaNorm.close();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        closeAll(layers, failure);
        if (failure != null) {
            throw failure;
        }
    }

    private static void closeAll(List<LayerBlocks> list, RuntimeException primary) {
        RuntimeException first = primary;
        for (LayerBlocks l : list) {
            first = l.close(first);
        }
        if (first != null && primary == null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuLayerExecutor{" + layers.size() + " layers" + (schedule != null ? ", " + schedule : "") + '}';
    }

}