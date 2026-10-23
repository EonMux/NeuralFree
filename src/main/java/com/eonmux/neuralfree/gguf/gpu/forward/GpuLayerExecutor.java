package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
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
import com.eonmux.neuralfree.gguf.spec.TensorRef;
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
            NormKernels.ROW_NORM, NormKernels.RESIDUAL_MIX, NormKernels.MUL_ROW_SCALE, GpuOps.COPY2D, GpuOps.AXPBY));

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
    private final int ubatch;
    private final int local;
    private final GpuBuffer placeholder;
    /** Adaptive FFN-norm vectors shared by the layers with {@code NormSpec.ffnAda()}; null when none has it. */
    private final GpuAdaNorm adaNorm;
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
        this.layers = Collections.unmodifiableList(bindAll(ctx, registry, stack, schedule));
        try {
            for (LayerBlocks l : layers) {
                if (l.outScale != null && x.offset() != 0) {
                    throw new IllegalStateException("layer_output_scale needs the X slot at element 0, it starts at "
                            + x.offset());
                }
            }
            this.adaNorm = adaNorm(ctx, layers);
            prewarmKernels(hrm);
        } catch (RuntimeException e) {
            closeAll(layers, e);
            throw e;
        }
    }

    /**
     * Prepares the kernels this executor launches at work-group size {@code L} ({@link GpuOps#prewarm}):
     * {@value NormKernels#RESIDUAL_MIX} for every residual, and {@value NormKernels#MUL_ROW_SCALE} when a layer has
     * {@code layer_output_scale} or a weighted-input post-FFN norm, or the schedule is HRM.
     */
    private void prewarmKernels(boolean hrm) {
        long[] size = {local};
        ops.prewarm(NormKernels.RESIDUAL_MIX, size);
        boolean rowScale = hrm;
        for (LayerBlocks l : layers) {
            rowScale |= l.outScale != null || (l.postFfn != null && l.postFfn.weightFirst);
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
            if (sink.wants(i)) {
                sink.capture(i, x, nTokens, l.streamWidth);
            }
            runLayer(l, pos0, nTokens, mode);
            if (hrm && schedule.passEnd(i)) {
                endHrmPass(i, l, nTokens);
            }
            ctx.queue().flush();
        }
        int out = layers.size();
        if (sink.wants(out)) {
            sink.capture(out, x, nTokens, outputWidth());
        }
    }

    private void runLayer(LayerBlocks l, int pos0, int n, StepMode mode) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        int il = l.spec.il();
        if (observed) {
            obs.onStageStart(il, ExecutionObserver.Stage.LAYER);
        }
        int streamLen = n * l.streamWidth;
        if (l.keepInput) {
            ops.copy(x.buffer(), x.offset(), layerInput.buffer(), layerInput.offset(), streamLen);
        }
        for (GpuLayerExtra e : l.extras) {
            e.beforeBlock(pos0, n, x);
        }
        ScratchSlot deferred = mixerBlock(l, pos0, n, mode);
        if (l.crossMixer != null) {
            crossBlock(l, pos0, n, mode);
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
        if (observed) {
            obs.onStageEnd(il, ExecutionObserver.Stage.LAYER, x, n, l.streamWidth);
        }
    }

    /**
     * Steps 2-6. Returns the mixer branch when a PARALLEL residual defers its add to step 11, else null.
     */
    private ScratchSlot mixerBlock(LayerBlocks l, int pos0, int n, StepMode mode) {
        int il = l.spec.il();
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
     * {@code ffn} tells the FFN sub-block from the mixer.
     */
    private void residual(LayerBlocks l, ScratchSlot in, ScratchSlot branchOut, int len, boolean ffn) {
        switch (l.kind) {
            case STANDARD:
            case PARALLEL:
                mix(MixMode.ADD, 1.0f, x, branchOut, len);
                break;
            case SCALED:
                mix(MixMode.SCALED, l.residualScale, x, branchOut, len);
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
            default:
                throw new IllegalStateException("layer " + l.spec.il() + ": no executor residual for " + l.kind);
        }
    }

    /** {@code X = mix(inp, cur)} over {@code n} floats ({@value NormKernels#RESIDUAL_MIX}). */
    private void mix(MixMode mode, float s, ScratchSlot inp, ScratchSlot cur, int n) {
        ops.enqueue(NormKernels.RESIDUAL_MIX, global(n), new long[]{local},
                NormKernels.residualMixArgs(mode, n, s, inp.buffer(), inp.offset(), cur.buffer(), cur.offset(),
                        placeholder, placeholder, 1, x.buffer(), x.offset(), placeholder));
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

    // ================================================================== bound layer

    /** Norm operands: a weight and bias, or neither for a weightless norm. */
    static final class Norm {
        static final Norm WEIGHTLESS = new Norm(null, null);

        final GpuWeight weight;
        final GpuWeight bias;
        /** Epsilon of this norm; NaN uses the layer's norm epsilon. */
        final float eps;
        /** {@code norm(x * weight)} instead of {@code norm(x) * weight} (muse-spark post-FFN norm). */
        final boolean weightFirst;
        /** Equal row slices normalized on their own before the full-width weight; 1 for a plain row norm. */
        final int groups;

        Norm(GpuWeight weight, GpuWeight bias) {
            this(weight, bias, Float.NaN, false, 1);
        }

        private Norm(GpuWeight weight, GpuWeight bias, float eps, boolean weightFirst, int groups) {
            this.weight = weight;
            this.bias = bias;
            this.eps = eps;
            this.weightFirst = weightFirst;
            this.groups = groups;
        }

        /** This norm applied with {@code eps}. */
        Norm withEps(float e) {
            return new Norm(weight, bias, e, weightFirst, groups);
        }

        /**
         * This norm over {@code g} equal slices of each row, each normalized on its own, then multiplied by the
         * full-width weight (k2-horizon.cpp:124-135).
         *
         * @throws IllegalArgumentException when the weight is not one single-buffer F32 vector of {@code width}
         *                                  values, a bias is present, or {@code width} is not a multiple of {@code g}
         */
        Norm grouped(String arch, int il, int width, int g) {
            if (g == 1) {
                return this;
            }
            if (weightFirst || weight == null || bias != null || weight.type() != GGMLType.F32
                    || !weight.isSingleShard() || weight.rows() != 1 || weight.cols() != width || width % g != 0) {
                throw new IllegalArgumentException(arch + " layer " + il + ": a norm over " + g + " groups needs a"
                        + " single-buffer F32 weight vector of " + width + " values, a width divisible by the groups"
                        + " and no bias, got " + weight);
            }
            return new Norm(weight, null, eps, false, g);
        }

        /**
         * This norm with its weight applied before the normalization.
         *
         * @throws IllegalArgumentException when the weight is not one single-buffer F32 vector without a bias
         */
        Norm weightFirst(String arch, int il, int width) {
            if (weight == null || bias != null || weight.type() != GGMLType.F32 || !weight.isSingleShard()
                    || weight.rows() != 1 || weight.cols() != width) {
                throw new IllegalArgumentException(arch + " layer " + il + ": a norm applied to the weighted input"
                        + " needs a single-buffer F32 weight vector of " + width + " values and no bias, got " + weight);
            }
            return new Norm(weight, null, eps, true, 1);
        }
    }

    /** Branch norm tensor names: {@link NormResolver} for main-stack layers, plain names for other stacks. */
    static final class NormNames {
        private final ModelSpec spec;
        private NormResolver resolver;

        NormNames(ModelSpec spec) {
            this.spec = spec;
        }

        private boolean resolvable(LayerSpec l) {
            return TensorNames.STACK_BLK.equals(l.stack()) && l.weightLayer() < spec.hparams().nLayerAll();
        }

        private NormResolver resolver() {
            if (resolver == null) {
                resolver = new NormResolver(spec.hparams(), spec.tensors());
            }
            return resolver;
        }

        /** FFN input norm tensor of a layer; null when the FFN input is not normed by a tensor. */
        TensorRef ffnNorm(LayerSpec l) {
            if (resolvable(l)) {
                return resolver().ffnNormTensor(l.weightLayer());
            }
            NormSpec ns = l.norm();
            if (ns.ffnInputIsPostAttnNorm()) {
                return layer(l, TensorNames.ATTN_POST_NORM);
            }
            if (ns.attnNorm2() && l.residual().ffnUsesAttnNorm2()) {
                return layer(l, TensorNames.ATTN_NORM_2);
            }
            return ns.ffnNorm() ? layer(l, TensorNames.FFN_NORM) : null;
        }

        /** Post-attention norm tensor of a layer; null when it has none. */
        TensorRef postAttnNorm(LayerSpec l) {
            if (resolvable(l)) {
                return resolver().postAttnNormTensor(l.weightLayer());
            }
            return layer(l, TensorNames.ATTN_POST_NORM);
        }

        /** Post-FFN norm tensor of a layer; null when it has none. */
        TensorRef postFfnNorm(LayerSpec l) {
            if (resolvable(l)) {
                return resolver().postFfnNormTensor(l.weightLayer());
            }
            return layer(l, TensorNames.FFN_POST_NORM);
        }

        private TensorRef layer(LayerSpec l, String base) {
            return spec.tensors().layer(l.stack(), l.weightLayer(), TensorNames.weight(base));
        }
    }

    /** One layer's bound blocks and resolved norm operands. */
    static final class LayerBlocks {
        final LayerSpec spec;
        final int streamWidth;
        final int width;
        final ResidualKind kind;
        final float residualScale;
        final boolean mixerResidual;
        final boolean discard;
        final boolean keepInput;
        final float blockScale;
        final NormKind normKind;
        final float normEps;
        final List<GpuLayerExtra> extras;
        GpuMixer mixer;
        GpuMixer parallelAttention;
        GpuMixer crossMixer;
        GpuFfnBlock ffn;
        Norm attnNorm;
        Norm postAttn;
        Norm attnOutBranch;
        Norm postLnAttn;
        Norm postLnAttn2;
        Norm postLnLayer;
        Norm crossNorm;
        Norm ffnNorm;
        Norm postFfn;
        boolean sharedFfnNorm;
        GpuWeight outScale;
        /** Adaptive conditioning of {@link #ffnNorm}'s output, or null. */
        GpuAdaNorm.Layer ada;

        private LayerBlocks(ModelSpec model, LayerSpec spec) {
            this.spec = spec;
            ResidualSpec r = spec.residual();
            this.streamWidth = r.streamWidth();
            this.width = r.streamWidth() / r.hcMult();
            this.kind = r.kind();
            this.residualScale = r.residualScale();
            this.mixerResidual = r.mixerResidual();
            this.discard = r.discardLayerOutput();
            this.keepInput = keepsLayerInput(model, spec);
            this.blockScale = r.blockOutputScale();
            this.normKind = spec.norm().kind();
            this.normEps = spec.norm().eps();
            this.extras = new ArrayList<>();
        }

        /**
         * BERT-graph post-LN: no {@code attn_norm}, {@code attn_output_norm} on the stream after the mixer residual
         * and {@code layer_output_norm} after the FFN residual (bert.cpp:156-215).
         */
        static boolean postLayerNorm(NormSpec ns) {
            return !ns.attnNorm() && ns.postRole() == PostNormRole.NONE
                    && (ns.attnOutputNorm() || ns.layerOutputNorm());
        }

        /**
         * falcon-40b: {@code attn_norm_2} of the layer input feeds the attention and {@code attn_norm} of the same
         * input feeds the FFN of the parallel residual (falcon.cpp:71-88, 125).
         */
        static boolean falconNorm2(LayerSpec l) {
            NormSpec ns = l.norm();
            return l.residual().kind() == ResidualKind.PARALLEL && l.residual().ffnUsesAttnNorm2() && ns.attnNorm2()
                    && ns.attnNorm() && ns.postRole() != PostNormRole.POST_ONLY;
        }

        /**
         * jina-bert-v2: after {@code attn_output_norm} the layer input is added again and {@code attn_norm_2} is
         * applied (bert.cpp:162-165).
         */
        static boolean postLnAttnNorm2(ModelSpec model, LayerSpec l) {
            NormSpec ns = l.norm();
            return postLayerNorm(ns) && ns.attnOutputNorm() && !ns.weightless() && l.mixerKind() != MixerKind.NONE
                    && model.tensors().layer(l.stack(), l.weightLayer(),
                    TensorNames.weight(TensorNames.ATTN_NORM_2)) != null;
        }

        /** True when the layer reads its input again after the block started ({@code discardLayerOutput}, BERT). */
        static boolean keepsLayerInput(ModelSpec model, LayerSpec l) {
            return l.residual().discardLayerOutput() || postLnAttnNorm2(model, l);
        }

        /** The mixer that owns the layer's KV rows (the falcon-h1 parallel attention), or null without KV. */
        GpuMixer cacheMixer() {
            if (!spec.usesKv()) {
                return null;
            }
            return parallelAttention != null ? parallelAttention : mixer;
        }

        static LayerBlocks bind(GpuForwardContext ctx, GpuBlockRegistry registry, LayerSpec spec, NormNames names) {
            LayerBlocks b = new LayerBlocks(ctx.spec(), spec);
            try {
                b.resolveNorms(ctx, names);
                b.bindBlocks(ctx, registry);
            } catch (RuntimeException e) {
                b.close(e);
                throw e;
            }
            return b;
        }

        private void resolveNorms(GpuForwardContext ctx, NormNames names) {
            NormSpec ns = spec.norm();
            GpuWeights w = ctx.weightSource().weightsFor(spec.weightLayer());
            String arch = ctx.spec().arch().ggufName();
            boolean weightless = ns.weightless();
            PostNormRole role = ns.postRole();
            boolean postLn = postLayerNorm(ns);
            boolean postBranches = role == PostNormRole.SANDWICH || role == PostNormRole.POST_ONLY;
            boolean falconNorm2 = falconNorm2(spec);
            if (ns.attnNorm() && role != PostNormRole.POST_ONLY) {
                attnNorm = named(w, arch, falconNorm2 ? TensorNames.ATTN_NORM_2 : TensorNames.ATTN_NORM, weightless);
            }
            if (postBranches && ns.postAttnNorm()) {
                postAttn = weightless || ns.postAttnWeightless() ? Norm.WEIGHTLESS
                        : fromRef(w, arch, names.postAttnNorm(spec), "post-attention norm");
                postAttn = postAttn.withEps(ns.postEps());
            }
            if (ns.attnOutputNorm()) {
                Norm n = named(w, arch, TensorNames.ATTN_OUT_NORM, weightless);
                if (postLn) {
                    postLnAttn = n;
                    if (postLnAttnNorm2(ctx.spec(), spec)) {
                        postLnAttn2 = named(w, arch, TensorNames.ATTN_NORM_2, false);
                    }
                } else {
                    attnOutBranch = n;
                }
            }
            if (postLn && ns.layerOutputNorm()) {
                postLnLayer = named(w, arch, TensorNames.LAYER_OUT_NORM, weightless);
            }
            if (spec.crossAttention() != null) {
                crossNorm = named(w, arch, TensorNames.DEC_CROSS_ATTN_NORM, weightless);
            }
            sharedFfnNorm = !falconNorm2 && ns.sharedAttnNormForFfn() && attnNorm != null
                    && spec.ffnKind() != FfnKind.NONE;
            if (spec.ffnKind() != FfnKind.NONE && !sharedFfnNorm) {
                ffnNorm = falconNorm2 ? named(w, arch, TensorNames.ATTN_NORM, weightless)
                        : ffnInputNorm(w, arch, names, ns, weightless);
            }
            if (spec.ffnKind() != FfnKind.NONE && postBranches && ns.postFfnNorm()) {
                postFfn = weightless ? Norm.WEIGHTLESS : fromRef(w, arch, names.postFfnNorm(spec), "post-FFN norm");
                postFfn = postFfn.withEps(ns.postEps());
                if (ns.postFfnWeightFirst()) {
                    postFfn = postFfn.weightFirst(arch, spec.il(), width);
                }
            }
            if (ns.groups() > 1) {
                if (attnNorm != null) {
                    attnNorm = attnNorm.grouped(arch, spec.il(), width, ns.groups());
                }
                if (ffnNorm != null) {
                    ffnNorm = ffnNorm.grouped(arch, spec.il(), width, ns.groups());
                }
            }
            if (ns.ffnAda()) {
                if (ffnNorm == null) {
                    throw new IllegalStateException(arch + " layer " + spec.il() + ": the ada norm conditions an"
                            + " ffn_norm the layer does not apply");
                }
                GpuWeight down = vector(w, arch, ns.ffnAdaDown().name());
                GpuWeight up = vector(w, arch, ns.ffnAdaUp().name());
                GpuProjectionDeltas deltas = ctx.projectionDeltas();
                ada = new GpuAdaNorm.Layer(down, up,
                        deltas.serve(spec, TensorNames.weight(TensorNames.FFN_NORM_ADA_DOWN), down),
                        deltas.serve(spec, TensorNames.weight(TensorNames.FFN_NORM_ADA_UP), up), width);
            }
            TensorRef scale = ns.layerOutputScale();
            if (scale != null) {
                outScale = vector(w, arch, scale.name());
                boolean fits = outScale.cols() == 1 || outScale.cols() == streamWidth;
                if (outScale.type() != GGMLType.F32 || !outScale.isSingleShard() || outScale.rows() != 1 || !fits) {
                    throw new IllegalArgumentException(arch + " layer " + spec.il() + ": layer_output_scale must be"
                            + " a single-buffer F32 vector of 1 or " + streamWidth + " values, got " + outScale);
                }
            }
        }

        private Norm ffnInputNorm(GpuWeights w, String arch, NormNames names, NormSpec ns, boolean weightless) {
            boolean expected = ns.ffnNorm() || ns.ffnInputIsPostAttnNorm()
                    || (ns.attnNorm2() && spec.residual().ffnUsesAttnNorm2());
            if (!expected) {
                return null;
            }
            if (weightless) {
                return Norm.WEIGHTLESS;
            }
            TensorRef ref = names.ffnNorm(spec);
            if (ref == null) {
                throw new MissingTensorException(arch, spec.il(), spec.tensorName(TensorNames.weight(
                        TensorNames.FFN_NORM)));
            }
            return fromRef(w, arch, ref, "FFN norm");
        }

        private Norm named(GpuWeights w, String arch, String base, boolean weightless) {
            if (weightless) {
                return Norm.WEIGHTLESS;
            }
            GpuWeight weight = w.layerOrNull(spec, TensorNames.weight(base));
            if (weight == null) {
                throw new MissingTensorException(arch, spec.il(), spec.tensorName(TensorNames.weight(base)));
            }
            return new Norm(weight, w.layerOrNull(spec, TensorNames.bias(base)));
        }

        private Norm fromRef(GpuWeights w, String arch, TensorRef ref, String what) {
            if (ref == null) {
                throw new IllegalStateException(arch + " layer " + spec.il() + ": the spec has a " + what
                        + " but no tensor names it");
            }
            GpuWeight weight = vector(w, arch, ref.name());
            String bias = biasName(ref.name());
            return new Norm(weight, bias != null ? w.find(bias) : null);
        }

        private GpuWeight vector(GpuWeights w, String arch, String name) {
            GpuWeight v = w.find(name);
            if (v == null) {
                v = w.getOrNull(name);
            }
            if (v == null) {
                throw new MissingTensorException(arch, spec.il(), name);
            }
            return v;
        }

        private static String biasName(String weightName) {
            String suffix = "." + TensorNames.WEIGHT;
            return weightName.endsWith(suffix)
                    ? weightName.substring(0, weightName.length() - suffix.length()) + "." + TensorNames.BIAS : null;
        }

        private void bindBlocks(GpuForwardContext ctx, GpuBlockRegistry registry) {
            if (spec.mixerKind() == MixerKind.CROSS_ATTENTION_T5) {
                LayerSpec self = attentionView(spec);
                mixer = registry.mixer(self);
                mixer.bind(ctx, self);
                crossMixer = registry.mixer(spec);
                crossMixer.bind(ctx, spec);
            } else if (spec.mixerKind() != MixerKind.NONE) {
                mixer = registry.mixer(spec);
                mixer.bind(ctx, spec);
                if (spec.parallelAttention()) {
                    LayerSpec attn = attentionView(spec);
                    parallelAttention = registry.mixer(attn);
                    parallelAttention.bind(ctx, attn);
                }
            }
            if (spec.ffnKind() != FfnKind.NONE) {
                ffn = registry.ffn(spec);
                ffn.bind(ctx, spec);
            }
            extras.addAll(registry.extras(spec));
            for (GpuLayerExtra e : extras) {
                e.bind(ctx, spec);
            }
        }

        /**
         * The layer's softmax-attention branch as a plain ATTENTION layer (T5 decoder self-attention, falcon-h1
         * parallel attention), bound through the registry's ATTENTION factory.
         */
        static LayerSpec attentionView(LayerSpec l) {
            return LayerSpec.builder(l.il(), MixerKind.ATTENTION, FfnKind.NONE)
                    .weightLayer(l.weightLayer())
                    .stack(l.stack())
                    .norm(l.norm())
                    .residual(l.residual())
                    .attention(l.attention())
                    .rope(l.rope())
                    .build();
        }

        /** Closes every bound block; returns {@code first} or the first close failure. */
        RuntimeException close(RuntimeException first) {
            RuntimeException err = first;
            List<AutoCloseable> all = new ArrayList<>();
            all.add(mixer);
            all.add(parallelAttention);
            all.add(crossMixer);
            all.add(ffn);
            all.addAll(extras);
            for (AutoCloseable c : all) {
                if (c == null) {
                    continue;
                }
                try {
                    c.close();
                } catch (Exception e) {
                    RuntimeException re = e instanceof RuntimeException ? (RuntimeException) e
                            : new IllegalStateException("closing " + c + " failed", e);
                    if (err == null) {
                        err = re;
                    } else if (err != re) {
                        err.addSuppressed(re);
                    }
                }
            }
            mixer = null;
            parallelAttention = null;
            crossMixer = null;
            ffn = null;
            extras.clear();
            return err;
        }
    }
}
