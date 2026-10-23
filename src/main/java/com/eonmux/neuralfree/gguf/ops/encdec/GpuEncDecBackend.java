package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.BuiltInGpuExtras;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuPrefillDriver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The encoder-decoder {@link ForwardBackend} on the GPU ({@link SupportClass#ENC_DEC}: T5), following llama.cpp's
 * encode-then-decode flow (llama.cpp tools/completion/completion.cpp:530-543, common/common.cpp:1515-1525): the source
 * is encoded once, then the decoder runs autoregressively from {@code decoder_start_token_id} over its own KV memory
 * and the cross-attention caches built from the encoder output.
 *
 * <p>{@link #prefill prefill(source, 0)}, in one queue submission with one host synchronization:
 * <ol>
 *   <li>the memory is reset (decoder self-attention KV, sequence) and the previous encoding is forgotten;</li>
 *   <li>the encoder pass ({@code llm_graph_type ENCODER}, llama.cpp src/models/t5.cpp:261-345): the source positions
 *       {@code 0 .. n-1} into {@link ScratchSlots#POS} (the bidirectional T5 bias of the memoryless attention reads
 *       them), the token embedding, the {@code enc.blk} stack ({@link GpuLayerExecutor#forStack}: non-causal
 *       attention without memory, ReLU or GEGLU FFN), and {@code enc.output_norm} (RMS, t5.cpp:340-344) from
 *       {@link ScratchSlots#X} into {@link ScratchSlots#NORMED};</li>
 *   <li>the cross-KV build: every decoder layer's {@link GpuCrossAttentionMixer#encode} projects the encoder output
 *       through {@code wk_cross} / {@code wv_cross} into its cross cache (t5.cpp:170-174);</li>
 *   <li>the decoder start token ({@code decoder_start_token_id}, else the vocabulary BOS, completion.cpp:535-538)
 *       at position 0 through the {@link GpuPrefillDriver}: the {@code dec.blk} trunk (causal T5-biased
 *       self-attention over the memory, {@code cross_attn_norm} and cross-attention, FFN; t5.cpp:117-235),
 *       {@code dec.output_norm} and the LM head (t5.cpp:237-254). The token is appended to the decoder
 *       {@code SequenceState}, so the next decoder position is 1, and its logits are returned.</li>
 * </ol>
 * {@link #decode}, {@link #decodeGreedyOnDevice} and {@link #forwardRows} continue the decoder sequence exactly as
 * {@code GpuForwardBackend} does for a decoder-only model (a call starting before the stored length rolls the tail
 * back first) and need a current encoding. {@link #hiddenRows} returns the encoder output (llama.cpp's encoder
 * {@code t_embd}, t5.cpp:344) without touching the memory or the cross caches; {@link #forwardAll} has no meaning
 * for an encoder-decoder model and is refused.</p>
 *
 * <p>The encoder runs the whole source in one step, as llama.cpp does (llama-context.cpp:1628), and the cross caches
 * hold {@code nCtx} rows, so a source longer than {@code min(ubatch, nCtx)} is refused with
 * {@link ContextFullException} before anything changes. The hidden tap observes the decoder stack. The context's
 * queue, scratch, weights, memory and placeholders belong to whoever built the context; this backend closes only the
 * blocks and head preprocessors it bound. Not thread-safe.</p>
 */
public final class GpuEncDecBackend implements ForwardBackend {

    private final GpuForwardContext ctx;
    private final ModelSpec spec;
    private final GpuEmbeddingStage embedding;
    private final GpuLayerExecutor encoder;
    private final GpuLayerExecutor decoder;
    private final GpuLogitsHead head;
    private final GpuPrefillDriver driver;
    private final List<GpuCrossAttentionMixer> crossMixers;
    private final GpuWeight encoderNorm;
    private final float encoderNormEps;
    private final int encoderWidth;
    private final int startToken;
    private final int sourceLimit;
    private final ScratchSlot x;
    private final ScratchSlot normed;
    private final ScratchSlot pos;
    private final MropeTextPositions textPositions;
    private final int[] posInts;
    private final ByteBuffer posBytes;
    private HiddenTap tap;
    private boolean encoded;
    private boolean closed;

    /**
     * Binds the encoder stack, the decoder trunk with its self- and cross-attention, and the head.
     *
     * @param ctx           context of the model, with the decoder memory (self-attention KV and cross caches)
     * @param registry      the mixer, FFN and extra factories of every family, including {@link EncDecBlocks} and
     *                      {@link BuiltInGpuExtras#factories()}
     * @param preprocessors head preprocessors of the families
     * @param vocabBos      the vocabulary's BOS id ({@code SpecialTokens.bos()}), the decoder start token when the
     *                      model declares none; {@code SpecialTokens.NULL} when the vocabulary has no BOS
     * @throws IllegalArgumentException      when the model is not an encoder-decoder model
     * @throws IllegalStateException         when the context has no memory, a decoder layer lacks cross-attention,
     *                                       or no decoder start token is defined
     * @throws UnsupportedOperationException when the registry lacks a block the model needs
     * @throws MissingTensorException        when {@code enc.output_norm.weight} is absent
     */
    public GpuEncDecBackend(GpuForwardContext ctx, GpuBlockRegistry registry, List<GpuHeadPreprocessor> preprocessors,
                            int vocabBos) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(preprocessors, "preprocessors");
        this.spec = ctx.spec();
        CpuEncDecBackend.checkModel(spec);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(arch() + ": the decoder needs memory for its self-attention KV and"
                    + " cross-attention caches; the context has none");
        }
        registry.validate(spec);
        ctx.kernels().prewarm(kernels());
        this.startToken = CpuEncDecBackend.decoderStartToken(spec, vocabBos);
        this.embedding = new GpuEmbeddingStage(ctx);
        if (startToken >= embedding.vocabRows()) {
            throw new IllegalStateException(arch() + ": decoder start token " + startToken + " is outside the "
                    + embedding.vocabRows() + " rows of the token table");
        }
        this.encoderNorm = encoderNorm(ctx.weights());
        this.encoderNormEps = spec.hparams().normRmsEps();
        this.encoderWidth = spec.encoderLayers().get(spec.encoderLayers().size() - 1).residual().streamWidth();
        this.textPositions = spec.hparams().mropeTextPositions();
        int ubatch = ctx.scratch().ubatch();
        this.x = ctx.scratch().slot(ScratchSlots.X).require((long) ubatch * encoderWidth);
        this.normed = ctx.scratch().slot(ScratchSlots.NORMED).require((long) ubatch * encoderWidth);
        this.pos = ctx.scratch().slot(ScratchSlots.POS).require((long) MropeTextPositions.SECTIONS * ubatch);
        this.posInts = new int[MropeTextPositions.SECTIONS * ubatch];
        this.posBytes = ByteBuffer.allocate(posInts.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.sourceLimit = Math.min(ubatch, CpuEncDecBackend.crossRows(spec, ctx.memory().plan()));

        GpuLayerExecutor enc = GpuLayerExecutor.forStack(ctx, registry, spec.encoderLayers());
        GpuLayerExecutor dec = null;
        GpuLogitsHead h = null;
        try {
            dec = GpuLayerExecutor.forTrunk(ctx, registry);
            h = new GpuLogitsHead(ctx, preprocessors);
            if (!h.hasLmHead()) {
                throw new IllegalStateException(arch() + ": the decoder has no LM head");
            }
            this.crossMixers = crossMixers(dec);
            this.driver = new GpuPrefillDriver(ctx, embedding, dec, h);
        } catch (RuntimeException e) {
            Closeables.close(h, e);
            Closeables.close(dec, e);
            Closeables.close(enc, e);
            throw e;
        }
        this.encoder = enc;
        this.decoder = dec;
        this.head = h;
        this.tap = ctx.hiddenTap();
    }

    /** Kernels of the executors, the head, the built-in extras and attention, for prewarming. */
    private static List<String> kernels() {
        Set<String> names = new LinkedHashSet<>();
        names.addAll(GpuLayerExecutor.KERNELS);
        names.addAll(BuiltInGpuExtras.KERNELS);
        names.addAll(GpuLogitsHead.KERNELS);
        names.add(GpuOps.UNARY);
        names.addAll(AttnKernels.all());
        return new ArrayList<>(names);
    }

    private GpuWeight encoderNorm(GpuWeights weights) {
        String name = TensorNames.weight(TensorNames.ENC_OUTPUT_NORM);
        GpuWeight w = weights.getOrNull(name);
        if (w == null) {
            throw new MissingTensorException(arch(), -1, name);
        }
        return w;
    }

    /** The bound cross-attention mixer of every decoder layer, in layer order. */
    private List<GpuCrossAttentionMixer> crossMixers(GpuLayerExecutor dec) {
        List<GpuMixer> bound = dec.crossAttentionMixers();
        int expected = 0;
        for (LayerSpec l : spec.layers()) {
            if (l.mixerKind() == MixerKind.CROSS_ATTENTION_T5) {
                expected++;
            }
        }
        if (expected != spec.nLayer() || bound.size() != expected) {
            throw new IllegalStateException(arch() + ": every one of the " + spec.nLayer() + " decoder layers needs"
                    + " cross-attention; the spec has " + expected + " and the executor bound " + bound.size());
        }
        List<GpuCrossAttentionMixer> out = new ArrayList<>(bound.size());
        for (GpuMixer m : bound) {
            if (!(m instanceof GpuCrossAttentionMixer)) {
                throw new IllegalStateException(arch() + ": the " + MixerKind.CROSS_ATTENTION_T5 + " factory built "
                        + m.getClass().getName() + ", the encoder-decoder backend needs "
                        + GpuCrossAttentionMixer.class.getSimpleName() + " to hand it the encoder output");
            }
            out.add((GpuCrossAttentionMixer) m);
        }
        return Collections.unmodifiableList(out);
    }

    // ================================================================== encoder + decoder start

    /**
     * Encodes {@code source}, builds the cross caches and evaluates the decoder start token at position 0.
     *
     * @param source encoder input tokens
     * @param pos0   must be 0: every prefill starts a new source and decoder sequence
     * @return the logits of the decoder start token
     * @throws ContextFullException     when the source is longer than {@code min(ubatch, nCtx)}; nothing changes
     * @throws IllegalArgumentException for an empty source, a token outside the vocabulary or {@code pos0 != 0}
     */
    @Override
    public float[] prefill(int[] source, int pos0) {
        ensureOpen();
        checkSource(source, pos0);
        GpuMemory memory = ctx.memory();
        encoded = false;
        memory.reset();
        try {
            encode(source, null);
            for (GpuCrossAttentionMixer m : crossMixers) {
                m.encode(normed.buffer(), normed.offset(), source.length);
            }
            encoded = true;
            return driver.run(new int[]{startToken}, null, 0, new int[]{0}, GpuPrefillDriver.Output.LOGITS, true,
                    tap).rows()[0];
        } catch (RuntimeException e) {
            encoded = false;
            drain(e);
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    private void checkSource(int[] source, int pos0) {
        Objects.requireNonNull(source, "source");
        if (pos0 != 0) {
            throw new IllegalArgumentException(arch() + ": an encoder-decoder prefill encodes a new source and"
                    + " restarts the decoder at position 0, got " + pos0);
        }
        if (source.length == 0) {
            throw new IllegalArgumentException("encoder source without tokens");
        }
        CpuEncDecBackend.checkSourceLength(source.length, ctx.scratch().ubatch(), sourceLimit);
        checkTokens(source);
    }

    private void checkTokens(int[] tokens) {
        long vocab = embedding.vocabRows();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] < 0 || tokens[i] >= vocab) {
                throw new IllegalArgumentException("token id " + tokens[i] + " at " + i + " is outside the " + vocab
                        + " rows of the token table");
            }
        }
    }

    /**
     * Enqueues the encoder pass over all of {@code tokens}: positions, embedding, the {@code enc.blk} stack and
     * {@code enc.output_norm} into {@link ScratchSlots#NORMED}.
     */
    private void encode(int[] tokens, int[] types) {
        int n = tokens.length;
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStepStart(0, n);
        }
        writePositions(n);
        embedding.embed(tokens, 0, n, types, 0);
        encoder.run(0, n, GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE);
        if (observed) {
            obs.onStageStart(-1, ExecutionObserver.Stage.OUTPUT_NORM);
        }
        ctx.ops().norm(NormKind.RMS, encoderNorm, null, encoderNormEps, x.buffer(), x.offset(), normed.buffer(),
                normed.offset(), n, encoderWidth);
        if (observed) {
            obs.onStageEnd(-1, ExecutionObserver.Stage.OUTPUT_NORM, normed, n, encoderWidth);
            obs.onStepEnd(0, n);
        }
    }

    /** Positions {@code 0 .. n-1} in the four-section layout into {@link ScratchSlots#POS} (one write). */
    private void writePositions(int n) {
        RopePositions.fillStep(textPositions, 0, n, posInts);
        int count = MropeTextPositions.SECTIONS * n;
        posBytes.clear();
        for (int i = 0; i < count; i++) {
            posBytes.putInt(posInts[i]);
        }
        GpuEvent ev = ctx.queue().enqueueWriteBuffer(pos.buffer(), (long) pos.offset() * Float.BYTES,
                (long) count * Integer.BYTES, posBytes.array());
        if (ev != null) {
            ev.close();
        }
    }

    // ================================================================== decoder steps

    @Override
    public float[] decode(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, GpuPrefillDriver.Output.LOGITS).rows()[0];
    }

    /** Greedy selection on the device ({@code LLM_Argmax}); only the id is read back. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, GpuPrefillDriver.Output.ARGMAX).ids()[0];
    }

    /** Decoder tokens at {@code pos0 ..} over the current encoding (speculative verification of the decoder). */
    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return causal(tokens, pos0, outputRows, GpuPrefillDriver.Output.LOGITS).rows();
    }

    private GpuPrefillDriver.Result causal(int[] tokens, int pos0, int[] rows, GpuPrefillDriver.Output output) {
        ensureOpen();
        requireEncoded();
        align(ctx.memory(), pos0, tokens.length);
        return driver.run(tokens, null, pos0, rows, output, true, tap);
    }

    /**
     * Rolls the decoder memory back to {@code pos0} when the call re-evaluates stored positions; checks the context
     * length first, so a call that cannot fit leaves the memory untouched.
     */
    private static void align(GpuMemory memory, int pos0, int nTokens) {
        if (pos0 < 0) {
            throw new IllegalArgumentException("position " + pos0 + " is negative");
        }
        int nCtx = memory.plan().nCtx();
        long end = (long) pos0 + nTokens;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(Integer.MAX_VALUE, end), nCtx);
        }
        int past = memory.sequence().nPast();
        if (pos0 > past) {
            throw new IllegalArgumentException("position " + pos0 + " leaves a gap after the " + past
                    + " stored decoder positions");
        }
        if (pos0 < past) {
            int drop = past - pos0;
            if (!memory.canRollback(drop)) {
                throw new IllegalStateException("the call starts at " + pos0 + " but the decoder memory holds " + past
                        + " positions and cannot drop the last " + drop);
            }
            memory.rollback(drop);
        }
    }

    private void requireEncoded() {
        if (!encoded) {
            throw new IllegalStateException(arch() + ": the decoder runs over an encoded source; call prefill(source,"
                    + " 0) after every reset");
        }
    }

    // ================================================================== encoder output

    /**
     * Not defined for an encoder-decoder model: the encoder has no LM head and the decoder needs an encoding.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        ensureOpen();
        throw new UnsupportedOperationException(arch() + " is an encoder-decoder model: encode a source with"
                + " prefill(source, 0) and decode, or read the encoder output with hiddenRows");
    }

    /**
     * The encoder output rows after {@code enc.output_norm} ({@code [n][n_embd]}), the embeddings llama.cpp reports
     * for an encoder-decoder model. Neither the decoder memory nor the cross caches change.
     *
     * @throws IllegalArgumentException for an empty input, a token outside the vocabulary, mismatched types or more
     *                                  tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        int n = tokens.length;
        int ubatch = ctx.scratch().ubatch();
        if (n == 0) {
            throw new IllegalArgumentException("encoder input without tokens");
        }
        if (n > ubatch) {
            throw new IllegalArgumentException(n + " tokens exceed the ubatch of " + ubatch + "; the encoder runs in"
                    + " one step, raise neuralfree.ubatch");
        }
        if (tokenTypes != null && tokenTypes.length != n) {
            throw new IllegalArgumentException(tokenTypes.length + " token types for " + n + " tokens");
        }
        checkTokens(tokens);
        ComputeQueue queue = ctx.queue();
        float[] all = new float[Math.multiplyExact(n, encoderWidth)];
        GpuEvent read = null;
        try {
            encode(tokens, tokenTypes);
            read = queue.enqueueReadBuffer(normed.buffer(), (long) normed.offset() * Float.BYTES,
                    (long) all.length * Float.BYTES, all);
            queue.finish();
        } catch (RuntimeException e) {
            drain(e);
            throw e;
        } finally {
            if (read != null) {
                read.close();
            }
        }
        float[][] rows = new float[n][];
        for (int t = 0; t < n; t++) {
            rows[t] = new float[encoderWidth];
            System.arraycopy(all, t * encoderWidth, rows[t], 0, encoderWidth);
        }
        return rows;
    }

    // ================================================================== memory

    /** Empties the decoder memory and forgets the encoding; the next decoder call needs a new prefill. */
    @Override
    public void reset() {
        ensureOpen();
        encoded = false;
        ctx.memory().reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        ensureOpen();
        return ctx.memory().canRollback(nTokens);
    }

    /** Drops decoder positions; the encoding stays valid. */
    @Override
    public void rollback(int nTokens) {
        ensureOpen();
        ctx.memory().rollback(nTokens);
    }

    /**
     * Context shift of the decoder self-attention cache, as {@code GpuForwardBackend}: refused before any row moves
     * by the model ({@link ModelSpec#contextShiftRefusal()}) or a decoder cache mixer
     * ({@link GpuLayerExecutor#shiftRefusal}); the cross caches hold the encoder output, which has no decoder
     * positions, and stay. A mixer failure after the rows moved resets the memory and forgets the encoding.
     *
     * @throws UnsupportedOperationException when the decoder memory cannot shift
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch() + ": context shift is not supported (" + refusal + ")");
        }
        GpuMemory memory = ctx.memory();
        if (!(memory instanceof DeviceModelMemory)) {
            throw new UnsupportedOperationException(arch() + ": context shift needs a device-owned memory, got "
                    + memory.getClass().getSimpleName());
        }
        refusal = decoder.shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch() + ": context shift is not supported (" + refusal + ")");
        }
        ((DeviceModelMemory) memory).shiftPositions(nKeep, nDiscard);
        try {
            decoder.shiftContext(nKeep, nDiscard);
        } catch (RuntimeException e) {
            encoded = false;
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            String msg = arch() + ": context shift failed after the cache rows moved; the memory was reset";
            if (e instanceof UnsupportedOperationException) {
                throw new UnsupportedOperationException(msg, e);
            }
            throw new IllegalStateException(msg, e);
        }
    }

    /** The decoder memory (self-attention KV, cross caches and the decoder sequence). */
    @Override
    public ModelMemory memory() {
        ensureOpen();
        return ctx.memory();
    }

    /** Observer of the decoder stack's tap points for every later call. */
    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        ensureOpen();
        this.tap = tap;
    }

    /** The decoder start token this backend evaluates after each encoding. */
    public int decoderStartToken() {
        return startToken;
    }

    /** Longest source {@link #prefill} accepts: {@code min(ubatch, cross cache rows)}. */
    public int maxSourceTokens() {
        return sourceLimit;
    }

    /** True once a source is encoded and until the next reset. */
    public boolean encoded() {
        return encoded;
    }

    private String arch() {
        return spec.arch().ggufName();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GPU encoder-decoder backend is closed");
        }
    }

    /** Waits for the queue after a failed enqueue, so no pending read outlives the call. */
    private void drain(RuntimeException e) {
        try {
            ctx.queue().finish();
        } catch (RuntimeException drainFailure) {
            if (drainFailure != e) {
                e.addSuppressed(drainFailure);
            }
        }
    }

    // ================================================================== lifecycle

    /** Waits for the queue, then closes the bound blocks and head preprocessors. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        try {
            ctx.queue().finish();
        } catch (RuntimeException e) {
            first = e;
        }
        first = Closeables.close(decoder, first);
        first = Closeables.close(encoder, first);
        first = Closeables.close(head, first);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuEncDecBackend{" + arch() + ", encoder " + spec.encoderLayers().size() + " layers, decoder "
                + spec.nLayer() + " layers, start " + startToken + (encoded ? ", encoded" : "")
                + (closed ? ", closed" : "") + '}';
    }
}
