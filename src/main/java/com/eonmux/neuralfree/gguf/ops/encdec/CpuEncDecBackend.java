package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuLayerExecutor;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.head.CpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.cpu.head.CpuLogitsHead;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The encoder-decoder {@link ForwardBackend} on the CPU, the twin of {@link GpuEncDecBackend} with the same call
 * semantics (T5; llama.cpp tools/completion/completion.cpp:530-543): {@link #prefill prefill(source, 0)} resets the
 * memory, runs the {@code enc.blk} stack over the source at positions {@code 0 .. n-1} (non-causal attention without
 * memory, llama.cpp src/models/t5.cpp:261-345), applies {@code enc.output_norm}, hands the result to every decoder
 * layer's {@link CpuCrossAttentionMixer#encode} (cross K/V, t5.cpp:170-174) and evaluates the decoder start token at
 * position 0 (t5.cpp:117-254), storing it in the decoder sequence and returning its logits. Later decoder calls
 * continue that sequence as {@code CpuForwardBackend} does and need a current encoding; {@link #hiddenRows} returns the
 * encoder output; {@link #forwardAll} is refused.
 *
 * <p>Per decoder step of at most {@code ubatch} tokens: the embedding stage writes the residual stream and the step
 * inputs, {@link CpuMemory#beginStep} marks the step's rows, the {@code dec.blk} executors run in order (each runs
 * the self-attention, {@code cross_attn_norm}, the cross-attention and the FFN), the tokens are appended to the
 * sequence and the head computes the requested rows. The hidden tap observes the decoder stack.</p>
 *
 * <p>The package-level helpers shared by both encoder-decoder backends (model checks, the decoder start token, the
 * source limit) live here. The backend owns its executors and head; the context's weights, scratch and memory
 * belong to whoever built the context. One thread at a time.</p>
 */
public final class CpuEncDecBackend implements ForwardBackend {

    private static final String TEMP_ROWS = "CpuEncDecBackend.rows";
    private static final String TEMP_OUTPUTS = "CpuEncDecBackend.outputs";

    private final ModelSpec spec;
    private final CpuScratch scratch;
    private final CpuMemory memory;
    private final int ubatch;
    private final CpuEmbeddingStage embedding;
    private final List<CpuLayerExecutor> encoder;
    private final List<CpuLayerExecutor> decoder;
    private final List<CpuCrossAttentionMixer> crossMixers;
    private final CpuLogitsHead head;
    private final float[] encoderNorm;
    private final float encoderNormEps;
    private final int encoderWidth;
    private final int decoderWidth;
    private final int startToken;
    private final int sourceLimit;
    private final int[] rowIdx;
    private final int[] outputIdx;
    private final int[] singleToken = new int[1];
    private HiddenTap tap;
    private float[] lastLogits;
    private boolean encoded;
    private boolean closed;

    /**
     * Builds and binds the encoder stack, the decoder trunk and the head.
     *
     * @param ctx               context of the model, with the decoder memory
     * @param registry          the CPU blocks, including {@link EncDecBlocks} and the built-in extras
     * @param headPreprocessors candidate head preprocessors; those that apply are bound and owned by the head
     * @param vocabBos          the vocabulary's BOS id, the decoder start token when the model declares none;
     *                          {@code SpecialTokens.NULL} when the vocabulary has none
     * @throws IllegalArgumentException      when the model is not an encoder-decoder model
     * @throws IllegalStateException         when the context has no memory, a decoder layer lacks cross-attention,
     *                                       or no valid decoder start token is defined
     * @throws UnsupportedOperationException when the registry lacks a block the model needs
     * @throws MissingTensorException        when {@code enc.output_norm.weight} is absent
     */
    public CpuEncDecBackend(CpuForwardContext ctx, CpuBlockRegistry registry,
                            List<CpuHeadPreprocessor> headPreprocessors, int vocabBos) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(headPreprocessors, "headPreprocessors");
        this.spec = ctx.spec();
        checkModel(spec);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(arch() + ": the decoder needs memory for its self-attention KV and"
                    + " cross-attention caches; the context has none");
        }
        registry.validate(spec);
        this.scratch = ctx.scratch();
        this.memory = ctx.memory();
        this.ubatch = Math.min(scratch.ubatch(), memory.plan().ubatch());
        if (ubatch < 1) {
            throw new IllegalArgumentException("memory plan ubatch " + memory.plan().ubatch() + " is below 1");
        }
        this.tap = ctx.hiddenTap();
        this.startToken = decoderStartToken(spec, vocabBos);
        this.encoderWidth = spec.encoderLayers().get(spec.encoderLayers().size() - 1).residual().streamWidth();
        this.decoderWidth = spec.layer(spec.nLayer() - 1).residual().streamWidth();
        this.encoderNorm = encoderNorm(ctx, encoderWidth);
        this.encoderNormEps = spec.hparams().normRmsEps();
        this.sourceLimit = Math.min(scratch.ubatch(), crossRows(spec, memory.plan()));
        this.rowIdx = scratch.intTemp(TEMP_ROWS, scratch.ubatch());
        this.outputIdx = scratch.intTemp(TEMP_OUTPUTS, scratch.ubatch());
        this.embedding = new CpuEmbeddingStage(ctx);
        try {
            embedding.checkInputs(new int[]{startToken}, null, 0);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(arch() + ": decoder start token " + startToken + " is not a valid input",
                    e);
        }
        List<CpuLayerExecutor> bound = new ArrayList<>();
        List<CpuLayerExecutor> enc;
        List<CpuLayerExecutor> dec;
        CpuLogitsHead h = null;
        try {
            enc = bind(registry, ctx, spec.encoderLayers(), bound);
            dec = bind(registry, ctx, spec.layers(), bound);
            this.crossMixers = crossMixers(dec);
            h = new CpuLogitsHead(ctx, headPreprocessors);
            if (!h.hasLmHead()) {
                throw new IllegalStateException(arch() + ": the decoder has no LM head");
            }
        } catch (RuntimeException e) {
            closeInto(h, e);
            closeInto(bound, e);
            throw e;
        }
        this.encoder = enc;
        this.decoder = dec;
        this.head = h;
    }

    private static List<CpuLayerExecutor> bind(CpuBlockRegistry registry, CpuForwardContext ctx,
                                               List<LayerSpec> layers, List<CpuLayerExecutor> bound) {
        List<CpuLayerExecutor> list = new ArrayList<>(layers.size());
        for (LayerSpec l : layers) {
            CpuLayerExecutor e = new CpuLayerExecutor(registry, ctx, l);
            bound.add(e);
            list.add(e);
        }
        return Collections.unmodifiableList(list);
    }

    private float[] encoderNorm(CpuForwardContext ctx, int width) {
        String name = TensorNames.weight(TensorNames.ENC_OUTPUT_NORM);
        float[] w = ctx.weights().floatsOrNull(name);
        if (w == null) {
            throw new MissingTensorException(arch(), -1, name);
        }
        if (w.length != width) {
            throw new IllegalArgumentException(arch() + ": " + name + " has " + w.length + " values, the encoder"
                    + " stream is " + width + " wide");
        }
        return w;
    }

    /** The bound cross-attention mixer of every decoder layer, in layer order. */
    private List<CpuCrossAttentionMixer> crossMixers(List<CpuLayerExecutor> dec) {
        List<CpuCrossAttentionMixer> out = new ArrayList<>(dec.size());
        for (CpuLayerExecutor e : dec) {
            CpuMixer m = e.crossAttentionMixer();
            if (m == null) {
                throw new IllegalStateException(arch() + ": decoder layer " + e.layer().il() + " has no"
                        + " cross-attention; every T5 decoder layer attends to the encoder output");
            }
            if (!(m instanceof CpuCrossAttentionMixer)) {
                throw new IllegalStateException(arch() + ": the " + MixerKind.CROSS_ATTENTION_T5 + " factory built "
                        + m.getClass().getName() + ", the encoder-decoder backend needs "
                        + CpuCrossAttentionMixer.class.getSimpleName() + " to hand it the encoder output");
            }
            out.add((CpuCrossAttentionMixer) m);
        }
        return Collections.unmodifiableList(out);
    }

    // ================================================================== shared helpers

    /**
     * Checks that {@code spec} is an encoder-decoder model: support class and memory kind {@code ENC_DEC}, a
     * non-empty encoder stack and decoder trunk.
     *
     * @throws IllegalArgumentException otherwise
     */
    static void checkModel(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        String arch = spec.arch().ggufName();
        if (spec.supportClass() != SupportClass.ENC_DEC || spec.memoryKind() != MemoryKind.ENC_DEC) {
            throw new IllegalArgumentException(arch + " (" + spec.supportClass() + ", memory " + spec.memoryKind()
                    + ") is not an encoder-decoder model");
        }
        if (spec.encoderLayers().isEmpty() || spec.nLayer() == 0) {
            throw new IllegalArgumentException(arch + ": an encoder-decoder model needs encoder and decoder layers, got "
                    + spec.encoderLayers().size() + " and " + spec.nLayer());
        }
    }

    /**
     * The token the decoder starts from: {@code <arch>.decoder_start_token_id} when the model declares it, else the
     * vocabulary's BOS (llama.cpp tools/completion/completion.cpp:535-538, examples/simple/simple.cpp:169-172).
     *
     * @param vocabBos the vocabulary BOS, or a negative id when it has none
     * @throws IllegalStateException when neither is defined
     */
    static int decoderStartToken(ModelSpec spec, int vocabBos) {
        int declared = spec.hparams().decStartTokenId();
        if (declared >= 0) {
            return declared;
        }
        if (vocabBos >= 0) {
            return vocabBos;
        }
        throw new IllegalStateException(spec.arch().ggufName() + " declares no decoder_start_token_id and the"
                + " vocabulary has no BOS token, so the decoder has no start token");
    }

    /**
     * Rows of the smallest cross-attention cache of the decoder layers: the longest encoder output the caches hold.
     *
     * @throws IllegalStateException when a decoder layer with cross-attention has no planned cross cache
     */
    static int crossRows(ModelSpec spec, MemoryPlan plan) {
        int rows = Integer.MAX_VALUE;
        for (LayerSpec l : spec.layers()) {
            if (l.crossAttention() == null) {
                continue;
            }
            LayerMemoryPlan lp = plan.layer(l.il());
            if (lp.crossKv() == null) {
                throw new IllegalStateException(spec.arch().ggufName() + ": decoder layer " + l.il()
                        + " has cross-attention but the memory plan has no cross-attention cache for it");
            }
            rows = Math.min(rows, lp.crossKv().rows());
        }
        if (rows == Integer.MAX_VALUE) {
            throw new IllegalStateException(spec.arch().ggufName() + ": no decoder layer has cross-attention");
        }
        return rows;
    }

    /**
     * Refuses a source longer than {@code limit} ({@code min(ubatch, cross cache rows)}) with
     * {@link ContextFullException}; when the ubatch is the bound, the cause says so (the encoder runs the whole
     * source in one ubatch, llama-context.cpp:1628).
     */
    static void checkSourceLength(int n, int ubatch, int limit) {
        if (n <= limit) {
            return;
        }
        ContextFullException e = new ContextFullException(n, limit);
        if (n > ubatch) {
            e.initCause(new IllegalArgumentException("the encoder runs the whole source of " + n + " tokens in one"
                    + " ubatch of " + ubatch + "; raise neuralfree.ubatch"));
        }
        throw e;
    }

    // ================================================================== encoder + decoder start

    /**
     * Encodes {@code source}, builds the cross caches and evaluates the decoder start token at position 0.
     *
     * @param pos0 must be 0: every prefill starts a new source and decoder sequence
     * @return the logits of the decoder start token
     * @throws ContextFullException     when the source is longer than {@code min(ubatch, nCtx)}; nothing changes
     * @throws IllegalArgumentException for an empty source, a token outside the vocabulary or {@code pos0 != 0}
     */
    @Override
    public float[] prefill(int[] source, int pos0) {
        checkOpen();
        Objects.requireNonNull(source, "source");
        if (pos0 != 0) {
            throw new IllegalArgumentException(arch() + ": an encoder-decoder prefill encodes a new source and"
                    + " restarts the decoder at position 0, got " + pos0);
        }
        if (source.length == 0) {
            throw new IllegalArgumentException("encoder source without tokens");
        }
        checkSourceLength(source.length, scratch.ubatch(), sourceLimit);
        embedding.checkInputs(source, null, 0);
        encoded = false;
        memory.reset();
        try {
            float[] enc = encode(source, null);
            for (CpuCrossAttentionMixer m : crossMixers) {
                m.encode(enc, 0, source.length);
            }
            encoded = true;
            singleToken[0] = startToken;
            TapRows taps = TapRows.open(tap, 1, decoder.size());
            step(singleToken, 0, 1, 0, taps);
            float[] logits = lastRowLogits(1);
            deliver(taps);
            return logits;
        } catch (RuntimeException e) {
            encoded = false;
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /**
     * The encoder pass over all of {@code tokens}: embedding at positions {@code 0 ..}, the {@code enc.blk} stack,
     * then {@code enc.output_norm} into {@link CpuScratch#NORMED}, which is returned.
     */
    private float[] encode(int[] tokens, int[] types) {
        int n = tokens.length;
        float[] x = scratch.slot(CpuScratch.X);
        embedding.embed(tokens, types, 0, n, 0, x);
        for (CpuLayerExecutor e : encoder) {
            e.forward(0, n, x);
        }
        float[] out = scratch.slot(CpuScratch.NORMED);
        CpuOps.norm(NormKind.RMS, encoderNormEps, x, 0, out, 0, n, encoderWidth, encoderNorm, null);
        return out;
    }

    // ================================================================== decoder steps

    @Override
    public float[] decode(int token, int pos) {
        checkOpen();
        singleToken[0] = token;
        prepareCausal(singleToken, pos);
        TapRows taps = TapRows.open(tap, 1, decoder.size());
        step(singleToken, 0, 1, pos, taps);
        float[] logits = lastRowLogits(1);
        deliver(taps);
        return logits;
    }

    /** {@link #decode} with the greedy choice taken on the scratch logits; no logits row is copied out. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        checkOpen();
        singleToken[0] = token;
        prepareCausal(singleToken, pos);
        TapRows taps = TapRows.open(tap, 1, decoder.size());
        step(singleToken, 0, 1, pos, taps);
        rowIdx[0] = 0;
        head.hidden(scratch.slot(CpuScratch.X), decoderWidth, rowIdx, 1);
        head.logits(1);
        int id = head.argmax(0);
        deliver(taps);
        return id;
    }

    /** Decoder tokens at {@code pos0 ..} over the current encoding (speculative verification of the decoder). */
    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        checkOpen();
        prepareCausal(tokens, pos0);
        int[] rows = rowsOrAll(outputRows, tokens.length);
        TapRows taps = TapRows.open(tap, tokens.length, decoder.size());
        float[][] out = new float[rows.length][];
        for (int off = 0; off < tokens.length; off += ubatch) {
            int n = Math.min(ubatch, tokens.length - off);
            step(tokens, off, n, pos0 + off, taps);
            emitRows(rows, off, n, out);
        }
        deliver(taps);
        return out;
    }

    /** One decoder step of {@code n} tokens: embedding, the step's memory rows, every decoder layer, bookkeeping. */
    private void step(int[] tokens, int off, int n, int pos0, TapRows taps) {
        float[] x = scratch.slot(CpuScratch.X);
        embedding.embed(tokens, null, off, n, pos0, x);
        memory.beginStep(pos0, n);
        for (int il = 0; il < decoder.size(); il++) {
            CpuLayerExecutor e = decoder.get(il);
            if (taps != null && taps.wants(il)) {
                taps.capture(il, x, off, n, e.streamWidth());
            }
            e.forward(pos0, n, x);
        }
        if (taps != null && taps.wants(decoder.size())) {
            taps.capture(decoder.size(), x, off, n, decoderWidth);
        }
        memory.sequence().append(tokens, off, n);
    }

    /**
     * Checks a decoder call before any work: a current encoding, the inputs, the context length, then aligns the
     * memory to {@code pos0} (rolling back the tail when the call re-evaluates stored positions).
     */
    private void prepareCausal(int[] tokens, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        requireEncoded();
        if (tokens.length == 0) {
            throw new IllegalArgumentException("no decoder tokens to evaluate");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("position " + pos0 + " is negative");
        }
        embedding.checkInputs(tokens, null, pos0);
        int nCtx = memory.sequence().nCtx();
        long end = (long) pos0 + tokens.length;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(end, Integer.MAX_VALUE), nCtx);
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

    /** Logits of the last row of the step just run, copied to the reused result row. */
    private float[] lastRowLogits(int n) {
        rowIdx[0] = n - 1;
        head.hidden(scratch.slot(CpuScratch.X), decoderWidth, rowIdx, 1);
        float[] logits = head.logits(1);
        int vocab = head.vocabRows();
        if (lastLogits == null || lastLogits.length != vocab) {
            lastLogits = new float[vocab];
        }
        System.arraycopy(logits, 0, lastLogits, 0, vocab);
        return lastLogits;
    }

    /** Logits of the requested rows that fall in the step {@code [off, off + n)}, batched through the head. */
    private void emitRows(int[] rows, int off, int n, float[][] out) {
        int max = head.maxRows();
        int count = 0;
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r < off || r >= off + n) {
                continue;
            }
            rowIdx[count] = r - off;
            outputIdx[count] = i;
            count++;
            if (count == max) {
                flush(count, out);
                count = 0;
            }
        }
        if (count > 0) {
            flush(count, out);
        }
    }

    private void flush(int count, float[][] out) {
        head.hidden(scratch.slot(CpuScratch.X), decoderWidth, rowIdx, count);
        float[] logits = head.logits(count);
        int vocab = head.vocabRows();
        for (int k = 0; k < count; k++) {
            out[outputIdx[k]] = Arrays.copyOfRange(logits, k * vocab, (k + 1) * vocab);
        }
    }

    private static int[] rowsOrAll(int[] outputRows, int nTokens) {
        if (outputRows == null) {
            int[] all = new int[nTokens];
            for (int i = 0; i < nTokens; i++) {
                all[i] = i;
            }
            return all;
        }
        for (int r : outputRows) {
            if (r < 0 || r >= nTokens) {
                throw new IllegalArgumentException("output row " + r + " outside [0, " + nTokens + ")");
            }
        }
        return outputRows;
    }

    // ================================================================== encoder output

    /**
     * Not defined for an encoder-decoder model: the encoder has no LM head and the decoder needs an encoding.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        checkOpen();
        throw new UnsupportedOperationException(arch() + " is an encoder-decoder model: encode a source with"
                + " prefill(source, 0) and decode, or read the encoder output with hiddenRows");
    }

    /**
     * The encoder output rows after {@code enc.output_norm} ({@code [n][n_embd]}). Neither the decoder memory nor the
     * cross caches change.
     *
     * @throws IllegalArgumentException for an empty input, a token outside the vocabulary, mismatched types or more
     *                                  tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        checkOpen();
        Objects.requireNonNull(tokens, "tokens");
        int n = tokens.length;
        if (n == 0) {
            throw new IllegalArgumentException("encoder input without tokens");
        }
        if (n > scratch.ubatch()) {
            throw new IllegalArgumentException(n + " tokens exceed the ubatch of " + scratch.ubatch() + "; the encoder"
                    + " runs in one step, raise neuralfree.ubatch");
        }
        embedding.checkInputs(tokens, tokenTypes, 0);
        float[] enc = encode(tokens, tokenTypes);
        float[][] rows = new float[n][];
        for (int t = 0; t < n; t++) {
            rows[t] = Arrays.copyOfRange(enc, t * encoderWidth, (t + 1) * encoderWidth);
        }
        return rows;
    }

    // ================================================================== memory

    /** Empties the decoder memory and forgets the encoding; the next decoder call needs a new prefill. */
    @Override
    public void reset() {
        checkOpen();
        encoded = false;
        memory.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        checkOpen();
        return memory.canRollback(nTokens);
    }

    /** Drops decoder positions; the encoding stays valid. */
    @Override
    public void rollback(int nTokens) {
        checkOpen();
        memory.rollback(nTokens);
    }

    /**
     * Context shift of the decoder self-attention cache through {@link HeapModelMemory#shiftPositions}, as
     * {@code CpuForwardBackend}: refused before any row moves by the model ({@link ModelSpec#contextShiftRefusal()})
     * or a decoder cache mixer ({@link CpuLayerExecutor#shiftRefusal}); the cross caches hold the encoder output and
     * stay. A layer failure after the rows moved resets the memory and forgets the encoding.
     *
     * @throws UnsupportedOperationException when the decoder memory cannot shift
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        checkOpen();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported for " + arch() + ": " + refusal);
        }
        if (!(memory instanceof HeapModelMemory)) {
            throw new UnsupportedOperationException("context shift needs HeapModelMemory, got "
                    + memory.getClass().getSimpleName());
        }
        for (CpuLayerExecutor e : decoder) {
            refusal = e.shiftRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException("context shift unsupported for " + arch() + ": " + refusal);
            }
        }
        ((HeapModelMemory) memory).shiftPositions(nKeep, nDiscard);
        try {
            for (CpuLayerExecutor e : decoder) {
                e.shiftContext(nKeep, nDiscard);
            }
        } catch (RuntimeException e) {
            encoded = false;
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            String msg = "context shift of " + arch() + " failed after the cache rows moved; the memory was reset";
            if (e instanceof UnsupportedOperationException) {
                throw new UnsupportedOperationException(msg, e);
            }
            throw new IllegalStateException(msg, e);
        }
    }

    /** The decoder memory (self-attention KV, cross caches and the decoder sequence). */
    @Override
    public ModelMemory memory() {
        checkOpen();
        return memory;
    }

    /** Observer of the decoder stack's tap points for every later call. */
    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        checkOpen();
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

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("CPU encoder-decoder backend is closed");
        }
    }

    private void deliver(TapRows taps) {
        if (taps != null) {
            taps.deliver();
        }
    }

    // ================================================================== lifecycle

    /** Closes the executors and the head. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = new IllegalStateException("closing the CPU encoder-decoder backend failed");
        List<CpuLayerExecutor> all = new ArrayList<>(decoder);
        all.addAll(encoder);
        closeInto(all, failure);
        closeInto(head, failure);
        if (failure.getSuppressed().length > 0) {
            throw failure;
        }
    }

    private static void closeInto(List<CpuLayerExecutor> executors, RuntimeException sink) {
        for (CpuLayerExecutor e : executors) {
            closeInto(e, sink);
        }
    }

    private static void closeInto(AutoCloseable c, RuntimeException sink) {
        if (c == null) {
            return;
        }
        try {
            c.close();
        } catch (Exception failure) {
            sink.addSuppressed(failure);
        }
    }

    @Override
    public String toString() {
        return "CpuEncDecBackend{" + arch() + ", encoder " + encoder.size() + " layers, decoder " + decoder.size()
                + " layers, start " + startToken + ", ubatch " + ubatch + (encoded ? ", encoded" : "")
                + (closed ? ", closed" : "") + '}';
    }

    // ================================================================== hidden tap

    /**
     * The rows of the wanted decoder tap points over one call, gathered step by step and delivered once per tap point
     * after the call ({@link HiddenTap}: {@code nRows} is every token of the call).
     */
    private static final class TapRows {
        private final HiddenTap tap;
        private final int total;
        private final float[][] rows;
        private final int[] widths;

        private TapRows(HiddenTap tap, int total, int points) {
            this.tap = tap;
            this.total = total;
            this.rows = new float[points][];
            this.widths = new int[points];
        }

        /** Collector of one call over tap points {@code [0, nLayer]}; null when the tap wants none of them. */
        static TapRows open(HiddenTap tap, int total, int nLayer) {
            for (int p = 0; p <= nLayer; p++) {
                if (tap.wants(p)) {
                    return new TapRows(tap, total, nLayer + 1);
                }
            }
            return null;
        }

        boolean wants(int point) {
            return tap.wants(point);
        }

        /** Copies the step's {@code n} rows, which are rows {@code [off, off + n)} of the call. */
        void capture(int point, float[] x, int off, int n, int width) {
            if (rows[point] == null) {
                rows[point] = new float[Math.multiplyExact(total, width)];
                widths[point] = width;
            } else if (widths[point] != width) {
                throw new IllegalStateException("tap point " + point + " changed width from " + widths[point]
                        + " to " + width + " within a call");
            }
            System.arraycopy(x, 0, rows[point], off * width, n * width);
        }

        void deliver() {
            for (int p = 0; p < rows.length; p++) {
                if (rows[p] != null) {
                    tap.onHidden(p, rows[p], total, widths[p]);
                }
            }
        }
    }
}
