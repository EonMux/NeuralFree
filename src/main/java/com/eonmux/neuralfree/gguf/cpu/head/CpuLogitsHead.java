package com.eonmux.neuralfree.gguf.cpu.head;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The output stage of the CPU forward pass, the twin of {@code GpuLogitsHead}. For the selected rows of a step:
 * <ol>
 *   <li>gather the rows of the final residual stream into a packed head buffer (llama.cpp {@code inp_out_ids});</li>
 *   <li>the bound {@link CpuHeadPreprocessor}s, in order (hyper-connection or AltUp collapse to {@code n_embd});</li>
 *   <li>{@code output_norm} (+ bias). The norm kind and epsilon are the trunk's block norm's ({@code LLM_NORM} with
 *       {@code f_norm_eps}, or {@code LLM_NORM_RMS} with {@code f_norm_rms_eps}), over
 *       {@link HeadSpec#outputNormGroups()} slices of the row (k2-horizon.cpp:124-135, 352). Without an
 *       {@code output_norm} tensor, a model whose block norms are weightless applies the same weightless norm
 *       (olmo.cpp, talkie.cpp:137), except the hierarchical schedule, whose output is {@code zH} itself
 *       (hrm-text.cpp:198-203). The {@link HeadSpec#embeddingProjection()} with its LoRA deltas then maps the rows to
 *       {@link #hiddenWidth()} (gemma-embedding2.cpp:206-212). The result is llama.cpp's {@code t_embd};</li>
 *   <li>{@code logits = output · h} ({@code token_embd} when tied), {@code + scale * B (A h)} of the LoRA deltas
 *       registered on {@code output.weight} ({@code build_lora_mm}, llama-graph.cpp:1629-1667; a tied head serves
 *       the same name, so a {@code token_embd} adapter adapts only the lookup), {@code + output.bias}, then the
 *       logit scale:
 *       {@code * scale}, or {@code * (1 / scale)} when {@link HeadSpec#logitScaleDivides()} (granite.cpp,
 *       grok.cpp:212), then the final softcap {@code cap * tanh(logits * (1 / cap))} (grok.cpp:214-218,
 *       gemma2.cpp), then {@code -FLT_MAX} over the masked logit range of every row (chameleon.cpp:186-198).</li>
 * </ol>
 * Logit rows are {@link #vocabRows()} wide and are written to {@link CpuScratch#LOGITS}, row-major. Not thread-safe;
 * one forward pass at a time. A call allocates nothing.
 */
public final class CpuLogitsHead implements AutoCloseable {

    private static final String TEMP_ROWS = "CpuLogitsHead.rows";
    private static final String TEMP_EMBD = "CpuLogitsHead.embd";

    private final CpuScratch scratch;
    private final CpuMatVec matVec;
    private final HeadSpec head;
    private final List<CpuHeadPreprocessor> preprocessors;
    private final int nEmbd;
    private final int maxStreamWidth;
    private final NormKind normKind;
    private final float normEps;
    private final boolean outputNorm;
    private final float[] normWeight;
    private final float[] normBias;
    private final int normGroups;
    private final CpuTensorRef projection;
    /** LoRA deltas of the embedding projection, or null without one. */
    private final CpuProjectionDeltas.Site projectionDeltas;
    private final int hiddenWidth;
    private final float[] embd;
    private final CpuTensorRef output;
    private final float[] outputBias;
    /** LoRA deltas of the output head, or null without an LM head. */
    private final CpuProjectionDeltas.Site outputDeltas;
    private final int vocab;
    private final ActParams softcap;
    private final float[] rows;
    private boolean closed;

    /**
     * Binds the preprocessors that apply to the model.
     *
     * @param preprocessors candidate preprocessors in registration order; each that {@code appliesTo} the model is
     *                      bound and then owned (closed) by this head
     * @throws IllegalArgumentException when the final stream is wider than {@code n_embd} and no preprocessor
     *                                  collapses it, or a head tensor disagrees with {@code n_embd}
     */
    public CpuLogitsHead(CpuForwardContext ctx, List<CpuHeadPreprocessor> preprocessors) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(preprocessors, "preprocessors");
        ModelSpec spec = ctx.spec();
        CpuWeights weights = ctx.weights();
        this.scratch = ctx.scratch();
        this.matVec = ctx.matVec();
        this.head = Objects.requireNonNull(spec.head(), "head spec");
        this.nEmbd = spec.hparams().nEmbd();
        this.maxStreamWidth = Math.max(nEmbd, spec.maxStreamWidth());
        int lastWidth = spec.layer(spec.nLayer() - 1).residual().streamWidth();

        NormSpec trunkNorm = spec.layer(0).norm();
        this.normKind = trunkNorm.kind();
        this.normEps = trunkNorm.eps();
        TensorRef normRef = head.outputNorm();
        this.normWeight = normRef != null ? checkWidth(weights.floats(normRef), "output_norm") : null;
        this.normBias = head.outputNormBias() != null
                ? checkWidth(weights.floats(head.outputNormBias()), "output_norm bias") : null;
        this.outputNorm = normRef != null
                || (trunkNorm.weightless() && spec.schedule().kind() != LayerSchedule.Kind.HRM);
        this.normGroups = head.outputNormGroups();
        if (normGroups > 1 && (normWeight == null || nEmbd % normGroups != 0)) {
            throw new IllegalArgumentException("an output norm over " + normGroups + " groups needs output_norm and an"
                    + " n_embd " + nEmbd + " divisible by the groups");
        }
        this.hiddenWidth = head.hiddenWidth(nEmbd);
        this.projection = head.embeddingProjection() != null ? weights.get(head.embeddingProjection()) : null;
        if (projection != null && (projection.cols() != nEmbd || projection.rows() != hiddenWidth)) {
            throw new IllegalArgumentException("embedding projection is " + projection.cols() + " x "
                    + projection.rows() + ", expected " + nEmbd + " x " + hiddenWidth);
        }
        this.projectionDeltas = projection != null ? ctx.projectionDeltas().serveModel(
                TensorNames.weight(TensorNames.OUTPUT), projection, matVec) : null;
        this.embd = projection != null
                ? scratch.temp(TEMP_EMBD, Math.multiplyExact(scratch.ubatch(), hiddenWidth)) : null;

        this.output = head.output() != null ? weights.get(head.output()) : null;
        if (output != null && output.cols() != nEmbd) {
            throw new IllegalArgumentException("LM head rows hold " + output.cols() + " values, expected n_embd "
                    + nEmbd);
        }
        this.vocab = (int) head.vocabRows();
        if (output != null && output.rows() != head.vocabRows()) {
            throw new IllegalArgumentException("LM head has " + output.rows() + " rows, the head spec expects "
                    + head.vocabRows());
        }
        this.outputDeltas = output != null ? ctx.projectionDeltas().serveModel(
                TensorNames.weight(TensorNames.OUTPUT), output, matVec) : null;
        this.outputBias = head.outputBias() != null ? weights.floats(head.outputBias()) : null;
        if (outputBias != null && outputBias.length != vocab) {
            throw new IllegalArgumentException("output bias has " + outputBias.length + " values, expected "
                    + vocab);
        }
        this.softcap = head.finalSoftcap() > 0f
                ? ActParams.builder().softcap(head.finalSoftcap(), 1.0f).build() : null;
        if (head.maskedLogitsEnd() > vocab) {
            throw new IllegalArgumentException("masked logit range ends at " + head.maskedLogitsEnd() + ", the head has "
                    + vocab + " rows");
        }
        this.rows = scratch.temp(TEMP_ROWS, Math.multiplyExact(scratch.ubatch(), maxStreamWidth));
        this.preprocessors = bind(ctx, spec, preprocessors);
        if (lastWidth != nEmbd && this.preprocessors.isEmpty()) {
            close();
            throw new IllegalArgumentException(spec.arch().ggufName() + ": the final stream has " + lastWidth
                    + " values per token but no head preprocessor collapses it to n_embd " + nEmbd);
        }
    }

    private float[] checkWidth(float[] v, String what) {
        if (v.length != nEmbd) {
            throw new IllegalArgumentException(what + " has " + v.length + " values, expected n_embd " + nEmbd);
        }
        return v;
    }

    private static List<CpuHeadPreprocessor> bind(CpuForwardContext ctx, ModelSpec spec,
                                                  List<CpuHeadPreprocessor> candidates) {
        List<CpuHeadPreprocessor> bound = new ArrayList<>();
        try {
            for (CpuHeadPreprocessor p : candidates) {
                Objects.requireNonNull(p, "head preprocessor");
                if (p.appliesTo(spec)) {
                    p.bind(ctx);
                    bound.add(p);
                }
            }
        } catch (RuntimeException e) {
            Closeables.closeAll(bound, e);
            throw e;
        }
        return Collections.unmodifiableList(bound);
    }

    /** Width of each logits row (LM head rows including vocabulary padding); 0 without an LM head. */
    public int vocabRows() {
        return vocab;
    }

    /** True when the model has an LM head. */
    public boolean hasLmHead() {
        return output != null;
    }

    /** Width of the hidden rows {@link #hidden} returns: {@code n_embd}, or the embedding projection's width. */
    public int hiddenWidth() {
        return hiddenWidth;
    }

    /** Most rows one call handles (the scratch ubatch). */
    public int maxRows() {
        return scratch.ubatch();
    }

    /**
     * Gathers rows of the final residual stream, collapses them to {@code n_embd}, applies the output norm and the
     * embedding projection.
     *
     * @param x           the final residual stream, rows of {@code streamWidth} floats
     * @param streamWidth floats per row of {@code x}
     * @param rowIdx      the rows of {@code x} to keep, in output order
     * @param nRows       entries of {@code rowIdx} to use, in {@code [1, maxRows()]}
     * @return the head buffer holding {@code nRows} packed rows of {@link #hiddenWidth()} floats (reused by the next
     *         call)
     * @throws IllegalArgumentException for a row count or index out of range
     */
    public float[] hidden(float[] x, int streamWidth, int[] rowIdx, int nRows) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(rowIdx, "rowIdx");
        checkOpen();
        if (nRows < 1 || nRows > scratch.ubatch() || nRows > rowIdx.length) {
            throw new IllegalArgumentException("head over " + nRows + " rows outside [1, "
                    + Math.min(scratch.ubatch(), rowIdx.length) + "]");
        }
        if (streamWidth < nEmbd || streamWidth > maxStreamWidth) {
            throw new IllegalArgumentException("stream width " + streamWidth + " outside [" + nEmbd + ", "
                    + maxStreamWidth + "]");
        }
        long xRows = x.length / streamWidth;
        for (int r = 0; r < nRows; r++) {
            int src = rowIdx[r];
            if (src < 0 || src >= xRows) {
                throw new IllegalArgumentException("output row " + src + " outside the " + xRows + " stream rows");
            }
            System.arraycopy(x, src * streamWidth, rows, r * streamWidth, streamWidth);
        }
        for (CpuHeadPreprocessor p : preprocessors) {
            p.apply(nRows, rows);
        }
        if (outputNorm) {
            if (normGroups > 1) {
                int slice = nEmbd / normGroups;
                NormMath.rowNorm(normKind, normEps, nRows * normGroups, slice, slice, normGroups, rows, 0,
                        normWeight, normBias, rows, 0);
            } else {
                CpuOps.norm(normKind, normEps, rows, 0, rows, 0, nRows, nEmbd, normWeight, normBias);
            }
        }
        if (projection == null) {
            return rows;
        }
        matVec.matmul(projection, rows, 0, embd, 0, nRows);
        projectionDeltas.applyRows(nRows, rows, 0, embd, 0);
        return embd;
    }

    /**
     * Logits of the rows the last {@link #hidden} call prepared.
     *
     * @param nRows the row count passed to that call
     * @return {@link CpuScratch#LOGITS}, holding {@code nRows} rows of {@link #vocabRows()} floats
     * @throws UnsupportedOperationException when the model has no LM head
     */
    public float[] logits(int nRows) {
        checkOpen();
        requireLmHead();
        if (nRows < 1 || nRows > scratch.ubatch()) {
            throw new IllegalArgumentException("logits of " + nRows + " rows outside [1, " + scratch.ubatch() + "]");
        }
        float[] logits = scratch.slot(CpuScratch.LOGITS);
        int n = Math.multiplyExact(nRows, vocab);
        matVec.matmul(output, rows, 0, logits, 0, nRows);
        outputDeltas.applyRows(nRows, rows, 0, logits, 0);
        if (outputBias != null) {
            CpuOps.addRowBias(logits, 0, nRows, vocab, outputBias, 0);
        }
        float scale = head.logitScale();
        if (scale != 1.0f) {
            CpuOps.scale(logits, 0, n, head.logitScaleDivides() ? 1.0f / scale : scale);
        }
        if (softcap != null) {
            CpuOps.unary(ActOp.SOFTCAP, softcap, logits, 0, logits, 0, n);
        }
        if (head.masksLogits()) {
            for (int r = 0; r < nRows; r++) {
                int base = r * vocab;
                Arrays.fill(logits, base + head.maskedLogitsStart(), base + head.maskedLogitsEnd(), -Float.MAX_VALUE);
            }
        }
        return logits;
    }

    /**
     * Greedy choice on logits row {@code row} of the last {@link #logits} call: the largest value, the lowest index
     * on ties, NaN never winning over an earlier value ({@code ForwardBackend.argmax}).
     *
     * @throws UnsupportedOperationException when the model has no LM head
     */
    public int argmax(int row) {
        checkOpen();
        requireLmHead();
        float[] logits = scratch.slot(CpuScratch.LOGITS);
        if (row < 0 || (long) (row + 1) * vocab > logits.length || vocab < 1) {
            throw new IllegalArgumentException("logits row " + row + " is out of range");
        }
        int base = row * vocab;
        int best = 0;
        float bestValue = logits[base];
        for (int i = 1; i < vocab; i++) {
            if (logits[base + i] > bestValue) {
                bestValue = logits[base + i];
                best = i;
            }
        }
        return best;
    }

    private void requireLmHead() {
        if (output == null) {
            throw new UnsupportedOperationException("model has no LM head; read hidden rows instead");
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("logits head is closed");
        }
    }

    /** Closes the bound preprocessors. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = new IllegalStateException("closing the head preprocessors failed");
        Closeables.closeAll(preprocessors != null ? preprocessors : Collections.<CpuHeadPreprocessor>emptyList(),
                failure);
        if (failure.getSuppressed().length > 0) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "CpuLogitsHead{vocab=" + vocab + ", norm=" + (outputNorm ? normKind : "none")
                + (head.tied() ? ", tied" : "") + ", preprocessors=" + preprocessors.size() + '}';
    }
}
