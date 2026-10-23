package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.DecisionOrder;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * The vllm-decision2 head over the qwen3 body's final hidden rows (models/vllm-decision2.cpp:42-179):
 * <ul>
 *   <li>rows (decision2_get_rows, :52-93): the last row of each option span and of the single question span; the
 *       order is usable when every code is known, there is exactly one question and at least one option, otherwise
 *       one option at row 0 with query row 0;</li>
 *   <li>{@code o = LayerNorm(option rows; option_norm)}, {@code q = LayerNorm(query row; query_norm)} with
 *       {@code eps = attention.layernorm_epsilon};</li>
 *   <li>{@code bilinear = sum((key o) * (query q)) / sqrt(n_embd_h)}, {@code n_embd_h = decision.key} rows;</li>
 *   <li>{@code nonlinear = scorer_out gelu_erf(option_mlp o + option_mlp.bias + query_mlp q)};</li>
 *   <li>score of option {@code i} = bilinear + nonlinear at output row {@code i}; rows past the options are 0; a
 *       given but unusable order adds NaN to every row (:130-131, :148-150).</li>
 * </ul>
 * Immutable; {@link #score} allocates its result.
 */
public final class VllmDecision2Head {

    private final CpuMatVec mv;
    private final int nEmbd;
    private final int nEmbdH;
    private final float eps;
    private final float[] optionNorm;
    private final float[] optionNormB;
    private final float[] queryNorm;
    private final float[] queryNormB;
    private final CpuTensorRef key;
    private final CpuTensorRef query;
    private final CpuTensorRef optionMlp;
    private final float[] optionMlpB;
    private final CpuTensorRef queryMlp;
    private final CpuTensorRef scorerOut;

    private VllmDecision2Head(ModelSpec spec, CpuWeights w) {
        this.mv = w.matVec();
        this.nEmbd = spec.hparams().nEmbd();
        this.eps = spec.hparams().normEps();
        this.key = w.get(TensorNames.weight(TensorNames.DECISION_KEY));
        if (key.cols() != nEmbd) {
            throw new IllegalArgumentException("decision.key takes " + key.cols() + " inputs, n_embd is " + nEmbd);
        }
        this.nEmbdH = HeadsBlocks.toInt(key.rows(), key);
        this.optionNorm = HeadsBlocks.vector(w, TensorNames.weight(TensorNames.DECISION_OPTION_NORM), nEmbd);
        this.optionNormB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.DECISION_OPTION_NORM), nEmbd);
        this.queryNorm = HeadsBlocks.vector(w, TensorNames.weight(TensorNames.DECISION_QUERY_NORM), nEmbd);
        this.queryNormB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.DECISION_QUERY_NORM), nEmbd);
        this.query = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_QUERY), nEmbd, nEmbdH);
        this.optionMlp = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_OPTION_MLP), nEmbd, nEmbdH);
        this.optionMlpB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.DECISION_OPTION_MLP), nEmbdH);
        this.queryMlp = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_QUERY_MLP), nEmbd, nEmbdH);
        this.scorerOut = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_SCORER_OUT), nEmbdH, 1);
    }

    /** The head of a vllm-decision2 model over the tensors of {@code weights}. */
    public static VllmDecision2Head of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        return new VllmDecision2Head(spec, weights);
    }

    /** Width of the head ({@code n_embd_h}). */
    public int headWidth() {
        return nEmbdH;
    }

    /**
     * Option scores of one input: output row {@code i} holds the score of option {@code i}.
     *
     * @param hiddenRows the final hidden rows of every token, after the output norm
     * @param order      one decision code per token, or null for an input without decision order
     * @return one value per token (the head's output is padded to the output rows)
     */
    public float[] score(float[][] hiddenRows, int[] order) {
        float[] hidden = HeadsBlocks.pack(hiddenRows, nEmbd);
        int nTokens = hiddenRows.length;
        DecisionOrder d = DecisionOrder.parse(order, nTokens);
        boolean valid = d.given() && d.wellFormed() && d.questionCount() == 1 && d.optionCount() > 0;
        int nOptions = valid ? d.optionCount() : 1;
        if (nOptions > nTokens) {
            throw new IllegalStateException(nOptions + " options cannot exceed " + nTokens + " output rows");
        }
        int queryRow = valid ? d.questionEnd(0) - 1 : 0;

        float[] options = new float[nOptions * nEmbd];
        for (int i = 0; i < nOptions; i++) {
            int row = valid ? d.optionEnd(i) - 1 : 0;
            System.arraycopy(hidden, row * nEmbd, options, i * nEmbd, nEmbd);
        }
        float[] q = new float[nEmbd];
        System.arraycopy(hidden, queryRow * nEmbd, q, 0, nEmbd);
        HeadsBlocks.layerNorm(options, nOptions, nEmbd, optionNorm, optionNormB, eps);
        HeadsBlocks.layerNorm(q, 1, nEmbd, queryNorm, queryNormB, eps);

        float[] k = HeadsBlocks.linear(mv, key, null, options, nOptions);
        float[] qh = HeadsBlocks.linear(mv, query, null, q, 1);
        float[] mlp = HeadsBlocks.linear(mv, optionMlp, optionMlpB, options, nOptions);
        float[] qm = HeadsBlocks.linear(mv, queryMlp, null, q, 1);
        for (int i = 0; i < nOptions; i++) {
            for (int j = 0; j < nEmbdH; j++) {
                mlp[i * nEmbdH + j] += qm[j];
            }
        }
        HeadsBlocks.activate(ActOp.GELU_ERF, mlp, mlp.length);
        float[] nonlinear = HeadsBlocks.linear(mv, scorerOut, null, mlp, nOptions);

        float inv = 1.0f / (float) Math.sqrt(nEmbdH);
        float status = DecisionOrder.status(d.given(), valid);
        float[] out = new float[nTokens];
        for (int i = 0; i < nOptions; i++) {
            float dot = 0.0f;
            for (int j = 0; j < nEmbdH; j++) {
                dot += k[i * nEmbdH + j] * qh[j];
            }
            out[i] = dot * inv + nonlinear[i];
        }
        for (int i = 0; i < nTokens; i++) {
            out[i] += status;
        }
        return out;
    }
}
