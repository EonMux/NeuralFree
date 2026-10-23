package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.DecisionOrder;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * The clef joint decision decoder over the qwen35 body's final hidden rows (models/clef.cpp:110-646, the
 * JointSchemaHead of the model repo). All norms are LayerNorms with weight and bias at
 * {@code eps = attention.layernorm_epsilon}; {@code H = n_embd_h} (rows of {@code decision.proj_memory}).
 * <ol>
 *   <li>Spans (clef_get_spans, :120-170): questions and options of the decision order; the order is usable when
 *       every code is known, every option follows a question, there is a question and every question has an option;
 *       otherwise one empty question with one empty option, and a given order adds NaN to every score.</li>
 *   <li>{@code h = LN(hidden; hidden_norm)}, {@code memory = proj_memory h}, {@code global = h[last]};
 *       question / option context = span means of {@code h}; lexical = span mean of the {@code output} rows of the
 *       option tokens (:555-566).</li>
 *   <li>{@code options = proj_option_context ctx + proj_option_lexical lex
 *       + (proj_option_question question)[owner]}.</li>
 *   <li>Routing blocks: {@code options += attn(LN(options; cross_norm), LN(memory; cross_norm_kv))},
 *       {@code options += ffn(options)} (:573-583).</li>
 *   <li>{@code fields = proj_question question}; each question attends its own options
 *       ({@code softmax(options . fields / sqrt(H))} with the owner mask) and adds
 *       {@code LN(summary; option_summary_norm) + proj_global global + token_types[type]} (:587-598).</li>
 *   <li>Joint blocks: self attention on {@code LN(fields; attn_norm)}, cross attention of
 *       {@code LN(fields; cross_attn_norm)} on {@code memory}, ffn (:600-608); then
 *       {@code LN(fields; field_norm)}.</li>
 *   <li>{@code prior = l2(lex) . l2(question + global)[owner]};
 *       {@code o = LN(options; option_norm)}, {@code f = fields[owner]}, {@code cosine = l2(f) . l2(o)};
 *       {@code residual = scorer_out gelu_erf(scorer [f | o | f*o | |f-o|] + b) + b};
 *       {@code score = prior s0 + (cosine s1 + residual) s2} with {@code s = decision.scales} (:613-645).</li>
 * </ol>
 * Head attention ({@code build_head_attn}, :497-518): biased Q/K/V/O projections, {@code n_head_decision} heads of
 * {@code H / n_head}, softmax scale {@code 1/sqrt(d_head)}, no mask. Head FFN (:520-525):
 * {@code down(gelu_erf(up LN(x; ffn_norm) + b)) + b}. Immutable; {@link #score} allocates its result.
 */
public final class ClefDecoder {

    /** L2 norm epsilon of the prior and cosine terms (clef.cpp:613). */
    static final float L2_EPS = 1e-12f;

    private final CpuMatVec mv;
    private final int nEmbd;
    private final int nEmbdH;
    private final int nHead;
    private final float eps;
    private final int nRouting;
    private final Block[] blocks;
    private final float[][] hiddenNorm;
    private final float[][] summaryNorm;
    private final float[][] fieldNorm;
    private final float[][] optionNorm;
    private final CpuTensorRef projMemory;
    private final CpuTensorRef projQuestion;
    private final CpuTensorRef projOptionQuestion;
    private final CpuTensorRef projGlobal;
    private final CpuTensorRef projOptionContext;
    private final CpuTensorRef projOptionLexical;
    private final float[] scales;
    private final float[] typeEmbd;
    private final CpuTensorRef scorer;
    private final float[] scorerB;
    private final CpuTensorRef scorerOut;
    private final float[] scorerOutB;
    private final CpuTensorRef output;

    private ClefDecoder(ModelSpec spec, CpuWeights w) {
        ModelHparams hp = spec.hparams();
        this.mv = w.matVec();
        this.nEmbd = hp.nEmbd();
        this.eps = hp.normEps();
        this.nHead = hp.decisionHeadCount();
        this.nRouting = hp.decisionRoutingLayers();
        int nJoint = hp.decisionJointLayers();
        this.projMemory = w.get(TensorNames.weight(TensorNames.DECISION_PROJ_MEMORY));
        if (projMemory.cols() != nEmbd) {
            throw new IllegalArgumentException("decision.proj_memory takes " + projMemory.cols() + " inputs, n_embd is "
                    + nEmbd);
        }
        this.nEmbdH = HeadsBlocks.toInt(projMemory.rows(), projMemory);
        if (nHead <= 0 || nEmbdH % nHead != 0) {
            throw new IllegalArgumentException("clef decision head width " + nEmbdH + " does not split into "
                    + nHead + " heads");
        }
        this.blocks = new Block[nRouting + nJoint];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = new Block(w, il, il < nRouting, nEmbdH);
        }
        this.hiddenNorm = norm(w, TensorNames.DECISION_HIDDEN_NORM, nEmbd);
        this.summaryNorm = norm(w, TensorNames.DECISION_OPTION_SUMMARY_NORM, nEmbdH);
        this.fieldNorm = norm(w, TensorNames.DECISION_FIELD_NORM, nEmbdH);
        this.optionNorm = norm(w, TensorNames.DECISION_OPTION_NORM, nEmbdH);
        this.projQuestion = proj(w, TensorNames.DECISION_PROJ_QUESTION);
        this.projOptionQuestion = proj(w, TensorNames.DECISION_PROJ_OPTION_QUESTION);
        this.projGlobal = proj(w, TensorNames.DECISION_PROJ_GLOBAL);
        this.projOptionContext = proj(w, TensorNames.DECISION_PROJ_OPTION_CONTEXT);
        this.projOptionLexical = proj(w, TensorNames.DECISION_PROJ_OPTION_LEXICAL);
        this.scales = HeadsBlocks.vector(w, TensorNames.DECISION_SCALES, 3);
        this.typeEmbd = HeadsBlocks.vector(w, TensorNames.weight(TensorNames.TOKEN_TYPES), 3 * nEmbdH);
        this.scorer = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_SCORER), 4L * nEmbdH, nEmbdH);
        this.scorerB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.DECISION_SCORER), nEmbdH);
        this.scorerOut = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.DECISION_SCORER_OUT), nEmbdH, 1);
        this.scorerOutB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.DECISION_SCORER_OUT), 1);
        if (spec.head().output() == null) {
            throw new IllegalArgumentException("clef needs the output embedding for the lexical option term");
        }
        this.output = w.get(spec.head().output());
        if (output.cols() != nEmbd) {
            throw new IllegalArgumentException("output embedding rows have " + output.cols() + " floats, n_embd is "
                    + nEmbd);
        }
    }

    /** The decoder of a clef model over the tensors of {@code weights}. */
    public static ClefDecoder of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        return new ClefDecoder(spec, weights);
    }

    private CpuTensorRef proj(CpuWeights w, String base) {
        return HeadsBlocks.matrix(w, TensorNames.weight(base), nEmbd, nEmbdH);
    }

    private static float[][] norm(CpuWeights w, String base, int n) {
        return new float[][]{HeadsBlocks.vector(w, TensorNames.weight(base), n),
                HeadsBlocks.vector(w, TensorNames.bias(base), n)};
    }

    /**
     * Option scores of one input: output row {@code i} holds the score of option {@code i}.
     *
     * @param hiddenRows the final hidden rows of every token, after the output norm
     * @param tokens     the token ids (the lexical term reads their output embeddings)
     * @param order      one decision code per token, or null for an input without decision order
     * @return one value per token (the head's output is padded to the token count)
     */
    public float[] score(float[][] hiddenRows, int[] tokens, int[] order) {
        float[] hidden = HeadsBlocks.pack(hiddenRows, nEmbd);
        int n = hiddenRows.length;
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length != n) {
            throw new IllegalArgumentException(tokens.length + " token ids for " + n + " hidden rows");
        }
        DecisionOrder d = DecisionOrder.parse(order, n);
        boolean valid = d.given() && d.wellFormed() && d.questionCount() > 0 && d.everyQuestionAnswered();
        Spans s = valid ? Spans.of(d) : Spans.empty();
        if (s.nOptions > n) {
            throw new IllegalStateException(s.nOptions + " options cannot exceed " + n + " output rows");
        }

        float[] h = HeadsBlocks.layerNormed(hidden, n, nEmbd, hiddenNorm[0], hiddenNorm[1], eps);
        float[] memory = HeadsBlocks.linear(mv, projMemory, null, h, n);
        float[] global = new float[nEmbd];
        System.arraycopy(h, (n - 1) * nEmbd, global, 0, nEmbd);
        float[] question = spanMeans(h, s.qStart, s.qEnd);
        float[] optCtx = spanMeans(h, s.oStart, s.oEnd);
        float[] lexical = lexicalMeans(tokens, s);

        float[] options = HeadsBlocks.linear(mv, projOptionContext, null, optCtx, s.nOptions);
        HeadsBlocks.addInPlace(options, HeadsBlocks.linear(mv, projOptionLexical, null, lexical, s.nOptions),
                options.length);
        float[] optQ = HeadsBlocks.linear(mv, projOptionQuestion, null, question, s.nQuestions);
        for (int o = 0; o < s.nOptions; o++) {
            for (int j = 0; j < nEmbdH; j++) {
                options[o * nEmbdH + j] += optQ[s.owner[o] * nEmbdH + j];
            }
        }
        for (int il = 0; il < nRouting; il++) {
            Block b = blocks[il];
            float[] q = HeadsBlocks.layerNormed(options, s.nOptions, nEmbdH, b.crossNorm[0], b.crossNorm[1], eps);
            float[] kv = HeadsBlocks.layerNormed(memory, n, nEmbdH, b.crossNormKv[0], b.crossNormKv[1], eps);
            HeadsBlocks.addInPlace(options, b.cross.apply(this, q, s.nOptions, kv, n), options.length);
            HeadsBlocks.addInPlace(options, b.ffn(this, options, s.nOptions), options.length);
        }

        float[] fields = HeadsBlocks.linear(mv, projQuestion, null, question, s.nQuestions);
        addOptionSummary(fields, options, s);
        HeadsBlocks.addInPlace(fields, broadcast(HeadsBlocks.linear(mv, projGlobal, null, global, 1), s.nQuestions),
                fields.length);
        for (int q = 0; q < s.nQuestions; q++) {
            HeadsBlocks.addInPlace(fields, q * nEmbdH, typeEmbd, s.qType[q] * nEmbdH, nEmbdH);
        }
        for (int il = nRouting; il < blocks.length; il++) {
            Block b = blocks[il];
            float[] cur = HeadsBlocks.layerNormed(fields, s.nQuestions, nEmbdH, b.selfNorm[0], b.selfNorm[1], eps);
            HeadsBlocks.addInPlace(fields, b.self.apply(this, cur, s.nQuestions, cur, s.nQuestions), fields.length);
            cur = HeadsBlocks.layerNormed(fields, s.nQuestions, nEmbdH, b.crossNorm[0], b.crossNorm[1], eps);
            HeadsBlocks.addInPlace(fields, b.cross.apply(this, cur, s.nQuestions, memory, n), fields.length);
            HeadsBlocks.addInPlace(fields, b.ffn(this, fields, s.nQuestions), fields.length);
        }
        HeadsBlocks.layerNorm(fields, s.nQuestions, nEmbdH, fieldNorm[0], fieldNorm[1], eps);

        return combine(s, d.given() && !valid, n, question, global, lexical, options, fields);
    }

    /** Prior, cosine and residual terms of every option, padded to {@code n} rows (clef.cpp:611-645). */
    private float[] combine(Spans s, boolean invalidGiven, int n, float[] question, float[] global, float[] lexical,
                            float[] options, float[] fields) {
        float[] anchor = question.clone();
        for (int q = 0; q < s.nQuestions; q++) {
            HeadsBlocks.addInPlace(anchor, q * nEmbd, global, 0, nEmbd);
        }
        HeadsBlocks.l2Norm(anchor, s.nQuestions, nEmbd, L2_EPS);
        float[] lex = lexical.clone();
        HeadsBlocks.l2Norm(lex, s.nOptions, nEmbd, L2_EPS);

        float[] o = HeadsBlocks.layerNormed(options, s.nOptions, nEmbdH, optionNorm[0], optionNorm[1], eps);
        float[] f = new float[s.nOptions * nEmbdH];
        for (int i = 0; i < s.nOptions; i++) {
            System.arraycopy(fields, s.owner[i] * nEmbdH, f, i * nEmbdH, nEmbdH);
        }
        float[] fn = f.clone();
        float[] on = o.clone();
        HeadsBlocks.l2Norm(fn, s.nOptions, nEmbdH, L2_EPS);
        HeadsBlocks.l2Norm(on, s.nOptions, nEmbdH, L2_EPS);

        int fw = 4 * nEmbdH;
        float[] features = new float[s.nOptions * fw];
        for (int i = 0; i < s.nOptions; i++) {
            int base = i * fw;
            for (int j = 0; j < nEmbdH; j++) {
                float fj = f[i * nEmbdH + j];
                float oj = o[i * nEmbdH + j];
                features[base + j] = fj;
                features[base + nEmbdH + j] = oj;
                features[base + 2 * nEmbdH + j] = fj * oj;
                features[base + 3 * nEmbdH + j] = Math.abs(fj - oj);
            }
        }
        float[] hidden = HeadsBlocks.linear(mv, scorer, scorerB, features, s.nOptions);
        HeadsBlocks.activate(ActOp.GELU_ERF, hidden, hidden.length);
        float[] residual = HeadsBlocks.linear(mv, scorerOut, scorerOutB, hidden, s.nOptions);

        float status = DecisionOrder.status(true, !invalidGiven);
        float[] out = new float[n];
        for (int i = 0; i < s.nOptions; i++) {
            float prior = 0.0f;
            int a = s.owner[i] * nEmbd;
            for (int j = 0; j < nEmbd; j++) {
                prior += lex[i * nEmbd + j] * anchor[a + j];
            }
            float cosine = 0.0f;
            for (int j = 0; j < nEmbdH; j++) {
                cosine += fn[i * nEmbdH + j] * on[i * nEmbdH + j];
            }
            float joint = cosine * scales[1] + residual[i];
            out[i] = prior * scales[0] + joint * scales[2];
        }
        for (int i = 0; i < n; i++) {
            out[i] += status;
        }
        return out;
    }

    /** {@code fields[q] += LN(sum_o softmax_o(options . fields[q] / sqrt(H) + mask) options[o])} (clef.cpp:587-594). */
    private void addOptionSummary(float[] fields, float[] options, Spans s) {
        float inv = 1.0f / (float) Math.sqrt(nEmbdH);
        float[] summary = new float[s.nQuestions * nEmbdH];
        float[] weights = new float[s.nOptions];
        for (int q = 0; q < s.nQuestions; q++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int o = 0; o < s.nOptions; o++) {
                float dot = 0.0f;
                for (int j = 0; j < nEmbdH; j++) {
                    dot += options[o * nEmbdH + j] * fields[q * nEmbdH + j];
                }
                weights[o] = s.owner[o] == q ? dot * inv : Float.NEGATIVE_INFINITY;
                max = Math.max(max, weights[o]);
            }
            float sum = 0.0f;
            for (int o = 0; o < s.nOptions; o++) {
                weights[o] = weights[o] == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(weights[o] - max);
                sum += weights[o];
            }
            for (int o = 0; o < s.nOptions; o++) {
                float p = weights[o] / sum;
                for (int j = 0; j < nEmbdH; j++) {
                    summary[q * nEmbdH + j] += p * options[o * nEmbdH + j];
                }
            }
        }
        HeadsBlocks.layerNorm(summary, s.nQuestions, nEmbdH, summaryNorm[0], summaryNorm[1], eps);
        HeadsBlocks.addInPlace(fields, summary, fields.length);
    }

    private float[] spanMeans(float[] h, int[] start, int[] end) {
        float[] out = new float[start.length * nEmbd];
        for (int i = 0; i < start.length; i++) {
            if (end[i] <= start[i]) {
                continue;
            }
            float w = 1.0f / (end[i] - start[i]);
            for (int t = start[i]; t < end[i]; t++) {
                for (int j = 0; j < nEmbd; j++) {
                    out[i * nEmbd + j] += h[t * nEmbd + j] * w;
                }
            }
        }
        return out;
    }

    private float[] lexicalMeans(int[] tokens, Spans s) {
        float[] out = new float[s.nOptions * nEmbd];
        float[] row = new float[nEmbd];
        for (int i = 0; i < s.nOptions; i++) {
            if (s.oEnd[i] <= s.oStart[i]) {
                continue;
            }
            float w = 1.0f / (s.oEnd[i] - s.oStart[i]);
            for (int t = s.oStart[i]; t < s.oEnd[i]; t++) {
                if (tokens[t] < 0 || tokens[t] >= output.rows()) {
                    throw new IllegalArgumentException("token " + tokens[t] + " at " + t + " is outside the "
                            + output.rows() + " output rows");
                }
                output.decodeRow(tokens[t], row, 0);
                for (int j = 0; j < nEmbd; j++) {
                    out[i * nEmbd + j] += row[j] * w;
                }
            }
        }
        return out;
    }

    private float[] broadcast(float[] row, int nRows) {
        float[] out = new float[nRows * row.length];
        for (int r = 0; r < nRows; r++) {
            System.arraycopy(row, 0, out, r * row.length, row.length);
        }
        return out;
    }

    /** Question and option spans of a usable order, or the single empty question and option of an unusable one. */
    private static final class Spans {
        final int nQuestions;
        final int nOptions;
        final int[] qType;
        final int[] qStart;
        final int[] qEnd;
        final int[] owner;
        final int[] oStart;
        final int[] oEnd;

        private Spans(int nq, int no) {
            nQuestions = nq;
            nOptions = no;
            qType = new int[nq];
            qStart = new int[nq];
            qEnd = new int[nq];
            owner = new int[no];
            oStart = new int[no];
            oEnd = new int[no];
        }

        static Spans empty() {
            return new Spans(1, 1);
        }

        static Spans of(DecisionOrder d) {
            Spans s = new Spans(d.questionCount(), d.optionCount());
            for (int i = 0; i < s.nQuestions; i++) {
                s.qType[i] = d.questionType(i);
                s.qStart[i] = d.questionStart(i);
                s.qEnd[i] = d.questionEnd(i);
            }
            for (int i = 0; i < s.nOptions; i++) {
                s.owner[i] = d.optionQuestion(i);
                s.oStart[i] = d.optionStart(i);
                s.oEnd[i] = d.optionEnd(i);
            }
            return s;
        }
    }

    /** One head block: routing blocks read the prompt, joint blocks also attend among the questions. */
    private static final class Block {
        final float[][] crossNormKv;
        final float[][] selfNorm;
        final Attention self;
        final float[][] crossNorm;
        final Attention cross;
        final float[][] ffnNorm;
        final CpuTensorRef up;
        final float[] upB;
        final CpuTensorRef down;
        final float[] downB;

        Block(CpuWeights w, int il, boolean routing, int width) {
            this.crossNormKv = routing ? stackNorm(w, il, "cross_attn_norm_kv", width) : null;
            this.selfNorm = routing ? null : stackNorm(w, il, "attn_norm", width);
            this.self = routing ? null : new Attention(w, il, "attn_q", "attn_k", "attn_v", "attn_o", width);
            this.crossNorm = stackNorm(w, il, "cross_attn_norm", width);
            this.cross = new Attention(w, il, "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o", width);
            this.ffnNorm = stackNorm(w, il, "ffn_norm", width);
            this.up = w.stack(TensorNames.STACK_DEC, il, TensorNames.weight("ffn_up"));
            if (up.cols() != width) {
                throw new IllegalArgumentException("clef head block " + il + " ffn_up takes " + up.cols()
                        + " inputs, the head width is " + width);
            }
            int ff = HeadsBlocks.toInt(up.rows(), up);
            this.upB = stackVector(w, il, TensorNames.bias("ffn_up"), ff);
            this.down = w.stack(TensorNames.STACK_DEC, il, TensorNames.weight("ffn_down"));
            if (down.cols() != ff || down.rows() != width) {
                throw new IllegalArgumentException("clef head block " + il + " ffn_down is " + down.cols() + " x "
                        + down.rows() + ", expected " + ff + " x " + width);
            }
            this.downB = stackVector(w, il, TensorNames.bias("ffn_down"), width);
        }

        /** {@code down(gelu_erf(up LN(x; ffn_norm) + b)) + b} of {@code nRows} rows (clef.cpp:520-525). */
        float[] ffn(ClefDecoder c, float[] x, int nRows) {
            float[] cur = HeadsBlocks.layerNormed(x, nRows, c.nEmbdH, ffnNorm[0], ffnNorm[1], c.eps);
            cur = HeadsBlocks.linear(c.mv, up, upB, cur, nRows);
            HeadsBlocks.activate(ActOp.GELU_ERF, cur, cur.length);
            return HeadsBlocks.linear(c.mv, down, downB, cur, nRows);
        }
    }

    /** Biased Q/K/V/O projections of one head attention (clef.cpp:497-518). */
    private static final class Attention {
        final CpuTensorRef wq;
        final float[] bq;
        final CpuTensorRef wk;
        final float[] bk;
        final CpuTensorRef wv;
        final float[] bv;
        final CpuTensorRef wo;
        final float[] bo;

        Attention(CpuWeights w, int il, String q, String k, String v, String o, int width) {
            this.wq = square(w, il, q, width);
            this.bq = stackVector(w, il, TensorNames.bias(q), width);
            this.wk = square(w, il, k, width);
            this.bk = stackVector(w, il, TensorNames.bias(k), width);
            this.wv = square(w, il, v, width);
            this.bv = stackVector(w, il, TensorNames.bias(v), width);
            this.wo = square(w, il, o, width);
            this.bo = stackVector(w, il, TensorNames.bias(o), width);
        }

        /** {@code wo attn(wq q + bq, wk kv + bk, wv kv + bv) + bo} for {@code nQ} query and {@code nKv} key rows. */
        float[] apply(ClefDecoder c, float[] q, int nQ, float[] kv, int nKv) {
            int width = c.nEmbdH;
            int dHead = width / c.nHead;
            float[] qh = HeadsBlocks.linear(c.mv, wq, bq, q, nQ);
            float[] k = HeadsBlocks.linear(c.mv, wk, bk, kv, nKv);
            float[] v = HeadsBlocks.linear(c.mv, wv, bv, kv, nKv);
            float[] packed = new float[nKv * 2 * width];
            for (int t = 0; t < nKv; t++) {
                System.arraycopy(k, t * width, packed, t * 2 * width, width);
                System.arraycopy(v, t * width, packed, t * 2 * width + width, width);
            }
            float[] ctx = HeadsBlocks.attendAll(qh, width, nQ, packed, 0, width, 2 * width, nKv, c.nHead, c.nHead,
                    dHead, 1.0f / (float) Math.sqrt(dHead));
            return HeadsBlocks.linear(c.mv, wo, bo, ctx, nQ);
        }

        private static CpuTensorRef square(CpuWeights w, int il, String base, int width) {
            CpuTensorRef m = w.stack(TensorNames.STACK_DEC, il, TensorNames.weight(base));
            if (m.cols() != width || m.rows() != width) {
                throw new IllegalArgumentException("clef head block " + il + " " + base + " is " + m.cols() + " x "
                        + m.rows() + ", expected " + width + " x " + width);
            }
            return m;
        }
    }

    private static float[][] stackNorm(CpuWeights w, int il, String base, int width) {
        return new float[][]{stackVector(w, il, TensorNames.weight(base), width),
                stackVector(w, il, TensorNames.bias(base), width)};
    }

    private static float[] stackVector(CpuWeights w, int il, String suffix, int n) {
        float[] v = w.stackFloatsOrNull(TensorNames.STACK_DEC, il, suffix);
        if (v == null) {
            throw new IllegalArgumentException("clef head block " + il + " has no " + suffix);
        }
        if (v.length != n) {
            throw new IllegalArgumentException("clef head block " + il + " " + suffix + " has " + v.length
                    + " floats, expected " + n);
        }
        return v;
    }
}
