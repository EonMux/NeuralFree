package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The ranking classifier of RANK pooling (llama-graph.cpp:4146-4176 build_pooling) over one pooled vector of
 * {@link #inputs()} floats (the embedding row width, {@code HeadSpec.embeddingWidth}):
 * <pre>
 *   if cls:     x = act_cls(cls x + cls_b);  if cls_norm: x = LayerNorm(x) * cls_norm   (no bias, eps = f_norm_eps)
 *   if cls_out: x = cls_out x + cls_out_b
 *   qwen3 / qwen3vl: x = softmax(x)
 * </pre>
 * with {@code act_cls} from {@code classifier.activation} (default tanh; modern-bert gelu_erf) and the labels from
 * {@code classifier.output_labels}. Without {@code cls} and {@code cls_out} the pooled vector is the output.
 *
 * <p>Also defines the rerank input ({@link #rerankInput}, with the token-level packing {@link #packRerank}) and the
 * decision order of the decision heads ({@link DecisionOrder}). Immutable after construction; {@link #apply}
 * allocates its result.</p>
 */
public final class ClassifierHead {

    /** Placeholder of the query in a rerank prompt. */
    public static final String RERANK_QUERY = "{query}";
    /** Placeholder of the document in a rerank prompt. */
    public static final String RERANK_DOCUMENT = "{document}";

    private final CpuMatVec mv;
    private final int nIn;
    private final CpuTensorRef cls;
    private final float[] clsB;
    private final ActOp act;
    private final float[] clsNorm;
    private final float eps;
    private final CpuTensorRef clsOut;
    private final float[] clsOutB;
    private final boolean softmax;
    private final String[] labels;
    private final int nOut;

    private ClassifierHead(CpuWeights w, ModelHparams hp, Arch arch, int inputs) {
        this.mv = w.matVec();
        this.nIn = inputs;
        this.cls = w.getOrNull(TensorNames.weight(TensorNames.CLS));
        this.clsB = cls == null ? null : w.floatsOrNull(TensorNames.bias(TensorNames.CLS));
        this.act = hp.classifierActivation();
        this.clsNorm = cls == null ? null : w.floatsOrNull(TensorNames.weight(TensorNames.CLS_NORM));
        this.eps = hp.normEps();
        this.clsOut = w.getOrNull(TensorNames.weight(TensorNames.CLS_OUT));
        this.clsOutB = clsOut == null ? null : w.floatsOrNull(TensorNames.bias(TensorNames.CLS_OUT));
        this.softmax = arch == Arch.QWEN3 || arch == Arch.QWEN3VL;
        this.labels = hp.classifierLabels();
        int width = nIn;
        if (cls != null) {
            width = checkInput(cls, width);
            if (clsNorm != null && clsNorm.length != width) {
                throw new IllegalArgumentException("cls.norm has " + clsNorm.length + " floats, cls outputs " + width);
            }
        }
        if (clsOut != null) {
            width = checkInput(clsOut, width);
        }
        this.nOut = width;
    }

    /**
     * The ranking classifier of {@code spec} over the tensors of {@code weights}, reading pooled embedding rows.
     *
     * @throws IllegalArgumentException when a classifier tensor does not take the previous stage's width
     */
    public static ClassifierHead of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        int inputs = spec.head().embeddingWidth(spec.hparams().nEmbd());
        return new ClassifierHead(weights, spec.hparams(), spec.arch(), inputs);
    }

    private static int checkInput(CpuTensorRef w, int width) {
        if (w.cols() != width) {
            throw new IllegalArgumentException("classifier tensor " + w + " takes " + w.cols() + " inputs, the"
                    + " previous stage gives " + width);
        }
        return HeadsBlocks.toInt(w.rows(), w);
    }

    /** Width of the pooled vector {@link #apply} reads. */
    public int inputs() {
        return nIn;
    }

    /** Number of scores {@link #apply} returns. */
    public int outputs() {
        return nOut;
    }

    /** True when the scores are probabilities (the qwen3 / qwen3vl softmax). */
    public boolean normalized() {
        return softmax;
    }

    /** {@code classifier.output_labels} (may be shorter than {@link #outputs()}). */
    public String[] labels() {
        return labels.clone();
    }

    /**
     * The classifier scores of the pooled vector {@code pooled[off .. off + inputs())}.
     *
     * @throws IllegalArgumentException when the vector does not fit {@code pooled}
     */
    public float[] apply(float[] pooled, int off) {
        Objects.requireNonNull(pooled, "pooled");
        if (off < 0 || (long) off + nIn > pooled.length) {
            throw new IllegalArgumentException("pooled vector [" + off + ", " + ((long) off + nIn)
                    + ") is outside an array of " + pooled.length);
        }
        float[] x = new float[nIn];
        System.arraycopy(pooled, off, x, 0, nIn);
        if (cls != null) {
            x = HeadsBlocks.linear(mv, cls, clsB, x, 1);
            HeadsBlocks.activate(act, x, x.length);
            if (clsNorm != null) {
                HeadsBlocks.layerNorm(x, 1, x.length, clsNorm, null, eps);
            }
        }
        if (clsOut != null) {
            x = HeadsBlocks.linear(mv, clsOut, clsOutB, x, 1);
        }
        if (softmax) {
            CpuOps.softmax(x, 0, x.length);
        }
        return x;
    }

    /** {@link #apply} wrapped with the labels. */
    public ClassifierOutput output(float[] pooled, int off) {
        return ClassifierOutput.of(apply(pooled, off), labels, softmax);
    }

    // ================================================================== rerank input

    /**
     * Rerank input of a query and a document (tools/server/server-common.cpp:1948-1991 format_prompt_rerank):
     * <ul>
     *   <li>with a {@code tokenizer.chat_template.rerank} prompt ({@link GGUFTokenizer#rerankTemplate()}): the prompt
     *       with every {@code {query}} replaced by the query and then every {@code {document}} by the document
     *       (string_replace_all, common/common.cpp:510-525), tokenized with special tokens parsed and none added;</li>
     *   <li>otherwise: the query and the document tokenized without added special tokens and without special-token
     *       parsing, and joined by {@link #packRerank}.</li>
     * </ul>
     *
     * @throws IllegalStateException when the rerank prompt metadata is not a string, or the packing needs a special
     *                               token the vocabulary lacks
     */
    public static int[] rerankInput(GGUFTokenizer tokenizer, String query, String document) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(document, "document");
        String template = tokenizer.rerankTemplate();
        if (template != null) {
            String prompt = template.replace(RERANK_QUERY, query).replace(RERANK_DOCUMENT, document);
            return tokenizer.encode(prompt, false);
        }
        return packRerank(tokenizer.encode(query, false, false), tokenizer.encode(document, false, false),
                tokenizer.vocab(), tokenizer.specialTokens());
    }

    /**
     * Rerank input of an already tokenized query and document: {@link #packRerank} with the tokenizer's vocabulary
     * flags and special tokens.
     *
     * @throws UnsupportedOperationException when the model has a rerank prompt, which is filled with the text of the
     *                                       query and document ({@link #rerankInput(GGUFTokenizer, String, String)})
     * @throws IllegalStateException         when the packing needs a special token the vocabulary lacks
     */
    public static int[] rerankInput(GGUFTokenizer tokenizer, int[] query, int[] document) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (tokenizer.rerankTemplate() != null) {
            throw new UnsupportedOperationException("the model has a " + GGUFTokenizer.KEY_RERANK_TEMPLATE
                    + " prompt, which takes the query and document text; rerank the text instead of tokens");
        }
        return packRerank(query, document, tokenizer.vocab(), tokenizer.specialTokens());
    }

    /**
     * Rerank input of a query and a document without a {@code rerank} chat template
     * (tools/server/server-common.cpp:1966-1986): {@code [BOS] query [EOS] [SEP] doc [EOS]}, each special token
     * present when the vocabulary's {@code add_bos} / {@code add_eos} / {@code add_sep} flag is set; EOS falls back to
     * SEP when the vocabulary has no EOS.
     *
     * @throws IllegalStateException when a flag requests a token the vocabulary does not define
     */
    public static int[] packRerank(int[] query, int[] document, TokenizerVocab vocab, SpecialTokens special) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(document, "document");
        Objects.requireNonNull(vocab, "vocab");
        Objects.requireNonNull(special, "special");
        int eos = special.eos() != SpecialTokens.NULL ? special.eos() : special.sep();
        int n = query.length + document.length + (vocab.addBos() ? 1 : 0) + (vocab.addEos() ? 2 : 0)
                + (vocab.addSep() ? 1 : 0);
        int[] out = new int[n];
        int at = 0;
        if (vocab.addBos()) {
            out[at++] = special.requireBos();
        }
        System.arraycopy(query, 0, out, at, query.length);
        at += query.length;
        if (vocab.addEos()) {
            out[at++] = requireToken(eos, "EOS (or SEP)");
        }
        if (vocab.addSep()) {
            out[at++] = special.requireSep();
        }
        System.arraycopy(document, 0, out, at, document.length);
        at += document.length;
        if (vocab.addEos()) {
            out[at++] = requireToken(eos, "EOS (or SEP)");
        }
        return out;
    }

    private static int requireToken(int id, String what) {
        if (id == SpecialTokens.NULL) {
            throw new IllegalStateException("rerank packing needs a " + what + " token, the vocabulary has none");
        }
        return id;
    }

    // ================================================================== decision order

    /**
     * The decision order of one input (llama-ext.h:100-125, {@code llama_batch_ext_set_decision_order}): one code
     * per token, a run of equal codes forming one span; questions {@link #QUESTION_NOUL},
     * {@link #QUESTION_CHOICE}, {@link #QUESTION_SCORE} (question type = code - 1), options {@link #OPTION}, which
     * belong to the last question before them, and {@link #NONE} elsewhere. Immutable.
     */
    public static final class DecisionOrder {

        public static final int NONE = 0;
        public static final int QUESTION_NOUL = 1;
        public static final int QUESTION_CHOICE = 2;
        public static final int QUESTION_SCORE = 3;
        public static final int OPTION = 4;

        private final boolean given;
        private final boolean wellFormed;
        private final int[] questions;
        private final int[] options;

        private DecisionOrder(boolean given, boolean wellFormed, int[] questions, int[] options) {
            this.given = given;
            this.wellFormed = wellFormed;
            this.questions = questions;
            this.options = options;
        }

        /**
         * Splits {@code order} into spans.
         *
         * @param order one code per token, or null when the input has no decision order
         * @throws IllegalArgumentException when {@code order} and the token count differ
         */
        public static DecisionOrder parse(int[] order, int nTokens) {
            if (order == null) {
                return new DecisionOrder(false, false, new int[0], new int[0]);
            }
            if (order.length != nTokens) {
                throw new IllegalArgumentException(order.length + " decision codes for " + nTokens + " tokens");
            }
            List<int[]> qs = new ArrayList<>();
            List<int[]> os = new ArrayList<>();
            boolean ok = true;
            for (int i = 0; i < nTokens; ) {
                int code = order[i];
                int end = i + 1;
                while (end < nTokens && order[end] == code) {
                    end++;
                }
                if (code >= QUESTION_NOUL && code <= QUESTION_SCORE) {
                    qs.add(new int[]{code - QUESTION_NOUL, i, end});
                } else if (code == OPTION) {
                    os.add(new int[]{qs.size() - 1, i, end});
                } else if (code != NONE) {
                    ok = false;
                }
                i = end;
            }
            return new DecisionOrder(true, ok, flatten(qs), flatten(os));
        }

        private static int[] flatten(List<int[]> spans) {
            int[] out = new int[spans.size() * 3];
            for (int i = 0; i < spans.size(); i++) {
                System.arraycopy(spans.get(i), 0, out, i * 3, 3);
            }
            return out;
        }

        /** True when the input carries a decision order. */
        public boolean given() { return given; }
        /** True when every code is known. */
        public boolean wellFormed() { return wellFormed; }
        public int questionCount() { return questions.length / 3; }
        public int optionCount() { return options.length / 3; }
        /** Question type: 0 noul, 1 choice, 2 score. */
        public int questionType(int i) { return questions[i * 3]; }
        public int questionStart(int i) { return questions[i * 3 + 1]; }
        public int questionEnd(int i) { return questions[i * 3 + 2]; }
        /** Owning question of option {@code i}, -1 when no question precedes it. */
        public int optionQuestion(int i) { return options[i * 3]; }
        public int optionStart(int i) { return options[i * 3 + 1]; }
        public int optionEnd(int i) { return options[i * 3 + 2]; }

        /** True when every question has at least one option and every option follows a question. */
        public boolean everyQuestionAnswered() {
            boolean[] has = new boolean[questionCount()];
            for (int i = 0; i < optionCount(); i++) {
                int q = optionQuestion(i);
                if (q < 0) {
                    return false;
                }
                has[q] = true;
            }
            for (boolean b : has) {
                if (!b) {
                    return false;
                }
            }
            return true;
        }

        /** Value added to every score: 0, or NaN for a given order the head cannot use (llama's status input). */
        public static float status(boolean given, boolean valid) {
            return valid || !given ? 0.0f : Float.NaN;
        }
    }
}
