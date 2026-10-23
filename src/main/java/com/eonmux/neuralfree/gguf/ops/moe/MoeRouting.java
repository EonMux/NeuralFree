package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import java.util.Objects;

/**
 * CPU MoE router, the reference twin of {@code LLM_MoE_Router} ({@code kernels/llm/moe.cl}), following llama.cpp
 * {@code build_moe_ffn} (src/llama-graph.cpp:2232-2345). Per token:
 * <ol>
 *   <li>{@code probs = gating(logits)}; SOFTMAX is {@code e_i * (1 / sum e)} with {@code e_i = exp(x_i - max)}
 *       (ggml_soft_max: f32 exponentials, sum in double, ggml-cpu ops.cpp soft_max_f32);</li>
 *   <li>selection scores by {@link Params#selMode()}: probs (+ {@code exp_probs_b}), raw logits (llama4) or
 *       {@code sigmoid(logits)} (grovemoe);</li>
 *   <li>with {@code nGroups > 1}, group-limited routing (llama-graph.cpp:2281-2305): a group scores the sum of its
 *       two largest selection scores, the {@code nGroupUsed} best groups are kept and the others masked to -INF;</li>
 *   <li>{@code k} rounds of arg-max over the unpicked scores, ties to the lower index, NaN ranking as -INF;</li>
 *   <li>grovemoe chunk pass ({@code idDiv > 0}): {@code id = (int)((float) id * (1f / idDiv))}
 *       (llama-graph.cpp:2312-2318);</li>
 *   <li>{@code w_j = probs[id_j]}; SOFTMAX_WEIGHT softmax over the k weights; {@code normWeights}:
 *       {@code w / clamp(sum w, NORM_SUM_MIN, INF)}; {@code weightsScale} not 0 or 1: {@code w * scale}.</li>
 * </ol>
 * The selected ids and their order equal the GPU kernel's for any work-group size: both use the same total order,
 * and the selection scores of the PROBS mode differ from the GPU only in the softmax sum's reduction order (a few
 * ulps), which cannot reorder distinct scores separated by more than that. Hash-routed layers (deepseek4) take
 * their ids from {@link MoeCombine#hashIds} and only compute the weights here ({@link Params#preselected}).
 */
public final class MoeRouting {

    private MoeRouting() {
    }

    /** Immutable router configuration; the field set and codes are those of the kernel arguments. */
    public static final class Params {
        private final int nExpert;
        private final int k;
        private final GatingFunc gating;
        private final int nGroups;
        private final int nGroupUsed;
        private final boolean normWeights;
        private final float weightsScale;
        private final boolean hasBias;
        private final int selMode;
        private final int idDiv;

        /**
         * @param nExpert      router logits per token
         * @param k            experts selected per token
         * @param nGroups      expert groups; 1 or less disables group-limited routing
         * @param nGroupUsed   groups kept when {@code nGroups > 1}
         * @param hasBias      {@code exp_probs_b} is added to the PROBS selection scores
         * @param selMode      {@link MoeKernels#SEL_PROBS}, {@link MoeKernels#SEL_RAW_LOGITS},
         *                     {@link MoeKernels#SEL_SIGMOID_LOGITS} or {@link MoeKernels#SEL_PRESELECTED}
         * @param idDiv        grovemoe chunk divisor of the selected ids; 0 disables it
         * @throws IllegalArgumentException when the configuration is inconsistent
         */
        public Params(int nExpert, int k, GatingFunc gating, int nGroups, int nGroupUsed, boolean normWeights,
                      float weightsScale, boolean hasBias, int selMode, int idDiv) {
            this.gating = Objects.requireNonNull(gating, "gating");
            if (nExpert <= 0 || k <= 0 || k > nExpert) {
                throw new IllegalArgumentException("router needs 0 < k <= n_expert, got k = " + k + ", n_expert = "
                        + nExpert);
            }
            if (selMode < MoeKernels.SEL_PROBS || selMode > MoeKernels.SEL_PRESELECTED) {
                throw new IllegalArgumentException("unknown selection mode " + selMode);
            }
            int groups = Math.max(1, nGroups);
            if (groups > 1 && (nExpert % groups != 0 || nExpert / groups < 2 || nGroupUsed <= 0
                    || nGroupUsed > groups)) {
                throw new IllegalArgumentException("group-limited routing needs n_expert divisible by n_groups with"
                        + " at least two experts per group and 0 < n_group_used <= n_groups, got " + nExpert + " / "
                        + groups + " / " + nGroupUsed);
            }
            if (idDiv < 0 || idDiv >= (1 << 24) || nExpert >= (1 << 24)) {
                throw new IllegalArgumentException("id divisor must be in [0, 2^24) with fewer than 2^24 experts, got "
                        + idDiv);
            }
            this.nExpert = nExpert;
            this.k = k;
            this.nGroups = groups;
            this.nGroupUsed = groups > 1 ? nGroupUsed : 1;
            this.normWeights = normWeights;
            this.weightsScale = weightsScale;
            this.hasBias = hasBias;
            this.selMode = selMode;
            this.idDiv = idDiv;
        }

        /**
         * The routed-expert pass of a MoE layer: {@code nUsed} of {@code nExpert}, with the spec's gating, groups,
         * selection mode and {@code exp_probs_b}; hash-routed layers get {@link MoeKernels#SEL_PRESELECTED}
         * without the bias (models/deepseek4.cpp:1297 passes {@code exp_probs_b = nullptr}).
         */
        public static Params forExperts(MoeSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (spec.hashRouting()) {
                return preselected(spec);
            }
            return new Params(spec.nExpert(), spec.nUsed(), spec.gating(), spec.nGroups(), spec.nGroupUsed(),
                    spec.normWeights(), spec.weightsScale(), spec.probBias(), spec.selectionMode().code(), 0);
        }

        /**
         * The grovemoe chunk-expert pass (models/grovemoe.cpp:153-164): the same router logits over all
         * {@code nExpert} experts, {@code k = min(nUsed, nChunkExperts)}, ids divided by {@code idDiv}, no
         * {@code exp_probs_b}.
         *
         * @throws IllegalStateException when the spec has no chunk experts or uses expert groups, which llama.cpp
         *                               cannot reshape for the chunk count
         */
        public static Params forChunks(MoeSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (!spec.chunked() || spec.nChunkExperts() <= 0) {
                throw new IllegalStateException("MoE spec has no chunk experts: " + spec);
            }
            if (spec.nGroups() > 1) {
                throw new IllegalStateException("chunk experts with expert groups are not defined by llama.cpp: " + spec);
            }
            return new Params(spec.nExpert(), Math.min(spec.nUsed(), spec.nChunkExperts()), spec.gating(), 1, 1,
                    spec.normWeights(), spec.weightsScale(), false, spec.selectionMode().code(), spec.idDiv());
        }

        /** Weights of given ids (hash-routed layers): selection, groups, bias and id division are skipped. */
        public static Params preselected(MoeSpec spec) {
            Objects.requireNonNull(spec, "spec");
            return new Params(spec.nExpert(), spec.nUsed(), spec.gating(), 1, 1, spec.normWeights(),
                    spec.weightsScale(), false, MoeKernels.SEL_PRESELECTED, 0);
        }

        public int nExpert() { return nExpert; }
        public int k() { return k; }
        public GatingFunc gating() { return gating; }
        public int nGroups() { return nGroups; }
        public int nGroupUsed() { return nGroupUsed; }
        public boolean normWeights() { return normWeights; }
        public float weightsScale() { return weightsScale; }
        public boolean hasBias() { return hasBias; }
        public int selMode() { return selMode; }
        public int idDiv() { return idDiv; }

        @Override
        public String toString() {
            return "MoeRouting.Params{experts=" + nExpert + ", k=" + k + ", gating=" + gating + ", groups=" + nGroups
                    + "/" + nGroupUsed + ", norm=" + normWeights + ", scale=" + weightsScale + ", bias=" + hasBias
                    + ", sel=" + selMode + ", idDiv=" + idDiv + '}';
        }
    }

    /** Routing result of a batch: {@code ids[t * k + j]} and {@code weights[t * k + j]}. Immutable. */
    public static final class Selection {
        private final int k;
        private final int[] ids;
        private final float[] weights;

        private Selection(int k, int[] ids, float[] weights) {
            this.k = k;
            this.ids = ids;
            this.weights = weights;
        }

        public int k() { return k; }
        public int nTokens() { return ids.length / k; }
        public int id(int t, int j) { return ids[t * k + j]; }
        public float weight(int t, int j) { return weights[t * k + j]; }
        public int[] ids() { return ids.clone(); }
        public float[] weights() { return weights.clone(); }
    }

    /**
     * Routes {@code nTokens} tokens whose logits are {@code logits[t * nExpert + e]} (router bias already added).
     *
     * @param expProbsB selection bias {@code [nExpert]}; required when {@link Params#hasBias()}, else ignored
     * @throws IllegalArgumentException when a buffer is too small or the mode is {@link MoeKernels#SEL_PRESELECTED}
     */
    public static Selection route(Params p, float[] logits, int nTokens, float[] expProbsB) {
        Objects.requireNonNull(p, "params");
        if (p.selMode() == MoeKernels.SEL_PRESELECTED) {
            throw new IllegalArgumentException("preselected routing takes its ids: use routePreselected");
        }
        checkLogits(p, logits, nTokens);
        if (p.hasBias() && (expProbsB == null || expProbsB.length < p.nExpert())) {
            throw new IllegalArgumentException("exp_probs_b must hold " + p.nExpert() + " values");
        }
        int[] ids = new int[nTokens * p.k()];
        float[] weights = new float[nTokens * p.k()];
        for (int t = 0; t < nTokens; t++) {
            routeToken(p, logits, t, expProbsB, null, ids, weights);
        }
        return new Selection(p.k(), ids, weights);
    }

    /**
     * Weights of given ids ({@code ids[t * k + j]}, e.g. from {@link MoeCombine#hashIds}); the ids are kept, an id
     * outside {@code [0, nExpert)} gets weight 0 before normalization, as in the kernel.
     */
    public static Selection routePreselected(Params p, float[] logits, int nTokens, int[] ids) {
        Objects.requireNonNull(p, "params");
        if (p.selMode() != MoeKernels.SEL_PRESELECTED) {
            throw new IllegalArgumentException("routePreselected needs SEL_PRESELECTED params, got " + p);
        }
        checkLogits(p, logits, nTokens);
        Objects.requireNonNull(ids, "ids");
        if (ids.length < nTokens * p.k()) {
            throw new IllegalArgumentException("ids must hold " + nTokens * p.k() + " values, got " + ids.length);
        }
        int[] outIds = new int[nTokens * p.k()];
        float[] weights = new float[nTokens * p.k()];
        for (int t = 0; t < nTokens; t++) {
            routeToken(p, logits, t, null, ids, outIds, weights);
        }
        return new Selection(p.k(), outIds, weights);
    }

    private static void checkLogits(Params p, float[] logits, int nTokens) {
        Objects.requireNonNull(logits, "logits");
        if (nTokens < 0 || (long) nTokens * p.nExpert() > logits.length) {
            throw new IllegalArgumentException("logits must hold " + nTokens + " x " + p.nExpert() + " values, got "
                    + logits.length);
        }
    }

    private static void routeToken(Params p, float[] logits, int t, float[] expProbsB, int[] given,
                                   int[] ids, float[] weights) {
        int n = p.nExpert();
        int xb = t * n;
        float[] probs = gate(p.gating(), logits, xb, n);
        int[] pick = new int[p.k()];
        if (p.selMode() == MoeKernels.SEL_PRESELECTED) {
            System.arraycopy(given, t * p.k(), pick, 0, p.k());
        } else {
            float[] sel = selectionScores(p, logits, xb, probs, expProbsB);
            if (p.nGroups() > 1) {
                maskGroups(sel, p.nGroups(), p.nGroupUsed());
            }
            topK(sel, pick);
            if (p.idDiv() > 0) {
                for (int j = 0; j < pick.length; j++) {
                    pick[j] = divId(pick[j], p.idDiv());
                }
            }
        }
        float[] w = weightsOf(p, probs, pick);
        System.arraycopy(pick, 0, ids, t * p.k(), p.k());
        System.arraycopy(w, 0, weights, t * p.k(), p.k());
    }

    /** Gating probabilities of one token's logits {@code x[off .. off + n)}. */
    static float[] gate(GatingFunc gating, float[] x, int off, int n) {
        float[] probs = new float[n];
        switch (gating) {
            case SOFTMAX: {
                float m = Float.NEGATIVE_INFINITY;
                for (int e = 0; e < n; e++) {
                    m = fmax(m, x[off + e]);
                }
                double sum = 0.0;
                for (int e = 0; e < n; e++) {
                    probs[e] = (float) Math.exp(x[off + e] - m);
                    sum += probs[e];
                }
                float inv = (float) (1.0 / sum);
                for (int e = 0; e < n; e++) {
                    probs[e] *= inv;
                }
                return probs;
            }
            case SIGMOID:
                for (int e = 0; e < n; e++) {
                    probs[e] = ActMath.sigmoid(x[off + e]);
                }
                return probs;
            case SOFTMAX_WEIGHT:
                System.arraycopy(x, off, probs, 0, n);
                return probs;
            case SQRT_SOFTPLUS:
                for (int e = 0; e < n; e++) {
                    probs[e] = (float) Math.sqrt(ActMath.softplus(x[off + e]));
                }
                return probs;
            default:
                throw new IllegalStateException("unhandled gating function " + gating);
        }
    }

    private static float[] selectionScores(Params p, float[] logits, int xb, float[] probs, float[] expProbsB) {
        int n = p.nExpert();
        float[] sel = new float[n];
        for (int e = 0; e < n; e++) {
            switch (p.selMode()) {
                case MoeKernels.SEL_RAW_LOGITS:
                    sel[e] = logits[xb + e];
                    break;
                case MoeKernels.SEL_SIGMOID_LOGITS:
                    sel[e] = ActMath.sigmoid(logits[xb + e]);
                    break;
                default:
                    sel[e] = p.hasBias() ? probs[e] + expProbsB[e] : probs[e];
                    break;
            }
        }
        return sel;
    }

    /**
     * Group-limited routing in place: a group's score is the sum of its two largest keys; the {@code nGroupUsed}
     * best groups (ties to the lower group) are kept and every other score becomes -INF.
     */
    static void maskGroups(float[] sel, int nGroups, int nGroupUsed) {
        int nPer = sel.length / nGroups;
        float[] score = new float[nGroups];
        for (int g = 0; g < nGroups; g++) {
            float a = Float.NEGATIVE_INFINITY;
            float b = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < nPer; i++) {
                float v = key(sel[g * nPer + i]);
                if (v > a) {
                    b = a;
                    a = v;
                } else if (v > b) {
                    b = v;
                }
            }
            score[g] = a + b;
        }
        boolean[] keep = new boolean[nGroups];
        for (int r = 0; r < nGroupUsed; r++) {
            float bv = Float.NEGATIVE_INFINITY;
            int bi = -1;
            for (int g = 0; g < nGroups; g++) {
                float v = key(score[g]);
                if (!keep[g] && better(v, g, bv, bi)) {
                    bv = v;
                    bi = g;
                }
            }
            if (bi >= 0) {
                keep[bi] = true;
            }
        }
        for (int e = 0; e < sel.length; e++) {
            if (!keep[e / nPer]) {
                sel[e] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    /** {@code pick.length} rounds of arg-max over the unpicked keys of {@code sel}, ties to the lower index. */
    static void topK(float[] sel, int[] pick) {
        boolean[] taken = new boolean[sel.length];
        for (int j = 0; j < pick.length; j++) {
            float bv = Float.NEGATIVE_INFINITY;
            int bi = -1;
            for (int e = 0; e < sel.length; e++) {
                float v = key(sel[e]);
                if (!taken[e] && better(v, e, bv, bi)) {
                    bv = v;
                    bi = e;
                }
            }
            pick[j] = bi;
            if (bi >= 0) {
                taken[bi] = true;
            }
        }
    }

    private static float[] weightsOf(Params p, float[] probs, int[] pick) {
        int k = pick.length;
        float[] w = new float[k];
        for (int j = 0; j < k; j++) {
            int id = pick[j];
            w[j] = id >= 0 && id < probs.length ? probs[id] : 0.0f;
        }
        if (p.gating() == GatingFunc.SOFTMAX_WEIGHT) {
            float wm = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                wm = fmax(wm, w[j]);
            }
            float s = 0.0f;
            for (int j = 0; j < k; j++) {
                w[j] = (float) Math.exp(w[j] - wm);
                s += w[j];
            }
            float inv = 1.0f / s;
            for (int j = 0; j < k; j++) {
                w[j] *= inv;
            }
        }
        if (p.normWeights()) {
            float s = 0.0f;
            for (int j = 0; j < k; j++) {
                s += w[j];
            }
            s = ActMath.clamp(s, MoeSpec.NORM_SUM_MIN, Float.POSITIVE_INFINITY);
            for (int j = 0; j < k; j++) {
                w[j] = w[j] / s;
            }
        }
        float scale = p.weightsScale();
        if (scale != 0.0f && scale != 1.0f) {
            for (int j = 0; j < k; j++) {
                w[j] *= scale;
            }
        }
        return w;
    }

    /** grovemoe id division {@code (int)((float) id * (1f / div))} (llama-graph.cpp:2312-2318). */
    public static int divId(int id, int div) {
        if (div <= 0) {
            throw new IllegalArgumentException("id divisor must be positive, got " + div);
        }
        return (int) ((float) id * (1.0f / (float) div));
    }

    /** Selection key: NaN ranks as -INF. */
    static float key(float v) {
        return Float.isNaN(v) ? Float.NEGATIVE_INFINITY : v;
    }

    /** Candidate {@code (cv, ci)} ranks before {@code (bv, bi)}: higher key, then lower index; {@code bi < 0} is empty. */
    static boolean better(float cv, int ci, float bv, int bi) {
        return ci >= 0 && (bi < 0 || cv > bv || (cv == bv && ci < bi));
    }

    /** C {@code fmax}: a NaN operand yields the other. */
    private static float fmax(float a, float b) {
        if (Float.isNaN(a)) {
            return b;
        }
        if (Float.isNaN(b)) {
            return a;
        }
        return Math.max(a, b);
    }
}
