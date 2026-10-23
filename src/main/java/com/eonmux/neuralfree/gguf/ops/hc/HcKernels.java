package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;

/**
 * Names, argument indices, mode codes, per-variant parameters and dispatch geometry of the hyper-connection and
 * attention-residual kernels ({@code kernels/llm/hc.cl} and its GLSL twin {@code shaders/llm/hc.comp}). Every
 * index constant is the position of that argument in the kernel signature; buffers follow the scalars. Every
 * activation buffer has an {@code *_OFF} element-offset argument, so a {@code ScratchSlot} view
 * binds as its buffer plus its offset; the offsets match the {@code *Off} parameters of the CPU twins.
 *
 * <p>Layouts shared by the kernels and the CPU twins ({@link CpuHyperConnections}, {@link CpuAttnResidualStack}),
 * element offsets with {@code n_embd} innermost as in ggml:
 * <ul>
 *   <li>stream {@code X[(t*hc + h)*n_embd + i]} (ggml {@code [n_embd, hc, n_tokens]}); the residual stream of a
 *       hyper-connection model is {@code hc * n_embd} values per token;</li>
 *   <li>{@code pre[t*hc + h]}, {@code post[t*hc + h]};</li>
 *   <li>{@code comb[t*hc*hc + dst + hc*src]} (dst innermost, ggml {@code [dst_hc, src_hc, n_tokens]});</li>
 *   <li>attention-residual checkpoint stack {@code stack[(c*n_tokens + t)*n_embd + i]} and mixing probabilities
 *       {@code probs[t*(n_ckpt + 1) + c]}, the last column belonging to the current stream.</li>
 * </ul>
 *
 * <p>Per-architecture use (llama.cpp src/models): the sub-block input is {@code mixes = W_fn . rms(flatten X)}
 * computed with the generic norm and matvec kernels, then
 * <ul>
 *   <li>deepseek4, glm5-next, dflash (mHC, deepseek4.cpp:353-446): {@link CoefficientParams#mhc},
 *       {@link PreParams#weights()}, {@link PostParams#mixing()};</li>
 *   <li>hy_v4 (iHC, hy-v4.cpp:175-245): {@link CoefficientParams#ihc}, {@link PreParams#weights()},
 *       {@link PostParams#identity()};</li>
 *   <li>HC head of deepseek4, dflash and hy_v4 (deepseek4.cpp:448-466, hy-v4.cpp:248-270):
 *       {@link CoefficientParams#head}, {@link PreParams#weights()};</li>
 *   <li>qwen4exp (qwen4exp.cpp:311-395): {@link PreParams#gated} over the grouped-RMS-normed streams and
 *       {@link PostParams#gatedInject} with {@code post = W_inject . xn};</li>
 *   <li>stream mean (build_hc_mean): layer-input taps of deepseek4 (deepseek4.cpp:1245-1249, 1337) and the
 *       glm5-next output collapse (glm5-next.cpp:791, which has no HC head): {@link PreParams#mean};</li>
 *   <li>kimi-k3 attention residuals (kimi-k3.cpp:188-235): {@link #RES_STACK_SCORE} then {@link #RES_STACK_MIX}.</li>
 * </ul>
 */
public final class HcKernels {

    /** Per-token pre/post/comb coefficients from the projected mixes (Sinkhorn for comb). */
    public static final String HC_PRE_POST_COMB = "LLM_HC_PrePostComb";
    /** Collapse of the hc streams into one row (weights, gated or mean). */
    public static final String HC_PRE = "LLM_HC_Pre";
    /** Expansion of a sub-block output back into hc streams (replaces the residual add). */
    public static final String HC_POST = "LLM_HC_Post";
    /** Attention-residual mixing probabilities over the checkpoint stack and the current stream. */
    public static final String RES_STACK_SCORE = "LLM_ResStack_Score";
    /** Attention-residual weighted sum of the checkpoint stack and the current stream. */
    public static final String RES_STACK_MIX = "LLM_ResStack_Mix";

    // ---- LLM_HC_PrePostComb: 11 scalars, 6 buffers ----
    /** Re-set per step. */
    public static final int COEF_ARG_N_TOKENS = 0;
    public static final int COEF_ARG_HC = 1;
    public static final int COEF_ARG_EPS = 2;
    public static final int COEF_ARG_ITERS = 3;
    public static final int COEF_ARG_N_OUT = 4;
    public static final int COEF_ARG_POST_MAG = 5;
    public static final int COEF_ARG_POST_EPS = 6;
    public static final int COEF_ARG_MIXES_OFF = 7;
    public static final int COEF_ARG_PRE_OFF = 8;
    public static final int COEF_ARG_POST_OFF = 9;
    public static final int COEF_ARG_COMB_OFF = 10;
    public static final int COEF_ARG_MIXES = 11;
    public static final int COEF_ARG_SCALE = 12;
    public static final int COEF_ARG_BASE = 13;
    public static final int COEF_ARG_PRE = 14;
    public static final int COEF_ARG_POST = 15;
    public static final int COEF_ARG_COMB = 16;
    public static final int COEF_ARG_COUNT = 17;

    // ---- LLM_HC_Pre: 8 scalars, 3 buffers ----
    /** Re-set per step. */
    public static final int PRE_ARG_N_TOKENS = 0;
    public static final int PRE_ARG_N_EMBD = 1;
    public static final int PRE_ARG_HC = 2;
    public static final int PRE_ARG_MODE = 3;
    public static final int PRE_ARG_GSCALE = 4;
    public static final int PRE_ARG_X_OFF = 5;
    public static final int PRE_ARG_W_OFF = 6;
    public static final int PRE_ARG_OUT_OFF = 7;
    public static final int PRE_ARG_X = 8;
    public static final int PRE_ARG_W_OR_GATE = 9;
    /** Must not overlap {@link #PRE_ARG_X}. */
    public static final int PRE_ARG_OUT = 10;
    public static final int PRE_ARG_COUNT = 11;

    // ---- LLM_HC_Post: 12 scalars, 5 buffers ----
    /** Re-set per step. */
    public static final int POST_ARG_N_TOKENS = 0;
    public static final int POST_ARG_N_EMBD = 1;
    public static final int POST_ARG_HC = 2;
    public static final int POST_ARG_HAS_COMB = 3;
    public static final int POST_ARG_POST_ACT = 4;
    public static final int POST_ARG_POST_IN_SCALE = 5;
    public static final int POST_ARG_POST_MAG = 6;
    public static final int POST_ARG_Y_OFF = 7;
    public static final int POST_ARG_X_OFF = 8;
    public static final int POST_ARG_POST_OFF = 9;
    public static final int POST_ARG_COMB_OFF = 10;
    public static final int POST_ARG_XOUT_OFF = 11;
    public static final int POST_ARG_Y = 12;
    /** Stream input; may be the same buffer as {@link #POST_ARG_XOUT} with equal offsets. */
    public static final int POST_ARG_X = 13;
    public static final int POST_ARG_POST = 14;
    public static final int POST_ARG_COMB = 15;
    public static final int POST_ARG_XOUT = 16;
    public static final int POST_ARG_COUNT = 17;

    // ---- LLM_ResStack_Score: 7 scalars, 4 buffers ----
    /** Re-set per step. */
    public static final int SCORE_ARG_N_TOKENS = 0;
    public static final int SCORE_ARG_N_EMBD = 1;
    /** Re-set per layer: the number of banked checkpoints grows through the forward pass. */
    public static final int SCORE_ARG_N_CKPT = 2;
    public static final int SCORE_ARG_EPS = 3;
    public static final int SCORE_ARG_STACK_OFF = 4;
    public static final int SCORE_ARG_CUR_OFF = 5;
    public static final int SCORE_ARG_PROBS_OFF = 6;
    public static final int SCORE_ARG_STACK = 7;
    public static final int SCORE_ARG_CUR = 8;
    public static final int SCORE_ARG_SCORE_W = 9;
    public static final int SCORE_ARG_PROBS = 10;
    public static final int SCORE_ARG_COUNT = 11;

    // ---- LLM_ResStack_Mix: 7 scalars, 4 buffers ----
    /** Re-set per step. */
    public static final int MIX_ARG_N_TOKENS = 0;
    public static final int MIX_ARG_N_EMBD = 1;
    /** Re-set per layer. */
    public static final int MIX_ARG_N_CKPT = 2;
    public static final int MIX_ARG_STACK_OFF = 3;
    public static final int MIX_ARG_CUR_OFF = 4;
    public static final int MIX_ARG_PROBS_OFF = 5;
    public static final int MIX_ARG_OUT_OFF = 6;
    public static final int MIX_ARG_STACK = 7;
    /** Current stream; may be the same buffer as {@link #MIX_ARG_OUT} with equal offsets. */
    public static final int MIX_ARG_CUR = 8;
    public static final int MIX_ARG_PROBS = 9;
    public static final int MIX_ARG_OUT = 10;
    public static final int MIX_ARG_COUNT = 11;

    /** {@code n_out}: pre only (HC head). */
    public static final int OUT_PRE = 1;
    /** {@code n_out}: pre and post (iHC). */
    public static final int OUT_PRE_POST = 2;
    /** {@code n_out}: pre, post and Sinkhorn comb (mHC). */
    public static final int OUT_PRE_POST_COMB = 3;

    /** {@code LLM_HC_Pre mode}: one weight per (token, stream), {@code w[t*hc + h]}. */
    public static final int PRE_MODE_WEIGHTS = 0;
    /** {@code LLM_HC_Pre mode}: per-element gate, {@code sigmoid(g[(t*hc + h)*n_embd + i])}. */
    public static final int PRE_MODE_GATED = 1;
    /** {@code LLM_HC_Pre mode}: unweighted sum (stream mean with {@code gscale = 1/hc}); the gate buffer is unused. */
    public static final int PRE_MODE_MEAN = 2;

    /** {@code LLM_HC_Post post_act}: post coefficients are used as given. */
    public static final int POST_ACT_NONE = 0;
    /** {@code LLM_HC_Post post_act}: {@code p = post_mag * sigmoid(post_in_scale * post)}. */
    public static final int POST_ACT_SIGMOID = 1;

    /** Largest stream count of every hc kernel ({@code NF_HC_MAX}). */
    public static final int MAX_HC = 8;
    /** mHC post coefficient magnitude: {@code post = 2 * sigmoid(...)} (deepseek4.cpp:386-389). */
    public static final float MHC_POST_MAGNITUDE = 2.0f;
    /** qwen4exp combine weight magnitude: {@code 2 * sigmoid(inject / hc)} (qwen4exp.cpp:376-378). */
    public static final float QWEN4EXP_INJECT_MAGNITUDE = 2.0f;

    /** Default local size of the per-token and per-element kernels. */
    public static final int DEFAULT_ELEMENT_LOCAL = 256;
    /** Smallest local size of {@code LLM_ResStack_Score} (one full warp / subgroup). */
    public static final int MIN_SCORE_LOCAL = 32;

    private HcKernels() {
    }

    /**
     * Width of one token's mixes row: {@code (2 + hc) * hc} for {@link #OUT_PRE_POST_COMB}, {@code nOut * hc}
     * otherwise (the row count of the {@code hc_*_fn} projection).
     */
    public static int mixDim(int hc, int nOut) {
        return nOut == OUT_PRE_POST_COMB ? (2 + hc) * hc : nOut * hc;
    }

    /** Values per token of the hyper-connection residual stream: {@code hc * nEmbd}. */
    public static int streamWidth(int hc, int nEmbd) {
        return hc * nEmbd;
    }

    /**
     * Number of checkpoints the attention-residual stack holds after all {@code nLayer} layers: layers
     * {@code 0, blockSize, 2*blockSize, ...} bank their input (kimi-k3.cpp:288), i.e. {@code ceil(nLayer / blockSize)}.
     *
     * @throws IllegalArgumentException when {@code nLayer < 0} or {@code blockSize <= 0}
     */
    public static int checkpointCount(int nLayer, int blockSize) {
        if (nLayer < 0 || blockSize <= 0) {
            throw new IllegalArgumentException("attention-residual stack needs nLayer >= 0 and a positive block size, got "
                    + nLayer + " / " + blockSize);
        }
        return (nLayer + blockSize - 1) / blockSize;
    }

    /** True when layer {@code il} banks its raw input on the attention-residual stack ({@code il % blockSize == 0}). */
    public static boolean isCheckpointLayer(int il, int blockSize) {
        return blockSize > 0 && il % blockSize == 0;
    }

    /** Floats of a checkpoint stack holding {@code nCkpt} checkpoints of {@code nTokens} rows of width {@code nEmbd}. */
    public static long stackFloats(int nCkpt, int nTokens, int nEmbd) {
        return (long) nCkpt * nTokens * nEmbd;
    }

    /** Element offset of checkpoint {@code c} in a stack of {@code nTokens} rows of width {@code nEmbd}. */
    public static long checkpointOffset(int c, int nTokens, int nEmbd) {
        return (long) c * nTokens * nEmbd;
    }

    /** Global size of {@code LLM_HC_PrePostComb}: one work-item per token, rounded up to {@code local}. */
    public static long perTokenGlobal(int nTokens, int local) {
        return roundUp(nTokens, local);
    }

    /** Global size of {@code LLM_HC_Pre}, {@code LLM_HC_Post} and {@code LLM_ResStack_Mix}: one work-item per (token, i). */
    public static long elementGlobal(int nTokens, int nEmbd, int local) {
        return roundUp((long) nTokens * nEmbd, local);
    }

    /**
     * Local size of {@code LLM_ResStack_Score}: the largest power of two in [{@value #MIN_SCORE_LOCAL},
     * {@code LLM_DIMX}] that does not exceed {@code preferred} ({@value #MIN_SCORE_LOCAL} when it is smaller).
     */
    public static int scoreLocal(int preferred) {
        int local = MIN_SCORE_LOCAL;
        while (local * 2 <= Math.min(preferred, LlmKernelConstants.LLM_DIMX)) {
            local *= 2;
        }
        return local;
    }

    /** Global size of {@code LLM_ResStack_Score}: one workgroup per token. */
    public static long scoreGlobal(int nTokens, int local) {
        return (long) nTokens * local;
    }

    /**
     * Checks a stream count.
     *
     * @throws IllegalArgumentException when {@code hc} is outside {@code [1, MAX_HC]}
     */
    public static void validateStreams(int hc) {
        if (hc < 1 || hc > MAX_HC) {
            throw new IllegalArgumentException("hyper-connection stream count " + hc + " is outside the supported range [1, "
                    + MAX_HC + "]");
        }
    }

    /**
     * Checks the attention-residual stack shape.
     *
     * @throws IllegalArgumentException naming the violated rule
     */
    public static void validateResStack(int nEmbd, int nCkpt) {
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("attention-residual rows need a positive width, got " + nEmbd);
        }
        if (nCkpt < 0) {
            throw new IllegalArgumentException("attention-residual checkpoint count must be non-negative, got " + nCkpt);
        }
    }

    private static long roundUp(long n, int local) {
        if (local <= 0) {
            throw new IllegalArgumentException("local size must be positive, got " + local);
        }
        return (n + local - 1) / local * local;
    }

    /** Immutable scalar arguments of {@code LLM_HC_PrePostComb}. */
    public static final class CoefficientParams {
        private final int hc;
        private final int nOut;
        private final float eps;
        private final int iters;
        private final float postMag;
        private final float postEps;

        private CoefficientParams(int hc, int nOut, float eps, int iters, float postMag, float postEps) {
            validateStreams(hc);
            if (nOut < OUT_PRE || nOut > OUT_PRE_POST_COMB) {
                throw new IllegalArgumentException("hyper-connection n_out must be 1, 2 or 3, got " + nOut);
            }
            if (nOut == OUT_PRE_POST_COMB && iters < 1) {
                throw new IllegalArgumentException("Sinkhorn needs at least one iteration, got " + iters);
            }
            if (!Float.isFinite(eps) || !Float.isFinite(postMag) || !Float.isFinite(postEps)) {
                throw new IllegalArgumentException("hyper-connection eps / post magnitude must be finite: eps=" + eps
                        + " post_mag=" + postMag + " post_eps=" + postEps);
            }
            this.hc = hc;
            this.nOut = nOut;
            this.eps = eps;
            this.iters = nOut == OUT_PRE_POST_COMB ? iters : 0;
            this.postMag = postMag;
            this.postEps = postEps;
        }

        /** mHC sub-block (deepseek4, glm5-next, dflash): pre, {@code post = 2*sigmoid(.)} and the Sinkhorn comb. */
        public static CoefficientParams mhc(int hc, float eps, int sinkhornIters) {
            return new CoefficientParams(hc, OUT_PRE_POST_COMB, eps, sinkhornIters, MHC_POST_MAGNITUDE, 0.0f);
        }

        /** iHC sub-block (hy_v4): pre and {@code post = magnitude*sigmoid(.) + eps}, no comb. */
        public static CoefficientParams ihc(int hc, float eps, float magnitude) {
            return new CoefficientParams(hc, OUT_PRE_POST, eps, 0, magnitude, eps);
        }

        /** HC head (deepseek4, dflash, hy_v4): pre only, from {@code hc_head_fn/scale/base}. */
        public static CoefficientParams head(int hc, float eps) {
            return new CoefficientParams(hc, OUT_PRE, eps, 0, 0.0f, 0.0f);
        }

        public int hc() { return hc; }

        /** {@link #OUT_PRE}, {@link #OUT_PRE_POST} or {@link #OUT_PRE_POST_COMB}. */
        public int nOut() { return nOut; }

        public float eps() { return eps; }

        /** Sinkhorn iterations; 0 unless {@link #nOut()} is {@link #OUT_PRE_POST_COMB}. */
        public int iters() { return iters; }

        public float postMag() { return postMag; }

        public float postEps() { return postEps; }

        /** Width of one token's mixes row ({@link HcKernels#mixDim}). */
        public int mixDim() { return HcKernels.mixDim(hc, nOut); }

        /** Elements of {@code scale}: one per produced coefficient group. */
        public int scaleCount() { return nOut; }

        /** Elements of {@code base}: equal to {@link #mixDim()}. */
        public int baseCount() { return mixDim(); }

        @Override
        public String toString() {
            return "HcCoefficients{hc=" + hc + ", nOut=" + nOut + ", eps=" + eps + ", iters=" + iters
                    + ", postMag=" + postMag + ", postEps=" + postEps + '}';
        }
    }

    /** Immutable scalar arguments of {@code LLM_HC_Pre}. */
    public static final class PreParams {
        private static final PreParams WEIGHTS = new PreParams(PRE_MODE_WEIGHTS, 1.0f);

        private final int mode;
        private final float gscale;

        private PreParams(int mode, float gscale) {
            this.mode = mode;
            this.gscale = gscale;
        }

        /** {@code out = sum_h X[h] * w[t*hc + h]} (ggml_dsv4_hc_pre). */
        public static PreParams weights() {
            return WEIGHTS;
        }

        /** {@code out = (1/hc) * sum_h X[h] * sigmoid(g[h])} (qwen4exp ggml_dsv4_hc_pre_gated). */
        public static PreParams gated(int hc) {
            validateStreams(hc);
            return new PreParams(PRE_MODE_GATED, 1.0f / hc);
        }

        /** {@code out = (1/hc) * sum_h X[h]} (build_hc_mean). */
        public static PreParams mean(int hc) {
            validateStreams(hc);
            return new PreParams(PRE_MODE_MEAN, 1.0f / hc);
        }

        /** {@link #PRE_MODE_WEIGHTS}, {@link #PRE_MODE_GATED} or {@link #PRE_MODE_MEAN}. */
        public int mode() { return mode; }

        public float gscale() { return gscale; }

        @Override
        public String toString() {
            return "HcPre{mode=" + mode + ", gscale=" + gscale + '}';
        }
    }

    /** Immutable scalar arguments of {@code LLM_HC_Post}. */
    public static final class PostParams {
        private static final PostParams MIXING = new PostParams(true, POST_ACT_NONE, 1.0f, 1.0f);
        private static final PostParams IDENTITY = new PostParams(false, POST_ACT_NONE, 1.0f, 1.0f);

        private final boolean hasComb;
        private final int postAct;
        private final float postInScale;
        private final float postMag;

        private PostParams(boolean hasComb, int postAct, float postInScale, float postMag) {
            this.hasComb = hasComb;
            this.postAct = postAct;
            this.postInScale = postInScale;
            this.postMag = postMag;
        }

        /** mHC: {@code Xout[dst] = y*post[dst] + sum_src X[src]*comb[dst + hc*src]}. */
        public static PostParams mixing() {
            return MIXING;
        }

        /** iHC: {@code Xout[dst] = y*post[dst] + X[dst]}. */
        public static PostParams identity() {
            return IDENTITY;
        }

        /** qwen4exp combine: {@code Xout[dst] = y*2*sigmoid(inject[dst]/hc) + X[dst]}. */
        public static PostParams gatedInject(int hc) {
            validateStreams(hc);
            return new PostParams(false, POST_ACT_SIGMOID, 1.0f / hc, QWEN4EXP_INJECT_MAGNITUDE);
        }

        public boolean hasComb() { return hasComb; }

        /** {@link #POST_ACT_NONE} or {@link #POST_ACT_SIGMOID}. */
        public int postAct() { return postAct; }

        public float postInScale() { return postInScale; }

        public float postMag() { return postMag; }

        /** Effective post coefficient for raw value {@code raw}. */
        public float coefficient(float raw) {
            return postAct == POST_ACT_SIGMOID ? postMag * ActMath.sigmoid(postInScale * raw) : raw;
        }

        @Override
        public String toString() {
            return "HcPost{comb=" + hasComb + ", act=" + postAct + ", inScale=" + postInScale + ", mag=" + postMag + '}';
        }
    }
}
