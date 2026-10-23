package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

/**
 * Names, argument indices, mode codes and dispatch geometry of the gated DeltaNet, KDA and lightning attention
 * kernels ({@code kernels/llm/recurrent/gdn.cl}, {@code recurrent/lightning.cl} and their GLSL twins
 * {@code gdn_scan.comp}, {@code gdn_gate.comp}, {@code lightning.comp}). Argument lists match the kernel signatures;
 * every index constant below is the position of that argument in the kernel signature.
 */
public final class DeltaNetKernels {

    /** Gated delta rule over consecutive tokens (GDN scalar gate or KDA per-key gate). */
    public static final String GDN_SCAN = "LLM_GDN_Scan";
    /** Decay gate {@code g} (log space) and update rate {@code beta} of GDN and KDA layers. */
    public static final String GDN_GATE_PREP = "LLM_GDN_GatePrep";
    /** MiniMax-01 lightning attention recurrence. */
    public static final String LIGHTNING_ATTN = "LLM_LightningAttn";

    // ---- LLM_GDN_Scan: 16 scalars, 8 buffers ----
    public static final int SCAN_ARG_S_K = 0;
    public static final int SCAN_ARG_S_V = 1;
    public static final int SCAN_ARG_N_V_HEADS = 2;
    public static final int SCAN_ARG_N_K_HEADS = 3;
    /** Re-set per step. */
    public static final int SCAN_ARG_N_TOKENS = 4;
    public static final int SCAN_ARG_KV_HEAD_MODE = 5;
    public static final int SCAN_ARG_GATE_MODE = 6;
    public static final int SCAN_ARG_Q_STRIDE = 7;
    public static final int SCAN_ARG_Q_OFF = 8;
    public static final int SCAN_ARG_K_STRIDE = 9;
    public static final int SCAN_ARG_K_OFF = 10;
    public static final int SCAN_ARG_V_STRIDE = 11;
    public static final int SCAN_ARG_V_OFF = 12;
    public static final int SCAN_ARG_O_STRIDE = 13;
    public static final int SCAN_ARG_O_OFF = 14;
    public static final int SCAN_ARG_SCALE = 15;
    public static final int SCAN_ARG_Q = 16;
    public static final int SCAN_ARG_K = 17;
    public static final int SCAN_ARG_V = 18;
    public static final int SCAN_ARG_G = 19;
    public static final int SCAN_ARG_BETA = 20;
    public static final int SCAN_ARG_STATE = 21;
    public static final int SCAN_ARG_STATE_ROWS = 22;
    public static final int SCAN_ARG_OUT = 23;
    public static final int SCAN_ARG_COUNT = 24;

    // ---- LLM_GDN_GatePrep: 11 scalars, 6 buffers ----
    /** Re-set per step. */
    public static final int GATE_ARG_N_TOKENS = 0;
    public static final int GATE_ARG_N_V_HEADS = 1;
    public static final int GATE_ARG_N_K_HEADS = 2;
    public static final int GATE_ARG_LAYOUT = 3;
    public static final int GATE_ARG_KDA_MODE = 4;
    public static final int GATE_ARG_HEAD_DIM = 5;
    public static final int GATE_ARG_LB = 6;
    public static final int GATE_ARG_A_STRIDE = 7;
    public static final int GATE_ARG_A_OFF = 8;
    public static final int GATE_ARG_B_STRIDE = 9;
    public static final int GATE_ARG_B_OFF = 10;
    public static final int GATE_ARG_A_IN = 11;
    public static final int GATE_ARG_B_IN = 12;
    public static final int GATE_ARG_DT_BIAS = 13;
    public static final int GATE_ARG_SSM_A = 14;
    public static final int GATE_ARG_G_OUT = 15;
    public static final int GATE_ARG_BETA_OUT = 16;
    public static final int GATE_ARG_COUNT = 17;

    // ---- LLM_LightningAttn: 4 scalars, 5 buffers ----
    /** Re-set per step. */
    public static final int LIGHTNING_ARG_N_TOKENS = 0;
    public static final int LIGHTNING_ARG_N_HEAD = 1;
    public static final int LIGHTNING_ARG_D = 2;
    public static final int LIGHTNING_ARG_LAYER_SCALE = 3;
    public static final int LIGHTNING_ARG_SLOPES = 4;
    public static final int LIGHTNING_ARG_QKV = 5;
    public static final int LIGHTNING_ARG_STATE = 6;
    public static final int LIGHTNING_ARG_STATE_ROWS = 7;
    public static final int LIGHTNING_ARG_OUT = 8;
    public static final int LIGHTNING_ARG_COUNT = 9;

    /** {@code gate_mode}: one decay value per (token, value head) (gated DeltaNet). */
    public static final int GATE_MODE_PER_HEAD = 0;
    /** {@code gate_mode}: one decay value per (token, value head, key dim) (KDA). */
    public static final int GATE_MODE_PER_KEY = 1;

    /** {@code layout}: separate alpha/f and beta projections. */
    public static final int LAYOUT_SEPARATE = 0;
    /** {@code layout}: qwen3next fused {@code ssm_ba}, per key head {@code [b(R) | a(R)]} with R = n_v / n_k. */
    public static final int LAYOUT_FUSED_BA = 1;

    /** Largest key head dim of {@code LLM_GDN_Scan} ({@code NF_GDN_MAX_SK}). */
    public static final int GDN_MAX_KEY_DIM = 256;
    /** Largest head dim of {@code LLM_LightningAttn} ({@code NF_LA_MAX_D}). */
    public static final int LIGHTNING_MAX_HEAD_DIM = 256;
    /** Work-items cooperating on one value column of a recurrence workgroup. */
    public static final int LANES_PER_COLUMN = 32;
    /** Default local size of the scan and lightning kernels: four value columns per workgroup. */
    public static final int DEFAULT_RECURRENCE_LOCAL = 128;

    private DeltaNetKernels() {
    }

    /**
     * Local size of {@code LLM_GDN_Scan} / {@code LLM_LightningAttn}: the largest power of two in
     * [{@value #LANES_PER_COLUMN}, {@code LLM_DIMX}] that does not exceed {@code preferred} (32 when it is
     * smaller). Every such size is a multiple of 32, as the 32-lane column reduction and the subgroup reductions
     * require.
     */
    public static int recurrenceLocal(int preferred) {
        int local = LANES_PER_COLUMN;
        while (local * 2 <= Math.min(preferred, LlmKernelConstants.LLM_DIMX)) {
            local *= 2;
        }
        return local;
    }

    /**
     * Global size of {@code LLM_GDN_Scan} ({@code columns} = S_v) or {@code LLM_LightningAttn} ({@code columns} = D):
     * {@code heads * ceil(columns / (local / 32)) * local}.
     */
    public static long recurrenceGlobal(int heads, int columns, int local) {
        if (local < LANES_PER_COLUMN || local % LANES_PER_COLUMN != 0 || local > LlmKernelConstants.LLM_DIMX) {
            throw new IllegalArgumentException("recurrence local size must be a multiple of " + LANES_PER_COLUMN
                    + " no larger than " + LlmKernelConstants.LLM_DIMX + " (the kernels' local arrays), got " + local);
        }
        if (heads <= 0 || columns <= 0) {
            throw new IllegalArgumentException("recurrence dispatch needs positive heads and columns: heads=" + heads
                    + " columns=" + columns);
        }
        int cols = local / LANES_PER_COLUMN;
        long blocks = (columns + cols - 1) / cols;
        return (long) heads * blocks * local;
    }

    /** Global size of {@code LLM_GDN_GatePrep}: one work-item per (token, value head, gated dim), rounded up to {@code local}. */
    public static long gatePrepGlobal(int nTokens, int nVHeads, int kdaMode, int headDim, int local) {
        if (local <= 0) {
            throw new IllegalArgumentException("gate prep local size must be positive, got " + local);
        }
        long items = (long) nTokens * nVHeads * (kdaMode < 0 ? 1 : headDim);
        return (items + local - 1) / local * local;
    }

    /**
     * Checks the {@code LLM_GDN_Scan} shape.
     *
     * @throws IllegalArgumentException naming the violated limit
     */
    public static void validateScan(int sK, int sV, int nVHeads, int nKHeads, int kvHeadMode, int gateMode) {
        if (sK <= 0 || sV <= 0) {
            throw new IllegalArgumentException("gated delta-net head dims must be positive: S_k=" + sK + " S_v=" + sV);
        }
        if (sK > GDN_MAX_KEY_DIM) {
            throw new IllegalArgumentException("gated delta-net key head dim " + sK + " exceeds the kernel limit "
                    + GDN_MAX_KEY_DIM);
        }
        if (nKHeads <= 0 || nVHeads <= 0 || nVHeads % nKHeads != 0) {
            throw new IllegalArgumentException("gated delta-net value heads (" + nVHeads
                    + ") must be a positive multiple of key heads (" + nKHeads + ")");
        }
        if (kvHeadMode != 0 && kvHeadMode != 1) {
            throw new IllegalArgumentException("kv_head_mode must be a GdnHeadMap code (0 or 1), got " + kvHeadMode);
        }
        if (gateMode != GATE_MODE_PER_HEAD && gateMode != GATE_MODE_PER_KEY) {
            throw new IllegalArgumentException("gate_mode must be 0 (per head) or 1 (per key dim), got " + gateMode);
        }
    }

    /**
     * Checks the {@code LLM_GDN_GatePrep} shape. The lower-bound modes (kda_mode 1 and 2) need a finite {@code lb}:
     * llama.cpp selects them only when {@code kda.gate_lower_bound} is set (kimi-k3.cpp:428, bailingmoe3.cpp:17-34).
     *
     * @throws IllegalArgumentException naming the violated rule
     */
    public static void validateGatePrep(int nVHeads, int nKHeads, int layout, int kdaMode, int headDim, float lb) {
        if (nKHeads <= 0 || nVHeads <= 0 || nVHeads % nKHeads != 0) {
            throw new IllegalArgumentException("gate prep value heads (" + nVHeads
                    + ") must be a positive multiple of key heads (" + nKHeads + ")");
        }
        if (kdaMode < -1 || kdaMode > 2) {
            throw new IllegalArgumentException("kda_mode must be -1 (GDN), 0, 1 or 2, got " + kdaMode);
        }
        if (layout != LAYOUT_SEPARATE && layout != LAYOUT_FUSED_BA) {
            throw new IllegalArgumentException("gate prep layout must be 0 (separate) or 1 (fused ssm_ba), got " + layout);
        }
        if (layout == LAYOUT_FUSED_BA && kdaMode >= 0) {
            throw new IllegalArgumentException("the fused ssm_ba layout exists only for gated delta-net (kda_mode -1)");
        }
        if (kdaMode >= 0 && headDim <= 0) {
            throw new IllegalArgumentException("KDA gate prep needs a positive head dim, got " + headDim);
        }
        if ((kdaMode == 1 || kdaMode == 2) && (Float.isNaN(lb) || Float.isInfinite(lb))) {
            throw new IllegalArgumentException("KDA lower-bound gate (kda_mode " + kdaMode
                    + ") needs a finite kda.gate_lower_bound, got " + lb);
        }
    }

    /**
     * Checks the {@code LLM_LightningAttn} shape.
     *
     * @throws IllegalArgumentException naming the violated limit
     */
    public static void validateLightning(int nHead, int headDim) {
        if (nHead <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("lightning attention needs positive heads and head dim: n_head=" + nHead
                    + " D=" + headDim);
        }
        if (headDim > LIGHTNING_MAX_HEAD_DIM) {
            throw new IllegalArgumentException("lightning attention head dim " + headDim + " exceeds the kernel limit "
                    + LIGHTNING_MAX_HEAD_DIM);
        }
    }
}
