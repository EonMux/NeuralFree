package com.eonmux.neuralfree.gguf.quant;

import java.util.Objects;

/**
 * The fused projections of vLLM's online quantization, by GGUF tensor name: the planned tensors vLLM loads into one
 * fused weight and so quantizes with one set of tensor-scoped scales ({@link RequantPolicy#quantizeFused}), in
 * fusion order:
 * <ul>
 *   <li>{@code attn_q}, {@code attn_k}, {@code attn_v}: {@code QKVParallelLinear} ({@code qkv_proj});</li>
 *   <li>{@code ffn_gate}, {@code ffn_up}: {@code MergedColumnParallelLinear} ({@code gate_up_proj});</li>
 *   <li>{@code ffn_gate_shexp}, {@code ffn_up_shexp}: the shared expert's {@code gate_up_proj};</li>
 *   <li>{@code ffn_gate_exps}, {@code ffn_up_exps}: the routed experts' {@code w13}, per expert
 *       ({@link Layout#EXPERT_SHARDS});</li>
 *   <li>{@code attn_q_a}, {@code attn_kv_a_mqa}: MLA's {@code fused_qkv_a_proj} (deepseek_v2.py:1052-1058);</li>
 *   <li>{@code attn_qkv}, {@code attn_gate}: the gated-delta-net {@code in_proj_qkvz} of Qwen3.5 and Qwen3-Next
 *       (qwen_gdn_linear_attn.py:841-865);</li>
 *   <li>{@code ssm_beta}, {@code ssm_alpha}: Qwen3.5's {@code in_proj_ba} (qwen_gdn_linear_attn.py:867-889);</li>
 *   <li>{@code attn_k_b}, {@code attn_v_b}: MLA's {@code kv_b_proj}, which the converter splits per head into a
 *       transposed key part and a value part ({@link Layout#MLA_KV_B}).</li>
 * </ul>
 * A group is the members under one name prefix ({@code blk.3.}) that come from one vLLM layer; the down
 * projections are never fused.
 */
public final class FusedProjections {

    /** How a group's members lie in the weight vLLM quantizes. */
    public enum Layout {
        /** The members' rows, concatenated in fusion order. */
        ROWS,
        /** Per expert, each member's rows padded to a multiple of 128 rows, concatenated in fusion order. */
        EXPERT_SHARDS,
        /**
         * {@code kv_b [heads * (nope + v), kv_lora]}: {@code attn_k_b [heads, kv_lora, nope]} holds each head's
         * key rows transposed, {@code attn_v_b [heads, v, kv_lora]} its value rows.
         */
        MLA_KV_B
    }

    /** Each group: its name, then its members' GGUF roles in fusion order. */
    private static final String[][] GROUPS = {
        {"qkv", "attn_q", "attn_k", "attn_v"},
        {"gate_up", "ffn_gate", "ffn_up"},
        {"gate_up_shexp", "ffn_gate_shexp", "ffn_up_shexp"},
        {"w13", "ffn_gate_exps", "ffn_up_exps"},
        {"fused_qkv_a", "attn_q_a", "attn_kv_a_mqa"},
        {"in_proj_qkvz", "attn_qkv", "attn_gate"},
        {"in_proj_ba", "ssm_beta", "ssm_alpha"},
        {"kv_b", "attn_k_b", "attn_v_b"},
    };

    private static final String WEIGHT_SUFFIX = ".weight";

    private FusedProjections() {
    }

    /** The group of {@code ggufName}, its name prefix followed by the group name ({@code blk.3.qkv}); null for none. */
    public static String groupOf(String ggufName) {
        int[] at = locate(ggufName);
        return at == null ? null : prefix(ggufName, at) + GROUPS[at[0]][0];
    }

    /** The position of {@code ggufName} in its group's fusion order; -1 when it is in no group. */
    public static int ordinalOf(String ggufName) {
        int[] at = locate(ggufName);
        return at == null ? -1 : at[1] - 1;
    }

    /** The layout of the group of {@code ggufName}; {@link Layout#ROWS} for a tensor in no group. */
    public static Layout layoutOf(String ggufName) {
        int[] at = locate(ggufName);
        if (at == null) {
            return Layout.ROWS;
        }
        switch (GROUPS[at[0]][0]) {
            case "w13":
                return Layout.EXPERT_SHARDS;
            case "kv_b":
                return Layout.MLA_KV_B;
            default:
                return Layout.ROWS;
        }
    }

    /**
     * Whether {@code ggufName} stores vLLM's weight transposed ({@code attn_k_b}): its rows run down the columns of
     * the weight vLLM quantizes, so only tensor-scoped scales that are constant along each stored block carry over.
     */
    public static boolean transposed(String ggufName) {
        int[] at = locate(ggufName);
        return at != null && GROUPS[at[0]][at[1]].equals("attn_k_b");
    }

    /** The number of members of the group of {@code ggufName}; 0 when it is in no group. */
    public static int sizeOf(String ggufName) {
        int[] at = locate(ggufName);
        return at == null ? 0 : GROUPS[at[0]].length - 1;
    }

    /** {group index, member index} of {@code ggufName}, or null. */
    private static int[] locate(String ggufName) {
        Objects.requireNonNull(ggufName, "ggufName");
        if (!ggufName.endsWith(WEIGHT_SUFFIX)) {
            return null;
        }
        String module = ggufName.substring(0, ggufName.length() - WEIGHT_SUFFIX.length());
        String role = module.substring(module.lastIndexOf('.') + 1);
        for (int g = 0; g < GROUPS.length; g++) {
            for (int m = 1; m < GROUPS[g].length; m++) {
                if (GROUPS[g][m].equals(role)) {
                    return new int[]{g, m};
                }
            }
        }
        return null;
    }

    private static String prefix(String ggufName, int[] at) {
        String role = GROUPS[at[0]][at[1]];
        return ggufName.substring(0, ggufName.length() - WEIGHT_SUFFIX.length() - role.length());
    }
}
