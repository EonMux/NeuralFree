package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A small nemotron_h_moe model with one NextN / MTP block after its trunk, for tests of NextN specs and MTP draft
 * files. Trunk block {@code il} is a Mamba-2 block for {@code il % 3 == 0}, an attention block for {@code 1} and a MoE
 * FFN block for {@code 2} (nemotron-h.cpp:226-234); the NextN block at {@code il = trunk} runs attention and the
 * routed experts with the shared expert. The width, trunk depth and vocabulary vary so that files of other models of
 * the family can be built.
 */
public final class NemotronNextnFixture {

    public static final String ARCH = "nemotron_h_moe";
    public static final int N_EMBD = 64;
    public static final int N_HEAD = 4;
    public static final int N_HEAD_KV = 2;
    public static final int N_VOCAB = 32;
    public static final int TRUNK = 3;

    private static final int N_EXPERT = 4;
    private static final int N_FF_EXP = 32;
    private static final int N_FF_SHEXP = 32;
    private static final int SSM_CONV = 4;
    private static final int SSM_INNER = 64;
    private static final int SSM_STATE = 16;
    private static final int SSM_HEADS = 8;
    private static final int SSM_GROUPS = 2;

    private NemotronNextnFixture() {
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    /**
     * The metadata of the default model: width {@link #N_EMBD}, {@link #TRUNK} trunk blocks, {@link #N_VOCAB}
     * tokens.
     */
    public static Map<String, Object> metadata() {
        return metadata(N_EMBD, TRUNK, N_VOCAB);
    }

    /**
     * The metadata of a model of width {@code nEmbd} with {@code trunk} trunk blocks, one NextN block and a
     * vocabulary of {@code nVocab} tokens ({@code vocab_size}; no token list).
     */
    public static Map<String, Object> metadata(int nEmbd, int trunk, int nVocab) {
        int blocks = trunk + 1;
        int[] ff = new int[blocks];
        int[] headKv = new int[blocks];
        for (int il = 0; il < trunk; il++) {
            ff[il] = il % 3 == 2 ? N_FF_EXP : 0;
            headKv[il] = il % 3 == 1 ? N_HEAD_KV : 0;
        }
        ff[trunk] = N_FF_EXP;
        headKv[trunk] = N_HEAD_KV;
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", ARCH);
        m.put(ARCH + ".context_length", 4096);
        m.put(ARCH + ".embedding_length", nEmbd);
        m.put(ARCH + ".vocab_size", nVocab);
        m.put(ARCH + ".block_count", blocks);
        m.put(ARCH + ".nextn_predict_layers", 1);
        m.put(ARCH + ".feed_forward_length", ff);
        m.put(ARCH + ".attention.head_count", N_HEAD);
        m.put(ARCH + ".attention.head_count_kv", headKv);
        m.put(ARCH + ".attention.layer_norm_epsilon", 1e-5f);
        m.put(ARCH + ".attention.layer_norm_rms_epsilon", 1e-5f);
        m.put(ARCH + ".ssm.conv_kernel", SSM_CONV);
        m.put(ARCH + ".ssm.inner_size", SSM_INNER);
        m.put(ARCH + ".ssm.state_size", SSM_STATE);
        m.put(ARCH + ".ssm.time_step_rank", SSM_HEADS);
        m.put(ARCH + ".ssm.group_count", SSM_GROUPS);
        m.put(ARCH + ".expert_count", N_EXPERT);
        m.put(ARCH + ".expert_used_count", 2);
        m.put(ARCH + ".expert_feed_forward_length", N_FF_EXP);
        m.put(ARCH + ".expert_shared_feed_forward_length", N_FF_SHEXP);
        m.put(ARCH + ".expert_weights_norm", true);
        return m;
    }

    /** The tensors of the default model. */
    public static List<GGUFTensorInfo> tensors() {
        return tensors(N_EMBD, TRUNK, N_VOCAB);
    }

    /** The tensors of the model of {@link #metadata(int, int, int)}. */
    public static List<GGUFTensorInfo> tensors(int nEmbd, int trunk, int nVocab) {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", nEmbd, nVocab));
        ts.add(t("output_norm.weight", nEmbd));
        for (int il = 0; il <= trunk; il++) {
            ts.add(t("blk." + il + ".attn_norm.weight", nEmbd));
        }
        for (int il = 0; il < trunk; il++) {
            switch (il % 3) {
                case 0:
                    mamba(ts, il, nEmbd);
                    break;
                case 1:
                    attention(ts, il, nEmbd);
                    break;
                default:
                    experts(ts, il, nEmbd);
                    break;
            }
        }
        String p = "blk." + trunk + ".";
        ts.add(t(p + "nextn.eh_proj.weight", 2L * nEmbd, nEmbd));
        ts.add(t(p + "nextn.enorm.weight", nEmbd));
        ts.add(t(p + "nextn.hnorm.weight", nEmbd));
        ts.add(t(p + "nextn.shared_head_norm.weight", nEmbd));
        attention(ts, trunk, nEmbd);
        ts.add(t(p + "post_attention_norm.weight", nEmbd));
        experts(ts, trunk, nEmbd);
        return ts;
    }

    /**
     * What {@code convert_hf_to_gguf.py --mtp} keeps of {@code full}, a model with {@code trunk} trunk blocks: the
     * embedding, the output norm and the NextN block.
     */
    public static List<GGUFTensorInfo> mtpOnly(List<GGUFTensorInfo> full, int trunk) {
        String nextn = "blk." + trunk + ".";
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (GGUFTensorInfo info : full) {
            if (!info.name().startsWith("blk.") || info.name().startsWith(nextn)) {
                ts.add(info);
            }
        }
        return ts;
    }

    /** {@code full} without its NextN block at {@code il = trunk}. */
    public static List<GGUFTensorInfo> withoutNextn(List<GGUFTensorInfo> full, int trunk) {
        String nextn = "blk." + trunk + ".";
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (GGUFTensorInfo info : full) {
            if (!info.name().startsWith(nextn)) {
                ts.add(info);
            }
        }
        return ts;
    }

    private static void mamba(List<GGUFTensorInfo> ts, int il, int nEmbd) {
        long convDim = SSM_INNER + 2L * SSM_GROUPS * SSM_STATE;
        long inProj = SSM_INNER + convDim + SSM_HEADS;
        String p = "blk." + il + ".";
        ts.add(t(p + "ssm_in.weight", nEmbd, inProj));
        ts.add(t(p + "ssm_conv1d.weight", SSM_CONV, convDim));
        ts.add(t(p + "ssm_dt.bias", SSM_HEADS));
        ts.add(t(p + "ssm_a", 1, SSM_HEADS));
        ts.add(t(p + "ssm_d", 1, SSM_HEADS));
        ts.add(t(p + "ssm_norm.weight", SSM_INNER / SSM_GROUPS, SSM_GROUPS));
        ts.add(t(p + "ssm_out.weight", SSM_INNER, nEmbd));
    }

    private static void attention(List<GGUFTensorInfo> ts, int il, int nEmbd) {
        long headDim = nEmbd / N_HEAD;
        String p = "blk." + il + ".";
        ts.add(t(p + "attn_q.weight", nEmbd, N_HEAD * headDim));
        ts.add(t(p + "attn_k.weight", nEmbd, N_HEAD_KV * headDim));
        ts.add(t(p + "attn_v.weight", nEmbd, N_HEAD_KV * headDim));
        ts.add(t(p + "attn_output.weight", N_HEAD * headDim, nEmbd));
    }

    private static void experts(List<GGUFTensorInfo> ts, int il, int nEmbd) {
        String p = "blk." + il + ".";
        ts.add(t(p + "ffn_gate_inp.weight", nEmbd, N_EXPERT));
        ts.add(t(p + "exp_probs_b.bias", N_EXPERT));
        ts.add(t(p + "ffn_up_exps.weight", nEmbd, N_FF_EXP, N_EXPERT));
        ts.add(t(p + "ffn_down_exps.weight", N_FF_EXP, nEmbd, N_EXPERT));
        ts.add(t(p + "ffn_up_shexp.weight", nEmbd, N_FF_SHEXP));
        ts.add(t(p + "ffn_down_shexp.weight", N_FF_SHEXP, nEmbd));
    }
}
