package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport.ExpertCollector;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport.MergedName;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * The hooks of {@code MiniMaxM2Model} (LC/conversion/minimax.py:124-161): the expert feed-forward length and partial
 * RoPE dimension count, the per-layer stacking of the {@code block_sparse_moe} experts, and the GPT-2 BPE vocabulary.
 */
public final class HfMiniMaxHooks {

    /** The merged expert parts, in the order the converter writes them (minimax.py:142). */
    private static final List<String> EXPERT_PARTS = List.of("w1", "w2", "w3");

    private static final String EXPERTS_STATE = "minimax-m2.experts";

    private HfMiniMaxHooks() {
    }

    /**
     * {@code MiniMaxM2Model.set_gguf_parameters} (minimax.py:128-132): the generic keys, then the expert feed-forward
     * length ({@code intermediate_size}) and the RoPE dimension count ({@code rotary_dim}).
     *
     * @throws GgufFormatException when {@code intermediate_size} or {@code rotary_dim} is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("rotary_dim"));
    }

    /**
     * {@code MiniMaxM2Model.modify_tensors} (minimax.py:134-161): the {@code block_sparse_moe.experts.<x>.w1|w2|w3}
     * tensors of a layer are stacked into one tensor per part, mapped under the merged name
     * {@code model.layers.<b>.block_sparse_moe.experts.<part>.weight}; every other tensor maps directly.
     *
     * @throws GgufFormatException when the expert count is missing, an expert tensor is not a member of a complete
     *                             expert set, or a name does not map
     * @throws IOException         as the default mapping does
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("block_sparse_moe.experts.")) {
            ExpertCollector experts = ctx.state(EXPERTS_STATE, ExpertCollector.class, () -> new ExpertCollector(
                    ctx.config().requireInt("num_local_experts", "num_experts"), EXPERT_PARTS,
                    MergedName.SAME_PREFIX));
            if (!experts.offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not an expert member"
                        + " <prefix>.experts.<x>.<part> with a part in " + EXPERT_PARTS);
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * {@code TextModel.set_vocab} ({@code _set_vocab_gpt2}, base.py:2156-2165).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }
}
