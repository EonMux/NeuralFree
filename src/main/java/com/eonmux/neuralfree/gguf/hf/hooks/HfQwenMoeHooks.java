package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.StringJoiner;

/**
 * Hooks of the Qwen mixture-of-experts converter classes (LC/conversion/qwen.py): {@code Qwen2MoeModel} (:81-161),
 * the base of Qwen3-Next and Qwen3.5, and {@code Qwen3MoeModel} (:268-284). Experts arrive either as Transformers 5
 * fused tensors ({@code mlp.experts.gate_up_proj} {@code [E, 2F, D]} and {@code mlp.experts.down_proj}) or as one
 * tensor per expert, which {@link HfHookSupport.ExpertCollector} stacks per layer.
 */
public final class HfQwenMoeHooks {

    /** The per-expert parts, in the order the converter merges them (qwen.py:139). */
    static final List<String> EXPERT_PARTS =
            Collections.unmodifiableList(Arrays.asList("down_proj", "gate_proj", "up_proj"));

    private static final String FUSED_DOWN = "mlp.experts.down_proj";
    private static final String FUSED_GATE_UP = "mlp.experts.gate_up_proj";
    private static final String WEIGHT = ".weight";
    private static final String EXPERTS_STATE = HfQwenMoeHooks.class.getName() + ".experts";

    private HfQwenMoeHooks() {
    }

    // ------------------------------------------------------------------ Qwen2MoeModel

    /**
     * {@code TextModel.set_vocab} (LC/conversion/base.py:1575-1576), which {@code Qwen2MoeModel} and its Qwen3-Next
     * and Qwen3.5 subclasses inherit: GPT-2 BPE.
     */
    public static Map<String, Object> qwen2MoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code Qwen2MoeModel.set_gguf_parameters} (qwen.py:86-93): the generic metadata, then
     * {@code expert_feed_forward_length} from {@code moe_intermediate_size} and
     * {@code expert_shared_feed_forward_length} from {@code shared_expert_intermediate_size} when present.
     */
    public static void qwen2MoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        OptionalInt ff = cfg.findInt("moe_intermediate_size");
        if (ff.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, ff.getAsInt());
        }
        OptionalInt shared = cfg.findInt("shared_expert_intermediate_size");
        if (shared.isPresent()) {
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, shared.getAsInt());
        }
    }

    /**
     * {@code Qwen2MoeModel.modify_tensors} (qwen.py:97-152):
     * <ul>
     *   <li>a fused {@code mlp.experts.down_proj} is mapped with a {@code .weight} suffix;</li>
     *   <li>a fused {@code mlp.experts.gate_up_proj} {@code [E, 2F, D]} is split in halves along dim -2, the gate
     *       first, into {@code mlp.experts.gate_proj.weight} and {@code mlp.experts.up_proj.weight};</li>
     *   <li>per-expert {@code down_proj}/{@code gate_proj}/{@code up_proj} tensors are stacked per layer, over
     *       {@code num_local_experts} (or {@code num_experts}) experts, into
     *       {@code model.layers.<b>.mlp.experts.<part>.weight};</li>
     *   <li>everything else is mapped by name.</li>
     * </ul>
     *
     * @throws GgufFormatException when a fused gate/up tensor is not {@code [..., 2F, D]}, or an expert tensor is not
     *                             one of the stacked parts ({@code Unprocessed experts})
     */
    public static void qwen2MoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.endsWith(FUSED_DOWN) || hfName.endsWith(FUSED_DOWN + WEIGHT)) {
            String mapped = hfName.endsWith(WEIGHT) ? hfName : hfName + WEIGHT;
            HfTensorHook.DEFAULT.apply(ctx, mapped, src, out);
            return;
        }
        if (hfName.endsWith(FUSED_GATE_UP) || hfName.endsWith(FUSED_GATE_UP + WEIGHT)) {
            long[] shape = src.shape();
            if (shape.length < 3 || shape[shape.length - 2] % 2 != 0) {
                throw new GgufFormatException("Unexpected gate_up_proj shape for " + hfName + ": " + tuple(shape));
            }
            RowSource[] gateUp = ExpertTransforms.splitExperts(src, -2, 2);
            String base = strip(strip(hfName, WEIGHT), ".gate_up_proj");
            HfTensorHook.DEFAULT.apply(ctx, base + ".gate_proj" + WEIGHT, gateUp[0], out);
            HfTensorHook.DEFAULT.apply(ctx, base + ".up_proj" + WEIGHT, gateUp[1], out);
            return;
        }
        if (hfName.contains("experts")) {
            HfHookSupport.layerOf(hfName); // the converter requires a layer number here (qwen.py:126)
            if (!experts(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * {@code Qwen2MoeModel.prepare_tensors}' check (qwen.py:154-161), run after every tensor is planned: every layer
     * whose per-expert tensors were offered was stacked.
     *
     * @throws GgufFormatException naming the first incomplete layer
     */
    public static void qwen2MoeExtra(HfLoadContext ctx, TensorSink out) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector != null) {
                e.collector.requireComplete();
            }
        }
    }

    /** The per-load expert collector, created on the first expert tensor. */
    private static final class Experts {
        HfHookSupport.ExpertCollector collector;
    }

    private static HfHookSupport.ExpertCollector experts(HfLoadContext ctx) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector == null) {
                int n = ctx.config().requireInt("num_local_experts", "num_experts");
                e.collector = new HfHookSupport.ExpertCollector(n, EXPERT_PARTS,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return e.collector;
        }
    }

    private static String strip(String s, String suffix) {
        return s.endsWith(suffix) ? s.substring(0, s.length() - suffix.length()) : s;
    }

    /** A shape as Python prints a tuple. */
    private static String tuple(long[] shape) {
        if (shape.length == 1) {
            return "(" + shape[0] + ",)";
        }
        StringJoiner j = new StringJoiner(", ", "(", ")");
        for (long d : shape) {
            j.add(Long.toString(d));
        }
        return j.toString();
    }

    // ------------------------------------------------------------------ Qwen3MoeModel

    /**
     * {@code Qwen3MoeModel.set_vocab} (qwen.py:278-283): the InternS1 vocabulary for
     * {@value HfQwenHooks#INTERN_S1} checkpoints, else {@link #qwen2MoeVocab}.
     */
    public static Map<String, Object> qwen3MoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (HfQwenHooks.INTERN_S1.equals(HfQwenHooks.originArchitecture(ctx.config()))) {
            return HfVocabWriters.internS1(HfHookSupport.vocabInput(ctx), pre);
        }
        return qwen2MoeVocab(ctx, pre);
    }
}
