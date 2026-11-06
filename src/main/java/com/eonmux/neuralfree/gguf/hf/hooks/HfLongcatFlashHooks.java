package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face LongCat-Flash conversion (vLLM longcat_flash.py, longcat_flash_ngram.py and longcat_flash_mtp.py,
 * which have no llama.cpp counterpart): a decoder whose logical layer is a dual attention and MLP structure.
 *
 * <p>Each logical layer {@code L} of the checkpoint becomes two GGUF blocks, {@code 2L} and {@code 2L + 1}: block
 * {@code 2L} carries {@code self_attn.0}, {@code mlps.0}, {@code input_layernorm.0},
 * {@code post_attention_layernorm.0} and the shortcut MoE ({@code mlp.router.*}, {@code mlp.experts.*}); block
 * {@code 2L + 1} carries the second attention and MLP. {@code num_hidden_layers} (the physical count) is the GGUF
 * block count. MLA is stored absorbed, so {@code kv_b_proj} is split into {@code attn_k_b}/{@code attn_v_b} with
 * {@link QkvLayouts#mlaKvB}, exactly as the DeepSeek converters do. The two LoRA layernorms absorb
 * {@code mla_scale_q_lora}/{@code mla_scale_kv_lora}, as vLLM's {@code scale_mla_lora_norms_on_load} folds them into
 * the weights at load time.
 *
 * <p>The n-gram variant writes the checkpoint's per-embedder hash tables as one concatenated
 * {@code ngram_embd.weight} and its projections as one stacked {@code ngram_proj.weight} (the transposed
 * {@code post_projs}). The MTP head under {@code model.mtp.*} is renamed to the NextN tensors of the appended block,
 * which {@code --no-mtp} drops.
 */
public final class HfLongcatFlashHooks {

    /** The merged expert parts, in the order the converter writes them. */
    private static final List<String> EXPERT_PARTS = List.of("down_proj", "gate_proj", "up_proj");
    private static final String EXPERT_STATE = "longcat.experts";
    private static final String NGRAM_STATE = "longcat.ngram";

    /** {@code model.layers.<L>.<part>} after the model prefix is removed. */
    private static final Pattern LAYER = Pattern.compile("layers\\.(\\d{1,9})\\.(.+)");
    /** {@code mlp.experts.<x>.<part>.weight}. */
    private static final Pattern EXPERT = Pattern.compile("mlp\\.experts\\.(\\d{1,9})\\.([a-z_]+)\\.weight");
    /** {@code ngram_embeddings.(embedders|post_projs).<e>.weight}. */
    private static final Pattern NGRAM = Pattern.compile("ngram_embeddings\\.(embedders|post_projs)\\.(\\d{1,9})\\.weight");
    private static final Pattern LAYER_NUMBER = Pattern.compile("layers\\.(\\d{1,9})\\.");

    private HfLongcatFlashHooks() {
    }

    // ------------------------------------------------------------------ resolver

    /** The n-gram variant is the architecture whose configuration carries the hash-embedding parameters. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return usesNgramEmbedding(cfg) ? Arch.LONGCAT_FLASH_NGRAM : Arch.LONGCAT_FLASH;
    }

    /** {@code uses_ngram_embedding} (longcat_flash_ngram.py:44-45). */
    static boolean usesNgramEmbedding(HfConfig cfg) {
        return cfg.find("ngram_vocab_size_ratio").isPresent();
    }

    // ------------------------------------------------------------------ filter

    /** Drops the rotary buffers, the tied head and, under {@code --no-mtp}, the multi-token prediction head. */
    public static String filter(String name, UnaryOperator<String> base) {
        if (name.contains("rotary_emb.inv_freq")) {
            return null;
        }
        return base.apply(name);
    }

    /**
     * The class's load-scoped filter: {@code model.mtp.*} is dropped when {@code --no-mtp} leaves the NextN head
     * out, and the rotary buffers are always dropped.
     */
    public static HfNameFilter loadFilter(HfLoadContext ctx) {
        boolean keepMtp = ctx.options().includeMtp();
        return (name, base) -> {
            if (!keepMtp && (name.startsWith("model.mtp.") || name.startsWith("mtp."))) {
                return null;
            }
            return filter(name, base);
        };
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The LongCat hyperparameters: the generic keys with the KV head count forced to one (MLA is stored as MQA),
     * then the LoRA ranks, MLA key and value lengths, FFN widths, expert counts, the identity-expert slots, the RoPE
     * dimension count, the NextN block count and, for the n-gram variant, the hash parameters.
     *
     * @throws GgufFormatException when a required hyperparameter is missing or inconsistent
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        String act = cfg.findStringOr("silu", "hidden_act");
        if (!"silu".equals(act)) {
            throw invalid(cfg, "hidden_act", "LongCat-Flash's dense and expert MLPs are SwiGLU, got '" + act + "'");
        }
        HfGenericMetadata.apply(ctx, kv);
        kv.remove(GgufKeys.ATTENTION_HEAD_COUNT_KV);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int qLora = cfg.requireInt("q_lora_rank");
        int kvLora = cfg.requireInt("kv_lora_rank");
        int ropeDim = cfg.requireInt("qk_rope_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int qkHead = cfg.findInt(qkNope + ropeDim, "qk_head_dim");
        if (qkHead != qkNope + ropeDim) {
            throw invalid(cfg, "qk_head_dim", "must be qk_nope_head_dim + qk_rope_head_dim = " + (qkNope + ropeDim)
                    + ", got " + qkHead);
        }
        kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora);
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + ropeDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, qkHead);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("ffn_hidden_size"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("expert_ffn_hidden_size", "moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("n_routed_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_topk"));
        Object zeroType = cfg.find("zero_expert_type").orElse(null);
        if (zeroType != null && !"identity".equals(String.valueOf(zeroType))) {
            throw invalid(cfg, "zero_expert_type", "only the identity experts are supported, got '" + zeroType + "'");
        }
        int zero = cfg.findInt(0, "zero_expert_num");
        if (zero > 0) {
            kv.set(GgufKeys.EXPERT_ZERO_COUNT, zero);
        }
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, (float) cfg.findFloat(1.0, "routed_scaling_factor"));
        int nextn = ctx.blockCount() - blockCount(cfg);
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
        }
        Object mscaleAllDim = cfg.rope().raw().get("mscale_all_dim");
        if (mscaleAllDim instanceof Number) {
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1 * ((Number) mscaleAllDim).doubleValue());
        }
        if (ctx.arch() == Arch.LONGCAT_FLASH_NGRAM) {
            readNgram(cfg, kv);
        }
    }

    /** The hash parameters of the fused input embedding ({@code NgramEmbedding.__init__}). */
    private static void readNgram(HfConfig cfg, HfMetadataBuilder kv) {
        int vocab = cfg.requireInt("vocab_size");
        double ratio = cfg.findFloat("ngram_vocab_size_ratio")
                .orElseThrow(() -> invalid(cfg, "ngram_vocab_size_ratio", "the n-gram variant needs the ratio"));
        int neighbors = cfg.requireInt("emb_neighbor_num");
        int splits = cfg.requireInt("emb_split_num");
        if (!(ratio > 0.0) || neighbors < 2 || splits < 1) {
            throw invalid(cfg, "emb_neighbor_num", "the n-gram embedding needs a positive ratio, at least two"
                    + " neighbors and one split, got " + ratio + ", " + neighbors + ", " + splits);
        }
        int base = (int) (ratio * vocab);
        if (base <= 0) {
            throw invalid(cfg, "ngram_vocab_size_ratio", "gives a base table of " + base + " rows");
        }
        kv.set(GgufKeys.NGRAM_BASE_TABLE_SIZE, base);
        kv.set(GgufKeys.NGRAM_NEIGHBOR_COUNT, neighbors);
        kv.set(GgufKeys.NGRAM_SPLIT_COUNT, splits);
        kv.set(GgufKeys.NGRAM_EOS_TOKEN_ID, cfg.requireInt("eos_token_id"));
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }

    // ------------------------------------------------------------------ block count

    /**
     * The checkpoint's physical block count ({@code 2 * num_layers}): the configuration derives
     * {@code num_hidden_layers} from {@code num_layers} (configuration_longcat_flash.py), so a file carrying both
     * must agree.
     */
    public static int blockCount(HfConfig cfg) {
        int layers = cfg.findInt(-1, "num_layers");
        int n = cfg.findInt(0, "num_hidden_layers");
        if (layers < 1) {
            if (n < 2 || n % 2 != 0) {
                throw invalid(cfg, "num_layers", "is missing and num_hidden_layers is not twice a layer count: " + n);
            }
            layers = n / 2;
        }
        if (n != 0 && n != 2 * layers) {
            throw invalid(cfg, "num_hidden_layers", "must be twice num_layers (" + layers + "), got " + n);
        }
        if (layers > com.eonmux.neuralfree.gguf.hparams.ModelHparams.MAX_LAYERS / 2) {
            throw invalid(cfg, "num_layers", "gives " + (2L * layers) + " blocks, more than "
                    + com.eonmux.neuralfree.gguf.hparams.ModelHparams.MAX_LAYERS);
        }
        return 2 * layers;
    }

    /**
     * The block count including the MTP block: the checkpoint stores its multi-token prediction head as
     * {@code model.mtp.layers.0} at {@code num_layers * 2} (longcat_flash_mtp.py:110), and the appended NextN block
     * keeps it unless {@code --no-mtp} drops it.
     */
    public static int blockCount(HfLoadContext ctx) {
        int base = blockCount(ctx.config());
        boolean keep = ctx.options().includeMtp() && !usesNgramEmbedding(ctx.config())
                && hasMtp(ctx.index(), ctx.config());
        return keep ? base + 1 : base;
    }

    /**
     * True when the checkpoint carries the multi-token prediction head, either under {@code mtp.} or as the extra
     * decoder layer the DeepSeek-family checkpoints append after the target's blocks.
     */
    static boolean hasMtp(HfTensorIndex index, HfConfig cfg) {
        if (index.has("model.mtp.layers.0.eh_proj.weight") || index.has("mtp.layers.0.eh_proj.weight")) {
            return true;
        }
        int base = blockCount(cfg);
        return index.has("model.layers." + base + ".eh_proj.weight")
                || index.has("model.layers." + base + ".transformer_layer.eh_proj.weight");
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code LongcatFlashForCausalLM.load_weights}: maps the dual-attention blocks, stacks the routed experts,
     * splits {@code kv_b_proj}, folds the MLA LoRA scales into the two layernorms, and renames the MTP head.
     *
     * @throws GgufFormatException when the checkpoint carries a tensor the layout does not define
     * @throws IOException         when a tensor cannot be read
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (isTiedHead(cfg, hfName)) {
            out.drop(hfName, "tied output layer; token_embd.weight serves as the output");
            return;
        }
        if (hfName.startsWith("model.mtp.") || hfName.startsWith("mtp.")) {
            mtpTensor(ctx, hfName, src, out);
            return;
        }
        String name = HfHookSupport.stripPrefixes(hfName, "model.");
        if (mapModelTensor(ctx, name, src, out)) {
            return;
        }
        if (name.startsWith("layers.")) {
            int mtpLayer = blockCount(cfg);
            Matcher at = LAYER.matcher(name);
            if (at.matches() && Integer.parseInt(at.group(1)) == mtpLayer) {
                mtpTensor(ctx, "mtp." + name.replaceFirst("^layers\\.\\d+\\.", "layers.0."), src, out);
                return;
            }
            layerTensor(ctx, name, src, out);
            return;
        }
        throw new GgufFormatException(cfg.source() + ": LongCat-Flash does not support tensor '" + hfName + "'");
    }

    private static boolean isTiedHead(HfConfig cfg, String hfName) {
        return cfg.findBool(false, "tie_word_embeddings")
                && (hfName.equals("lm_head.weight") || hfName.equals("model.lm_head.weight"));
    }

    /** The model-level tensors, and true when {@code name} is one of them. */
    private static boolean mapModelTensor(HfLoadContext ctx, String name, RowSource src, TensorSink out)
            throws IOException {
        switch (name) {
            case "embed_tokens.weight":
                out.add("token_embd.weight", src);
                return true;
            case "norm.weight":
                out.add("output_norm.weight", src);
                return true;
            default:
                break;
        }
        if (name.equals("lm_head.weight")) {
            out.add("output.weight", src);
            return true;
        }
        Matcher ngram = NGRAM.matcher(name);
        if (ngram.matches()) {
            ngramTensor(ctx, ngram.group(1), Integer.parseInt(ngram.group(2)), out);
            return true;
        }
        return false;
    }

    /** One physical block of the checkpoint's dual-attention layers. */
    private static void layerTensor(HfLoadContext ctx, String name, RowSource src, TensorSink out)
            throws IOException {
        Matcher m = LAYER.matcher(name);
        if (!m.matches()) {
            throw new GgufFormatException(ctx.config().source() + ": LongCat-Flash does not support tensor '"
                    + name + "'");
        }
        int layer = Integer.parseInt(m.group(1));
        String part = m.group(2);
        Matcher expert = EXPERT.matcher(part);
        if (expert.matches()) {
            experts(ctx, layer, Integer.parseInt(expert.group(1)), expert.group(2), out);
            return;
        }
        if (part.equals("mlp.router.classifier.weight")) {
            out.add(block(layer, 0, "ffn_gate_inp.weight"), src);
            return;
        }
        if (part.equals("mlp.router.e_score_correction_bias")
                || part.equals("mlp.router.e_score_correction.bias")) {
            out.add(block(layer, 0, "exp_probs_b.bias"), src);
            return;
        }
        int half = half(part);
        String suffix = part.replaceFirst("^(self_attn|mlps|input_layernorm|post_attention_layernorm)\\.\\d\\.", "$1.");
        if (suffix.equals("input_layernorm.weight")) {
            out.add(block(layer, half, "attn_norm.weight"), src);
            return;
        }
        if (suffix.equals("post_attention_layernorm.weight")) {
            out.add(block(layer, half, "ffn_norm.weight"), src);
            return;
        }
        if (suffix.equals("self_attn.q_a_layernorm.weight")) {
            out.add(block(layer, half, "attn_q_a_norm.weight"), scaleLoraNorm(ctx, src, true));
            return;
        }
        if (suffix.equals("self_attn.kv_a_layernorm.weight")) {
            out.add(block(layer, half, "attn_kv_a_norm.weight"), scaleLoraNorm(ctx, src, false));
            return;
        }
        if (suffix.equals("self_attn.kv_b_proj.weight")) {
            splitKvB(ctx, layer, half, src, out);
            return;
        }
        String attention = attentionSuffix(suffix);
        if (attention != null) {
            out.add(block(layer, half, attention), src);
            return;
        }
        String mlp = mlpSuffix(suffix);
        if (mlp != null) {
            out.add(block(layer, half, mlp), src);
            return;
        }
        throw new GgufFormatException(ctx.config().source() + ": LongCat-Flash does not support tensor '"
                + name + "'");
    }

    /** {@code <group>.<index>.<rest>} to its index: the dual attention and MLP groups have exactly two. */
    private static int half(String part) {
        Matcher m = Pattern.compile("(?:self_attn|mlps|input_layernorm|post_attention_layernorm)\\.(\\d)\\..+")
                .matcher(part);
        if (!m.matches()) {
            throw new GgufFormatException("LongCat-Flash tensor part '" + part + "' is outside the dual structure");
        }
        int i = Integer.parseInt(m.group(1));
        if (i != 0 && i != 1) {
            throw new GgufFormatException("LongCat-Flash tensor part '" + part + "' has index " + i
                    + "; a layer has exactly two of each");
        }
        return i;
    }

    private static String attentionSuffix(String suffix) {
        switch (suffix) {
            case "self_attn.q_a_proj.weight":
                return "attn_q_a.weight";
            case "self_attn.q_b_proj.weight":
                return "attn_q_b.weight";
            case "self_attn.kv_a_proj_with_mqa.weight":
                return "attn_kv_a_mqa.weight";
            case "self_attn.o_proj.weight":
                return "attn_output.weight";
            default:
                return null;
        }
    }

    private static String mlpSuffix(String suffix) {
        switch (suffix) {
            case "mlps.gate_proj.weight":
                return "ffn_gate.weight";
            case "mlps.up_proj.weight":
                return "ffn_up.weight";
            case "mlps.down_proj.weight":
                return "ffn_down.weight";
            case "mlp.gate_proj.weight":
                return "ffn_gate.weight";
            case "mlp.up_proj.weight":
                return "ffn_up.weight";
            case "mlp.down_proj.weight":
                return "ffn_down.weight";
            default:
                return null;
        }
    }

    /** The GGUF name of {@code <suffix>} on physical block {@code 2 * layer + half}. */
    private static String block(int layer, int half, String suffix) {
        return "blk." + (2 * layer + half) + "." + suffix;
    }

    /**
     * {@code scale_mla_lora_norms_on_load} (longcat_flash.py:82-94): {@code sqrt(hidden_size / rank)} onto the Q and
     * KV LoRA layernorms the configuration marks.
     */
    private static RowSource scaleLoraNorm(HfLoadContext ctx, RowSource src, boolean query) {
        HfConfig cfg = ctx.config();
        boolean enabled = cfg.findBool(false, query ? "mla_scale_q_lora" : "mla_scale_kv_lora");
        if (!enabled) {
            return src;
        }
        int rank = query ? cfg.requireInt("q_lora_rank") : cfg.requireInt("kv_lora_rank");
        float scale = (float) Math.sqrt((double) cfg.requireInt("hidden_size") / rank);
        return ElementTransforms.mulScalar(src, scale);
    }

    /** {@code kv_b [n_head*(nope+v), kv_lora]} to {@code attn_k_b}/{@code attn_v_b}, as the DeepSeek converters do. */
    private static void splitKvB(HfLoadContext ctx, int layer, int half, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor 'kv_b_proj.weight' of layer " + layer + " has shape "
                    + java.util.Arrays.toString(shape) + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, cfg.requireInt("qk_nope_head_dim"), cfg.requireInt("v_head_dim"),
                    Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor 'kv_b_proj.weight' of layer " + layer + " does not split into "
                    + nHead + " heads: " + e.getMessage(), e);
        }
        out.add(block(layer, half, "attn_k_b.weight"), kvB[0]);
        out.add(block(layer, half, "attn_v_b.weight"), kvB[1]);
    }

    // ------------------------------------------------------------------ experts

    /**
     * Stacks the routed experts of one layer into the three GGUF expert tensors, as the converter's {@code _experts}
     * buffers do: the members are verified and consumed when the first one is offered, and the stacked tensors are
     * planned when the member that comes last in the index does.
     */
    private static void experts(HfLoadContext ctx, int layer, int expert, String part, TensorSink out) {
        expertGroups(ctx).offer(ctx, layer, expert, part, out);
    }

    private static ExpertGroups expertGroups(HfLoadContext ctx) {
        return ctx.state(EXPERT_STATE, ExpertGroups.class, () -> new ExpertGroups(
                ctx.config().requireInt("n_routed_experts")));
    }

    /** Checks that every expert of each offered layer and every n-gram table set were planned. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        expertGroups(ctx).requireComplete();
        if (usesNgramEmbedding(ctx.config())) {
            ngramTables(ctx).requireComplete();
        }
    }

    private static NgramTables ngramTables(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        return ctx.state(NGRAM_STATE, NgramTables.class,
                () -> new NgramTables(cfg.requireInt("emb_split_num") * (cfg.requireInt("emb_neighbor_num") - 1)));
    }

    /** One layer's expert members, stacked when the last of them arrives. */
    private static final class ExpertGroups {
        private final int nExperts;
        private final Map<String, Group> groups = new LinkedHashMap<>();

        ExpertGroups(int nExperts) {
            this.nExperts = nExperts;
        }

        void offer(HfLoadContext ctx, int layer, int expert, String part, TensorSink out) {
            if (!EXPERT_PARTS.contains(part)) {
                throw new GgufFormatException("layer " + layer + ": expert part '" + part + "' is not one of "
                        + EXPERT_PARTS);
            }
            if (expert < 0 || expert >= nExperts) {
                throw new GgufFormatException("layer " + layer + ": expert " + expert + " is outside [0, "
                        + nExperts + ")");
            }
            String key = layer + ":" + part;
            Group g = groups.get(key);
            if (g == null) {
                g = open(ctx, layer, part);
                groups.put(key, g);
            }
            if (member(layer, expert, part).equals(g.last)) {
                emit(ctx, layer, part, out);
                g.done = true;
            }
        }

        private Group open(HfLoadContext ctx, int layer, String part) {
            List<String> names = ctx.index().names();
            String last = null;
            int lastPos = -1;
            for (int x = 0; x < nExperts; x++) {
                String name = member(layer, x, part);
                if (!ctx.index().has(name)) {
                    throw new GgufFormatException("layer " + layer + ": expert tensor '" + name + "' is missing"
                            + " (expected " + nExperts + " experts)");
                }
                int pos = names.indexOf(name);
                if (pos > lastPos) {
                    lastPos = pos;
                    last = name;
                }
            }
            return new Group(last);
        }

        private void emit(HfLoadContext ctx, int layer, String part, TensorSink out) {
            List<RowSource> sources = new ArrayList<>(nExperts);
            for (int x = 0; x < nExperts; x++) {
                sources.add(ctx.index().get(member(layer, x, part)));
            }
            out.add(block(layer, 0, "ffn_" + expertShort(part) + "_exps.weight"), ExpertTransforms.stack(sources));
        }

        void requireComplete() {
            for (Map.Entry<String, Group> e : groups.entrySet()) {
                if (!e.getValue().done) {
                    throw new GgufFormatException("experts of '" + e.getKey() + "' were not all planned");
                }
            }
        }

        private static String member(int layer, int expert, String part) {
            return "model.layers." + layer + ".mlp.experts." + expert + "." + part + ".weight";
        }
    }

    private static String expertShort(String part) {
        switch (part) {
            case "gate_proj":
                return "gate";
            case "up_proj":
                return "up";
            default:
                return "down";
        }
    }

    private static final class Group {
        final String last;
        boolean done;

        Group(String last) {
            this.last = last;
        }
    }

    // ------------------------------------------------------------------ n-gram embedding

    /**
     * Concatenates the per-embedder hash tables ({@code ngram_embd.weight}) and stacks their transposed projections
     * ({@code ngram_proj.weight}, an {@code [embedders * n_embd / embedders, n_embd]} matrix).
     */
    private static void ngramTensor(HfLoadContext ctx, String group, int index, TensorSink out) {
        ngramTables(ctx).offer(ctx, index, group.equals("post_projs"), out);
    }

    /** The embedder tables and projections of one checkpoint, planned when their last member arrives. */
    private static final class NgramTables {
        private final int embedders;
        private boolean touched;
        private boolean tablesDone;
        private boolean projectionsDone;

        NgramTables(int embedders) {
            this.embedders = embedders;
        }

        void offer(HfLoadContext ctx, int index, boolean projection, TensorSink out) {
            if (index < 0 || index >= embedders) {
                throw new GgufFormatException("n-gram embedder " + index + " is outside [0, " + embedders + ")");
            }
            touched = true;
            List<String> names = ctx.index().names();
            if (names.indexOf(name(projection, index)) != maxPosition(names, projection)) {
                out.consume(name(projection, index));
                return;
            }
            emit(ctx, projection, out);
        }

        private int maxPosition(List<String> names, boolean projection) {
            int max = -1;
            for (int i = 0; i < embedders; i++) {
                max = Math.max(max, names.indexOf(name(projection, i)));
            }
            return max;
        }

        private void emit(HfLoadContext ctx, boolean projection, TensorSink out) {
            List<RowSource> sources = new ArrayList<>(embedders);
            for (int i = 0; i < embedders; i++) {
                RowSource s = ctx.index().get(name(projection, i));
                if (s == null) {
                    throw new GgufFormatException("n-gram tensor '" + name(projection, i) + "' is missing (expected "
                            + embedders + " " + (projection ? "projections" : "tables") + ")");
                }
                sources.add(s);
            }
            if (!projection) {
                out.add("ngram_embd.weight", RowTransforms.concatRows(sources));
                tablesDone = true;
                return;
            }
            long[] shape = sources.get(0).shape();
            long nEmbd = shape[0];
            long width = shape[1];
            try {
                // one [n_embd, embedders * width] matrix whose column blocks are the embedders' projections
                RowSource stacked = ExpertTransforms.stack(sources);
                RowSource permuted = ElementTransforms.permuteDims(stacked, 1, 0, 2);
                out.add("ngram_proj.weight", ElementTransforms.reshape(permuted, nEmbd, embedders * width));
            } catch (IOException e) {
                throw new GgufFormatException("n-gram projections cannot be stacked: " + e.getMessage(), e);
            }
            projectionsDone = true;
        }

        /**
         * @throws GgufFormatException when a checkpoint offered hash tables or projections whose last member never
         *                             arrived
         */
        void requireComplete() {
            if (touched && !(tablesDone && projectionsDone)) {
                throw new GgufFormatException("the n-gram input embedding was not planned completely: tables "
                        + (tablesDone ? "done" : "missing") + ", projections "
                        + (projectionsDone ? "done" : "missing"));
            }
        }

        private static String name(boolean projection, int index) {
            return "model.ngram_embeddings." + (projection ? "post_projs" : "embedders") + "." + index + ".weight";
        }
    }

    // ------------------------------------------------------------------ MTP head

    /**
     * {@code LongCatFlashMTP.load_weights} (longcat_flash_mtp.py:174-249): the single MTP layer becomes the appended
     * NextN block at {@code num_layers * 2}, its {@code enorm}/{@code hnorm}/{@code eh_proj} and final norm become the
     * {@code nextn} tensors, and the embedding it shares with the target is dropped.
     */
    private static void mtpTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String name = HfHookSupport.stripPrefixes(hfName, "mtp.", "model.mtp.");
        if (name.equals("embed_tokens.weight") || name.equals("layers.0.embed_tokens.weight")) {
            out.drop(hfName, "the MTP head shares the target's token embedding");
            return;
        }
        int nextn = ctx.blockCount() - 1;
        if (name.equals("norm.weight") || name.equals("layers.0.norm.weight")
                || name.equals("layers.0.final_layernorm.weight")
                || name.equals("layers.0.shared_head.norm.weight")) {
            out.add("blk." + nextn + ".nextn.shared_head_norm.weight", src);
            return;
        }
        if (name.equals("layers.0.enorm.m.weight") || name.equals("layers.0.enorm.weight")
                || name.equals("enorm.weight")) {
            out.add("blk." + nextn + ".nextn.enorm.weight", src);
            return;
        }
        if (name.equals("layers.0.hnorm.m.weight") || name.equals("layers.0.hnorm.weight")
                || name.equals("hnorm.weight")) {
            out.add("blk." + nextn + ".nextn.hnorm.weight", src);
            return;
        }
        if (name.equals("layers.0.eh_proj.weight") || name.equals("eh_proj.weight")) {
            out.add("blk." + nextn + ".nextn.eh_proj.weight", src);
            return;
        }
        String block = name.replaceFirst("^layers\\.0\\.transformer_layer\\.", "layers.0.");
        Matcher m = LAYER.matcher(block);
        if (m.matches() && m.group(2).startsWith("self_attn.")) {
            String suffix = m.group(2);
            if (suffix.equals("self_attn.kv_b_proj.weight")) {
                splitKvB(ctx, nextn / 2, nextn % 2, src, out);
                return;
            }
            if (suffix.equals("self_attn.q_a_layernorm.weight")) {
                out.add("blk." + nextn + ".attn_q_a_norm.weight", scaleLoraNorm(ctx, src, true));
                return;
            }
            if (suffix.equals("self_attn.kv_a_layernorm.weight")) {
                out.add("blk." + nextn + ".attn_kv_a_norm.weight", scaleLoraNorm(ctx, src, false));
                return;
            }
            String mapped = attentionSuffix(suffix);
            if (mapped != null) {
                out.add("blk." + nextn + "." + mapped, src);
                return;
            }
        }
        if (m.matches() && m.group(2).equals("input_layernorm.weight")) {
            out.add("blk." + nextn + ".attn_norm.weight", src);
            return;
        }
        if (m.matches() && m.group(2).equals("post_attention_layernorm.weight")) {
            out.add("blk." + nextn + ".ffn_norm.weight", src);
            return;
        }
        if (m.matches() && mlpSuffix(mtpMlp(m.group(2))) != null) {
            out.add("blk." + nextn + "." + mlpSuffix(mtpMlp(m.group(2))), src);
            return;
        }
        throw new GgufFormatException(ctx.config().source() + ": LongCat-Flash does not support MTP tensor '"
                + hfName + "'");
    }

    /** The MTP decoder layer spells its MLP {@code mlp.*}; the trunk's dense MLPs are {@code mlps.<i>.*}. */
    private static String mtpMlp(String part) {
        return part.startsWith("mlp.") ? "mlps." + part.substring("mlp.".length()) : part;
    }

    // ------------------------------------------------------------------ vocabulary

    /** The vocabulary of the checkpoint: the GPT-2 tokenizer.json writer, as the DeepSeek converters use. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }
}
