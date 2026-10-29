package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The hooks of three Gemma converter classes {@link HfGemmaHooks} (the Gemma 3 hooks of HK-B2) does not cover:
 * {@code GemmaModel} (LC/conversion/gemma.py:20-70), {@code Gemma2Model} (:73-122) and {@code Gemma3NModel}
 * (:489-629): the class's own hand-written metadata, the +1 norm shift and {@code lm_head} filter of Gemma 1 and 2,
 * and the AltUp stacking, embedding padding and activation-sparsity keys Gemma 3n adds over the shared Gemma 3 path.
 */
public final class HfDenseHooksA {

    /** The tied output head both classes drop (gemma.py:59-61, :111-113). */
    private static final String LM_HEAD = "lm_head.weight";
    private static final String SCALE_TAIL = "_scale";

    /** The CodeGemma infill ids of {@code set_vocab} (gemma.py:27-35). */
    public static final int FIM_PRE_ID = 67;
    public static final int FIM_SUF_ID = 69;
    public static final int FIM_MID_ID = 68;
    public static final int FIM_SEP_ID = 70;
    public static final int FIM_EOT_ID = 107;

    /** The {@code altup_num_inputs} the class converts (gemma.py:497). */
    public static final int ALTUP_NUM_INPUTS = 4;
    /** The AltUp slots, the indexes the checkpoint stores per family (gemma.py:597-621). */
    public static final int ALTUP_SLOTS = 3;
    /** The token vocabulary the converter pads to when the configuration omits it (gemma.py:576). */
    public static final int DEFAULT_GEMMA3N_VOCAB = 262400;

    private static final String LAND_MARKER = "altup_projections";
    private static final String UNEMBED_MARKER = "altup_unembed_projections";
    private static final String EMBED_TOKENS = "embed_tokens.weight";
    private static final String EMBED_PER_LAYER = "embed_tokens_per_layer";
    private static final String ALTUP_STATE = HfDenseHooksA.class.getName() + ".altup";

    private HfDenseHooksA() {
    }

    // ------------------------------------------------------------------ Gemma (v1) and Gemma2

    /**
     * The class filter of {@code GemmaModel} and {@code Gemma2Model} (gemma.py:53-63, :105-115): the tied
     * {@value #LM_HEAD} tensor is dropped before the inherited filters, so a checkpoint that carries it converts.
     */
    public static final HfNameFilter GEMMA_FILTER = (name, base) ->
            LM_HEAD.equals(name) ? null : base.apply(name);

    /**
     * {@code GemmaModel.set_vocab} (gemma.py:23-37): the SentencePiece vocabulary, with the CodeGemma infill tokens
     * (prefix {@value #FIM_PRE_ID}, suffix {@value #FIM_SUF_ID}, middle {@value #FIM_MID_ID}, file separator
     * {@value #FIM_SEP_ID}, eot {@value #FIM_EOT_ID}), no chat template of its own, and {@code add_space_prefix}.
     */
    public static Map<String, Object> gemmaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_FIM_PRE_ID, FIM_PRE_ID);
        md.put(GgufKeys.TOKENIZER_FIM_SUF_ID, FIM_SUF_ID);
        md.put(GgufKeys.TOKENIZER_FIM_MID_ID, FIM_MID_ID);
        md.put(GgufKeys.TOKENIZER_FIM_SEP_ID, FIM_SEP_ID);
        md.put(GgufKeys.TOKENIZER_EOT_ID, FIM_EOT_ID);
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code Gemma2Model.set_vocab} (gemma.py:79-82): the SentencePiece vocabulary with {@code add_space_prefix}
     * off and no CodeGemma tokens.
     */
    public static Map<String, Object> gemma2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code GemmaModel.set_gguf_parameters} (gemma.py:39-51): the class's own keys, without the RoPE rows it never
     * wrote: context, embedding, blocks, FFN, head and KV-head counts (the KV heads fall back to the head count),
     * RMS epsilon and the equal key and value lengths {@code head_dim}.
     */
    public static void gemmaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_position_embeddings"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        int heads = cfg.requireInt("num_attention_heads");
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.findInt(heads, "num_key_value_heads"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, requireDouble(cfg, "rms_norm_eps"));
        int headDim = cfg.requireInt("head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
    }

    /**
     * {@code Gemma2Model.set_gguf_parameters} (gemma.py:84-103): the Gemma 1 keys, then the required attention and
     * final logit soft-cappings and the required sliding window.
     */
    public static void gemma2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        gemmaMeta(ctx, kv, plan);
        kv.set(GgufKeys.ATTN_LOGIT_SOFTCAPPING, requireDouble(cfg, "attn_logit_softcapping"));
        kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, requireDouble(cfg, "final_logit_softcapping"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
    }

    /**
     * The common tensor step of {@code GemmaModel} and {@code Gemma2Model} modify_tensors (gemma.py:65-70,
     * :117-122): every {@code *norm.weight} gets +1 (the stored-minus-one RMSNorm multiplier), then the mapped
     * tensor.
     */
    public static void gemmaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = HfHookSupport.isNorm(hfName) ? HfHookSupport.normPlusOne(src) : src;
        HfTensorHook.DEFAULT.apply(ctx, hfName, s, out);
    }

    // ------------------------------------------------------------------ Gemma3N

    /** {@code norm_shift} of {@code Gemma3NModel} (gemma.py:509-511): no shift on any norm. */
    private static final boolean NO_NORM_SHIFT = false;

    /**
     * The class filter of {@code Gemma3NModel} (gemma.py:559-566): every scale parameter loses its bare
     * {@code *_scale} tail ({@code *_scale} becomes {@code *_scale.weight}), then the inherited filters apply.
     */
    public static final HfNameFilter GEMMA3N_FILTER = (name, base) -> {
        String n = name.endsWith(SCALE_TAIL) ? name + ".weight" : name;
        return base.apply(n);
    };

    /**
     * {@code Gemma3NModel.set_vocab} (gemma.py:513-531): the Gemma 3 vocabulary over the full {@code vocab_size};
     * the per-layer-input size the converter detaches for the same call plays no part in the writers' size rule
     * (they read {@code vocab_size}, never {@code vocab_size_per_layer_input}).
     */
    public static Map<String, Object> gemma3nVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.tokenizerFiles().hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(ctx.tokenizerFiles());
        }
        if (ctx.tokenizerFiles().file(HfTokenizerFiles.TOKENIZER_MODEL) == null) {
            return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
        }
        return HfGemmaHooks.vocab(ctx, pre);
    }

    /**
     * {@code Gemma3NModel.set_gguf_parameters} (gemma.py:533-550) over the Gemma 3 metadata of
     * {@link HfGemmaHooks#meta}: the AltUp index and input count, the per-layer-input width, the KV-shared layer
     * count, the per-layer activation sparsity (the standard normal inverse CDF of
     * {@code activation_sparsity_pattern}) and the sliding-window pattern of {@code layer_types}.
     *
     * @throws GgufFormatException when a required value is missing or the pattern is not a number array
     */
    public static void gemma3nMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGemmaHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ALTUP_ACTIVE_IDX, cfg.requireInt("altup_active_idx"));
        int altup = cfg.requireInt("altup_num_inputs");
        if (altup != ALTUP_NUM_INPUTS) {
            throw new GgufFormatException(cfg.source() + ": altup_num_inputs must be " + ALTUP_NUM_INPUTS + ", got "
                    + altup);
        }
        kv.set(GgufKeys.ALTUP_NUM_INPUTS, altup);
        kv.set(GgufKeys.EMBEDDING_LENGTH_PER_LAYER, cfg.requireInt("hidden_size_per_layer_input"));
        kv.set(GgufKeys.ATTENTION_SHARED_KV_LAYERS, cfg.requireInt("num_kv_shared_layers"));
        kv.set(GgufKeys.ACTIVATION_SPARSITY_SCALE, activationSparsityScale(cfg, 0));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, HfBaseDenseHooks.slidingFlags(cfg));
    }

    /**
     * {@link #activationSparsityScale}' list of {@code torch.distributions.normal.Normal(0, 1).icdf(s)} values
     * (gemma.py:540-545).
     *
     * @throws GgufFormatException when a pattern entry is not a number of (0, 1)
     */
    static List<Float> activationSparsityScale(HfConfig cfg, int depth) {
        Object v = cfg.flat().get("activation_sparsity_pattern");
        if (!(v instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'activation_sparsity_pattern' must be an array of"
                    + " sparsity fractions, got " + (v == null ? "nothing" : v.getClass().getSimpleName()));
        }
        List<?> pattern = (List<?>) v;
        if (pattern.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'activation_sparsity_pattern' must not be empty");
        }
        List<Float> out = new ArrayList<>(pattern.size());
        for (int i = 0; i < pattern.size(); i++) {
            Object p = pattern.get(i);
            if (!(p instanceof Number) || p instanceof Boolean) {
                throw new GgufFormatException(cfg.source() + ": activation_sparsity_pattern[" + i
                        + "] must be a number, got " + p);
            }
            double sparsity = ((Number) p).doubleValue();
            if (sparsity <= 0.0 || sparsity >= 1.0) {
                throw new GgufFormatException(cfg.source() + ": activation_sparsity_pattern[" + i
                        + "] must be between 0 and 1 (exclusive), got " + sparsity);
            }
            out.add((float) normalInverseCdf(sparsity));
        }
        return out;
    }

    /**
     * The standard normal inverse CDF. Newton iterations over the standard normal CDF {@link #phi}, whose
     * complementary error function keeps an absolute error below 1.4e-7 for {@code |x| <= 8.25} (the form of
     * Numerical Recipes' erfc), seeded at zero and converged; the float round of gemma.py:543-544 then fixes the
     * precision of the key. Sparsity fractions outside the code's range never push |x| far enough for the
     * approximation to lose more float digits than the rounding itself does.
     */
    static double normalInverseCdf(double p) {
        if (!(p > 0.0) || !(p < 1.0)) {
            throw new GgufFormatException("the normal inverse CDF is defined for probabilities in (0, 1), got "
                    + p);
        }
        double x = p - 0.5;
        for (int i = 0; i < 64; i++) {
            double err = phi(x) - p;
            double d = Math.exp(-0.5 * x * x) / CDF_ROOT_INV_2PI;
            double step = err / d;
            x -= step;
            if (Math.abs(step) < 1e-17) {
                break;
            }
        }
        return x;
    }

    /** The standard normal CDF of {@code x}, through the complementary error function. */
    private static double phi(double x) {
        // the normal CDF: 0.5 * erfc(-x / sqrt(2)) — a positive argument gives the upper tail instead
        return 0.5 * erfc(-x / Math.sqrt(2.0));
    }

    /** The scaling {@code 1/sqrt(2*pi)} of the normal CDF's derivative. */
    private static final double CDF_ROOT_INV_2PI = Math.sqrt(2.0 * Math.PI);

    /**
     * {@code erfc(x)} for {@code x >= 0}, through the rational form of Numerical Recipes' {@code erfc}
     * (an absolute error below 1.4e-7 at every argument in range here); the deep tails ({@code |x| > 8.25}) never
     * occur for a sparsity fraction in (0, 1) within float rounds. Negative arguments give {@code 2 - erfc(|x|)}.
     */
    private static double erfc(double x) {
        boolean negative = x < 0.0;
        double a = negative ? -x : x;
        if (a >= 8.25) {
            return negative ? 2.0 : 0.0;
        }
        double t = 2.0 / (2.0 + a);
        double poly = 1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
                + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))));
        double out = t * Math.exp(-a * a - 1.26551223 + t * poly);
        return negative ? 2.0 - out : out;
    }

    private static double requireDouble(HfConfig cfg, String key) {
        if (!cfg.flat().containsKey(key)) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        Object v = cfg.flat().get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": key '" + key + "' must be a number, got " + v);
        }
        return ((Number) v).doubleValue();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code Gemma3NModel.modify_tensors} (gemma.py:568-629): the token and per-layer embeddings are zero-padded up
     * to {@code vocab_size} (the vision/audio special tokens get no embedding rows of their own); the AltUp
     * projection and unembed-projection matrices of the three slots are stacked along dim 0 into one tensor per
     * family ({@code model.altup_projections.weight}, {@code model.altup_unembed_projections.weight}); every other
     * tensor follows the Gemma 3 path without the norm shift (gemma.py:509-511) and without the embedding
     * truncation its caller never reaches.
     */
    public static void gemma3nTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (hfName.contains(EMBED_TOKENS)) {
            padEmbedding(ctx, hfName, src, out, cfg.findInt(DEFAULT_GEMMA3N_VOCAB, "vocab_size"), "token"
                    + " embeddings");
            return;
        }
        if (hfName.contains(EMBED_PER_LAYER)) {
            padEmbedding(ctx, hfName, src, out, cfg.findInt(DEFAULT_GEMMA3N_VOCAB, "vocab_size"),
                    "per-layer embeddings");
            return;
        }
        if (hfName.contains(UNEMBED_MARKER) || hfName.contains(LAND_MARKER)) {
            altup(ctx, hfName, src, out, hfName.contains(UNEMBED_MARKER));
            return;
        }
        planGemma3NoShift(ctx, hfName, src, out);
    }

    /**
     * The embedded zeros of an under-sized embedding ({@code torch.cat([data_torch, padding], dim=0)},
     * gemma.py:586-587); a tensor of the configured vocabulary or more is kept whole.
     */
    private static void padEmbedding(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, int size,
                                     String what) throws IOException {
        long[] shape = src.shape();
        if (shape.length < 2) {
            throw new GgufFormatException("tensor '" + hfName + "' has the unusable shape " + Arrays.toString(shape));
        }
        long current = shape[0];
        RowSource data = current < size ? RowTransforms.padRows(src, size, 0.0f) : src;
        if (current < size) {
            ctx.log("Padding " + what + " shape " + Arrays.toString(shape) + " from " + current + " to " + size
                    + " (adding " + (size - current) + " vision/audio token slots)");
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /** The per-load AltUp accumulator: one slot per id, in id order. */
    private static final class Altup {
        final boolean unembed;
        final Map<Integer, RowSource> slots = new LinkedHashMap<>(ALTUP_SLOTS);

        Altup(boolean unembed) {
            this.unembed = unembed;
        }
    }

    /** The two accumulators of one load ({@code __init__}, gemma.py:495-507). */
    private static final class AltupState {
        Altup projections;
        Altup unembed;
    }

    /**
     * The id of one AltUp slot tensor: the {@code .<id>.} member the converter keys on (gemma.py:601-608, whose
     * unknown names raise).
     */
    private static int slotOf(String hfName, boolean unembed) {
        java.util.regex.Matcher m = SLOT.matcher(hfName);
        if (!m.find()) {
            throw new GgufFormatException("tensor '" + hfName + "' names no " + (unembed ? UNEMBED_MARKER
                    : LAND_MARKER) + " slot");
        }
        return Integer.parseInt(m.group(1));
    }

    private static final java.util.regex.Pattern SLOT = java.util.regex.Pattern.compile("\\.(\\d+)\\.");

    /**
     * One AltUp family tensor: each slot's matrix stored and consumed by its id (the converter's 0, 1 and 2
     * slots of gemma.py:601-610; unknown ids raise), and the stacked {@code [3, <rows>, <cols>]} tensor planned
     * under the family's GGUF name in id order when the last slot arrived ({@code _stack_matrices},
     * gemma.py:552-557, which yields only when every slot has content).
     */
    private static void altup(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean unembed)
            throws IOException {
        AltupState state = ctx.state(ALTUP_STATE, AltupState.class, AltupState::new);
        Altup a = unembed ? state.unembed : state.projections;
        if (a == null) {
            a = new Altup(unembed);
            if (unembed) {
                state.unembed = a;
            } else {
                state.projections = a;
            }
        }
        int slot = slotOf(hfName, unembed);
        if (slot < 0 || slot >= ALTUP_SLOTS) {
            throw new GgufFormatException("tensor '" + hfName + "' is the unknown " + (unembed ? UNEMBED_MARKER
                    : LAND_MARKER) + " slot " + slot);
        }
        if (a.slots.containsKey(slot)) {
            throw new GgufFormatException("tensor '" + hfName + "' is a repeated " + (unembed ? UNEMBED_MARKER
                    : LAND_MARKER) + " slot " + slot);
        }
        out.consume(hfName);
        a.slots.put(slot, src);
        if (a.slots.size() < ALTUP_SLOTS) {
            return;
        }
        List<RowSource> stacked = new ArrayList<>(ALTUP_SLOTS);
        for (int s = 0; s < ALTUP_SLOTS; s++) {
            stacked.add(a.slots.get(s));
        }
        String merged = "model." + (unembed ? UNEMBED_MARKER : LAND_MARKER) + ".weight";
        out.add(ctx.names().mapOrThrow(merged), RowTransforms.concatRows(stacked));
    }

    /** The Gemma 3 wide path with 3n's zero norm shift, and no truncation step (gemma.py:629). */
    private static void planGemma3NoShift(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfGemmaHooks.plan(ctx, hfName, src, out, NO_NORM_SHIFT);
    }
}
