package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Hooks of the plain state-space-model converter classes (LC/conversion/mamba.py): {@code MambaModel}
 * (mamba.py:18-100) and {@code Mamba2Model} (mamba.py:105-200), plus the shared helpers of the SSM sub-family
 * ({@code FalconH1Model}, {@code JambaModel} and {@code Plamo2Model} live in {@link HfSsmHooksA}).
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}. The {@code mamba_}-prefixed hyperparameter lookups ({@link #prefixedInt}) serve Falcon-H1 and
 * GraniteHybrid.</p>
 */
public final class HfSsmHooks {

    /** {@link #mamba2Filter} as an {@link HfNameFilter}, for entries and subclass filters that wrap it. */
    public static final HfNameFilter MAMBA2_FILTER = HfSsmHooks::mamba2Filter;

    /** The arbitrary context length of a plain Mamba / Mamba-2 converter (mamba.py:63, :155). */
    static final long DEFAULT_CONTEXT_LENGTH = 1L << 20;
    /** The suffix a weight tensor carries; the SSM A and D vectors have none. */
    static final String WEIGHT = ".weight";
    /** The per-load state holding the token embedding seen first, for the tied-output drop. */
    private static final String TOK_EMBD_STATE = HfSsmHooks.class.getName() + ".tokEmbd";
    /** The per-expert weight parts a Jamba layer merges, in the converter's order (jamba.py:84). */
    static final List<String> JAMBA_EXPERT_PARTS = Arrays.asList("down_proj", "gate_proj", "up_proj");

    private HfSsmHooks() {
    }

    // ------------------------------------------------------------------ shared helpers

    /**
     * A prefixed {@code find_hparam} (falcon_h1.py:47-55, granite.py:480-487): every alias of {@code keys} is
     * searched first, then each of its {@code <prefix>_...}-joined spellings; the first present value wins.
     */
    public static List<String> prefixedAliases(String prefix, String... keys) {
        List<String> out = new ArrayList<>(keys.length * 2);
        out.addAll(Arrays.asList(keys));
        for (String k : keys) {
            out.add(prefix + "_" + k);
        }
        return out;
    }

    /** The first present of {@code keys} and their {@code mamba_}-prefixed spellings, or {@code dflt}. */
    public static int prefixedInt(HfConfig cfg, int dflt, String... keys) {
        OptionalDouble d = cfg.findFloat(prefixedAliases("mamba", keys).toArray(new String[0]));
        if (d.isPresent() && d.getAsDouble() == Math.rint(d.getAsDouble())) {
            return (int) d.getAsDouble();
        }
        return dflt;
    }

    /** The first present of {@code keys} and their {@code mamba_}-prefixed spellings, required. */
    public static int prefixedRequireInt(HfConfig cfg, String... keys) {
        OptionalDouble d = cfg.findFloat(prefixedAliases("mamba", keys).toArray(new String[0]));
        if (!d.isPresent()) {
            List<String> aliases = prefixedAliases("mamba", keys);
            throw new GgufFormatException(cfg.source() + ": required key " + aliases + " is missing");
        }
        return (int) d.getAsDouble();
    }

    /** {@code ceil(a / b)} of positive integers, as the converter's {@code -(a // -b)} (mamba.py:54). */
    static int ceilDiv(int a, int b) {
        return -Math.floorDiv(-a, b);
    }

    /** The padded vocab size of {@code v}: the next multiple of {@code multiple} at or above it. */
    private static long padVocab(long v, int multiple) {
        if (multiple < 1) {
            throw new GgufFormatException("config.json: 'pad_vocab_size_multiple' must be positive, got " + multiple);
        }
        return -Math.floorDiv(-v, multiple) * multiple;
    }

    /** {@code vocab_size} padded to the next multiple of {@code multiple}, when it is not already. */
    public static HfConfig padVocabSize(HfConfig cfg, int multiple) {
        long vocab = cfg.requireInt("vocab_size");
        long padded = padVocab(vocab, multiple);
        return padded == vocab ? cfg : cfg.withFlat(Map.of("vocab_size", padded));
    }

    /** The SSM B/C group count and, with it, the grouped norm width (Mamba-2 builds); required by the reshape. */
    static long[] groupShape(HfConfig cfg, long dInner, String configKey, String prefixedKey) {
        OptionalDouble nGroupD = cfg.findFloat(configKey, prefixedKey);
        long nGroup = nGroupD.isPresent() ? (long) nGroupD.getAsDouble() : 1L;
        if (nGroup < 1) {
            throw new GgufFormatException(cfg.source() + ": '" + configKey + "' " + nGroup
                    + " is not a positive group count");
        }
        if (dInner % nGroup != 0) {
            throw new GgufFormatException(cfg.source() + ": SSM inner size " + dInner
                    + " is not a multiple of the group count " + nGroup);
        }
        return new long[]{nGroup, dInner / nGroup};
    }

    // ------------------------------------------------------------------ MambaModel (mamba.py:18)

    /** {@link Arch#MAMBA} for every {@code MambaModel} checkpoint. */
    public static Arch mambaArch(HfConfig cfg, HfTensorIndex index) {
        return Arch.MAMBA;
    }

    /**
     * {@code MambaModel.set_vocab}'s padding (mamba.py:30-36): {@code vocab_size} rounded up to the next multiple of
     * {@code pad_vocab_size_multiple} (default 8), which the vocabulary writers then size their token list to.
     */
    public static HfConfig mambaFixup(HfConfig cfg) {
        return padVocabSize(cfg, cfg.findInt(8, "pad_vocab_size_multiple"));
    }

    /**
     * {@code MambaModel.set_gguf_parameters} (mamba.py:46-74): the Mamba-1 sizes with the defaults {@code d_conv} 4,
     * {@code d_inner} {@code 2 * d_model}, {@code d_state} 16 and {@code dt_rank} {@code ceil(d_model / 16)}, no
     * attention, and the arbitrary {@code 2^20} context. The B/DT/C RMS flag is set for the {@code falcon_mamba}
     * model type (:58-59). The class does not extend {@code set_gguf_parameters}, so only these keys are written.
     *
     * @throws GgufFormatException when the expansion factor is not 2 (the converter asserts, mamba.py:61)
     */
    public static void mambaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        long dModel = cfg.requireInt("hidden_size", "d_model");
        long dInner = cfg.findInt(0, "intermediate_size", "d_inner");
        if (dInner == 0) {
            dInner = 2L * dModel;
        }
        if (dInner != 2L * dModel) {
            throw new GgufFormatException(cfg.source() + ": d_inner " + dInner + " differs from 2 * d_model "
                    + dModel + "; only an expansion factor of 2 is supported (mamba.py:61)");
        }
        long dtRank = cfg.findInt(0, "time_step_rank", "dt_rank");
        if (dtRank == 0) {
            dtRank = ceilDiv((int) dModel, 16);
        }
        OptionalDouble eps = cfg.findFloat("layer_norm_epsilon", "rms_norm_eps");
        kv.set(GgufKeys.CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH);
        kv.set(GgufKeys.EMBEDDING_LENGTH, dModel);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 0L);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0L);
        kv.set(GgufKeys.BLOCK_COUNT, (long) ctx.blockCount());
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) cfg.findInt(4, "conv_kernel", "d_conv"));
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) cfg.findInt(16, "state_size", "d_state"));
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, dtRank);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps.isPresent() ? eps.getAsDouble() : 1e-5);
        kv.set(GgufKeys.SSM_DT_B_C_RMS, "falcon_mamba".equals(cfg.modelType()));
    }

    /**
     * {@code MambaModel.modify_tensors} (mamba.py:78-100): {@code .A_log} becomes {@code -exp(.A_log)}, the
     * convolution kernel is squeezed, and the output is dropped when it equals the token embedding byte for byte
     * (:93-96, one per load); every other tensor is mapped.
     */
    public static void mambaTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = hfName.endsWith(".A_log") ? ElementTransforms.negExp(src) : src;
        String gguf = ctx.names().mapOrThrow(hfName);
        int bid = HfLoadContext.bidOf(hfName);
        if (ctx.names().matches(gguf, TensorNames.SSM_CONV1D, bid, WEIGHT)) {
            s = ElementTransforms.squeezeConv(s);
        }
        AtomicReference<RowSource> tokEmbd = ctx.state(TOK_EMBD_STATE, AtomicReference.class, AtomicReference::new);
        if (tokEmbd.get() != null) {
            if (ctx.names().matches(gguf, TensorNames.OUTPUT, -1, WEIGHT) && equalValues(tokEmbd.get(), s)) {
                out.drop(hfName, "equals token_embd byte for byte (mamba.py:95)");
                return;
            }
        } else if (ctx.names().matches(gguf, TensorNames.TOKEN_EMBD, -1, WEIGHT)) {
            tokEmbd.set(s);
        }
        out.add(gguf, s);
    }

    /** The {@code torch.equal} test: same shape and element format, and the same decoded values. */
    static boolean equalValues(RowSource a, RowSource b) throws IOException {
        if (!Arrays.equals(a.shape(), b.shape()) || !a.format().equals(b.format())) {
            return false;
        }
        long bytes = a.rows() * a.rowBytes();
        if (bytes != b.rows() * b.rowBytes() || bytes > Integer.MAX_VALUE - 8) {
            return false;
        }
        ByteBuffer ba = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        a.readRows(0, (int) a.rows(), ba);
        ByteBuffer bb = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        b.readRows(0, (int) b.rows(), bb);
        return ba.equals(bb);
    }

    /**
     * {@code MambaModel.set_vocab} (mamba.py:29-44): GPT-2 BPE for {@code tokenizer.json}, SentencePiece for
     * {@code tokenizer.model}, else the builtin GPT-NeoX vocabulary at the padded size when no tokenizer file is
     * present (mamba.py:44).
     */
    public static Map<String, Object> mambaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return ssmVocab(ctx, pre, 8, true);
    }

    /** {@code Mamba2Model.set_vocab} (mamba.py:128-141): SentencePiece first, then the {@code .v3} refusal. */
    public static Map<String, Object> mamba2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return ssmVocab(ctx, pre, 16, false);
    }

    /**
     * The shared Mamba-family vocabulary chain: the padded vocabulary of {@code padMultiple}, then —
     * {@code tokenizer.json} first for the Mamba classes ({@code jsonFirst}, mamba.py:29-36) or SentencePiece
     * first for Mamba-2 and hybrids (mamba.py:128-141) — the SentencePiece writer, the restricted
     * {@code tokenizer.model.v3} refusal (Mamba-2 subclasses only), the GPT-2 writer, and finally the builtin
     * GPT-NeoX vocabulary at the padded size when no tokenizer file is present (mamba.py:44, :141).
     */
    public static Map<String, Object> ssmVocab(HfLoadContext ctx, PreNameResolver pre, int padMultiple,
                                               boolean jsonFirst) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        if (ctx.tokenizerFiles().hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(ctx.tokenizerFiles());
        }
        if (ctx.tokenizerFiles().tokenizerJson() != null && jsonFirst) {
            return HfVocabWriters.gpt2(in, pre);
        }
        if (ctx.tokenizerFiles().sentencePieceModel() != null) {
            return HfVocabWriters.sentencePiece(in, pre);
        }
        if (!jsonFirst && ctx.tokenizerFiles().file(HfTokenizerFiles.TOKENIZER_MODEL_V3) != null) {
            throw new GgufFormatException("Please rename " + ctx.tokenizerFiles().file(HfTokenizerFiles.TOKENIZER_MODEL_V3)
                    + " to tokenizer.model (mamba.py:136)");
        }
        if (ctx.tokenizerFiles().tokenizerJson() != null) {
            return HfVocabWriters.gpt2(in, pre);
        }
        // No tokenizer files: the builtin GPT-NeoX tokenizer (mamba.py:44/:141)
        return HfVocabWriters.builtin(in, "gpt-neox",
                (int) padVocab(ctx.config().requireInt("vocab_size"), padMultiple));
    }

    // ------------------------------------------------------------------ Mamba2Model (mamba.py:105)

    /** {@link Arch#MAMBA2} for every {@code Mamba2Model} checkpoint. */
    public static Arch mamba2Arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.MAMBA2;
    }

    /**
     * {@code Mamba2Model.__init__}'s vocabulary padding (mamba.py:126-129): {@code pad_vocab_size_multiple} 16.
     */
    public static HfConfig mamba2Fixup(HfConfig cfg) {
        return padVocabSize(cfg, cfg.findInt(16, "pad_vocab_size_multiple"));
    }

    /** The Mamba-2 model width of {@code cfg}: {@code hidden_size}, {@code d_model} or {@code dim}. */
    private static long mamba2ModelWidth(HfConfig cfg) {
        return cfg.requireInt("hidden_size", "d_model", "dim");
    }

    /** The Mamba-2 inner size of {@code cfg}: {@code mamba_d_ssm}, {@code inter…}, or {@code expand * d_model}. */
    static long mamba2InnerSize(HfConfig cfg) {
        long dModel = mamba2ModelWidth(cfg);
        long dInner = cfg.findInt(0, "mamba_d_ssm", "intermediate_size", "d_inner");
        if (dInner == 0) {
            long expand = peekExpand(cfg);
            dInner = expand * dModel;
        }
        return dInner;
    }

    private static long peekExpand(HfConfig cfg) {
        return cfg.findFloat("mamba_expand", "expand").isPresent() ? cfg.findInt("mamba_expand", "expand").getAsInt()
                : 2L;
    }

    /**
     * {@code Mamba2Model.set_gguf_parameters} (mamba.py:143-166): the Mamba-2 sizes with the defaults {@code d_conv}
     * 4, {@code d_state} 128 and {@code head_dim} 64, {@code ssm_dt_rank = d_inner / head_dim}, the group count, and
     * the arbitrary {@code 2^20} context. Like Mamba the class does not extend the generic metadata.
     *
     * @throws GgufFormatException when {@code d_inner} differs from {@code expand * d_model} or is not a whole
     *                             number of head widths (mamba.py:152-153)
     */
    public static void mamba2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        long dModel = mamba2ModelWidth(cfg);
        long dInner = mamba2InnerSize(cfg);
        long expand = peekExpand(cfg);
        long headDim = cfg.findInt(64, "mamba_d_head", "head_dim");
        if (dInner != expand * dModel) {
            throw new GgufFormatException(cfg.source() + ": d_inner " + dInner + " is not mamba_expand " + expand
                    + " times d_model " + dModel + " (mamba.py:152)");
        }
        if (dInner % headDim != 0) {
            throw new GgufFormatException(cfg.source() + ": d_inner " + dInner + " is not a multiple of head_dim "
                    + headDim + " (mamba.py:153)");
        }
        long[] group = groupShape(cfg, dInner, "n_groups", null);
        OptionalDouble eps = cfg.findFloat("layer_norm_epsilon", "rms_norm_eps");
        kv.set(GgufKeys.CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH);
        kv.set(GgufKeys.EMBEDDING_LENGTH, dModel);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 0L);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0L);
        kv.set(GgufKeys.BLOCK_COUNT, (long) ctx.blockCount());
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) cfg.findInt(4, "conv_kernel", "d_conv"));
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) cfg.findInt(128, "state_size", "d_state"));
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, dInner / headDim);
        kv.set(GgufKeys.SSM_GROUP_COUNT, group[0]);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps.isPresent() ? eps.getAsDouble() : 1e-5);
    }

    /**
     * {@code Mamba2Model.filter_tensors} (mamba.py:169-179): the Codestral {@code model.backbone*} and
     * {@code model.lm_head*} prefixes drop their {@code model.}, and every {@code .dt_bias} tail becomes
     * {@code .dt_proj.bias}.
     */
    public static String mamba2Filter(String name, java.util.function.UnaryOperator<String> base) {
        String out = name;
        if (out.startsWith("model.backbone") || out.startsWith("model.lm_head")) {
            out = out.substring("model.".length());
        }
        if (out.endsWith(".dt_bias")) {
            out = out.substring(0, out.length() - ".dt_bias".length()) + ".dt_proj.bias";
        }
        return base.apply(out);
    }

    /**
     * {@code Mamba2Model.modify_tensors} (mamba.py:181-200): the conv squeeze, the {@code ssm_a}/{@code ssm_d}
     * trailing-1 reshapes (stored F32 through their unsuffixed GGUF names, DtypePolicy step 0), the grouped-norm
     * reshape {@code (n_group, d_inner/n_group)}, and {@code A_log} → {@code -exp}, in the converter's order.
     */
    public static void mamba2Tensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String gguf = ctx.names().mapOrThrow(hfName);
        int bid = HfLoadContext.bidOf(hfName);
        RowSource s = src;
        if (ctx.names().matches(gguf, TensorNames.SSM_CONV1D, bid, WEIGHT)) {
            s = ElementTransforms.squeezeConv(s);
        } else if (ctx.names().matches(gguf, TensorNames.SSM_A, bid, "")
                || ctx.names().matches(gguf, TensorNames.SSM_D, bid, "")) {
            s = reshapeAppendOne(s);
        } else if (ctx.names().matches(gguf, TensorNames.SSM_NORM, bid, WEIGHT)) {
            s = ElementTransforms.reshape(s, groupShape(ctx.config(), mamba2InnerSize(ctx.config()),
                    "n_groups", null));
        }
        if (hfName.endsWith(".A_log")) {
            s = ElementTransforms.negExp(s);
        }
        out.add(gguf, s);
    }

    /** {@code data_torch.reshape((*shape, 1))} (mamba.py:192): a new trailing 1. */
    static RowSource reshapeAppendOne(RowSource src) {
        long[] shape = src.shape();
        long[] out = Arrays.copyOf(shape, shape.length + 1);
        out[out.length - 1] = 1L;
        return ElementTransforms.reshape(src, out);
    }
}

