package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Map;

/**
 * The hooks of llama.cpp's {@code GptOssModel} (LC/conversion/gpt_oss.py:15-130): the attention-sink scaling,
 * metadata, tiled gate/up output and vocabulary. The MXFP4 experts of QR-2 (quant/Fp4Scheme's gpt-oss pairing,
 * Fp4Scheme.java:318-359) arrive as {@code [E, R, K]} sources from the {@code _blocks}/{@code _scales} pairs, so the
 * gate/up rows are de-interleaved ({@code data[:, ::2]} / {@code [:, 1::2]} over the fused rows,
 * gpt_oss.py:100-104, 113-118); a BF16 {@code gate_up_proj [E, D, 2F]} is split into gate and up as
 * {@code transposeLast2(interleaveCols(x, 2, part))}, which equals the converter's transpose to {@code [E, 2F, D]}
 * followed by the row de-interleave {@code [:, ::2, :]} (gpt_oss.py:116-119); a BF16 {@code down_proj [E, D, F]} is
 * transposed (gpt_oss.py:73-74). The {@code [E, 2F]} biases are de-interleaved through
 * {@link ColumnTransforms#interleaveCols} (gpt_oss.py:110-112) and stay F32 by {@code DtypePolicy}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfGptOssHooks {

    private static final String WEIGHT = ".weight";
    private static final String BIAS = ".bias";
    private static final String GATE_UP = "gate_up_proj";
    private static final String DOWN = "down_proj";
    private static final String SINKS = "sinks";

    private HfGptOssHooks() {
    }

    // ------------------------------------------------------------------ architecture and metadata

    /** The {@model_arch} of the class: GPT_OSS sources resolve to NeuralFree's openai-moe architecture. */
    public static Arch arch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        return Arch.OPENAI_MOE;
    }

    /**
     * {@code GptOssModel.set_gguf_parameters} (gpt_oss.py:122-130) after the generic metadata: the sliding window and
     * the expert feed-forward length
     * (models/openai-moe.cpp:5-6 read them).
     *
     * @throws GgufFormatException when a required value is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code GptOssModel.modify_tensors} with the class's {@code filter_tensors} sink rename
     * (gpt_oss.py:55-119): {@code sinks} gains {@code .weight}; a {@code down_proj} bias keeps its name, a float
     * {@code down_proj} is transposed and named, an MXFP4 one is named as stored; the fused {@code gate_up_proj} is
     * split over the gate/up interleaved rows (MXFP4) or transposed and split columns (float); the biases are
     * de-interleaved; every other tensor maps unchanged.
     *
     * @throws GgufFormatException when a fused tensor's split does not fit its shape
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.contains(SINKS) && !name.endsWith(WEIGHT) && !name.endsWith(BIAS)) {
            name = name + WEIGHT;
        }
        if (name.contains(DOWN)) {
            down(ctx, name, src, out);
            return;
        }
        if (name.contains(GATE_UP)) {
            gateUp(ctx, name, src, out);
            return;
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /** {@code down_proj} (gpt_oss.py:66-74): the bias keeps its name, a float stack is transposed, MXFP4 stays. */
    private static void down(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        String mapped = name.endsWith(BIAS) ? name.replace(DOWN + "_bias", DOWN + BIAS)
                : name.endsWith(WEIGHT) ? name : name + WEIGHT;
        RowSource data = src.format().isFloat() && src.shape().length > 1 && !name.endsWith(BIAS)
                ? ExpertTransforms.transposeLast2(src) : src;
        out.add(ctx.names().mapOrThrow(mapped), data);
    }

    /**
     * The fused {@code gate_up_proj} (gpt_oss.py:107-118): the MXFP4 logical rows {@code [E, 2F, K]} split
     * {@code 0::2} into gate and {@code 1::2} into up; a float {@code [E, D, 2F]} split through transposed and
     * de-interleaved columns; a bias {@code [E, 2F]} de-interleaved into two {@code [E, F]} rows.
     */
    private static void gateUp(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        if (name.endsWith(BIAS)) {
            biased(ctx, name, src, out);
            return;
        }
        String gate = name.replace(GATE_UP, "gate_proj" + (name.endsWith(WEIGHT) ? "" : WEIGHT));
        String up = name.replace(GATE_UP, "up_proj" + (name.endsWith(WEIGHT) ? "" : WEIGHT));
        if (src.format().isFloat()) {
            out.add(ctx.names().mapOrThrow(gate),
                    ExpertTransforms.transposeLast2(ColumnTransforms.interleaveCols(src, 2, 0)));
            out.add(ctx.names().mapOrThrow(up),
                    ExpertTransforms.transposeLast2(ColumnTransforms.interleaveCols(src, 2, 1)));
            return;
        }
        long[] shape = src.shape();
        long rowsPerExpert = shape[1];
        if (rowsPerExpert % 2 != 0 || rowsPerExpert <= 0) {
            throw new GgufFormatException("tensor '" + name + ": the fused gate/up row width " + rowsPerExpert
                    + " is not an even split");
        }
        long e = shape[0];
        long f = rowsPerExpert / 2;
        long k = shape[shape.length - 1];
        RowSource flat = ElementTransforms.reshape(src, Math.multiplyExact(e, rowsPerExpert), k);
        out.add(ctx.names().mapOrThrow(gate),
                ElementTransforms.reshape(RowTransforms.interleaveRows(flat, 2, 0), e, f, k));
        out.add(ctx.names().mapOrThrow(up),
                ElementTransforms.reshape(RowTransforms.interleaveRows(flat, 2, 1), e, f, k));
    }

    /** The {@code [E, 2F]} biases, gate rows {@code [i::2]} and up rows {@code [i+1::2]} (gpt_oss.py:110-112). */
    private static void biased(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        String base = name.replace(GATE_UP + "_bias", "gate_proj" + BIAS);
        out.add(ctx.names().mapOrThrow(base), ColumnTransforms.interleaveCols(src, 2, 0));
        out.add(ctx.names().mapOrThrow(name.replace(GATE_UP + "_bias", "up_proj" + BIAS)),
                ColumnTransforms.interleaveCols(src, 2, 1));
    }

    // ------------------------------------------------------------------ vocabulary

    /** {@code GptOssModel.set_vocab} (gpt_oss.py:121): the GPT-2 BPE vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }
}
