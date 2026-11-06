package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * Hooks of vLLM's InternS1-Pro checkpoint (interns1_pro.py): the Qwen3-VL MoE text backbone with the Fourier
 * position embedding. The RoPE metadata follows {@code InternS1ProMoeDecoderLayer.__init__} (:383-392), which reads
 * {@code fope_init_factor}, {@code fope_sep_head} and {@code num_inv_freq} from the rope parameters and enables FoPE
 * when any of them is present. The learned coefficient matrices are model-level tensors the rotation reads; the
 * checkpoint stores one pair under the text model's {@code rotary_emb}.
 */
public final class HfInternS1ProHooks {

    /** The configuration key of the grouped router (vLLM interns1_pro.py:169). */
    private static final String ROUTER_N_GROUPS = "router_n_groups";
    private static final String COS_COEF = "rotary_emb.cos_coef";
    private static final String SIN_COEF = "rotary_emb.sin_coef";
    private static final String TIME_SERIES = "model.time_series.";

    private HfInternS1ProHooks() {
    }

    /**
     * The grouped router the configuration declares as {@code router_n_groups} (vLLM's InternS1-Pro name for
     * {@code n_group}, interns1_pro.py:169): copied to {@code n_group} and {@code topk_group} so the generic
     * metadata writes the group counts, and every group contributes its share of the selected experts.
     *
     * @throws GgufFormatException naming {@code decoder_sparse_step} or {@code mlp_only_layers} on a checkpoint that
     *                             mixes dense and expert blocks, which the Qwen3-VL MoE text graph does not define
     */
    public static HfConfig fixup(HfConfig cfg) {
        OptionalInt step = cfg.findInt("decoder_sparse_step");
        if (step.isPresent() && step.getAsInt() != 1) {
            throw new GgufFormatException("config.json: 'decoder_sparse_step' = " + step.getAsInt()
                    + " makes InternS1-Pro alternate dense and expert blocks; the Qwen3-VL MoE text graph builds"
                    + " expert FFNs on every block");
        }
        Object only = cfg.flat().get("mlp_only_layers");
        if (only instanceof List && !((List<?>) only).isEmpty()) {
            throw new GgufFormatException("config.json: 'mlp_only_layers' " + only + " selects dense blocks in an"
                    + " InternS1-Pro checkpoint; the Qwen3-VL MoE text graph builds expert FFNs on every block");
        }
        OptionalInt groups = cfg.findInt(ROUTER_N_GROUPS);
        if (!groups.isPresent() || groups.getAsInt() <= 0) {
            return cfg;
        }
        Map<String, Object> fix = new LinkedHashMap<>();
        fix.put("n_group", groups.getAsInt());
        fix.put("topk_group", groups.getAsInt());
        return cfg.withFlat(fix);
    }

    /**
     * {@code Qwen3VLMoeTextModel.set_gguf_parameters} plus the FoPE keys vLLM's decoder layer derives from the rope
     * parameters and the per-group router flag.
     *
     * @throws IOException when a side file the inherited metadata reads cannot be read
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfMoeClassicHooksD.qwen3VlMoeMeta(ctx, kv, plan);
        HfRope.Params rope = ctx.config().rope().global();
        kv.set(GgufKeys.ROPE_FOPE_USE, rope.useFope());
        kv.set(GgufKeys.ROPE_FOPE_SEP_HEAD, rope.fopeSepHead().orElse(Boolean.FALSE));
        kv.set(GgufKeys.ROPE_FOPE_NUM_INV_FREQ, rope.fopeNumInvFreq().orElse(0));
        kv.set(GgufKeys.ROPE_FOPE_INIT_FACTOR, rope.fopeInitFactor().orElse(0.0));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, ctx.config().findBool(true, "norm_topk_prob"));
        OptionalInt groups = ctx.config().findInt(ROUTER_N_GROUPS);
        kv.set(GgufKeys.EXPERT_PER_GROUP_TOP_K, groups.isPresent() && groups.getAsInt() > 0);
    }

    /**
     * The Fourier coefficient matrices stay F32 under every output type: vLLM computes their projection in float and
     * they are the rotation's own arithmetic, so a quantized copy would change every angle of the model.
     *
     * @return {@code F32} for the two coefficient tensors, null for every other tensor
     */
    public static GGMLType forceType(String ggufName, int nDims, ElementFormat source,
                                     ModelSourceOptions.OutType outType) {
        return TensorNames.weight(TensorNames.ROPE_COS_COEF).equals(ggufName)
                || TensorNames.weight(TensorNames.ROPE_SIN_COEF).equals(ggufName) ? GGMLType.F32 : null;
    }

    /**
     * {@code Qwen3VLMoeTextModel.modify_tensors} with the Fourier coefficient matrices, which the text name map does
     * not cover: they become the model-level {@code rope_cos_coef}/{@code rope_sin_coef} tensors.
     *
     * @throws GgufFormatException naming a time-series tensor, the one module of the checkpoint's family the engine
     *                             has no graph for
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (hfName.startsWith(TIME_SERIES)) {
            throw new GgufFormatException("tensor '" + hfName + "': InternS1-Pro's time-series tower has no"
                    + " NeuralFree graph; the text model and the vision tower (through --mmproj) are the supported"
                    + " parts of the checkpoint");
        }
        if (hfName.endsWith(COS_COEF)) {
            out.add(TensorNames.weight(TensorNames.ROPE_COS_COEF), src);
            return;
        }
        if (hfName.endsWith(SIN_COEF)) {
            out.add(TensorNames.weight(TensorNames.ROPE_SIN_COEF), src);
            return;
        }
        HfMoeClassicHooksD.qwen3VlMoeTensors(ctx, hfName, src, out);
    }
}
