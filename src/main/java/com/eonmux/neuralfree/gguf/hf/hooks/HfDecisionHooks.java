package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The Qwen3.5 decision-model classes (LC/conversion): {@code PplxDeciderModel} (pplx_decider.py:18-105),
 * {@code VllmD3Model} (vllm_d3.py:12-60), {@code OpenJevModel} (qwen.py:684-742), {@code H2OLightningModel}
 * (h2o_lightning.py:13-96) and the plain-LoRA adapters {@code LevModel}, {@code KevModel} and {@code NimbleModel}
 * (lev.py:17-280). Every class is architecture QWEN35 and builds through the {@link HfQwenLinearHooks} Qwen3.5
 * chain; its own part is the decision type, the per-type temperatures and labels, the systemone chat template,
 * and the readout:
 *
 * <ul>
 *   <li>{@code PplxDeciderModel}: the {@code readout.safetensors} side file becomes {@code output.weight}, one row
 *       per option label on zero rows at the label token ids;</li>
 *   <li>{@code VllmD3Model}: the same readout, and the non-causal full attention of the Decision 3.0 format;</li>
 *   <li>{@code OpenJevModel} and {@code H2OLightningModel}: fixed temperatures, and the template texts and labels
 *       of {@code serve_config.json};</li>
 *   <li>the LoRA classes: the adapter merges at load as {@code W + (alpha/r)·B·A} over the local base checkpoint
 *       (never a download; {@code ctx.resources()} keeps it open), and Kev's pointer head is {@code cat(q, k)} of
 *       its training head, a {@code torch} pickle {@code head.pt} that NeuralFree does not unpickle (PAR FX-1).</li>
 * </ul>
 */
public final class HfDecisionHooks {

    private static final String SYSTEMONE = "systemone";
    private static final String[] TEMPERATURE_TYPES = {"choice", "score", "noul"};

    private HfDecisionHooks() {
    }

    /** A resolver to {@link Arch#QWEN35} (every class sets {@code model_arch} QWEN35). */
    public static HfArchResolver arch() {
        return HfArchResolver.of(Arch.QWEN35);
    }

    /** {@code no_mtp} of every decision class: the base model's {@code mtp.*} tensors are dropped
     *  (lev.py:42, pplx_decider.py:41, qwen.py:702). */
    public static HfNameFilter noMtpFilter() {
        return (name, base) -> {
            String t = base.apply(name);
            if (t == null) {
                return null;
            }
            for (String part : t.split("\\.", -1)) {
                if (part.equals("mtp")) {
                    return null;
                }
            }
            return t;
        };
    }

    // ------------------------------------------------------------------ PplxDeciderModel / VllmD3Model

    /** {@code PplxDeciderModel.set_gguf_parameters} (pplx_decider.py:75-79): {@code pplx-decider} with one
     *  temperature per type from {decision_config.json. */
    public static void pplxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        readoutMeta(ctx, kv, plan, "pplx-decider");
    }

    /** {@code PplxDeciderModel.modify_tensors}: the Qwen3.5 chain, with the checkpoint's {lm_head} dropped —
     *  the readout of {@link #pplxExtra} replaces it as the output layer. */
    public static void readoutTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith("lm_head.weight")) {
            out.drop(hfName, "the readout lm_head of decision_config.json token ids replaces the checkpoint's");
            return;
        }
        HfQwenLinearHooks.vReorderTensors(ctx, hfName, src, out);
    }

    /** {@code PplxDeciderModel.generate_extra_tensors} (pplx_decider.py:81-92): the {readout.safetensors}
     *  {weight} rows scattered onto the {token_ids} ofDecision_config.jsonExtra on zero rows, stored as
     *  {output.weight}; the readout must hold exactly one row per distinct label token. */
    public static void pplxExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        StTensor readout = Readout.tensor(ctx);
        List<Integer> ids = HfDecisionSupport.Decisions.distinctInts(HfDecisionSupport.Decisions.map(decisionConfig(ctx).get("token_ids"),
                "decision_config.json token_ids"), "decision_config.json token_ids");
        if (readout.shape().length != 2 || readout.shape()[0] != ids.size()) {
            throw new GgufFormatException(cfg.source() + ": " + Readout.FILE + " has "
                    + Arrays.toString(readout.shape()) + " rows, expected one per distinct token id ("
                    + ids.size() + ")");
        }
        int vocab = cfg.requireInt("vocab_size");
        RowSource lmHead = Readout.scatter(StRowSource.of(ctx.checkpoint().openSideFile(Readout.FILE), readout),
                ids, vocab);
        String ggufName = ctx.names().mapOrThrow(Readout.LM_HEAD);
        out.add(ggufName, lmHead);
        ctx.log("Decision readout stored as " + ggufName + " over " + vocab + " rows");
    }

    /** {@code VllmD3Model.__init__} (vllm_d3.py:45-49): prompt d3, non-causal full attention, last-token pooling;
     *  {@code set_gguf_parameters} (vllm_d3.py:51-54): full attention layers see the whole prompt. */
    public static void vllmD3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        Map<String, Object> decision = decisionConfig(ctx);
        check(decision, "prompt", "d3", cfg.source());
        check(decision, "attention_mode", "noncausal_full_attention", cfg.source());
        check(decision, "pooling", "last", cfg.source());
        readoutMeta(ctx, kv, plan, "vllm-d3");
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
    }

    public static void vllmD3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        readoutTensors(ctx, hfName, src, out);
    }

    public static void vllmD3Extra(HfLoadContext ctx, TensorSink out) throws IOException {
        pplxExtra(ctx, out);
    }

    // ------------------------------------------------------------------ OpenJevModel / H2OLightningModel

    /** {@code OpenJevModel.set_gguf_parameters} (qwen.py:730-736): the decision type {openjev} and the fixed
     *  temperatures {TEMPERATURE} (choice and score) and {TEMPERATURE} * {TEMPERATURE_NOUL} (noul). */
    public static void openjevMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        kv.set(GgufKeys.DECISION_TYPE, "openjev");
        double temperature = HfDecisionSupport.OpenJev.TEMPERATURE;
        kv.setAt(GgufKeys.DECISION_TEMPERATURE, temperature, "choice");
        kv.setAt(GgufKeys.DECISION_TEMPERATURE, temperature, "score");
        kv.setAt(GgufKeys.DECISION_TEMPERATURE, temperature * HfDecisionSupport.OpenJev.TEMPERATURE_NOUL, "noul");
    }

    public static void openjevTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfQwenLinearHooks.vReorderTensors(ctx, hfName, src, out);
    }

    /** {@code H2OLightningModel.set_gguf_parameters} (h2o_lightning.py:82-91): the decision type
     *  {h2o-lightning}, the labels of {serve_config.json} with their leading space, the per-type temperatures
     *  and the image temperature. */
    public static void h2oMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        Map<String, Object> serve = HfDecisionSupport.JsonObjects.readJsonObject(ctx.config().sideFile("serve_config.json"));
        HfDecisionSupport.Serve.check(serve, cfg.source());
        kv.set(GgufKeys.DECISION_TYPE, "h2o-lightning");
        kv.set(GgufKeys.DECISION_LABELS, HfDecisionSupport.Serve.labels(serve, cfg.source()));
        Map<String, Object> temperatures = HfDecisionSupport.Decisions.map(serve.get("temperature"), "serve_config.json temperature");
        for (Map.Entry<String, Object> e : temperatures.entrySet()) {
            kv.setAt(GgufKeys.DECISION_TEMPERATURE,
                    HfDecisionSupport.Decisions.number(e.getValue(), "serve_config.json temperature." + e.getKey()), e.getKey());
        }
    }

    public static void h2oTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        openjevTensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ LevModel / KevModel / NimbleModel

    /** {@code LevModel.set_gguf_parameters} (lev.py:139-148): the mode-A temperatures of the adapter's
     *  {calibration.json}; {choice:A:small} becomes {choice.small} because only the label readout is supported. */
    public static void levMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        loraMeta(ctx, kv, plan, "lev");
        HfConfig cfg = ctx.config();
        Map<String, Object> calibration = HfDecisionSupport.JsonObjects.readJsonObject(cfg.sideFile("calibration.json"));
        Map<String, Object> temperatures = HfDecisionSupport.Decisions.map(calibration.get("temperatures"),
                "calibration.json temperatures");
        for (Map.Entry<String, Object> e : temperatures.entrySet()) {
            String[] parts = e.getKey().split(":");
            if (parts.length < 2 || !HfDecisionSupport.Serve.ANNOTATION.equals(parts[1])) {
                continue;
            }
            List<String> key = new ArrayList<>();
            key.add(parts[0]);
            for (int i = 2; i < parts.length; i++) {
                key.add(parts[i]);
            }
            kv.setAt(GgufKeys.DECISION_TEMPERATURE,
                    HfDecisionSupport.Decisions.number(e.getValue(), "calibration.json temperatures." + e.getKey()),
                    String.join(".", key));
        }
    }

    /** {@code KevModel.__init__} and {set_gguf_parameters} (lev.py:173-201): the decision type {kev} and the
     *  output embedding length {2 * args.head_dim} of the training config; the head of {head.pt} is a pickle it
     *  never reads — reading it fails with the named pickle error, so a Kev checkpoint refuses here. */
    public static void kevMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        loraMeta(ctx, kv, plan, "kev");
        HfConfig cfg = ctx.config();
        Map<String, Object> training = HfDecisionSupport.JsonObjects.readJsonObject(cfg.sideFile("training_config.json"));
        Map<String, Object> args = HfDecisionSupport.Decisions.map(training.get("args"), "training_config.json args");
        long headDim = HfDecisionSupport.Decisions.longOf(args.get("head_dim"), "training_config.json args.head_dim");
        kv.set(GgufKeys.EMBEDDING_LENGTH_OUT, 2L * headDim);
        throw new GgufFormatException("Invalid Hugging Face checkpoint '" + cfg.dir() + "': head.pt is a PyTorch"
                + " pickle checkpoint, which NeuralFree does not support; use the model's safetensors files");
    }

    /** {@code NimbleModel.set_gguf_parameters} (lev.py:278-280): the decision type {nimble}. */
    public static void nimbleMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        loraMeta(ctx, kv, plan, "nimble");
    }

    /**
     * The LoRA classes' index (lev.py:40-85): the plain-LoRA assertion over {adapter_config.json}, the local base
     * checkpoint opened with {CheckpointContainers.open} and registered in {ctx.resources()}, and every base-model
     * tensor adjusted by its LoRA delta {scale} * B @ A. The adapter's {lora_A}/{lora_B} tensors leave the index;
     * a named error when one of them merged nowhere, mirroring prepare_tensors.
     */
    public static HfTensorIndex decisionLora(HfLoadContext ctx, HfTensorIndex index) throws IOException {
        HfConfig cfg = ctx.config();
        HfDecisionSupport.PlainLora.check(HfDecisionSupport.PlainLora.adapter(ctx), cfg);
        float scale = HfDecisionSupport.PlainLora.scale(HfDecisionSupport.PlainLora.adapter(ctx), cfg.source());
        Map<String, HfDecisionSupport.LoraPart[]> pieces = HfDecisionSupport.PlainLora.pieces(index, cfg);
        Path baseDir = HfDecisionSupport.RequireDir.baseDir(ctx);
        CheckpointContainer base = CheckpointContainers.open(baseDir, baseDir, false);
        ctx.resources().register(base);
        HfTensorIndex baseIndex = HfTensorIndex.of(base);
        Map<String, RowSource> merged = new LinkedHashMap<>();
        Set<String> mergedKeys = new HashSet<>();
        for (String publication : baseIndex.names()) {
            merged.put(publication, merged(baseIndex, publication, pieces, mergedKeys, scale));
        }
        if (mergedKeys.size() != pieces.size()) {
            throw new GgufFormatException(cfg.source() + ": only " + mergedKeys.size() + " of " + pieces.size()
                    + " LoRA tensors merged into the base model (lev.py:82-85)");
        }
        return HfTensorIndex.of(merged);
    }

    /**
     * The merge of one base-model publication (lev.py:57-63): when a LoRA pair targets it, the rows are
     * recomputed as {@code scale * (rows of B) @ (rows of A)} added to the base's and the pair is consumed;
     * otherwise the base tensor is carried over unchanged.
     */
    private static RowSource merged(HfTensorIndex base, String name, Map<String, HfDecisionSupport.LoraPart[]> pieces,
                                    Set<String> consumed, float scale) {
        int at = name.indexOf("layers.");
        HfDecisionSupport.LoraPart[] pair = at >= 0 ? pieces.get(name.substring(at)) : null;
        if (pair == null || pair[0].source == null || pair[1].source == null) {
            return base.get(name);
        }
        RowSource baseRows = base.get(name);
        try {
            return ElementTransforms.addLowRank(baseRows, pair[0].source, pair[1].source, scale);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("tensor '" + name + "': the LoRA does not fit the base model: "
                    + e.getMessage(), e);
        } finally {
            consumed.add(name.substring(at));
        }
    }

    /** The Qwen3.5 tensor chain over the merged base checkpoint (lev.py:63: the merge, then the parent). */
    public static void loraTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfQwenLinearHooks.vReorderTensors(ctx, hfName, src, out);
    }

    /** {@code _DecisionLoraMixin.prepare_tensors} has no extra tensors of its own (lev.py:81-87). */
    public static void decisionLoraExtra(HfLoadContext ctx, TensorSink out) {
    }

    // ------------------------------------------------------------------ the systemone templates

    public static Map<String, Object> pplxVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.PPLX);
    }

    public static Map<String, Object> vllmD3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.VLLM_D3);
    }

    public static Map<String, Object> openjevVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.OPENJEV);
    }

    public static Map<String, Object> levVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.LEV);
    }

    public static Map<String, Object> kevVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.KEV);
    }

    public static Map<String, Object> nimbleVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return withSystemone(ctx, pre, HfDecisionSupport.Templates.NIMBLE);
    }

    /** {@code h2o_lightning.py:58-80}: the served prompt layout with the {system} and {prefill} of
     *  {serve_config.json}. */
    public static Map<String, Object> h2oVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> serve = HfDecisionSupport.JsonObjects.readJsonObject(ctx.config().sideFile("serve_config.json"));
        Map<String, Object> prompt = HfDecisionSupport.Decisions.map(serve.get("prompt"), "serve_config.json prompt");
        return withSystemone(ctx, pre, HfDecisionSupport.Serve.systemone(
                String.valueOf(prompt.containsKey("system") ? prompt.get("system") : "\n").trim(),
                String.valueOf(prompt.get("prefill"))));
    }

    private static Map<String, Object> withSystemone(HfLoadContext ctx, PreNameResolver pre, String template)
            throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx).withFallbackChatTemplate(nullNamedSystemone(template)), pre);
    }

    /**
     * The fallback of {@code withSystemone}: the systemone-only classes write the one {@code systemone} template
     * these checkpoints carry (pplx_decider.py:47-49, lev.py:121-123 and its neighbours) as their chat template,
     * and the vocabulary writer stores it as the fallback.
     */
    private static String nullNamedSystemone(String template) {
        return template;
    }

    private static void readoutMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, String type)
            throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.DECISION_TYPE, type);
        Map<String, Object> decision = decisionConfig(ctx);
        double temperature = HfDecisionSupport.Decisions.number(decision.get("temperature"), "decision_config.json temperature");
        for (String name : TEMPERATURE_TYPES) {
            kv.setAt(GgufKeys.DECISION_TEMPERATURE, temperature, name);
        }
    }

    private static Map<String, Object> decisionConfig(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        Object v = cfg.find("decision").orElse(null);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key 'decision' (decision_config.json) is"
                    + " missing");
        }
        return (Map<String, Object>) v;
    }

    private static void loraMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, String type)
            throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        kv.set(GgufKeys.DECISION_TYPE, type);
    }

    private static void check(Map<String, Object> decision, String key, String want, String source) {
        Object got = decision.get(key);
        if (!want.equals(String.valueOf(got))) {
            throw new GgufFormatException(source + ": unsupported Decision 3.0 configuration: " + key + " is "
                    + got + ", expected " + want);
        }
    }

    // ------------------------------------------------------------------ the readout

    /** The readout side file of the PPLX decider (readout.safetensors → sparse lm_head rows). */
    private static final class Readout {
        static final String FILE = "readout.safetensors";
        static final String LM_HEAD = "lm_head.weight";

        static StTensor tensor(HfLoadContext ctx) throws IOException {
            SafetensorsFile f = ctx.checkpoint().openSideFile(FILE);
            StTensor t = f.tensor("weight");
            if (t == null) {
                throw new IOException("Invalid safetensors file '" + f.path() + "': no tensor 'weight'");
            }
            return t;
        }

        /** The rows of {readout} at the label ids over zero rows: {output.weight} (pplx_decider.py:85-92). */
        static RowSource scatter(RowSource readout, List<Integer> tokenIds, int vocab) {
            long inner = readout.shape()[readout.shape().length - 1];
            List<RowSource> parts = new ArrayList<>();
            LinkedHashMap<Integer, Integer> at = new LinkedHashMap<>();
            for (int i = 0; i < tokenIds.size(); i++) {
                Object log = tokenIds.get(i);
                at.put(tokenIds.get(i), i);
            }
            long pos = 0;
            for (Map.Entry<Integer, Integer> e : new java.util.TreeMap<>(at).entrySet()) {
                int token = e.getKey();
                if (token < pos) {
                    throw new IllegalArgumentException("readout token id " + token + " repeats");
                }
                if (token > pos) {
                    parts.add(zeros(token - pos, readout.format(), inner));
                }
                if (token >= vocab) {
                    throw new IllegalArgumentException("readout token id " + token
                            + " is not below the vocabulary of " + vocab);
                }
                parts.add(RowTransforms.sliceRows(readout, e.getValue(), 1));
                pos = token + 1L;
            }
            if (pos < vocab) {
                parts.add(zeros(vocab - pos, readout.format(), inner));
            }
            return RowTransforms.concatRows(parts);
        }

        private static RowSource zeros(long rows, ElementFormat format, long inner) {
            return new Zeros(rows, format, inner);
        }
    }

    /** {count} rows of zero bytes in one format, the filler of the sparse readout lm_head. */
    private static final class Zeros implements RowSource {
        private final long rows;
        private final ElementFormat format;
        private final long inner;
        private final byte[] zero;

        Zeros(long rows, ElementFormat format, long inner) {
            if (rows <= 0) {
                throw new IllegalArgumentException("zeros need rows > 0, got " + rows);
            }
            this.rows = rows;
            this.format = format;
            this.inner = inner;
            this.zero = new byte[(int) format.rowBytes(inner)];
        }

        @Override public String describe() { return "zeros"; }
        @Override public ElementFormat format() { return format; }
        @Override public long[] shape() { return new long[]{rows, inner}; }
        @Override public long rows() { return rows; }
        @Override public long rowBytes() { return zero.length; }
        @Override public boolean rowLocal() { return false; }
        @Override public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            for (int i = 0; i < count; i++) {
                dst.put(zero);
            }
        }
    }
}
