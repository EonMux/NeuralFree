package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

import java.util.Objects;

/**
 * Tiny, structurally faithful GGUF fixtures for the decoder and encoder architectures the engine runs, in the shape
 * the llama.cpp converters write and its loader reads. The builders live with their model families, which this class
 * aggregates: dense attention decoders in {@link SyntheticDenseModels}, routed experts in
 * {@link SyntheticMoeModels}, attention/SSM and attention/convolution hybrids in {@link SyntheticHybridModels}, SSM
 * decoders in {@link SyntheticRecurrentModels}, latent-attention and DSA-indexer models in
 * {@link SyntheticMlaDsaModels}, encoders in {@link SyntheticEncoderModels}, decision readouts in
 * {@link SyntheticDecisionModels} and {@link SyntheticD1Models}, the audio and drafting fixtures in
 * {@link SyntheticAudioModels} and the classifier heads in {@link SyntheticClassifiers}. The per-family classes
 * beyond this package's core are gemma ({@link SyntheticGemmaModels}), HF-converted checkpoints ({@link SyntheticHf}),
 * speculative drafting ({@link SyntheticEagle}, {@link SyntheticDflash}, {@link SyntheticGemma4Assistant}) and the
 * MTP, routed-expert and newer MLA families ({@link SyntheticMtpArchModels}, {@link SyntheticRoutedModels},
 * {@link SyntheticQwen4ExpModels}, {@link SyntheticMlaModels}).
 *
 * <p>This class keeps the shared fixture vocabulary every family uses: the entry points tests call, the token-type
 * and pooling constants, the chat templates, and the {@link Config} and {@link Tensors} kernel the builders share.
 * Each builder returns a fresh {@link GgufTestWriter} holding the metadata keys and value types the llama.cpp
 * converters write and every tensor the llama.cpp loader requires, with the loader's shapes ({@code ne[0]} first).
 * Tests may edit the writer before writing it, e.g. to drop a tensor or to replace a key with an array.
 *
 * <p>Values are deterministic per (config seed, tensor name). Projection matrices use
 * {@link Config#weightType()}: random values when the writer can encode floats into that type,
 * zero blocks otherwise. Norm weights are 1, norm biases 0, and SSM parameters F32 with
 * realistic signs (negative {@code ssm_a}, unit {@code ssm_d}). Every fixture carries a complete
 * tokenizer: SPM with byte fallback (llama, gemma2), byte-level BPE with merges (qwen3next,
 * mamba2) or WordPiece in phantom-space form (bert).
 */
public final class SyntheticModels {

    /** llama.cpp {@code llama_token_type} values. */
    public static final int TOKEN_TYPE_NORMAL = 1;
    public static final int TOKEN_TYPE_UNKNOWN = 2;
    public static final int TOKEN_TYPE_CONTROL = 3;
    public static final int TOKEN_TYPE_BYTE = 6;

    /** llama.cpp {@code LLAMA_POOLING_TYPE_MEAN}, written for the bert fixture. */
    public static final int POOLING_MEAN = 1;
    /** llama.cpp {@code LLAMA_POOLING_TYPE_RANK}, written for the rank fixture. */
    public static final int POOLING_RANK = 4;
    /** Token type count written for the bert fixture ("sequence A" / "sequence B"). */
    public static final int BERT_TOKEN_TYPES = 2;

    /** The t5 fixture's decoder start token, its {@code <pad>} (as the T5 checkpoints declare it). */
    public static final int T5_DECODER_START = 0;

    /** Geometry of {@link #wavtokenizer(int, int, int, int, int)}: the fixed test shape. */
    public static final int WAVTOK_E = 4;
    public static final int WAVTOK_C = 8;
    public static final int WAVTOK_FF = 16;
    public static final int WAVTOK_OUT = 10;
    public static final int WAVTOK_CODES = 260;
    public static final int WAVTOK_POSNET = 6;
    public static final int WAVTOK_CONVNEXT = 2;

    /**
     * The systemone template of the vllm-decision2 checkpoints (vllm_decision2.py:75-100), as the converter
     * writes it into {@code tokenizer.chat_template.systemone} (HfEncoderHooks.TEMPLATE_DECISION2).
     */
    public static final String VLLM_DECISION2_TEMPLATE =
            "{% for q in questions %}{% set d = o.description if o is defined else none %}"
            + "Context:\n{{ state | tojson }}\n\nTask type: {{ q.type }}\nQuestion:\n"
            + "{{ q.instructions | tojson }}\nOptions:"
            + "{% for o in q.options %}{{ sep }}\n<option>\n{{ sep }}{{ mark_option }}"
            + "{{ {'description': o.description, 'key': o.key} | tojson }}\n</option>{% endfor %}"
            + "{{ sep }}{{ mark_question }}\n\nSelect the single option best supported by the context and"
            + " instructions.\nDecision:{% endfor %}";

    /**
     * The systemone template of the label-family fixture: one question per prompt, its options with their labels,
     * and the answer cue the last prompt token ends on.
     */
    public static final String LABEL_TEMPLATE =
            "{{ type }}|{{ instructions }}|"
                    + "{% for o in options %}{{ o.label }}{{ o.key }}:{{ o.description }};{% endfor %}"
                    + "|{{ state }}|Answer:";

    /**
     * The kev systemone template: one question per prompt over a flattened state, every option ended on the
     * {@code <|box_end|>} token whose hidden state the kev read scores at (llama.cpp's {@code token_marker}).
     */
    public static final String KEV_TEMPLATE =
            "{{ type }}|{{ instructions }}|"
                    + "{% for o in options %}{{ o.key }}:{{ o.description }}<|box_end|>;{% endfor %}"
                    + "|{{ state }}|Answer:";

    /** The laya systemone template (bert.py:694-713): [cls] type question [sep] ([mask] option)* [sep] state [sep]. */
    public static final String LAYA_TEMPLATE =
            "[CLS]{{ type }} question: {{ instructions | tojson }}[SEP]"
                    + "{% for o in options %}[MASK] {% if type == 'choice' %}{{ o.key }}"
                    + "{% if o.description %}: {{ o.description | tojson }}{% endif %}"
                    + "{% elif type == 'score' %}level {{ o.key }}: {{ o.description | tojson }}"
                    + "{% else %}{{ o.key }}: {% if o.description %}{{ o.description | tojson }}"
                    + "{% elif o.key == 'true' %}yes, the statement holds"
                    + "{% else %}no, the statement does not hold{% endif %}{% endif %}{% endfor %}"
                    + "[SEP]{{ state | tojson }}[SEP]";

    /** The lfm2-d1 systemone template: the state, the question, the labelled options and the answer cue. */
    public static final String LFM2_D1_TEMPLATE =
            "<|startoftext|><|im_start|>user\n{{ state }}\n\nQUESTION:\n{{ instructions }}"
                    + "\n\nOptions:\n{% for o in options %}{{ o.label }} {{ o.key }}"
                    + "{% if o.description %}: {{ o.description }}{% endif %}\n{% endfor %}"
                    + "\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n";

    /**
     * The lfm2-d1-omni systemone template (D1OmniModel._systemone_template, lfm2.py:212-239, prompt.py of the model
     * repo): the media first, then the state, the question and the options, each marked piece cut to its token
     * budget by the server.
     */
    public static final String LFM2_D1_OMNI_TEMPLATE =
            "{% set ns = namespace(criteria=false) %}"
                    + "{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}"
                    + "{% endfor %}"
                    + "{% for image in images %}{{ image }}{% endfor %}{{ sep }}"
                    + "<|startoftext|><|reserved_7|>{{ sep }}{{ mark_state }}"
                    + "{% if state is string %}{{ state }}{% elif state is not none %}{{ state | tojson }}"
                    + "{% elif audio %}{}{% endif %}"
                    + "{{ sep }}{{ mark_question }}<|reserved_8|>"
                    + "{{ instructions if instructions is string else instructions | tojson }}"
                    + "{% for o in options %}{{ sep }}<|reserved_9|><|mask|>{{ sep }}{{ mark_option }} "
                    + "{% if type == 'choice' and audio %}option_{% if loop.index0 < 10 %}00"
                    + "{% elif loop.index0 < 100 %}0{% endif %}{{ loop.index0 }}: "
                    + "{% if o.description is not none and o.description != '' %}"
                    + "{{ o.description if o.description is string else o.description | tojson }}{% else %}{{ o.key }}"
                    + "{% endif %}"
                    + "{% elif type == 'choice' %}{{ o.key }}{% if o.description is not none and o.description"
                    + " != '' %}: {{ o.description if o.description is string else o.description | tojson }}"
                    + "{% endif %}"
                    + "{% elif type == 'score' %}level {{ o.key }}: "
                    + "{{ o.description if o.description is string else o.description | tojson }}"
                    + "{% elif audio %}{{ o.key }}: {{ 'yes' if o.key == 'true' else 'no' }}"
                    + "{% else %}{{ o.key }}: {% if o.description is not none and o.description != '' %}"
                    + "{{ o.description if o.description is string else o.description | tojson }}"
                    + "{% elif images and not ns.criteria %}{{ 'yes' if o.key == 'true' else 'no' }}"
                    + "{% elif o.key == 'true' %}yes, the statement holds"
                    + "{% else %}no, the statement does not hold{% endif %}{% endif %}"
                    + "{{ sep }}<|reserved_10|>{% endfor %}{{ sep }}<|reserved_11|>";

    /**
     * The h2o-lightning systemone template (h2o_lightning.py:62-80): the record, the question and the options of one
     * prompt in a chat turn, the option descriptions following the question type, the user turn trimmed by the
     * template, the answer following the "Answer:" prefill.
     */
    public static final String H2O_LIGHTNING_TEMPLATE =
            "<|im_start|>system\n<|im_end|>\n"
                    + "<|im_start|>user\n{% for image in images %}{{ image }}{% endfor %}"
                    + "{% set user %}record: {{ state }}\n"
                    + "question: {{ instructions if instructions else 'Answer the question below.' }}\n"
                    + "options:{% for o in options %}\n{{ o.label | trim }}) {{ o.key }}: "
                    + "{% if type == 'noul' %}{% if o.description %}{{ o.description }}"
                    + "{% elif o.key == 'true' %}the statement holds{% else %}it does not{% endif %}"
                    + "{% elif o.description is none %}{{ o.key }}{% else %}{{ o.description }}{% endif %}"
                    + "{% endfor %}{% endset %}{{ user | trim }}<|im_end|>\n"
                    + "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:";

    /** The minicpm-ocr-jev systemone template (MiniCPMOCRSeqClsModel._systemone_template, qwen.py:726-736): the
     * fixed OCR consistency question over the original image, the OCR text and its rendering. */
    public static final String MINICPM_OCR_JEV_TEMPLATE = "<|im_start|>user\n{{ images[0] }}\nOCR文本：\n{{ state }}\n"
            + "{{ images[1] }}\n第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
            + "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n这是一个二分类任务：一致标为 1，不一致标为 0。"
            + "{{ '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}";

    private SyntheticModels() {
    }

    // ================================================================== dense decoders

    /** LLM_ARCH_LLAMA: dense GQA attention + SwiGLU FFN, SPM tokenizer; see {@link SyntheticDenseModels#llama()}. */
    public static GgufTestWriter llama() {
        return SyntheticDenseModels.llama();
    }

    /** {@link #llama()} over {@code c}; see {@link SyntheticDenseModels#llama(SyntheticModels.Config)}. */
    public static GgufTestWriter llama(Config c) {
        return SyntheticDenseModels.llama(c);
    }

    /**
     * {@link #llama(Config)} with its scored pieces rotated by {@code pieceShift} ids, so the file describes a
     * token-level-incompatible vocabulary; see
     * {@link SyntheticDenseModels#llamaRotatedVocab(SyntheticModels.Config, int)}.
     */
    public static GgufTestWriter llamaRotatedVocab(Config c, int pieceShift) {
        return SyntheticDenseModels.llamaRotatedVocab(c, pieceShift);
    }

    /**
     * LLM_ARCH_QWEN2 with optional MiMo MTP {@code nextn} blocks; see
     * {@link SyntheticDenseModels#qwen2Mtp(SyntheticModels.Config, int)}.
     */
    public static GgufTestWriter qwen2Mtp(Config c, int nextn) {
        return SyntheticDenseModels.qwen2Mtp(c, nextn);
    }

    /**
     * LLM_ARCH_QWEN2 running Qwen2.5-1M dual-chunk attention.
     *
     * @param chunkSize   {@code chunk_size} of the block
     * @param localSize   {@code local_size} (the chunk length is {@code chunk_size - local_size})
     * @param originalMax {@code original_max_position_embeddings}; 0 disables the logit scaling
     * @see SyntheticDenseModels#qwen2DualChunk(SyntheticModels.Config, int, int, int)
     */
    public static GgufTestWriter qwen2DualChunk(Config c, int chunkSize, int localSize, int originalMax) {
        return SyntheticDenseModels.qwen2DualChunk(c, chunkSize, localSize, originalMax);
    }

    /** {@link #qwen2DualChunk(Config, int, int, int)} with the sparse-attention switch set. */
    public static GgufTestWriter qwen2DualChunk(Config c, int chunkSize, int localSize, int originalMax,
                                                boolean sparseEnabled) {
        return SyntheticDenseModels.qwen2DualChunk(c, chunkSize, localSize, originalMax, sparseEnabled);
    }

    /** LLM_ARCH_GEMMA2: sandwich norms, logit soft-capping, SWA pattern 2, tied head; see {@link SyntheticDenseModels}. */
    public static GgufTestWriter gemma2() {
        return SyntheticDenseModels.gemma2();
    }

    /** {@link #gemma2()} over {@code c}; see {@link SyntheticDenseModels#gemma2(SyntheticModels.Config)}. */
    public static GgufTestWriter gemma2(Config c) {
        return SyntheticDenseModels.gemma2(c);
    }

    /** LLM_ARCH_QWEN2VL: the qwen2 tensor set with the four M-RoPE sections; see {@link SyntheticDenseModels}. */
    public static GgufTestWriter qwen2vl() {
        return SyntheticDenseModels.qwen2vl();
    }

    /** {@link #qwen2vl()} over {@code c}; see {@link SyntheticDenseModels#qwen2vl(SyntheticModels.Config)}. */
    public static GgufTestWriter qwen2vl(Config c) {
        return SyntheticDenseModels.qwen2vl(c);
    }

    /**
     * LLM_ARCH_LIMITE: value embeddings, XSA with per-head alphas, learned residual scales, MUDD connections and the
     * sigmoid logit cap; see {@link SyntheticDenseModels#limite(SyntheticModels.Config)}.
     */
    public static GgufTestWriter limite() {
        return SyntheticDenseModels.limite();
    }

    /** {@link #limite()} over {@code c}; see {@link SyntheticDenseModels#limite(SyntheticModels.Config)}. */
    public static GgufTestWriter limite(Config c) {
        return SyntheticDenseModels.limite(c);
    }

    // ================================================================== routed experts

    /**
     * LLM_ARCH_GLM4_MOE: one leading dense block, sigmoid-gated experts with a selection bias and a shared expert,
     * then {@code nextn} NextN / MTP blocks; see {@link SyntheticMoeModels#glm4moe(SyntheticModels.Config, int)}.
     */
    public static GgufTestWriter glm4moe(Config c, int nextn) {
        return SyntheticMoeModels.glm4moe(c, nextn);
    }

    /** LLM_ARCH_QWEN3MOE: GQA attention over softmax-gated, renormalized experts; see {@link SyntheticMoeModels}. */
    public static GgufTestWriter qwen3moe() {
        return SyntheticMoeModels.qwen3moe();
    }

    /** {@link #qwen3moe()} over {@code c}; see {@link SyntheticMoeModels#qwen3moe(SyntheticModels.Config)}. */
    public static GgufTestWriter qwen3moe(Config c) {
        return SyntheticMoeModels.qwen3moe(c);
    }

    // ================================================================== hybrids

    /**
     * LLM_ARCH_QWEN3NEXT: hybrid gated-delta-net + attention + MoE; see
     * {@link SyntheticHybridModels#qwen3next(SyntheticModels.Config)}.
     */
    public static GgufTestWriter qwen3next() {
        return SyntheticHybridModels.qwen3next();
    }

    /** {@link #qwen3next()} over {@code c}; see {@link SyntheticHybridModels#qwen3next(SyntheticModels.Config)}. */
    public static GgufTestWriter qwen3next(Config c) {
        return SyntheticHybridModels.qwen3next(c);
    }

    /**
     * LLM_ARCH_LFM3: gated SWA attention alternating with short convolutions, causal by default; see
     * {@link SyntheticHybridModels#lfm3(SyntheticModels.Config, boolean)}.
     */
    public static GgufTestWriter lfm3() {
        return SyntheticHybridModels.lfm3();
    }

    /** {@link #lfm3()} over {@code c} with the causal switch; see {@link SyntheticHybridModels}. */
    public static GgufTestWriter lfm3(Config c, boolean causal) {
        return SyntheticHybridModels.lfm3(c, causal);
    }

    /**
     * LLM_ARCH_LFM2: short convolutions alternating with attention, tied head by default; see
     * {@link SyntheticHybridModels#lfm2(SyntheticModels.Config)}.
     */
    public static GgufTestWriter lfm2() {
        return SyntheticHybridModels.lfm2();
    }

    /** {@link #lfm2()} over {@code c}; see {@link SyntheticHybridModels#lfm2(SyntheticModels.Config)}. */
    public static GgufTestWriter lfm2(Config c) {
        return SyntheticHybridModels.lfm2(c);
    }

    /** LLM_ARCH_PLAMO2: alternating PLaMo-2 Mamba and fused-QKV attention layers; see {@link SyntheticHybridModels}. */
    public static GgufTestWriter plamo2() {
        return SyntheticHybridModels.plamo2();
    }

    /** {@link #plamo2()} over {@code c}; see {@link SyntheticHybridModels#plamo2(SyntheticModels.Config)}. */
    public static GgufTestWriter plamo2(Config c) {
        return SyntheticHybridModels.plamo2(c);
    }

    /**
     * LLM_ARCH_KIMI_LINEAR: alternating KDA and absorbed-MLA layers over sigmoid-routed experts; see
     * {@link SyntheticHybridModels#kimiLinear(SyntheticModels.Config)}.
     */
    public static GgufTestWriter kimiLinear() {
        return SyntheticHybridModels.kimiLinear();
    }

    /** {@link #kimiLinear()} over {@code c}; see {@link SyntheticHybridModels#kimiLinear(SyntheticModels.Config)}. */
    public static GgufTestWriter kimiLinear(Config c) {
        return SyntheticHybridModels.kimiLinear(c);
    }

    // ================================================================== SSM decoders

    /** LLM_ARCH_MAMBA2: attention-free SSD blocks; see {@link SyntheticRecurrentModels#mamba2(SyntheticModels.Config)}. */
    public static GgufTestWriter mamba2() {
        return SyntheticRecurrentModels.mamba2();
    }

    /** {@link #mamba2()} over {@code c}; see {@link SyntheticRecurrentModels#mamba2(SyntheticModels.Config)}. */
    public static GgufTestWriter mamba2(Config c) {
        return SyntheticRecurrentModels.mamba2(c);
    }

    /** LLM_ARCH_MAMBA: selective-scan blocks in the Mamba-1 layout; see {@link SyntheticRecurrentModels}. */
    public static GgufTestWriter mamba() {
        return SyntheticRecurrentModels.mamba();
    }

    /** {@link #mamba()} over {@code c}; see {@link SyntheticRecurrentModels#mamba(SyntheticModels.Config)}. */
    public static GgufTestWriter mamba(Config c) {
        return SyntheticRecurrentModels.mamba(c);
    }

    // ================================================================== latent attention and DSA

    /** LLM_ARCH_DEEPSEEK2: MLA with the q_lora path and a dense lead block; see {@link SyntheticMlaDsaModels}. */
    public static GgufTestWriter deepseek2() {
        return SyntheticMlaDsaModels.deepseek2();
    }

    /** {@link #deepseek2()} over {@code c}; see {@link SyntheticMlaDsaModels#deepseek2(SyntheticModels.Config)}. */
    public static GgufTestWriter deepseek2(Config c) {
        return SyntheticMlaDsaModels.deepseek2(c);
    }

    /** LLM_ARCH_DEEPSEEK32: the deepseek2 layers plus the DSA lightning indexer; see {@link SyntheticMlaDsaModels}. */
    public static GgufTestWriter deepseek32() {
        return SyntheticMlaDsaModels.deepseek32();
    }

    /** {@link #deepseek32()} over {@code c}; see {@link SyntheticMlaDsaModels#deepseek32(SyntheticModels.Config)}. */
    public static GgufTestWriter deepseek32(Config c) {
        return SyntheticMlaDsaModels.deepseek32(c);
    }

    /**
     * LLM_ARCH_MISTRAL_LARGE4: MLA (no Q LoRA) with the DSA indexer, SSMax and a KV source map; see
     * {@link SyntheticMlaDsaModels#mistralLarge4(SyntheticModels.Config)}.
     */
    public static GgufTestWriter mistralLarge4() {
        return SyntheticMlaDsaModels.mistralLarge4();
    }

    /** {@link #mistralLarge4()} over {@code c}; see {@link SyntheticMlaDsaModels#mistralLarge4(SyntheticModels.Config)}. */
    public static GgufTestWriter mistralLarge4(Config c) {
        return SyntheticMlaDsaModels.mistralLarge4(c);
    }

    /**
     * A mistral-large4 file whose layer 0 rotates and whose other layers are NoPE ({@code attention.rope_pattern});
     * see {@link SyntheticMlaDsaModels#mistralLarge4NoPe(SyntheticModels.Config)}.
     */
    public static GgufTestWriter mistralLarge4NoPe(Config c) {
        return SyntheticMlaDsaModels.mistralLarge4NoPe(c);
    }

    // ================================================================== encoders

    /** LLM_ARCH_BERT: non-causal encoder with learned positions and mean pooling; see {@link SyntheticEncoderModels}. */
    public static GgufTestWriter bert() {
        return SyntheticEncoderModels.bert();
    }

    /** {@link #bert()} over {@code c}; see {@link SyntheticEncoderModels#bert(SyntheticModels.Config)}. */
    public static GgufTestWriter bert(Config c) {
        return SyntheticEncoderModels.bert(c);
    }

    /** LLM_ARCH_T5: encoder-decoder with relative-position bias and cross-attention; see {@link SyntheticEncoderModels}. */
    public static GgufTestWriter t5() {
        return SyntheticEncoderModels.t5();
    }

    /** {@link #t5()} over {@code c}; see {@link SyntheticEncoderModels#t5(SyntheticModels.Config)}. */
    public static GgufTestWriter t5(Config c) {
        return SyntheticEncoderModels.t5(c);
    }

    // ================================================================== decision readouts

    /** LLM_ARCH_VLLM_DECISION2 with the systemone template and calibration keys; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter vllmDecision2() {
        return SyntheticDecisionModels.vllmDecision2();
    }

    /** {@link #vllmDecision2()} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter vllmDecision2(Config c) {
        return SyntheticDecisionModels.vllmDecision2(c);
    }

    /** {@link #vllmDecision2(Config)} with {@code systemoneTemplate}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter vllmDecision2(Config c, String systemoneTemplate) {
        return SyntheticDecisionModels.vllmDecision2(c, systemoneTemplate);
    }

    /** A label decision readout over a qwen2 trunk; see {@link SyntheticDecisionModels#decisionLabels(String)}. */
    public static GgufTestWriter decisionLabels(String type) {
        return SyntheticDecisionModels.decisionLabels(type);
    }

    /** {@link #decisionLabels(String)} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter decisionLabels(String type, Config c) {
        return SyntheticDecisionModels.decisionLabels(type, c);
    }

    /** {@link #decisionLabels(String)} with {@code systemoneTemplate}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter decisionLabels(String type, Config c, String systemoneTemplate) {
        return SyntheticDecisionModels.decisionLabels(type, c, systemoneTemplate);
    }

    /** The kev decision readout with its {@code <|box_end|>} marker token; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter decisionKev() {
        return SyntheticDecisionModels.decisionKev();
    }

    /** {@link #decisionKev()} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter decisionKev(Config c) {
        return SyntheticDecisionModels.decisionKev(c);
    }

    /** {@link #decisionKev(Config)} with {@code systemoneTemplate}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter decisionKev(Config c, String systemoneTemplate) {
        return SyntheticDecisionModels.decisionKev(c, systemoneTemplate);
    }

    /** The modern-bert decision fixture with the laya readout; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter modernBertDecision() {
        return SyntheticDecisionModels.modernBertDecision();
    }

    /** {@link #modernBertDecision()} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter modernBertDecision(Config c) {
        return SyntheticDecisionModels.modernBertDecision(c);
    }

    /** The h2o-lightning fixture: the label readout over {@code decision.labels}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter h2oLightning() {
        return SyntheticDecisionModels.h2oLightning();
    }

    /** {@link #h2oLightning()} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter h2oLightning(Config c) {
        return SyntheticDecisionModels.h2oLightning(c);
    }

    /** The minicpm-ocr-jev fixture: the pooled two-class readout; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter minicpmOcrJev() {
        return SyntheticDecisionModels.minicpmOcrJev();
    }

    /** {@link #minicpmOcrJev()} over {@code c}; see {@link SyntheticDecisionModels}. */
    public static GgufTestWriter minicpmOcrJev(Config c) {
        return SyntheticDecisionModels.minicpmOcrJev(c);
    }

    // ================================================================== d1 decision readouts

    /** The lfm2-d1 fixture: the plain LFM2 trunk with the d1 decision type; see {@link SyntheticD1Models}. */
    public static GgufTestWriter lfm2Decision() {
        return SyntheticD1Models.lfm2Decision();
    }

    /** {@link #lfm2Decision()} over {@code c}; see {@link SyntheticD1Models#lfm2Decision(SyntheticModels.Config)}. */
    public static GgufTestWriter lfm2Decision(Config c) {
        return SyntheticD1Models.lfm2Decision(c);
    }

    /** The lfm2-d1-omni fixture: the mask-marker read over one decision block; see {@link SyntheticD1Models}. */
    public static GgufTestWriter lfm2D1Omni() {
        return SyntheticD1Models.lfm2D1Omni();
    }

    /** {@link #lfm2D1Omni()} over {@code c}; see {@link SyntheticD1Models#lfm2D1Omni(SyntheticModels.Config)}. */
    public static GgufTestWriter lfm2D1Omni(Config c) {
        return SyntheticD1Models.lfm2D1Omni(c);
    }

    /** The lfm3-d1 fixture: the non-causal LFM3 trunk with the d1 readout; see {@link SyntheticD1Models}. */
    public static GgufTestWriter lfm3D1() {
        return SyntheticD1Models.lfm3D1();
    }

    /** {@link #lfm3D1()} over {@code c}; see {@link SyntheticD1Models#lfm3D1(SyntheticModels.Config)}. */
    public static GgufTestWriter lfm3D1(Config c) {
        return SyntheticD1Models.lfm3D1(c);
    }

    /** The lfm3-d1-omni fixture: the d1 readout with the think marker; see {@link SyntheticD1Models}. */
    public static GgufTestWriter lfm3D1Omni() {
        return SyntheticD1Models.lfm3D1Omni();
    }

    /** {@link #lfm3D1Omni()} over {@code c}; see {@link SyntheticD1Models#lfm3D1Omni(SyntheticModels.Config)}. */
    public static GgufTestWriter lfm3D1Omni(Config c) {
        return SyntheticD1Models.lfm3D1Omni(c);
    }

    // ================================================================== audio and drafting

    /** The WavTokenizer decoder fixture; see {@link SyntheticAudioModels#wavtokenizer}. */
    public static GgufTestWriter wavtokenizer(int sampleRate, int nFft, int hopLength) {
        return SyntheticAudioModels.wavtokenizer(sampleRate, nFft, hopLength);
    }

    /** The DFlash drafter fixture; see {@link SyntheticAudioModels#dflashDrafter}. */
    public static GgufTestWriter dflashDrafter(Config c, int[] targetLayers, int blockSize) {
        return SyntheticAudioModels.dflashDrafter(c, targetLayers, blockSize);
    }

    /** {@link #dflashDrafter(Config, int[], int)} with the causal flag and an interleaved sliding window. */
    public static GgufTestWriter dflashDrafter(Config c, int[] targetLayers, int blockSize, boolean causal, int nSwa,
                                               boolean[] swaPattern) {
        return SyntheticAudioModels.dflashDrafter(c, targetLayers, blockSize, causal, nSwa, swaPattern);
    }

    /** {@link #dflashDrafter(Config, int[], int)} as a DFlash2 drafter. */
    public static GgufTestWriter dflashDrafter(Config c, int[] targetLayers, int blockSize, boolean causal, int nSwa,
                                               boolean[] swaPattern, Dflash2 dflash2) {
        return SyntheticAudioModels.dflashDrafter(c, targetLayers, blockSize, causal, nSwa, swaPattern, dflash2);
    }

    /**
     * The DFlash2 settings of a drafter fixture: conv kernel and group sizes and selector rank and top-k
     * (dflash.cpp:152-177).
     */
    public static final class Dflash2 {
        final int convKernel;
        final int convGroup;
        final int selectorRank;
        final int selectorTopK;

        public Dflash2(int convKernel, int convGroup, int selectorRank, int selectorTopK) {
            this.convKernel = convKernel;
            this.convGroup = convGroup;
            this.selectorRank = selectorRank;
            this.selectorTopK = selectorTopK;
        }
    }

    // ================================================================== classifier heads

    /** The RANK classifier fixture; see {@link SyntheticClassifiers#rank}. */
    public static GgufTestWriter rank(String... labels) {
        return SyntheticClassifiers.rank(labels);
    }

    /** The per-token classifier fixture; see {@link SyntheticClassifiers#tokenClassifier}. */
    public static GgufTestWriter tokenClassifier(String... labels) {
        return SyntheticClassifiers.tokenClassifier(labels);
    }

    /** The reward-model fixture; see {@link SyntheticClassifiers#reward}. */
    public static GgufTestWriter reward() {
        return SyntheticClassifiers.reward();
    }

    /** The process-reward fixture; see {@link SyntheticClassifiers#processReward}. */
    public static GgufTestWriter processReward() {
        return SyntheticClassifiers.processReward();
    }

    /** The sequence-classifier fixture; see {@link SyntheticClassifiers#sequenceClassify}. */
    public static GgufTestWriter sequenceClassify(String... labels) {
        return SyntheticClassifiers.sequenceClassify(labels);
    }

    /** The ColBERT late-interaction fixture; see {@link SyntheticClassifiers#colbert}. */
    public static GgufTestWriter colbert() {
        return SyntheticClassifiers.colbert();
    }

    /** The SPLADE sparse-embedding fixture; see {@link SyntheticClassifiers#splade}. */
    public static GgufTestWriter splade(String kind) {
        return SyntheticClassifiers.splade(kind);
    }

    /** The BGE-M3 multi-output fixture; see {@link SyntheticClassifiers#bgeM3}. */
    public static GgufTestWriter bgeM3() {
        return SyntheticClassifiers.bgeM3();
    }

    // ================================================================== tokenizers

    /** The SPM vocab of the llama fixture; see {@link SyntheticText#spm}. */
    public static void spm(GgufTestWriter w, int nVocab) {
        SyntheticText.spm(w, nVocab);
    }

    // ================================================================== tensor emission

    /** Adds tensors to one writer with values derived from the config seed and the tensor name. */
    static final class Tensors {
        private final GgufTestWriter w;
        private final Config c;

        Tensors(GgufTestWriter w, Config c) {
            this.w = w;
            this.c = c;
        }

        /** Projection weight in the configured weight type. */
        void matrix(String name, long... ne) {
            GGMLType type = c.weightType();
            if (GgufTestWriter.canEncodeFloats(type)) {
                w.tensorRandom(name, type, seedFor(name), -0.1f, 0.1f, ne);
            } else {
                w.tensorZeros(name, type, ne);
            }
        }

        void random(String name, float lo, float hi, long... ne) {
            w.tensorRandom(name, GGMLType.F32, seedFor(name), lo, hi, ne);
        }

        void ones(String name, long... ne) {
            w.tensorFilled(name, GGMLType.F32, 1.0f, ne);
        }

        void bias(String name, long n) {
            random(name, -0.02f, 0.02f, n);
        }

        void layerNorm(String prefix, long n) {
            ones(prefix + ".weight", n);
            w.tensorFilled(prefix + ".bias", GGMLType.F32, 0.0f, n);
        }

        /** Separate Q/K/V/O projections; {@code qRows} covers gated-Q layouts. */
        void qkvo(int il, long qRows) {
            long kvRows = (long) c.nHeadKv() * c.headDim();
            matrix(blk(il, "attn_q.weight"), c.nEmbd(), qRows);
            matrix(blk(il, "attn_k.weight"), c.nEmbd(), kvRows);
            matrix(blk(il, "attn_v.weight"), c.nEmbd(), kvRows);
            matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
        }

        void gatedFfn(int il, long nFf) {
            matrix(blk(il, "ffn_gate.weight"), c.nEmbd(), nFf);
            matrix(blk(il, "ffn_up.weight"), c.nEmbd(), nFf);
            matrix(blk(il, "ffn_down.weight"), nFf, c.nEmbd());
        }

        private long seedFor(String name) {
            return c.seed() ^ (name.hashCode() * 0x9E3779B97F4A7C15L);
        }
    }

    // ================================================================== config

    /**
     * Immutable fixture dimensions. Start from an arch default ({@link #llama()}, {@link #gemma2()},
     * {@link #qwen3next()}, {@link #mamba2()}, {@link #bert()}) and adjust with {@link #toBuilder()}.
     * Each arch builder reads only the fields that apply to it; {@code normEps} is the RMS epsilon
     * for RMS-norm archs and the LayerNorm epsilon for bert.
     */
    public static final class Config {
        private final int nVocab;
        private final int nEmbd;
        private final int nLayer;
        private final int nHead;
        private final int nHeadKv;
        private final int headDim;
        private final int nRot;
        private final int kvLora;
        private final int nFf;
        private final int nCtxTrain;
        private final float normEps;
        private final float ropeFreqBase;
        private final int slidingWindow;
        private final float attnLogitSoftcap;
        private final float finalLogitSoftcap;
        private final int nExpert;
        private final int nExpertUsed;
        private final int nFfExp;
        private final int nFfShexp;
        private final int fullAttentionInterval;
        private final int ssmConv;
        private final int ssmInner;
        private final int ssmState;
        private final int ssmGroups;
        private final int ssmDtRank;
        private final int kdaHeadDim;
        private final GGMLType weightType;
        private final long seed;
        private final boolean tiedOutput;
        private final boolean expertBiases;

        private Config(Builder b) {
            this.nVocab = b.nVocab;
            this.nEmbd = b.nEmbd;
            this.nLayer = b.nLayer;
            this.nHead = b.nHead;
            this.nHeadKv = b.nHeadKv;
            this.headDim = b.headDim;
            this.nRot = b.nRot;
            this.kvLora = b.kvLora;
            this.nFf = b.nFf;
            this.nCtxTrain = b.nCtxTrain;
            this.normEps = b.normEps;
            this.ropeFreqBase = b.ropeFreqBase;
            this.slidingWindow = b.slidingWindow;
            this.attnLogitSoftcap = b.attnLogitSoftcap;
            this.finalLogitSoftcap = b.finalLogitSoftcap;
            this.nExpert = b.nExpert;
            this.nExpertUsed = b.nExpertUsed;
            this.nFfExp = b.nFfExp;
            this.nFfShexp = b.nFfShexp;
            this.fullAttentionInterval = b.fullAttentionInterval;
            this.ssmConv = b.ssmConv;
            this.ssmInner = b.ssmInner;
            this.ssmState = b.ssmState;
            this.ssmGroups = b.ssmGroups;
            this.ssmDtRank = b.ssmDtRank;
            this.kdaHeadDim = b.kdaHeadDim;
            this.weightType = b.weightType;
            this.seed = b.seed;
            this.tiedOutput = b.tiedOutput;
            this.expertBiases = b.expertBiases;
        }

        /** 2 layers, n_embd 64, 4 heads / 2 KV heads of dim 16, FFN 128, vocab 320, F32. */
        public static Config llama() {
            return attentionDefaults().nLayer(2).normEps(1e-5f).build();
        }

        /** 4 layers (SWA, global, SWA, global), window 32, soft-caps 50 / 30. */
        public static Config gemma2() {
            return attentionDefaults().nLayer(4).normEps(1e-6f).slidingWindow(32).attnLogitSoftcap(50.0f)
                    .finalLogitSoftcap(30.0f).tiedOutput(true).build();
        }

        /** 4 layers (GDN, GDN, GDN, attention), 4 experts / 2 used, GDN 2 k-heads / 4 v-heads of dim 16. */
        public static Config qwen3next() {
            return attentionDefaults().nLayer(4).nRot(4).normEps(1e-6f).ropeFreqBase(1.0e7f).nExpert(4)
                    .nExpertUsed(2).nFfExp(32).nFfShexp(32).fullAttentionInterval(4).ssmConv(4).ssmState(16)
                    .ssmGroups(2).ssmDtRank(4).build();
        }

        /** 2 layers of plain GQA attention with the Qwen2 biases, FFN 128, RMS eps 1e-6. */
        public static Config qwen2() {
            return attentionDefaults().nLayer(2).nRot(16).normEps(1e-6f).build();
        }

        /** 2 layers of plain GQA attention over a routed MoE (4 experts, 2 used, FFN 32). */
        public static Config qwen3moe() {
            return attentionDefaults().nLayer(2).nRot(16).normEps(1e-6f).ropeFreqBase(1.0e7f).nExpert(4)
                    .nExpertUsed(2).nFfExp(32).build();
        }

        /** 2 layers, d_inner 128, d_state 16, 1 group, 8 heads of dim 16, conv 4. */
        public static Config mamba2() {
            return new Builder().nLayer(2).nHead(0).nHeadKv(0).headDim(0).nRot(0).nFf(0).normEps(1e-5f)
                    .ssmConv(4).ssmInner(128).ssmState(16).ssmGroups(1).ssmDtRank(8).build();
        }

        /** 2 layers, d_inner 128 (= 2 * n_embd), d_state 16, dt rank 8, conv 4; the Mamba-1 layout, no groups. */
        public static Config mamba() {
            return new Builder().nLayer(2).nHead(0).nHeadKv(0).headDim(0).nRot(0).nFf(0).normEps(1e-5f)
                    .ssmConv(4).ssmInner(128).ssmState(16).ssmDtRank(8).build();
        }

        /** 4 layers (Mamba, attention, Mamba, attention), 4 heads of dim 16, FFN 128, d_inner 128 in 4 heads. */
        public static Config plamo2() {
            return attentionDefaults().nLayer(4).nHeadKv(4).normEps(1e-6f).ssmConv(4).ssmInner(128).ssmState(16)
                    .ssmDtRank(4).build();
        }

        /** 4 layers (KDA, MLA, KDA, MLA), MLA head 32 (nope-only) over latent 32, KDA head 16, 4 experts / 2 used. */
        public static Config kimiLinear() {
            return attentionDefaults().nLayer(4).nHeadKv(4).headDim(32).nRot(0).normEps(1e-6f).nExpert(4)
                    .nExpertUsed(2).nFfExp(32).ssmConv(4).kdaHeadDim(16).build();
        }

        /** 4 layers (SWA attention, convolution, full attention, convolution), l_cache 4, window 32. */
        public static Config lfm3() {
            return attentionDefaults().nLayer(4).nRot(16).normEps(1e-6f).slidingWindow(32).ssmConv(4)
                    .tiedOutput(true).build();
        }

        /** 4 layers (convolution, attention, convolution, attention), l_cache 4, tied output head. */
        public static Config lfm2() {
            return attentionDefaults().nLayer(4).nRot(16).normEps(1e-6f).ssmConv(4).tiedOutput(true).build();
        }

        /** 2 layers (SWA attention with the ve gate, full attention with the output gate), mudd width 32. */
        public static Config limite() {
            return attentionDefaults().nLayer(2).nRot(16).normEps(1e-6f).slidingWindow(16).nFfExp(32).build();
        }

        /** 2 layers, 4 heads of MLA dim 32 (16 rotary), 4 experts of width 48, layer 1 MoE. */
        public static Config mistralLarge4() {
            return attentionDefaults().nLayer(2).headDim(32).nRot(16).nFf(128).nFfExp(48).nExpert(4).normEps(1e-6f)
                    .build();
        }

        /** 2 layers, 4 heads of dim 16 (MHA), FFN 128, 128 positions, LayerNorm eps 1e-12. */
        public static Config bert() {
            return attentionDefaults().nLayer(2).nHeadKv(4).nCtxTrain(128).normEps(1e-12f).build();
        }

        /** 2 encoder and 2 decoder layers, 4 heads of dim 16 (MHA), FFN 128, RMS eps 1e-6. */
        public static Config t5() {
            return attentionDefaults().nLayer(2).nHeadKv(4).normEps(1e-6f).build();
        }

        private static Builder attentionDefaults() {
            return new Builder().nHead(4).nHeadKv(2).headDim(16).nRot(16).nFf(128);
        }

        public Builder toBuilder() {
            return new Builder(this);
        }

        public int nVocab() { return nVocab; }
        public int nEmbd() { return nEmbd; }
        public int nLayer() { return nLayer; }
        public int nHead() { return nHead; }
        public int nHeadKv() { return nHeadKv; }
        public int headDim() { return headDim; }
        public int nRot() { return nRot; }
        /** Rank of the DeepSeek MLA latent rows; the absorbed cache stores {@code kvLora + nRot} values per row. */
        public int kvLora() { return kvLora; }
        public int nFf() { return nFf; }
        public int nCtxTrain() { return nCtxTrain; }
        public float normEps() { return normEps; }
        public float ropeFreqBase() { return ropeFreqBase; }
        public int slidingWindow() { return slidingWindow; }
        public float attnLogitSoftcap() { return attnLogitSoftcap; }
        public float finalLogitSoftcap() { return finalLogitSoftcap; }
        public int nExpert() { return nExpert; }
        public int nExpertUsed() { return nExpertUsed; }
        public int nFfExp() { return nFfExp; }
        public int nFfShexp() { return nFfShexp; }
        public int fullAttentionInterval() { return fullAttentionInterval; }
        public int ssmConv() { return ssmConv; }
        public int ssmInner() { return ssmInner; }
        public int ssmState() { return ssmState; }
        public int ssmGroups() { return ssmGroups; }
        public int ssmDtRank() { return ssmDtRank; }
        /** Head width of the KDA layers; unrelated to the attention head dim. */
        public int kdaHeadDim() { return kdaHeadDim; }
        public GGMLType weightType() { return weightType; }
        public long seed() { return seed; }
        public boolean tiedOutput() { return tiedOutput; }
        public boolean expertBiases() { return expertBiases; }

        /** Mutable builder for {@link Config}; {@link #build()} validates the shared fields. */
        public static final class Builder {
            private int nVocab = 320;
            private int nEmbd = 64;
            private int nLayer = 2;
            private int nHead;
            private int nHeadKv;
            private int headDim;
            private int nRot;
            private int kvLora = SyntheticMlaDsaModels.KV_LORA;
            private int nFf;
            private int nCtxTrain = 256;
            private float normEps = 1e-5f;
            private float ropeFreqBase = 10000.0f;
            private int slidingWindow;
            private float attnLogitSoftcap;
            private float finalLogitSoftcap;
            private int nExpert;
            private int nExpertUsed;
            private int nFfExp;
            private int nFfShexp;
            private int fullAttentionInterval;
            private int ssmConv;
            private int ssmInner;
            private int ssmState;
            private int ssmGroups;
            private int ssmDtRank;
            private int kdaHeadDim;
            private GGMLType weightType = GGMLType.F32;
            private long seed = 42L;
            private boolean tiedOutput;
            private boolean expertBiases;

            public Builder() {
            }

            private Builder(Config c) {
                nVocab = c.nVocab;
                nEmbd = c.nEmbd;
                nLayer = c.nLayer;
                nHead = c.nHead;
                nHeadKv = c.nHeadKv;
                headDim = c.headDim;
                nRot = c.nRot;
                nFf = c.nFf;
                nCtxTrain = c.nCtxTrain;
                normEps = c.normEps;
                ropeFreqBase = c.ropeFreqBase;
                slidingWindow = c.slidingWindow;
                attnLogitSoftcap = c.attnLogitSoftcap;
                finalLogitSoftcap = c.finalLogitSoftcap;
                nExpert = c.nExpert;
                nExpertUsed = c.nExpertUsed;
                nFfExp = c.nFfExp;
                nFfShexp = c.nFfShexp;
                fullAttentionInterval = c.fullAttentionInterval;
                ssmConv = c.ssmConv;
                ssmInner = c.ssmInner;
                ssmState = c.ssmState;
                ssmGroups = c.ssmGroups;
                ssmDtRank = c.ssmDtRank;
                kdaHeadDim = c.kdaHeadDim;
                weightType = c.weightType;
                seed = c.seed;
                tiedOutput = c.tiedOutput;
                expertBiases = c.expertBiases;
            }

            public Builder nVocab(int v) { nVocab = v; return this; }
            public Builder nEmbd(int v) { nEmbd = v; return this; }
            public Builder nLayer(int v) { nLayer = v; return this; }
            public Builder nHead(int v) { nHead = v; return this; }
            public Builder nHeadKv(int v) { nHeadKv = v; return this; }
            public Builder headDim(int v) { headDim = v; return this; }
            public Builder nRot(int v) { nRot = v; return this; }
            public Builder kvLora(int v) { kvLora = v; return this; }
            public Builder nFf(int v) { nFf = v; return this; }
            public Builder nCtxTrain(int v) { nCtxTrain = v; return this; }
            public Builder normEps(float v) { normEps = v; return this; }
            public Builder ropeFreqBase(float v) { ropeFreqBase = v; return this; }
            public Builder slidingWindow(int v) { slidingWindow = v; return this; }
            public Builder attnLogitSoftcap(float v) { attnLogitSoftcap = v; return this; }
            public Builder finalLogitSoftcap(float v) { finalLogitSoftcap = v; return this; }
            public Builder nExpert(int v) { nExpert = v; return this; }
            public Builder nExpertUsed(int v) { nExpertUsed = v; return this; }
            public Builder nFfExp(int v) { nFfExp = v; return this; }
            public Builder nFfShexp(int v) { nFfShexp = v; return this; }
            public Builder fullAttentionInterval(int v) { fullAttentionInterval = v; return this; }
            public Builder ssmConv(int v) { ssmConv = v; return this; }
            public Builder ssmInner(int v) { ssmInner = v; return this; }
            public Builder ssmState(int v) { ssmState = v; return this; }
            public Builder ssmGroups(int v) { ssmGroups = v; return this; }
            public Builder ssmDtRank(int v) { ssmDtRank = v; return this; }
            public Builder kdaHeadDim(int v) { kdaHeadDim = v; return this; }
            public Builder weightType(GGMLType v) { weightType = Objects.requireNonNull(v, "weightType"); return this; }
            public Builder seed(long v) { seed = v; return this; }
            public Builder tiedOutput(boolean v) { tiedOutput = v; return this; }
            /** Writes the per-expert {@code ffn_{gate,up,down}_exps.bias} rows of every MoE layer. */
            public Builder expertBiases(boolean v) { expertBiases = v; return this; }

            public Config build() {
                requirePositive("vocab size", nVocab);
                requirePositive("embedding length", nEmbd);
                requirePositive("layer count", nLayer);
                requirePositive("training context length", nCtxTrain);
                if (nHead < 0 || nHeadKv < 0 || headDim < 0 || nRot < 0 || nFf < 0) {
                    throw new IllegalArgumentException("head counts, head dim, rope dim and FFN length must be >= 0");
                }
                if (!(normEps > 0.0f)) {
                    throw new IllegalArgumentException("norm epsilon must be positive, got " + normEps);
                }
                if (weightType == GGMLType.UNKNOWN || weightType.getBlockSize() <= 0
                        || weightType.getTypeSize() <= 0) {
                    throw new IllegalArgumentException(weightType + " is not a storage type");
                }
                return new Config(this);
            }
        }
    }
}
