package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooks;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooksEncoder;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooksHeads;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirements;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirementsEncoder;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirementsHeads;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Architecture table for the encoder, classifier, diffusion, drafter, audio, encoder-decoder and looped architectures.
 * {@link ArchRegistry} merges it with the other tables; an architecture appears in exactly one table.
 *
 * <p>{@link Arch#CLIP}, {@link Arch#GPTJ} and {@link Arch#UNKNOWN} are registered by {@link ArchRegistry}
 * itself as rejected entries and must not appear here.
 *
 * <p>This class holds the encoder and classifier rows; {@link ArchTableSpecialGen} holds the diffusion,
 * looped, audio and drafter rows. Rope types follow llama.cpp {@code llama_model_rope_type}
 * (llama-model.cpp:3476-3667), memory kinds {@code llama_model::create_memory} (llama-model.cpp:2778-2807).
 * The BERT-style encoders take {@code attention.causal} from the file and default to bidirectional attention,
 * which is what their converters always write (conversion/bert.py:34, 451); llama-embed keeps llama.cpp's
 * causal default because its converter writes the key only for bidirectional checkpoints
 * (conversion/base.py:1629-1630); the T5 encoder always runs bidirectionally (llama-context.cpp:1659-1662).
 * Architectures whose llama.cpp class derives from another one (llama-embed, vllm-decision2, clef, qwen3tts)
 * reuse that row's hook and tensor list.
 */
public final class ArchTableSpecial {

    private ArchTableSpecial() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        Map<Arch, ArchEntry> bases = baseRows();
        List<ArchEntry> out = new ArrayList<>();
        out.addAll(bertRows());
        out.addAll(encoderRows(bases));
        out.addAll(ArchTableSpecialGen.rows(bases));
        return Collections.unmodifiableList(out);
    }

    /**
     * Rows of the other tables, for the architectures that extend a base architecture. Read from the tables
     * directly: {@link ArchRegistry} is still being built while this table is constructed.
     */
    private static Map<Arch, ArchEntry> baseRows() {
        Map<Arch, ArchEntry> m = new EnumMap<>(Arch.class);
        for (List<ArchEntry> table : Arrays.asList(ArchTableDense.entries(), ArchTableMoe.entries(),
                ArchTableHybrid.entries())) {
            for (ArchEntry e : table) {
                m.put(e.arch(), e);
            }
        }
        return m;
    }

    /**
     * Row of the base architecture whose llama.cpp class {@code derived} subclasses.
     *
     * @throws IllegalStateException when no other table registers {@code base}
     */
    static ArchEntry base(Map<Arch, ArchEntry> bases, Arch derived, Arch base) {
        ArchEntry e = bases.get(base);
        if (e == null) {
            throw new IllegalStateException("architecture '" + derived.ggufName() + "' reuses the hparams and tensors"
                    + " of '" + base.ggufName() + "', which no architecture table registers");
        }
        return e;
    }

    /**
     * Row reusing {@code base}'s hook and tensors, followed by its own hook and tensors (the C++ subclass
     * calling the base {@code load_arch_hparams}/{@code load_arch_tensors} first).
     */
    static ArchEntry derived(Map<Arch, ArchEntry> bases, ArchTraits traits, Arch base, ArchHparamsHook ownHook,
                             List<TensorRequirement> ownTensors) {
        ArchEntry b = base(bases, traits.arch(), base);
        return ArchEntry.of(traits, SpecialHooks.extend(b.hook(), ownHook),
                SpecialTensorRequirements.extend(b.tensors(), ownTensors));
    }

    /** BERT-graph encoders (models.h {@code using graph = llama_model_bert::graph}), post-LN with LayerNorm. */
    private static List<ArchEntry> bertRows() {
        return Arrays.asList(
                ArchEntry.of(bert(Arch.BERT, RopeType.NEOX).ffnAct(ActOp.GELU_TANH).ffnGating(FfnGating.SEQ).build(),
                        SpecialHooksEncoder.BERT, SpecialTensorRequirementsEncoder.bert()),
                ArchEntry.of(bert(Arch.NOMIC_BERT, RopeType.NEOX).ffnAct(ActOp.SILU).ffnGating(FfnGating.PAR)
                                .graphFamily(Arch.BERT).build(),
                        SpecialHooksEncoder.NOMIC_BERT, SpecialTensorRequirementsEncoder.nomicBert()),
                ArchEntry.of(bert(Arch.NOMIC_BERT_MOE, RopeType.NEOX).ffnAct(ActOp.GELU_TANH).ffnGating(FfnGating.SEQ)
                                .defaultGating(GatingFunc.SOFTMAX).graphFamily(Arch.BERT).build(),
                        SpecialHooksEncoder.NOMIC_BERT_MOE, SpecialTensorRequirementsEncoder.nomicBertMoe()),
                ArchEntry.of(bert(Arch.JINA_BERT_V2, RopeType.NONE).ffnAct(ActOp.GELU_TANH).ffnGating(FfnGating.PAR)
                                .graphFamily(Arch.BERT).build(),
                        SpecialHooksEncoder.JINA_BERT_V2, SpecialTensorRequirementsEncoder.jinaBertV2()),
                ArchEntry.of(bert(Arch.JINA_BERT_V3, RopeType.NEOX).ffnAct(ActOp.GELU_TANH).ffnGating(FfnGating.SEQ)
                                .graphFamily(Arch.BERT).build(),
                        SpecialHooksEncoder.JINA_BERT_V3, SpecialTensorRequirementsEncoder.jinaBertV3()));
    }

    /** Traits shared by the BERT-graph encoders: no memory, bidirectional by default, LayerNorm. */
    private static ArchTraits.Builder bert(Arch arch, RopeType rope) {
        return encoder(arch, rope).normKind(NormKind.LAYER);
    }

    /** Embedding encoder evaluated without memory whose causality defaults to bidirectional. */
    private static ArchTraits.Builder encoder(Arch arch, RopeType rope) {
        return ArchTraits.builder(arch).ropeType(rope).memoryKind(MemoryKind.NONE).causalDefault(false)
                .support(SupportClass.EMBED);
    }

    /** Other encoders, encoder-decoder and classifier rows. */
    private static List<ArchEntry> encoderRows(Map<Arch, ArchEntry> bases) {
        return Arrays.asList(
                // modern-bert.cpp:20-24: GEGLU-erf with the gate packed in ffn_up
                ArchEntry.of(encoder(Arch.MODERN_BERT, RopeType.NEOX).normKind(NormKind.LAYER).ffnAct(ActOp.GELU_ERF)
                                .ffnGating(FfnGating.FUSED_UP).build(),
                        SpecialHooksEncoder.MODERN_BERT, SpecialTensorRequirementsEncoder.modernBert()),
                // neo-bert.cpp:116: SwiGLU with the gate packed in ffn_up
                ArchEntry.of(encoder(Arch.NEO_BERT, RopeType.NORM).ffnAct(ActOp.SILU).ffnGating(FfnGating.FUSED_UP)
                                .build(),
                        SpecialHooksEncoder.NEO_BERT, SpecialTensorRequirementsEncoder.neoBert()),
                ArchEntry.of(encoder(Arch.EUROBERT, RopeType.NEOX).build(),
                        SpecialHooksEncoder.EUROBERT, SpecialTensorRequirementsEncoder.eurobert()),
                // gemma-embedding.cpp:7, gemma-embedding2.cpp:7: always bidirectional; gemma sandwich norms
                ArchEntry.of(gemmaEmbedding(Arch.GEMMA_EMBEDDING),
                        SpecialHooksEncoder.GEMMA_EMBEDDING, SpecialTensorRequirementsEncoder.gemmaEmbedding()),
                ArchEntry.of(gemmaEmbedding(Arch.GEMMA_EMBEDDING2),
                        SpecialHooksEncoder.GEMMA_EMBEDDING2, SpecialTensorRequirementsEncoder.gemmaEmbedding2()),
                // t5.cpp:231-232: GELU-gated FFN when ffn_gate exists, ReLU otherwise
                ArchEntry.of(ArchTraits.builder(Arch.T5).ropeType(RopeType.NONE).memoryKind(MemoryKind.ENC_DEC)
                                .ffnAct(ActOp.GELU_TANH).support(SupportClass.ENC_DEC).build(),
                        SpecialHooksEncoder.T5, SpecialTensorRequirementsEncoder.t5()),
                ArchEntry.of(encoder(Arch.T5ENCODER, RopeType.NONE).forcedNonCausal(true).ffnAct(ActOp.GELU_TANH)
                                .graphFamily(Arch.T5).build(),
                        SpecialHooksEncoder.T5ENCODER, SpecialTensorRequirementsEncoder.t5Encoder()),
                // models.h:195-203: llama hparams, tensors and the cache-less embedding graph (llama.cpp:113-119)
                derived(bases, encoder(Arch.LLAMA_EMBED, RopeType.NORM).causalDefault(true).graphFamily(Arch.LLAMA)
                                .build(), Arch.LLAMA,
                        ArchHparamsHook.NONE, Collections.<TensorRequirement>emptyList()),
                // vllm-decision2.cpp:8-39: qwen3 backbone with a per-option decision head
                derived(bases, ArchTraits.builder(Arch.VLLM_DECISION2).ropeType(RopeType.NEOX)
                                .memoryKind(MemoryKind.KV).support(SupportClass.CLS).graphFamily(Arch.QWEN3).build(),
                        Arch.QWEN3, SpecialHooksHeads.VLLM_DECISION2_HEAD,
                        SpecialTensorRequirementsHeads.vllmDecision2Head()),
                // clef.cpp:8-98: qwen35 backbone without memory and a joint decision head
                derived(bases, ArchTraits.builder(Arch.CLEF).ropeType(RopeType.IMROPE).memoryKind(MemoryKind.NONE)
                                .postNormRole(PostNormRole.PRE_FFN).support(SupportClass.CLS)
                                .graphFamily(Arch.QWEN35).build(),
                        Arch.QWEN35, SpecialHooksHeads.CLEF_HEAD, SpecialTensorRequirementsHeads.clefHead()));
    }

    /** EmbeddingGemma traits: forced bidirectional, GELU-gated FFN, sandwich norms, no memory. */
    private static ArchTraits gemmaEmbedding(Arch arch) {
        return encoder(arch, RopeType.NEOX).forcedNonCausal(true).ffnAct(ActOp.GELU_TANH)
                .postNormRole(PostNormRole.SANDWICH).build();
    }
}
