package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.Zamba2Spec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Load-time checks of a model against its architecture, in the order {@link ModelSpecBuilder} runs them:
 * <ol>
 *   <li>{@link #checkSupported}: an architecture of {@link SupportClass#NA} is rejected with
 *       {@link ArchTraits#unsupportedReason()};</li>
 *   <li>{@link #requireTensors}: every required entry of the architecture's tensor list ({@code gguf.spec.req},
 *       {@link ArchEntry#tensors()}) is present, over each entry's own layer slots
 *       ({@link TensorRequirement#layerSlots}); all unmet entries are reported together;</li>
 *   <li>{@link #checkGraphAsserts}: the {@code GGML_ASSERT}s llama.cpp's graph builders make on the hparams;</li>
 *   <li>{@link #checkShapes}: the token embedding, the untied output and the separate Q/K/V/O projections of the
 *       softmax-attention layers agree with the hparams of the resolved spec and the vocabulary size
 *       ({@link #vocabTokens}).</li>
 * </ol>
 * The MTP drafter of a draft file ({@code ModelSpecBuilder.buildMtpDraft}) runs the same checks with its trunk optional
 * and its shapes checked over the NextN blocks ({@link #checkMtpDraftShapes}). Stateless.
 */
public final class ModelSpecValidator {

    /**
     * Graphs asserting {@code n_embd_head_v == n_embd_head_k == n_rot} on the layer-0 values
     * ({@code hparams.n_embd_head_k()}, {@code hparams.n_rot()}), with the architectures that reuse them through
     * models.h ({@code using graph = ...} or a derived model class): models/llama.cpp:102-103 (llama, llama-embed),
     * granite.cpp:130-131 (granite, granitemoe, minicpm), hunyuan-vl.cpp:63-64 (hunyuan_vl, hunyuan-dense),
     * qwen3.cpp:70-71 (qwen3, vllm-decision2 through {@code build_backbone}), qwen3vl.cpp:66-67 (qwen3vl, qwen3tts),
     * qwen3vlmoe.cpp:70 (interns1-pro, which reuses that graph through {@link ArchTraits#graphFamily()}),
     * and the own graphs of apertus.cpp:63, arcee.cpp:51, arctic.cpp:59, baichuan.cpp:49, chameleon.cpp:56,
     * codeshell.cpp:56, cogvlm.cpp:58, cohere2moe.cpp:146, dbrx.cpp:50, deci.cpp:83, deepseek.cpp:78, dots1.cpp:78,
     * dream.cpp:57, ernie4-5.cpp:80, ernie4-5-moe.cpp:11, exaone.cpp:50, exaone4.cpp:86, exaone-moe.cpp:112,
     * falcon.cpp:53, flint.cpp:81, granite-swa.cpp:128, granite-switch.cpp:234, grok.cpp:89, grovemoe.cpp:72,
     * hunyuan-moe.cpp:59, hy-v3.cpp:100, internlm2.cpp:44, jais2.cpp:57, llada.cpp:70, llada-moe.cpp:57, llama4.cpp:108,
     * maincoder.cpp:50, maion-coder.cpp:74, mellum.cpp:111, mistral3.cpp:133, olmo.cpp:46, olmo2.cpp:66, olmoe.cpp:56,
     * orion.cpp:46, paddleocr.cpp:14, pangu-embed.cpp:61, plamo.cpp:42, pockettts.cpp:51, qwen2.cpp:56, qwen2moe.cpp:68,
     * qwen2vl.cpp:45, qwen3moe.cpp:65, qwen3vlmoe.cpp:70, rnd1.cpp:68, seed-oss.cpp:52, smallthinker.cpp:78,
     * smollm3.cpp:49, starcoder2.cpp:62, talkie.cpp:43 and xverse.cpp:44.
     */
    private static final Set<Arch> ASSERT_HEAD_ROT = Collections.unmodifiableSet(EnumSet.of(
            Arch.LLAMA, Arch.LLAMA_EMBED, Arch.GRANITE, Arch.GRANITE_MOE, Arch.MINICPM, Arch.HUNYUAN_VL,
            Arch.HUNYUAN_DENSE, Arch.QWEN3, Arch.VLLM_DECISION2, Arch.QWEN3VL, Arch.QWEN3TTS, Arch.APERTUS,
            Arch.ARCEE, Arch.ARCTIC, Arch.BAICHUAN, Arch.CHAMELEON, Arch.CODESHELL, Arch.COGVLM, Arch.COHERE2MOE,
            Arch.DBRX, Arch.DECI, Arch.DEEPSEEK, Arch.DOTS1, Arch.DREAM, Arch.ERNIE4_5, Arch.ERNIE4_5_MOE,
            Arch.EXAONE, Arch.EXAONE4, Arch.EXAONE_MOE, Arch.FALCON, Arch.FLINT, Arch.GRANITE_SWA,
            Arch.GRANITE_SWITCH, Arch.GROK, Arch.GROVEMOE, Arch.HUNYUAN_MOE, Arch.HY_V3, Arch.INTERNLM2,
            Arch.INTERN_S1_PRO, Arch.JAIS2,
            Arch.LLADA, Arch.LLADA_MOE, Arch.LLAMA4, Arch.MAINCODER, Arch.MAION_CODER, Arch.MELLUM, Arch.MISTRAL3,
            Arch.OLMO, Arch.OLMO2, Arch.OLMOE, Arch.ORION, Arch.PADDLEOCR, Arch.PANGU_EMBED, Arch.PANGU_MOE,
            Arch.PLAMO,
            Arch.POCKETTTS, Arch.QWEN2, Arch.QWEN2MOE, Arch.QWEN2VL, Arch.QWEN3MOE, Arch.QWEN3VLMOE, Arch.RND1,
            Arch.SEED_OSS, Arch.SMALLTHINKER, Arch.SMOLLM3, Arch.STARCODER2, Arch.TALKIE, Arch.XVERSE));

    /**
     * Graphs asserting only {@code n_embd_head_v == n_embd_head_k} (layer 0): the BERT graph (bert.cpp:71; bert,
     * jina-bert-v2/v3, nomic-bert, nomic-bert-moe), phi3.cpp:71 (phi3, phimoe), t5.cpp:114, 266 (t5, t5encoder),
     * nemotron-h.cpp:205 (nemotron_h, nemotron_h_moe), and the own graphs of afmoe.cpp:109, bailingmoe2.cpp:94,
     * beam.cpp:117, bitnet.cpp:54, bloom.cpp:68, chatglm.cpp:61, clef.cpp:249, cohere2.cpp:53, command-r.cpp:50,
     * dflash.cpp:811, eagle3.cpp:155, eurobert.cpp:41, glm4.cpp:71, glm4-moe.cpp:288, gpt2.cpp:61, gptneox.cpp:90,
     * granite-hybrid.cpp:138, jais.cpp:58, k2-horizon.cpp:215, kolibri1.cpp:70, laguna.cpp:157, maple.cpp:69,
     * minimax-01.cpp:195, minimax-m2.cpp:50, minimax-m3.cpp:214, modern-bert.cpp:141, mpt.cpp:70, muse-glimmer.cpp:60,
     * muse-spark.cpp:81, nanbeige.cpp:85, nemotron.cpp:53, neo-bert.cpp:46, openelm.cpp:53, phi2.cpp:50, qwen.cpp:46,
     * qwen3next.cpp:240, qwen35.cpp:141, qwen35moe.cpp:160, qwen4exp.cpp:403, refact.cpp:82, sensenova-u1.cpp:79,
     * spark2-5.cpp:59, stablelm.cpp:54 and starcoder.cpp:63, and the deepseek4 attention (deepseek4.cpp:906; its
     * RoPE spans the last {@code n_rot} head dims, which its {@code Dsv4Spec} checks). The minimax graphs leave
     * their {@code n_rot} assert commented out (minimax-m2.cpp:51, minimax-01.cpp:196), as does nemotron.cpp:54.
     */
    private static final Set<Arch> ASSERT_HEAD_KV = Collections.unmodifiableSet(EnumSet.of(
            Arch.BERT, Arch.JINA_BERT_V2, Arch.JINA_BERT_V3, Arch.NOMIC_BERT, Arch.NOMIC_BERT_MOE, Arch.PHI3,
            Arch.PHIMOE, Arch.T5, Arch.T5ENCODER, Arch.NEMOTRON_H, Arch.NEMOTRON_H_MOE, Arch.AFMOE, Arch.BAILINGMOE2,
            Arch.BEAM, Arch.BITNET, Arch.BLOOM, Arch.CHATGLM, Arch.CLEF, Arch.COHERE2, Arch.COMMAND_R, Arch.DEEPSEEK4,
            Arch.DEEPSEEK41, Arch.DFLASH, Arch.EAGLE3, Arch.EUROBERT, Arch.GLM4, Arch.GLM4_MOE, Arch.GPT2, Arch.GPTNEOX,
            Arch.GRANITE_HYBRID, Arch.JAIS, Arch.K2_HORIZON, Arch.KOLIBRI1, Arch.LAGUNA, Arch.MAPLE, Arch.MINIMAX_01,
            Arch.MINIMAX_M2, Arch.MINIMAX_M3, Arch.MODERN_BERT, Arch.MPT, Arch.MUSE_GLIMMER, Arch.MUSE_SPARK,
            Arch.NANBEIGE, Arch.NEMOTRON, Arch.NEO_BERT, Arch.OPENELM, Arch.PHI2, Arch.QWEN, Arch.QWEN3NEXT,
            Arch.QWEN35, Arch.QWEN35MOE, Arch.QWEN4EXP, Arch.REFACT, Arch.SENSENOVA_U1, Arch.SPARK2_5, Arch.STABLELM,
            Arch.STARCODER));

    /**
     * Loaders that take their trunk and MTP flags from {@code nextn_flags} (llama-model.cpp:3901-3920), with the
     * architectures that reuse them through models.h (mistral4 from deepseek2, clef from qwen35, nemotron_h_moe from
     * nemotron_h): cohere2moe.cpp, deepseek2.cpp, deepseek32.cpp, deepseek4.cpp, glm-dsa.cpp, glm4-moe.cpp,
     * glm5-next.cpp, hy-v3.cpp, mellum.cpp, mimo2.cpp, bailingmoe3.cpp, nemotron-h.cpp, qwen3next.cpp, qwen35.cpp,
     * qwen35moe.cpp, qwen4exp.cpp and step35.cpp.
     */
    private static final Set<Arch> NEXTN_FLAGS = Collections.unmodifiableSet(EnumSet.of(
            Arch.COHERE2MOE, Arch.DEEPSEEK2, Arch.MISTRAL4, Arch.DEEPSEEK32, Arch.DEEPSEEK4, Arch.DEEPSEEK41,
            Arch.GLM_DSA,
            Arch.GLM4_MOE, Arch.GLM5_NEXT, Arch.HY_V3, Arch.MELLUM, Arch.MIMO2, Arch.BAILINGMOE3, Arch.NEMOTRON_H,
            Arch.NEMOTRON_H_MOE, Arch.QWEN3NEXT, Arch.QWEN35, Arch.CLEF, Arch.QWEN35MOE, Arch.QWEN4EXP,
            Arch.STEP35));

    /** Untied output heads whose rows are not {@code n_vocab} (eagle3.cpp:67, dflash.cpp:246, qwen3vl.cpp:29). */
    private static final Set<Arch> OUTPUT_VOCAB_DIFFERS = Collections.unmodifiableSet(EnumSet.of(
            Arch.EAGLE3, Arch.DFLASH, Arch.INKLING, Arch.QWEN3VL, Arch.QWEN3TTS, Arch.WAVTOKENIZER_DEC));

    private ModelSpecValidator() {
    }

    /**
     * Rejects an architecture the engine cannot run.
     *
     * @throws UnsupportedArchException with {@link ArchTraits#unsupportedReason()} for {@link SupportClass#NA}
     */
    public static void checkSupported(ArchEntry entry) {
        Objects.requireNonNull(entry, "entry");
        ArchTraits traits = entry.traits();
        if (traits.support() == SupportClass.NA) {
            throw new UnsupportedArchException(entry.arch(), traits.unsupportedReason());
        }
    }

    /**
     * Checks every required tensor of the architecture's list. Layer entries scan their own slots
     * ({@link TensorRequirement#layerSlots}: {@code n_layer_all}, or the decoder / drafter spans of entries over
     * other stacks); names resolve through the catalog's model and layer aliases. In an MTP-only file
     * ({@link #isMtpOnly}) the trunk tensors are not required, as {@code nextn_flags} makes them
     * {@code TENSOR_NOT_REQUIRED}.
     *
     * @throws MissingTensorException listing every unmet entry
     */
    public static void requireTensors(ModelHparams hp, TensorCatalog tc, List<TensorRequirement> requirements) {
        Objects.requireNonNull(hp, "hp");
        Objects.requireNonNull(tc, "tc");
        requireTensors(hp, tc, requirements, isMtpOnly(hp, tc));
    }

    /**
     * {@link #requireTensors(ModelHparams, TensorCatalog, List)} with the trunk tensors optional or not: the MTP
     * drafter of a draft file reads none of the file's trunk ({@code ModelSpecBuilder.buildMtpDraft}), so its trunk
     * is optional whether or not the file holds it.
     *
     * @param trunkOptional true when trunk slots {@code [0, nLayer)} and the trunk-flagged model entries are not
     *                      required
     * @throws MissingTensorException listing every unmet entry
     */
    public static void requireTensors(ModelHparams hp, TensorCatalog tc, List<TensorRequirement> requirements,
                                      boolean trunkOptional) {
        Objects.requireNonNull(hp, "hp");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(requirements, "requirements");
        Predicate<String> present = name -> isPresent(tc, name);
        List<String> missing = new ArrayList<>();
        for (TensorRequirement r : requirements) {
            missing.addAll(r.missing(hp, present, trunkOptional));
        }
        if (!missing.isEmpty()) {
            throw new MissingTensorException(tc.archName(), missing);
        }
    }

    /**
     * True when the file holds only the MTP blocks of its model ({@code nextn_flags}, llama-model.cpp:3901-3920): the
     * architecture's loader uses {@code nextn_flags} ({@link #NEXTN_FLAGS}), NextN blocks are declared, and the trunk
     * probe {@code blk.0.attn_norm.weight} ({@code blk.0.hc_attn_norm.weight} for qwen4exp, qwen4exp.cpp:178) is
     * absent. llama.cpp loads such a file only as the MTP draft context of its target model
     * (common/speculative.cpp:2414-2420, 2660-2661); it cannot run as a model of its own.
     */
    public static boolean isMtpOnly(ModelHparams hp, TensorCatalog tc) {
        Objects.requireNonNull(hp, "hp");
        Objects.requireNonNull(tc, "tc");
        if (!NEXTN_FLAGS.contains(hp.arch()) || hp.nLayerAll() <= hp.nLayer()) {
            return false;
        }
        return !isPresent(tc, trunkProbe(hp.arch()));
    }

    /**
     * True when {@code storedName} is a tensor of a trunk block of a model with {@code nLayer} trunk blocks:
     * {@code blk.N.*} or its Hugging Face form {@code model.layers.N.*} with {@code N < nLayer}. The NextN blocks
     * ({@code N >= nLayer}) and model-level tensors are not.
     */
    public static boolean isTrunkTensor(String storedName, int nLayer) {
        Objects.requireNonNull(storedName, "storedName");
        return blockIndexOf(storedName) >= 0 && blockIndexOf(storedName) < nLayer;
    }

    /**
     * The block index {@code N} of a stored {@code blk.N.*} or Hugging Face {@code model.layers.N.*} tensor name,
     * or -1 for a model-level tensor or a name of another form. The weight-layer key a layer split routes tensors
     * by.
     */
    public static int blockIndexOf(String storedName) {
        Objects.requireNonNull(storedName, "storedName");
        int il = blockIndex(storedName, TensorNames.STACK_BLK + ".");
        if (il < 0) {
            il = blockIndex(storedName, TensorNames.HF_LAYER_PREFIX + ".");
        }
        return il;
    }

    /** The block index {@code N} of {@code <prefix>N.<suffix>}, or -1 when the name has another form. */
    private static int blockIndex(String name, String prefix) {
        if (!name.startsWith(prefix)) {
            return -1;
        }
        int dot = name.indexOf('.', prefix.length());
        if (dot <= prefix.length() || dot == name.length() - 1) {
            return -1;
        }
        try {
            return Integer.parseInt(name.substring(prefix.length(), dot));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /** The first trunk tensor {@code nextn_flags} probes for {@code arch} ({@code trunk_probe} at layer 0). */
    static String trunkProbe(Arch arch) {
        String base = arch == Arch.QWEN4EXP ? TensorNames.HC_ATTN_NORM : TensorNames.ATTN_NORM;
        return TensorNames.layerName(0, TensorNames.weight(base));
    }

    /**
     * True when {@code name} is in the catalog: exactly, through a model-level alias ({@link TensorCatalog#get}), or,
     * for a main-stack {@code blk.N.<suffix>} name, through the Hugging Face layer aliases
     * ({@link TensorCatalog#layer(int, String)}).
     */
    static boolean isPresent(TensorCatalog tc, String name) {
        return find(tc, name) != null;
    }

    /** The stored tensor {@link #isPresent} finds for {@code name}; null when absent. */
    static TensorRef find(TensorCatalog tc, String name) {
        TensorRef direct = tc.get(name);
        if (direct != null) {
            return direct;
        }
        String prefix = TensorNames.STACK_BLK + ".";
        int il = blockIndex(name, prefix);
        if (il < 0) {
            return null;
        }
        return tc.layer(il, name.substring(name.indexOf('.', prefix.length()) + 1));
    }

    /**
     * The hparams asserts of the architecture's llama.cpp graph: equal K/V head sizes (and RoPE over the whole head)
     * on layer 0 ({@link #ASSERT_HEAD_KV}, {@link #ASSERT_HEAD_ROT}), and the single decoder layer of eagle3
     * (eagle3.cpp:156).
     *
     * @throws GgufFormatException naming the architecture, the values and the graph source when an assert fails
     */
    public static void checkGraphAsserts(ModelHparams hp) {
        Objects.requireNonNull(hp, "hp");
        Arch arch = hp.arch();
        String name = arch.ggufName();
        boolean rot = ASSERT_HEAD_ROT.contains(arch);
        // an EAGLE, EAGLE-3 or Mistral Large 3 drafter over a DeepSeek target runs MLA, whose key row is kv_lora +
        // rotary and whose value row is kv_lora (deepseek2 has no equal-K/V assertion)
        boolean eagleMla = (arch == Arch.EAGLE || arch == Arch.EAGLE3 || arch == Arch.EAGLE_MISTRAL_LARGE3)
                && hp.isMla();
        // openPangu's static sink attention is vLLM-only: its key head (qk_nope + qk_rope) need not equal its value
        // head (v_channels) and only qk_rope of the key is rotated, so the pangu-embedded graph's asserts do not hold
        boolean sinkTrunk = arch == Arch.PANGU_MOE && hp.sinkTokenCount() > 0;
        if ((rot || ASSERT_HEAD_KV.contains(arch)) && !eagleMla && !sinkTrunk) {
            int headK = hp.headK(0);
            int headV = hp.headV(0);
            if (headK != headV) {
                throw new GgufFormatException(name + ": the llama.cpp graph requires equal K and V head sizes, got "
                        + headK + " and " + headV + " (GGML_ASSERT n_embd_head == n_embd_head_k, src/models)");
            }
            if (rot && hp.nRot(0) != headV) {
                throw new GgufFormatException(name + ": the llama.cpp graph rotates the whole head, but"
                        + " rope.dimension_count is " + hp.nRot(0) + " for heads of " + headV
                        + " (GGML_ASSERT n_embd_head == n_rot, src/models)");
            }
        }
        if (arch == Arch.EAGLE3 && hp.nLayer() != 1) {
            throw new GgufFormatException(name + ": the eagle3 drafter has exactly one decoder layer, got "
                    + hp.nLayer() + " (eagle3.cpp:156)");
        }
    }

    /**
     * The vocabulary size llama.cpp loads the embedding and output tensors with ({@code vocab.n_tokens()}): the
     * {@code vocab_size} key for a vocabulary-less tokenizer ({@code no_vocab} / {@code none},
     * llama-vocab.cpp:2206-2225), else the length of {@code tokenizer.ggml.tokens} (llama-vocab.cpp:2731-2778); a
     * file without a token list falls back to {@link ModelHparams#nVocab()}. 0 when unknown.
     */
    public static int vocabTokens(GgufKv kv, ModelHparams hp) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(hp, "hp");
        String model = kv.str(GgufKeys.TOKENIZER_MODEL, null);
        if ("no_vocab".equals(model) || "none".equals(model)) {
            return kv.u32(GgufKeys.VOCAB_SIZE, 0);
        }
        int listed = kv.arrayLength(GgufKeys.TOKENIZER_LIST);
        return listed >= 0 ? listed : Math.max(0, hp.nVocab());
    }

    /**
     * Shapes of the resolved spec against its hparams: {@code token_embd} is {@code [n_embd, nVocab]}
     * (eagle3: the target vocabulary), an untied {@code output} reads {@code n_embd} values and, for heads over
     * the model vocabulary, writes {@code nVocab} rows, and every trunk and NextN layer with softmax attention over
     * separate projections has {@code attn_q} rows of {@code n_head * head_k} (twice that with an interleaved
     * gate), {@code attn_k} rows of {@code n_head_kv * head_k}, {@code attn_v} rows of {@code n_head_kv * head_v}
     * and {@code attn_output} columns of {@code n_head * head_v}. Absent optional tensors are not checked.
     *
     * @param nVocab the vocabulary size of the file ({@link #vocabTokens}); 0 skips the vocabulary checks
     * @throws GgufFormatException naming the tensor, its shape and the expected size
     */
    public static void checkShapes(ModelSpec spec, int nVocab) {
        checkModelShapes(spec, nVocab);
        for (LayerSpec l : spec.layers()) {
            checkAttentionShapes(spec, l);
            checkZamba2Shapes(spec, l);
        }
        for (LayerSpec l : spec.nextnLayers()) {
            checkAttentionShapes(spec, l);
            checkZamba2Shapes(spec, l);
        }
    }

    /**
     * {@link #checkShapes} for the spec of an MTP drafter ({@code ModelSpecBuilder.buildMtpDraft}): the token
     * embedding, the output head and the NextN layers; the trunk slots are the target's layers, checked with the
     * target.
     *
     * @throws GgufFormatException naming the tensor, its shape and the expected size
     */
    public static void checkMtpDraftShapes(ModelSpec spec, int nVocab) {
        checkModelShapes(spec, nVocab);
        for (LayerSpec l : spec.nextnLayers()) {
            checkAttentionShapes(spec, l);
        }
    }

    private static void checkModelShapes(ModelSpec spec, int nVocab) {
        Objects.requireNonNull(spec, "spec");
        if (nVocab < 0) {
            throw new IllegalArgumentException("vocabulary size must be >= 0 (0 skips its checks), got " + nVocab);
        }
        ModelHparams hp = spec.hparams();
        Arch arch = spec.arch();
        TensorRef embd = spec.embedding().tokenEmbd();
        if (embd != null) {
            expect(arch, embd, 0, spec.embedding().tokenWidth(), "the token row width");
            if (arch == Arch.INKLING) {
                // vLLM allocates the embedding over padded_vocab_size and slices the padded logits off; the padding
                // rows extend the real vocabulary, and the LM head carries the same rows.
                TensorRef paddedHead = spec.head().output();
                if (nVocab > 0 && embd.ne(1) < nVocab) {
                    throw new GgufFormatException(arch.ggufName() + ": tensor '" + embd.name() + "' has "
                            + embd.ne(1) + " rows, fewer than the vocabulary of " + nVocab);
                }
                if (paddedHead != null) {
                    expect(arch, embd, 1, paddedHead.ne(1), "the LM head's row count");
                }
            } else if (nVocab > 0 && arch != Arch.EAGLE3) {
                expect(arch, embd, 1, nVocab, "n_vocab");
            }
        }
        TensorRef embdProjection = spec.embedding().inputProjection();
        if (embdProjection != null) {
            expect(arch, embdProjection, 1, hp.nEmbd(), "n_embd");
        }
        HeadSpec head = spec.head();
        TensorRef out = head.output();
        if (out != null && !head.tied() && arch != Arch.WAVTOKENIZER_DEC) {
            expect(arch, out, 0, head.headInputWidth(hp.nEmbd()), "the head input width");
            TensorRef d2t = head.d2t();
            if (d2t != null) {
                // a drafter's reduced (FR-Spec) head has one row per d2t entry, not the vocabulary
                // (eagle3.cpp:47-50, qwen4exp.cpp:287-296)
                expect(arch, out, 1, d2t.ne(0), "d2t");
            } else if (nVocab > 0 && !OUTPUT_VOCAB_DIFFERS.contains(arch)) {
                expect(arch, out, 1, nVocab, "n_vocab");
            }
        }
        TensorRef proj = head.embeddingProjection();
        if (proj != null) {
            expect(arch, proj, 0, hp.nEmbd(), "n_embd");
            expect(arch, proj, 1, hp.nEmbdOut(), "n_embd_out");          // gemma-embedding2.cpp:41
        }
        TensorRef outProj = head.outputProjection();
        if (outProj != null) {
            expect(arch, outProj, 0, hp.nEmbd(), "n_embd");
        }
    }

    private static void checkAttentionShapes(ModelSpec spec, LayerSpec l) {
        AttentionSpec a = l.attention();
        if (a == null || l.mla() != null || a.linearReplace() || a.fusedQkv()
                || !TensorNames.STACK_BLK.equals(l.stack())) {
            return;
        }
        TensorCatalog tc = spec.tensors();
        Arch arch = spec.arch();
        int wl = l.weightLayer();
        long qRows = a.gate() == AttnGateKind.Q_INTERLEAVED ? 2L * a.qDim() : a.qDim();
        if (a.qLatentDim() > 0) {
            TensorRef qa = tc.layer(wl, TensorNames.weight(TensorNames.ATTN_Q_A));
            expectLayer(arch, qa, 1, a.qLatentDim(), "shared query width", l);
            expectLayer(arch, qa, 0, spec.hparams().nEmbd(), "shared query input width", l);
            TensorRef qb = tc.layer(wl, TensorNames.weight(TensorNames.ATTN_Q_B));
            expectLayer(arch, qb, 1, a.qDim(), "query head width", l);
            expectLayer(arch, qb, 0, a.qLatentDim(), "shared query width", l);
            TensorRef qn = tc.layer(wl, TensorNames.weight(TensorNames.ATTN_Q_A_NORM));
            if (qn != null && qn.numElements() != a.qLatentDim()) {
                throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + qn.name()
                        + "' has " + qn.numElements() + " values, expected the shared query width " + a.qLatentDim());
            }
        } else {
            expectLayer(arch, tc.layer(wl, TensorNames.weight(TensorNames.ATTN_Q)), 1, qRows, "Q width", l);
        }
        expectLayer(arch, tc.layer(wl, TensorNames.weight(TensorNames.ATTN_K)), 1, a.kDim(), "K width", l);
        expectLayer(arch, tc.layer(wl, TensorNames.weight(TensorNames.ATTN_V)), 1, a.vDim(), "V width", l);
        expectLayer(arch, tc.layer(wl, TensorNames.weight(TensorNames.ATTN_OUT)), 0, a.outDim(),
                "attention output width", l);
        checkStaticSinks(spec, l, a);
    }

    /**
     * The static sink tensors of an openPangu sink layer: {@code attn_sink_k} is required and holds
     * {@code n_sink * n_head_kv} key rows of {@code headDimK} values, {@code attn_sink_v} optionally holds the
     * matching value rows (absent ones are zeros, as vLLM's non-persistent buffer is). The layer's K norm is what
     * prepares both the projected keys and the sink keys, so the graph needs it.
     */
    private static void checkStaticSinks(ModelSpec spec, LayerSpec l, AttentionSpec a) {
        if (a.sinkTokens() == 0) {
            return;
        }
        Arch arch = spec.arch();
        TensorCatalog tc = spec.tensors();
        int wl = l.weightLayer();
        long rows = (long) a.sinkTokens() * a.nHeadKv();
        TensorRef key = tc.layer(wl, TensorNames.weight(TensorNames.ATTN_SINK_K));
        if (key == null) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": attention.sink_token_count = "
                    + a.sinkTokens() + " needs tensor " + TensorNames.weight(TensorNames.ATTN_SINK_K));
        }
        expectLayer(arch, key, 0, a.headDimK(), "sink key width", l);
        expectLayer(arch, key, 1, rows, "sink key count", l);
        TensorRef value = tc.layer(wl, TensorNames.weight(TensorNames.ATTN_SINK_V));
        expectLayer(arch, value, 0, a.headDimV(), "sink value width", l);
        expectLayer(arch, value, 1, rows, "sink value count", l);
        if (a.qkNorm() == QkNormKind.NONE) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": static sink attention needs the"
                    + " K norm in " + TensorNames.weight(TensorNames.ATTN_K_NORM));
        }
    }

    /**
     * Shapes of a Zamba2 layer's shared transformer block and per-use adapters: the block reads the
     * {@code 2 * n_embd} concatenation of the residual stream and the model input, its MLP maps {@code n_embd} to
     * twice the feed-forward length, and {@code shared_linear} maps the block output back to {@code n_embd}.
     *
     * @throws GgufFormatException naming the tensor, its shape and the expected one
     */
    private static void checkZamba2Shapes(ModelSpec spec, LayerSpec l) {
        Zamba2Spec z = l.zamba2();
        if (z == null) {
            return;
        }
        ModelHparams hp = spec.hparams();
        TensorCatalog tc = spec.tensors();
        Arch arch = spec.arch();
        int nEmbd = hp.nEmbd();
        expectLayer(arch, tc.layer(l.weightLayer(), TensorNames.weight(TensorNames.ATTN_NORM)), 0, nEmbd,
                "n_embd", l);
        if (!z.hybrid()) {
            return;
        }
        int b = z.sharedBlock();
        long attnIn = 2L * nEmbd;
        AttentionSpec a = l.attention();
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_NORM)), 0,
                attnIn, "the concatenated block input 2 * n_embd", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_Q)), 0,
                attnIn, "the concatenated block input 2 * n_embd", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_Q)), 1,
                a.qDim(), "Q width", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_K)), 1,
                a.kDim(), "K width", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_V)), 1,
                a.vDim(), "V width", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_OUT)), 0,
                a.outDim(), "attention output width", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, TensorNames.weight(TensorNames.ATTN_OUT)), 1,
                nEmbd, "n_embd", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, "attn_pre_ff_norm.weight"), 0, nEmbd,
                "n_embd", l);
        TensorRef gateUp = tc.layer(TensorNames.STACK_SHARED_BLK, b, "ffn_gate_up.weight");
        expectShared(arch, gateUp, 0, nEmbd, "n_embd", l);
        if (gateUp != null && gateUp.ne(1) % 2 != 0) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + gateUp.name()
                    + "' has " + gateUp.ne(1) + " rows, expected twice the feed-forward length");
        }
        long nFf = gateUp != null ? gateUp.ne(1) / 2 : 0;
        if (gateUp != null && hp.nFf(l.il()) > 0 && nFf != hp.nFf(l.il())) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + gateUp.name()
                    + "' has " + nFf + " gate/up rows per half, expected the feed-forward length " + hp.nFf(l.il()));
        }
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, "ffn_down.weight"), 0, nFf,
                "the feed-forward length", l);
        expectShared(arch, tc.layer(TensorNames.STACK_SHARED_BLK, b, "ffn_down.weight"), 1, nEmbd, "n_embd", l);
        int rank = z.adapterRank();
        expectLayer(arch, tc.layer(l.weightLayer(), TensorNames.weight("shared_linear")), 0, nEmbd, "n_embd", l);
        expectLayer(arch, tc.layer(l.weightLayer(), TensorNames.weight("shared_linear")), 1, nEmbd, "n_embd", l);
        expectAdapter(arch, tc.layer(l.weightLayer(), TensorNames.weight("ffn_gate_up_adapter_a")), nEmbd, rank, l);
        expectAdapter(arch, tc.layer(l.weightLayer(), TensorNames.weight("ffn_gate_up_adapter_b")), rank,
                gateUp != null ? gateUp.ne(1) : 0, l);
        if (z.attentionAdapter()) {
            String[] heads = {"attn_q", "attn_k", "attn_v"};
            int[] widths = {a.qDim(), a.kDim(), a.vDim()};
            for (int i = 0; i < heads.length; i++) {
                expectAdapter(arch, tc.layer(l.weightLayer(), TensorNames.weight(heads[i] + "_adapter_a")), attnIn,
                        rank, l);
                expectAdapter(arch, tc.layer(l.weightLayer(), TensorNames.weight(heads[i] + "_adapter_b")), rank,
                        widths[i], l);
            }
        }
    }

    /** {@link #expectLayer} over a tensor of another stack, naming the layer in the message. */
    private static void expectShared(Arch arch, TensorRef t, int dim, long expected, String what, LayerSpec l) {
        if (t != null && t.ne(dim) != expected) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + t.name() + "' has ne["
                    + dim + "] = " + t.ne(dim) + ", expected the " + what + " " + expected);
        }
    }

    /** A projection of {@code cols} inputs and {@code rows} outputs, checked when present. */
    private static void expectAdapter(Arch arch, TensorRef t, long cols, long rows, LayerSpec l) {
        if (t == null) {
            return;
        }
        if (t.ne(0) != cols || t.ne(1) != rows) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + t.name() + "' is ["
                    + t.ne(0) + " x " + t.ne(1) + "], expected [" + cols + " x " + rows + "]");
        }
    }

    private static void expect(Arch arch, TensorRef t, int dim, long expected, String what) {
        if (t.ne(dim) != expected) {
            throw new GgufFormatException(arch.ggufName() + ": tensor '" + t.name() + "' has ne[" + dim + "] = "
                    + t.ne(dim) + ", expected " + what + " = " + expected);
        }
    }

    private static void expectLayer(Arch arch, TensorRef t, int dim, long expected, String what, LayerSpec l) {
        if (t != null && t.ne(dim) != expected) {
            throw new GgufFormatException(arch.ggufName() + " layer " + l.il() + ": tensor '" + t.name() + "' has ne["
                    + dim + "] = " + t.ne(dim) + ", expected the " + what + " " + expected + " of the hparams");
        }
    }
}
