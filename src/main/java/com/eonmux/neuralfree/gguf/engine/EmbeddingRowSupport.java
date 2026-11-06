package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Whether a model's forward pass can take precomputed embedding rows ({@code prompt_embeds}, media embeddings) in
 * place of token rows, and what an embedding row stands for where a block reads token ids. An embedding row carries
 * no token id (llama.cpp {@code LLAMA_TOKEN_NULL}); the blocks that read ids follow llama.cpp:
 * <ul>
 *   <li>the per-layer token embeddings of gemma3n and gemma4 read the padding token's row (gemma3n.cpp:329-349,
 *       gemma4.cpp:490-509);</li>
 *   <li>the qwen4exp n-gram per-layer embedding hashes a stand-in id, {@code ple.image_token_id} or the EOS id when
 *       that is 0, and records it as the row's history token (qwen4exp.cpp:1326-1335, llama-kv-cache.cpp:1192-1200;
 *       {@link #recordToken});</li>
 *   <li>the deepseek4 hash-routed layers route an embedding row through the learned router instead of
 *       {@code ffn_gate_tid2eid} (deepseek4.cpp:1290-1301).</li>
 * </ul>
 * Switched-LoRA routing and value embeddings have no embedding-row path in llama.cpp (granite-switch.cpp:231,
 * limite.cpp:86), and LongCat-Flash-Lite's n-gram input embedding hashes the row's token history
 * (longcat_flash_ngram.py:298-374), so a model with any of them refuses embedding rows by name. Shared by the CPU and GPU forward backends.
 */
public final class EmbeddingRowSupport {

    /** The token id an embedding row records on a model whose blocks read none: no token. */
    public static final int NO_TOKEN = -1;

    private EmbeddingRowSupport() {
    }

    /**
     * Null when every block of {@code spec} runs on embedding rows, else the reason naming the first block that
     * reads token ids without an embedding-row definition.
     */
    public static String refusal(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        for (LayerSpec l : spec.layers()) {
            String reader = tokenIdReader(l);
            if (reader != null) {
                return spec.arch().ggufName() + " layer " + l.il() + " reads the step's token ids (" + reader
                        + "); embedding rows carry none";
            }
        }
        return null;
    }

    /**
     * The id an embedding row records in the sequence's token record: the n-gram stand-in on a model with n-gram
     * per-layer embeddings ({@link #ngramStandIn}), else {@link #NO_TOKEN}.
     */
    public static int recordToken(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        for (LayerSpec l : spec.layers()) {
            if (l.hasExtra(LayerSpec.Extras.NGRAM_PLE)) {
                return ngramStandIn(spec.hparams());
            }
        }
        return NO_TOKEN;
    }

    /**
     * The id the n-gram per-layer embedding hashes for an embedding row: {@code ple.image_token_id}, or
     * {@code ple.eos_token_id} when the file gives no image token (llama.cpp {@code ple_embd_token},
     * qwen4exp.cpp:118-120).
     */
    public static int ngramStandIn(ModelHparams hp) {
        Objects.requireNonNull(hp, "hp");
        return hp.pleImageTokenId() != 0 ? hp.pleImageTokenId() : hp.pleEosTokenId();
    }

    private static String tokenIdReader(LayerSpec l) {
        if (l.hasExtra(LayerSpec.Extras.SWITCHED_LORA)) {
            return "switched-LoRA routing";
        }
        if (l.attention() != null && l.attention().valueEmbeddings()) {
            return "value embeddings";
        }
        if (l.hasExtra(LayerSpec.Extras.NGRAM_EMBD)) {
            return "the n-gram input embedding";
        }
        return null;
    }
}
