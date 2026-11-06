package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * The tokenizer of a CLIP two-tower file's text tower. llama.cpp's legacy converter stores only the token list
 * ({@code tokenizer.ggml.tokens}, no {@code tokenizer.ggml.model} and no BPE merges), which cannot tokenize, so
 * the tokenizer comes from, in order: the tokenizer source of the load options ({@code neuralfree.hf.tokenizer},
 * vLLM's {@code --tokenizer}) — a Hugging Face checkpoint directory whose {@code tokenizer.json} (or
 * {@code vocab.json} + {@code merges.txt}) converts as the GPT-2 BPE family does, the CLIP end-of-word suffix,
 * normalizer and post-processor included, or a GGUF file whose {@code tokenizer.*} metadata is used; then the
 * file's own {@code tokenizer.*} metadata when it names a tokenizer model. Without either the text tower has no
 * tokenizer and text inputs refuse by name. Stateless.
 */
final class ClipDualVocab {

    /** The configuration the vocabulary writer reads; the vocabulary bound comes from the embedding rows. */
    private static final String CONFIG = "{\"architectures\": [\"CLIPModel\"]}";

    private ClipDualVocab() {
    }

    /**
     * The tokenizer of the text tower, or null when neither the tokenizer source nor the file carries one.
     *
     * @param tokenizerSource the tokenizer directory or GGUF file of the load options, or null
     * @param vocabRows       the rows of the token embedding, which bound the vocabulary
     * @throws IOException when the tokenizer source cannot be read or converted
     */
    static GGUFTokenizer resolve(ModelSource source, GgufKv kv, Path tokenizerSource, int vocabRows)
            throws IOException {
        if (tokenizerSource != null) {
            return new GGUFTokenizer(TokenizerVocab.fromMetadata(fromSource(tokenizerSource, vocabRows)));
        }
        if (kv.has(GgufKeys.TOKENIZER_MODEL)) {
            return new GGUFTokenizer(source);
        }
        return null;
    }

    /** Why the text tower has no tokenizer, naming the way to supply one. */
    static String missingReason(String path) {
        return "the CLIP file '" + path + "' carries no tokenizer model (no " + GgufKeys.TOKENIZER_MODEL
                + ": llama.cpp's legacy convert_image_encoder_to_gguf.py writes the vocabulary without the BPE"
                + " merges, and a SigLIP checkpoint without its spiece.model has none); pass the checkpoint's"
                + " tokenizer directory with -D" + com.eonmux.neuralfree.gguf.ModelSourceOptions.PROP_TOKENIZER
                + "=<dir>, or send token ids";
    }

    private static Map<String, Object> fromSource(Path tokenizerSource, int vocabRows) throws IOException {
        HfTokenizerFiles files = Files.isDirectory(tokenizerSource) ? HfTokenizerFiles.locate(tokenizerSource)
                : HfTokenizerFiles.empty().withTokenizerDir(tokenizerSource);
        Path dir = Files.isDirectory(tokenizerSource) ? tokenizerSource : tokenizerSource.toAbsolutePath().getParent();
        HfVocabWriters.Input in = HfVocabWriters.Input.of(HfConfigLoader.fromJson(dir, CONFIG), files,
                PreTokenizerCanonicalizer::toSpec).withEmbeddingRows(vocabRows);
        return HfVocabWriters.gpt2(in, new HfPreTokenizerResolver());
    }
}
