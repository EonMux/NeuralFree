package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.Function;

/**
 * The converter's vocabulary writers (LC/conversion/base.py:2153-2800, t5.py:30-117, bert.py:40-67): each turns the
 * Hugging Face tokenizer files of a checkpoint into the GGUF tokenizer metadata llama.cpp writes for it, plus the
 * {@code neuralfree.tokenizer.*} overlays ({@link HfTokenOverlays}) and the chat templates ({@link HfChatTemplates}).
 *
 * <p>Every writer returns an unmodifiable map of GGUF keys to values typed as the GGUF reader decodes them
 * ({@code String}, {@code Integer}, {@code Float}, {@code Boolean}, {@code List}). When the tokenizer source is a
 * GGUF file, every writer returns its {@code tokenizer.*} keys instead ({@link #fromGguf}). A tokenizer.json with a
 * {@code WordLevel} model is rejected: llama.cpp has no word-level vocabulary.</p>
 */
public final class HfVocabWriters {

    private static final String WORD_LEVEL_MESSAGE =
            "WordLevel tokenizers are not supported (llama.cpp has no word-level vocabulary)";
    private static final String END_OF_TEXT = "<|endoftext|>";
    private static final String QWEN_TIKTOKEN = "qwen.tiktoken";
    private static final int RWKV_DEFAULT_VOCAB = 65536;
    private static final int RWKV_EOT = 261;
    private static final String RWKV_TEMPLATE = "llama-cpp-rwkv-world.jinja";

    private HfVocabWriters() {}

    /** The inputs every writer reads. Immutable; the withers return copies. */
    public static final class Input {
        private final HfConfig config;
        private final HfTokenizerFiles files;
        private final Function<JsonView, PreTokenizerSpec> preTokenizerSpecs;
        private final OptionalInt embeddingRows;
        private final String ggufArch;
        private final String fallbackChatTemplate;

        private Input(HfConfig config, HfTokenizerFiles files, Function<JsonView, PreTokenizerSpec> specs,
                      OptionalInt embeddingRows, String ggufArch, String fallbackChatTemplate) {
            this.config = Objects.requireNonNull(config, "config");
            this.files = Objects.requireNonNull(files, "files");
            this.preTokenizerSpecs = Objects.requireNonNull(specs, "preTokenizerSpecs");
            this.embeddingRows = Objects.requireNonNull(embeddingRows, "embeddingRows");
            this.ggufArch = ggufArch;
            this.fallbackChatTemplate = fallbackChatTemplate;
        }

        /**
         * @param config            the model configuration
         * @param files             the tokenizer files
         * @param preTokenizerSpecs the exact {@link PreTokenizerSpec} of a tokenizer.json {@code pre_tokenizer}
         *                          (a null view meaning none), used to encode the fingerprint text and stored when no
         *                          llama.cpp pre-tokenizer name matches
         */
        public static Input of(HfConfig config, HfTokenizerFiles files,
                               Function<JsonView, PreTokenizerSpec> preTokenizerSpecs) {
            return new Input(config, files, preTokenizerSpecs, OptionalInt.empty(), null, null);
        }

        /** With the row count of the token embedding, which bounds the vocabulary size. */
        public Input withEmbeddingRows(int rows) {
            if (rows <= 0) throw new IllegalArgumentException("embedding rows " + rows + " is not positive");
            return new Input(config, files, preTokenizerSpecs, OptionalInt.of(rows), ggufArch, fallbackChatTemplate);
        }

        /** With the GGUF architecture name, for the {@code <arch>.vocab_size} key the Mistral writer adds. */
        public Input withGgufArch(String arch) {
            return new Input(config, files, preTokenizerSpecs, embeddingRows, arch, fallbackChatTemplate);
        }

        /** With the chat template used when the checkpoint has none. */
        public Input withFallbackChatTemplate(String template) {
            return new Input(config, files, preTokenizerSpecs, embeddingRows, ggufArch, template);
        }

        public HfConfig config() { return config; }

        public HfTokenizerFiles files() { return files; }

        public Function<JsonView, PreTokenizerSpec> preTokenizerSpecs() { return preTokenizerSpecs; }

        public OptionalInt embeddingRows() { return embeddingRows; }

        public String ggufArch() { return ggufArch; }

        public String fallbackChatTemplate() { return fallbackChatTemplate; }
    }

    // ------------------------------------------------------------------ sources and trivial writers

    /** The {@code tokenizer.*} keys of the GGUF tokenizer source, as stored. */
    public static Map<String, Object> fromGguf(HfTokenizerFiles files) {
        if (!files.hasGgufTokenizer()) throw new IllegalArgumentException("the tokenizer source is not a GGUF file");
        return Collections.unmodifiableMap(new LinkedHashMap<>(files.ggufTokenizerKeys()));
    }

    /** {@code _set_vocab_none} (base.py:2153-2154): no vocabulary. */
    public static Map<String, Object> none(Input in, PreNameResolver pre) {
        if (in.files().hasGgufTokenizer()) return fromGguf(in.files());
        return Collections.singletonMap(GgufKeys.TOKENIZER_MODEL, "none");
    }

    /**
     * {@code _set_vocab_hybriddna} (base.py:2183-2238) builds its k-mer vocabulary with the checkpoint's Python
     * tokenizer code, which NeuralFree does not run; the checkpoint needs a GGUF tokenizer source.
     */
    public static Map<String, Object> hybridDna(Input in, PreNameResolver pre) {
        if (in.files().hasGgufTokenizer()) return fromGguf(in.files());
        throw new GgufFormatException("HybridDNATokenizer vocabularies are built by the checkpoint's Python tokenizer"
                + " code; convert the tokenizer with llama.cpp and pass the GGUF file as the tokenizer source ("
                + ModelSourceOptions.PROP_TOKENIZER + ")");
    }

    /** {@code _set_vocab_builtin} ({@link HfBuiltinVocab}) for a checkpoint without tokenizer files. */
    public static Map<String, Object> builtin(Input in, String name, int vocabSize) throws IOException {
        if (in.files().hasGgufTokenizer()) return fromGguf(in.files());
        Map<String, Object> md = new LinkedHashMap<>(HfBuiltinVocab.of(name, vocabSize));
        md.putAll(HfChatTemplates.resolve(in.files(), in.fallbackChatTemplate()));
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ GPT-2 BPE family

    /** {@code _set_vocab_gpt2} (base.py:2156-2165). */
    public static Map<String, Object> gpt2(Input in, PreNameResolver pre) throws IOException {
        return bpeFamily(in, pre, "gpt2", null, true, SpecialContext::vocab);
    }

    /** {@code _set_vocab_whitespace} (base.py:2167-2176): pre-tokenizer pinned to {@code whitespace}. */
    public static Map<String, Object> whitespace(Input in, PreNameResolver pre) throws IOException {
        return bpeFamily(in, pre, "whitespace", "whitespace", true, SpecialContext::vocab);
    }

    /** {@code _set_vocab_glmedge} (base.py:2562-2574). */
    public static Map<String, Object> glmEdge(Input in, PreNameResolver pre) throws IOException {
        return bpeFamily(in, pre, "gpt2", null, true, s -> s
                .withSpecialToken(HfSpecialVocab.EOS, s.addedId(END_OF_TEXT))
                .withSpecialToken(HfSpecialVocab.EOT, s.addedId("<|user|>"))
                .withSpecialToken(HfSpecialVocab.UNK, s.addedId(END_OF_TEXT))
                .withSpecialToken(HfSpecialVocab.BOS, s.addedId(END_OF_TEXT)));
    }

    /** {@code _set_vocab_glm} (base.py:2576-2590). */
    public static Map<String, Object> glm(Input in, PreNameResolver pre) throws IOException {
        return bpeFamily(in, pre, "gpt2", null, true, s -> s
                .withSpecialToken(HfSpecialVocab.BOS, s.addedId("[gMASK]"))
                .withSpecialToken(HfSpecialVocab.EOT, s.addedId("<|user|>"))
                .withSpecialToken(HfSpecialVocab.UNK, s.addedId(END_OF_TEXT))
                .withSpecialToken(HfSpecialVocab.EOM, s.addedId("<|observation|>")));
    }

    /** {@code _set_vocab_interns1} (base.py:2592-2633): added tokens keep U+2581; bos is 151643. */
    public static Map<String, Object> internS1(Input in, PreNameResolver pre) throws IOException {
        return bpeFamily(in, pre, "gpt2", null, false, s -> s.withSpecialToken(HfSpecialVocab.BOS, 151643));
    }

    /** Special-vocabulary adjustments of one writer, given the added tokens. */
    @FunctionalInterface
    private interface SpecialEdit {
        HfSpecialVocab apply(SpecialContext s);
    }

    /** A special vocabulary with the {@code tokenizer.get_added_vocab()} lookup the writers use. */
    private static final class SpecialContext {
        private final HfSpecialVocab vocab;
        private final Map<String, Integer> added;

        SpecialContext(HfSpecialVocab vocab, Map<String, Integer> added) {
            this.vocab = vocab;
            this.added = added;
        }

        Integer addedId(String content) {
            Integer id = added.get(content);
            if (id == null) throw new GgufFormatException("the tokenizer has no added token '" + content + "'");
            return id;
        }

        HfSpecialVocab vocab() { return vocab; }

        HfSpecialVocab withSpecialToken(String type, Integer id) { return vocab.withSpecialToken(type, id); }
    }

    /**
     * The GPT-2 BPE pipeline: {@code get_vocab_base} tokens, {@code SpecialVocab(load_merges=True)}, the pre-tokenizer
     * name (pinned, or resolved by fingerprint, else the exact spec under {@code default}), the overlays and the chat
     * templates.
     */
    private static Map<String, Object> bpeFamily(Input in, PreNameResolver pre, String model, String pinnedPre,
                                                 boolean metaspaceToSpace, SpecialEdit edit) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        HfTokenizerJson tj = files.readTokenizerJson();
        rejectWordLevel(tj);
        JsonView tc = files.readTokenizerConfig();
        BpeSource src = tj != null ? BpeSource.of(tj, in) : BpeSource.slow(files, tc);
        Map<String, Integer> vocab = src.base.vocab();
        int size = HfVocabBase.vocabSize(in.config().findInt("vocab_size"), vocab.size(), HfVocabBase.maxId(vocab),
                in.embeddingRows(), "tokenizer vocabulary");
        HfVocabBase.Tokens tokens = HfVocabBase.vocabBase(src.base, size,
                HfTokenOverlays.roundTrip(src.decoder, HfTokenOverlays.converterCleanSpaces(tc)), metaspaceToSpace);
        Map<String, Integer> added = new LinkedHashMap<>();
        for (HfTokenizerJson.AddedToken t : src.base.added()) added.put(t.content(), t.id());
        HfSpecialVocab special = edit.apply(new SpecialContext(
                HfSpecialVocab.load(files, tj, in.config().raw(), true, tokens.size()), added));

        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, model);
        tokens.writeTo(md);
        special.writeTo(md);
        HfTokenOverlays.writeTokenAttrs(md, src.base.added(), tokens.size());
        if (src.model != null) {
            HfTokenOverlays.writeBpeModel(md, src.model, tokens.ids());
        } else {
            md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, false);
            md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, false);
        }
        String normalizer = HfTokenOverlays.normalizerJson(tj != null ? tj.normalizer() : null);
        if (normalizer != null) md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, normalizer);
        String decoder = tj != null ? HfTokenOverlays.bpeDecoderJson(tj.decoder(), src.spec.byteEncode()) : null;
        if (decoder != null) md.put(GgufKeys.NF_TOKENIZER_DECODER, decoder);
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, HfTokenOverlays.cleanSpaces(tc, true));

        HfTokenOverlays.PostIds post = HfTokenOverlays.postProcessorIds(tj);
        String name = pinnedPre;
        if (name == null) {
            Map<String, Object> candidate = new LinkedHashMap<>(md);
            candidate.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
            candidate.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, src.spec.toCanonicalJson());
            name = pre.resolve(tj, HfTokenOverlays.encoder(candidate, post));
        }
        if (name != null) {
            md.put(GgufKeys.TOKENIZER_PRE, name);
            if (pinnedPre == null && src.spec.addPrefixSpace()) {
                md.put(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE, true);
            }
        } else {
            md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
            md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, src.spec.toCanonicalJson());
        }
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        HfTokenOverlays.writePostProcessorIds(md, post);
        return Collections.unmodifiableMap(md);
    }

    /** The tokenizer of a GPT-2 BPE writer: its vocabulary, decoder, exact pre-tokenizer and BPE model. */
    private static final class BpeSource {
        final HfVocabBase.BaseVocab base;
        final TokenDecoder decoder;
        final PreTokenizerSpec spec;
        final HfTokenizerJson.Model model;

        private BpeSource(HfVocabBase.BaseVocab base, TokenDecoder decoder, PreTokenizerSpec spec,
                          HfTokenizerJson.Model model) {
            this.base = base;
            this.decoder = decoder;
            this.spec = spec;
            this.model = model;
        }

        static BpeSource of(HfTokenizerJson tj, Input in) {
            if (tj.model().type() != HfTokenizerJson.ModelType.BPE) {
                throw new GgufFormatException(tj.source() + ": the " + tj.model().type().jsonName()
                        + " model is not a BPE vocabulary");
            }
            PreTokenizerSpec spec = in.preTokenizerSpecs().apply(tj.preTokenizer());
            if (spec == null) throw new IllegalStateException("the pre-tokenizer spec function returned null");
            return new BpeSource(new HfVocabBase.BaseVocab(tj.model().vocab(), tj.addedTokens()),
                    HfTokenOverlays.decoder(tj.decoder()), spec, tj.model());
        }

        /** {@code vocab.json} + {@code merges.txt}: a GPT-2 tokenizer with ByteLevel pre-tokenizer and decoder. */
        static BpeSource slow(HfTokenizerFiles files, JsonView tc) throws IOException {
            if (files.vocabJson() == null) {
                throw new GgufFormatException("the checkpoint has no BPE tokenizer (tokenizer.json or vocab.json with"
                        + " merges.txt)");
            }
            String[] base = HfVocabBase.byId(files.readVocabJson(), files.vocabJson());
            boolean prefixSpace = Boolean.TRUE.equals(tc.raw("add_prefix_space"));
            PreTokenizerSpec spec = new PreTokenizerSpec(Collections.singletonList(PreTokenizerSpec.Step.byteLevel()),
                    true, prefixSpace, true, true);
            return new BpeSource(new HfVocabBase.BaseVocab(base, HfSpecialVocab.slowAddedTokens(files, tc, base)),
                    TokenDecoder.byteLevel(prefixSpace, true, true), spec, null);
        }
    }

    // ------------------------------------------------------------------ WordPiece

    /**
     * BERT's {@code set_vocab} (bert.py:40-67): {@code get_vocab_base} tokens with the phantom-space conversion
     * (CONTROL kept, {@code ##x} → {@code x}, others prefixed with U+2581), {@code token_type_count} from
     * {@code type_vocab_size}, model {@code bert} and the fingerprinted pre-tokenizer ({@code default} when none
     * matches).
     */
    public static Map<String, Object> bertWpm(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        HfTokenizerJson tj = files.readTokenizerJson();
        rejectWordLevel(tj);
        JsonView tc = files.readTokenizerConfig();
        HfVocabBase.BaseVocab base;
        TokenDecoder decoder;
        if (tj != null) {
            if (tj.model().type() != HfTokenizerJson.ModelType.WORD_PIECE) {
                throw new GgufFormatException(tj.source() + ": the " + tj.model().type().jsonName()
                        + " model is not a WordPiece vocabulary");
            }
            base = new HfVocabBase.BaseVocab(tj.model().vocab(), tj.addedTokens());
            decoder = HfTokenOverlays.decoder(tj.decoder());
        } else {
            if (files.vocabTxt() == null) throw new GgufFormatException("the checkpoint has no WordPiece tokenizer"
                    + " (tokenizer.json or vocab.txt)");
            String[] ids = files.readVocabTxt().toArray(new String[0]);
            base = new HfVocabBase.BaseVocab(ids, HfSpecialVocab.slowAddedTokens(files, tc, ids));
            decoder = TokenDecoder.wordPiece("##", true);
        }
        Map<String, Integer> vocab = base.vocab();
        int size = HfVocabBase.vocabSize(in.config().findInt("vocab_size"), vocab.size(), HfVocabBase.maxId(vocab),
                in.embeddingRows(), "tokenizer vocabulary");
        HfVocabBase.Tokens tokens = HfVocabBase.vocabBase(base, size,
                HfTokenOverlays.roundTrip(decoder, HfTokenOverlays.converterCleanSpaces(tc)), true)
                .mapTexts((t, type) -> type == HfVocabBase.CONTROL ? t
                        : t.startsWith("##") ? t.substring(2) : HfVocabBase.METASPACE + t);
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, in.config().findInt(1, "type_vocab_size"));
        md.put(GgufKeys.TOKENIZER_MODEL, "bert");
        tokens.writeTo(md);
        HfSpecialVocab.load(files, tj, in.config().raw(), false, tokens.size()).writeTo(md);
        HfTokenOverlays.writeTokenAttrs(md, base.added(), tokens.size());
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, HfTokenOverlays.cleanSpaces(tc, false));
        Map<String, Object> candidate = new LinkedHashMap<>(md);
        candidate.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        String name = pre.resolve(tj, HfTokenOverlays.encoder(candidate, HfTokenOverlays.postProcessorIds(tj)));
        md.put(GgufKeys.TOKENIZER_PRE, name != null ? name : PreTokenizerRegistry.DEFAULT_NAME);
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ SentencePiece

    /** Whether the checkpoint has the {@code tokenizer.model} {@link #sentencePiece} reads. */
    public static boolean canSentencePiece(Input in) {
        return in.files().hasGgufTokenizer() || in.files().file(HfTokenizerFiles.TOKENIZER_MODEL) != null;
    }

    /**
     * {@code _set_vocab_sentencepiece} (base.py:2296-2306, 2308-2396): model {@code llama}, pre {@code default}, the
     * SentencePiece token list sized by {@code vocab_size_per_layer_input} or {@code vocab_size}.
     *
     * @throws GgufFormatException when the checkpoint has no {@code tokenizer.model}
     */
    public static Map<String, Object> sentencePiece(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path model = files.file(HfTokenizerFiles.TOKENIZER_MODEL);
        if (model == null) throw new GgufFormatException("the checkpoint has no " + HfTokenizerFiles.TOKENIZER_MODEL);
        SentencePieceModel sp = SentencePieceModel.read(model);
        HfTokenizerJson tj = files.readTokenizerJson();
        rejectWordLevel(tj);
        JsonView tc = files.readTokenizerConfig();
        OptionalInt configured = in.config().findInt("vocab_size_per_layer_input", "vocab_size");
        if (configured.isPresent() && configured.getAsInt() == 0) configured = OptionalInt.empty();
        int size = HfVocabBase.vocabSize(configured, sp.pieceCount(), -1, in.embeddingRows(), model.toString());
        HfVocabBase.Tokens tokens = HfVocabBase.sentencePiece(sp, size, files.readAddedTokens(), tc);
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "llama");
        md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        tokens.writeTo(md);
        HfSpecialVocab.load(files, tj, in.config().raw(), false, tokens.size()).writeTo(md);
        boolean bpeBackend = tj != null ? tj.model().type() == HfTokenizerJson.ModelType.BPE
                : sp.trainerSpec().modelType() == SentencePieceModel.ModelType.BPE;
        writeSpmOverlays(md, tj, tc, tokens.size(), bpeBackend);
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        HfTokenOverlays.writePostProcessorIds(md, HfTokenOverlays.postProcessorIds(tj));
        return Collections.unmodifiableMap(md);
    }

    /** Whether tokenizer.json is the Llama BPE tokenizer {@link #llamaHf} reads. */
    public static boolean canLlamaHf(Input in) throws IOException {
        return in.files().hasGgufTokenizer() || HfVocabBase.isLlamaHf(in.files().readTokenizerJson());
    }

    /**
     * {@code _set_vocab_llama_hf} (base.py:2398-2418, vocab.py:563-684): model {@code llama}, pre {@code default},
     * the tokenizer.json vocabulary with every score -1000.
     */
    public static Map<String, Object> llamaHf(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        HfTokenizerJson tj = files.readTokenizerJson();
        rejectWordLevel(tj);
        JsonView tc = files.readTokenizerConfig();
        HfVocabBase.Tokens tokens = HfVocabBase.llamaHf(tj,
                HfSpecialVocab.declaredSpecialTokens(tc, files.readSpecialTokensMap()));
        if (in.embeddingRows().isPresent() && tokens.size() > in.embeddingRows().getAsInt()) {
            throw new GgufFormatException(tj.source() + ": " + tokens.size() + " tokens exceed the "
                    + in.embeddingRows().getAsInt() + "-row token embedding");
        }
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "llama");
        md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        tokens.writeTo(md);
        HfSpecialVocab.load(files, tj, in.config().raw(), false, tokens.size()).writeTo(md);
        writeSpmOverlays(md, tj, tc, tokens.size(), true);
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        HfTokenOverlays.writePostProcessorIds(md, HfTokenOverlays.postProcessorIds(tj));
        return Collections.unmodifiableMap(md);
    }

    /**
     * T5's {@code set_vocab} (t5.py:30-117): a Unigram SentencePiece model ({@code tokenizer.model}, else
     * {@code spiece.model}) becomes model {@code t5} with its normalizer flags and charsmap; a BPE model named
     * {@code tokenizer.model} is written by {@link #sentencePiece}.
     */
    public static Map<String, Object> t5Ugm(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path model = files.sentencePieceModel();
        if (model == null) {
            throw new GgufFormatException("the checkpoint has no " + HfTokenizerFiles.TOKENIZER_MODEL + " or "
                    + HfTokenizerFiles.SPIECE_MODEL);
        }
        SentencePieceModel sp = SentencePieceModel.read(model);
        SentencePieceModel.ModelType type = sp.trainerSpec().modelType();
        if (type == SentencePieceModel.ModelType.BPE) {
            if (!model.getFileName().toString().equals(HfTokenizerFiles.TOKENIZER_MODEL)) {
                throw new GgufFormatException(model + ": a BPE SentencePiece model must be named "
                        + HfTokenizerFiles.TOKENIZER_MODEL);
            }
            return sentencePiece(in, pre);
        }
        if (type != SentencePieceModel.ModelType.UNIGRAM) {
            throw new GgufFormatException(model + ": the " + type + " SentencePiece model is not a Unigram model");
        }
        HfTokenizerJson tj = files.readTokenizerJson();
        rejectWordLevel(tj);
        JsonView tc = files.readTokenizerConfig();
        int size = HfVocabBase.vocabSize(in.config().findInt("vocab_size"), sp.pieceCount(), -1, in.embeddingRows(),
                model.toString());
        HfVocabBase.Tokens tokens = HfVocabBase.t5(sp, size, files.readAddedTokens());
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "t5");
        md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        tokens.writeTo(md);
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, sp.normalizerSpec().addDummyPrefix());
        md.put(GgufKeys.TOKENIZER_REMOVE_EXTRA_WS, sp.normalizerSpec().removeExtraWhitespaces());
        byte[] charsmap = sp.normalizerSpec().precompiledCharsmap();
        if (charsmap != null && charsmap.length > 0) {
            List<Integer> bytes = new ArrayList<>(charsmap.length);
            for (byte b : charsmap) bytes.add(b & 0xFF);
            md.put(GgufKeys.TOKENIZER_PRECOMPILED_CHARSMAP, Collections.unmodifiableList(bytes));
        }
        HfSpecialVocab.load(files, tj, in.config().raw(), false, tokens.size()).writeTo(md);
        HfTokenOverlays.writeTokenAttrs(md, addedTokens(tj, tc), tokens.size());
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, HfTokenOverlays.cleanSpaces(tc,
                tj != null && tj.model().type() == HfTokenizerJson.ModelType.BPE));
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        return Collections.unmodifiableMap(md);
    }

    private static void writeSpmOverlays(Map<String, Object> md, HfTokenizerJson tj, JsonView tc, int n,
                                         boolean bpeBackend) {
        HfTokenOverlays.writeTokenAttrs(md, addedTokens(tj, tc), n);
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, HfTokenOverlays.cleanSpaces(tc, bpeBackend));
        String scheme = HfTokenOverlays.prependScheme(tj, tc);
        if (scheme != null) md.put(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, scheme);
    }

    // ------------------------------------------------------------------ tiktoken, RWKV, PLaMo, Mistral

    /**
     * {@code _set_vocab_qwen} (base.py:2240-2294): a tiktoken vocabulary with merges rebuilt from its ranks and the
     * special tokens of Qwen-1 ({@code qwen.tiktoken}) or of tokenizer_config {@code added_tokens_decoder}. The
     * fingerprint encoder splits words as the tokenizer's code does: the {@code qwen2} pattern for Qwen-1, the
     * {@code stablelm2} pattern otherwise (StableLM 2's Arcade100k), which is also the name used when no fingerprint
     * matches.
     */
    public static Map<String, Object> qwen(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path file = files.tiktokenFile();
        if (file == null) throw new GgufFormatException("the checkpoint has no tiktoken vocabulary");
        TiktokenReader ranks = TiktokenReader.read(file);
        boolean qwen1 = file.getFileName().toString().equals(QWEN_TIKTOKEN);
        Map<String, Integer> specials = qwen1 ? HfSpecialVocab.qwenSpecials()
                : HfSpecialVocab.tiktokenSpecials(files.readTokenizerConfig(), file);
        int maxId = ranks.maxRank();
        for (int id : specials.values()) maxId = Math.max(maxId, id);
        OptionalInt configured = in.config().findInt("vocab_size");
        if (!configured.isPresent()) throw new GgufFormatException("config.json has no vocab_size");
        int size = HfVocabBase.vocabSize(configured, 0, maxId, in.embeddingRows(), file.toString());
        HfVocabBase.Tokens tokens = HfVocabBase.tiktoken(ranks, specials, size);
        HfSpecialVocab special = HfSpecialVocab.load(files, files.readTokenizerJson(), in.config().raw(), false,
                tokens.size()).withMerges(HfVocabBase.tiktokenMerges(ranks));
        Integer endOfText = specials.get(END_OF_TEXT);
        if (endOfText == null) throw new GgufFormatException(file + ": the special tokens have no " + END_OF_TEXT);
        if (special.ids().isEmpty()) {
            special = special.withSpecialToken(HfSpecialVocab.BOS, endOfText)
                    .withSpecialToken(HfSpecialVocab.EOS, endOfText);
        }
        special = special.withSpecialToken(HfSpecialVocab.UNK, endOfText);
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "gpt2");
        tokens.writeTo(md);
        special.writeTo(md);
        md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, true);
        md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, false);
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, false);
        String patternName = qwen1 ? "qwen2" : "stablelm2";
        Map<String, Object> candidate = new LinkedHashMap<>(md);
        candidate.put(GgufKeys.TOKENIZER_PRE, patternName);
        String name = pre.resolve(null, HfTokenOverlays.encoder(candidate, HfTokenOverlays.PostIds.NONE));
        md.put(GgufKeys.TOKENIZER_PRE, name != null ? name : patternName);
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        return Collections.unmodifiableMap(md);
    }

    /** The tiktoken merges {@code _set_vocab_qwen} rebuilds (base.py:2253-2260), for writers that reuse them. */
    public static List<String> tiktokenMerges(TiktokenReader ranks) {
        return Collections.unmodifiableList(HfVocabBase.tiktokenMerges(ranks));
    }

    /**
     * {@code _set_vocab_rwkv_world} (base.py:2420-2462): {@code <s>} then the world vocabulary as {@code repr} texts,
     * padded to {@code vocab_size} (65536 by default); eot 261, bos and eos 0; the chat template falls back to the
     * bundled {@code llama-cpp-rwkv-world.jinja}.
     */
    public static Map<String, Object> rwkvWorld(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path file = files.rwkvVocabFile();
        if (file == null) throw new GgufFormatException("the checkpoint has no RWKV world vocabulary");
        RwkvVocabReader reader = RwkvVocabReader.read(file);
        int count = reader.size() + 1;
        int size = HfVocabBase.vocabSize(OptionalInt.of(in.config().findInt(RWKV_DEFAULT_VOCAB, "vocab_size")), 0,
                count - 1, in.embeddingRows(), file.toString());
        HfVocabBase.Tokens tokens = HfVocabBase.rwkvWorld(reader, size);
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "rwkv");
        tokens.writeTo(md);
        HfSpecialVocab.load(files, files.readTokenizerJson(), in.config().raw(), false, size)
                .withSpecialToken(HfSpecialVocab.EOT, RWKV_EOT)
                .withSpecialTokenReplaced(HfSpecialVocab.BOS, 0)
                .withSpecialTokenReplaced(HfSpecialVocab.EOS, 0)
                .writeTo(md);
        String fallback = in.fallbackChatTemplate() != null ? in.fallbackChatTemplate()
                : HfChatTemplates.bundled(RWKV_TEMPLATE);
        md.putAll(HfChatTemplates.resolve(files, fallback));
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code _set_vocab_plamo} (base.py:2720-2815): PLaMo's {@code tokenizer.jsonl} with {@code normalTokens} typed
     * NORMAL and {@code <|plamo:...|>} tokens CONTROL, padded to {@code vocab_size}; model {@code plamo2} or
     * {@code plamo3} by tokenizer_config {@code tokenizer_class}; special ids by token text; eot {@code eotToken}.
     */
    public static Map<String, Object> plamo(Input in, String eotToken, Collection<String> normalTokens)
            throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path jsonl = files.plamoJsonl();
        if (jsonl == null) throw new GgufFormatException("the checkpoint has no PLaMo tokenizer.jsonl");
        if (files.tokenizerConfigJson() == null) {
            throw new GgufFormatException("the PLaMo checkpoint has no " + HfTokenizerFiles.TOKENIZER_CONFIG_JSON);
        }
        JsonView tc = files.readTokenizerConfig();
        String tokenizerClass = tc.optStr("tokenizer_class", null);
        String model = "Plamo2Tokenizer".equals(tokenizerClass) ? "plamo2"
                : "Plamo3Tokenizer".equals(tokenizerClass) ? "plamo3" : null;
        if (model == null) throw new GgufFormatException("unsupported PLaMo tokenizer class: " + tokenizerClass);
        PlamoVocabReader reader = PlamoVocabReader.read(jsonl);
        Set<String> normals = new HashSet<>(normalTokens);
        OptionalInt configured = in.config().findInt("vocab_size");
        if (!configured.isPresent()) throw new GgufFormatException("config.json has no vocab_size");
        int size = HfVocabBase.vocabSize(OptionalInt.of(Math.max(configured.getAsInt(), reader.size())), 0,
                reader.size() - 1, in.embeddingRows(), jsonl.toString());
        HfVocabBase.Tokens tokens = HfVocabBase.plamo(reader, size, normals);
        Map<String, Integer> ids = tokens.ids();
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, model);
        md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        tokens.writeTo(md);
        String[][] idKeys = {{"bos_token", GgufKeys.TOKENIZER_BOS_ID}, {"eos_token", GgufKeys.TOKENIZER_EOS_ID},
                {"pad_token", GgufKeys.TOKENIZER_PAD_ID}, {"sep_token", GgufKeys.TOKENIZER_SEP_ID},
                {"unk_token", GgufKeys.TOKENIZER_UNK_ID}};
        for (String[] k : idKeys) {
            if (tc.raw(k[0]) != null) md.put(k[1], plamoId(ids, HfSpecialVocab.tokenContent(tc.raw(k[0])), k[0]));
        }
        md.put(GgufKeys.TOKENIZER_EOT_ID, plamoId(ids, eotToken, "eot token"));
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        if (tc.raw("add_bos_token") instanceof Boolean) md.put(GgufKeys.TOKENIZER_ADD_BOS, tc.raw("add_bos_token"));
        if (tc.raw("add_eos_token") instanceof Boolean) md.put(GgufKeys.TOKENIZER_ADD_EOS, tc.raw("add_eos_token"));
        md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
        return Collections.unmodifiableMap(md);
    }

    private static int plamoId(Map<String, Integer> ids, String token, String what) {
        Integer id = token == null ? null : ids.get(token);
        if (id == null) {
            throw new GgufFormatException("the PLaMo " + what + " '" + token + "' is not in the vocabulary");
        }
        return id;
    }

    /**
     * {@code _set_vocab_mistral} (base.py:2635-2718): the mistral_common tokenizer ({@code tekken.json}, else the
     * versioned SentencePiece model) with its bos/eos/unk/pad ids, add_bos true and add_eos false, and the chat
     * template of a Mistral-format checkpoint's {@code chat_template.jinja}, else the community template of
     * mistral.py:54-90 ({@link HfChatTemplates#mistralCommunity}).
     */
    public static Map<String, Object> mistral(Input in, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer()) return fromGguf(files);
        Path file = files.mistralTokenizerFile();
        if (file == null) throw new GgufFormatException("the checkpoint has no mistral_common tokenizer file");
        Map<String, Object> md = new LinkedHashMap<>();
        HfVocabBase.Tokens tokens;
        boolean tekken = file.equals(files.tekkenJson());
        int version;
        int[] ids;
        if (tekken) {
            TekkenReader t = TekkenReader.read(file);
            tokens = HfVocabBase.tekken(t);
            version = t.versionNumber();
            int eos = requireSpecial(t, "</s>");
            int pad = t.specialId("<pad>");
            ids = new int[]{requireSpecial(t, "<s>"), eos, requireSpecial(t, "<unk>"), pad < 0 ? eos : pad};
            md.put(GgufKeys.TOKENIZER_MODEL, "gpt2");
            md.put(GgufKeys.TOKENIZER_PRE, "tekken");
            md.put(GgufKeys.TOKENIZER_MERGES, Collections.unmodifiableList(HfVocabBase.tekkenMerges(t)));
        } else {
            SentencePieceModel sp = SentencePieceModel.read(file);
            tokens = HfVocabBase.mistralSentencePiece(sp);
            version = HfVocabBase.mistralSpmVersion(file);
            SentencePieceModel.TrainerSpec ts = sp.trainerSpec();
            ids = new int[]{ts.bosId(), ts.eosId(), ts.unkId(), ts.padId() == -1 ? ts.eosId() : ts.padId()};
            md.put(GgufKeys.TOKENIZER_MODEL, "llama");
        }
        if (in.embeddingRows().isPresent() && tokens.size() > in.embeddingRows().getAsInt()) {
            throw new GgufFormatException(file + ": " + tokens.size() + " tokens exceed the "
                    + in.embeddingRows().getAsInt() + "-row token embedding");
        }
        String[] idKeys = {GgufKeys.TOKENIZER_BOS_ID, GgufKeys.TOKENIZER_EOS_ID, GgufKeys.TOKENIZER_UNK_ID,
                GgufKeys.TOKENIZER_PAD_ID};
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] < 0 || ids[i] >= tokens.size()) {
                throw new GgufFormatException(file + ": " + idKeys[i] + " " + ids[i] + " is outside the vocabulary");
            }
            md.put(idKeys[i], ids[i]);
        }
        tokens.writeTo(md);
        if (in.ggufArch() != null) md.put(String.format(GgufKeys.VOCAB_SIZE, in.ggufArch()), tokens.size());
        md.put(GgufKeys.TOKENIZER_ADD_BOS, true);
        md.put(GgufKeys.TOKENIZER_ADD_EOS, false);
        String local = in.config().mistralNative() ? files.readChatTemplate() : null;
        md.put(GgufKeys.TOKENIZER_CHAT_TEMPLATE,
                local != null ? local : HfChatTemplates.mistralCommunity(version, tekken));
        return Collections.unmodifiableMap(md);
    }

    private static int requireSpecial(TekkenReader t, String token) {
        int id = t.specialId(token);
        if (id < 0) throw new GgufFormatException("the tekken tokenizer has no special token " + token);
        return id;
    }

    // ------------------------------------------------------------------ shared helpers

    private static void rejectWordLevel(HfTokenizerJson tj) {
        if (tj != null && tj.model().type() == HfTokenizerJson.ModelType.WORD_LEVEL) {
            throw new GgufFormatException(WORD_LEVEL_MESSAGE);
        }
    }

    /** The tokenizer.json added tokens, else tokenizer_config {@code added_tokens_decoder}. */
    private static List<HfTokenizerJson.AddedToken> addedTokens(HfTokenizerJson tj, JsonView tc) {
        return tj != null ? tj.addedTokens() : HfSpecialVocab.configAddedTokens(tc);
    }
}
