package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.Normalizer;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfVocabWriters} on synthetic checkpoints: the token lists, types, scores, merges and padding the converter
 * writes, the special ids and add flags, the {@code neuralfree.tokenizer.*} overlays, the added-token round trip, the
 * post-processor ids of the fingerprint encoder, the WordLevel rejection and the Mistral community template.
 */
public class HfVocabWritersTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The exact pre-tokenizer of the fixtures: ByteLevel, or no step when tokenizer.json has none. */
    private static final Function<JsonView, PreTokenizerSpec> SPECS = v -> v == null
            ? new PreTokenizerSpec(Collections.emptyList(), false, false, false, false)
            : new PreTokenizerSpec(Collections.singletonList(PreTokenizerSpec.Step.byteLevel()), true,
                    v.optBool("add_prefix_space", true), v.optBool("trim_offsets", true), v.optBool("use_regex", true));

    /** A GPT-2 style tokenizer.json: byte-level BPE, a leading {@code <|begin|>} post-processor token. */
    private static final String GPT2_TOKENIZER = "{'added_tokens':["
            + "{'id':0,'content':'<s>','special':true,'normalized':false},"
            + "{'id':1,'content':'</s>','special':true,'normalized':false},"
            + "{'id':5,'content':'<|begin|>','special':true,'normalized':false},"
            + "{'id':6,'content':'Ġhi','special':false,'normalized':false,'lstrip':true}],"
            + "'normalizer':null,"
            + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,'use_regex':true},"
            + "'post_processor':{'type':'TemplateProcessing',"
            + "'single':[{'SpecialToken':{'id':'<|begin|>','type_id':0}},{'Sequence':{'id':'A','type_id':0}}],"
            + "'pair':[{'SpecialToken':{'id':'<|begin|>','type_id':0}},{'Sequence':{'id':'A','type_id':0}},"
            + "{'Sequence':{'id':'B','type_id':1}}],"
            + "'special_tokens':{'<|begin|>':{'id':'<|begin|>','ids':[5],'tokens':['<|begin|>']}}},"
            + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,'use_regex':true},"
            + "'model':{'type':'BPE','unk_token':null,'byte_fallback':false,'ignore_merges':false,"
            + "'vocab':{'<s>':0,'</s>':1,'a':2,'b':3,'ab':4},'merges':['a b']}}";

    private static final String GPT2_CONFIG = "{'bos_token':'<s>','eos_token':'</s>','add_bos_token':false,"
            + "'clean_up_tokenization_spaces':true,'chat_template':'T'}";

    /** Records what the writer asks to fingerprint and answers {@code name}. */
    private static final class CapturingResolver implements PreNameResolver {
        private final String name;
        private HfTokenizerJson tokenizerJson;
        private Function<String, int[]> encoder;

        CapturingResolver(String name) {
            this.name = name;
        }

        @Override
        public String resolve(HfTokenizerJson tj, Function<String, int[]> enc) {
            this.tokenizerJson = tj;
            this.encoder = enc;
            return name;
        }
    }

    private static String json(String singleQuoted) {
        return singleQuoted.replace('\'', '"');
    }

    private static void write(Path dir, String name, String content) throws IOException {
        Path file = dir.resolve(name);
        Files.createDirectories(file.getParent());
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
    }

    private Path dir(String... namesAndJson) throws IOException {
        Path dir = tmp.newFolder().toPath();
        for (int i = 0; i < namesAndJson.length; i += 2) write(dir, namesAndJson[i], json(namesAndJson[i + 1]));
        return dir;
    }

    /**
     * The writer input of {@code dir} with the config.json object {@code config} (single-quoted JSON), to which the
     * {@code architectures} member every Hugging Face configuration needs is added.
     */
    private static HfVocabWriters.Input input(Path dir, String config) throws IOException {
        String members = config.trim();
        members = members.substring(1, members.length() - 1).trim();
        String full = "{'architectures':['LlamaForCausalLM']" + (members.isEmpty() ? "" : "," + members) + "}";
        return HfVocabWriters.Input.of(HfConfigLoader.fromJson(dir, json(full)), HfTokenizerFiles.locate(dir), SPECS);
    }

    private static <T> List<T> list(Object value) {
        @SuppressWarnings("unchecked")
        List<T> out = (List<T>) value;
        return out;
    }

    // ------------------------------------------------------------------ GPT-2 BPE

    @Test
    public void gpt2WritesTokensSpecialIdsAndAddFlags() throws IOException {
        Path dir = dir("tokenizer.json", GPT2_TOKENIZER, "tokenizer_config.json", GPT2_CONFIG);
        CapturingResolver pre = new CapturingResolver("gpt-2");
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':8}"), pre);

        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("gpt-2", md.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(Arrays.asList("<s>", "</s>", "a", "b", "ab", "<|begin|>", " hi", "[PAD7]"),
                md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(3, 3, 1, 1, 1, 3, 4, 5), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertNull("GPT-2 vocabularies have no scores", md.get(GgufKeys.TOKENIZER_SCORES));
        assertEquals(Collections.singletonList("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals("the post-processor wins over add_bos_token", true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertNull(md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertEquals("T", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertThrows(UnsupportedOperationException.class, () -> md.put("x", 1));
    }

    @Test
    public void gpt2WritesTheBpeOverlays() throws IOException {
        Path dir = dir("tokenizer.json", GPT2_TOKENIZER, "tokenizer_config.json", GPT2_CONFIG);
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':8}"), new CapturingResolver("gpt-2"));

        assertEquals(false, md.get(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES));
        assertEquals(false, md.get(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK));
        assertFalse("no unk_token, no fuse_unk", md.containsKey(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE));
        assertEquals("transformers does not clean up BPE output", false, md.get(GgufKeys.NF_TOKENIZER_CLEAN_SPACES));
        assertFalse("the byte-level decoder is implied", md.containsKey(GgufKeys.NF_TOKENIZER_DECODER));
        assertFalse(md.containsKey(GgufKeys.NF_TOKENIZER_NORMALIZER));
        assertFalse("a resolved name needs no spec", md.containsKey(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER));
        assertFalse(md.containsKey(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE));
        assertEquals(Arrays.asList(0, 0, 0, 0, 0, 0, TokenizerVocab.ATTR_LSTRIP, 0),
                md.get(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS));
    }

    @Test
    public void addedTokensAreRoundTrippedThroughTheDecoder() throws IOException {
        Path dir = dir("tokenizer.json", GPT2_TOKENIZER, "tokenizer_config.json", GPT2_CONFIG);
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':8}"), new CapturingResolver(null));
        assertEquals("'Ġhi' decodes to ' hi'", " hi", list(md.get(GgufKeys.TOKENIZER_LIST)).get(6));
    }

    @Test
    public void postProcessorIdsReachTheEncoderAndTheOverlay() throws IOException {
        Path dir = dir("tokenizer.json", GPT2_TOKENIZER, "tokenizer_config.json", GPT2_CONFIG);
        CapturingResolver pre = new CapturingResolver("gpt-2");
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':8}"), pre);

        assertNotNull(pre.tokenizerJson);
        assertArrayEquals("<|begin|> then the merged 'ab'", new int[]{5, 4}, pre.encoder.apply("ab"));
        assertArrayEquals(new int[]{5, 5, 4}, pre.encoder.apply("<|begin|>ab"));
        assertEquals("add_bos would add <s>, the post-processor adds <|begin|>", Collections.singletonList(5),
                md.get(GgufKeys.NF_TOKENIZER_PREFIX_IDS));
        assertEquals(Collections.emptyList(), md.get(GgufKeys.NF_TOKENIZER_SUFFIX_IDS));
    }

    @Test
    public void anUnresolvedPreTokenizerStoresTheExactSpec() throws IOException {
        String tokenizer = GPT2_TOKENIZER.replace("'add_prefix_space':false", "'add_prefix_space':true");
        Path dir = dir("tokenizer.json", tokenizer, "tokenizer_config.json", GPT2_CONFIG);
        HfVocabWriters.Input in = input(dir, "{'vocab_size':8}");

        Map<String, Object> unresolved = HfVocabWriters.gpt2(in, new CapturingResolver(null));
        assertEquals("default", unresolved.get(GgufKeys.TOKENIZER_PRE));
        PreTokenizerSpec expected = SPECS.apply(HfTokenizerJson.of(Json.parse(json(tokenizer)), "t").preTokenizer());
        assertEquals(expected.toCanonicalJson(), unresolved.get(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER));
        assertFalse(unresolved.containsKey(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE));

        Map<String, Object> named = HfVocabWriters.gpt2(in, new CapturingResolver("gpt-2"));
        assertEquals("a named pre-tokenizer keeps add_prefix_space", true,
                named.get(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE));
        assertEquals("a pinned name ignores the resolver", "whitespace",
                HfVocabWriters.whitespace(in, new CapturingResolver("gpt-2")).get(GgufKeys.TOKENIZER_PRE));
    }

    @Test
    public void aSentencePieceStyleBpeWritesItsNormalizerDecoderAndModelOptions() throws IOException {
        String normalizer = "{'type':'Sequence','normalizers':[{'type':'Prepend','prepend':'▁'},"
                + "{'type':'Replace','pattern':{'String':' '},'content':'▁'}]}";
        String decoder = "{'type':'Sequence','decoders':[{'type':'Replace','pattern':{'String':'▁'},'content':' '},"
                + "{'type':'ByteFallback'},{'type':'Fuse'},{'type':'Strip','content':' ','start':1,'stop':0}]}";
        String tokenizer = "{'added_tokens':[],'normalizer':" + normalizer + ",'pre_tokenizer':null,"
                + "'post_processor':null,'decoder':" + decoder + ","
                + "'model':{'type':'BPE','unk_token':'<unk>','fuse_unk':true,'byte_fallback':true,"
                + "'ignore_merges':true,'vocab':{'<unk>':0,'<s>':1,'</s>':2,'▁':3,'a':4,'▁a':5},'merges':['▁ a']}}";
        Path dir = dir("tokenizer.json", tokenizer, "tokenizer_config.json",
                "{'clean_up_tokenization_spaces':true,"
                        + "'clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output':true}");
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':6}"), new CapturingResolver(null));

        assertEquals(true, md.get(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES));
        assertEquals(true, md.get(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK));
        assertEquals(true, md.get(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE));
        assertEquals("the unk id comes from the model's unk_token", 0, md.get(GgufKeys.TOKENIZER_UNK_ID));
        assertEquals("the opt-in flag keeps the clean-up", true, md.get(GgufKeys.NF_TOKENIZER_CLEAN_SPACES));
        assertEquals(Normalizer.fromJson(Json.parse(json(normalizer)), "n"),
                Normalizer.fromCanonicalJson((String) md.get(GgufKeys.NF_TOKENIZER_NORMALIZER)));
        assertEquals(TokenDecoder.fromJson(Json.parse(json(decoder)), "d"),
                TokenDecoder.fromCanonicalJson((String) md.get(GgufKeys.NF_TOKENIZER_DECODER)));
        assertEquals("default", md.get(GgufKeys.TOKENIZER_PRE));
        assertFalse("no post-processor, no overlay", md.containsKey(GgufKeys.NF_TOKENIZER_PREFIX_IDS));
    }

    @Test
    public void aSlowGpt2TokenizerReadsVocabJsonAndMergesTxt() throws IOException {
        Path dir = dir("vocab.json", "{'a':0,'b':1,'ab':2,'<|endoftext|>':3}",
                "tokenizer_config.json", "{'eos_token':'<|endoftext|>'}");
        write(dir, "merges.txt", "#version: 0.2\na b\n");
        Map<String, Object> md = HfVocabWriters.gpt2(input(dir, "{'vocab_size':4,'eos_token_id':3}"),
                new CapturingResolver(null));

        assertEquals(Arrays.asList("a", "b", "ab", "<|endoftext|>"), md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals("a declared special token is CONTROL", Arrays.asList(1, 1, 1, 3),
                md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals(Collections.singletonList("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(3, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(false, md.get(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES));
        assertEquals("default", md.get(GgufKeys.TOKENIZER_PRE));
        assertNotNull(md.get(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER));
    }

    @Test
    public void wordLevelTokenizersAreRejected() throws IOException {
        Path dir = dir("tokenizer.json",
                "{'model':{'type':'WordLevel','vocab':{'a':0,'[UNK]':1},'unk_token':'[UNK]'}}");
        HfVocabWriters.Input in = input(dir, "{'vocab_size':2}");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfVocabWriters.gpt2(in, new CapturingResolver(null)));
        assertTrue(e.getMessage(), e.getMessage().contains("WordLevel"));
        assertThrows(GgufFormatException.class, () -> HfVocabWriters.bertWpm(in, new CapturingResolver(null)));
    }

    // ------------------------------------------------------------------ WordPiece

    @Test
    public void bertWritesPhantomSpaceTokensAndWrapsTheEncoderInClsSep() throws IOException {
        String tokenizer = "{'added_tokens':["
                + "{'id':0,'content':'[PAD]','special':true,'normalized':false},"
                + "{'id':1,'content':'[UNK]','special':true,'normalized':false},"
                + "{'id':2,'content':'[CLS]','special':true,'normalized':false},"
                + "{'id':3,'content':'[SEP]','special':true,'normalized':false}],"
                + "'normalizer':{'type':'BertNormalizer','clean_text':true,'handle_chinese_chars':true,"
                + "'strip_accents':null,'lowercase':true},"
                + "'pre_tokenizer':{'type':'BertPreTokenizer'},"
                + "'post_processor':{'type':'TemplateProcessing',"
                + "'single':[{'SpecialToken':{'id':'[CLS]','type_id':0}},{'Sequence':{'id':'A','type_id':0}},"
                + "{'SpecialToken':{'id':'[SEP]','type_id':0}}],"
                + "'special_tokens':{'[CLS]':{'id':'[CLS]','ids':[2],'tokens':['[CLS]']},"
                + "'[SEP]':{'id':'[SEP]','ids':[3],'tokens':['[SEP]']}}},"
                + "'decoder':{'type':'WordPiece','prefix':'##','cleanup':true},"
                + "'model':{'type':'WordPiece','unk_token':'[UNK]','continuing_subword_prefix':'##',"
                + "'max_input_chars_per_word':100,"
                + "'vocab':{'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,'hello':4,'##lo':5}}}";
        Path dir = dir("tokenizer.json", tokenizer, "tokenizer_config.json",
                "{'cls_token':'[CLS]','sep_token':'[SEP]','pad_token':'[PAD]','unk_token':'[UNK]',"
                        + "'clean_up_tokenization_spaces':true}");
        CapturingResolver pre = new CapturingResolver(null);
        Map<String, Object> md = HfVocabWriters.bertWpm(input(dir, "{'vocab_size':6,'type_vocab_size':2}"), pre);

        assertEquals("bert", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("default", md.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(2, md.get(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT));
        assertEquals(Arrays.asList("[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "lo"),
                md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(3, 3, 3, 3, 1, 1), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals("cls stands in for bos", 2, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals("sep stands in for eos", 3, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(3, md.get(GgufKeys.TOKENIZER_SEP_ID));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_PAD_ID));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_UNK_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_NORMALIZER_LOWERCASE));
        assertEquals("WordPiece keeps the clean-up", true, md.get(GgufKeys.NF_TOKENIZER_CLEAN_SPACES));
        assertArrayEquals(new int[]{2, 4, 3}, pre.encoder.apply("hello"));
    }

    // ------------------------------------------------------------------ SentencePiece

    /** A minimal protobuf encoder for SentencePiece model files. */
    private static final class Proto {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        Proto varint(int field, long value) {
            tag(field, 0);
            raw(value);
            return this;
        }

        Proto fixed32(int field, float value) {
            tag(field, 5);
            int bits = Float.floatToIntBits(value);
            for (int i = 0; i < 4; i++) out.write(bits >>> (8 * i));
            return this;
        }

        Proto string(int field, String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            tag(field, 2);
            raw(bytes.length);
            out.write(bytes, 0, bytes.length);
            return this;
        }

        Proto message(int field, Proto value) {
            byte[] bytes = value.out.toByteArray();
            tag(field, 2);
            raw(bytes.length);
            out.write(bytes, 0, bytes.length);
            return this;
        }

        byte[] toBytes() {
            return out.toByteArray();
        }

        private void tag(int field, int wire) {
            raw(((long) field << 3) | wire);
        }

        private void raw(long v) {
            while ((v & ~0x7FL) != 0) {
                out.write((int) ((v & 0x7F) | 0x80));
                v >>>= 7;
            }
            out.write((int) v);
        }
    }

    /** A SentencePiece model of {@code pieces} ({text, score, type} triples) with trainer {@code modelType}. */
    private static byte[] sentencePiece(int modelType, Object... pieces) {
        Proto model = new Proto();
        for (int i = 0; i < pieces.length; i += 3) {
            model.message(1, new Proto().string(1, (String) pieces[i]).fixed32(2, (Float) pieces[i + 1])
                    .varint(3, (Integer) pieces[i + 2]));
        }
        return model.message(2, new Proto().varint(3, modelType)).toBytes();
    }

    private static final Object[] SPM_PIECES = {"<unk>", 0f, 2, "<s>", 0f, 3, "</s>", 0f, 3, "▁a", -1.5f, 1};

    @Test
    public void sentencePieceWritesPiecesAddedTokensAndPadding() throws IOException {
        Path dir = dir("tokenizer_config.json", "{'add_bos_token':true,'add_eos_token':false,'legacy':false,"
                        + "'clean_up_tokenization_spaces':true,"
                        + "'added_tokens_decoder':{'5':{'content':'<|im_end|>','special':true}}}",
                "added_tokens.json", "{'<x>':4}");
        Files.write(dir.resolve("tokenizer.model"), sentencePiece(2, SPM_PIECES));
        HfVocabWriters.Input in = input(dir, "{'vocab_size':7,'bos_token_id':1,'eos_token_id':2}");
        assertTrue(HfVocabWriters.canSentencePiece(in));
        Map<String, Object> md = HfVocabWriters.sentencePiece(in, new CapturingResolver(null));

        assertEquals("llama", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("default", md.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(Arrays.asList("<unk>", "<s>", "</s>", "▁a", "<x>", "<|im_end|>", "[PAD6]"),
                md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(0f, 0f, 0f, -1.5f, -1000f, -1000f, -10000f), md.get(GgufKeys.TOKENIZER_SCORES));
        assertEquals(Arrays.asList(2, 3, 3, 1, 4, 3, 5), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(2, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(false, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertEquals("a BPE SentencePiece backend is not cleaned up", false,
                md.get(GgufKeys.NF_TOKENIZER_CLEAN_SPACES));
        assertEquals("legacy=false prepends only to the first piece", "first",
                md.get(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME));
        assertFalse("without tokenizer.json the GGUF add flags stand",
                md.containsKey(GgufKeys.NF_TOKENIZER_PREFIX_IDS));
        assertFalse(md.containsKey(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES));
    }

    @Test
    public void sentencePieceNeedsTokenizerModel() throws IOException {
        HfVocabWriters.Input in = input(dir("tokenizer_config.json", "{}"), "{'vocab_size':4}");
        assertFalse(HfVocabWriters.canSentencePiece(in));
        assertThrows(GgufFormatException.class, () -> HfVocabWriters.sentencePiece(in, new CapturingResolver(null)));
    }

    @Test
    public void t5WritesTheUnigramVocabularyAndNormalizerFlags() throws IOException {
        Path dir = dir("tokenizer_config.json", "{'eos_token':'</s>'}");
        Files.write(dir.resolve("spiece.model"),
                sentencePiece(1, "<pad>", 0f, 3, "</s>", 0f, 3, "<unk>", 0f, 2, "▁a", -2f, 1));
        Map<String, Object> md = HfVocabWriters.t5Ugm(input(dir, "{'vocab_size':6,'eos_token_id':1}"),
                new CapturingResolver(null));

        assertEquals("t5", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(Arrays.asList("<pad>", "</s>", "<unk>", "▁a", "[PAD4]", "[PAD5]"),
                md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(0f, 0f, 0f, -2f, -10000f, -10000f), md.get(GgufKeys.TOKENIZER_SCORES));
        assertEquals(Arrays.asList(3, 3, 2, 1, 5, 5), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_PREFIX));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_REMOVE_EXTRA_WS));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_EOS_ID));
    }

    // ------------------------------------------------------------------ tiktoken, RWKV, Mistral

    private static String b64(String s) {
        return Base64.getEncoder().encodeToString(s.getBytes(StandardCharsets.UTF_8));
    }

    @Test
    public void tiktokenRebuildsMergesAndDefaultsBosEosUnkToEndOfText() throws IOException {
        Path dir = dir("tokenizer_config.json",
                "{'added_tokens_decoder':{'3':{'content':'<|endoftext|>','special':true}}}");
        write(dir, "tokenizer.tiktoken", b64("a") + " 0\n" + b64("b") + " 1\n" + b64("ab") + " 2\n");
        CapturingResolver pre = new CapturingResolver(null);
        Map<String, Object> md = HfVocabWriters.qwen(input(dir, "{'vocab_size':5}"), pre);

        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("stablelm2", md.get(GgufKeys.TOKENIZER_PRE));
        assertNull("a tiktoken vocabulary has no tokenizer.json", pre.tokenizerJson);
        assertEquals(Arrays.asList("a", "b", "ab", "<|endoftext|>", "[PAD4]"), md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(1, 1, 1, 3, 5), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals(Collections.singletonList("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(3, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(3, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(3, md.get(GgufKeys.TOKENIZER_UNK_ID));
        assertEquals(true, md.get(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES));
        assertEquals(false, md.get(GgufKeys.NF_TOKENIZER_CLEAN_SPACES));
    }

    @Test
    public void rwkvWorldPadsTheVocabularyAndUsesTheBundledTemplate() throws IOException {
        Path dir = dir();
        write(dir, "rwkv_vocab_v20230424.txt", "1 'a' 1\n2 'bc' 2\n");
        Map<String, Object> md = HfVocabWriters.rwkvWorld(input(dir, "{'vocab_size':5}"), new CapturingResolver(null));

        assertEquals("rwkv", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(Arrays.asList("<s>", "a", "bc", "[PAD3]", "[PAD4]"), md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(Arrays.asList(3, 1, 1, 5, 5), md.get(GgufKeys.TOKENIZER_TOKEN_TYPE));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertFalse("eot 261 is outside a 5-token vocabulary", md.containsKey(GgufKeys.TOKENIZER_EOT_ID));
        assertEquals(HfChatTemplates.bundled("llama-cpp-rwkv-world.jinja"), md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
    }

    /** A tekken.json with {@code numSpecial} default special tokens, the 256 bytes and {@code extra} tokens. */
    private static String tekken(String version, int numSpecial, String... extra) {
        StringBuilder vocab = new StringBuilder();
        for (int i = 0; i < 256 + extra.length; i++) {
            byte[] bytes = i < 256 ? new byte[]{(byte) i} : extra[i - 256].getBytes(StandardCharsets.UTF_8);
            if (i > 0) vocab.append(',');
            vocab.append("{\"rank\":").append(i).append(",\"token_bytes\":\"")
                    .append(Base64.getEncoder().encodeToString(bytes)).append("\",\"token_str\":null}");
        }
        return "{\"config\":{\"pattern\":\"\\\\p{L}+|\\\\s+\",\"num_vocab_tokens\":" + (256 + extra.length)
                + ",\"default_vocab_size\":" + (numSpecial + 256 + extra.length)
                + ",\"default_num_special_tokens\":" + numSpecial + ",\"version\":\"" + version + "\"},"
                + "\"vocab\":[" + vocab + "]}";
    }

    @Test
    public void tekkenV7WritesTheMistralVocabularyAndTheV7TekkenTemplate() throws IOException {
        Path dir = dir();
        write(dir, "tekken.json", tekken("v7", 20, "ab"));
        TekkenReader reader = TekkenReader.read(dir.resolve("tekken.json"));
        HfVocabWriters.Input in = input(dir, "{'vocab_size':277}").withGgufArch("llama");
        Map<String, Object> md = HfVocabWriters.mistral(in, new CapturingResolver(null));

        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("tekken", md.get(GgufKeys.TOKENIZER_PRE));
        List<String> tokens = list(md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(277, tokens.size());
        assertEquals("<s>", tokens.get(1));
        assertEquals("[INST]", tokens.get(3));
        assertEquals("a", tokens.get(20 + 'a'));
        assertEquals("Ġ", tokens.get(20 + ' '));
        assertEquals("ab", tokens.get(276));
        assertEquals(Integer.valueOf(3), list(md.get(GgufKeys.TOKENIZER_TOKEN_TYPE)).get(19));
        assertEquals(Integer.valueOf(1), list(md.get(GgufKeys.TOKENIZER_TOKEN_TYPE)).get(20));
        assertEquals(Collections.singletonList("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(2, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_UNK_ID));
        int pad = reader.specialId("<pad>");
        assertEquals(pad < 0 ? 2 : pad, md.get(GgufKeys.TOKENIZER_PAD_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(false, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertEquals(277, md.get("llama.vocab_size"));
        assertEquals("mistral-v7-tekken", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));

        String template = ChatTemplateSource.resolve((String) md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE), null);
        ChatFormatter formatter = new ChatFormatter(template, "<s>", "</s>", true, false, "llama");
        assertEquals("[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]U1[/INST]A1</s>[INST]U2[/INST]",
                formatter.format(Arrays.asList(msg("system", "S"), msg("user", "U1"), msg("assistant", "A1"),
                        msg("user", "U2")), false));
    }

    @Test
    public void aVersionedSentencePieceFileSelectsItsCommunityTemplate() throws IOException {
        Path dir = dir();
        Files.write(dir.resolve("tokenizer.model.v3"), sentencePiece(2, SPM_PIECES));
        Map<String, Object> md = HfVocabWriters.mistral(input(dir, "{'vocab_size':4}"), new CapturingResolver(null));

        assertEquals("llama", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(Arrays.asList("<unk>", "<s>", "</s>", "▁a"), md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(2, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals("pad_id -1 falls back to eos", 2, md.get(GgufKeys.TOKENIZER_PAD_ID));
        assertEquals("mistral-v3", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
    }

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
