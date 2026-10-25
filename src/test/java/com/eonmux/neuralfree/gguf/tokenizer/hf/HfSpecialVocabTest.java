package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfSpecialVocab}, the port of llama.cpp's {@code gguf.SpecialVocab}: the post-processor add flags and the
 * EOS demotion, the {@code add_sep} parse, tokenizer_config and config.json ids, the vocabulary bound, the merges
 * sources, the normalizer flags and the written keys.
 */
public class HfSpecialVocabTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String ADDED = "'added_tokens':["
            + "{'id':0,'content':'<s>','special':true},{'id':1,'content':'</s>','special':true},"
            + "{'id':2,'content':'<|eot|>','special':true},{'id':3,'content':'<|eom|>','special':true}]";
    private static final String MODEL = "'model':{'type':'BPE','vocab':{'<s>':0,'</s>':1,'<|eot|>':2,'<|eom|>':3,"
            + "'a':4,'b':5,'ab':6},'merges':['a b']}";

    private static String json(String singleQuoted) {
        return singleQuoted.replace('\'', '"');
    }

    private static String special(String id) {
        return "{'SpecialToken':{'id':'" + id + "','type_id':0}}";
    }

    private static String sequence(String id) {
        return "{'Sequence':{'id':'" + id + "','type_id':0}}";
    }

    private static String template(String single, String pair) {
        return "'post_processor':{'type':'TemplateProcessing','single':[" + single + "]"
                + (pair == null ? "" : ",'pair':[" + pair + "]") + ",'special_tokens':{}}";
    }

    private static HfTokenizerJson tokenizer(String postProcessor, String normalizer) {
        String doc = "{" + ADDED + "," + MODEL + (postProcessor == null ? "" : "," + postProcessor)
                + (normalizer == null ? "" : ",'normalizer':" + normalizer) + "}";
        return HfTokenizerJson.of(Json.parse(json(doc)), "tokenizer.json");
    }

    private HfTokenizerFiles files(String tokenizerConfig) throws IOException {
        Path dir = tmp.newFolder().toPath();
        if (tokenizerConfig != null) {
            Files.write(dir.resolve("tokenizer_config.json"), json(tokenizerConfig).getBytes(StandardCharsets.UTF_8));
        }
        return HfTokenizerFiles.locate(dir);
    }

    private static JsonView config(String json) {
        return JsonView.of(Json.parse(json(json)), "config.json");
    }

    private static Map<String, Object> written(HfSpecialVocab v) {
        Map<String, Object> out = new LinkedHashMap<>();
        v.writeTo(out);
        return out;
    }

    // ------------------------------------------------------------------ post-processor

    @Test
    public void templateProcessingDecidesAddBosAndAddEos() throws IOException {
        HfTokenizerJson tj = tokenizer(template(special("<s>") + "," + sequence("A") + "," + special("</s>"), null),
                null);
        HfSpecialVocab v = HfSpecialVocab.load(files("{'bos_token':'<s>','eos_token':'</s>','add_eos_token':false}"),
                tj, config("{}"), false, 7);
        Map<String, Object> md = written(v);
        assertEquals(0, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals("the post-processor wins over add_eos_token", true, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertFalse("merges were not requested", md.containsKey(GgufKeys.TOKENIZER_MERGES));
    }

    @Test
    public void aTrailingTokenOtherThanEosReplacesItAndDemotesItToEot() throws IOException {
        HfTokenizerJson tj = tokenizer(template(sequence("A") + "," + special("<|eot|>"), null), null);
        HfSpecialVocab v = HfSpecialVocab.load(files("{'eos_token':'</s>'}"), tj, config("{}"), false, 7);
        Map<String, Object> md = written(v);
        assertEquals(2, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals("the replaced eos becomes eot", 1, md.get(GgufKeys.TOKENIZER_EOT_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertNull("a single piece decides no add_bos", md.get(GgufKeys.TOKENIZER_ADD_BOS));
    }

    @Test
    public void aSeparatorBetweenPairSequencesSetsAddSep() throws IOException {
        HfTokenizerJson tj = tokenizer(template(special("<s>") + "," + sequence("A"),
                special("<s>") + "," + sequence("A") + "," + special("</s>") + "," + sequence("B")), null);
        HfSpecialVocab v = HfSpecialVocab.load(files("{'bos_token':'<s>','eos_token':'</s>'}"), tj, config("{}"),
                false, 7);
        Map<String, Object> md = written(v);
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_SEP));
        assertEquals("sep falls back to the eos text", 1, md.get(GgufKeys.TOKENIZER_SEP_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
    }

    @Test
    public void byteLevelAndRobertaProcessorsSetTheAddFlags() throws IOException {
        HfSpecialVocab byteLevel = HfSpecialVocab.load(files("{}"),
                tokenizer("'post_processor':{'type':'ByteLevel'}", null), config("{}"), false, 7);
        assertEquals(false, byteLevel.addFlags().get("bos"));
        assertEquals(false, byteLevel.addFlags().get("eos"));

        HfSpecialVocab roberta = HfSpecialVocab.load(files("{'bos_token':'<s>','eos_token':'</s>'}"),
                tokenizer("'post_processor':{'type':'RobertaProcessing','sep':['</s>',1],'cls':['<s>',0]}", null),
                config("{}"), false, 7);
        Map<String, Object> md = written(roberta);
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_EOS));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_SEP));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_SEP_ID));
        assertEquals("cls has no GGUF key", Integer.valueOf(0), roberta.id("cls"));
        assertFalse(md.keySet().stream().anyMatch(k -> k.contains("cls")));
    }

    // ------------------------------------------------------------------ ids

    @Test
    public void configIdsFillUnsetSlotsIncludingTextConfig() throws IOException {
        HfSpecialVocab v = HfSpecialVocab.load(files(null), null,
                config("{'pad_token_id':3,'text_config':{'bos_token_id':0,'eos_token_id':1}}"), false, 7);
        assertEquals(Integer.valueOf(0), v.id("bos"));
        assertEquals(Integer.valueOf(1), v.id("eos"));
        assertEquals(Integer.valueOf(3), v.id("pad"));
        assertTrue("no add flag without a source", v.addFlags().isEmpty());
    }

    @Test
    public void idsOutsideTheVocabularyAreSkippedAndNegativeIdsRejected() throws IOException {
        HfSpecialVocab v = HfSpecialVocab.load(files(null), null, config("{'bos_token_id':7,'eos_token_id':6}"),
                false, 7);
        assertNull(v.id("bos"));
        assertEquals(Integer.valueOf(6), v.id("eos"));
        assertThrows(GgufFormatException.class,
                () -> HfSpecialVocab.load(files(null), null, config("{'bos_token_id':-1}"), false, 7));
    }

    @Test
    public void withSpecialTokenSetsOnlyUnsetInRangeIds() throws IOException {
        HfSpecialVocab v = HfSpecialVocab.load(files(null), null, config("{'eos_token_id':1}"), false, 7);
        assertSame(v, v.withSpecialToken("eos", 2));
        assertSame(v, v.withSpecialToken("bos", null));
        assertEquals(Integer.valueOf(2), v.withSpecialToken("eot", 2).id("eot"));
        assertNull("out of range is skipped", v.withSpecialToken("eot", 7).id("eot"));
        assertThrows(GgufFormatException.class, () -> v.withSpecialToken("eot", -2));
        assertEquals(Integer.valueOf(0), v.withSpecialTokenReplaced("eos", 0).id("eos"));
        assertThrows(GgufFormatException.class, () -> v.withSpecialTokenReplaced("eos", 7));
        assertEquals("the original is unchanged", Integer.valueOf(1), v.id("eos"));
    }

    // ------------------------------------------------------------------ merges and normalizer

    @Test
    public void mergesComeFromTokenizerJsonElseMergesTxt() throws IOException {
        HfSpecialVocab fromJson = HfSpecialVocab.load(files(null), tokenizer(null, null), config("{}"), true, 7);
        assertEquals(Collections.singletonList("a b"), fromJson.merges());

        Path dir = tmp.newFolder().toPath();
        Files.write(dir.resolve("merges.txt"), "#version: 0.2\na b\nab c\n".getBytes(StandardCharsets.UTF_8));
        HfSpecialVocab fromTxt = HfSpecialVocab.load(HfTokenizerFiles.locate(dir), null, config("{}"), true, 7);
        assertEquals(Arrays.asList("a b", "ab c"), fromTxt.merges());
        assertEquals(fromTxt.merges(), written(fromTxt).get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(Collections.singletonList("x y"),
                fromTxt.withMerges(Collections.singletonList("x y")).merges());
    }

    @Test
    public void normalizerFlagsFollowBertAndSequenceNormalizers() throws IOException {
        Map<String, Object> bert = written(HfSpecialVocab.load(files(null),
                tokenizer(null, "{'type':'BertNormalizer','lowercase':true,'strip_accents':null}"), config("{}"),
                false, 7));
        assertEquals(true, bert.get(GgufKeys.TOKENIZER_NORMALIZER_LOWERCASE));
        assertFalse(bert.containsKey(GgufKeys.TOKENIZER_NORMALIZER_STRIP_ACCENTS));

        Map<String, Object> sequence = written(HfSpecialVocab.load(files(null),
                tokenizer(null, "{'type':'Sequence','normalizers':[{'type':'NFC'},{'type':'StripAccents'}]}"),
                config("{}"), false, 7));
        assertEquals(true, sequence.get(GgufKeys.TOKENIZER_NORMALIZER_NFC));
        assertEquals(true, sequence.get(GgufKeys.TOKENIZER_NORMALIZER_STRIP_ACCENTS));
        assertFalse(sequence.containsKey(GgufKeys.TOKENIZER_NORMALIZER_LOWERCASE));
    }

    @Test
    public void declaredSpecialTokensReadBothFilesInOrder() {
        JsonView tc = config("{'bos_token':{'content':'<s>'},'additional_special_tokens':['<a>',{'content':'<b>'}]}");
        JsonView map = config("{'eos_token':'</s>','extra_special_tokens':{'x':'<c>'}}");
        assertEquals(Arrays.asList("<s>", "<a>", "<b>", "</s>", "<c>"),
                Arrays.asList(HfSpecialVocab.declaredSpecialTokens(tc, map).toArray()));
    }
}
