package com.eonmux.neuralfree.gguf.tokenizer;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Parity tests for {@link SpecialTokens} against the special-token rules of llama.cpp
 * {@code llama_vocab::impl::load} and {@code tokenizer_st_partition}, on synthetic vocabularies.
 * Token types follow {@code LLAMA_TOKEN_TYPE_*}: 1 normal, 3 control, 4 user-defined.
 */
public class SpecialTokensTest {

    private static final int NORMAL = 1;
    private static final int CONTROL = 3;
    private static final int USER_DEFINED = 4;

    /** A BPE vocabulary of the given texts; a text listed in {@code control} gets type 3. */
    private static Map<String, Object> bpe(String pre, List<String> tokens, List<String> control) {
        List<Object> types = new ArrayList<>();
        for (String t : tokens) types.add(control.contains(t) ? CONTROL : NORMAL);
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, pre);
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        return md;
    }

    private static SpecialTokens special(Map<String, Object> md) {
        return SpecialTokens.of(TokenizerVocab.fromMetadata(md));
    }

    @Test
    public void bpeDefaultsOutOfRangeAreDisabledAndOverridesApply() {
        Map<String, Object> md = bpe("gpt-2", Arrays.asList("a", "b", "<|endoftext|>"), Arrays.asList("<|endoftext|>"));
        md.put("tokenizer.ggml.bos_token_id", 2);
        md.put("tokenizer.ggml.padding_token_id", 99);
        SpecialTokens s = special(md);
        assertEquals(2, s.bos());
        assertEquals(2, s.cls());
        assertEquals(SpecialTokens.NULL, s.eos());
        assertEquals(SpecialTokens.NULL, s.pad());
        assertEquals(2, s.eot());
        assertEquals(Arrays.asList(2), new ArrayList<>(s.eogIds()));
        assertThrows(IllegalStateException.class, s::requireEos);
    }

    @Test
    public void detectedEndOfTurnBecomesControlAndEog() {
        SpecialTokens s = special(bpe("qwen2", Arrays.asList("x", "<|im_end|>", "<|eom_id|>"),
                Arrays.<String>asList()));
        assertEquals(1, s.eot());
        assertEquals(2, s.eom());
        assertTrue(s.hasAttr(1, TokenizerVocab.ATTR_CONTROL));
        assertTrue(s.isEog(1) && s.isEog(2));
        assertTrue(s.isHidden(1));
        assertFalse(s.isHidden(0));
    }

    @Test
    public void fimTokensAreDetectedAndPadRepoSepEndGeneration() {
        List<String> tokens = Arrays.asList("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
                "<|repo_name|>", "<|file_sep|>");
        SpecialTokens s = special(bpe("qwen2", tokens, tokens));
        assertEquals(0, s.fimPre());
        assertEquals(1, s.fimSuf());
        assertEquals(2, s.fimMid());
        assertEquals(3, s.fimPad());
        assertEquals(4, s.fimRep());
        assertEquals(5, s.fimSep());
        assertEquals(Arrays.asList(3, 4, 5), new ArrayList<>(s.eogIds()));
    }

    @Test
    public void harmonyEndIsADelimiterNotEog() {
        List<String> tokens = Arrays.asList("<|return|>", "<|call|>", "<|end|>", "<|channel|>", "<|message|>",
                "<|start|>", "hi");
        SpecialTokens s = special(bpe("gpt-4o", tokens, tokens.subList(0, 6)));
        assertEquals(Arrays.asList(0, 1), new ArrayList<>(s.eogIds()));
        assertEquals(TokenizerVocab.ATTR_USER_DEFINED, s.attr(2));
        assertEquals(TokenizerVocab.ATTR_USER_DEFINED, s.attr(3));
        assertFalse(s.isHidden(2));
        assertFalse(s.isHidden(3));
        assertTrue(s.isHidden(0));
    }

    @Test
    public void endOfSentenceIsNormalNextToToolResponse() {
        List<String> tokens = Arrays.asList("</s>", "<|tool_response>", "<eos>");
        SpecialTokens s = special(bpe("gemma4", tokens, tokens));
        assertEquals(Arrays.asList(1, 2), new ArrayList<>(s.eogIds()));
        assertEquals(TokenizerVocab.ATTR_NORMAL, s.attr(0));
        assertFalse(s.isHidden(0));
    }

    @Test
    public void flintUsesUserAndObservationAsStops() {
        List<String> tokens = Arrays.asList("<|user|>", "<|observation|>", "x");
        assertEquals(Arrays.asList(0, 1), new ArrayList<>(special(bpe("flint", tokens, tokens)).eogIds()));
        assertTrue(special(bpe("gpt-2", tokens, tokens)).eogIds().isEmpty());
    }

    @Test
    public void unusedControlTokensAreMarked() {
        List<String> tokens = Arrays.asList("<unused0>", "unused", "<ctl>");
        SpecialTokens s = special(bpe("gpt-2", tokens, Arrays.asList("<unused0>", "<ctl>")));
        assertTrue(s.hasAttr(0, TokenizerVocab.ATTR_UNUSED));
        assertFalse(s.hasAttr(1, TokenizerVocab.ATTR_UNUSED));
        assertFalse(s.hasAttr(2, TokenizerVocab.ATTR_UNUSED));
    }

    @Test
    public void maskLstripForJinaCode() {
        Map<String, Object> md = bpe("jina-v2-code", Arrays.asList("a", "<mask>", "b"), Arrays.asList("<mask>"));
        SpecialTokens s = special(md);
        assertTrue(s.hasAttr(1, TokenizerVocab.ATTR_LSTRIP));
        List<SpecialTokens.Fragment> f = s.partition("a  <mask> b", true);
        assertEquals(3, f.size());
        assertEquals("a", f.get(0).text());
        assertEquals(1, f.get(1).token());
        assertEquals(" b", f.get(2).text());
    }

    @Test
    public void phi3RstripsSpecialTokensExceptUnkBosAndEndOfText() {
        List<String> tokens = Arrays.asList("<unk>", "<s>", "</s>", "<|endoftext|>", "<|assistant|>", "x");
        Map<String, Object> md = bpe("gpt-2", tokens, tokens.subList(0, 5));
        md.put(TokenizerVocab.KEY_NAME, "Phi-3 Mini");
        SpecialTokens s = special(md);
        assertTrue(s.hasAttr(4, TokenizerVocab.ATTR_RSTRIP));
        assertFalse(s.hasAttr(0, TokenizerVocab.ATTR_RSTRIP));
        assertFalse(s.hasAttr(1, TokenizerVocab.ATTR_RSTRIP));
        assertFalse(s.hasAttr(3, TokenizerVocab.ATTR_RSTRIP));
        List<SpecialTokens.Fragment> f = s.partition("<|assistant|>  x", true);
        assertEquals(2, f.size());
        assertEquals("x", f.get(1).text());

        Map<String, Object> broken = bpe("gpt-2", Arrays.asList("<s>", "x"), Arrays.asList("<s>"));
        broken.put(TokenizerVocab.KEY_NAME, "phi3-small");
        assertThrows(IllegalArgumentException.class, () -> special(broken));
    }

    @Test
    public void partitionKeepsControlTextUnlessParsingSpecial() {
        List<String> tokens = Arrays.asList("<ctl>", "<ud>", "a");
        Map<String, Object> md = bpe("gpt-2", tokens, Arrays.asList("<ctl>"));
        @SuppressWarnings("unchecked")
        List<Object> types = (List<Object>) md.get(TokenizerVocab.KEY_TOKEN_TYPE);
        types.set(1, USER_DEFINED);
        SpecialTokens s = special(md);
        List<SpecialTokens.Fragment> plain = s.partition("a<ctl>a<ud>", false);
        assertEquals(2, plain.size());
        assertEquals("a<ctl>a", plain.get(0).text());
        assertEquals(1, plain.get(1).token());
        assertEquals(4, s.partition("a<ctl>a<ud>", true).size());
    }

    @Test
    public void suppressTokensDropOutOfRangeIds() {
        Map<String, Object> md = bpe("gpt-2", Arrays.asList("a", "b"), Arrays.<String>asList());
        md.put("tokenizer.ggml.suppress_tokens", new ArrayList<Object>(Arrays.asList(1, 7, 0)));
        assertArrayEquals(new int[]{1, 0}, special(md).suppressTokens());
    }

    @Test
    public void malformedModelNameFailsTheLoad() {
        Map<String, Object> md = bpe("gpt-2", Arrays.asList("a", "b"), Arrays.<String>asList());
        md.put(TokenizerVocab.KEY_NAME, 42);
        assertThrows(IllegalArgumentException.class, () -> special(md));
    }

    @Test
    public void glmClearsDefaultBos() {
        List<String> tokens = new ArrayList<>();
        for (int i = 0; i < 12; i++) tokens.add("t" + i);
        SpecialTokens s = special(bpe("glm4", tokens, Arrays.<String>asList()));
        assertEquals(SpecialTokens.NULL, s.bos());
        assertEquals(11, s.eos());
    }
}
