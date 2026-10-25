package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfBuiltinVocab}: the bundled llama.cpp vocabularies cut to a configured size, as {@code _set_vocab_builtin}
 * writes them for checkpoints without tokenizer files.
 */
public class HfBuiltinVocabTest {

    /** Mamba's {@code vocab_size} rounded up to {@code pad_vocab_size_multiple} (mamba.py:30-44). */
    private static final int MAMBA_VOCAB = 50280;

    @Test
    public void aMambaShapedConfigCutsTheGptNeoxVocabulary() throws IOException {
        Map<String, Object> md = HfBuiltinVocab.of(HfBuiltinVocab.GPT_NEOX, MAMBA_VOCAB);
        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("the converter's default pre-tokenizer for gpt-neox", "mpt", md.get(GgufKeys.TOKENIZER_PRE));
        List<?> tokens = (List<?>) md.get(GgufKeys.TOKENIZER_LIST);
        assertEquals(MAMBA_VOCAB, tokens.size());
        assertEquals(MAMBA_VOCAB, ((List<?>) md.get(GgufKeys.TOKENIZER_TOKEN_TYPE)).size());
        assertNull("gpt-neox has no scores", md.get(GgufKeys.TOKENIZER_SCORES));
        assertFalse(((List<?>) md.get(GgufKeys.TOKENIZER_MERGES)).isEmpty());
        assertEquals(0, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(0, md.get(GgufKeys.TOKENIZER_UNK_ID));
        assertThrows(UnsupportedOperationException.class, () -> md.put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> tokens.remove(0));
    }

    @Test
    public void aLargerSizeIsNotPadded() throws IOException {
        List<?> all = (List<?>) HfBuiltinVocab.of(HfBuiltinVocab.LLAMA_SPM, 40000).get(GgufKeys.TOKENIZER_LIST);
        assertEquals(32000, all.size());
        assertFalse(all.stream().anyMatch(t -> ((String) t).startsWith("[PAD")));
    }

    @Test
    public void llamaSpmKeepsScoresFlagsAndNoMerges() throws IOException {
        Map<String, Object> md = HfBuiltinVocab.of(HfBuiltinVocab.LLAMA_SPM, 32000);
        assertEquals("llama", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("default", md.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(32000, ((List<?>) md.get(GgufKeys.TOKENIZER_SCORES)).size());
        assertNull(md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(1, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(2, md.get(GgufKeys.TOKENIZER_EOS_ID));
        assertEquals(true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(false, md.get(GgufKeys.TOKENIZER_ADD_EOS));
    }

    @Test
    public void specialIdsBeyondTheCutAreDropped() throws IOException {
        Map<String, Object> md = HfBuiltinVocab.of(HfBuiltinVocab.LLAMA_SPM, 2);
        assertEquals(2, ((List<?>) md.get(GgufKeys.TOKENIZER_LIST)).size());
        assertEquals(1, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertTrue(md.containsKey(GgufKeys.TOKENIZER_UNK_ID));
        assertFalse("eos 2 is outside two tokens", md.containsKey(GgufKeys.TOKENIZER_EOS_ID));
    }

    @Test
    public void unknownNamesAndSizesAreRejected() {
        assertEquals(Arrays.asList(HfBuiltinVocab.GPT_NEOX, HfBuiltinVocab.LLAMA_SPM), HfBuiltinVocab.names());
        assertThrows(IllegalArgumentException.class, () -> HfBuiltinVocab.of("gpt-2", 10));
        assertThrows(IllegalArgumentException.class, () -> HfBuiltinVocab.of(HfBuiltinVocab.GPT_NEOX, 0));
    }
}
