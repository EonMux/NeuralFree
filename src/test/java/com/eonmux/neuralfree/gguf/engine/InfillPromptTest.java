package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** llama.cpp's FIM prompt layout ({@code format_prompt_infill}): order, truncation, extra context, refusals. */
public class InfillPromptTest {

    private static final int PRE = 10;
    private static final int SUF = 11;
    private static final int MID = 12;
    private static final int REP = 13;
    private static final int SEP = 14;

    private static GGUFTokenizer tokenizer(boolean repoTokens) throws IOException {
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = SyntheticModels.llama()
                .u32("tokenizer.ggml.fim_pre_token_id", PRE)
                .u32("tokenizer.ggml.fim_suf_token_id", SUF)
                .u32("tokenizer.ggml.fim_mid_token_id", MID)
                .bool("tokenizer.ggml.add_bos_token", false);
        if (repoTokens) {
            w = w.u32("tokenizer.ggml.fim_rep_token_id", REP).u32("tokenizer.ggml.fim_sep_token_id", SEP);
        }
        Path path = w.writeTempFile();
        try (GGUFFileParser p = new GGUFFileParser(path.toString())) {
            return new GGUFTokenizer(p);
        }
    }

    @Test
    public void prefixSuffixMiddleAndTheSpmOrder() throws IOException {
        GGUFTokenizer tok = tokenizer(false);
        int[] prefix = {100, 101, 102};
        int[] suffix = {200, 201};
        int[] prompt = {50};
        assertArrayEquals(new int[]{PRE, 100, 101, 102, 50, SUF, 200, 201, MID},
                InfillPrompt.build(tok, prefix, suffix, Collections.emptyList(), prompt, 2048, 16, 4096, false));
        assertArrayEquals(new int[]{SUF, 200, 201, PRE, 100, 101, 102, 50, MID},
                InfillPrompt.build(tok, prefix, suffix, Collections.emptyList(), prompt, 2048, 16, 4096, true));
    }

    @Test
    public void theBatchKeepsThePrefixTailAndTheSuffixHead() throws IOException {
        GGUFTokenizer tok = tokenizer(false);
        int[] prefix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int[] suffix = {21, 22, 23, 24, 25, 26};
        // nBatch 8: the prefix keeps its last 6, the suffix its first 8/4 - 2 = 0
        assertArrayEquals(new int[]{PRE, 4, 5, 6, 7, 8, 9, SUF, MID},
                InfillPrompt.build(tok, prefix, suffix, Collections.emptyList(), new int[0], 8, 0, 4096, false));
    }

    @Test
    public void extraChunksLeadUnderRepoAndFileSeparators() throws IOException {
        GGUFTokenizer tok = tokenizer(true);
        int[] ids = InfillPrompt.build(tok, new int[]{100}, new int[]{200},
                Arrays.asList(new InfillPrompt.Chunk("a.py", "x = 1")), new int[0], 2048, 16, 4096, false);
        assertEquals(REP, ids[0]);
        int seps = 0;
        for (int id : ids) {
            seps += id == SEP ? 1 : 0;
        }
        assertEquals("one separator per chunk and one for the current file", 2, seps);
        assertArrayEquals(new int[]{PRE, 100, SUF, 200, MID}, Arrays.copyOfRange(ids, ids.length - 5, ids.length));
        int[] tight = InfillPrompt.build(tok, new int[]{100}, new int[]{200},
                Arrays.asList(new InfillPrompt.Chunk(null, "x = 1")), new int[0], 2048, 16, 2048, false);
        assertArrayEquals("no room beyond the batch: no extra context", new int[]{PRE, 100, SUF, 200, MID}, tight);
    }

    @Test
    public void aVocabularyWithoutFimTokensRefusesByName() throws IOException {
        Path path = SyntheticModels.llama().writeTempFile();
        try (GGUFFileParser p = new GGUFFileParser(path.toString())) {
            GGUFTokenizer plain = new GGUFTokenizer(p);
            String missing = InfillPrompt.missingTokens(plain);
            assertTrue(missing, missing != null && missing.contains("prefix token is missing"));
            assertThrows(IllegalArgumentException.class, () -> InfillPrompt.build(plain, new int[0], new int[0],
                    Collections.emptyList(), new int[0], 2048, 0, 4096, false));
        }
        assertNull(InfillPrompt.missingTokens(tokenizer(false)));
    }
}
