package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LookupDrafterFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * Per-sequence lookup drafting ({@link BatchLookupDrafter}) over a paged record holding media rows: the drafters
 * search the sequence's text tokens only, as llama.cpp's server drafts over {@code get_text_tokens}, so neither the
 * stand-in id an n-gram model records for a media row nor the {@code -1} of a model without one ever enters a
 * proposal. vLLM's prompt-lookup size range searches the same text, and the suffix drafters of the sequences share
 * one request tree.
 */
public class BatchLookupDrafterMediaTest {

    private static final int PAGE = 4;
    private static final int N_CTX = 32;
    private static final int STAND_IN = 99;

    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX).ubatch(8)
                .parallel(2).kvPageSize(PAGE).kvUnified(true).build();
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, N_CTX, false, KvWindow.FULL_CAUSAL, PAGE, N_CTX / PAGE);
        KvLayout kv = KvLayout.of(N_CTX, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, cfg.ubatch()).group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build()).config(cfg).build();
        return new PagedSequences(plan, 2);
    }

    private static void write(PagedSequences mem, int[] tokens, int[] positions, int embeddingToken) {
        write(mem, 0, tokens, positions, embeddingToken);
    }

    private static void write(PagedSequences mem, int seq, int[] tokens, int[] positions, int embeddingToken) {
        int n = tokens.length;
        StepLayout layout = StepLayout.of(n, new int[] {seq}, new int[] {0}, new int[] {n},
                new int[] {positions[0]}, new int[] {n - 1}, true);
        mem.commit(mem.prepare(layout, positions));
        mem.appendRows(seq, tokens, positions, 0, n, embeddingToken);
    }

    private static int[] positions(int from, int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = from + i;
        }
        return p;
    }

    /** {@code 1 5 6}, three media rows at position 3, then {@code 7 8 9 10 12 last} from position 5. */
    private static PagedSequences history(int embeddingToken, int last) {
        PagedSequences mem = memory();
        write(mem, new int[] {1, 5, 6}, new int[] {0, 1, 2}, embeddingToken);
        write(mem, new int[] {-1, -1, -1}, new int[] {3, 3, 3}, embeddingToken);
        write(mem, new int[] {7, 8, 9, 10, 12, last}, new int[] {5, 6, 7, 8, 9, 10}, embeddingToken);
        return mem;
    }

    /** The {@code ngram-simple} drafter with a 2-token lookup and a 3-token draft over {@code memory}. */
    private static BatchLookupDrafter simple(PagedSequences memory) throws IOException {
        NgramDrafter.Options sizes = NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(2).sizeM(3).build();
        EngineConfig cfg = EngineConfig.builder().draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(sizes).build()).build();
        return new BatchLookupDrafter(LookupDrafterFactory.open(cfg), memory);
    }

    /** The {@code ngram-simple} drafter of the lookup sizes {@code sizeNMin .. sizeN} and a 3-token draft. */
    private static BatchLookupDrafter range(PagedSequences memory, int sizeNMin, int sizeN) throws IOException {
        NgramDrafter.Options sizes = NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE)
                .sizeN(sizeN).sizeNMin(sizeNMin).sizeM(3).build();
        EngineConfig cfg = EngineConfig.builder().draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(sizes).build()).build();
        return new BatchLookupDrafter(LookupDrafterFactory.open(cfg), memory);
    }

    /** The {@code suffix} drafter over {@code memory}. */
    private static BatchLookupDrafter suffix(PagedSequences memory) throws IOException {
        EngineConfig cfg = EngineConfig.builder().draftType(EngineConfig.DraftType.SUFFIX).build();
        return new BatchLookupDrafter(LookupDrafterFactory.open(cfg), memory);
    }

    @Test
    public void theProposalSkipsTheStandInOfMediaRows() throws IOException {
        BatchLookupDrafter drafter = simple(history(STAND_IN, 5));
        assertArrayEquals("the text after the matched 5 6, the media rows left out", new int[] {7, 8, 9},
                drafter.draft(0, 6, 8));
    }

    @Test
    public void theProposalSkipsMediaRowsWithoutAnId() throws IOException {
        BatchLookupDrafter drafter = simple(history(-1, 5));
        assertArrayEquals(new int[] {7, 8, 9}, drafter.draft(0, 6, 8));
    }

    @Test
    public void aSampledStandInMatchesNoMediaRow() throws IOException {
        BatchLookupDrafter drafter = simple(history(STAND_IN, 6));
        assertArrayEquals("6 then the stand-in is a pattern of the record, not of the text", new int[0],
                drafter.draft(0, STAND_IN, 8));
    }

    @Test
    public void thePromptLookupRangeSearchesTheTextTokens() throws IOException {
        // text 1 5 6 7 8 9 10 12 5, sampled 6: only the 2-gram (5, 6) recurs, after the media rows were left out
        assertArrayEquals("vLLM's 2..4 range falls back to the matching 2-gram", new int[] {7, 8, 9},
                range(history(STAND_IN, 5), 2, 4).draft(0, 6, 8));
        assertArrayEquals("llama.cpp's single size 4 needs (10, 12, 5, 6)", new int[0],
                range(history(STAND_IN, 5), 4, 4).draft(0, 6, 8));
        assertArrayEquals("the range skips rows without an id the same way", new int[] {7, 8, 9},
                range(history(-1, 5), 2, 4).draft(0, 6, 8));
    }

    /** A {@code suffix} drafter whose spec factor lets a one-token match draft several tokens. */
    private static BatchLookupDrafter sharedSuffix(PagedSequences memory, int maxCachedRequests) throws IOException {
        EngineConfig cfg = EngineConfig.builder().draftType(EngineConfig.DraftType.SUFFIX)
                .speculativeOptions(SpeculativeOptions.builder().suffix(SuffixDrafter.Options.builder()
                        .maxSpecFactor(4).maxCachedRequests(maxCachedRequests).build()).build()).build();
        return new BatchLookupDrafter(LookupDrafterFactory.open(cfg), memory);
    }

    /** Sequence {@code seq} samples each of {@code response}, drafting before each, and commits it. */
    private static void respond(BatchLookupDrafter drafter, PagedSequences mem, int seq, int... response) {
        for (int token : response) {
            drafter.draft(seq, token, 4);
            drafter.accept(seq, 0);
            write(mem, seq, new int[] {token}, positions(mem.sequence(seq).textTokens().length, 1), -1);
        }
    }

    @Test
    public void theSuffixDraftersOfTheSequencesShareTheirResponses() throws IOException {
        PagedSequences mem = memory();
        write(mem, 0, new int[] {1, 2}, positions(0, 2), -1);
        write(mem, 1, new int[] {3}, positions(0, 1), -1);
        BatchLookupDrafter drafter = sharedSuffix(mem, SuffixDrafter.DEFAULT_MAX_CACHED_REQUESTS);
        respond(drafter, mem, 0, 7, 8, 9);
        assertEquals(1, drafter.suffixCachedRequests());
        // sequence 1 never saw 7: the still running response of sequence 0 continues it
        assertArrayEquals(new int[] {8, 9}, drafter.draft(1, 7, 4));
        assertEquals("both running requests in one tree", 2, drafter.suffixCachedRequests());

        // a reset slot ends its request; the finished response stays in the shared tree
        drafter.reset(0);
        drafter.reset(1);
        assertArrayEquals(new int[] {8, 9}, drafter.draft(1, 7, 4));
        assertEquals(3, drafter.suffixCachedRequests());
        drafter.close();
    }

    @Test
    public void theSharedRequestTreeKeepsItsCapacityAcrossSequences() throws IOException {
        PagedSequences mem = memory();
        write(mem, 0, new int[] {1}, positions(0, 1), -1);
        write(mem, 1, new int[] {2}, positions(0, 1), -1);
        BatchLookupDrafter drafter = sharedSuffix(mem, 1);
        respond(drafter, mem, 0, 7);
        respond(drafter, mem, 1, 5);
        assertEquals("the second sequence's request evicted the first", 1, drafter.suffixCachedRequests());
        // sequence 0 runs on, evicted: its 8 9 no longer enter the tree
        respond(drafter, mem, 0, 8, 9);
        assertEquals(1, drafter.suffixCachedRequests());
        assertArrayEquals("no response continues 8 for sequence 1", new int[0], drafter.draft(1, 8, 4));
        drafter.close();
    }

    @Test
    public void theSuffixTreeLeavesMediaRowsOut() throws IOException {
        BatchLookupDrafter drafter = suffix(history(STAND_IN, 5));
        int[] proposal = drafter.draft(0, 6, 8);
        assertTrue("the suffix tree continues the matched 5 6", proposal.length > 0);
        assertEquals("the text after 5 6, the media rows left out", 7, proposal[0]);
        for (int token : proposal) {
            assertNotEquals("no media stand-in in a suffix proposal", STAND_IN, token);
            assertTrue("no unnamed media row in a suffix proposal", token >= 0);
        }
        drafter.close();
    }
}
