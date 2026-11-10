package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.GbnfParser;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.ref.Reference;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The heap a grammar stage keeps between masks: a constrained sequence that is open but not masking holds its parse
 * state and its mask memo, not row-sized scratch. Each stage masks a whole row of a 151,669-token vocabulary twice,
 * once from the decoded piece table and once with a character pending across tokens (the path that decodes the row
 * itself), and the heap retained by the stages is read after a collection while they stay reachable.
 */
public class GrammarStageRetainedHeapTest {

    private static final int VOCAB_SIZE = 151_669;
    private static final int STAGES = 32;
    /** Retained bytes allowed per stage: the mask memo's budget, the largest thing a stage keeps by design. */
    private static final long BOUND_PER_STAGE = GrammarMaskMemo.BUDGET_BYTES;
    /** A piece that leaves the lead byte of a three-byte character pending. */
    private static final int PENDING = 1;

    @Test
    public void anOpenConstrainedSequenceKeepsNoRowSizedScratch() {
        SamplerVocab vocab = vocab();
        Grammar grammar = GbnfParser.parse("root ::= [^\"]*\n");
        float[] row = new float[VOCAB_SIZE];
        Reference.reachabilityFence(drive(new GrammarStage(grammar, vocab), vocab, row)); // the shared table, warm
        long before = usedAfterCollection();
        List<GrammarStage> open = new ArrayList<>();
        for (int i = 0; i < STAGES; i++) {
            open.add(drive(new GrammarStage(grammar, vocab), vocab, row));
        }
        long retained = usedAfterCollection() - before;
        Reference.reachabilityFence(open);
        long perStage = retained / STAGES;
        assertTrue("a stage between masks retains " + perStage + " bytes, bound " + BOUND_PER_STAGE,
                perStage <= BOUND_PER_STAGE);
    }

    /** Masks a whole row, accepts a piece that leaves a character pending, and masks the row again. */
    private static GrammarStage drive(GrammarStage stage, SamplerVocab vocab, float[] row) {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], Map.of(), List.of(stage), vocab);
        sampler.apply(row);
        sampler.accept(PENDING);
        Sampler.Candidates masked = sampler.apply(row);
        assertTrue("the pending character leaves its continuations admitted",
                masked.logit(VOCAB_SIZE - 1) == 0f);
        return stage;
    }

    private static long usedAfterCollection() {
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        long used = Long.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            System.gc();
            used = Math.min(used, memory.getHeapMemoryUsage().getUsed());
        }
        return used;
    }

    /**
     * Id 0 ends generation, id {@link #PENDING} is {@code a} and the lead byte of {@code 中}, the last id is the
     * character's two continuation bytes, and every other id is two to six random ASCII letters or a whole
     * {@code 中} after them.
     */
    private static SamplerVocab vocab() {
        byte[][] pieces = new byte[VOCAB_SIZE][];
        Random rnd = new Random(7);
        pieces[0] = new byte[0];
        pieces[PENDING] = new byte[]{'a', (byte) 0xE4};
        pieces[VOCAB_SIZE - 1] = new byte[]{(byte) 0xB8, (byte) 0xAD};
        for (int id = 2; id < VOCAB_SIZE - 1; id++) {
            StringBuilder text = new StringBuilder();
            int letters = 2 + rnd.nextInt(5);
            for (int i = 0; i < letters; i++) {
                text.append((char) ('a' + rnd.nextInt(26)));
            }
            if (rnd.nextInt(8) == 0) {
                text.append('中');
            }
            pieces[id] = text.toString().getBytes(StandardCharsets.UTF_8);
        }
        return new SamplerVocab() {
            @Override public int vocabSize() { return VOCAB_SIZE; }
            @Override public boolean isEog(int id) { return id == 0; }
            @Override public int eotToken() { return 0; }
            @Override public int eosToken() { return 0; }
            @Override public byte[] piece(int id) { return pieces[id]; }
            @Override public String pieceText(int id) { return new String(pieces[id], StandardCharsets.UTF_8); }
            @Override public int[] tokenize(String text) { return new int[0]; }
        };
    }
}
