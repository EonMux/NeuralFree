package com.eonmux.neuralfree.gguf.engine.sampling;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The grammar piece tables are built once per vocabulary object, each under its own slot: a table still building
 * for one vocabulary does not hold back the table of another, and a second caller of the same vocabulary receives
 * the one table the first built.
 */
public class GrammarPiecesCacheTest {

    /** A vocabulary of single-letter pieces whose first piece read waits on {@code gate} when one is given. */
    private static final class LetterVocab implements SamplerVocab {
        private final CountDownLatch started;
        private final CountDownLatch gate;

        LetterVocab(CountDownLatch started, CountDownLatch gate) {
            this.started = started;
            this.gate = gate;
        }

        @Override public int vocabSize() { return 4; }
        @Override public boolean isEog(int id) { return false; }
        @Override public int eotToken() { return -1; }
        @Override public int eosToken() { return -1; }
        @Override public String pieceText(int id) { return String.valueOf((char) ('a' + id)); }
        @Override public int[] tokenize(String text) { return new int[0]; }

        @Override
        public byte[] piece(int id) {
            if (id == 0 && gate != null) {
                started.countDown();
                try {
                    if (!gate.await(30, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("the gate never opened");
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(e);
                }
            }
            return pieceText(id).getBytes(StandardCharsets.UTF_8);
        }
    }

    @Test
    public void aSlowBuildDoesNotHoldBackAnotherVocabularysTable() throws Exception {
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch gate = new CountDownLatch(1);
        LetterVocab slow = new LetterVocab(started, gate);
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<GrammarPieces> slowTable = pool.submit(() -> GrammarPieces.of(slow));
            assertTrue(started.await(30, TimeUnit.SECONDS));
            Future<GrammarPieces> other = pool.submit(() -> GrammarPieces.of(new LetterVocab(null, null)));
            assertEquals("the other table builds while the first waits", 4,
                    other.get(10, TimeUnit.SECONDS).size());
            gate.countDown();
            GrammarPieces built = slowTable.get(30, TimeUnit.SECONDS);
            assertSame("a later caller receives the table the first built", built, GrammarPieces.of(slow));
        } finally {
            gate.countDown();
            pool.shutdownNow();
        }
    }
}
