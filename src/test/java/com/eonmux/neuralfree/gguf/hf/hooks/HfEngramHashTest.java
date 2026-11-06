package com.eonmux.neuralfree.gguf.hf.hooks;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfEngramLayout} and {@link HfEngramHash} against numpy and the converter: the bucket moduli of
 * {@code EngramLayout.from_args}, and numpy's {@code default_rng(10007 * layer).integers(...)} stream captured from
 * numpy 2.5.3 ({@code SeedSequence.generate_state(4, uint64)} and the odd multipliers).
 */
public class HfEngramHashTest {

    @Test
    public void theSeedSequenceMatchesNumpy() {
        assertArrayEquals(new long[]{
                        Long.parseUnsignedLong("15793235383387715774"),
                        Long.parseUnsignedLong("12390638538380655177"),
                        Long.parseUnsignedLong("2361836109651742017"),
                        Long.parseUnsignedLong("3188717715514472916")},
                HfEngramHash.generateState(0));
        assertArrayEquals(new long[]{
                        Long.parseUnsignedLong("8634291315755388167"),
                        Long.parseUnsignedLong("11042498673593962405"),
                        Long.parseUnsignedLong("15418481100450270793"),
                        Long.parseUnsignedLong("9041516862649747678")},
                HfEngramHash.generateState(10007));
        assertArrayEquals(new long[]{
                        Long.parseUnsignedLong("3524533697516052101"),
                        Long.parseUnsignedLong("3575824834223264401"),
                        Long.parseUnsignedLong("4318004044551998539"),
                        Long.parseUnsignedLong("3075780137898789324")},
                HfEngramHash.generateState(30021));
    }

    @Test
    public void theMultipliersMatchNumpy() {
        assertArrayEquals(new long[]{19930781263555467L, 1258774283041005L, 9352535037976283L},
                HfEngramHash.multipliers(new int[]{1}, 3, 381));
        assertArrayEquals(new long[]{7159325749683359L, 21238261240028655L, 2829175741814069L},
                HfEngramHash.multipliers(new int[]{3}, 3, 381));
        assertArrayEquals(new long[]{19930781263555467L, 1258774283041005L, 9352535037976283L,
                        7159325749683359L, 21238261240028655L, 2829175741814069L},
                HfEngramHash.multipliers(new int[]{1, 3}, 3, 381));
        assertArrayEquals(new long[]{7593627661414631L, 479593001838623L, 3563315849468963L,
                        2727703110629359L, 8091777532450917L, 1077915957631161L},
                HfEngramHash.multipliers(new int[]{1, 3}, 3, 1000));
    }

    @Test
    public void thePrimesAreNeverReusedAcrossLayers() {
        int[][] primes = HfEngramLayout.primes(2, 3, 2, 100);
        assertArrayEquals(new int[]{101, 103, 107, 109}, primes[0]);
        assertArrayEquals(new int[]{113, 127, 131, 137}, primes[1]);
        assertTrue(HfEngramLayout.isPrime(2));
        assertTrue(HfEngramLayout.isPrime(999983));
        assertFalse(HfEngramLayout.isPrime(1));
        assertFalse(HfEngramLayout.isPrime(999981));
    }

    @Test
    public void theLayoutFlattensTheModuliIntoRowOffsets() {
        HfEngramLayout layout = layout(2, 3, 2, 100, 420, 508);
        assertArrayEquals(new long[]{420, 508}, layout.rows());
        assertArrayEquals(new long[]{101, 103, 107, 109, 113, 127, 131, 137}, layout.headSizes());
        assertArrayEquals(new long[]{0, 101, 204, 311, 420, 533, 660, 791}, layout.headOffsets());
        assertEquals(3, layout.maxNgram());
        assertEquals(2, layout.heads());
    }

    /** A layout of {@code layers} Engram layers, its row counts checked against {@code rows}. */
    static HfEngramLayout layout(int layers, int ngram, int heads, int vocab, long... rows) {
        StringBuilder ids = new StringBuilder();
        StringBuilder embeddings = new StringBuilder();
        for (int i = 0; i < layers; i++) {
            if (i > 0) {
                ids.append(", ");
                embeddings.append(", ");
            }
            ids.append(i);
            embeddings.append(rows[i]);
        }
        return HfEngramLayout.of(HfLlamaHooksTest.config("{'architectures': ['DeepseekV41ForCausalLM'],"
                + " 'num_hidden_layers': " + layers + ", 'engram_layer_ids': [" + ids
                + "], 'engram_max_ngram_size': " + ngram + ", 'engram_n_heads': " + heads
                + ", 'engram_vocab_size': " + vocab + ", 'engram_num_embeddings': [" + embeddings + "]"
                + ", 'engram_head_dim': 32, 'engram_pad_token_id': 7"
                + ", 'engram_compressed_vocab_size': 8}"));
    }
}
