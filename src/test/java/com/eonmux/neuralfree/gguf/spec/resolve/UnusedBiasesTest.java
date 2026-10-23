package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** The loaded biases whose llama.cpp graph passes {@code NULL} in their place. */
public class UnusedBiasesTest {

    @Test
    public void nomicBertFfnRunsWithoutBiases() {
        assertTrue(UnusedBiases.ffnUnused(Arch.NOMIC_BERT));
        assertTrue(UnusedBiases.ffn().contains(Arch.NOMIC_BERT));
    }

    @Test
    public void nomicBertMoeDenseLayersKeepTheirBiases() {
        assertFalse(UnusedBiases.ffnUnused(Arch.NOMIC_BERT_MOE));
        assertFalse(UnusedBiases.ffnUnused(Arch.BERT));
    }
}
