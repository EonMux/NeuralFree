package com.eonmux.neuralfree.gguf.gpu;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Guards the S5 input-transform sites: every consumer of a GPU weight either routes the product through the
 * {@code GpuOps} entry points (whose {@code projectionInput} applies the full HADAMARD &rarr; ACT_QUANT &rarr;
 * PERMUTE chain) or refuses a non-identity input transform. A source-scan test on purpose: the sites gain methods
 * over time, and a reflection sweep sees only the weights a model actually binds, while every site must refuse at
 * bind time too.
 *
 * <p>Sites that only use {@link com.eonmux.neuralfree.gguf.spec.DsaSpec#hadamard()} (the lightning-indexer KV
 * rotation) carry no weight-transform duty; the test pins that they reference no weight input transform at all.
 * Runs from the project directory (the Gradle test working directory).</p>
 */
public class InputTransformSitesTest {

    private static final Path MAIN = Paths.get("src", "main", "java", "com", "eonmux", "neuralfree", "gguf");

    /**
     * Sites whose products route through {@code GpuOps} matvec/matmul, which applies the full input chain. The
     * sparse block drivers ({@code MsaBlocks}/{@code SparseBlocks}) serve their absorbed products through the
     * mixer sites below them, so they live in {@link #REFUSING} and {@link #INDEXER_ROTATION_ONLY}.
     */
    private static final String[] DELEGATING = {
        "ops/act/GpuDenseFfn.java",
        "ops/attn/GpuAttentionMixer.java",
        "ops/dsv4/GpuDsv4Mixer.java",
        "ops/mla/GpuMlaMixer.java",
        "ops/rwkv/RwkvBlocks.java",
        "ops/sparse/GpuDsaMlaMixer.java"
    };

    /** Sites that launch a weight directly (LoRA delta sites, id-matvec products) and refuse the chain at bind. */
    private static final String[] REFUSING = {
        "gpu/block/GpuProjectionDeltas.java",
        "ops/act/GpuDenseFfn.java",
        "ops/attn/GpuAttentionMixer.java",
        "ops/attn/GpuValueExperts.java",
        "ops/dsv4/GpuDsv4Mixer.java",
        "ops/mla/GpuMlaMixer.java",
        "ops/moe/MoeExpertWeights.java",
        "ops/sparse/GpuDsaMlaMixer.java",
        "ops/sparse/MsaBlocks.java"
    };

    /** Sites that hold no weight: their only Hadamard use is the DSA indexer KV rotation. */
    private static final String[] INDEXER_ROTATION_ONLY = {
        "memory/MemoryPlanner.java",
        "spec/DsaSpec.java",
        "ops/sparse/SparseBlocks.java"
    };

    /** Source of a main class, keyed by its path below the {@code gguf} package. */
    private static String source(String relativePath) {
        Path p = MAIN.resolve(relativePath);
        try {
            return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
        } catch (IOException e) {
            fail("cannot read " + p + ": " + e);
            return null;
        }
    }

    @Test
    public void everyRefusalSiteAlsoRejectsANonIdentityInputTransform() {
        for (String site : REFUSING) {
            String src = source(site);
            assertTrue(site + ": every refusal must check a hadamard part", src.contains("hadamard()"));
            if (!refusesTransform(src)) {
                fail(site + " refuses a directly attached Hadamard but not the input-transform chain; extend the"
                        + " refusal to w.hadamard() != null || (w.inputTransform() != null"
                        + " && !w.inputTransform().isIdentity())");
            }
        }
    }

    /** The refusal messages stay one family: each refusal names the input transform it rejects. */
    @Test
    public void refusalMessagesStayOneFamily() {
        for (String site : REFUSING) {
            String src = source(site);
            assertTrue(site + ": every refusal message must name the refused input transform",
                    src.contains("input transform"));
        }
    }

    /** The row-slicing site must carry the full chain to its sliced block, not only the direct rotation. */
    @Test
    public void rowSlicingSitePropagatesTheChain() {
        String src = source("ops/rwkv/RwkvBlocks.java");
        assertTrue("RwkvBlocks.rowBlock must reattach the input transform of the weight it slices rows of",
                src.contains("withInputTransform(") && src.contains("inputPerm()"));
    }

    @Test
    public void delegatingSitesRouteTheirProductsThroughGpuOps() {
        for (String site : DELEGATING) {
            String src = source(site);
            assertTrue(site + " no longer routes its products through the GpuOps entry points",
                    src.contains("ops.matmul(") || src.contains("ops.matvec("));
        }
    }

    @Test
    public void indexerRotationOnlySitesReferenceNoWeightTransform() {
        for (String site : INDEXER_ROTATION_ONLY) {
            String src = source(site);
            assertFalse(site + " now consumes a weight input transform and must join the site table",
                    src.contains(".inputTransform()") || src.contains(".hasInputTransform()"));
        }
    }

    @Test
    public void gpuOpsAppliesTheWholeChainIntoReservedStaging() {
        String src = source("gpu/forward/GpuOps.java");
        assertTrue("GpuOps lost the projectionInput chain application", src.contains("projectionInput("));
        assertTrue(src.contains("actFakeQuant("));
        assertTrue(src.contains("permuteCols("));
        assertTrue("GpuOps must reserve its own staging kind for the PERMUTE gather",
                src.contains("GpuScratch.Staging.PERMUTED"));
        String scratch = source("gpu/forward/GpuScratch.java");
        assertTrue("GpuScratch lost the PERMUTED staging kind", scratch.contains("PERMUTED"));
    }

    /**
     * True when {@code src} performs a transform refusal: an occurrence of
     * {@code x.hadamard() != null || (x.inputTransform() != null && !x.inputTransform().isIdentity())} with any
     * receiver {@code x} and any spacing.
     */
    private static boolean refusesTransform(String src) {
        String flat = src.replaceAll("\\s+", " ");
        int hadamard = flat.indexOf("hadamard() != null");
        while (hadamard >= 0) {
            String window = flat.substring(hadamard, Math.min(flat.length(), hadamard + 400));
            int chain = window.indexOf(".inputTransform() != null && !");
            if (chain >= 0 && window.indexOf(".inputTransform().isIdentity()", chain) >= 0) {
                return true;
            }
            hadamard = flat.indexOf("hadamard() != null", hadamard + 1);
        }
        return false;
    }
}
