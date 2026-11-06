package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The memory budget of {@link CpuRepack}: qualifying weights beyond the allowance stay in place, their products
 * stay bit-exact, the summary names what was repacked and why the rest was not, and lookup tables are never
 * copied.
 */
public class CpuRepackBudgetTest {

    /** A Q8_0 weight of 128 columns (one Q8_0x4 group per row) and {@code rows} rows costs rows * 136 bytes. */
    private static GgufTestWriter model() {
        return new GgufTestWriter()
                .architecture("llama")
                .tensorZeros("blk.0.ffn_up.weight", GGMLType.Q8_0, 128, 4)
                .tensorZeros("blk.0.ffn_down.weight", GGMLType.Q8_0, 128, 4)
                .tensorZeros("blk.1.ffn_up.weight", GGMLType.Q8_0, 128, 4)
                .tensorZeros("token_embd.weight", GGMLType.Q8_0, 128, 8);
    }

    private static final String[] MATMULS = {
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.1.ffn_up.weight"};

    @Test
    public void weightsBeyondTheBudgetStayInPlaceAndTheirProductsMatch() throws IOException {
        Assume.assumeTrue(CpuSimd.available());
        final long budget = 1000L;
        Path path = model().writeTempFile();
        String previousMode = System.getProperty(CpuRepack.PROP_REPACK);
        String previousBudget = System.getProperty(CpuRepack.PROP_BUDGET);
        try {
            System.setProperty(CpuRepack.PROP_REPACK, "on");
            System.setProperty(CpuRepack.PROP_BUDGET, Long.toString(budget));
            try (ModelSource source = ModelSources.open(path.toString(), ModelSourceOptions.defaults());
                 CpuWeights budgeted = CpuWeights.open(source, TensorCatalog.of(source), null,
                         TensorReadMode.MMAP)) {
                System.setProperty(CpuRepack.PROP_REPACK, "off");
                System.clearProperty(CpuRepack.PROP_BUDGET);
                try (ModelSource plainSource = ModelSources.open(path.toString(), ModelSourceOptions.defaults());
                     CpuWeights plain = CpuWeights.open(plainSource, TensorCatalog.of(plainSource), null,
                             TensorReadMode.MMAP)) {
                    Random rnd = new Random(0xB0D);
                    for (String name : MATMULS) {
                        CpuTensorRef weighted = budgeted.get(name);
                        CpuTensorRef unweighted = plain.get(name);
                        float[] x = new float[(int) weighted.cols()];
                        for (int i = 0; i < x.length; i++) {
                            x[i] = (float) rnd.nextGaussian();
                        }
                        float[] repackedOut = new float[(int) weighted.rows()];
                        float[] plainOut = new float[(int) unweighted.rows()];
                        budgeted.matVec().matvec(weighted, x, 0, repackedOut, 0);
                        plain.matVec().matvec(unweighted, x, 0, plainOut, 0);
                        assertArrayEquals("the product of " + name + " must not depend on the budget",
                                plainOut, repackedOut, 0f);
                    }
                    assertEquals("only the first 544-byte copy fits a 1000-byte budget", 1,
                            budgeted.repack().repackedCount());
                    assertEquals(544L, budgeted.repack().repackedBytes());
                    assertTrue(budgeted.repack().summary(),
                            budgeted.repack().summary().contains("over budget 2"));
                    assertTrue(budgeted.repack().summary(),
                            budgeted.repack().summary().contains("repacked 1 weight(s)"));
                    assertEquals("the unrepacked run copies nothing", 0, plain.repack().repackedCount());
                }
            }
        } finally {
            restore(CpuRepack.PROP_REPACK, previousMode);
            restore(CpuRepack.PROP_BUDGET, previousBudget);
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void aLookupTableIsKeptInPlaceAndNamedInTheSummary() throws IOException {
        Assume.assumeTrue(CpuSimd.available());
        Path path = model().writeTempFile();
        String previousMode = System.getProperty(CpuRepack.PROP_REPACK);
        String previousBudget = System.getProperty(CpuRepack.PROP_BUDGET);
        try {
            System.setProperty(CpuRepack.PROP_REPACK, "on");
            System.clearProperty(CpuRepack.PROP_BUDGET);
            try (ModelSource source = ModelSources.open(path.toString(), ModelSourceOptions.defaults());
                 CpuWeights weights = CpuWeights.open(source, TensorCatalog.of(source), null,
                         TensorReadMode.MMAP)) {
                CpuTensorRef table = weights.get("token_embd.weight");
                float[] row = new float[(int) table.cols()];
                weights.embedRow(table, 0L, row, 0);
                float[] x = new float[(int) table.cols()];
                float[] y = new float[(int) table.rows()];
                weights.matVec().matvec(table, x, 0, y, 0);
                assertEquals("a lookup table is never copied, even when a product also reads it", 0,
                        weights.repack().repackedCount());
                assertTrue(weights.repack().summary(),
                        weights.repack().summary().contains("lookup table 1"));
            }
        } finally {
            restore(CpuRepack.PROP_REPACK, previousMode);
            restore(CpuRepack.PROP_BUDGET, previousBudget);
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void theBudgetPropertyRejectsInvalidByteCountsByName() {
        String previous = System.getProperty(CpuRepack.PROP_BUDGET);
        try {
            for (String bad : new String[]{"12 MiB", "-1"}) {
                System.setProperty(CpuRepack.PROP_BUDGET, bad);
                try {
                    new CpuRepack();
                    fail("expected the rejection of " + CpuRepack.PROP_BUDGET + "=" + bad);
                } catch (IllegalArgumentException expected) {
                    assertTrue(expected.getMessage(), expected.getMessage().contains(CpuRepack.PROP_BUDGET));
                    assertTrue(expected.getMessage(), expected.getMessage().contains(bad));
                }
            }
            System.setProperty(CpuRepack.PROP_BUDGET, "4096");
            assertEquals(4096L, new CpuRepack().budgetBytes());
        } finally {
            restore(CpuRepack.PROP_BUDGET, previous);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
