package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The quantizer-side lattices against llama.cpp's {@code iq2xs_init_impl} / {@code iq3xs_init_impl}: the
 * level vectors recovered from the dequantization grids equal the reference {@code kgrid_*} tables, and
 * the point map with the neighbour lists hashes to the reference tables over every vector an encoder can
 * form. The reference figures come from ggml-quants.c built as is and walked over the same vectors.
 */
public class IqGridTest {

    /** Sum and {@code h = 31 h + key} hash of the packed level vectors, in grid order. */
    private static void assertKgrid(GGMLType type, long sum, long hash) {
        IqGrid grid = IqGrid.of(type);
        long s = 0;
        int h = 0;
        for (int g = 0; g < grid.size(); g++) {
            s += grid.key(g);
            h = h * 31 + grid.key(g);
            if (g > 0) {
                assertTrue(type + " keys ascend like the reference table", grid.key(g) > grid.key(g - 1));
            }
        }
        assertEquals(type + " kgrid sum", sum, s);
        assertEquals(type + " kgrid hash", hash, Integer.toUnsignedLong(h));
    }

    @Test
    public void levelsReproduceTheReferenceKgridTables() {
        assertKgrid(GGMLType.IQ2_XXS, 3697703L, 3568743679L);  // kgrid_2bit_256
        assertKgrid(GGMLType.IQ2_XS, 8004683L, 392942767L);    // kgrid_2bit_512
        assertKgrid(GGMLType.IQ2_S, 16519059L, 1737912421L);   // kgrid_2bit_1024
        assertKgrid(GGMLType.IQ1_S, 44428132L, 924341650L);    // kgrid_1bit_2048
        assertKgrid(GGMLType.IQ3_XXS, 391138L, 1962385806L);   // kgrid_256
        assertKgrid(GGMLType.IQ3_S, 777331L, 2909333787L);     // kgrid_512
        assertEquals(0, IqGrid.of(GGMLType.IQ2_XXS).key(0));
        assertEquals(43268, IqGrid.of(GGMLType.IQ2_XXS).key(255));
        assertEquals(4032, IqGrid.of(GGMLType.IQ3_S).key(511));
    }

    /**
     * The reference walk: over every vector with lanes in [0, maxLevel], an on-grid vector mixes its grid
     * index, an off-grid one mixes 0xFFFF, its list length and its entries.
     */
    private static void assertNeighbours(GGMLType type, int maxLevel, long hash, long count) {
        IqGrid grid = IqGrid.of(type);
        int bits = grid.bits();
        int lanes = grid.lanes();
        int h = 0;
        long n = 0;
        int limit = 1 << (bits * lanes);
        for (int u = 0; u < limit; u++) {
            boolean reachable = true;
            for (int j = 0; j < lanes; j++) {
                reachable &= ((u >>> (bits * j)) & ((1 << bits) - 1)) <= maxLevel;
            }
            if (!reachable) {
                continue;
            }
            if (grid.indexOf(u) >= 0) {
                h = h * 31 + grid.indexOf(u);
                continue;
            }
            int[] list = grid.neighbours(u);
            h = h * 31 + 0xFFFF;
            h = h * 31 + list.length;
            n += list.length;
            for (int g : list) {
                h = h * 31 + g;
            }
        }
        assertEquals(type + " neighbour hash", hash, Integer.toUnsignedLong(h));
        assertEquals(type + " neighbour entries", count, n);
    }

    @Test
    public void neighbourListsMatchTheReference() {
        assertNeighbours(GGMLType.IQ2_XXS, 2, 1845234394L, 113157L);
        assertNeighbours(GGMLType.IQ2_XS, 2, 3890916458L, 181721L);
        assertNeighbours(GGMLType.IQ2_S, 2, 680209465L, 44716L);
        assertNeighbours(GGMLType.IQ1_M, 2, 643584522L, 327721L);
        assertNeighbours(GGMLType.IQ3_XXS, 7, 3621848050L, 18985L);
        assertNeighbours(GGMLType.IQ3_S, 7, 1540519711L, 24733L);
    }

    @Test
    public void onGridVectorsHaveNoNeighbourList() {
        IqGrid grid = IqGrid.of(GGMLType.IQ2_XXS);
        try {
            grid.neighbours(grid.key(3));
            fail("an on-grid vector has no neighbour list");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no neighbour list"));
        }
        try {
            IqGrid.of(GGMLType.Q4_K);
            fail("Q4_K has no grid");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Q4_K"));
        }
    }
}
