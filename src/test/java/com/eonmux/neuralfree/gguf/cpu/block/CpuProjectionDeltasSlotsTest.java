package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The adapter slots of {@link CpuProjectionDeltas} (vLLM's punica wrapper over {@code token_lora_mapping}): every
 * slot's deltas — slot 0's, the startup adapters, included — cover only the rows the pass runs in that slot, run by
 * run; a uniform table applies one slot to a product over any rows, a mixed one requires the product to span the
 * table's rows, and a pass without a table runs the session slot.
 */
public final class CpuProjectionDeltasSlotsTest {

    private static final String SITE = "blk.0.attn_q.weight";
    private static final int WIDTH = 2;

    /** A delta adding {@code tag} to every output element of its rows. */
    private static CpuProjectionDeltas.Delta tag(float tag) {
        return (scale, n, x, xOff, xStride, y, yOff, yStride) -> {
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < WIDTH; c++) {
                    y[yOff + t * yStride + c] += scale * tag;
                }
            }
        };
    }

    private static LayerSpec layer0() {
        return LayerSpec.builder(0, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .build();
    }

    private static CpuProjectionDeltas threeSlots() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        deltas.serveModel(SITE, WIDTH, WIDTH, null);
        deltas.register(SITE, tag(1f));
        deltas.register(SITE, 1, tag(10f));
        deltas.register(SITE, 2, tag(100f));
        return deltas;
    }

    private static float[] applied(CpuProjectionDeltas deltas, int n) {
        float[] x = new float[n * WIDTH];
        float[] y = new float[n * WIDTH];
        ((CpuProjectionDeltas.Site) deltas.served(SITE)).applyRows(n, x, 0, y, 0);
        return y;
    }

    private static float[] rowsOf(float... perRow) {
        float[] out = new float[perRow.length * WIDTH];
        for (int r = 0; r < perRow.length; r++) {
            Arrays.fill(out, r * WIDTH, (r + 1) * WIDTH, perRow[r]);
        }
        return out;
    }

    @Test
    public void aMixedTableAppliesEachSlotsDeltasToItsRowsAlone() {
        CpuProjectionDeltas deltas = threeSlots();
        deltas.rows().set(new int[]{1, 1, 0, 2, 2, 1});
        assertArrayEquals(rowsOf(10, 10, 1, 100, 100, 10), applied(deltas, 6), 0f);
    }

    @Test
    public void aPassWithoutATableRunsTheSessionSlot() {
        CpuProjectionDeltas deltas = threeSlots();
        deltas.rows().session(2);
        assertArrayEquals(rowsOf(100, 100), applied(deltas, 2), 0f);
        deltas.rows().set(new int[]{1, 0});
        assertArrayEquals("a table overrides the session", rowsOf(10, 1), applied(deltas, 2), 0f);
        deltas.rows().clear();
        assertArrayEquals("clearing the table returns to the session", rowsOf(100), applied(deltas, 1), 0f);
        deltas.rows().session(0);
        assertArrayEquals(rowsOf(1), applied(deltas, 1), 0f);
    }

    @Test
    public void aSlotWithoutDeltasRunsTheBareProduct() {
        CpuProjectionDeltas deltas = threeSlots();
        deltas.openSlot(5);
        deltas.rows().set(new int[]{5, 0, 5});
        assertArrayEquals(rowsOf(0, 1, 0), applied(deltas, 3), 0f);
    }

    @Test
    public void anInactiveTableRunsTheSlotZeroDeltasAlone() {
        CpuProjectionDeltas deltas = threeSlots();
        assertArrayEquals(rowsOf(1, 1, 1), applied(deltas, 3), 0f);
        deltas.rows().set(new int[]{0, 0, 0});
        assertFalse("an all-zero table is no table", deltas.rows().active());
        assertArrayEquals(rowsOf(1, 1, 1), applied(deltas, 3), 0f);
        deltas.rows().set(new int[]{2, 1});
        deltas.rows().clear();
        assertArrayEquals(rowsOf(1, 1), applied(deltas, 2), 0f);
    }

    @Test
    public void aUniformTableAppliesItsSlotToAProductOverAnyRows() {
        CpuProjectionDeltas deltas = threeSlots();
        deltas.rows().set(new int[]{2, 2, 2, 2});
        // the logits head runs over the output rows only: one row of the four
        assertArrayEquals(rowsOf(100), applied(deltas, 1), 0f);
    }

    @Test
    public void aMixedTableRefusesAProductOverOtherRowsNamingTheProduct() {
        CpuProjectionDeltas deltas = threeSlots();
        deltas.rows().set(new int[]{1, 2, 0});
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> applied(deltas, 2));
        assertTrue(e.getMessage(), e.getMessage().contains(SITE));
    }

    @Test
    public void slotDeltasAloneMakeASiteNonEmptyAndOpenSlotsAreHeld() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        CpuProjectionDeltas.Site site = deltas.serveModel(SITE, WIDTH, WIDTH, null);
        CpuProjectionDeltas.Delta d = tag(5f);
        deltas.openSlot(3);
        deltas.register(SITE, 3, d);
        assertFalse(site.isEmpty());
        assertTrue(deltas.holdsSlot(3));
        assertFalse(deltas.holdsSlot(1));

        deltas.rows().set(new int[]{3, 0});
        assertArrayEquals(rowsOf(5, 0), applied(deltas, 2), 0f);

        deltas.unregister(SITE, d);
        assertTrue(site.isEmpty());
        assertTrue("the slot stays open without deltas", deltas.holdsSlot(3));
        deltas.closeSlot(3);
        assertFalse(deltas.holdsSlot(3));
        assertThrows(IllegalArgumentException.class, () -> deltas.register(SITE, -1, tag(1f)));
        assertThrows(IllegalArgumentException.class, () -> deltas.openSlot(0));
    }

    @Test
    public void expertDeltasOfAMixedTableRunOverTheirTokensPairs() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        // four experts of {3 -> 5}: the tensor holds 4 * 5 rows of 3 columns
        ByteBuffer buf = ByteBuffer.allocate(4 * 5 * 3 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        CpuProjectionDeltas.ExpertSite site = deltas.serveExperts(
                layer0(), "ffn_up_exps.weight",
                CpuTensorRef.of(GGMLType.F32, buf, 4 * 5, 3), 4, null);
        assertEquals(3, site.inDim());
        assertEquals(5, site.outDim());
        List<String> calls = new ArrayList<>();
        String name = CpuProjectionDeltas.key(layer0(),
                "ffn_up_exps.weight");
        deltas.registerExperts(name, (scale, pairs, k, ids, x, xOff, tokenRows, y, yOff) ->
                calls.add("base pairs=" + pairs + " ids=" + Arrays.toString(ids)));
        deltas.registerExperts(name, 1, (scale, pairs, k, ids, x, xOff, tokenRows, y, yOff) ->
                calls.add("slot1 pairs=" + pairs + " ids=" + Arrays.toString(ids) + " x=" + xOff + " y=" + yOff));

        // three tokens of two experts each; token 1 runs in slot 1
        deltas.rows().set(new int[]{0, 1, 0});
        int[] ids = {0, 1, 2, 3, 1, 0};
        site.apply(6, 2, ids, new float[64], 0, true, new float[64], 0);
        assertEquals(Arrays.asList("base pairs=2 ids=[0, 1]",
                "slot1 pairs=2 ids=[2, 3] x=" + site.inDim() + " y=" + 2 * site.outDim(),
                "base pairs=2 ids=[1, 0]"), calls);
    }

    @Test
    public void embeddingDeltasFollowTheRowsSlotsInBatchAndPerRow() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        CpuProjectionDeltas.EmbeddingSite site = deltas.serveEmbedding("token_embd.weight", 8, WIDTH, null);
        deltas.registerEmbedding(site.name(), (scale, n, ids, idsOff, y, yOff, yStride) -> {
            for (int t = 0; t < n; t++) {
                y[yOff + t * yStride] += scale * ids[idsOff + t];
            }
        });
        deltas.registerEmbedding(site.name(), 2, (scale, n, ids, idsOff, y, yOff, yStride) -> {
            for (int t = 0; t < n; t++) {
                y[yOff + t * yStride + 1] += scale * 100 * ids[idsOff + t];
            }
        });
        deltas.rows().set(new int[]{0, 2, 2});
        int[] ids = {9, 5, 6, 7};
        float[] y = new float[3 * WIDTH];
        site.apply(3, ids, 1, y, 0, WIDTH);
        assertArrayEquals(new float[]{5, 0, 0, 600, 0, 700}, y, 0f);

        float[] one = new float[WIDTH];
        site.applyRow(1, ids, 2, one, 0);
        assertArrayEquals(new float[]{0, 600}, one, 0f);
        site.applyRow(0, ids, 2, one, 0);
        assertArrayEquals(new float[]{6, 600}, one, 0f);
    }

    @Test
    public void theRowsTableReportsItsRunsAndGathersUbatchRows() {
        AdapterRows rows = new AdapterRows();
        rows.set(new int[]{1, 1, 0, 0, 0, 2});
        assertEquals(-1, rows.uniform());
        assertEquals(6, rows.rows());
        assertEquals(2, rows.runEnd(0, 6));
        assertEquals(5, rows.runEnd(2, 6));
        assertEquals(6, rows.runEnd(5, 6));
        assertEquals(2, rows.slot(5));
        assertThrows(IndexOutOfBoundsException.class, () -> rows.slot(6));
        assertThrows(IllegalArgumentException.class, () -> rows.set(new int[]{1, -1}));
        assertEquals(2, rows.slot(5)); // a refused table changes nothing

        rows.set(new int[]{4, 4});
        assertEquals(4, rows.uniform());
        assertEquals(4, rows.slot(17));
        assertEquals(9, rows.runEnd(0, 9));
        rows.checkSpan("any", 9);

        assertArrayEquals(new int[]{7, 5, 5}, AdapterRows.ofRows(new int[]{5, 6, 7}, new int[]{2, 0, 0}));
        assertEquals(null, AdapterRows.ofRows(null, new int[]{0}));
    }
}
