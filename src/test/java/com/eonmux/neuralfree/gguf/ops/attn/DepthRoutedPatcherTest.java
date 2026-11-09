package com.eonmux.neuralfree.gguf.ops.attn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import java.util.function.IntPredicate;
import org.junit.Test;

/** The decode-replay contract of the two one-row patchers of a depth-routed flash plan. */
public class DepthRoutedPatcherTest {
    /** The route bound of the tests: steps at positions below it run the MMA prefill. */
    private static final int BOUND = 128;
    private static final IntPredicate PREFILL_ROUTE = pos -> pos + 1 <= BOUND;

    /** A patcher that records its calls and answers patch and verify with true. */
    private static final class Recording implements GpuStepPatcher {
        boolean claimed;
        int patches;
        int verifies;
        int resets;

        @Override
        public boolean accepts(GpuGraphDispatch d) {
            return claimed;
        }

        @Override
        public void resetClaims() {
            resets++;
        }

        @Override
        public boolean patch(int pos, int recordIndex, GpuGraphPatchSink sink) {
            patches++;
            return true;
        }

        @Override
        public boolean verify(int pos) {
            verifies++;
            return true;
        }
    }

    @Test
    public void theCapturedRoutesPatcherPatchesItsStepsAndDeclinesTheFlip() {
        Recording prefill = new Recording();
        prefill.claimed = true;
        DepthRoutedPatcher routed = new DepthRoutedPatcher(prefill, () -> prefill.claimed, PREFILL_ROUTE);
        assertTrue(routed.patch(BOUND - 1, 0, null));
        assertFalse("the step past the bound leaves the prefill route", routed.patch(BOUND, 0, null));
        assertEquals("a declined step reaches no delegate patch", 1, prefill.patches);
        assertTrue(routed.verify(BOUND - 1));
        assertFalse(routed.verify(BOUND));
    }

    @Test
    public void theOtherRoutesPatcherVerifiesVacuouslyWhileTheStepStaysOffItsRoute() {
        Recording decode = new Recording();
        DepthRoutedPatcher routed = new DepthRoutedPatcher(decode, () -> decode.claimed, PREFILL_ROUTE.negate());
        assertTrue("a prefill-route step holds no decode kernel to verify", routed.verify(BOUND - 1));
        assertFalse("a decode-route step the capture holds no decode kernel for", routed.verify(BOUND));
        assertEquals("an unclaimed delegate is never asked to verify", 0, decode.verifies);
    }

    @Test
    public void claimsAndResetsReachTheDelegate() {
        Recording decode = new Recording();
        decode.claimed = true;
        DepthRoutedPatcher routed = new DepthRoutedPatcher(decode, () -> decode.claimed, PREFILL_ROUTE.negate());
        assertTrue(routed.accepts(null));
        routed.resetClaims();
        assertEquals(1, decode.resets);
        assertTrue(routed.verify(BOUND));
        assertEquals(1, decode.verifies);
    }
}
