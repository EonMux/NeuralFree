package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.IntPredicate;

/**
 * The decode-replay view of one of the two one-row attention patchers of a depth-routed plan
 * ({@link FlashPlan#routesDecodeByDepth}), whose one-row step runs either the MMA prefill launch or the decode
 * kernel by its live key count. A capture holds the kernels of exactly one route, so the patcher of the other
 * route claims nothing: it verifies vacuously while the step stays off its route. The patcher of the captured
 * route declines a step whose route left the capture (the step at the workgroup bound of keys), so the replay
 * runs that step's body and recaptures on the new route. Immutable; the claims live in the delegate.
 */
final class DepthRoutedPatcher implements GpuStepPatcher {
    private final GpuStepPatcher delegate;
    private final BooleanSupplier claimed;
    private final IntPredicate runsAt;

    /**
     * @param delegate the route's patcher
     * @param claimed  whether the delegate claimed a dispatch of the current capture
     * @param runsAt   whether the one-row step at a position runs this route's kernels
     */
    DepthRoutedPatcher(GpuStepPatcher delegate, BooleanSupplier claimed, IntPredicate runsAt) {
        this.delegate = delegate;
        this.claimed = claimed;
        this.runsAt = runsAt;
    }

    /**
     * The decode kernel's and the MMA prefill's one-row patchers of {@code attention}, each declining the steps
     * {@code prefillRoute} sends to the other; the decode patcher alone where the attention has no prefill launch.
     *
     * @param prefillRoute whether the one-row step at a position runs the MMA prefill launch
     */
    static List<GpuStepPatcher> of(GpuFlashAttention attention, IntPredicate prefillRoute) {
        FlashDecodePatcher decode = attention.patcher();
        FlashPrefillPatcher prefill = attention.prefillPatcher();
        DepthRoutedPatcher decodeRoute = new DepthRoutedPatcher(decode, decode::claimed, prefillRoute.negate());
        return prefill == null
                ? List.of(decodeRoute)
                : List.of(decodeRoute, new DepthRoutedPatcher(prefill, prefill::claimed, prefillRoute));
    }

    @Override
    public boolean accepts(GpuGraphDispatch d) {
        return delegate.accepts(d);
    }

    @Override
    public void resetClaims() {
        delegate.resetClaims();
    }

    @Override
    public boolean patch(int pos, int recordIndex, GpuGraphPatchSink sink) {
        return runsAt.test(pos) && delegate.patch(pos, recordIndex, sink);
    }

    @Override
    public boolean verify(int pos) {
        return claimed.getAsBoolean() ? runsAt.test(pos) && delegate.verify(pos) : !runsAt.test(pos);
    }
}
