package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The patched decode replay over the fake backend's updatable graphs: after the capture and two
 * verified body steps the trunk replays from the patch plan alone — the step's kernel arguments and
 * work size are patched into the captured dispatch and reach the emulated device without the body
 * running. The guard cases refuse a replay whose step leaves the capture: a position that does not
 * follow the previous one, a pattern the patcher declines, a patcher that misses the capture's values,
 * and a prediction that does not reproduce the captured arguments.
 */
public class GpuDecodeReplayPatchedTest {

    private static final String KERNEL = "TEST_STEP_KERNEL";

    @After
    public void tearDown() {
        System.clearProperty(GpuDecodeReplay.PROPERTY);
        System.clearProperty(GpuDecodeReplay.WARMUP_PROPERTY);
    }

    /** One decode step of the harness: sets the step's kernel argument and runs the trunk. */
    private void step(Harness h, int pos) {
        h.patcher.setStepValue(pos);
        h.replay.trunk(pos, List.of(h.patcher), true, () -> {
            h.bodyRuns++;
            h.backend.computeQueue().enqueueKernelNoEvent(h.kernel, new long[]{pos + 1}, new long[]{1},
                    new Object[]{pos});
        });
    }

    @Test
    public void aTrustedReplayPatchesTheStepValuesWithoutRunningTheBody() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, true)) {
            for (int pos = 0; pos < 8; pos++) {
                step(h, pos);
            }
            assertEquals("the capture, the two verified steps and no trusted step run the body", 3, h.bodyRuns);
            assertEquals("the capture is submitted once", 1, h.count(FakeOpKind.LAUNCH));
            assertEquals("every step after the capture reaches the device as one replay", 7, h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step's patched argument, including the no-body steps",
                    List.of(0, 1, 2, 3, 4, 5, 6, 7), h.executedArgs);
            assertEquals("the device saw every step's patched work size",
                    List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L), h.executedGlobals);
        }
    }

    @Test
    public void aPositionThatDoesNotFollowThePreviousOneNeverReplays() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            assertEquals(3, h.bodyRuns);

            long replaysBefore = h.count(FakeOpKind.REPLAY);
            long launchesBefore = h.count(FakeOpKind.LAUNCH);
            step(h, 99);
            assertEquals("the stale step runs the body", 4, h.bodyRuns);
            assertEquals("the stale graph is not replayed", replaysBefore, h.count(FakeOpKind.REPLAY));
            assertEquals("the stale step does not capture", launchesBefore, h.count(FakeOpKind.LAUNCH));

            step(h, 100);
            assertEquals("the next step captures again", 2, h.count(FakeOpKind.LAUNCH));
            step(h, 101);
            step(h, 102);
            step(h, 103);
            assertEquals("two verified steps and one trusted step of each capture", 6, h.count(FakeOpKind.REPLAY));
            assertEquals("the stale step and the recapture's steps run the body", 7, h.bodyRuns);
            assertEquals("the device saw every step's argument",
                    List.of(0, 1, 2, 3, 99, 100, 101, 102, 103), h.executedArgs);
        }
    }

    @Test
    public void aDeclinedPatternRunsTheBodyAndRecaptures() {
        try (Harness h = new Harness(false, false, 4, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            assertEquals(3, h.bodyRuns);

            long replaysBefore = h.count(FakeOpKind.REPLAY);
            long launchesBefore = h.count(FakeOpKind.LAUNCH);
            step(h, 4);
            assertEquals("the declined step runs the body", 4, h.bodyRuns);
            assertEquals("the capture is not replayed against the diverging pattern", replaysBefore,
                    h.count(FakeOpKind.REPLAY));
            assertEquals("the declined step does not capture", launchesBefore, h.count(FakeOpKind.LAUNCH));

            step(h, 5);
            step(h, 6);
            step(h, 7);
            step(h, 8);
            assertEquals("the recapture submits once", 2, h.count(FakeOpKind.LAUNCH));
            assertEquals("three replays of each capture's verified and trusted steps", 6, h.count(FakeOpKind.REPLAY));
            assertEquals("every step's value reached the device",
                    List.of(0, 1, 2, 3, 4, 5, 6, 7, 8), h.executedArgs);
        }
    }

    /** One two-row step of the harness: sets the step's kernel argument and runs the width-2 trunk. */
    private void stepWidth2(Harness h, int pos) {
        h.patcher.setStepValue(pos);
        h.replay.trunk(pos, 2, List.of(h.patcher), true, () -> {
            h.bodyRuns++;
            h.backend.computeQueue().enqueueKernelNoEvent(h.kernel, new long[]{pos + 1}, new long[]{1},
                    new Object[]{pos});
        });
    }

    @Test
    public void aWidthTwoTrunkPatchesContiguousStepsWithoutRunningTheBody() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, true)) {
            for (int pos = 0; pos <= 8; pos += 2) {
                stepWidth2(h, pos);
            }
            assertEquals("the capture and the two verified steps run the body", 3, h.bodyRuns);
            assertEquals("the capture is submitted once", 1, h.count(FakeOpKind.LAUNCH));
            assertEquals("the verified steps body-replay and the trusted steps replay patched", 4,
                    h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step's patched argument",
                    List.of(0, 2, 4, 6, 8), h.executedArgs);
            assertEquals("the device saw every step's patched work size",
                    List.of(1L, 3L, 5L, 7L, 9L), h.executedGlobals);
        }
    }

    @Test
    public void aWidthTwoTrunkBodyReplaysAGapAndKeepsThePlanForTheNextRun() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, true)) {
            for (int pos = 0; pos <= 6; pos += 2) {
                stepWidth2(h, pos);
            }
            long launchesBefore = h.count(FakeOpKind.LAUNCH);

            stepWidth2(h, 20);
            assertEquals("the gap's step runs the body through the capture's replay", 4, h.bodyRuns);
            assertEquals("the gap's step does not capture", launchesBefore, h.count(FakeOpKind.LAUNCH));

            stepWidth2(h, 22);
            assertEquals("the next run's steps follow the gap's step, so they trust the kept plan", 4,
                    h.bodyRuns);
            assertEquals("the kept plan still needs no capture", launchesBefore, h.count(FakeOpKind.LAUNCH));
            assertEquals("the device saw every step's argument, the gap's through the body",
                    List.of(0, 2, 4, 6, 20, 22), h.executedArgs);
        }
    }

    @Test
    public void aPatcherThatMissesTheCaptureValuesNeverEarnsTrust() {
        try (Harness h = new Harness(false, true, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 5; pos++) {
                step(h, pos);
            }
            assertEquals("two captures and their four verified steps ran the body, then the attempts stop", 6,
                    h.bodyRuns);
            assertEquals(2, h.count(FakeOpKind.LAUNCH));
            assertEquals(2, h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step through the body replay",
                    List.of(0, 1, 2, 3, 4, 5), h.executedArgs);
        }
    }

    @Test
    public void anInvalidationStartsANewChainTheGuardAccepts() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            h.replay.invalidate();

            step(h, 50);
            assertEquals("the invalidation made the step capture again", 2, h.count(FakeOpKind.LAUNCH));
            step(h, 51);
            step(h, 52);
            step(h, 53);
            assertEquals("three replays of each capture's verified and trusted steps", 6,
                    h.count(FakeOpKind.REPLAY));
            assertEquals("the new chain's positions follow each other, so the trusted step skips the body",
                    6, h.bodyRuns);
            assertEquals("the device saw the new chain's arguments",
                    List.of(0, 1, 2, 3, 50, 51, 52, 53), h.executedArgs);
        }
    }

    @Test
    public void aDistrustedCaptureIsVerifiedByTwoBodyStepsAndThenTrustedAgain() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, true)) {
            for (int pos = 0; pos <= 5; pos++) {
                step(h, pos);
            }
            assertEquals("the capture and its two verified steps run the body", 3, h.bodyRuns);
            h.replay.distrust();

            step(h, 0);
            step(h, 1);
            assertEquals("the two steps after the distrust verify the kept capture through the body", 5,
                    h.bodyRuns);
            step(h, 2);
            step(h, 3);
            assertEquals("the verified capture is trusted again: no body", 5, h.bodyRuns);
            assertEquals("the kept capture is never submitted again", 1, h.count(FakeOpKind.LAUNCH));
            assertEquals("every step after the capture reaches the device as one replay", 9,
                    h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step's argument and work size from the rewound position on",
                    List.of(0, 1, 2, 3, 4, 5, 0, 1, 2, 3), h.executedArgs);
            assertEquals(List.of(1L, 2L, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L), h.executedGlobals);
        }
    }

    @Test
    public void aDistrustedCaptureWhoseBodyLeavesItIsDroppedAndCapturedAgain() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            h.replay.distrust();
            KernelHandle other = h.backend.compileKernel(KERNEL + "_OTHER", "void " + KERNEL + "_OTHER(){}", null);
            h.replay.trunk(0, List.of(h.patcher), true, () -> {
                h.bodyRuns++;
                h.backend.computeQueue().enqueueKernelNoEvent(other, new long[]{1}, new long[]{1},
                        new Object[]{0});
            });
            assertEquals("the diverging body ran directly once its replay mismatched", 5, h.bodyRuns);
            assertEquals("the diverging step did not capture", 1, h.count(FakeOpKind.LAUNCH));

            step(h, 1);
            assertEquals("the dropped capture is recorded again from the next step", 2, h.count(FakeOpKind.LAUNCH));
            step(h, 2);
            step(h, 3);
            step(h, 4);
            assertEquals("the recapture verifies twice and then trusts", 8, h.bodyRuns);
            assertEquals(List.of(0, 1, 2, 3, 1, 2, 3, 4), h.executedArgs);
        }
    }

    @Test
    public void aStepThatIsNotCapturableDropsAKeptCapture() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            h.replay.distrust();
            h.patcher.setStepValue(0);
            h.replay.trunk(0, List.of(h.patcher), false, () -> {
                h.bodyRuns++;
                h.backend.computeQueue().enqueueKernelNoEvent(h.kernel, new long[]{1}, new long[]{1},
                        new Object[]{0});
            });
            assertEquals("the uncapturable step ran its body directly", 4, h.bodyRuns);
            long replays = h.count(FakeOpKind.REPLAY);
            step(h, 1);
            assertEquals("the kept capture was dropped: the next step captures again", 2,
                    h.count(FakeOpKind.LAUNCH));
            assertEquals("no replay of the dropped capture", replays, h.count(FakeOpKind.REPLAY));
        }
    }

    @Test
    public void aPredictionThatMissesTheCapturedArgumentsKeepsTheBodyReplay() {
        try (Harness h = new Harness(true, false, Integer.MIN_VALUE, false)) {
            for (int pos = 0; pos <= 7; pos++) {
                step(h, pos);
            }
            assertEquals("no plan was built, so every step runs the body", 8, h.bodyRuns);
            assertEquals("the capture is submitted once", 1, h.count(FakeOpKind.LAUNCH));
            assertEquals("every step after the capture body-replays", 7, h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step's argument", List.of(0, 1, 2, 3, 4, 5, 6, 7), h.executedArgs);
        }
    }

    @Test
    public void aGraphWithoutUpdatableArgumentsStaysWithTheBodyReplay() {
        try (Harness h = new Harness(false, false, Integer.MIN_VALUE, false, false)) {
            for (int pos = 0; pos <= 3; pos++) {
                step(h, pos);
            }
            assertEquals("a verify-only graph refuses the changing arguments, so the replays mismatch:", 6,
                    h.bodyRuns);
            assertEquals("the retries recapture once before the attempts stop", 2, h.count(FakeOpKind.LAUNCH));
            assertEquals("no replay reached the device", 0, h.count(FakeOpKind.REPLAY));
            assertEquals("the device saw every step through the direct bodies",
                    List.of(0, 1, 2, 3), h.executedArgs);
        }
    }

    private static final class Harness implements AutoCloseable {
        final FakeComputeBackend backend;
        final KernelHandle kernel;
        final TestPatcher patcher;
        final GpuDecodeReplay replay;
        final List<Integer> executedArgs = new ArrayList<>();
        final List<Long> executedGlobals = new ArrayList<>();
        int bodyRuns;

        Harness(boolean wrongPrediction, boolean refuseVerify, int declineAt, boolean patchGrid) {
            this(wrongPrediction, refuseVerify, declineAt, patchGrid, true);
        }

        Harness(boolean wrongPrediction, boolean refuseVerify, int declineAt, boolean patchGrid,
                boolean argUpdates) {
            System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "0");
            System.setProperty(GpuDecodeReplay.PROPERTY, "true");
            backend = FakeComputeBackend.builder()
                    .graphCapture(true)
                    .graphArgUpdates(argUpdates)
                    .emulate(KERNEL, d -> {
                        executedArgs.add(d.intArg(0));
                        executedGlobals.add(d.global()[0]);
                    })
                    .build();
            kernel = backend.compileKernel(KERNEL, "void " + KERNEL + "(){}", null);
            patcher = new TestPatcher(wrongPrediction, refuseVerify, declineAt, patchGrid);
            replay = new GpuDecodeReplay(backend.computeQueue());
        }

        long count(FakeOpKind kind) {
            return backend.ops().stream().filter(o -> o.kind() == kind).count();
        }

        @Override
        public void close() {
            replay.close();
            backend.close();
        }
    }

    /** A patcher of the harness kernel: one per-step scalar argument and, optionally, the work size. */
    private static final class TestPatcher implements GpuStepPatcher {
        private final boolean wrongPrediction;
        private final boolean refuseVerify;
        private final int declineAt;
        private final boolean patchGrid;
        private int claimed = -1;
        private int stepValue = -1;

        TestPatcher(boolean wrongPrediction, boolean refuseVerify, int declineAt, boolean patchGrid) {
            this.wrongPrediction = wrongPrediction;
            this.refuseVerify = refuseVerify;
            this.declineAt = declineAt;
            this.patchGrid = patchGrid;
        }

        void setStepValue(int value) {
            stepValue = value;
        }

        @Override
        public void resetClaims() {
            claimed = -1;
        }

        @Override
        public boolean accepts(GpuGraphDispatch d) {
            if (d.kernelName().equals(KERNEL) && claimed < 0) {
                claimed = d.index();
                return true;
            }
            return false;
        }

        @Override
        public boolean patch(int pos, int recordIndex, GpuGraphPatchSink sink) {
            if (pos == declineAt) {
                return false;
            }
            int value = wrongPrediction ? pos + 1000 : pos;
            sink.patchKernel(recordIndex, patchGrid ? new long[]{pos + 1} : null, new int[]{0},
                    new Object[]{value});
            return true;
        }

        @Override
        public boolean verify(int pos) {
            return !refuseVerify && stepValue == pos;
        }
    }
}
