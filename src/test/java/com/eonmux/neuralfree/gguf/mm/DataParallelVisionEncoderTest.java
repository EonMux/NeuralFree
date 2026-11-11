package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraFixtures;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DataParallelVisionEncoder}, the data-parallel side of vLLM's {@code mm_encoder_tp_mode}: a batch is
 * assigned over the ranks by pixel count the way vLLM's {@code get_load_balance_assignment} does, every rank
 * encodes its share on its own thread, the results come back in input order and equal the sequential encoder's, a
 * failing rank fails the batch, and closing the encoder closes every rank.
 */
public class DataParallelVisionEncoderTest {

    private static final int WIDTH = 16;

    /** Which images one rank encoded, and whether it was closed. */
    private static final class Tracking implements VisionEncoder {
        private final VisionEncoder delegate;
        private final String name;
        private final Map<ClipImage, Boolean> failures = new IdentityHashMap<>();
        final List<ClipImage> encoded = Collections.synchronizedList(new ArrayList<>());
        /** The calls running on this rank now, and the most that ever ran at once. */
        final AtomicInteger active = new AtomicInteger();
        final AtomicInteger mostActive = new AtomicInteger();
        volatile boolean closed;

        Tracking(VisionEncoder delegate, String name) {
            this.delegate = delegate;
            this.name = name;
        }

        void failOn(ClipImage image) {
            failures.put(image, Boolean.TRUE);
        }

        @Override
        public float[][] encode(ClipImage image) {
            return counted(() -> {
                encoded.add(image);
                if (failures.containsKey(image)) {
                    throw new IllegalStateException(name + " cannot encode its share");
                }
                return delegate.encode(image);
            });
        }

        @Override
        public float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY,
                                      int origWidth, int origHeight) {
            return counted(() -> delegate.encodeAnyres(overview, tiles, gridX, gridY, origWidth, origHeight));
        }

        /** Runs {@code call} counted as active, held a moment so overlapping callers would meet. */
        private float[][] counted(Supplier<float[][]> call) {
            mostActive.accumulateAndGet(active.incrementAndGet(), Math::max);
            try {
                Thread.sleep(1);
                return call.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(name + " was interrupted", e);
            } finally {
                active.decrementAndGet();
            }
        }

        @Override
        public LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings) {
            return delegate.attachLoras(bindings);
        }

        @Override
        public int openLoraSession(List<LoraAttachment.Binding> bindings) {
            return delegate.openLoraSession(bindings);
        }

        @Override
        public void closeLoraSession(int slot) {
            delegate.closeLoraSession(slot);
        }

        @Override
        public ClipVisionModel model() {
            return delegate.model();
        }

        @Override
        public String device() {
            return name;
        }

        @Override
        public void close() {
            closed = true;
            delegate.close();
        }
    }

    /** A rank whose encode waits for the test's gate and ignores interrupts meanwhile, as a device encode does. */
    private static final class Gated implements VisionEncoder {
        private final VisionEncoder delegate;
        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch gate = new CountDownLatch(1);
        final AtomicBoolean running = new AtomicBoolean();
        volatile boolean closedWhileRunning;

        Gated(VisionEncoder delegate) {
            this.delegate = delegate;
        }

        @Override
        public float[][] encode(ClipImage image) {
            running.set(true);
            entered.countDown();
            try {
                boolean interrupted = false;
                while (true) {
                    try {
                        gate.await();
                        break;
                    } catch (InterruptedException e) {
                        interrupted = true;
                    }
                }
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                return delegate.encode(image);
            } finally {
                running.set(false);
            }
        }

        @Override
        public ClipVisionModel model() {
            return delegate.model();
        }

        @Override
        public LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings) {
            return delegate.attachLoras(bindings);
        }

        @Override
        public int openLoraSession(List<LoraAttachment.Binding> bindings) {
            return delegate.openLoraSession(bindings);
        }

        @Override
        public void closeLoraSession(int slot) {
            delegate.closeLoraSession(slot);
        }

        @Override
        public String device() {
            return "gated";
        }

        @Override
        public void close() {
            if (running.get()) {
                closedWhileRunning = true;
            }
            delegate.close();
        }
    }

    private static ClipImage image(int seed) {
        return ClipImage.of(8, 8, MmprojFixture.pixels(8, 8, seed));
    }

    /** Row-by-row equality of two encodes. */
    private static void assertRows(String what, float[][] expected, float[][] actual) {
        assertRows(what, expected, actual, 0.0f);
    }

    private static void assertRows(String what, float[][] expected, float[][] actual, float delta) {
        assertEquals(what, expected.length, actual.length);
        for (int r = 0; r < expected.length; r++) {
            assertArrayEquals(what + " row " + r, expected[r], actual[r], delta);
        }
    }

    @Test
    public void theBatchAssignmentIsVllmsGreedyPixelBalance() {
        // the example of vLLM's get_load_balance_assignment: [1000, 100, 200, 50] over two GPUs
        assertArrayEquals(new int[]{0, 1, 1, 1}, DataParallelVisionEncoder.assignment(new int[]{1000, 100, 200, 50}, 2));
        assertArrayEquals(new int[]{0}, DataParallelVisionEncoder.assignment(new int[]{7}, 3));
        assertArrayEquals(new int[]{0, 1, 2}, DataParallelVisionEncoder.assignment(new int[]{1, 1, 1}, 3));
        // equal sizes keep input order among the equally loaded ranks
        assertArrayEquals(new int[]{0, 1, 0, 1}, DataParallelVisionEncoder.assignment(new int[]{4, 4, 4, 4}, 2));
        // the largest image first: a small one joins the free rank after it
        assertArrayEquals(new int[]{0, 1, 1}, DataParallelVisionEncoder.assignment(new int[]{10, 1, 1}, 2));
    }

    @Test
    public void aBatchIsSpreadOverTheRanksAndComesBackInInputOrder() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Tracking a = new Tracking(mm.openCpuVisionEncoder(), "rank-a");
            Tracking b = new Tracking(mm.openCpuVisionEncoder(), "rank-b");
            List<ClipImage> images = Arrays.asList(image(1), image(2), image(3));
            try (DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(Arrays.asList(a, b))) {
                List<float[][]> rows = dp.encodeBatch(images);
                assertEquals(2, dp.ranks());
                assertEquals(3, rows.size());
                for (int i = 0; i < images.size(); i++) {
                    // each result is the rows of its own image, in input order
                    assertRows("image " + i, a.delegate.encode(images.get(i)), rows.get(i));
                }
                Map<ClipImage, Integer> rankOf = new IdentityHashMap<>();
                for (ClipImage encoded : a.encoded) {
                    rankOf.put(encoded, 0);
                }
                for (ClipImage encoded : b.encoded) {
                    rankOf.put(encoded, 1);
                }
                assertEquals("every image ran on exactly one rank", 3, rankOf.size());
                assertTrue("both ranks took part", !a.encoded.isEmpty() && !b.encoded.isEmpty());
                assertEquals("rank-a+rank-b", dp.device());
            }
            assertTrue(a.closed);
            assertTrue(b.closed);
        }
    }

    @Test
    public void dataParallelRowsEqualASequentialEncode() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            CpuVisionEncoder single = mm.openCpuVisionEncoder();
            List<ClipImage> images = Arrays.asList(image(4), image(5), image(6), image(7));
            List<float[][]> sequential = single.encodeBatch(images);
            try (DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(Arrays.asList(
                    mm.openCpuVisionEncoder(), mm.openCpuVisionEncoder(), mm.openCpuVisionEncoder()))) {
                List<float[][]> parallel = dp.encodeBatch(images);
                assertEquals(sequential.size(), parallel.size());
                for (int i = 0; i < sequential.size(); i++) {
                    assertRows("image " + i, sequential.get(i), parallel.get(i));
                }
                assertEquals(new ArrayList<>(), dp.encodeBatch(new ArrayList<ClipImage>()));
            }
            single.close();
        }
    }

    @Test
    public void oneImageGoesToTheNextRankInTurn() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Tracking a = new Tracking(mm.openCpuVisionEncoder(), "rank-a");
            Tracking b = new Tracking(mm.openCpuVisionEncoder(), "rank-b");
            try (DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(Arrays.asList(a, b))) {
                ClipImage first = image(8);
                ClipImage second = image(9);
                dp.encode(first);
                dp.encode(second);
                assertSame(first, a.encoded.get(0));
                assertSame(second, b.encoded.get(0));
            }
        }
    }

    /**
     * Callers on several threads encode through one data-parallel encoder at once, and no rank ever runs two calls
     * at a time: a one-rank batch and an anyres encode run on the rank's thread like every other call.
     */
    @Test
    public void concurrentCallersNeverOverlapOnARank() throws Exception {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Tracking only = new Tracking(mm.openCpuVisionEncoder(), "rank-only");
            Tracking a = new Tracking(mm.openCpuVisionEncoder(), "rank-a");
            Tracking b = new Tracking(mm.openCpuVisionEncoder(), "rank-b");
            ExecutorService callers = Executors.newFixedThreadPool(4);
            try (DataParallelVisionEncoder single = DataParallelVisionEncoder.of(List.of(only));
                 DataParallelVisionEncoder pair = DataParallelVisionEncoder.of(Arrays.asList(a, b))) {
                assertTrue(single.concurrent());
                List<Future<Object>> calls = new ArrayList<>();
                for (int t = 0; t < 4; t++) {
                    int seed = t;
                    calls.add(callers.submit(() -> {
                        for (int i = 0; i < 6; i++) {
                            assertEquals(2, single.encodeBatch(Arrays.asList(image(seed), image(seed + 4))).size());
                            pair.encode(image(seed + i));
                            assertEquals(3, pair.encodeBatch(Arrays.asList(image(seed), image(seed + 1),
                                    image(seed + 2))).size());
                            assertThrows(IllegalArgumentException.class,
                                    () -> single.encodeAnyres(image(seed), List.of(), 1, 1, 8, 8));
                        }
                        return null;
                    }));
                }
                for (Future<Object> call : calls) {
                    call.get(120, TimeUnit.SECONDS);
                }
            } finally {
                callers.shutdownNow();
            }
            assertEquals("the one rank ran one call at a time", 1, only.mostActive.get());
            assertEquals("rank a ran one call at a time", 1, a.mostActive.get());
            assertEquals("rank b ran one call at a time", 1, b.mostActive.get());
        }
    }

    /**
     * A caller interrupted while its call runs on a rank returns only once that call ended, on a one-rank encoder
     * and on a batch spread over two ranks: the media lock it holds is never released under a running encode.
     */
    @Test
    public void anInterruptedCallerReturnsOnlyOnceItsRankCallEnded() throws Exception {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            for (int rankCount = 1; rankCount <= 2; rankCount++) {
                Gated gated = new Gated(mm.openCpuVisionEncoder());
                List<VisionEncoder> ranks = new ArrayList<>(List.of(gated));
                if (rankCount == 2) {
                    ranks.add(mm.openCpuVisionEncoder());
                }
                try (DataParallelVisionEncoder encoder = DataParallelVisionEncoder.of(ranks)) {
                    AtomicReference<Throwable> outcome = new AtomicReference<>();
                    AtomicBoolean rankRanAtReturn = new AtomicBoolean();
                    Thread caller = new Thread(() -> {
                        try {
                            encoder.encodeBatch(Arrays.asList(image(1), image(2)).subList(0, ranks.size()));
                        } catch (RuntimeException e) {
                            outcome.set(e);
                        }
                        rankRanAtReturn.set(gated.running.get());
                    });
                    caller.start();
                    assertTrue("the rank started the call", gated.entered.await(30, TimeUnit.SECONDS));
                    caller.interrupt();
                    caller.join(300);
                    assertTrue(rankCount + " rank(s): the interrupted caller waits for its rank's call",
                            caller.isAlive());
                    gated.gate.countDown();
                    caller.join(30_000);
                    assertFalse(caller.isAlive());
                    assertFalse("no call ran when the caller returned", rankRanAtReturn.get());
                    assertTrue("the caller learns it was interrupted: " + outcome.get(),
                            outcome.get() instanceof IllegalStateException);
                }
            }
        }
    }

    /** Closing the encoder while a rank encodes waits for the encode before it closes the rank. */
    @Test
    public void closeWaitsForARunningRankCallBeforeClosingTheRank() throws Exception {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Gated gated = new Gated(mm.openCpuVisionEncoder());
            DataParallelVisionEncoder encoder = DataParallelVisionEncoder.of(List.of(gated));
            Thread caller = new Thread(() -> {
                try {
                    encoder.encode(image(3));
                } catch (RuntimeException ignored) {
                    // the close may fail the call; the test watches the rank
                }
            });
            caller.start();
            assertTrue("the rank started the call", gated.entered.await(30, TimeUnit.SECONDS));
            Thread closer = new Thread(encoder::close);
            closer.start();
            closer.join(300);
            assertTrue("the close waits for the running call", closer.isAlive());
            gated.gate.countDown();
            closer.join(30_000);
            caller.join(30_000);
            assertFalse(closer.isAlive());
            assertFalse("the rank was never closed under its encode", gated.closedWhileRunning);
        }
    }

    @Test
    public void aFailingRankFailsTheBatchAndTheOthersFinish() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Tracking a = new Tracking(mm.openCpuVisionEncoder(), "rank-a");
            Tracking b = new Tracking(mm.openCpuVisionEncoder(), "rank-b");
            ClipImage broken = image(10);
            b.failOn(broken);
            // a batch of one per rank, so the failing rank is the second
            List<ClipImage> images = Arrays.asList(image(11), broken);
            try (DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(Arrays.asList(a, b))) {
                IllegalStateException e = assertThrows(IllegalStateException.class, () -> dp.encodeBatch(images));
                assertTrue(e.getMessage(), e.getMessage().contains("rank-b"));
                assertEquals("the other rank finished its share", 1, a.encoded.size());
            }
        }
    }

    @Test
    public void aLoraSessionOpensOnEveryRankAndClosesOnAll() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            Tracking a = new Tracking(mm.openCpuVisionEncoder(), "rank-a");
            Tracking b = new Tracking(mm.openCpuVisionEncoder(), "rank-b");
            LoraAdapter adapter = LoraFixtures.adapter("s.gguf", 8.0f,
                    LoraFixtures.visionProjection("v.blk.0.attn_qkv.weight", 8, 24, 2, 40),
                    LoraFixtures.visionProjection("mm.0.weight", 8, WIDTH, 2, 41));
            List<ClipImage> images = Arrays.asList(image(12), image(13));
            try (DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(Arrays.asList(a, b))) {
                assertEquals(VisionEncoder.LORA_SESSION_SLOT, dp.openLoraSession(
                        List.of(new LoraAttachment.Binding(adapter, 1.0f))));
                // the encoder serves one caller's session: a second open refuses while the first holds
                IllegalStateException again = assertThrows(IllegalStateException.class,
                        () -> dp.openLoraSession(List.of()));
                assertTrue(again.getMessage(), again.getMessage().contains("session"));
                assertEquals(images.size(), dp.encodeBatch(images).size());
                dp.closeLoraSession(VisionEncoder.LORA_SESSION_SLOT);
                // the closed session's slot is free for the next one
                assertEquals(VisionEncoder.LORA_SESSION_SLOT, dp.openLoraSession(List.of()));
                IllegalStateException wrongSlot = assertThrows(IllegalStateException.class,
                        () -> dp.closeLoraSession(VisionEncoder.LORA_SESSION_SLOT + 1));
                assertTrue(wrongSlot.getMessage(), wrongSlot.getMessage().contains("slot"));
                dp.closeLoraSession(VisionEncoder.LORA_SESSION_SLOT);
            }
        }
    }

    @Test
    public void ranksOfDifferentModelsAreRefused() throws IOException {
        Path narrow = MmprojTestFiles.llava(WIDTH);
        Path wide = MmprojTestFiles.llava(WIDTH * 2);
        try (MmprojModel one = MmprojModel.open(narrow.toString());
             MmprojModel other = MmprojModel.open(wide.toString())) {
            CpuVisionEncoder a = one.openCpuVisionEncoder();
            CpuVisionEncoder b = other.openCpuVisionEncoder();
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> DataParallelVisionEncoder.of(Arrays.asList(a, b)));
            assertTrue(e.getMessage(), e.getMessage().contains("one model on every rank"));
            a.close();
            b.close();
        }
        assertThrows(IllegalArgumentException.class,
                () -> DataParallelVisionEncoder.of(new ArrayList<VisionEncoder>()));
    }

    @Test
    public void aClosedEncoderRefusesCalls() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            DataParallelVisionEncoder dp = DataParallelVisionEncoder.of(
                    Arrays.asList(mm.openCpuVisionEncoder(), mm.openCpuVisionEncoder()));
            dp.close();
            dp.close();
            assertThrows(IllegalStateException.class, () -> dp.encodeBatch(Arrays.asList(image(12))));
            assertThrows(IllegalStateException.class, () -> dp.encode(image(13)));
        }
    }

    @Test
    public void theModelBindsOneWholeEncoderPerRankDevice() throws IOException {
        Assume.assumeTrue("no OpenCL device running the LLM kernels", GpuTestUtil.isOpenCLAvailable());
        Path file = MmprojTestFiles.llava(WIDTH);
        // one physical device twice: '0' is its enumeration index, 'opencl' its name prefix
        MmprojOptions options = MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA)
                .withDevices(Arrays.asList("0", "opencl"));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            VisionEncoder encoder = mm.openVisionEncoder(options);
            Assume.assumeTrue("the OpenCL device cannot run this encoder", encoder instanceof DataParallelVisionEncoder);
            try {
                DataParallelVisionEncoder dp = (DataParallelVisionEncoder) encoder;
                assertEquals(2, dp.ranks());
                List<ClipImage> images = Arrays.asList(image(14), image(15));
                List<float[][]> parallel = dp.encodeBatch(images);
                CpuVisionEncoder cpu = mm.openCpuVisionEncoder();
                try {
                    for (int i = 0; i < images.size(); i++) {
                        assertRows("image " + i, cpu.encode(images.get(i)), parallel.get(i), 1e-3f);
                    }
                } finally {
                    cpu.close();
                }
            } finally {
                encoder.close();
            }
        }
    }
}
