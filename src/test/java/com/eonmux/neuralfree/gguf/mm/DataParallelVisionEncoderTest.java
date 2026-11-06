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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
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
            encoded.add(image);
            if (failures.containsKey(image)) {
                throw new IllegalStateException(name + " cannot encode its share");
            }
            return delegate.encode(image);
        }

        @Override
        public float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY,
                                      int origWidth, int origHeight) {
            return delegate.encodeAnyres(overview, tiles, gridX, gridY, origWidth, origHeight);
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
