package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.LoraFixtures;
import com.eonmux.neuralfree.gguf.video.VideoOptions;
import org.junit.ClassRule;
import org.junit.Assume;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Per-request adapters on the media path (vLLM keys its encoder and prefix caches by the adapter name and the
 * media hash): a request's selection changes the rows its image encodes to, the encoded-media key carries the
 * adapter identity while the processed layer keeps the plain item, a request that repeats a uuid another adapter
 * encoded re-encodes it under its own adapters without the data, and two concurrent requests under different
 * adapters each get their own adapter's rows. A request without a selection encodes under the startup adapters as
 * before.
 */
public class MediaContextLoraTest {

    private static final int WIDTH = 16;
    private static final String QKV = "v.blk.0.attn_qkv.weight";
    private static final String CONNECTOR = "mm.0.weight";
    private static final float TOL = 2e-3f;

    @ClassRule
    public static final TemporaryFolder TMP = new TemporaryFolder();

    /** A text encoder of one token per character. */
    private static final class CharText implements MediaContext.TextEncoder {
        @Override
        public int[] encode(String text) {
            int[] out = new int[text.length()];
            for (int i = 0; i < out.length; i++) {
                out[i] = text.charAt(i);
            }
            return out;
        }
    }

    /** A vision adapter over the llava fixture's fused attention and connector tensors. */
    private static LoraAdapter visionAdapter(String name, long seed) {
        return LoraFixtures.adapter(name + ".gguf", 8.0f,
                LoraFixtures.visionProjection(QKV, 8, 24, 2, seed),
                LoraFixtures.visionProjection(CONNECTOR, 8, 16, 2, seed + 1));
    }

    /** A registry holding adapters {@code a} and {@code b} of {@link #visionAdapter}. */
    private static LoraRegistry registry(LoraAdapter a, LoraAdapter b) {
        return LoraRegistry.of(path -> path.startsWith("a") ? a : b,
                LoraConfig.builder().maxAdapters(4).maxRank(64).cacheEntries(4).build());
    }

    /** The keys of adapters {@code a} and {@code b} at scale 1, resolved once. */
    private static LoraSlotKey[] keys(LoraRegistry registry) throws IOException {
        registry.load("a", "a.gguf", 1.0f);
        registry.load("b", "b.gguf", 1.0f);
        return new LoraSlotKey[]{registry.resolve(LoraSelection.of("a")),
                registry.resolve(LoraSelection.of("b"))};
    }

    /** A CPU-encoder context over the llava fixture with the given encoded-cache budget in bytes. */
    private static MediaContext open(Path file, long cacheBytes) throws IOException {
        return MediaContext.open(file.toString(), WIDTH, false, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), cacheBytes);
    }

    /** The embedding rows the context's image chunk of one append carries. */
    private static float[][] imageRows(MediaContext ctx, String uuid, byte[] png, LoraSlotKey lora)
            throws IOException {
        MediaPrompt.Builder b = MediaPrompt.builder().appendText(new CharText().encode("a"));
        ctx.appendImage(b, uuid, png == null ? null : () -> png, 0, new CharText(), lora);
        return b.build().chunks().get(1).embeddings();
    }

    private static float maxAbsDiff(float[][] a, float[][] b) {
        assertEquals(a.length, b.length);
        float max = 0.0f;
        for (int r = 0; r < a.length; r++) {
            assertEquals(a[r].length, b[r].length);
            for (int c = 0; c < a[r].length; c++) {
                max = Math.max(max, Math.abs(a[r][c] - b[r][c]));
            }
        }
        return max;
    }

    @Test
    public void aRequestsSelectionChangesItsImageRowsAndEachAdapterItsOwn() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = MediaContextTest.png(12, 8);
        LoraAdapter a = visionAdapter("a", 40);
        LoraAdapter b = visionAdapter("b", 60);
        LoraSlotKey[] keys = keys(registry(a, b));

        try (MediaContext plain = open(file, MediaContext.DEFAULT_CACHE_BYTES);
             MediaContext startup = MediaContext.open(file.toString(), WIDTH, false,
                     ImageTokenBudget.none(), t -> false,
                     MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES, MediaIoOptions.defaults(),
                     VideoOptions.defaults(), List.of(new LoraAttachment.Binding(a, 1.0f)));
             MediaContext session = open(file, MediaContext.DEFAULT_CACHE_BYTES)) {
            float[][] base = imageRows(plain, null, png, null);
            float[][] rowsA = imageRows(session, null, png, keys[0]);
            float[][] rowsB = imageRows(session, null, png, keys[1]);
            float[][] startupA = imageRows(startup, null, png, null);

            assertTrue(maxAbsDiff(base, rowsA) > 1e-3f);
            assertTrue(maxAbsDiff(base, rowsB) > 1e-3f);
            assertTrue(maxAbsDiff(rowsA, rowsB) > 1e-3f);
            // the session slot applies the same pairs the startup attachment does
            assertRowsExact(startupA, rowsA);
        }
    }

    @Test
    public void theEncodedKeyCarriesTheAdapterAndTheProcessedLayerStaysShared() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = MediaContextTest.png(12, 8);
        LoraSlotKey[] keys = keys(registry(visionAdapter("a", 40), visionAdapter("b", 60)));

        try (MediaContext ctx = open(file, MediaContext.DEFAULT_CACHE_BYTES)) {
            MediaKey plainKey = ctx.keyOf(MediaLimits.Modality.IMAGE, "m", null);
            MediaKey keyA = ctx.keyOf(MediaLimits.Modality.IMAGE, "m", null, keys[0]);
            MediaKey keyB = ctx.keyOf(MediaLimits.Modality.IMAGE, "m", null, keys[1]);
            assertNotEquals(plainKey, keyA);
            assertNotEquals(plainKey, keyB);
            assertNotEquals(keyA, keyB);

            float[][] first = imageRows(ctx, "m", png, keys[0]);
            assertEquals(1, ctx.cacheQueries());
            assertEquals(0, ctx.cacheHits());

            // the same item under the same adapter comes from the encoded layer
            assertRowsExact(first, imageRows(ctx, "m", null, keys[0]));
            assertEquals(2, ctx.cacheQueries());
            assertEquals(1, ctx.cacheHits());

            // another adapter re-encodes the shared processed tiles without the data
            float[][] other = imageRows(ctx, "m", null, keys[1]);
            assertTrue(maxAbsDiff(first, other) > 1e-3f);

            // with the encodings dropped the request encodes its adapter's rows again
            ctx.clearEncodings();
            assertRowsExact(first, imageRows(ctx, "m", null, keys[0]));

            // with both layers dropped the uuid alone is refused by name
            ctx.clearCache();
            MediaInputException e = assertThrows(MediaInputException.class,
                    () -> imageRows(ctx, "m", null, keys[0]));
            assertTrue(e.getMessage(), e.getMessage().contains("not in the media cache"));
            // the session closed on the way out, the encoder still serves a full encode
            assertRowsExact(first, imageRows(ctx, "m", png, keys[0]));
        }
    }

    @Test
    public void twoConcurrentRequestsUnderDifferentAdaptersKeepTheirOwnRows() throws Exception {
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = MediaContextTest.png(12, 8);
        LoraSlotKey[] keys = keys(registry(visionAdapter("a", 40), visionAdapter("b", 60)));

        try (MediaContext ctx = open(file, 0)) {
            float[][] refA = imageRows(ctx, null, png, keys[0]);
            float[][] refB = imageRows(ctx, null, png, keys[1]);
            assertTrue(maxAbsDiff(refA, refB) > 1e-3f);

            int runs = 6;
            ExecutorService pool = Executors.newFixedThreadPool(2);
            CountDownLatch start = new CountDownLatch(1);
            Future<List<float[][]>> fa = pool.submit(() -> encodeAll(ctx, png, keys[0], runs, start));
            Future<List<float[][]>> fb = pool.submit(() -> encodeAll(ctx, png, keys[1], runs, start));
            start.countDown();
            List<float[][]> rowsA = fa.get(60, TimeUnit.SECONDS);
            List<float[][]> rowsB = fb.get(60, TimeUnit.SECONDS);
            pool.shutdown();

            assertEquals(runs, rowsA.size());
            assertEquals(runs, rowsB.size());
            for (int i = 0; i < runs; i++) {
                assertRowsExact(refA, rowsA.get(i));
                assertRowsExact(refB, rowsB.get(i));
            }
        }
    }

    /** {@code runs} fresh encodes of the same image under one adapter, released together. */
    private static List<float[][]> encodeAll(MediaContext ctx, byte[] png, LoraSlotKey lora, int runs,
                                             CountDownLatch start) throws Exception {
        start.await();
        List<float[][]> out = new ArrayList<>(runs);
        for (int i = 0; i < runs; i++) {
            out.add(imageRows(ctx, null, png, lora));
        }
        return out;
    }

    @Test
    public void aGpuEncoderAppliesTheRequestAdapterLikeTheCpuOne() throws IOException {
        Assume.assumeTrue("no OpenCL device", GpuTestUtil.isOpenCLAvailable());
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = MediaContextTest.png(12, 8);
        LoraSlotKey[] keys = keys(registry(visionAdapter("a", 40), visionAdapter("b", 60)));

        try (MediaContext cpu = open(file, MediaContext.DEFAULT_CACHE_BYTES);
             MediaContext gpu = MediaContext.open(file.toString(), WIDTH, false,
                     ImageTokenBudget.none(), t -> false,
                     MmprojOptions.defaults(), MediaContext.DEFAULT_CACHE_BYTES)) {
            Assume.assumeFalse("no GPU device runs the vision encoder",
                    VisionEncoder.CPU_DEVICE.equals(gpu.visionDevice()));
            assertRows(imageRows(cpu, null, png, keys[0]), imageRows(gpu, null, png, keys[0]));
            assertRows(imageRows(cpu, null, png, keys[1]), imageRows(gpu, null, png, keys[1]));
            assertRows(imageRows(cpu, null, png, null), imageRows(gpu, null, png, null));
        }
    }

    private static void assertRows(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int r = 0; r < want.length; r++) {
            assertArrayEquals("row " + r, want[r], got[r], TOL);
        }
    }

    /** Bitwise equality of two row blocks: the same deterministic CPU encode. */
    private static void assertRowsExact(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int r = 0; r < want.length; r++) {
            assertArrayEquals("row " + r, want[r], got[r], 0f);
        }
    }
}
