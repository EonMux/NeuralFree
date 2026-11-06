package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A request's adapter selection reaches its media (vLLM runs the request's adapter while its images encode and
 * keys the encoder cache by the adapter): an embeddings request naming a {@code --lora-modules} adapter by its
 * {@code model} (or listing it under {@code lora}) pools the image the same startup adapter does, another
 * adapter's request pools its own rows off the same shared image, an empty or zero-scale selection runs none, and
 * two concurrent requests under different adapters each get their own adapter's embedding.
 */
class PerRequestVisionLoraTest {

    private static final String BASE = "base";
    private static final int RANK = 2;
    private static final float ALPHA = 4.0f;
    private static final String TOWER_TARGET = "v.blk.0.attn_qkv.weight";
    private static final String CONNECTOR_TARGET = "mm.0.weight";

    private static Path adapterA;
    private static Path adapterB;
    private static EngineService service;
    private static EngineService referenceA;
    private static EngineService referenceB;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int nEmbd = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(nEmbd);
        adapterA = visionAdapter(nEmbd, 170);
        adapterB = visionAdapter(nEmbd, 270);
        ServerOptions options = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT);
        service = new EngineService(model.toString(), config(), options, MediaOptions.of(mmproj.toString()),
                new ServerLoras(List.of(new ServerLoras.Module("a", adapterA.toString()),
                        new ServerLoras.Module("b", adapterB.toString())), true));
        referenceA = new EngineService(model.toString(), config().toBuilder()
                .addLora(adapterA.toString(), 1.0f).build(), options, MediaOptions.of(mmproj.toString()));
        referenceB = new EngineService(model.toString(), config().toBuilder()
                .addLora(adapterB.toString(), 1.0f).build(), options, MediaOptions.of(mmproj.toString()));
    }

    @AfterAll
    static void close() {
        service.close();
        referenceA.close();
        referenceB.close();
    }

    private static EngineConfig config() {
        return ServerTestSupport.config().pooling(PoolingKind.LAST)
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A GGUF adapter whose only pairs adapt the mmproj's tower and connector tensors. */
    private static Path visionAdapter(int nEmbd, long seed) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", ALPHA);
        pair(w, TOWER_TARGET, 8, 24, seed);
        pair(w, CONNECTOR_TARGET, 8, nEmbd, seed + 1);
        return w.writeTempFile();
    }

    private static void pair(GgufTestWriter w, String target, int in, int out, long seed) {
        w.tensorF32(target + ".lora_a", new long[]{in, RANK}, random(seed, RANK * in));
        w.tensorF32(target + ".lora_b", new long[]{RANK, out}, random(seed + 1, out * RANK));
    }

    private static float[] random(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = r.nextFloat() - 0.5f;
        }
        return out;
    }

    private static String dataUrl(byte[] png) {
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
    }

    /** The pooled vector of the image request, naming adapter {@code model} or carrying {@code loraExtra}. */
    private static double[] imageEmbedding(EngineService on, String model, String loraExtra, byte[] png) {
        String messages = "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": "
                + "{\"url\": \"" + dataUrl(png) + "\"}}, {\"type\": \"text\", \"text\": \"a cat\"}]}]";
        String body = "{\"messages\": " + messages + (model == null ? "" : ", \"model\": \"" + model + "\"")
                + (loraExtra == null ? "" : loraExtra) + "}";
        return pooled(new PoolingResource(on).embeddings(body));
    }

    /** The pooled vector of a text-only request, naming adapter {@code model}. */
    private static double[] textEmbedding(EngineService on, String model) {
        String body = "{\"input\": \"hello world\"" + (model == null ? "" : ", \"model\": \"" + model + "\"")
                + "}";
        return pooled(new PoolingResource(on).embeddings(body));
    }

    private static double[] pooled(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        List<?> values = (List<?>) at(view(response.getEntity()), "data", 0).raw("embedding");
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).doubleValue();
        }
        return out;
    }

    private static double maxDiff(double[] a, double[] b) {
        assertEquals(a.length, b.length);
        double max = 0.0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    /** The two runs carry the same vector: the same adapter served both. */
    private static void assertClose(double[] want, double[] got) {
        assertTrue(maxDiff(want, got) < 1e-9, "max diff " + maxDiff(want, got));
    }

    private static void assertChanged(double[] a, double[] b) {
        assertTrue(maxDiff(a, b) > 1e-6, "max diff " + maxDiff(a, b));
    }

    @Test
    void aRequestNamingAnAdapterEmbedsItsImageAsTheStartupAdapterWould() {
        byte[] png = MultimodalChatResourceTest.png(12, 8);
        double[] base = imageEmbedding(service, null, null, png);
        double[] a = imageEmbedding(service, "a", null, png);
        double[] b = imageEmbedding(service, "b", null, png);
        double[] refA = imageEmbedding(referenceA, null, null, png);
        double[] refB = imageEmbedding(referenceB, null, null, png);

        assertClose(refA, a);
        assertClose(refB, b);
        assertChanged(base, a);
        assertChanged(base, b);
        assertChanged(a, b);

        // the llama.cpp lora list selects the same adapter, an empty list or a zero scale runs none
        assertClose(a, imageEmbedding(service, null, ", \"lora\": [{\"id\": 0, \"scale\": 1.0}]", png));
        assertClose(base, imageEmbedding(service, null, ", \"lora\": []", png));
        assertClose(base, imageEmbedding(service, null, ", \"lora\": [{\"id\": 0, \"scale\": 0}]", png));
    }

    @Test
    void theAdapterScopedCacheServesEachAdaptersEncodingOfTheSharedImage() {
        byte[] png = MultimodalChatResourceTest.png(14, 8);
        long queries = service.media().cacheQueries();

        double[] first = imageEmbedding(service, "a", null, png);
        assertEquals(queries + 1, service.media().cacheQueries());
        assertClose(first, imageEmbedding(service, "a", null, png));
        assertEquals(queries + 2, service.media().cacheQueries());
        assertTrue(service.media().cacheHits() >= 1, "the repeat must hit the adapter-scoped cache");

        // another adapter's request of the same image is a miss of its own scoped key and encodes again
        double[] other = imageEmbedding(service, "b", null, png);
        assertEquals(queries + 3, service.media().cacheQueries());
        assertChanged(first, other);
    }

    @Test
    void twoConcurrentRequestsUnderDifferentAdaptersGetTheirOwnEmbedding() throws Exception {
        List<byte[]> images = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            images.add(MultimodalChatResourceTest.png(12 + i, 8));
        }
        List<double[]> refA = new ArrayList<>();
        List<double[]> refB = new ArrayList<>();
        for (byte[] image : images) {
            refA.add(imageEmbedding(service, "a", null, image));
            refB.add(imageEmbedding(service, "b", null, image));
        }
        assertChanged(refA.get(0), refB.get(0));

        // each thread encodes its fresh images under its own adapter, both released together
        ExecutorService pool = Executors.newFixedThreadPool(2);
        CountDownLatch start = new CountDownLatch(1);
        Future<List<double[]>> fa = pool.submit(() -> runAll(service, "a", images, start));
        Future<List<double[]>> fb = pool.submit(() -> runAll(service, "b", images, start));
        start.countDown();
        List<double[]> rowsA = fa.get(120, TimeUnit.SECONDS);
        List<double[]> rowsB = fb.get(120, TimeUnit.SECONDS);
        pool.shutdown();

        assertEquals(images.size(), rowsA.size());
        assertEquals(images.size(), rowsB.size());
        for (int i = 0; i < images.size(); i++) {
            assertClose(refA.get(i), rowsA.get(i));
            assertClose(refB.get(i), rowsB.get(i));
            assertChanged(rowsA.get(i), rowsB.get(i));
        }
    }

    /** The pooled vectors of every image under one adapter, the requests released together. */
    private static List<double[]> runAll(EngineService on, String model, List<byte[]> images,
                                         CountDownLatch start) throws Exception {
        start.await();
        List<double[]> out = new ArrayList<>(images.size());
        for (byte[] image : images) {
            out.add(imageEmbedding(on, model, null, image));
        }
        return out;
    }

    @Test
    void anUnknownAdapterModelIsRefusedAndAVisionOnlyAdapterLeavesTextAlone() {
        byte[] png = MultimodalChatResourceTest.png(12, 8);
        assertEquals(404, statusOf(() -> imageEmbedding(service, "z", null, png)));

        double[] base = textEmbedding(service, null);
        assertClose(base, textEmbedding(service, "a"));
        assertClose(base, textEmbedding(referenceA, null));
    }
}
