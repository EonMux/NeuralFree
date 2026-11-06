package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Random;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A vision-encoder and connector LoRA adapter loaded at startup ({@code --lora}) reaches the media path: the
 * service opens the adapter against the text model's weights plus the mmproj tensor table (a tower-only target
 * would otherwise fail as an unknown base tensor), and the media projector applies its vision pairs, so the same
 * image pools to different embeddings than the base model's.
 */
class VisionLoraMediaTest {

    private static final int RANK = 2;
    private static final float ALPHA = 4.0f;
    private static final String TOWER_TARGET = "v.blk.0.attn_qkv.weight";
    private static final String CONNECTOR_TARGET = "mm.0.weight";

    private static EngineConfig config() {
        return ServerTestSupport.config().pooling(PoolingKind.LAST)
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A GGUF adapter of the synthetic llama's architecture whose only pairs adapt the mmproj's tower and
     *  connector. */
    private static Path visionAdapter(int nEmbd) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", ALPHA);
        pair(w, TOWER_TARGET, 8, 24, 17);
        pair(w, CONNECTOR_TARGET, 8, nEmbd, 19);
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

    private static double[] embedding(PoolingResource pooling, String image) {
        String messages = "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": "
                + "{\"url\": \"" + image + "\"}}, {\"type\": \"text\", \"text\": \"a cat\"}]}]";
        Response response = pooling.embeddings("{\"messages\": " + messages + "}");
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        List<?> values = (List<?>) at(view(response.getEntity()), "data", 0).raw("embedding");
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).doubleValue();
        }
        return out;
    }

    @Test
    void aStartupTowerAndConnectorAdapterChangesTheMediaEmbeddings() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int nEmbd = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(nEmbd);
        Path adapter = visionAdapter(nEmbd);
        EngineConfig base = config();
        EngineConfig adapted = config().toBuilder().addLora(adapter.toString(), 1.0f).build();
        assertEquals(1, adapted.loras().size());
        assertTrue(EngineService.withVisionCatalog(adapted, mmproj.toString()).sourceOptions()
                .loraVisionCatalog().has(TOWER_TARGET), "the mmproj table must carry the tower tensor");
        String image = dataUrl(MultimodalChatResourceTest.png(12, 8));

        double[] plain;
        double[] withAdapter;
        try (EngineService plainService = new EngineService(model.toString(), base, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
             EngineService adaptedService = new EngineService(model.toString(), adapted, ServerOptions.defaults(),
                     MediaOptions.of(mmproj.toString()))) {
            plain = embedding(new PoolingResource(plainService), image);
            withAdapter = embedding(new PoolingResource(adaptedService), image);
        }

        assertEquals(nEmbd, plain.length);
        assertEquals(plain.length, withAdapter.length);
        assertFalse(Arrays.equals(plain, withAdapter),
                "the vision and connector adapter must change the pooled media embedding");
        double max = 0.0;
        for (int i = 0; i < plain.length; i++) {
            max = Math.max(max, Math.abs(plain[i] - withAdapter[i]));
        }
        assertTrue(max > 1e-4, "the adapter changed the embedding by only " + max);
    }

    @Test
    void aTowerOnlyAdapterFailsWithoutTheMmprojTable() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path adapter = visionAdapter(SyntheticModels.Config.llama().nEmbd());
        EngineConfig adapterOnly = config().toBuilder().addLora(adapter.toString(), 1.0f).build();

        // Without a projector the service never parses the mmproj table, so the adapter's vision targets are
        // unknown base tensors and the load is refused by name rather than silently skipped.
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new EngineService(model.toString(), adapterOnly, ServerOptions.defaults()));
        assertTrue(e.getMessage().contains(TOWER_TARGET), e.getMessage());
    }
}
