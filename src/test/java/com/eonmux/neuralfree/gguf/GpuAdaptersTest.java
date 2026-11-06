package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The adapters of a GPU load over the synthetic two-layer llama: no adapter reserves nothing and never opens the host
 * weights; control vectors reserve one {@code n_embd} float row per trunk layer; a LoRA adapter is validated against
 * the host weights and reserves its pairs plus one {@code ubatch} row per non-expert target.
 */
public class GpuAdaptersTest {

    private static final int RANK = 4;
    private static final int UBATCH = 4;
    private static final String TARGET = "blk.0.attn_q.weight";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private GgufTestWriter writer;
    private Path file;
    private GGUFFileParser parser;
    private ModelSpec spec;
    private TensorCatalog catalog;
    private CpuWeights host;
    private int hostOpens;

    @Before
    public void open() throws IOException {
        writer = SyntheticModels.llama();
        file = writer.writeTempFile();
        parser = new GGUFFileParser(file.toString());
        catalog = TensorCatalog.of(parser);
        spec = ModelSpecBuilder.build(GgufKv.of(parser), catalog, memory());
    }

    @After
    public void close() throws IOException {
        try {
            if (host != null) {
                host.close();
            }
            parser.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void noAdapterReservesNothing() throws IOException {
        GpuAdapters adapters = GpuAdapters.load("m", spec, EngineConfig.builder().build(), memory(), this::hostWeights);

        assertNull(adapters.controlVectors());
        assertFalse(adapters.hasLoras());
        assertEquals(0L, adapters.deviceBytes());
        assertEquals(0, hostOpens);
    }

    @Test
    public void controlVectorsReserveOneRowPerTrunkLayer() throws IOException {
        Path vector = controlVector();
        EngineConfig cfg = EngineConfig.builder().addControlVector(vector.toString(), 0.5f).build();
        GpuAdapters adapters = GpuAdapters.load("m", spec, cfg, memory(), this::hostWeights);

        assertNotNull(adapters.controlVectors());
        assertEquals(1, adapters.controlVectors().list().size());
        assertFalse(adapters.hasLoras());
        assertEquals((long) spec.nLayer() * spec.hparams().nEmbd() * Float.BYTES, adapters.deviceBytes());
        assertEquals(0, hostOpens);
    }

    @Test
    public void aLoraAdapterReservesItsPairsAndItsStepRows() throws IOException {
        long[] ne = writer.tensorShape(TARGET);
        Path adapter = loraAdapter(ne);
        EngineConfig cfg = EngineConfig.builder().addLora(adapter.toString(), 1.0f).build();
        GpuAdapters adapters = GpuAdapters.load("m", spec, cfg, memory(), this::hostWeights);

        assertNull(adapters.controlVectors());
        assertTrue(adapters.hasLoras());
        long floats = ne[0] * RANK + RANK * ne[1] + UBATCH;
        assertEquals(floats * Float.BYTES, adapters.deviceBytes());
        assertEquals(1, hostOpens);
    }

    private CpuWeights hostWeights() throws IOException {
        hostOpens++;
        if (host == null) {
            host = CpuWeights.open(parser);
        }
        return host;
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(UBATCH).build();
    }

    private Path controlVector() throws IOException {
        int nEmbd = spec.hparams().nEmbd();
        List<float[]> rows = new ArrayList<>();
        for (int il = 0; il < spec.nLayer(); il++) {
            float[] row = new float[nEmbd];
            row[il] = 1f;
            rows.add(row);
        }
        return CvectorGenerator.export(rows, nEmbd, "llama", tmp.getRoot().toPath().resolve("cv.gguf"), true);
    }

    /** A rank-{@value #RANK} adapter of {@link #TARGET}: {@code A {in, r}} and {@code B {r, out}}. */
    private Path loraAdapter(long[] ne) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", RANK);
        w.tensorF32(TARGET + ".lora_a", new long[]{ne[0], RANK}, new float[Math.toIntExact(ne[0] * RANK)]);
        w.tensorF32(TARGET + ".lora_b", new long[]{RANK, ne[1]}, new float[Math.toIntExact(RANK * ne[1])]);
        Path out = tmp.getRoot().toPath().resolve("lora.gguf");
        w.writeTo(out);
        return out;
    }
}
