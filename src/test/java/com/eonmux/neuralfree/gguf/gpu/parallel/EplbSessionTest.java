package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRankingFile;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The EPLB session over a synthetic routed-MoE spec: the profile it reads fills the placement, the routed rows it
 * records are merged with the file's heats and written at close, a profile naming an expert the layer does not have
 * is refused when placed, and a save-only session records without balancing.
 */
public class EplbSessionTest {

    private Path dir;
    private Path modelFile;
    private GGUFFileParser parser;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        dir = Files.createTempDirectory("eplb-session");
        modelFile = dir.resolve("model.gguf");
        SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build())
                .writeTo(modelFile);
        parser = new GGUFFileParser(modelFile.toString());
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                MemoryConfig.defaults().toBuilder().ubatch(3).build());
    }

    @After
    public void close() throws IOException {
        try {
            parser.close();
        } finally {
            Files.walk(dir).sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    @Test
    public void theProfileFillsThePlacementAndTheCountedRowsAreSaved() throws IOException {
        Path profile = dir.resolve("profile.bin");
        MoeRankingFile.write(profile, entries(40, 40, 1, 1), 4);
        try (EplbSession session = EplbSession.open(profile, profile, spec)) {
            assertTrue(session.balances());
            EplbPlacement placement = session.placement(2);
            assertNotNull(placement);
            LayerSpec layer = spec.layers().get(0);
            assertTrue(placement.covers(layer));
            assertEquals(Arrays.toString(new int[]{0, 2}), Arrays.toString(placement.experts(layer, 0)));
            assertEquals(Arrays.toString(new int[]{1, 3}), Arrays.toString(placement.experts(layer, 1)));
            session.heat().note(0, new int[]{0, 0, 1}, 3);
        }
        List<RankedExpert> saved = MoeRankingFile.read(profile).entries();
        assertEquals("e0 keeps its 40 and gains two routed rows", 42, heat(saved, 0, 0));
        assertEquals("e1 keeps its 40 and gains one routed row", 41, heat(saved, 0, 1));
        assertEquals(1, heat(saved, 0, 2));
        assertEquals(1, heat(saved, 0, 3));
    }

    @Test
    public void aSaveOnlySessionRecordsWithoutBalancing() throws IOException {
        Path save = dir.resolve("save.bin");
        try (EplbSession session = EplbSession.open(null, save, spec)) {
            assertFalse(session.balances());
            assertNull(session.placement(2));
            session.heat().note(0, new int[]{3, 3}, 2);
        }
        assertEquals(2, heat(MoeRankingFile.read(save).entries(), 0, 3));
    }

    @Test
    public void aProfileNamingAnUnknownExpertIsRefusedWhenPlaced() throws IOException {
        Path profile = dir.resolve("profile.bin");
        MoeRankingFile.write(profile, List.of(new RankedExpert(0, 5, 1)), 6);
        try (EplbSession session = EplbSession.open(profile, null, spec)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> session.placement(2));
            assertTrue(e.getMessage(), e.getMessage().contains("5"));
            assertTrue(e.getMessage(), e.getMessage().contains("4"));
        }
    }

    @Test
    public void aMissingProfileIsRefusedByName() {
        Path missing = dir.resolve("absent.bin");
        IOException e = assertThrows(IOException.class, () -> EplbSession.open(missing, null, spec));
        assertTrue(e.getMessage(), e.getMessage().contains(missing.toString()));
    }

    @Test
    public void aSessionNeedsAProfileOrASave() {
        assertThrows(IllegalArgumentException.class, () -> EplbSession.open(null, null, spec));
    }

    @Test
    public void aRebalanceOnlySessionNeedsNoPaths() throws IOException {
        EplbConfig cfg = EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 0, 0, 1, true, 3000, 1000);
        try (EplbSession session = EplbSession.open(cfg, spec)) {
            assertFalse(session.balances());
            assertNull(session.placement(2));
            assertNull(session.heat());
            assertNull(session.profilePath());
            assertNull(session.savePath());
        }

        EplbConfig redundant = EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 2, 0, 1, true, 3000, 1000);
        try (EplbSession session = EplbSession.open(redundant, spec)) {
            assertNotNull("the redundant replicas spread from zero load", session.placement(2));
        }
    }

    private static List<RankedExpert> entries(int e0, int e1, int e2, int e3) {
        return Arrays.asList(new RankedExpert(0, 0, e0), new RankedExpert(0, 1, e1),
                new RankedExpert(0, 2, e2), new RankedExpert(0, 3, e3));
    }

    private static long heat(List<RankedExpert> entries, int layer, int expert) {
        for (RankedExpert e : entries) {
            if (e.layer() == layer && e.expert() == expert) {
                return e.heat();
            }
        }
        throw new AssertionError("no entry for layer " + layer + " expert " + expert);
    }
}
