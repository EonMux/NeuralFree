package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@link MoeCacheRanking} session: the options of the properties (the default beside the model, a blank value
 * keeping it, off, a named file, the profile save), the preload of the ranking beside a model file or inside a model
 * directory with its demand shares and the cache's bounds filter, the named refusal of a missing configured ranking,
 * the logged skip of unreadable ranking bytes, and the profile save at close that merges the saved file's heats of
 * the same model shape with the counted routed rows.
 */
public class MoeCacheRankingTest {

    private Path dir;

    @Before
    public void makeDir() throws IOException {
        dir = Files.createTempDirectory("moe-cache-ranking-test");
    }

    @After
    public void removeDir() throws IOException {
        delete(dir);
    }

    @Test
    public void defaultsPreloadBesideTheModelAndSaveNothing() {
        Properties props = new Properties();
        MoeCacheRanking.Options options = MoeCacheRanking.Options.fromProperties(props);
        assertEquals(MoeCacheRanking.PreloadMode.AUTO, options.preloadMode());
        assertNull(options.preloadPath());
        assertNull(options.savePath());
    }

    @Test
    public void blankKeepsTheDefaultAndOffDisablesThePreload() {
        Properties empty = new Properties();
        empty.setProperty(MoeCacheRanking.PROP_PRELOAD, "  ");
        empty.setProperty(MoeCacheRanking.PROP_SAVE, "");
        assertEquals(MoeCacheRanking.Options.auto(), MoeCacheRanking.Options.fromProperties(empty));
        assertTrue(MoeCacheRanking.Options.fromProperties(empty).isDefault());
        Properties off = new Properties();
        off.setProperty(MoeCacheRanking.PROP_PRELOAD, "Off");
        assertEquals(MoeCacheRanking.PreloadMode.OFF,
                MoeCacheRanking.Options.fromProperties(off).preloadMode());
    }

    @Test
    public void namedPreloadAndSavePathsAreTaken() {
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_PRELOAD, "ranking.bin");
        props.setProperty(MoeCacheRanking.PROP_SAVE, "profile.bin");
        MoeCacheRanking.Options options = MoeCacheRanking.Options.fromProperties(props);
        assertEquals(MoeCacheRanking.PreloadMode.PATH, options.preloadMode());
        assertEquals("ranking.bin", options.preloadPath().toString());
        assertEquals("profile.bin", options.savePath().toString());
    }

    @Test
    public void withPreloadReadsAPreloadValueAndKeepsTheSavePath() {
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_SAVE, "profile.bin");
        MoeCacheRanking.Options base = MoeCacheRanking.Options.fromProperties(props);

        MoeCacheRanking.Options named = base.withPreload(" other.bin ");
        assertEquals(MoeCacheRanking.PreloadMode.PATH, named.preloadMode());
        assertEquals("other.bin", named.preloadPath().toString());
        assertEquals("profile.bin", named.savePath().toString());

        MoeCacheRanking.Options off = named.withPreload("OFF");
        assertEquals(MoeCacheRanking.PreloadMode.OFF, off.preloadMode());
        assertNull(off.preloadPath());
        assertEquals("profile.bin", off.savePath().toString());

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> base.withPreload(" "));
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheRanking.PROP_PRELOAD));
    }

    @Test
    public void unreadablePathsAreRefusedByName() {
        Properties preload = new Properties();
        preload.setProperty(MoeCacheRanking.PROP_PRELOAD, "bad\u0000name");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeCacheRanking.Options.fromProperties(preload));
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheRanking.PROP_PRELOAD));
        Properties save = new Properties();
        save.setProperty(MoeCacheRanking.PROP_SAVE, "bad\u0000name");
        e = assertThrows(IllegalArgumentException.class, () -> MoeCacheRanking.Options.fromProperties(save));
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheRanking.PROP_SAVE));
    }

    @Test
    public void inconsistentOptionsAreRefused() {
        NullPointerException missing = assertThrows(NullPointerException.class,
                () -> new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.PATH, null, null));
        assertEquals("preloadPath", missing.getMessage());
        IllegalArgumentException extra = assertThrows(IllegalArgumentException.class,
                () -> new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.OFF, dir.resolve("x"), null));
        assertTrue(extra.getMessage(), extra.getMessage().contains("OFF"));
    }

    @Test
    public void theRankingBesideTheModelFilePreloads() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        writeRanking(dir.resolve(MoeCacheRanking.RANKING_FILE_NAME),
                new int[]{0, 1, 30, 2, 4, 30, 1, 3, 40});
        MoeCacheRanking session = MoeCacheRanking.open(model, 8, MoeCacheRanking.Options.auto());
        try {
            assertTrue(session.preloaded());
            assertEquals("the ranking keeps the file's order, hottest first",
                    Arrays.asList(
                            new RankedExpert(0, 1, 30),
                            new RankedExpert(2, 4, 30),
                            new RankedExpert(1, 3, 40)), session.ranking());
            assertEquals(30.0 / (30 + 30 + 40), session.share(0), 1e-12);
            assertEquals(30.0 / (30 + 30 + 40), session.share(1), 1e-12);
            assertEquals(40.0 / (30 + 30 + 40), session.share(2), 1e-12);
            assertEquals("the header's layers", 3, session.rankingLayers());
            assertEquals("the header's experts", 5, session.rankingExperts());
            assertEquals(dir.resolve(MoeCacheRanking.RANKING_FILE_NAME).toAbsolutePath(), session.preloadedFrom());
            assertNull("no profile save, no recorder", session.heat());
        } finally {
            session.close();
        }
    }

    /** A converted checkpoint is a directory: its ranking sits inside it. */
    @Test
    public void theRankingInsideTheModelDirectoryPreloads() throws IOException {
        Path model = Files.createDirectory(dir.resolve("model"));
        writeRanking(model.resolve(MoeCacheRanking.RANKING_FILE_NAME), new int[]{0, 0, 5});
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.auto());
        try {
            assertTrue(session.preloaded());
            assertEquals(Collections.singletonList(new RankedExpert(0, 0, 5)), session.ranking());
        } finally {
            session.close();
        }
    }

    @Test
    public void zeroHeatsShareEvenly() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        writeRanking(dir.resolve(MoeCacheRanking.RANKING_FILE_NAME), new int[]{0, 1, 0, 0, 2, 0});
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.auto());
        try {
            assertEquals(0.0, session.share(0), 0.0);
            assertEquals(0.0, session.share(1), 0.0);
        } finally {
            session.close();
        }
    }

    @Test
    public void sharesOutsideTheRankingAreRefusedByName() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.auto());
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> session.share(0));
            assertTrue(e.getMessage(), e.getMessage().contains("[0, 0)"));
        } finally {
            session.close();
        }
    }

    @Test
    public void anExplicitPathPreloadsAndAMissingOneIsRefusedByName() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        Path ranking = dir.resolve("other.bin");
        writeRanking(ranking, new int[]{0, 0, 5});
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_PRELOAD, ranking.toString());
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.fromProperties(props));
        try {
            assertTrue(session.preloaded());
        } finally {
            session.close();
        }
        Properties missing = new Properties();
        missing.setProperty(MoeCacheRanking.PROP_PRELOAD, dir.resolve("absent.bin").toString());
        IOException e = assertThrows(IOException.class,
                () -> MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.fromProperties(missing)));
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheRanking.PROP_PRELOAD));
        assertTrue(e.getMessage(), e.getMessage().contains("absent.bin"));
    }

    @Test
    public void foreignRankingBytesAreNamedAndSkipped() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        Path ranking = dir.resolve("garbage.bin");
        Files.write(ranking, new byte[24]);
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_PRELOAD, ranking.toString());
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.fromProperties(props));
        try {
            assertFalse(session.preloaded());
        } finally {
            session.close();
        }
    }

    @Test
    public void offPreloadsNothing() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        writeRanking(dir.resolve(MoeCacheRanking.RANKING_FILE_NAME), new int[]{0, 0, 5});
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_PRELOAD, "off");
        MoeCacheRanking session = MoeCacheRanking.open(model, 4, MoeCacheRanking.Options.fromProperties(props));
        try {
            assertFalse(session.preloaded());
        } finally {
            session.close();
        }
    }

    /** With a save configured the routed rows count, and close merges the saved file's heats of the same shape. */
    @Test
    public void theProfileSaveMergesTheSavedHeatsOfTheSameShape() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        Path save = dir.resolve("profile.bin");
        MoeCacheRanking.Options options = optionsWithSave(save);
        MoeCacheRanking first = MoeCacheRanking.open(model, 8, options);
        try {
            first.heat().note(0, new int[]{1, 1, 1}, 3);
        } finally {
            first.close();
        }
        assertEquals("the header names the model's expert count, so the next session's shape check passes",
                8, MoeRankingFile.read(save).nExperts());
        assertEquals(Arrays.asList(new RankedExpert(0, 1, 3)), MoeRankingFile.read(save).entries());

        MoeCacheRanking second = MoeCacheRanking.open(model, 8, optionsWithSave(save));
        try {
            second.heat().note(0, new int[]{1, 2}, 2);
        } finally {
            second.close();
        }
        assertEquals(Arrays.asList(
                new RankedExpert(0, 1, 4),
                new RankedExpert(0, 2, 1)), MoeRankingFile.read(save).entries());
    }

    /** A saved ranking of another expert count is left out of the merge, like the reference's shape check. */
    @Test
    public void aSavedRankingOfAnotherShapeIsNotMerged() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        Path save = dir.resolve("profile.bin");
        writeRanking(save, new int[]{0, 1, 500});
        MoeCacheRanking session = MoeCacheRanking.open(model, 8, optionsWithSave(save));
        try {
            session.heat().note(0, new int[]{3}, 1);
        } finally {
            session.close();
        }
        assertEquals(Collections.singletonList(new RankedExpert(0, 3, 1)), MoeRankingFile.read(save).entries());
    }

    @Test
    public void closeWithoutCountedRowsWritesNothing() throws IOException {
        Path model = dir.resolve("model.gguf");
        Files.write(model, new byte[1]);
        Path save = dir.resolve("profile.bin");
        MoeCacheRanking session = MoeCacheRanking.open(model, 8, optionsWithSave(save));
        try {
            assertTrue(session.heat().isEmpty());
        } finally {
            session.close();
        }
        assertTrue(Files.notExists(save));
    }

    /** The cache's bounds filter keeps the file order and drops the entries outside the served layers and experts. */
    @Test
    public void withinKeepsTheFileOrderInsideTheBounds() throws IOException {
        Path ranking = dir.resolve("ranking.bin");
        writeRanking(ranking, new int[]{0, 3, 900, 1, 5, 400, 0, 1, 70});
        MoeCacheRanking session = MoeCacheRanking.open(null, 8,
                new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.PATH, ranking, null));
        try {
            assertEquals(3, session.within(2, 8).size());
            assertEquals(Arrays.asList(new RankedExpert(0, 3, 900), new RankedExpert(0, 1, 70)),
                    session.within(2, 4));
            assertEquals(Arrays.asList(new RankedExpert(0, 3, 900), new RankedExpert(0, 1, 70)),
                    session.within(1, 8));
        } finally {
            session.close();
        }
    }

    /** A model without a file has nothing next to it: the default preload reads nothing. */
    @Test
    public void aModelWithoutAFileHasNoDefaultPreload() throws IOException {
        MoeCacheRanking session = MoeCacheRanking.open(null, 4, MoeCacheRanking.Options.auto());
        try {
            assertFalse(session.preloaded());
            assertNull(session.preloadedFrom());
            assertEquals(0, session.rankingLayers());
            assertEquals(Collections.emptyList(), session.within(4, 4));
        } finally {
            session.close();
        }
    }

    /** The model's routed MoE layers give the largest expert count, 0 without MoE. */
    @Test
    public void maxExpertsReadsTheRoutedLayers() {
        assertEquals(0, MoeCacheRanking.maxExperts(spec(4, false)));
        assertEquals(4, MoeCacheRanking.maxExperts(spec(4, true)));
        assertEquals(8, MoeCacheRanking.maxExperts(spec(8, true)));
    }

    // ------------------------------------------------------------------ helpers

    private static ModelSpec spec(int nExpert, boolean moe) {
        FfnKind kind = moe ? FfnKind.MOE : FfnKind.DENSE;
        LayerSpec.Builder layer = LayerSpec.builder(0, MixerKind.NONE, kind)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, 16).build());
        if (moe) {
            layer.moe(MoeSpec.builder().experts(nExpert, 1, 4).gating(GatingFunc.SOFTMAX)
                    .glu(GluOp.SWIGLU).build());
        } else {
            layer.ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(8).glu(FfnGating.PAR, GluOp.SWIGLU).build());
        }
        return ModelSpec.builder(new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA))
                        .layerCounts(1, 0).nCtxTrain(64).nEmbd(16).build(),
                        TensorCatalog.of("llama", Collections.emptyList()))
                .layer(layer.build())
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, 16).build())
                .head(HeadSpec.builder().build())
                .build();
    }

    private static MoeCacheRanking.Options optionsWithSave(Path save) {
        Properties props = new Properties();
        props.setProperty(MoeCacheRanking.PROP_SAVE, save.toString());
        return MoeCacheRanking.Options.fromProperties(props);
    }

    /** A version 2 ranking from {@code [layer, expert, heat, ...]} triples. */
    private static void writeRanking(Path file, int[] triples) throws IOException {
        ByteBuffer out = ByteBuffer.allocate(24 + triples.length / 3 * 8).order(ByteOrder.LITTLE_ENDIAN);
        out.put("STRP".getBytes(StandardCharsets.US_ASCII));
        out.putInt(2);
        int layers = 0;
        int experts = 0;
        for (int i = 0; i < triples.length; i += 3) {
            layers = Math.max(layers, triples[i] + 1);
            experts = Math.max(experts, triples[i + 1] + 1);
        }
        out.putInt(layers);
        out.putInt(experts);
        out.putInt(triples.length / 3);
        out.putInt(triples.length / 3);
        for (int i = 0; i < triples.length; i += 3) {
            out.putShort((short) triples[i]);
            out.putShort((short) triples[i + 1]);
            out.putInt(triples[i + 2]);
        }
        Files.write(file, out.array());
    }

    private static void delete(Path path) throws IOException {
        if (!Files.isDirectory(path)) {
            Files.deleteIfExists(path);
            return;
        }
        try (java.util.stream.Stream<Path> entries = Files.list(path)) {
            for (Path entry : entries.toArray(Path[]::new)) {
                delete(entry);
            }
        }
        Files.deleteIfExists(path);
    }
}
