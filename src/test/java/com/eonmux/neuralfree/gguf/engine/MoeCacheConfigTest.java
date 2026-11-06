package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeTailSkip;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Parsing of {@link MoeCacheConfig} and its wiring through {@link EngineConfig}: the {@code off} default, the
 * {@code off | on | auto | MiB} values, the ranking properties carried with it, the prefill expert prefetch zone
 * count, and the refusals that name the property.
 */
public class MoeCacheConfigTest {

    @Test
    public void theDefaultIsOff() {
        MoeCacheConfig cfg = MoeCacheConfig.fromProperties(new Properties());

        assertEquals(MoeCacheConfig.Mode.OFF, cfg.mode());
        assertEquals(MoeCacheConfig.BUDGET_AUTO, cfg.budgetMiB());
        assertEquals(MoeCacheRanking.Options.auto(), cfg.ranking());
        assertFalse(cfg.enabled());
        assertFalse(EngineConfig.defaults().moeCache().enabled());
    }

    @Test
    public void parsesModesAndBudgets() {
        assertEquals(MoeCacheConfig.Mode.OFF, MoeCacheConfig.parse("off").mode());
        assertEquals(MoeCacheConfig.Mode.OFF, MoeCacheConfig.parse("OFF").mode());
        assertEquals(MoeCacheConfig.Mode.ON, MoeCacheConfig.parse("on").mode());
        assertEquals(MoeCacheConfig.BUDGET_AUTO, MoeCacheConfig.parse("on").budgetMiB());
        assertEquals(MoeCacheConfig.Mode.AUTO, MoeCacheConfig.parse("auto").mode());
        assertTrue(MoeCacheConfig.parse("auto").enabled());
        assertEquals(MoeCacheConfig.Mode.ON, MoeCacheConfig.parse("512").mode());
        assertEquals(512L, MoeCacheConfig.parse("512").budgetMiB());
        assertTrue(MoeCacheConfig.parse("0").enabled());
    }

    @Test
    public void refusesUnsupportedValuesByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeCacheConfig.parse("soft"));
        assertTrue(e.getMessage().contains("neuralfree.moe.cache"));
        e = assertThrows(IllegalArgumentException.class, () -> MoeCacheConfig.parse("-4"));
        assertTrue(e.getMessage().contains(">= 0"));
    }

    @Test
    public void readsThePropertiesThroughEngineConfig() {
        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "1024");
        p.setProperty(MoeCacheRanking.PROP_PRELOAD, "/tmp/ranking.bin");
        p.setProperty(MoeCacheRanking.PROP_SAVE, "/tmp/profile.bin");
        EngineConfig cfg = EngineConfig.fromProperties(p);

        assertEquals(MoeCacheConfig.Mode.ON, cfg.moeCache().mode());
        assertEquals(1024L, cfg.moeCache().budgetMiB());
        assertEquals(MoeCacheRanking.PreloadMode.PATH, cfg.moeCache().ranking().preloadMode());
        assertEquals(Paths.get("/tmp/ranking.bin"), cfg.moeCache().ranking().preloadPath());
        assertEquals(Paths.get("/tmp/profile.bin"), cfg.moeCache().ranking().savePath());
        assertTrue(cfg.toString(), cfg.toString().contains("moeCache=1024 MiB"));
        assertTrue(cfg.toString(), cfg.toString().contains("rankingSave=/tmp/profile.bin"));
        assertEquals(cfg.moeCache(), cfg.toBuilder().build().moeCache());
    }

    @Test
    public void aBlankRankingPropertyIsIgnored() {
        Properties p = new Properties();
        p.setProperty(MoeCacheRanking.PROP_PRELOAD, "  ");
        EngineConfig cfg = EngineConfig.fromProperties(p);

        assertEquals(MoeCacheRanking.PreloadMode.AUTO, cfg.moeCache().ranking().preloadMode());
        assertNull(cfg.moeCache().ranking().preloadPath());
    }

    /** The ranking options ride along with an off cache too: the profile save works without the cache. */
    @Test
    public void theRankingKeysApplyWithTheCacheOff() {
        Properties p = new Properties();
        p.setProperty(MoeCacheRanking.PROP_PRELOAD, "off");
        p.setProperty(MoeCacheRanking.PROP_SAVE, "/tmp/profile.bin");
        EngineConfig cfg = EngineConfig.fromProperties(p);

        assertFalse(cfg.moeCache().enabled());
        assertEquals(MoeCacheRanking.PreloadMode.OFF, cfg.moeCache().ranking().preloadMode());
        assertEquals(Paths.get("/tmp/profile.bin"), cfg.moeCache().ranking().savePath());
        assertTrue(cfg.toString(), cfg.toString().contains("ranking=off, rankingSave=/tmp/profile.bin"));
        assertEquals(cfg.moeCache(), cfg.toBuilder().build().moeCache());
        assertEquals(cfg.moeCache().ranking(), cfg.moeCache().withBudgetMiB(64).ranking());
    }

    @Test
    public void thePrefetchDefaultsToTwoZonesAndReadsThroughEngineConfig() {
        assertEquals(MoeCacheConfig.PREFETCH_ZONES_DEFAULT, MoeCacheConfig.off().prefetchZones());
        assertEquals(2, MoeCacheConfig.parse("on").prefetchZones());

        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH, "4");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(4, cfg.moeCache().prefetchZones());
        assertTrue(cfg.toString().contains("moeCachePrefetch=4"));
        assertEquals(cfg.moeCache(), cfg.toBuilder().build().moeCache());

        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH, "off");
        assertEquals(0, EngineConfig.fromProperties(p).moeCache().prefetchZones());
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH, "0");
        assertEquals(0, EngineConfig.fromProperties(p).moeCache().prefetchZones());
    }

    @Test
    public void thePrefetchZoneCountIsKeptByTheOtherSetters() {
        MoeCacheConfig cfg = MoeCacheConfig.parse("512").withPrefetchZones(3).withRanking(rankingAt("/tmp/r.bin"));
        assertEquals(3, cfg.prefetchZones());
        assertEquals(512L, cfg.budgetMiB());
        assertTrue(cfg.toString().contains("moeCachePrefetch=3"));
        assertFalse(cfg.equals(cfg.withPrefetchZones(1)));
        assertEquals(3, cfg.withRanking(MoeCacheRanking.Options.auto()).prefetchZones());
        assertTrue(cfg.toString(), cfg.toString().contains("ranking=/tmp/r.bin"));
    }

    @Test
    public void refusesAnInvalidPrefetchZoneCountByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeCacheConfig.parsePrefetchZones("-1"));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH));
        e = assertThrows(IllegalArgumentException.class, () -> MoeCacheConfig.parsePrefetchZones("many"));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH));
        e = assertThrows(IllegalArgumentException.class, () -> MoeCacheConfig.off().withPrefetchZones(-2));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH));
        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH, "two");
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
    }

    @Test
    public void theHostedRowBoundDefaultsToZeroAndIsRefusedWhenNegative() {
        assertEquals(0, MoeCacheConfig.parse("on").hostedMaxRows());
        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_HOSTED_ROWS, "8191");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(8191, cfg.moeCache().hostedMaxRows());
        assertTrue(cfg.toString().contains("moeCacheHostedRows=8191"));
        assertEquals(8191, cfg.moeCache().withPrefetchZones(1).withRanking(rankingAt("/r")).hostedMaxRows());

        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_HOSTED_ROWS, "-1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_HOSTED_ROWS));
        e = assertThrows(IllegalArgumentException.class, () -> MoeCacheConfig.parseHostedMaxRows("lots"));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_HOSTED_ROWS));
    }

    @Test
    public void theBudgetSettersKeepTheRankingAndThePrefetchSettings() {
        MoeCacheConfig base = MoeCacheConfig.off().withPrefetchZones(3).withHostedMaxRows(64)
                .withRanking(rankingAt("/tmp/r.bin"));
        MoeCacheConfig on = base.onWithBudgetMiB(256);
        assertEquals(MoeCacheConfig.Mode.ON, on.mode());
        assertEquals(256L, on.budgetMiB());
        assertEquals(3, on.prefetchZones());
        assertEquals(64, on.hostedMaxRows());
        assertEquals(rankingAt("/tmp/r.bin"), on.ranking());
        assertEquals(MoeCacheConfig.parse("256"), MoeCacheConfig.off().onWithBudgetMiB(256));

        MoeCacheConfig budget = MoeCacheConfig.parse("on").withPrefetchZones(0).withHostedMaxRows(8)
                .withBudgetMiB(128);
        assertEquals(128L, budget.budgetMiB());
        assertEquals(0, budget.prefetchZones());
        assertEquals(8, budget.hostedMaxRows());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> base.onWithBudgetMiB(-1));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE));
    }

    private static MoeCacheRanking.Options rankingAt(String path) {
        return new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.PATH, Paths.get(path), null);
    }

    @Test
    public void theTailSkipRankIsOffByDefaultAndReadsThroughEngineConfig() {
        assertEquals(MoeTailSkip.OFF, MoeCacheConfig.off().tailSkipRank());
        assertEquals(MoeTailSkip.OFF, MoeCacheConfig.parse("on").tailSkipRank());

        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP, "7");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(7, cfg.moeCache().tailSkipRank());
        assertTrue(cfg.toString(), cfg.toString().contains("moeCacheTailSkip=7"));
        assertEquals(cfg.moeCache(), cfg.toBuilder().build().moeCache());

        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP, "off");
        assertEquals(0, EngineConfig.fromProperties(p).moeCache().tailSkipRank());
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP, "0");
        assertEquals(0, EngineConfig.fromProperties(p).moeCache().tailSkipRank());

        MoeCacheConfig ranked = cfg.moeCache().withPrefetchZones(1).withHostedMaxRows(2)
                .withRanking(rankingAt("/tmp/r.bin"));
        assertEquals("the other setters keep the rank", 7, ranked.tailSkipRank());
        assertEquals(7, cfg.moeCache().withBudgetMiB(64).tailSkipRank());
        assertEquals(7, cfg.moeCache().withTailSkipRank(3).withRanking(MoeCacheRanking.Options.auto())
                .withTailSkipRank(7).tailSkipRank());
    }

    @Test
    public void refusesAnInvalidTailSkipRankByName() {
        for (String bad : new String[]{"10", "-1", "seven"}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> MoeCacheConfig.parseTailSkipRank(bad));
            assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP));
        }
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeCacheConfig.off().withTailSkipRank(MoeCacheConfig.TAIL_SKIP_MAX_RANK + 1));
        assertTrue(e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP));

        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP, "10");
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
    }

    @Test
    public void theTailSkipRankNeedsTheCache() {
        Properties p = new Properties();
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP, "7");
        EngineConfig off = EngineConfig.fromProperties(p);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> HostPlacement.moeCacheSettings(off));
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP));

        p.setProperty(EngineConfig.PROP_MOE_CACHE_MB, "64");
        MoeCacheConfig settings = HostPlacement.moeCacheSettings(EngineConfig.fromProperties(p));
        assertTrue("the MiB budget enables the cache", settings.enabled());
        assertEquals(7, settings.tailSkipRank());
    }
}
