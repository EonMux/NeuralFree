package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The hub cache listing behind {@code --cache-list} over a cache built in a temporary directory: one
 * entry per {@code repository:tag} pair, the first shard standing for its split, sidecars and
 * untagged model files left out, and llama.cpp's count line and {@code <n>. repo:tag} lines. No
 * network is touched: the fixture writes the cache layout the downloader writes.
 */
public class CacheListingTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** Writes one snapshot file into the cache through the layout's own paths. */
    private void writeSnapshot(HubCache cache, HubRef ref, String commit, String repoPath)
            throws IOException {
        Path snapshot = cache.snapshotPath(ref, commit, repoPath);
        Files.createDirectories(snapshot.getParent());
        Files.write(snapshot, new byte[]{1});
    }

    private static String repeat(String s, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) {
            sb.append(s);
        }
        return sb.toString();
    }

    @Test
    public void theListingKeepsOneEntryPerRepositoryAndTag() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("ab", 20);
        HubRef a = HubRef.parse("user/model-a");
        cache.writeRef(a, "main", commit);
        writeSnapshot(cache, a, commit, "Model-Q4_K_M.gguf");
        HubRef b = HubRef.parse("user/model-b");
        cache.writeRef(b, "main", commit);
        writeSnapshot(cache, b, commit, "Other-F16.gguf");

        List<CacheListing.CachedModel> models = CacheListing.models(cache);
        assertEquals(2, models.size());
        assertTrue(models.toString(), models.stream().anyMatch(
                m -> m.repo().equals("user/model-a") && m.tag().equals("Q4_K_M")));
        assertTrue(models.toString(), models.stream().anyMatch(
                m -> m.repo().equals("user/model-b") && m.tag().equals("F16")));
    }

    @Test
    public void theFirstShardStandsForItsSplit() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("cd", 20);
        HubRef ref = HubRef.parse("user/model");
        cache.writeRef(ref, "main", commit);
        writeSnapshot(cache, ref, commit, "Model-Q8_0-00001-of-00002.gguf");
        writeSnapshot(cache, ref, commit, "Model-Q8_0-00002-of-00002.gguf");

        List<CacheListing.CachedModel> models = CacheListing.models(cache);
        assertEquals(1, models.size());
        assertEquals("user/model:Q8_0", models.get(0).toString());
    }

    @Test
    public void sidecarsAndUntaggedModelFilesAreNotModels() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("ef", 20);
        HubRef ref = HubRef.parse("user/model");
        cache.writeRef(ref, "main", commit);
        writeSnapshot(cache, ref, commit, "Model-mmproj-Q8_0.gguf");
        writeSnapshot(cache, ref, commit, "Model-mtp-Q8_0.gguf");
        writeSnapshot(cache, ref, commit, "Model-imatrix-Q8_0.gguf");
        writeSnapshot(cache, ref, commit, "Model.gguf");
        writeSnapshot(cache, ref, commit, "tokenizer.json");

        assertTrue(CacheListing.models(cache).isEmpty());
    }

    @Test
    public void aTagRepeatedAcrossRepositoriesListsBoth() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("12", 20);
        for (String repo : new String[]{"user/one", "user/two"}) {
            HubRef ref = HubRef.parse(repo);
            cache.writeRef(ref, "main", commit);
            writeSnapshot(cache, ref, commit, "Model-Q4_K_M.gguf");
        }
        List<CacheListing.CachedModel> models = CacheListing.models(cache);
        assertEquals(2, models.size());
    }

    @Test
    public void anEmptyOrAbsentCacheListsNothing() throws IOException {
        assertTrue(CacheListing.models(new HubCache(tmp.newFolder("empty").toPath())).isEmpty());
        assertTrue(CacheListing.models(new HubCache(tmp.getRoot().toPath().resolve("absent"))).isEmpty());
    }

    @Test
    public void theLinesUseLlamaCppCountAndEntryForm() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("34", 20);
        HubRef ref = HubRef.parse("user/model");
        cache.writeRef(ref, "main", commit);
        writeSnapshot(cache, ref, commit, "Model-Q4_K_M.gguf");

        String lines = CacheListing.lines(CacheListing.models(cache));
        assertTrue(lines, lines.startsWith("number of models in cache: 1\n"));
        assertTrue(lines, lines.contains("   1. user/model:Q4_K_M\n"));
    }

    @Test
    public void thePrintWritesTheLines() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat("56", 20);
        HubRef ref = HubRef.parse("user/model");
        cache.writeRef(ref, "main", commit);
        writeSnapshot(cache, ref, commit, "Model-Q4_K_M.gguf");

        StringBuilder out = new StringBuilder();
        CacheListing.print(out, cache);
        assertTrue(out.toString(), out.toString().contains("user/model:Q4_K_M"));
    }
}
