package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code hf} dataset's sources: a local path stays local, a Hub repository expression is a
 * dataset repository (its cache folder and API namespace differ from a model's), and a Parquet
 * directory samples through the same schema dispatch as the JSONL files.
 */
public class HfDatasetSourceTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A tokenizer that counts one token per character and carries no chat template. */
    private static final BenchTokenizer CHARS = text -> new int[text.length()];

    @Test
    public void aLocalPathStaysLocal() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("hf")
                .datasetPath(tmp.getRoot().getAbsolutePath()).hfSubset("sub").hfSplit("test").build();
        HfDatasetSource.Source source = HfDatasetSource.resolve(opts);
        assertEquals(tmp.getRoot().toPath(), source.path);
        assertEquals("sub", source.subset);
        assertEquals("test", source.split);
    }

    @Test
    public void aMissingPathThatIsNoRepositoryIsRefused() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("hf")
                .datasetPath("/definitely/not/a/local/file").build();
        try {
            HfDatasetSource.resolve(opts);
            fail("a missing path and no repository id must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("Hub repository"));
        }
    }

    @Test
    public void aDatasetReferenceUsesTheDatasetCacheLayout() throws IOException {
        HubRef ref = HubRef.parseDataset("openai/gsm8k@main");
        assertEquals(HubRef.RepoType.DATASET, ref.type());
        assertEquals("openai/gsm8k", ref.repository());
        assertEquals("datasets--openai--gsm8k", ref.folderName());
        assertEquals("datasets/openai/gsm8k@main", ref.toString());

        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        Path snapshot = cache.snapshotDir(ref, repeat('a', 40));
        assertTrue(snapshot.toString(), snapshot.toString().endsWith(
                "datasets--openai--gsm8k/snapshots/" + repeat('a', 40)));
        assertEquals(snapshot, cache.snapshotPath(ref, repeat('a', 40), "data.parquet").getParent());
    }

    @Test
    public void aDatasetRepositoryRejectsAQuantTag() {
        try {
            HubRef.parseDataset("user/model:Q4_K_M");
            fail("a dataset repository has no quant tag");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("quant"));
        }
    }

    @Test
    public void aParquetDirectorySamplesThroughTheHfSchema() throws IOException {
        Path dir = tmp.newFolder("hf-parquet").toPath();
        Files.copy(Path.of("src", "test", "resources", "parquet", "nested_snappy.parquet"),
                dir.resolve("data.parquet"), StandardCopyOption.REPLACE_EXISTING);
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("hf")
                .datasetPath(dir.toString()).numPrompts(2).seed(1).disableShuffle(true)
                .hfOutputLen(5).requestIdPrefix("h-").build();
        List<BenchSample> samples = new HfDataset(opts, CHARS).sample();
        assertEquals(2, samples.size());
        assertEquals("hi", samples.get(0).text());
        assertEquals(2, samples.get(0).promptLen());
        assertEquals(5, samples.get(0).outputLen());
        assertEquals("yo", samples.get(1).text());
    }

    private static String repeat(char c, int n) {
        StringBuilder out = new StringBuilder(n);
        for (int i = 0; i < n; i++) {
            out.append(c);
        }
        return out.toString();
    }
}
