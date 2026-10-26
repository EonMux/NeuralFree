package com.eonmux.neuralfree.gguf.hub;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HubRef} parsing, the filename and format parsing of {@link HubCache.GgufFilename}, and the
 * path confinement of {@link HubCache} (PARITY.md WP TL-7): the {@code user/repo[:quant][@revision]}
 * grammar and its rejections, the llama.cpp GGUF filename grammar
 * ({@code <prefix>[-<tag>][-NNNNN-of-NNNNN].gguf}) and the model/sidecar split, the snapshot paths an
 * untrusted repository path may take ({@code ..} and absolute paths rejected), the cache's ref,
 * commit and listing round trip over the Hub layout, and the oids the cache accepts. No network
 * starts here: the download client is never constructed in a test, TL-7 wires it and the plan runs it
 * nowhere.
 */
public class HubRefTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== HubRef parsing

    @Test
    public void parsesRepoQuantAndRevision() {
        HubRef r = HubRef.parse("ggml-org/gemma-3-GGUF");
        assertEquals("ggml-org/gemma-3-GGUF", r.repository());
        assertEquals("ggml-org", r.user());
        assertEquals("gemma-3-GGUF", r.model());
        assertNull(r.quant());
        assertNull(r.revision());
        assertEquals("ggml-org/gemma-3-GGUF", r.toString());

        HubRef tagged = HubRef.parse("ggml-org/gemma-3-GGUF:Q4_K_M");
        assertEquals("Q4_K_M", tagged.quant());
        assertEquals("ggml-org/gemma-3-GGUF:Q4_K_M", tagged.toString());

        assertEquals("main", HubRef.parse("ggml-org/gemma-3-GGUF@main").revision());
        HubRef both = HubRef.parse("ggml-org/gemma-3-GGUF:Q4_K_M@abc123");
        assertEquals("Q4_K_M", both.quant());
        assertEquals("abc123", both.revision());
    }

    @Test
    public void normalizesWhitespaceAndKeepsCase() {
        HubRef r = HubRef.parse(" The-org/Model.2:j6q9 ");
        assertEquals("The-org/Model.2", r.repository());
        assertEquals("j6q9", r.quant());
        assertEquals("The-org/Model.2:j6q9", r.toString());
        assertEquals("models--The-org--Model.2", r.folderName());
    }

    @Test
    public void rejectsMissingUserAndWrongSegmentCounts() {
        assertInvalid("");
        assertInvalid("gemma");
        assertInvalid("ggml-org/gemma-3-GGUF/extra");
        assertInvalid("/ggml-org");
        assertInvalid("ggml-org/");
    }

    @Test
    public void rejectsSpecialCharEdgesOfTheRepositoryId() {
        assertInvalid("ggml-.org/model");
        assertInvalid("ggml-org./model");
        assertInvalid("ggml-org/model.");
        assertInvalid("ggml-org/model//tail");
        assertInvalid("ggml-org/model w spaces");
        assertInvalid("ggml-org/model'quote");
        assertInvalid("a/" + repeat('b', 300));
    }

    @Test
    public void rejectsEmptyAndIllformedQuantAndRevision() {
        assertInvalid("user/model:");
        assertInvalid("user/model:@");
        assertInvalid("user/model:q4 k m");
        assertInvalid("user/model@main/../../x");
        assertInvalid("user/model@. ");
    }

    /** {@link HubRef#parse} raising with a message that names the hub grammar or the broken part. */
    private static void assertInvalid(String spec) {
        try {
            HubRef.parse(spec);
            fail("parse should reject '" + spec + "'");
        } catch (IllegalArgumentException expected) {
            assertTrue("the message should name the hub expression: " + expected.getMessage(),
                    expected.getMessage().contains("invalid HF repo"));
        }
    }

    private static String repeat(char c, int n) {
        return repeat(String.valueOf(c), n);
    }

    /** {@code s} repeated {@code n} times, the test's own hex generators. */
    private static String repeat(String s, int n) {
        StringBuilder out = new StringBuilder(s.length() * n);
        for (int i = 0; i < n; i++) {
            out.append(s);
        }
        return out.toString();
    }

    // ================================================================== filename and format parsing

    /** The filename parts of one repo path, llama.cpp's get_gguf_split_info surface. */
    private static HubCache.GgufFilename name(String path) {
        return HubCache.GgufFilename.parse(path);
    }

    @Test
    public void parsesPlainNamesTagsAndSplitSuffixes() {
        HubCache.GgufFilename f = name("Model-Q4_K_M.gguf");
        assertEquals("Model-Q4_K_M", f.prefix());
        assertEquals("Q4_K_M", f.tag());
        assertEquals(1, f.index());
        assertEquals(1, f.count());
        assertFalse(f.isShard());

        HubCache.GgufFilename split = name("sub/dir/Model-Q4_K_M-00002-of-00003.gguf");
        assertEquals("sub/dir/Model-Q4_K_M", split.prefix());
        assertEquals("Q4_K_M", split.tag());
        assertEquals(2, split.index());
        assertEquals(3, split.count());
        assertTrue(split.isShard());

        HubCache.GgufFilename dot = name("model.f16.gguf");
        assertEquals("model.f16", dot.prefix());
        assertEquals("F16", dot.tag());

        HubCache.GgufFilename lacks = name("Model.gguf");
        assertEquals("Model", lacks.prefix());
        assertEquals("", lacks.tag());
    }

    /** {@code Q4_0}→4, {@code F16}→16, {@code NVFP4}→4, {@code Q8_K_M}→8, no digit→0 (llama.cpp extract_quant_bits). */
    @Test
    public void quantBitsReadTheFirstNumberOfTheTag() {
        assertEquals(4, name("m-Q4_0.gguf").quantBits());
        assertEquals(16, name("m-F16.gguf").quantBits());
        assertEquals(4, name("m-NVFP4.gguf").quantBits());
        assertEquals(8, name("m-Q8_K_M.gguf").quantBits());
        assertEquals(0, name("m-Line.gguf").quantBits());
        assertEquals(0, name("m.gguf").quantBits());
    }

    @Test
    public void nonGgufPathsHaveNoFilenameParts() {
        assertNull(name("model.safetensors"));
        assertNull(name("Model.Q4_K_M"));
        assertNull(name(""));
    }

    @Test
    public void modelFilesSplitFromSidecarsAndImatrix() {
        assertTrue(HubCache.isModel("Model-Q4_K_M.gguf"));
        assertTrue(HubCache.isModel("sub/Model-Q4_K_M-00001-of-00003.gguf"));
        assertFalse(HubCache.isModel("imatrix-Model-Q4_K_M.gguf"));
        assertFalse(HubCache.isModel("Model.imatrix.gguf"));
        assertFalse(HubCache.isModel("mmproj-Model-F16.gguf"));
        assertFalse(HubCache.isModel("sub/mmproj-model-f16.gguf"));
        assertFalse(HubCache.isModel("mtp-Model-Q4_K_M.gguf"));
        assertFalse(HubCache.isModel("dflash-Model-Q8_0.gguf"));
        assertFalse(HubCache.isModel("eagle3-Model-Q8_0.gguf"));
        assertFalse(HubCache.isModel("model.txt"));
    }

    // ================================================================== path confinement

    @Test
    public void snapshotPathsStayInsideTheSnapshotTree() throws IOException {
        HubRef ref = HubRef.parse("user/model");
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        String commit = repeat('a', 40);
        Path snapshot = cache.snapshotPath(ref, commit, "sub/Model-Q4_K_M.gguf");
        assertTrue(snapshot.toString(),
                snapshot.endsWith("models--user--model/snapshots/" + commit
                        + "/sub/Model-Q4_K_M.gguf"));
    }

    @Test
    public void parentAbsoluteAndEmptyRepositoryPathsAreRejected() throws IOException {
        String commit = repeat('a', 40);
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, "../escape.gguf"));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, "a/../b.gguf"));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, "sub/../../b.gguf"));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, "/etc/passwd"));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, ""));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), commit, "./Model.gguf"));
    }

    @Test
    public void snapshotCommitsAreKeyedOnFortyHexCharacters() throws IOException {
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), "not-hex", "Model.gguf"));
        assertThrows(IllegalArgumentException.class,
                () -> cache.snapshotPath(HubRef.parse("user/model"), repeat('a', 39), "Model.gguf"));
    }

    @Test
    public void refsAndOidsObeyTheSameHexRules() throws IOException {
        HubRef ref = HubRef.parse("user/model");
        HubCache cache = new HubCache(tmp.newFolder("hub").toPath());
        assertThrows(IllegalArgumentException.class, () -> cache.writeRef(ref, "main", repeat('z', 40)));
        assertThrows(IllegalArgumentException.class, () -> cache.writeRef(ref, "m/ain", repeat('a', 40)));
        assertThrows(IllegalArgumentException.class, () -> cache.writeRef(ref, "main", repeat('a', 9)));
        assertThrows(IllegalArgumentException.class, () -> cache.blobPath(ref, repeat('e', 31)));
        assertTrue(HubCache.isValidOid(repeat("ab", 32), 64));
        assertFalse(HubCache.isValidOid(repeat("gh", 32), 64));
        assertFalse(HubCache.isValidOid(repeat('a', 41), 64));
        assertTrue(HubCache.isSha256Oid(repeat("ab", 32)));
        assertFalse(HubCache.isSha256Oid(repeat('a', 40)));
        // a commit the cache stores as written hex comes back lowercase, the LFS pointer form
        Path snapshot = cache.snapshotPath(ref, repeat('C', 40), "Model.gguf");
        assertEquals(repeat('c', 40), snapshot.getParent().getFileName().toString());
    }

    // ================================================================== the cache round trip

    /** The cached listing decodes the repository folder, resolves main's commit and sees the snapshot files. */
    @Test
    public void cachedFilesListTheSnapshotThroughMain() throws IOException {
        Path root = tmp.newFolder("hub").toPath();
        HubCache cache = new HubCache(root);
        HubRef ref = HubRef.parse("user/model");
        String commit = repeat("ab", 20);
        cache.writeRef(ref, "main", commit);
        Path snapshot = cache.snapshotPath(ref, commit, "sub/Model-Q8_0.gguf");
        Files.createDirectories(snapshot.getParent());
        Files.write(snapshot, new byte[]{1, 2, 3});
        assertEquals(commit, cache.cachedCommit(ref));
        List<HubCache.CachedFile> files = cache.cachedFiles("user/model");
        assertEquals(1, files.size());
        assertEquals("user/model", files.get(0).repoId());
        assertEquals("sub/Model-Q8_0.gguf", files.get(0).repoPath());
        assertEquals("Q8_0", files.get(0).filename().tag());
        assertEquals(snapshot, files.get(0).localPath());
        assertTrue(cache.cachedFiles("other/model").isEmpty());
        assertTrue(new HubCache(tmp.newFolder("empty").toPath()).cachedFiles(null).isEmpty());
    }

    /** A snapshot keyed on a revision that is itself the commit resolves through the cache unchanged. */
    @Test
    public void commitRevisionsResolveThroughTheCachedLayout() throws IOException {
        Path root = tmp.newFolder("hub").toPath();
        HubCache cache = new HubCache(root);
        String commit = repeat("cd", 20);
        HubRef ref = HubRef.parse("user/model@" + commit);
        assertEquals(commit, cache.cachedCommit(ref));
        assertFalse(Files.isDirectory(cache.repoPath(ref).resolve("refs")));
    }

    /** A repository without a main ref lists its first sorted ref's snapshot (llama.cpp get_cached_ref). */
    @Test
    public void cachedCommitFallsBackToTheFirstRef() throws IOException {
        Path root = tmp.newFolder("hub").toPath();
        HubCache cache = new HubCache(root);
        String commit = repeat("ef", 20);
        cache.writeRef(HubRef.parse("user/model"), "develop", commit);
        assertEquals(commit, cache.cachedCommit(HubRef.parse("user/model")));
        Path snapshot = cache.snapshotPath(HubRef.parse("user/model"), commit, "Model.gguf");
        Files.createDirectories(snapshot.getParent());
        Files.write(snapshot, new byte[]{9});
        List<HubCache.CachedFile> files = cache.cachedFiles("user/model");
        assertEquals(1, files.size());
        assertEquals("Model.gguf", files.get(0).repoPath());
    }

    /** The blob path of a valid LFS oid lands under blobs of the repository directory. */
    @Test
    public void blobPathsUseTheValidatedOid() throws IOException {
        Path root = tmp.newFolder("hub").toPath();
        HubCache cache = new HubCache(root);
        Path blob = cache.blobPath(HubRef.parse("user/model"), repeat("ef", 32));
        assertTrue(blob.toString(), blob.endsWith("models--user--model/blobs/" + repeat("ef", 32)));
    }
}
