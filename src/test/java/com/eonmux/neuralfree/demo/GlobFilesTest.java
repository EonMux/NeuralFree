package com.eonmux.neuralfree.demo;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link GlobMatcher} follows llama.cpp common/common.cpp {@code glob_match} ({@code **} crosses {@code /},
 * {@code *} and {@code ?} do not, classes with ranges, negation and literal leading {@code ]}), and
 * {@link GlobFiles} splits a pattern's directory part off and walks the regular files below it.
 */
public class GlobFilesTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void starStopsAtSlashAndDoubleStarCrossesIt() {
        assertTrue(GlobMatcher.matches("*.txt", "a.txt"));
        assertFalse(GlobMatcher.matches("*.txt", "dir/a.txt"));
        assertTrue(GlobMatcher.matches("**.txt", "dir/sub/a.txt"));
        assertTrue(GlobMatcher.matches("**/*.txt", "dir/a.txt"));
        assertTrue(GlobMatcher.matches("dir/*", "dir/a"));
        assertFalse(GlobMatcher.matches("dir/*", "dir/sub/a"));
    }

    @Test
    public void questionMarkMatchesOneNonSlashCharacter() {
        assertTrue(GlobMatcher.matches("a?c", "abc"));
        assertFalse(GlobMatcher.matches("a?c", "a/c"));
        assertFalse(GlobMatcher.matches("a?c", "ac"));
    }

    @Test
    public void classesTakeRangesNegationAndLiteralBrackets() {
        assertTrue(GlobMatcher.matches("f[a-c].md", "fb.md"));
        assertFalse(GlobMatcher.matches("f[a-c].md", "fd.md"));
        assertTrue(GlobMatcher.matches("f[!a-c].md", "fd.md"));
        assertFalse(GlobMatcher.matches("f[!a-c].md", "fa.md"));
        assertTrue("a leading ] is literal", GlobMatcher.matches("[]x]", "]"));
        assertTrue("a leading - is literal", GlobMatcher.matches("[-x]", "-"));
        assertTrue("an unclosed [ is literal", GlobMatcher.matches("a[b", "a[b"));
        assertFalse("^ is not negation in the reference", GlobMatcher.matches("[^a]", "b"));
    }

    @Test
    public void thePatternDirectoryPartIsResolvedAndTheRestMatched() throws IOException {
        Path root = tmp.getRoot().toPath();
        write(root.resolve("docs/a.md"));
        write(root.resolve("docs/b.txt"));
        write(root.resolve("docs/sub/c.md"));
        write(root.resolve("top.md"));
        GlobFiles glob = GlobFiles.split(root, "docs/*.md");
        assertEquals(root.resolve("docs"), glob.baseDir());
        assertEquals("*.md", glob.pattern());
        assertEquals(List.of(root.resolve("docs/a.md")), glob.matches());
        assertEquals(List.of(root.resolve("docs/a.md"), root.resolve("docs/sub/c.md")),
                GlobFiles.split(root, "docs/**.md").matches());
    }

    @Test
    public void aPatternWithoutDirectoryWalksTheWorkingDirectory() throws IOException {
        Path root = tmp.getRoot().toPath();
        write(root.resolve("x.md"));
        write(root.resolve("d/y.md"));
        List<Path> all = GlobFiles.split(root, "**.md").matches();
        assertEquals(List.of(root.resolve("d/y.md"), root.resolve("x.md")), all);
        assertEquals("no glob character matches the literal relative path", List.of(root.resolve("d/y.md")),
                GlobFiles.split(root, "d/y.md").matches());
    }

    @Test
    public void anAbsoluteDirectoryAndAMissingOneAreHandled() throws IOException {
        Path root = tmp.getRoot().toPath();
        write(root.resolve("abs/q.txt"));
        Path elsewhere = tmp.newFolder("cwd").toPath();
        assertEquals(List.of(root.resolve("abs/q.txt")),
                GlobFiles.split(elsewhere, root.resolve("abs") + "/*.txt").matches());
        assertEquals(new ArrayList<Path>(), GlobFiles.split(root, "missing/*.txt").matches());
    }

    private static void write(Path file) throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, file.getFileName().toString().getBytes(StandardCharsets.UTF_8));
    }
}
