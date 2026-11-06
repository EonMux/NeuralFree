package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool merge} against llama.cpp's {@code gguf-split --merge}: the shards of one split join into a
 * single GGUF that carries the split's metadata without the {@code split.*} keys, every tensor round trips,
 * the merged output refuses to merge again, {@code --dry-run} prints the plan and writes nothing, and
 * {@code --delete-splits} removes every shard once the merged file is complete.
 */
public class MergeCommandTest {

    private static final int TENSORS = 5;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== fixtures

    /** A GGUF of {@value #TENSORS} F32 tensors of ten floats each, ramping from their index. */
    private Path inputGguf(String name) throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str(GgufKeys.GENERAL_NAME, "m");
        for (int i = 0; i < TENSORS; i++) {
            w.tensorF32("t" + i, new long[]{10}, ramp(i));
        }
        Path p = tmp.getRoot().toPath().resolve(name);
        w.writeTo(p);
        return p;
    }

    private static float[] ramp(int start) {
        float[] v = new float[10];
        for (int j = 0; j < v.length; j++) {
            v[j] = start + 0.25f * j;
        }
        return v;
    }

    /** The input split into three shards; the first shard's path is the merge input. */
    private List<Path> shards() throws Exception {
        return split(inputGguf("in.gguf"), out("sp.gguf"), "--split-max-tensors", "2");
    }

    private static List<Path> split(Object... args) throws Exception {
        String[] a = new String[args.length];
        for (int i = 0; i < args.length; i++) {
            a[i] = args[i].toString();
        }
        return SplitCommand.split(a);
    }

    private Path out(String name) {
        return tmp.getRoot().toPath().resolve(name);
    }

    private static List<Path> merge(Object... args) throws Exception {
        String[] a = new String[args.length];
        for (int i = 0; i < args.length; i++) {
            a[i] = args[i].toString();
        }
        return MergeCommand.merge(a);
    }

    private static MergeCommand.UsageException usage(Object... args) {
        return assertThrows(MergeCommand.UsageException.class, () -> merge(args));
    }

    private static void assertMentions(Exception e, String... parts) {
        for (String part : parts) {
            assertTrue("'" + part + "' missing from: " + e.getMessage(), e.getMessage().contains(part));
        }
    }

    /** Runs {@code run} and returns everything it printed on stdout. */
    private static String printed(Callable<List<Path>> run) throws Exception {
        PrintStream old = System.out;
        ByteArrayOutputStream cap = new ByteArrayOutputStream();
        System.setOut(new PrintStream(cap, true, StandardCharsets.UTF_8.name()));
        try {
            run.call();
        } finally {
            System.out.flush();
            System.setOut(old);
        }
        return cap.toString(StandardCharsets.UTF_8.name());
    }

    // ================================================================== merge output

    @Test
    public void mergeJoinsTheShardsIntoOnePlainGguf() throws Exception {
        List<Path> shards = shards();
        Path merged = out("merged.gguf");
        assertEquals(List.of(merged), merge(shards.get(0), merged));
        try (GGUFFileParser p = new GGUFFileParser(merged.toString())) {
            assertFalse(p.isMultiShard());
            assertEquals(TENSORS, p.getTensors().size());
            for (int i = 0; i < TENSORS; i++) {
                assertArrayEquals(ramp(i), p.readTensorDataAsFloat(p.getTensor("t" + i)), 0f);
            }
            assertFalse(p.getMetadata().containsKey(GgufKeys.SPLIT_NO));
            assertFalse(p.getMetadata().containsKey(GgufKeys.SPLIT_COUNT));
            assertFalse(p.getMetadata().containsKey(GgufKeys.SPLIT_TENSORS_COUNT));
            assertEquals("llama", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals("m", p.getString(GgufKeys.GENERAL_NAME));
        }
    }

    @Test
    public void theMergedOutputRefusesToMergeAgain() throws Exception {
        List<Path> shards = shards();
        Path merged = out("merged.gguf");
        merge(shards.get(0), merged);
        assertMentions(usage(merged, out("again.gguf")), "merge needs the first file of a split GGUF",
                "single-file GGUF", "split.count");
    }

    // ================================================================== --dry-run

    @Test
    public void dryRunPrintsThePlanAndWritesNothing() throws Exception {
        List<Path> shards = shards();
        Path merged = out("dry.gguf");
        String printed = printed(() -> merge(shards.get(0), merged, "--dry-run"));
        assertMentions(new Exception(printed), "dry run, not writing files", "3 shards", "n_tensors = 5",
                merged.toString(), "total_size = ");
        assertFalse(Files.exists(merged));
        for (Path shard : shards) {
            assertTrue("the dry run deleted " + shard, Files.exists(shard));
        }
    }

    // ================================================================== --delete-splits

    @Test
    public void deleteSplitsRemovesEveryShardAfterTheMerge() throws Exception {
        List<Path> shards = shards();
        Path merged = out("merged.gguf");
        String printed = printed(() -> merge(shards.get(0), merged, "--delete-splits"));
        for (Path shard : shards) {
            assertFalse("the shard " + shard + " survived", Files.exists(shard));
        }
        assertTrue(printed.contains("deleted"));
        try (GGUFFileParser p = new GGUFFileParser(merged.toString())) {
            assertEquals(TENSORS, p.getTensors().size());
            assertArrayEquals(ramp(4), p.readTensorDataAsFloat(p.getTensor("t4")), 0f);
        }
    }

    // ================================================================== refusals

    @Test
    public void aSingleFileInputIsRefusedByName() throws Exception {
        Path in = inputGguf("in.gguf");
        assertMentions(usage(in, out("m.gguf")), "merge needs the first file of a split GGUF",
                "single-file GGUF", "split.count");
        assertFalse(Files.exists(out("m.gguf")));
    }

    @Test
    public void anOutputThatNamesAShardFileIsRefused() throws Exception {
        List<Path> shards = shards();
        assertMentions(usage(shards.get(0), shards.get(2)), "is one of the merge's shard files",
                "name an output outside the split");
        assertTrue("the refusal removed " + shards.get(2), Files.exists(shards.get(2)));
    }

    @Test
    public void anExistingOutputIsRefusedWithoutOverwriteAndReplacedWithIt() throws Exception {
        List<Path> shards = shards();
        Path merged = out("merged.gguf");
        merge(shards.get(0), merged);
        IOException e = assertThrows(IOException.class, () -> merge(shards.get(0), merged));
        assertMentions(e, "already exists", "overwrite");
        assertEquals(List.of(merged), merge(shards.get(0), merged, "--overwrite"));
        assertEquals(TENSORS, tensorCountOf(merged));
    }

    @Test
    public void missingFilesAndArgumentsKeepTheirNamedErrors() throws Exception {
        Path in = inputGguf("in.gguf");
        assertMentions(usage(), "merge needs an input path and an output path");
        assertMentions(usage(in), "merge needs an output path after");
        assertMentions(usage(in, out("a.gguf"), out("b.gguf")), "too many file arguments");
        assertMentions(usage(in, out("a.gguf"), "--frobnicate"), "unknown option --frobnicate");
        assertMentions(usage(in, out("a.gguf"), "--split"), "--split is not an option of nftool merge",
                "nftool split");
        assertThrows(FileNotFoundException.class, () -> merge(out("missing.gguf"), out("a.gguf")));
    }

    @Test
    public void helpPrintsEveryFlag() throws Exception {
        String printed = printed(() -> merge("--help"));
        for (String flag : new String[]{"--merge", "--dry-run", "--delete-splits", "--overwrite",
                "<prefix>-00001-of-0000N.gguf"}) {
            assertTrue(flag, printed.contains(flag));
        }
        assertNull(merge("--help"));
    }

    // ================================================================== helper

    private int tensorCountOf(Path file) throws IOException {
        try (GGUFFileParser p = new GGUFFileParser(file.toString())) {
            return p.getTensors().size();
        }
    }
}
