package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
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
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool split} against llama.cpp's {@code gguf-split --split}: the llama.cpp shard names derived
 * from the output, the {@code split.*} keys and metadata placement its loader reads back, the 128-tensor
 * default limit, the byte limit cutting before the tensor that would overflow it, {@code --no-tensor-first-split},
 * {@code --dry-run}, the refusals naming their flags, and the re-split of a split's first file, which the
 * writer merges before it cuts again.
 */
public class SplitCommandTest {

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

    private Path out(String name) {
        return tmp.getRoot().toPath().resolve(name);
    }

    private static List<Path> split(Object... args) throws Exception {
        String[] a = new String[args.length];
        for (int i = 0; i < args.length; i++) {
            a[i] = args[i].toString();
        }
        return SplitCommand.split(a);
    }

    private static SplitCommand.UsageException usage(Object... args) {
        return assertThrows(SplitCommand.UsageException.class, () -> split(args));
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

    /** The llama.cpp shard name of the output base {@code prefix}. */
    private static String shard(String prefix, int oneBased, int count) {
        return String.format(Locale.ROOT, "%s-%05d-of-%05d.gguf", prefix, oneBased, count);
    }

    // ================================================================== split output

    @Test
    public void tensorLimitSplitWritesLlamaShardNamesAndRoundTrips() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("sp.gguf");
        List<Path> written = split(in, out, "--split-max-tensors", "2");
        assertEquals(Arrays.asList(out.resolveSibling(shard("sp", 1, 3)),
                out.resolveSibling(shard("sp", 2, 3)), out.resolveSibling(shard("sp", 3, 3))), written);

        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertTrue(p.isMultiShard());
            assertEquals(TENSORS, p.getTensors().size());
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(3, shards.size());
            assertEquals(2, shards.get(0).tensorCount());
            assertEquals(2, shards.get(1).firstTensor());
            assertEquals(1, shards.get(2).tensorCount());
            assertEquals(4, shards.get(2).firstTensor());
            // the split.* keys of shard 0, over the full metadata
            assertEquals(GGUFValueType.UINT16, p.getValueType(GgufKeys.SPLIT_NO));
            assertEquals(0, p.getInt(GgufKeys.SPLIT_NO).intValue());
            assertEquals(3, p.getInt(GgufKeys.SPLIT_COUNT).intValue());
            assertEquals(TENSORS, p.getInt(GgufKeys.SPLIT_TENSORS_COUNT).intValue());
            assertEquals("m", p.getString(GgufKeys.GENERAL_NAME));
            for (int i = 0; i < TENSORS; i++) {
                assertTrue("tensor t" + i + " drifted", Arrays.equals(ramp(i),
                        p.readTensorDataAsFloat(p.getTensor("t" + i))));
            }
        }
    }

    @Test
    public void theTensorLimitDefaultsTo128SoASmallTableWritesTheOutputItself() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("whole.gguf");
        assertEquals(List.of(out), split(in, out));
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertFalse(p.isMultiShard());
            assertEquals(TENSORS, p.getTensors().size());
            assertFalse(p.getMetadata().containsKey(GgufKeys.SPLIT_COUNT));
        }
        assertEquals(0, tmp.getRoot().listFiles((d, n) -> n.startsWith("whole-")).length);
    }

    @Test
    public void byteLimitSplitsBeforeTheTensorThatWouldOverflow() throws Exception {
        // three 40-byte tensors fit 128 bytes at alignment 32 (t1 ends at 104); t2 would end at 168
        Path in = inputGguf("in.gguf");
        Path out = out("sp.gguf");
        List<Path> written = split(in, out, "--split-max-size", "128");
        assertEquals(Arrays.asList(out.resolveSibling(shard("sp", 1, 3)),
                out.resolveSibling(shard("sp", 2, 3)), out.resolveSibling(shard("sp", 3, 3))), written);
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(2, shards.get(0).tensorCount());
            assertEquals(2, shards.get(1).tensorCount());
            assertEquals(1, shards.get(2).tensorCount());
        }
    }

    @Test
    public void aTensorLargerThanTheSizeLimitIsRefusedByNameAndWritesNothing() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("sp.gguf");
        IOException e = assertThrows(IOException.class, () -> split(in, out, "--split-max-size", "32"));
        assertMentions(e, "t0", "no split can produce a smaller shard");
        assertEquals(1, tmp.getRoot().listFiles().length);
    }

    @Test
    public void noTensorFirstSplitPutsTheMetadataAloneInTheFirstShard() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("nt.gguf");
        List<Path> written = split(in, out, "--no-tensor-first-split");
        assertEquals(Arrays.asList(out.resolveSibling(shard("nt", 1, 2)),
                out.resolveSibling(shard("nt", 2, 2))), written);
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            // the first shard's own header: the metadata and the split keys, no tensors
            assertEquals("m", p.getMetadata().get(GgufKeys.GENERAL_NAME));
            assertEquals(0, p.getInt(GgufKeys.SPLIT_NO).intValue());
            assertEquals(2, p.getInt(GgufKeys.SPLIT_COUNT).intValue());
            assertEquals(TENSORS, p.getInt(GgufKeys.SPLIT_TENSORS_COUNT).intValue());
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(0, shards.get(0).tensorCount());
            assertEquals(TENSORS, shards.get(1).tensorCount());
            assertEquals(0, shards.get(1).firstTensor());
            assertArrayEquals(ramp(2), p.readTensorDataAsFloat(p.getTensor("t2")), 0f);
        }
    }

    @Test
    public void splittingASplitRejoinsItBeforeItCutsAgain() throws Exception {
        Path in = inputGguf("in.gguf");
        List<Path> shards = split(in, out("sp.gguf"), "--split-max-tensors", "3");
        assertEquals(2, shards.size());
        Path out = out("rs.gguf");
        List<Path> written = split(shards.get(0), out, "--split", "--split-max-tensors", "2");
        assertEquals(Arrays.asList(out.resolveSibling(shard("rs", 1, 3)),
                out.resolveSibling(shard("rs", 2, 3)), out.resolveSibling(shard("rs", 3, 3))), written);
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertEquals(TENSORS, p.getTensors().size());
            assertEquals(Integer.valueOf(3), p.getInt(GgufKeys.SPLIT_COUNT));
            for (int i = 0; i < TENSORS; i++) {
                assertArrayEquals(ramp(i), p.readTensorDataAsFloat(p.getTensor("t" + i)), 0f);
            }
        }
    }

    // ================================================================== --dry-run

    @Test
    public void dryRunPrintsThePlanAndWritesNothing() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("dry.gguf");
        String printed = printed(() -> split(in, out, "--dry-run", "--split-max-tensors", "2"));
        assertMentions(new Exception(printed), "dry run, not writing files",
                shard("dry", 1, 3), shard("dry", 3, 3), "n_tensors = 2", "n_tensors = 1", "total_size = ");
        assertFalse(Files.exists(out));
        assertFalse(Files.exists(out.resolveSibling(shard("dry", 1, 3))));
        assertEquals(1, tmp.getRoot().listFiles().length);
    }

    // ================================================================== overwrite and named refusals

    @Test
    public void anExistingOutputIsRefusedWithoutOverwriteAndReplacedWithIt() throws Exception {
        Path in = inputGguf("in.gguf");
        Path out = out("sp.gguf");
        List<Path> written = split(in, out, "--split-max-tensors", "3");
        IOException e = assertThrows(IOException.class, () -> split(in, out, "--split-max-tensors", "3"));
        assertMentions(e, "already exists", "overwrite");
        assertEquals(written, split(in, out, "--split-max-tensors", "3", "--overwrite"));
    }

    @Test
    public void theSameInputAndOutputIsRefused() throws Exception {
        Path in = inputGguf("in.gguf");
        assertMentions(usage(in, in), "is the input", "name another output");
    }

    @Test
    public void missingFilesAndArgumentsKeepTheirNamedErrors() throws Exception {
        Path in = inputGguf("in.gguf");
        assertMentions(usage(), "split needs an input path and an output path");
        assertMentions(usage(in), "split needs an output path after");
        assertMentions(usage(in, out("a.gguf"), out("b.gguf")), "too many file arguments");
        assertMentions(usage(in, out("a.gguf"), "--frobnicate"), "unknown option --frobnicate");
        assertMentions(usage(in, out("a.gguf"), "--split-max-tensors"), "--split-max-tensors needs a value");
        assertMentions(usage(in, out("a.gguf"), "--split-max-tensors", "-2"),
                "--split-max-tensors needs a non-negative integer");
        assertMentions(usage(in, out("a.gguf"), "--split-max-size", "2000x"), "--split-max-size",
                "needs a size in bytes");
        assertMentions(usage(in, out("a.gguf"), "--merge"), "--merge is not an option of nftool split",
                "nftool merge");
        assertThrows(FileNotFoundException.class, () -> split(out("missing.gguf"), out("a.gguf")));
    }

    @Test
    public void theTwoLimitsAreRefusedTogether() throws Exception {
        Path in = inputGguf("in.gguf");
        assertMentions(usage(in, out("a.gguf"), "--split-max-tensors", "2", "--split-max-size", "1024"),
                "either --split-max-tensors or --split-max-size", "not both");
        assertFalse(Files.exists(out("a.gguf")));
    }

    @Test
    public void helpPrintsEveryFlag() throws Exception {
        String printed = printed(() -> split("--help"));
        for (String flag : new String[]{"--split", "--split-max-tensors", "--split-max-size",
                "--no-tensor-first-split", "--dry-run", "--overwrite", "GGUF_IN GGUF_OUT"}) {
            assertTrue(flag, printed.contains(flag));
        }
        assertNull(split("--help"));
    }
}
