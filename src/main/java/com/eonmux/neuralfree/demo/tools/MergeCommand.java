package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSources;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code nftool merge}: the GGUF merger of llama.cpp's {@code gguf-split} (tools/gguf-split/gguf-split.cpp,
 * {@code --merge}), joining the shard files of one split into a single GGUF through {@link GgufWriter} in
 * {@link GgufWriter.Mode#NATIVE}. The input is the split's first file
 * ({@code <prefix>-00001-of-0000N.gguf}), whose {@code split.count} names the other shards; the parser
 * ({@link GGUFFileParser#loadShards()}) joins their tensors at virtual offsets and the writer lays them
 * into one file carrying the split's metadata without the {@code split.*} keys, so the merged file is a
 * plain GGUF that no longer merges. Flags first ({@code --dry-run}, {@code --delete-splits},
 * {@code --overwrite}), then the positionals {@code GGUF_IN GGUF_OUT}: {@code --delete-splits} deletes
 * every shard file of the split once the merged file is complete (llama.cpp's disk-space option, with the
 * same warning — a failure there leaves the shards in place and the merge reported), and {@code --dry-run}
 * prints the one file and its tensor count and bytes without writing anything.
 */
final class MergeCommand {

    private static final String USAGE =
            "nftool merge [--merge] [--dry-run] [--delete-splits] [--overwrite] GGUF_IN GGUF_OUT\n"
            + "options:\n"
            + "  --merge                               the operation of this command, as gguf-split names it\n"
            + "  --dry-run                             print the file the merge writes, write nothing\n"
            + "  --delete-splits                       delete the shard files once the merge is complete;\n"
            + "                                        a shard that cannot be deleted is a named error and\n"
            + "                                        every shard stays in place\n"
            + "  --overwrite                           replace an existing output\n"
            + "  --help, -h                            this text\n"
            + "GGUF_IN is the split's first file, named <prefix>-00001-of-0000N.gguf.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The command line {@link #parse} fills; read-only once parsed. */
    private static final class Request {
        Path input;
        Path out;
        boolean dryRun;
        boolean deleteSplits;
        boolean overwrite;
        boolean help;
    }

    private MergeCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code merge}. */
    static void run(String[] args) {
        try {
            List<Path> written = merge(args);
            if (written != null) {
                System.out.println("nftool merge: wrote " + written.get(0));
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags and merges; {@code null} for {@code --help} and {@code --dry-run}, which print and
     * write nothing.
     */
    static List<Path> merge(String[] args) throws IOException, UsageException {
        Request r = parse(args);
        if (r.help) {
            System.out.println(USAGE);
            return null;
        }
        validate(r);
        GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(r.overwrite);
        List<Path> shards;
        try (GGUFFileParser src = ModelSources.parseGguf(r.input.toString())) {
            if (!src.isMultiShard()) {
                throw new UsageException("merge needs the first file of a split GGUF; '" + r.input
                        + "' is a single-file GGUF without split.count metadata");
            }
            shards = shardPaths(src);
            if (namesShard(r.out, shards)) {
                throw new UsageException("the output '" + r.out + "' is one of the merge's shard files;"
                        + " name an output outside the split");
            }
            if (r.dryRun) {
                printPlan(GgufWriter.plan(src, r.out, options), shards.size());
                return null;
            }
            GgufWriter.write(src, r.out, options);
        }
        if (r.deleteSplits) {
            deleteShards(shards);
        }
        return List.of(r.out);
    }

    // ------------------------------------------------------------------ parsing

    /** The command line as a {@link Request}; every flag value checked as it is read. */
    private static Request parse(String[] args) throws UsageException {
        Request r = new Request();
        for (int i = 0; i < args.length && !r.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                r.help = true;
            } else if (a.startsWith("-")) {
                i = flag(r, a, args, i);
            } else if (r.input == null) {
                r.input = Paths.get(a);
            } else if (r.out == null) {
                r.out = Paths.get(a);
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + r.input + " and "
                        + r.out);
            }
        }
        return r;
    }

    /** Applies the flag {@code a} at {@code args[i]}; the index of its last argument. */
    private static int flag(Request r, String a, String[] args, int i) throws UsageException {
        switch (a) {
            case "--merge":
                return i;
            case "--split":
                throw new UsageException("--split is not an option of nftool merge; run 'nftool split"
                        + " <in.gguf> <out.gguf>'");
            case "--dry-run":
                r.dryRun = true;
                return i;
            case "--delete-splits":
                r.deleteSplits = true;
                return i;
            case "--overwrite":
                r.overwrite = true;
                return i;
            default:
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
        }
    }

    /** Refuses a missing positional. */
    private static void validate(Request r) throws UsageException {
        if (r.input == null) {
            throw new UsageException("merge needs an input path and an output path");
        }
        if (r.out == null) {
            throw new UsageException("merge needs an output path after " + r.input);
        }
    }

    // ------------------------------------------------------------------ shard files

    /** The paths of the split's shard files, in split order. */
    private static List<Path> shardPaths(GGUFFileParser src) {
        List<Path> shards = new ArrayList<>(src.getShards().size());
        for (GGUFFileParser.Shard shard : src.getShards()) {
            shards.add(Paths.get(shard.path()));
        }
        return shards;
    }

    /** Whether {@code out} names one of the split's shard files, the merged file must never be. */
    private static boolean namesShard(Path out, List<Path> shards) {
        Path normalized = out.toAbsolutePath().normalize();
        for (Path shard : shards) {
            if (shard.toAbsolutePath().normalize().equals(normalized)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Deletes every shard file of the merge ({@code --delete-splits}); the merged file is complete at
     * this point. Every file is attempted, and the files that could not be deleted are one named error.
     */
    private static void deleteShards(List<Path> shards) throws IOException {
        List<Path> failed = new ArrayList<>();
        for (Path shard : shards) {
            try {
                Files.delete(shard);
            } catch (IOException e) {
                failed.add(shard);
                continue;
            }
            System.out.println("nftool merge: deleted " + shard);
        }
        if (!failed.isEmpty()) {
            throw new IOException("the merge is written, but these shard files could not be deleted: "
                    + failed);
        }
    }

    // ------------------------------------------------------------------ prints

    /** The plan of a {@code --dry-run} merge: the file it would write, with its tensor count and bytes. */
    private static void printPlan(List<GgufWriter.PlannedShard> plan, int shardCount) {
        System.out.println("nftool merge: dry run, not writing files; the merge of " + shardCount
                + " shards writes");
        for (GgufWriter.PlannedShard shard : plan) {
            System.out.println("  " + shard);
        }
    }
}
