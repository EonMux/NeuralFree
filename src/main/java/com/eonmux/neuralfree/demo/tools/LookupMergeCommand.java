package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code nftool lookup-merge}: llama.cpp's {@code examples/lookup/lookup-merge.cpp}. Every cache file but the last
 * loads in the {@code common_ngram_cache_save} layout and merges into the first ({@link NgramCache#merge}), and the
 * merged cache saves to the last path ({@link NgramCacheFile#save}). With two paths the first file is rewritten,
 * accepting the same counts, as the reference does.
 */
final class LookupMergeCommand {

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool lookup-merge <cache1> <cache2> ... <out>\n"
            + "Merges lookup cache files into the last path, the counts of equal n-grams and tokens summed.\n"
            + "With one input path and <out>, the input is rewritten with the same counts.\n"
            + "  --help, -h    this text\n";

    private LookupMergeCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code lookup-merge}. */
    static void run(String[] args) {
        try {
            Path out = execute(args);
            if (out != null) {
                System.out.println("nftool lookup-merge: wrote " + out);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the paths and merges the caches; {@code null} for {@code --help}, which prints the usage.
     *
     * @throws IOException when a cache file cannot be read or the output cannot be written
     */
    static Path execute(String[] args) throws IOException, UsageException {
        List<Path> paths = new ArrayList<>();
        for (String a : args) {
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            }
            paths.add(Paths.get(a));
        }
        if (paths.size() < 2) {
            throw new UsageException("lookup-merge needs at least one input cache and an output path");
        }
        Path out = paths.get(paths.size() - 1);
        System.err.println("lookup-merge: loading file " + paths.get(0));
        NgramCache merged = NgramCacheFile.load(paths.get(0));
        for (int i = 1; i < paths.size() - 1; i++) {
            System.err.println("lookup-merge: loading file " + paths.get(i));
            merged.merge(NgramCacheFile.load(paths.get(i)));
        }
        System.err.println("lookup-merge: saving file " + out);
        NgramCacheFile.save(merged, out);
        System.err.println("lookup-merge: " + merged.ngrams() + " n-grams written to " + out);
        return out;
    }
}
