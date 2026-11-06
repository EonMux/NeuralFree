package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * {@code nftool lookup-create}: llama.cpp's {@code examples/lookup/lookup-create.cpp}. The prompt tokenizes with
 * the model's tokenizer, its {@value NgramDrafter#CACHE_STATIC_NGRAM}-grams (llama.cpp's
 * {@code LLAMA_NGRAM_STATIC}) fill an {@link NgramCache}, and the cache saves in the
 * {@code common_ngram_cache_save} layout ({@link NgramCacheFile}) to the {@code -lcs} file the {@code -lcs} /
 * {@code --lookup-cache-static} decoding flag reads.
 */
final class LookupCreateCommand {

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool lookup-create <model> [options]\n"
            + "options (llama.cpp lookup-create names):\n"
            + "  -m, --model <file>                  the model whose tokenizer reads the prompt\n"
            + "  -lcs, --lookup-cache-static <file>  where the static lookup cache is written (required)\n"
            + "  -p, --prompt <text>                 the prompt whose n-grams fill the cache\n"
            + "  -f, --file <fname>                  read the prompt from this file instead\n"
            + "  --help, -h                          this text\n"
            + "Tokenizes the prompt and writes its 2-grams to the static lookup cache file, ready for the\n"
            + "-lcs/--lookup-cache-static decoding flag.\n";

    private LookupCreateCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code lookup-create}. */
    static void run(String[] args) {
        try {
            Path out = execute(args);
            if (out != null) {
                System.out.println("nftool lookup-create: wrote " + out);
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
     * Parses the flags and writes the cache; {@code null} for {@code --help}, which prints the usage.
     *
     * @throws IOException when the model cannot be read or the cache cannot be written
     */
    static Path execute(String[] args) throws IOException, UsageException {
        Path model = null;
        Path out = null;
        String prompt = "";
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            switch (a) {
                case "-m":
                case "--model":
                    model = model(a, model, value(a, args, i++));
                    break;
                case "-lcs":
                case "--lookup-cache-static":
                    out = Paths.get(value(a, args, i++));
                    break;
                case "-p":
                case "--prompt":
                    prompt = value(a, args, i++);
                    break;
                case "-f":
                case "--file": {
                    Path file = Paths.get(value(a, args, i++));
                    if (!Files.isReadable(file)) {
                        throw new UsageException(a + " " + file + " is not a readable file");
                    }
                    prompt = LookupToolSupport.readPrompt(file);
                    break;
                }
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    model = model(a, model, a);
            }
        }
        if (model == null) {
            throw new UsageException("lookup-create needs a model path");
        }
        if (out == null) {
            throw new UsageException("-lcs/--lookup-cache-static needs the output cache file");
        }

        LookupToolSupport.TokenizerModel m = LookupToolSupport.open(model);
        int[] tokens = m.encode(prompt);
        NgramCache cache = new NgramCache();
        cache.addSequence(NgramDrafter.CACHE_STATIC_NGRAM, NgramDrafter.CACHE_STATIC_NGRAM, tokens);
        NgramCacheFile.save(cache, out);
        System.err.println("nftool lookup-create: " + cache.ngrams() + " n-grams of " + tokens.length
                + " tokens written to " + out);
        return out;
    }

    /** {@code path} as the model, refusing a second model argument. */
    private static Path model(String flag, Path current, String path) throws UsageException {
        if (current != null) {
            throw new UsageException("too many model arguments; " + flag + " " + path + " follows " + current);
        }
        return Paths.get(path);
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }
}
