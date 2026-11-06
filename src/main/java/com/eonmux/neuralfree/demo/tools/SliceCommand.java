package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.gguf.hub.slice.ModelSlicer;
import com.eonmux.neuralfree.gguf.hub.slice.SliceOptions;
import com.eonmux.neuralfree.gguf.hub.slice.SliceReport;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * {@code nftool slice}: the developer and test tool of the model slice, the mirror of the reference
 * {@code slice-model} skill. It builds a small version of a Hugging Face safetensors model — fewer layers, fewer
 * experts, optionally fewer vocab rows — by reading the shard headers and then fetching only the kept tensors over
 * HTTP range requests ({@link ModelSlicer}); the weights never pass through the hub cache. The slice keeps the
 * requested layers with their indices renumbered to {@code 0..n-1}, keeps the first experts of every expert
 * family, patches {@code config.json} to the new counts, copies the small auxiliary repository files and writes
 * {@code .slice_meta.json}. {@code --inspect} prints the analysis and the plan without writing, and
 * {@code --dry-run} reports the planned size.
 *
 * <p>The repository expression is {@code <user>/<model>[@revision]}; a quant tag is refused because the tool reads
 * safetensors weights. The tool is online only, sends {@code --hf-token} or the environment's token
 * ({@code HF_TOKEN}, else the token file of {@code HF_TOKEN_PATH} or {@code HF_HOME/token}) in the
 * {@code Authorization} header and never prints it.</p>
 */
final class SliceCommand {

    /** An invalid command line; the message names the flag or the expression. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** A parsed command line: the repository, the options and the explicit token. */
    static final class Parsed {

        private final HubRef ref;
        private final SliceOptions options;
        private final String token;

        private Parsed(HubRef ref, SliceOptions options, String token) {
            this.ref = ref;
            this.options = options;
            this.token = token;
        }

        /** The repository to slice. */
        HubRef ref() {
            return ref;
        }

        /** The slice options. */
        SliceOptions options() {
            return options;
        }

        /** The explicit token, or null for the environment's token. */
        String token() {
            return token;
        }
    }

    private static final String USAGE =
            "nftool slice <user>/<model>[@revision] [options]\n"
            + "Builds a small test slice of a Hugging Face safetensors model, fetching only the kept tensors\n"
            + "with HTTP range requests. Layers and experts are renumbered to 0..n-1 and config.json is patched.\n"
            + "options:\n"
            + "  -o, --output <dir>       output directory (default ./tmp/hf_slices/<user>--<model>)\n"
            + "  --revision <rev>         branch or commit, instead of <user>/<model>@<rev>\n"
            + "  --num-layers <n>         keep the first n layers (default 2)\n"
            + "  --keep-layers <i,j,...>  keep exactly these layer indices (overrides --num-layers)\n"
            + "  --num-experts <n>        keep the first n experts of each expert layer (default 8)\n"
            + "  --vocab-size <n>         row-slice vocab-sized tensors to n rows (default 0: off; the\n"
            + "                           tokenizer is not sliced, which breaks conversion for most models)\n"
            + "  --max-size <size>        refuse a slice over this size (default 1G; K, M, G, T suffixes)\n"
            + "  --jobs <n>               concurrent range requests (default 8)\n"
            + "  --inspect                print the config analysis, tensor families and plan; write nothing\n"
            + "  --dry-run                plan and report the size; write nothing\n"
            + "  --overwrite              replace the files of a non-empty output directory\n"
            + "  --hf-token <token>       access token (default: HF_TOKEN, else the token file\n"
            + "                           HF_TOKEN_PATH names, else HF_HOME/token, as `hf auth login`\n"
            + "                           writes it)\n"
            + "  --help, -h               this text\n";

    private static final Pattern SIZE =
            Pattern.compile("([0-9]+(?:\\.[0-9]+)?)\\s*([kKmMgGtT]?)[bB]?");

    private SliceCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code slice}. */
    static void run(String[] args) {
        try {
            Parsed parsed = parse(args);
            if (parsed != null) {
                execute(parsed);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Error: the slice was interrupted");
            System.exit(1);
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the arguments; null for {@code --help}, which prints the usage and runs nothing. */
    static Parsed parse(String[] args) throws UsageException {
        String spec = null;
        String revision = null;
        String token = null;
        Path output = null;
        List<Integer> keepLayers = null;
        SliceOptions.Builder options = SliceOptions.builder();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return null;
            }
            if (a.equals("-o") || a.equals("--output")) {
                output = Paths.get(value(a, args, ++i));
            } else if (a.equals("--revision")) {
                revision = value(a, args, ++i);
            } else if (a.equals("--num-layers")) {
                options.numLayers(integer(a, value(a, args, ++i)));
            } else if (a.equals("--keep-layers")) {
                keepLayers = layers(value(a, args, ++i));
            } else if (a.equals("--num-experts")) {
                options.numExperts(integer(a, value(a, args, ++i)));
            } else if (a.equals("--vocab-size")) {
                options.vocabSize(integer(a, value(a, args, ++i)));
            } else if (a.equals("--max-size")) {
                options.maxBytes(size(value(a, args, ++i)));
            } else if (a.equals("--jobs")) {
                options.jobs(integer(a, value(a, args, ++i)));
            } else if (a.equals("--inspect")) {
                options.inspect(true);
            } else if (a.equals("--dry-run")) {
                options.dryRun(true);
            } else if (a.equals("--overwrite")) {
                options.overwrite(true);
            } else if (a.equals("--hf-token")) {
                token = value(a, args, ++i);
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (spec == null) {
                spec = a;
            } else {
                throw new UsageException("too many repository arguments; '" + a + "' follows " + spec);
            }
        }
        if (spec == null) {
            throw new UsageException("slice needs a repository, as <user>/<model>[@revision]");
        }
        if (revision != null) {
            if (spec.indexOf('@') >= 0) {
                throw new UsageException("the revision is already in '" + spec + "'; drop --revision");
            }
            spec = spec + "@" + revision;
        }
        HubRef ref;
        try {
            ref = HubRef.parse(spec);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        if (ref.quant() != null) {
            throw new UsageException("slice reads safetensors weights, so the quant tag ':" + ref.quant()
                    + "' does not apply");
        }
        if (keepLayers != null) {
            options.keepLayers(keepLayers);
        }
        if (output == null) {
            output = Paths.get("tmp", "hf_slices", ref.repository().replace("/", "--"));
        }
        options.output(output);
        try {
            return new Parsed(ref, options.build(), token);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** Resolves the repository and runs the slicer, printing the plan, the config changes and the result. */
    private static void execute(Parsed parsed) throws IOException, InterruptedException, UsageException {
        HubCache cache = new HubCache(HubCache.defaultRoot());
        HubClient client = parsed.token() == null
                ? new HubClient(cache) : new HubClient(cache, parsed.token());
        SliceReport report = ModelSlicer.run(client, parsed.ref(), parsed.options());
        print(report, parsed.options());
    }

    /** Prints the inspect report, the config changes, the warnings and the summary of the run. */
    private static void print(SliceReport report, SliceOptions options) {
        for (String warning : report.warnings()) {
            System.err.println("warning: " + warning);
        }
        if (options.inspect()) {
            System.out.print(report.inspect());
            return;
        }
        for (String patch : report.configPatches()) {
            System.out.println("config: " + patch);
        }
        for (String skipped : report.skippedAux()) {
            System.err.println("warning: skipped large aux file " + skipped);
        }
        String summary = "nftool slice: " + report.repository() + "@" + report.commit() + ": "
                + report.keptTensors() + "/" + report.totalTensors() + " tensors, "
                + report.humanBytes();
        if (report.dryRun()) {
            System.out.println(summary + " (dry run, nothing written)");
        } else {
            System.out.println(summary + " -> " + report.directory());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index];
    }

    /** An integer flag value, or an error naming the flag and the text. */
    private static int integer(String flag, String text) throws UsageException {
        try {
            return Integer.parseInt(text.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + ": '" + text + "' is not an integer");
        }
    }

    /** A comma-separated layer index list. */
    private static List<Integer> layers(String text) throws UsageException {
        List<Integer> layers = new ArrayList<>();
        for (String part : text.split(",", -1)) {
            String trimmed = part.trim();
            if (trimmed.isEmpty()) {
                throw new UsageException("--keep-layers needs comma-separated layer indices, got '" + text + "'");
            }
            layers.add(integer("--keep-layers", trimmed));
        }
        return layers;
    }

    /** A byte size with an optional K, M, G or T suffix. */
    static long size(String text) throws UsageException {
        Matcher m = SIZE.matcher(text.trim());
        if (!m.matches()) {
            throw new UsageException("--max-size: '" + text + "' is not a size (use bytes, K, M, G or T)");
        }
        int power = "kKmMgGtT".indexOf(m.group(2)) / 2;
        double multiplier = Math.pow(1024.0, m.group(2).isEmpty() ? 0 : power + 1);
        double value = Double.parseDouble(m.group(1)) * multiplier;
        if (value >= (double) Long.MAX_VALUE) {
            throw new UsageException("--max-size: '" + text + "' is too large");
        }
        return (long) value;
    }
}
