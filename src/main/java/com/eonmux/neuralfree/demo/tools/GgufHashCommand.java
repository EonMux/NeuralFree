package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.inspect.GgufHasher;
import com.eonmux.neuralfree.gguf.inspect.HashManifest;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

/**
 * {@code nftool hash}: the tensor hashes of llama.cpp's {@code gguf-hash}. Each selected algorithm prints one
 * line per tensor ({@code <algorithm> <hash> <file>:<tensor>}) and one for the whole model
 * ({@code <algorithm> <hash> <file>}), covering the tensor bytes in tensor-table order and never the metadata.
 * {@code --xxh64} is the default; {@code --sha1}, {@code --sha256} and {@code --uuid} select the others and
 * {@code --all} selects xxh64, sha1 and sha256 (the reference leaves the UUID out of {@code --all} because it is
 * an identifier, not a hash). {@code --no-layer} omits the per-tensor lines.
 *
 * <p>{@code -c}/{@code --check <manifest>} verifies the hashes against a manifest of
 * {@code <type> <hash> <target>} lines, suffixing each line with {@code Ok}, {@code Mismatch} or {@code Not Found}
 * and exiting with the reference's codes: 0 success, 1 a hash mismatch or a per-tensor failure, 2 a mismatch,
 * 3 no manifest entry matched the model, 4 the manifest holds no known hash type, 5 the manifest cannot be read.
 * Without an explicit algorithm, a manifest selects the most secure type it holds (sha256, then sha1, xxh64,
 * uuid).</p>
 */
final class GgufHashCommand {

    /** All hashes of a model matched, or hashing only. */
    static final int EXIT_SUCCESS = 0;
    /** A hash mismatched, or a per-tensor hash failed. */
    static final int EXIT_FAILURE = 1;
    /** A hash mismatched during validation. */
    static final int EXIT_MISMATCH = 2;
    /** Validation ran but no manifest line named the model or any of its tensors. */
    static final int EXIT_MANIFEST_MISSING_ENTRY = 3;
    /** The manifest exists but holds no hash type the tool knows. */
    static final int EXIT_MANIFEST_UNKNOWN_HASH = 4;
    /** The manifest cannot be read. */
    static final int EXIT_MANIFEST_FILE_ERROR = 5;

    private static final String USAGE =
            "nftool hash [options] <model.gguf>\n"
            + "options:\n"
            + "  --xxh64               xxHash64 of every tensor and of the model (the default)\n"
            + "  --sha1                SHA-1 of every tensor and of the model\n"
            + "  --sha256              SHA-256 of every tensor and of the model\n"
            + "  --uuid                UUIDv5 of the model tensor bytes in llama.cpp's namespace\n"
            + "  --all                 --xxh64, --sha1 and --sha256 together\n"
            + "  --no-layer            omit the per-tensor lines\n"
            + "  -c, --check <file>    verify against a manifest, line by line\n"
            + "  --help, -h            this text\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line. */
    static final class Parsed {
        String model;
        Path check;
        final Set<GgufHasher.Algorithm> algorithms = EnumSet.noneOf(GgufHasher.Algorithm.class);
        boolean noLayer;
        boolean help;
        /** True when no algorithm flag was given, so a manifest picks one or xxh64 is the default. */
        boolean autoSelect;
    }

    private GgufHashCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code hash} and exits with its code. */
    static void run(String[] args) {
        int code;
        try {
            code = execute(parse(args), System.out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
            return;
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(EXIT_FAILURE);
            return;
        }
        if (code != EXIT_SUCCESS) {
            System.exit(code);
        }
    }

    /** Parses the flags and the model path; {@code --help} needs no model. */
    static Parsed parse(String[] args) throws UsageException {
        Parsed p = new Parsed();
        for (int i = 0; i < args.length && !p.help; i++) {
            String raw = args[i];
            // the reference accepts --no_layer for --no-layer
            String a = raw.startsWith("--") ? raw.replace('_', '-') : raw;
            switch (a) {
                case "--help":
                case "-h":
                    p.help = true;
                    break;
                case "--xxh64":
                    p.algorithms.add(GgufHasher.Algorithm.XXH64);
                    break;
                case "--sha1":
                    p.algorithms.add(GgufHasher.Algorithm.SHA1);
                    break;
                case "--sha256":
                    p.algorithms.add(GgufHasher.Algorithm.SHA256);
                    break;
                case "--uuid":
                    p.algorithms.add(GgufHasher.Algorithm.UUID);
                    break;
                case "--all":
                    p.algorithms.add(GgufHasher.Algorithm.XXH64);
                    p.algorithms.add(GgufHasher.Algorithm.SHA1);
                    p.algorithms.add(GgufHasher.Algorithm.SHA256);
                    break;
                case "--no-layer":
                    p.noLayer = true;
                    break;
                case "-c":
                case "--check": {
                    p.check = Paths.get(value(a, args, i));
                    i++;
                    break;
                }
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + raw);
                    }
                    if (p.model != null) {
                        throw new UsageException("too many file arguments; '" + raw + "' follows " + p.model);
                    }
                    p.model = raw;
            }
        }
        if (p.help) {
            return p;
        }
        if (p.model == null) {
            throw new UsageException("hash needs a GGUF file");
        }
        p.autoSelect = p.algorithms.isEmpty();
        return p;
    }

    /**
     * Hashes and, with {@code --check}, verifies; returns the exit code.
     */
    static int execute(Parsed p, Appendable out) throws IOException {
        if (p.help) {
            out.append(USAGE);
            return EXIT_SUCCESS;
        }
        HashManifest manifest = null;
        Set<GgufHasher.Algorithm> algorithms = p.algorithms;
        if (p.check != null) {
            try {
                manifest = HashManifest.read(p.check);
            } catch (IOException e) {
                out.append("ERROR cannot open manifest ").append(p.check.toString()).append('\n');
                return EXIT_MANIFEST_FILE_ERROR;
            }
            if (manifest.knownTypes().isEmpty()) {
                out.append("ERROR manifest does not have any known hash format in ")
                        .append(p.check.toString()).append('\n');
                return EXIT_MANIFEST_UNKNOWN_HASH;
            }
            out.append("manifest  ").append(p.check.toString());
            for (GgufHasher.Algorithm a : new GgufHasher.Algorithm[] {GgufHasher.Algorithm.SHA256,
                    GgufHasher.Algorithm.SHA1, GgufHasher.Algorithm.XXH64, GgufHasher.Algorithm.UUID}) {
                if (manifest.knownTypes().contains(a.text())) {
                    out.append("  ").append(a.text());
                }
            }
            out.append('\n');
            if (p.autoSelect) {
                algorithms = EnumSet.of(mostSecure(manifest));
            }
        } else if (algorithms.isEmpty()) {
            algorithms = EnumSet.of(GgufHasher.Algorithm.XXH64);
        }

        GgufHasher.Result result;
        try (GGUFFileParser src = ModelSources.parseGguf(p.model)) {
            result = GgufHasher.hash(src, algorithms, p.noLayer);
        }

        boolean tensorInManifest = false;
        boolean modelInManifest = false;
        boolean tensorMismatch = false;
        boolean modelMismatch = false;
        for (GgufHasher.Line line : result.lines()) {
            out.append(String.format(Locale.ROOT, "%-8s  %s  %s", line.algorithm().text(), line.hash(),
                    line.target()));
            if (manifest != null) {
                HashManifest.Status status = manifest.verify(line.algorithm().text(), line.hash(),
                        line.target());
                out.append("  -  ").append(status.text());
                if (status != HashManifest.Status.NOT_FOUND) {
                    if (line.target().equals(p.model)) {
                        modelInManifest = true;
                        modelMismatch |= status == HashManifest.Status.MISMATCH;
                    } else {
                        tensorInManifest = true;
                        tensorMismatch |= status == HashManifest.Status.MISMATCH;
                    }
                }
            }
            out.append('\n');
        }
        if (manifest == null) {
            return EXIT_SUCCESS;
        }
        int code = verifyCode(tensorInManifest, modelInManifest, tensorMismatch, modelMismatch);
        out.append('\n').append("Verification results for ").append(p.check.toString()).append(" - ")
                .append(codeText(code)).append('\n');
        return code;
    }

    /** The reference's verification outcome from what the manifest covered. */
    private static int verifyCode(boolean tensorInManifest, boolean modelInManifest, boolean tensorMismatch,
                                  boolean modelMismatch) {
        if (!modelInManifest) {
            if (!tensorInManifest) {
                return EXIT_MANIFEST_MISSING_ENTRY;
            }
            return tensorMismatch ? EXIT_FAILURE : EXIT_SUCCESS;
        }
        if (tensorInManifest && tensorMismatch) {
            return EXIT_FAILURE;
        }
        return modelMismatch ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    /** The most secure hash type the manifest holds, the reference's sha256, sha1, xxh64, uuid order. */
    private static GgufHasher.Algorithm mostSecure(HashManifest manifest) {
        for (GgufHasher.Algorithm a : new GgufHasher.Algorithm[] {GgufHasher.Algorithm.SHA256,
                GgufHasher.Algorithm.SHA1, GgufHasher.Algorithm.XXH64, GgufHasher.Algorithm.UUID}) {
            if (manifest.knownTypes().contains(a.text())) {
                return a;
            }
        }
        throw new IllegalArgumentException("the manifest holds no known hash type");
    }

    private static String codeText(int code) {
        switch (code) {
            case EXIT_SUCCESS:
                return "Success";
            case EXIT_FAILURE:
                return "Failure";
            case EXIT_MISMATCH:
                return "Mismatch";
            case EXIT_MANIFEST_MISSING_ENTRY:
                return "Manifest Missing Entry";
            case EXIT_MANIFEST_UNKNOWN_HASH:
                return "Manifest Unknown Hash";
            default:
                return "Manifest File Error";
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }
}
