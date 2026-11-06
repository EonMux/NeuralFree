package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.SourceOrigin;
import com.eonmux.neuralfree.gguf.hf.HfFileType;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.lora.PeftAdapterConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.load.ConverterMetadata;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.load.OverriddenSource;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@code nftool export}: exports any {@link ModelSource}, resolved from its path exactly as the demos resolve a model
 * ({@link ModelSources#open}, which routes a GGUF file, a split's first file, a Hugging Face checkpoint and the rest),
 * to a GGUF file through {@link GgufWriter#write(ModelSource, Path, GgufWriter.Options)}, with
 * the flags of llama.cpp's {@code convert_hf_to_gguf.py} and {@code convert_lora_to_gguf.py}. Positionals first
 * ({@code nftool export <model> <out.gguf>}), then the flags:
 * <ul>
 *   <li>{@code --mode native|compat|compat-exact}, {@code --overwrite}, {@code --vocab-only};</li>
 *   <li>{@code --outtype}: the float tensor type of a converted checkpoint (f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
 *       or source), refused for a source already stored as GGUF;</li>
 *   <li>{@code --mmproj}: the vision / audio projector view of a Hugging Face checkpoint
 *       ({@link ModelSources#openProjector});</li>
 *   <li>{@code --mtp}, {@code --no-nextn} (alias {@code --no-mtp}) and {@code --dspark}: the converter's draft
 *       selection ({@link ModelSourceOptions.DraftMode}), mutually exclusive and refused for an architecture whose
 *       converter does not support the MTP export;</li>
 *   <li>{@code --lora} with {@code --base <path>}: a PEFT LoRA adapter directory converted for its base model
 *       ({@link ModelSources#openAdapter}), the base an HF directory holding at least its {@code config.json} or a
 *       GGUF model; without {@code --base} the adapter's local {@code base_model_name_or_path};</li>
 *   <li>{@code --metadata <file>} and {@code --model-name <name>}: authorship metadata overrides
 *       ({@link ConverterMetadata}); {@code --override-kv key=type:value} sets any scalar metadata value of the
 *       exported view on top, with the load-time flag's grammar ({@link MetadataOverrides#parse});</li>
 *   <li>{@code --dry-run}: the files the export writes, with their tensor counts and bytes, nothing written;</li>
 *   <li>{@code --print-supported-models}: the convertible architectures, then exit;</li>
 *   <li>{@code --split-max-tensors <n>}, {@code --split-max-size <n>} (bytes, or with a {@code k}/{@code m}/{@code
 *       g} suffix as llama.cpp's split tool takes it) and {@code --no-tensor-first-split}: a split into the llama.cpp
 *       shard names derived from {@code out} ({@link com.eonmux.neuralfree.gguf.GgufTensorSink#shards}).</li>
 * </ul>
 * The output name may carry the converter's {@code {ftype}} / {@code {outtype}} (lower case) and {@code {FTYPE}} /
 * {@code {OUTTYPE}} (upper case) placeholders, filled with the exported {@code general.file_type}. Flags parse in the
 * {@code LLMDemoOptions} style; {@code -Dneuralfree.*} properties still apply to the model's open, and the flags
 * override them.
 */
final class ExportCommand {

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool export <model> <out.gguf> [options]\n"
            + "options:\n"
            + "  --mode <native|compat|compat-exact>   export mode, default native\n"
            + "  --overwrite                           replace the output (or every derived shard)\n"
            + "  --vocab-only                          metadata and tokenizer keys, no tensors\n"
            + "  --outtype <t>                         float tensor type of a converted checkpoint: f32, f16,\n"
            + "                                        bf16, q8_0, tq1_0, tq2_0, auto or source; default source\n"
            + "  --mmproj                              export the vision/audio projector of a checkpoint\n"
            + "  --mtp                                 export only the NextN/MTP draft head as its own GGUF\n"
            + "  --no-nextn, --no-mtp                  exclude the NextN/MTP tensors from the target\n"
            + "  --dspark                              export the DeepSeek-V4, V4.1 or Mistral Large 4 DSpark\n"
            + "                                        draft as a DFLASH GGUF\n"
            + "  --lora                                <model> is a PEFT LoRA adapter directory\n"
            + "  --base <path>                         the adapter's base model: an HF directory with its\n"
            + "                                        config.json (weights optional) or a GGUF model\n"
            + "  --metadata <file>                     authorship metadata overrides from one JSON object file\n"
            + "  --model-name <name>                   the exported model's general.name\n"
            + "  --override-kv <key=type:value>        set a scalar metadata value on the exported view\n"
            + "  --dry-run                             print the files the export writes, write nothing\n"
            + "  --print-supported-models              list the convertible architectures and exit\n"
            + "  --split-max-tensors <n>               largest tensor count of one shard\n"
            + "  --split-max-size <n>                  largest shard size in bytes, e.g. 2000m\n"
            + "  --no-tensor-first-split               the first shard of a split carries no tensors\n"
            + "  --help, -h                            this text\n"
            + "The output name may hold {ftype} or {FTYPE}, replaced by the exported file type (e.g. q8_0).\n";

    /** The command line {@link #parse} fills; read-only once parsed. */
    private static final class Request {
        Path model;
        Path out;
        GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE);
        ModelSourceOptions.OutType outType;
        Path metadataFile;
        String modelName;
        final List<MetadataOverrides.Override> overrides = new ArrayList<>();
        Path base;
        boolean lora;
        boolean mmproj;
        ModelSourceOptions.DraftMode draft;
        boolean dryRun;
        boolean printSupported;
        boolean help;
    }

    private ExportCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code export}. */
    static void run(String[] args) {
        try {
            List<Path> written = export(args);
            if (written != null) {
                StringBuilder names = new StringBuilder();
                for (int i = 0; i < written.size(); i++) {
                    if (i > 0) {
                        names.append(", ");
                    }
                    names.append(written.get(i));
                }
                System.out.println("nftool export: wrote " + names);
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
     * Parses the flags and exports; {@code null} for {@code --help}, {@code --dry-run} and
     * {@code --print-supported-models}, which print and write nothing.
     */
    static List<Path> export(String[] args) throws IOException, UsageException {
        Request r = parse(args);
        if (r.help) {
            System.out.println(USAGE);
            return null;
        }
        if (r.printSupported) {
            printSupportedModels();
            return null;
        }
        validate(r);
        ConverterMetadata metadata = converterMetadata(r.metadataFile, r.modelName);
        ModelSource raw = open(r, sourceOptions(r.outType, r.draft));
        ModelSource applied;
        try {
            applied = metadata.apply(raw);
        } catch (RuntimeException e) {
            closeAfter(raw, e);
            throw e;
        }
        ModelSource view = r.overrides.isEmpty() ? applied : new OverriddenSource(applied, r.overrides);
        try (ModelSource src = view) {
            Path out = templated(r.out, src);
            if (r.dryRun) {
                printPlan(GgufWriter.plan(src, out, r.options));
                return null;
            }
            return GgufWriter.write(src, out, r.options);
        }
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
            } else if (r.model == null) {
                r.model = Paths.get(a);
            } else if (r.out == null) {
                r.out = Paths.get(a);
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + r.model + " and " + r.out);
            }
        }
        return r;
    }

    /** Applies the flag {@code a} at {@code args[i]}; the index of its last argument. */
    private static int flag(Request r, String a, String[] args, int i) throws UsageException {
        try {
            switch (a) {
                case "--mode":
                    r.options = r.options.mode(mode(a, value(a, args, i)));
                    return i + 1;
                case "--overwrite":
                    r.options = r.options.overwrite(true);
                    return i;
                case "--vocab-only":
                    r.options = r.options.vocabOnly(true);
                    return i;
                case "--no-tensor-first-split":
                    r.options = r.options.smallFirstShard(true);
                    return i;
                case "--split-max-tensors":
                    r.options = r.options.splitMaxTensors(longValue(a, value(a, args, i)));
                    return i + 1;
                case "--split-max-size":
                    r.options = r.options.splitMaxBytes(splitSize(a, value(a, args, i)));
                    return i + 1;
                default:
                    return sourceFlag(r, a, args, i);
            }
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** The flags that choose what is opened and how; the index of the flag's last argument. */
    private static int sourceFlag(Request r, String a, String[] args, int i) throws UsageException {
        switch (a) {
            case "--outtype":
                r.outType = outType(a, value(a, args, i));
                return i + 1;
            case "--mmproj":
                r.mmproj = true;
                return i;
            case "--mtp":
                return draft(r, ModelSourceOptions.DraftMode.MTP_ONLY, i);
            case "--no-nextn":
            case "--no-mtp":
                return draft(r, ModelSourceOptions.DraftMode.NO_NEXTN, i);
            case "--dspark":
                return draft(r, ModelSourceOptions.DraftMode.DSPARK, i);
            case "--lora":
                r.lora = true;
                return i;
            case "--base":
                r.base = Paths.get(value(a, args, i));
                return i + 1;
            case "--metadata":
                r.metadataFile = Paths.get(value(a, args, i));
                return i + 1;
            case "--model-name":
                r.modelName = value(a, args, i);
                return i + 1;
            case "--override-kv":
                try {
                    r.overrides.add(MetadataOverrides.parse(value(a, args, i), a));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(e.getMessage());
                }
                return i + 1;
            case "--dry-run":
                r.dryRun = true;
                return i;
            case "--print-supported-models":
                r.printSupported = true;
                return i;
            default:
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
        }
    }

    /** Records one of {@code --mtp}, {@code --no-nextn} / {@code --no-mtp} and {@code --dspark}. */
    private static int draft(Request r, ModelSourceOptions.DraftMode mode, int i) throws UsageException {
        if (r.draft != null && r.draft != mode) {
            throw new UsageException("--mtp, --no-nextn, and --dspark are mutually exclusive");
        }
        r.draft = mode;
        return i;
    }

    /** Refuses a missing positional and the flag combinations no converter accepts, naming the flags. */
    private static void validate(Request r) throws IOException, UsageException {
        if (r.model == null) {
            throw new UsageException("export needs a model path and an output path");
        }
        if (r.out == null) {
            throw new UsageException("export needs an output path after " + r.model);
        }
        if (r.base != null && !r.lora) {
            throw new UsageException("--base names the base model of a --lora export; add --lora");
        }
        if (r.lora && r.mmproj) {
            throw new UsageException("--lora and --mmproj are exclusive: an adapter has no projector");
        }
        if (r.options.vocabOnly && (r.lora || r.mmproj)) {
            throw new UsageException("--vocab-only exports a model's tokenizer; " + (r.lora ? "--lora" : "--mmproj")
                    + " exports have none");
        }
        if (r.draft != null && r.mmproj) {
            throw new UsageException("--" + draftFlag(r.draft) + " and --mmproj are exclusive: a projector export"
                    + " selects no draft");
        }
        if (r.draft != null && r.lora) {
            throw new UsageException("--" + draftFlag(r.draft) + " and --lora are exclusive: an adapter has no"
                    + " NextN draft");
        }
        if (r.draft != null && ModelSources.detect(r.model) != ModelFormat.SAFETENSORS) {
            throw new UsageException("--" + draftFlag(r.draft) + " applies to a converted Hugging Face checkpoint;"
                    + " '" + r.model + "' is stored as GGUF and keeps its tensors");
        }
        if (r.outType != null && ModelSources.detect(r.model) != ModelFormat.SAFETENSORS) {
            throw new UsageException("--outtype " + name(r.outType) + " applies to a converted Hugging Face"
                    + " checkpoint; '" + r.model + "' is stored as GGUF and keeps its tensor types");
        }
        if (r.lora && (r.outType == ModelSourceOptions.OutType.TQ1_0
                || r.outType == ModelSourceOptions.OutType.TQ2_0)) {
            throw new UsageException("--outtype " + name(r.outType) + " is not an adapter type; a --lora export"
                    + " takes f32, f16, bf16, q8_0, auto or source");
        }
    }

    // ------------------------------------------------------------------ opening

    /** The flag that selected {@code mode}, for the messages that name it. */
    private static String draftFlag(ModelSourceOptions.DraftMode mode) {
        return mode == ModelSourceOptions.DraftMode.MTP_ONLY ? "mtp"
                : mode == ModelSourceOptions.DraftMode.NO_NEXTN ? "no-nextn" : "dspark";
    }

    /** The source the request exports: the model, its projector view or the adapter on its base. */
    private static ModelSource open(Request r, ModelSourceOptions src) throws IOException, UsageException {
        if (r.mmproj) {
            return ModelSources.openProjector(r.model, src);
        }
        if (!r.lora) {
            return ModelSources.open(r.model.toString(), src);
        }
        if (ModelSources.detect(r.model) == ModelFormat.GGUF) {
            if (r.base != null) {
                throw new UsageException("--base applies to a PEFT adapter directory; the GGUF adapter '" + r.model
                        + "' carries its own metadata");
            }
            return ModelSources.open(r.model.toString(), src);
        }
        return ModelSources.openAdapter(r.model.toString(), baseOrigin(r.model, r.base), src);
    }

    /**
     * The base model of the PEFT adapter at {@code adapter}: {@code base}, or the adapter's
     * {@code base_model_name_or_path} when it names a local directory (absolute, or relative to the adapter). A
     * directory with a {@code config.json} is an HF base whatever weights it holds, as the converter's
     * {@code --base} only needs the configuration; any other path is detected like a model.
     *
     * @throws UsageException naming {@code --base} when there is no base to resolve
     */
    private static SourceOrigin baseOrigin(Path adapter, Path base) throws IOException, UsageException {
        Path p = base;
        if (p == null) {
            String named = PeftAdapterConfig.read(adapter).baseModelNameOrPath();
            if (named == null) {
                throw new UsageException("--lora needs --base <dir>: '" + adapter + "/"
                        + PeftAdapterConfig.ADAPTER_CONFIG_JSON + "' names no base_model_name_or_path");
            }
            p = adapter.resolve(named).normalize();
            if (!Files.isDirectory(p)) {
                throw new UsageException("--lora needs --base <dir>: the adapter's base model '" + named
                        + "' is not a local directory");
            }
        } else if (!Files.exists(p)) {
            throw new UsageException("--base '" + p + "' does not exist");
        }
        if (Files.isDirectory(p) && Files.isRegularFile(p.resolve("config.json"))) {
            return new SourceOrigin(p, ModelFormat.SAFETENSORS);
        }
        return new SourceOrigin(p, ModelSources.detect(p));
    }

    /**
     * The model's open options: the {@code -Dneuralfree.*} properties with {@code --outtype} and the draft selection
     * on top.
     *
     * @throws UsageException with the message of a property that cannot be applied
     */
    private static ModelSourceOptions sourceOptions(ModelSourceOptions.OutType outType,
                                                    ModelSourceOptions.DraftMode draft) throws UsageException {
        try {
            ModelSourceOptions.Builder b = ModelSourceOptions.fromSystemProperties().toBuilder();
            if (outType != null) {
                b.outType(outType);
            }
            if (draft != null) {
                b.draftMode(draft);
            }
            return b.build();
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /**
     * The overrides of {@code --metadata} (one JSON object file) and {@code --model-name}, the name last.
     *
     * @throws UsageException naming the flag when the file is unreadable, is not one JSON object, or holds a value the
     *                        converter's file cannot
     * @throws IOException    when the file cannot be read
     */
    private static ConverterMetadata converterMetadata(Path file, String modelName)
            throws IOException, UsageException {
        try {
            ConverterMetadata m = file == null ? ConverterMetadata.none()
                    : ConverterMetadata.fromJson(metadataObject(file), "--metadata '" + file + "'");
            return modelName == null ? m : m.withModelName(modelName);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** The {@code --metadata} file's single JSON object. */
    private static Map<?, ?> metadataObject(Path file) throws IOException, UsageException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new UsageException("--metadata '" + file + "' is not a readable file");
        }
        Object parsed;
        try {
            parsed = Json.parse(file, Json.MAX_FILE_BYTES, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw new UsageException("--metadata '" + file + "' is not valid JSON: " + e.getMessage());
        }
        if (!(parsed instanceof Map)) {
            throw new UsageException("--metadata '" + file + "' is not a JSON object; use {\"key\": value, ...}");
        }
        return (Map<?, ?>) parsed;
    }

    /** Closes {@code s} after {@code failure}, keeping a close failure as suppressed. */
    private static void closeAfter(ModelSource s, RuntimeException failure) {
        try {
            s.close();
        } catch (IOException | RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    // ------------------------------------------------------------------ output name

    /**
     * {@code out} with the converter's file type placeholders of its file name filled ({@code fill_templated_filename}):
     * {@code {ftype}}, {@code {outtype}} and {@code {}} in lower case, {@code {FTYPE}} and {@code {OUTTYPE}} in upper
     * case, from the source's {@code general.file_type}.
     *
     * @throws UsageException when the name holds a placeholder and the source names no tensor file type
     */
    private static Path templated(Path out, ModelSource src) throws UsageException {
        String name = out.getFileName().toString();
        if (!name.contains("{ftype}") && !name.contains("{outtype}") && !name.contains("{}")
                && !name.contains("{FTYPE}") && !name.contains("{OUTTYPE}")) {
            return out;
        }
        String type = fileTypeName(src, name);
        String lower = type.toLowerCase(Locale.ROOT);
        String upper = type.toUpperCase(Locale.ROOT);
        String filled = name.replace("{ftype}", lower).replace("{outtype}", lower).replace("{}", lower)
                .replace("{FTYPE}", upper).replace("{OUTTYPE}", upper);
        return out.resolveSibling(filled);
    }

    /** The tensor type name of the source's {@code general.file_type}, e.g. {@code Q8_0}. */
    private static String fileTypeName(ModelSource src, String outName) throws UsageException {
        String key = GgufKeys.GENERAL_FILE_TYPE;
        Object value = src.getMetadata().get(key);
        GGMLType type = null;
        if (value != null) {
            try {
                type = HfFileType.typeOf(GGUFValueType.uint32Value(key, value, src.getValueType(key)));
            } catch (IllegalArgumentException e) {
                throw new UsageException("the output name '" + outName + "' needs the file type: " + e.getMessage());
            }
        }
        if (type == null) {
            throw new UsageException("the output name '" + outName + "' holds a file type placeholder, but '"
                    + src.getFilePath() + "' names no tensor file type (" + key + " " + value + ")");
        }
        return type.name();
    }

    // ------------------------------------------------------------------ prints

    /** {@code --print-supported-models}: the convertible architectures, as the converter lists them. */
    private static void printSupportedModels() {
        System.out.println("Supported models:");
        System.out.println("TEXT models:");
        for (String name : HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT)) {
            System.out.println("  - " + name);
        }
        System.out.println("MMPROJ models:");
        for (String name : HfArchRegistry.supportedNames(HfArchEntry.Kind.MMPROJ)) {
            System.out.println("  - " + name);
        }
    }

    /** The plan of a {@code --dry-run} export: the files it would write, with their tensor counts and bytes. */
    private static void printPlan(List<GgufWriter.PlannedShard> plan) {
        System.out.println("nftool export: dry run, not writing files; the export writes");
        for (GgufWriter.PlannedShard shard : plan) {
            System.out.println("  " + shard);
        }
    }

    // ------------------------------------------------------------------ values

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static GgufWriter.Mode mode(String flag, String v) throws UsageException {
        try {
            return GgufWriter.Mode.valueOf(v.trim().toUpperCase(Locale.ROOT).replace('-', '_'));
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + " is invalid; use native, compat or compat-exact");
        }
    }

    /** The {@code --outtype} value, named like the converter's flag. */
    private static ModelSourceOptions.OutType outType(String flag, String v) throws UsageException {
        try {
            return ModelSourceOptions.OutType.parse(v);
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + " '" + v + "' is invalid; use f32, f16, bf16, q8_0, tq1_0, tq2_0, auto"
                    + " or source");
        }
    }

    private static String name(ModelSourceOptions.OutType t) {
        return t.name().toLowerCase(Locale.ROOT);
    }

    /** A non-negative count of the flag, as llama.cpp parses one. */
    private static long longValue(String flag, String v) throws UsageException {
        try {
            long n = Long.parseLong(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
    }

    /**
     * A byte size: a non-negative integer with an optional {@code k}/{@code m}/{@code g} suffix (a power
     * of 1024 like llama.cpp's {@code --split-max-size} N, e.g. {@code 2000m}), never one that overflows;
     * {@code 0} keeps the no-limit meaning.
     */
    private static long splitSize(String flag, String v) throws UsageException {
        String s = v.trim().toLowerCase(Locale.ROOT);
        long factor = 1;
        if (!s.isEmpty()) {
            char suffix = s.charAt(s.length() - 1);
            if (suffix == 'k') {
                factor = 1L << 10;
            } else if (suffix == 'm') {
                factor = 1L << 20;
            } else if (suffix == 'g') {
                factor = 1L << 30;
            }
            if (factor != 1) {
                s = s.substring(0, s.length() - 1);
            }
        }
        long n;
        try {
            n = Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a size in bytes, e.g. 2000000 or 2000m, not '" + v + "'");
        }
        if (n < 0 || factor != 1 && n > Long.MAX_VALUE / factor) {
            throw new UsageException(flag + " '" + v + "' overflows a byte size");
        }
        return n * factor;
    }
}
