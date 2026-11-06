package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.load.ConverterMetadata;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.load.OverriddenSource;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@code nftool new-metadata}: copies a GGUF file to a new one with metadata added, changed or removed, the
 * operation of llama.cpp's {@code gguf_new_metadata.py}. The tensor table and every tensor byte are copied
 * unchanged through {@link GgufWriter}; only the key/value section differs.
 *
 * <p>Values are set with the {@code --override-kv} grammar ({@code --set key=type:value}, every scalar GGUF type),
 * with a converter-style JSON object file ({@code --metadata}), or with the reference's convenience flags:
 * {@code --general-name}, {@code --general-description}, {@code --pre-tokenizer}, the chat template flags
 * ({@code --chat-template}, {@code --chat-template-file}, {@code --chat-template-config}) and the special-token
 * flags ({@code --special-token NAME TOKEN}, {@code --special-token-by-id NAME ID}, which resolve the token index
 * in the model's own vocabulary). {@code --remove} drops a key. Flags apply in command-line order, so a later
 * value of a key wins.</p>
 *
 * <p>Setting a chat template replaces the file's whole chat template family: the old
 * {@code tokenizer.chat_template}, {@code tokenizer.chat_template.*} and {@code tokenizer.chat_templates} keys are
 * removed before the new ones are written. The reference's {@code --force} confirmation is not needed: a
 * {@code --dry-run} prints the files the rewrite would produce and writes nothing, and an existing output is
 * replaced only with {@code --overwrite}.</p>
 */
final class GgufNewMetadataCommand {

    /** Tokenizer special-token names and the keys they set, as the reference derives them from the key name. */
    private static final Map<String, String> TOKEN_ID_KEYS = tokenKeys();

    private static final String USAGE =
            "nftool new-metadata <in.gguf> <out.gguf> [options]\n"
            + "options:\n"
            + "  --set <key=type:value>       set or add a scalar value; type is str, bool, int, float,\n"
            + "                               uint8, int8, uint16, int16, uint32, int64 or float64\n"
            + "  --metadata <file>            set values from one JSON object file (arrays and every GGUF type)\n"
            + "  --remove <key>               remove a metadata key\n"
            + "  --general-name <name>        set general.name\n"
            + "  --general-description <text> set general.description\n"
            + "  --pre-tokenizer <name>       set tokenizer.ggml.pre\n"
            + "  --chat-template <text|JSON>  set the chat template; a JSON list of {name, template} objects\n"
            + "                               writes named templates too\n"
            + "  --chat-template-file <file>  set the chat template from a Jinja file\n"
            + "  --chat-template-config <f>   set the chat template from a tokenizer_config.json\n"
            + "  --special-token <name> <tok> set a special token id by its token text\n"
            + "  --special-token-by-id <name> <id>\n"
            + "                               set a special token id directly\n"
            + "  --overwrite                  replace an existing output\n"
            + "  --dry-run                    print the files the rewrite writes, write nothing\n"
            + "  --help, -h                   this text\n"
            + "special-token names: " + String.join(", ", TOKEN_ID_KEYS.keySet()) + "\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** A {@code --special-token} or {@code --special-token-by-id} request, resolved once the model is open. */
    private static final class SpecialToken {
        final String name;
        final String value;
        final boolean byId;

        SpecialToken(String name, String value, boolean byId) {
            this.name = name;
            this.value = value;
            this.byId = byId;
        }
    }

    /** The parsed command line. */
    static final class Parsed {
        String input;
        String output;
        final List<MetadataOverrides.Override> overrides = new ArrayList<>();
        final Set<String> remove = new LinkedHashSet<>();
        final List<SpecialToken> specialTokens = new ArrayList<>();
        Path metadataFile;
        Object chatTemplate;
        boolean overwrite;
        boolean dryRun;
        boolean help;
    }

    private GgufNewMetadataCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code new-metadata}. */
    static void run(String[] args) {
        try {
            execute(parse(args), System.out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the flags and the two positionals; every value is checked as it is read. */
    static Parsed parse(String[] args) throws UsageException {
        Parsed p = new Parsed();
        for (int i = 0; i < args.length && !p.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                p.help = true;
            } else if (a.equals("--overwrite")) {
                p.overwrite = true;
            } else if (a.equals("--dry-run")) {
                p.dryRun = true;
            } else if (a.equals("--set")) {
                p.overrides.add(override(value(a, args, i), "--set"));
                i++;
            } else if (a.equals("--metadata")) {
                p.metadataFile = Paths.get(value(a, args, i));
                i++;
            } else if (a.equals("--remove")) {
                p.remove.add(value(a, args, i));
                i++;
            } else if (a.equals("--general-name")) {
                p.overrides.add(stringOverride(GgufKeys.GENERAL_NAME, value(a, args, i)));
                i++;
            } else if (a.equals("--general-description")) {
                p.overrides.add(stringOverride(GgufKeys.GENERAL_DESCRIPTION, value(a, args, i)));
                i++;
            } else if (a.equals("--pre-tokenizer")) {
                p.overrides.add(stringOverride(GgufKeys.TOKENIZER_PRE, value(a, args, i)));
                i++;
            } else if (a.equals("--chat-template")) {
                p.chatTemplate = chatTemplateText(value(a, args, i));
                i++;
            } else if (a.equals("--chat-template-file")) {
                p.chatTemplate = readText("--chat-template-file", Paths.get(value(a, args, i)));
                i++;
            } else if (a.equals("--chat-template-config")) {
                p.chatTemplate = chatTemplateConfig(Paths.get(value(a, args, i)));
                i++;
            } else if (a.equals("--special-token")) {
                String name = value(a, args, i);
                p.specialTokens.add(new SpecialToken(name, value("--special-token", args, i + 1), false));
                i += 2;
            } else if (a.equals("--special-token-by-id")) {
                String name = value(a, args, i);
                p.specialTokens.add(new SpecialToken(name, value("--special-token-by-id", args, i + 1), true));
                i += 2;
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (p.input == null) {
                p.input = a;
            } else if (p.output == null) {
                p.output = a;
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + p.output);
            }
        }
        if (!p.help && (p.input == null || p.output == null)) {
            throw new UsageException("new-metadata needs an input and an output GGUF file");
        }
        return p;
    }

    /** Applies the requested metadata changes and writes the output through {@link GgufWriter}. */
    static void execute(Parsed p, Appendable out) throws IOException, UsageException {
        if (p.help) {
            out.append(USAGE);
            return;
        }
        if (ModelSources.detect(Paths.get(p.input)) != ModelFormat.GGUF) {
            throw new IOException("new-metadata rewrites a GGUF file; '" + p.input + "' is a Hugging Face"
                    + " checkpoint, convert it with nftool export first");
        }
        try (GGUFFileParser src = ModelSources.parseGguf(p.input)) {
            if (src.getShardCount() > 1) {
                throw new IOException("'" + p.input + "' is a split model of " + src.getShardCount()
                        + " files; new-metadata rewrites one file, so merge the splits with nftool merge"
                        + " first");
            }
            List<MetadataOverrides.Override> overrides = new ArrayList<>();
            Set<String> removed = new LinkedHashSet<>(p.remove);
            if (p.metadataFile != null) {
                addJsonMetadata(p, src, overrides, removed);
            }
            overrides.addAll(p.overrides);
            addSpecialTokens(p, src, overrides);
            if (p.chatTemplate != null) {
                removeChatTemplateKeys(src, removed);
                overrides.addAll(chatTemplateOverrides(p.chatTemplate));
            }
            OverriddenSource view = new OverriddenSource(src, overrides, removed);
            Path output = Paths.get(p.output);
            GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE, p.overwrite,
                    GGUFFileParser.DEFAULT_ALIGNMENT);
            if (p.dryRun) {
                out.append("nftool new-metadata: dry run, not writing files; the rewrite writes\n");
                for (GgufWriter.PlannedShard shard : GgufWriter.plan(view, output, options)) {
                    out.append("  ").append(shard.toString()).append('\n');
                }
                return;
            }
            List<Path> written = GgufWriter.write(view, output, options);
            out.append("nftool new-metadata: wrote ").append(written.get(0).toString()).append('\n');
        }
    }

    // ------------------------------------------------------------------ metadata sources

    /** Adds the {@code --metadata} file's overrides and the keys its entry lists replace. */
    private static void addJsonMetadata(Parsed p, GGUFFileParser src, List<MetadataOverrides.Override> overrides,
                                        Set<String> removed) throws IOException, UsageException {
        Map<?, ?> object = jsonObject(p.metadataFile, "--metadata");
        ConverterMetadata metadata;
        try {
            metadata = ConverterMetadata.fromJson(object, "--metadata '" + p.metadataFile + "'");
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        overrides.addAll(metadata.overrides());
        if (object.containsKey(ConverterMetadata.BASE_MODELS)) {
            removePrefix(src, removed, "general.base_model.");
        }
        if (object.containsKey(ConverterMetadata.DATASETS)) {
            removePrefix(src, removed, "general.dataset.");
        }
    }

    private static void addSpecialTokens(Parsed p, GGUFFileParser src, List<MetadataOverrides.Override> overrides)
            throws IOException {
        if (p.specialTokens.isEmpty()) {
            return;
        }
        String[] tokens = tokenizerTokens(src);
        for (SpecialToken request : p.specialTokens) {
            String key = TOKEN_ID_KEYS.get(request.name);
            if (key == null) {
                throw new IOException("unknown special token name '" + request.name + "'; use one of "
                        + String.join(", ", TOKEN_ID_KEYS.keySet()));
            }
            int id = request.byId ? tokenId(request, tokens.length) : tokenIndex(request, tokens);
            overrides.add(MetadataOverrides.Override.scalar(key, GGUFValueType.UINT32, Integer.valueOf(id)));
        }
    }

    /** The model's tokenizer list, refused by name when the model has none. */
    private static String[] tokenizerTokens(GGUFFileParser src) throws IOException {
        try {
            String[] tokens = src.getStringArray(GgufKeys.TOKENIZER_LIST);
            if (tokens != null) {
                return tokens;
            }
        } catch (IllegalArgumentException e) {
            throw new IOException("'" + src.getFilePath() + "' has no usable " + GgufKeys.TOKENIZER_LIST + ": "
                    + e.getMessage(), e);
        }
        throw new IOException("'" + src.getFilePath() + "' has no " + GgufKeys.TOKENIZER_LIST
                + ", so a special token cannot be resolved by its text");
    }

    private static int tokenIndex(SpecialToken request, String[] tokens) throws IOException {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals(request.value)) {
                return i;
            }
        }
        throw new IOException("token '" + request.value + "' is not in the model's "
                + GgufKeys.TOKENIZER_LIST);
    }

    private static int tokenId(SpecialToken request, int tokenCount) throws IOException {
        int id;
        try {
            id = Integer.parseInt(request.value.trim());
        } catch (NumberFormatException e) {
            throw new IOException("--special-token-by-id " + request.name + " needs a token id, not '"
                    + request.value + "'");
        }
        if (id < 0 || id >= tokenCount) {
            throw new IOException("token id " + id + " is outside the model's " + tokenCount + " tokens");
        }
        return id;
    }

    private static void removePrefix(GGUFFileParser src, Set<String> removed, String prefix) {
        for (String key : src.getMetadata().keySet()) {
            if (key.startsWith(prefix)) {
                removed.add(key);
            }
        }
    }

    /**
     * Removes the whole chat template family of {@code src}: the default template, every named template and the
     * name list. The new template's keys are written by {@link #chatTemplateOverrides(Object)}.
     */
    private static void removeChatTemplateKeys(GGUFFileParser src, Set<String> removed) {
        String namedPrefix = GgufKeys.TOKENIZER_CHAT_TEMPLATE + ".";
        for (String key : src.getMetadata().keySet()) {
            if (key.equals(GgufKeys.TOKENIZER_CHAT_TEMPLATE) || key.equals(GgufKeys.TOKENIZER_CHAT_TEMPLATES)
                    || key.startsWith(namedPrefix)) {
                removed.add(key);
            }
        }
    }

    /** The chat template keys of {@code template}, as {@code GGUFWriter.add_chat_template} writes them. */
    private static List<MetadataOverrides.Override> chatTemplateOverrides(Object template) throws UsageException {
        Map<String, Object> keys;
        try {
            keys = HfChatTemplates.toGguf(template);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        List<MetadataOverrides.Override> out = new ArrayList<>(keys.size());
        for (Map.Entry<String, Object> e : keys.entrySet()) {
            Object value = e.getValue();
            if (value instanceof String) {
                out.add(MetadataOverrides.Override.scalar(e.getKey(), GGUFValueType.STRING, value));
            } else if (value instanceof List) {
                out.add(MetadataOverrides.Override.array(e.getKey(), GGUFValueType.STRING, (List<?>) value));
            } else {
                throw new UsageException("chat template key '" + e.getKey() + "' has no string or list value");
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ flag values

    private static MetadataOverrides.Override override(String entry, String what) throws UsageException {
        try {
            return MetadataOverrides.parse(entry, what);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    private static MetadataOverrides.Override stringOverride(String key, String value) {
        return MetadataOverrides.Override.scalar(key, GGUFValueType.STRING, value);
    }

    /** A {@code --chat-template} value: a plain template string, or a JSON list of named templates. */
    private static Object chatTemplateText(String text) throws UsageException {
        if (!text.trim().startsWith("[")) {
            return text;
        }
        Object parsed;
        try {
            parsed = Json.parse(text, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw new UsageException("--chat-template is not valid JSON: " + e.getMessage());
        }
        validateTemplates(parsed);
        return parsed;
    }

    /** The {@code chat_template} member of a tokenizer configuration file. */
    private static Object chatTemplateConfig(Path file) throws UsageException {
        Map<?, ?> object = jsonObject(file, "--chat-template-config");
        Object template = object.get("chat_template");
        if (template == null) {
            throw new UsageException("--chat-template-config '" + file + "' has no chat_template member");
        }
        if (template instanceof String) {
            return template;
        }
        if (template instanceof List) {
            validateTemplates(template);
            return template;
        }
        throw new UsageException("--chat-template-config '" + file + "': chat_template is neither a string"
                + " nor a list of {name, template} objects");
    }

    /** Refuses a template list with an entry the writer would silently skip. */
    private static void validateTemplates(Object parsed) throws UsageException {
        if (!(parsed instanceof List)) {
            throw new UsageException("a JSON chat template list must hold {name, template} objects");
        }
        for (Object entry : (List<?>) parsed) {
            if (!(entry instanceof Map)) {
                throw new UsageException("--chat-template: every list entry must be a {name, template} object");
            }
            Map<?, ?> m = (Map<?, ?>) entry;
            if (!(m.get("name") instanceof String) || !(m.get("template") instanceof String)) {
                throw new UsageException("--chat-template: every list entry needs a string name and a string"
                        + " template");
            }
        }
    }

    private static Map<?, ?> jsonObject(Path file, String flag) throws UsageException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new UsageException(flag + " '" + file + "' is not a readable file");
        }
        Object parsed;
        try {
            parsed = Json.parse(file, Json.MAX_FILE_BYTES, Json.Limits.STRICT);
        } catch (JsonException | IOException e) {
            throw new UsageException(flag + " '" + file + "' is not valid JSON: " + e.getMessage());
        }
        if (!(parsed instanceof Map)) {
            throw new UsageException(flag + " '" + file + "' is not a JSON object; use {\"key\": value, ...}");
        }
        return (Map<?, ?>) parsed;
    }

    private static String readText(String flag, Path file) throws UsageException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new UsageException(flag + " '" + file + "' is not a readable file");
        }
        try {
            return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UsageException(flag + " '" + file + "' cannot be read: " + e.getMessage());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /**
     * The tokenizer special-token names and keys, derived as the reference derives them: each
     * {@code tokenizer.*_token_id} key named by its last segment without {@code _token_id}.
     */
    private static Map<String, String> tokenKeys() {
        List<String> keys = Arrays.asList(GgufKeys.TOKENIZER_BOS_ID, GgufKeys.TOKENIZER_EOS_ID,
                GgufKeys.TOKENIZER_EOT_ID, GgufKeys.TOKENIZER_EOM_ID, GgufKeys.TOKENIZER_UNK_ID,
                GgufKeys.TOKENIZER_SEP_ID, GgufKeys.TOKENIZER_PAD_ID, GgufKeys.TOKENIZER_CLS_ID,
                GgufKeys.TOKENIZER_MASK_ID, GgufKeys.TOKENIZER_FIM_PRE_ID, GgufKeys.TOKENIZER_FIM_SUF_ID,
                GgufKeys.TOKENIZER_FIM_MID_ID, GgufKeys.TOKENIZER_FIM_PAD_ID, GgufKeys.TOKENIZER_FIM_REP_ID,
                GgufKeys.TOKENIZER_FIM_SEP_ID);
        Map<String, String> map = new LinkedHashMap<>();
        for (String key : keys) {
            String last = key.substring(key.lastIndexOf('.') + 1);
            map.put(last.substring(0, last.length() - "_token_id".length()), key);
        }
        return Collections.unmodifiableMap(map);
    }
}
