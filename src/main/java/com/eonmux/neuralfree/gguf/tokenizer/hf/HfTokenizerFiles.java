package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * The tokenizer files of a Hugging Face checkpoint directory, located once and read on demand.
 *
 * <p>{@link #locate(Path)} records which of the files the llama.cpp converter and transformers read are present:
 * {@code tokenizer.json}, {@code tokenizer_config.json}, {@code special_tokens_map.json}, {@code added_tokens.json},
 * the SentencePiece models ({@code tokenizer.model}, {@code spiece.model}, {@code sentencepiece.bpe.model}, Mistral's
 * versioned {@code tokenizer.model.v<N>}), {@code vocab.json} + {@code merges.txt}, {@code vocab.txt}, tiktoken ranks
 * ({@code *.tiktoken}, {@code tiktoken.model}), {@code tekken.json}, {@code rwkv_vocab_v*.txt}, PLaMo's
 * {@code tokenizer.jsonl}, the chat templates ({@code chat_template.jinja}, {@code chat_template.json},
 * {@code additional_chat_templates/*.jinja}), {@code processor_config.json} and {@code generation_config.json}.
 * Directory entries may be symbolic links, as in the Hugging Face cache snapshot layout, and are followed.</p>
 *
 * <p>A tokenizer source given apart from the model ({@link ModelSourceOptions#tokenizerDir()}) is applied with
 * {@link #withTokenizerDir(Path)}: a directory replaces the located tokenizer files, while
 * {@code generation_config.json} and {@code processor_config.json} stay the model's, as vLLM reads them from the
 * model path; a GGUF file supplies its {@code tokenizer.*} metadata ({@link #ggufTokenizerKeys()}).</p>
 *
 * <p>Every file is at most {@link #MAX_FILE_BYTES} long. A file that cannot be decoded or has the wrong shape throws
 * {@code IOException("Invalid Hugging Face checkpoint '<dir>': <file>: <detail>")}. Instances are immutable.</p>
 */
public final class HfTokenizerFiles {

    private static final Logger LOGGER = LogManager.getLogger(HfTokenizerFiles.class);

    /** Largest tokenizer file read. */
    public static final long MAX_FILE_BYTES = 512L << 20;
    /** Most tokens a vocabulary may hold; token ids are below this bound. */
    public static final int MAX_TOKENS = 16_777_216;

    public static final String TOKENIZER_JSON = "tokenizer.json";
    public static final String TOKENIZER_CONFIG_JSON = "tokenizer_config.json";
    public static final String SPECIAL_TOKENS_MAP_JSON = "special_tokens_map.json";
    public static final String ADDED_TOKENS_JSON = "added_tokens.json";
    public static final String TOKENIZER_MODEL = "tokenizer.model";
    public static final String TOKENIZER_MODEL_V3 = "tokenizer.model.v3";
    public static final String SPIECE_MODEL = "spiece.model";
    public static final String SENTENCEPIECE_BPE_MODEL = "sentencepiece.bpe.model";
    public static final String VOCAB_JSON = "vocab.json";
    public static final String MERGES_TXT = "merges.txt";
    public static final String VOCAB_TXT = "vocab.txt";
    public static final String QWEN_TIKTOKEN = "qwen.tiktoken";
    public static final String TIKTOKEN_MODEL = "tiktoken.model";
    public static final String TEKKEN_JSON = "tekken.json";
    public static final String RWKV_WORLD_VOCAB = "rwkv_vocab_v20230424.txt";
    public static final String PLAMO_JSONL = "tokenizer.jsonl";
    public static final String CHAT_TEMPLATE_JINJA = "chat_template.jinja";
    public static final String CHAT_TEMPLATE_JSON = "chat_template.json";
    public static final String ADDITIONAL_CHAT_TEMPLATES_DIR = "additional_chat_templates";
    public static final String PROCESSOR_CONFIG_JSON = "processor_config.json";
    public static final String GENERATION_CONFIG_JSON = "generation_config.json";

    /** The model's own files: a tokenizer directory does not replace them. */
    private static final List<String> MODEL_FILES = Collections.unmodifiableList(
            Arrays.asList(GENERATION_CONFIG_JSON, PROCESSOR_CONFIG_JSON));
    private static final Pattern TIKTOKEN_FILE = Pattern.compile(".+\\.tiktoken|tiktoken\\.model");
    private static final String JINJA_SUFFIX = ".jinja";
    private static final String TOKENIZER_KEY_PREFIX = "tokenizer.";
    private static final Pattern RWKV_VOCAB = Pattern.compile("rwkv_vocab_v.*\\.txt");
    /**
     * mistral_common's SentencePiece file names: the whole suffix chain ({@code Path.suffixes}) is {@code .model} or
     * {@code .model.v<N>[m<N>]}, so the stem holds no dot ({@code sentencepiece.bpe.model} is not one).
     */
    private static final Pattern MISTRAL_SPM = Pattern.compile("\\.*[^.]+\\.model(\\.v\\d+(m\\d+)?)?");
    private static final Pattern WINDOWS_DRIVE = Pattern.compile("^[A-Za-z]:.*");

    private static final HfTokenizerFiles EMPTY = new HfTokenizerFiles(null, Collections.emptyMap(),
            Collections.emptyList(), null, Collections.emptyMap(), Collections.emptyMap(), Collections.emptyMap());

    private final Path dir;
    /** Every regular file of {@link #dir}, links followed, by name in name order. */
    private final Map<String, Path> files;
    private final List<Path> additionalChatTemplates;
    private final Path ggufTokenizerFile;
    private final Map<String, Object> ggufKeys;
    private final Map<String, GGUFValueType> ggufTypes;
    private final Map<String, GGUFValueType> ggufElementTypes;

    private HfTokenizerFiles(Path dir, Map<String, Path> files, List<Path> additionalChatTemplates,
                             Path ggufTokenizerFile, Map<String, Object> ggufKeys, Map<String, GGUFValueType> ggufTypes,
                             Map<String, GGUFValueType> ggufElementTypes) {
        this.dir = dir;
        this.files = files;
        this.additionalChatTemplates = additionalChatTemplates;
        this.ggufTokenizerFile = ggufTokenizerFile;
        this.ggufKeys = ggufKeys;
        this.ggufTypes = ggufTypes;
        this.ggufElementTypes = ggufElementTypes;
    }

    /** No tokenizer files and no GGUF tokenizer metadata. */
    public static HfTokenizerFiles empty() {
        return EMPTY;
    }

    /**
     * The tokenizer files present in {@code dir}.
     *
     * @throws FileNotFoundException when {@code dir} is not a readable directory
     * @throws IOException           when the directory cannot be listed
     */
    public static HfTokenizerFiles locate(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir) || !Files.isReadable(dir)) {
            throw new FileNotFoundException("tokenizer files: '" + dir + "' is not a readable directory");
        }
        Path templatesDir = dir.resolve(ADDITIONAL_CHAT_TEMPLATES_DIR);
        List<Path> templates = Files.isDirectory(templatesDir)
                ? Collections.unmodifiableList(new ArrayList<>(regularFiles(templatesDir, JINJA_SUFFIX).values()))
                : Collections.emptyList();
        return new HfTokenizerFiles(dir, Collections.unmodifiableMap(regularFiles(dir, null)), templates, null,
                Collections.emptyMap(), Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * These files with the tokenizer source {@code tokenizerDir} applied: a directory replaces every located file
     * with its own except the model's {@code generation_config.json} and {@code processor_config.json}, which are
     * kept (vLLM's {@code --tokenizer} reads only tokenizer files from it); a GGUF file keeps the located files and
     * supplies its {@code tokenizer.*} metadata, which takes precedence over them ({@link #hasGgufTokenizer()}). A
     * null argument returns this.
     *
     * @throws FileNotFoundException when {@code tokenizerDir} does not exist or is not readable
     * @throws IOException           when it is neither a directory nor a GGUF file, or the GGUF file is invalid
     */
    public HfTokenizerFiles withTokenizerDir(Path tokenizerDir) throws IOException {
        if (tokenizerDir == null) {
            return this;
        }
        if (Files.isDirectory(tokenizerDir)) {
            HfTokenizerFiles tokenizer = locate(tokenizerDir);
            Map<String, Path> merged = new TreeMap<>(tokenizer.files);
            for (String name : MODEL_FILES) {
                merged.remove(name);
                Path own = files.get(name);
                if (own != null) {
                    merged.put(name, own);
                }
            }
            return new HfTokenizerFiles(tokenizer.dir, Collections.unmodifiableMap(merged),
                    tokenizer.additionalChatTemplates, null, Collections.emptyMap(), Collections.emptyMap(),
                    Collections.emptyMap());
        }
        if (!Files.isRegularFile(tokenizerDir) || !Files.isReadable(tokenizerDir)) {
            throw new FileNotFoundException("tokenizer source '" + tokenizerDir + "' does not exist or is not"
                    + " readable; pass a tokenizer directory or a GGUF file");
        }
        ModelFormat format;
        try {
            format = ModelSources.detect(tokenizerDir);
        } catch (FileNotFoundException e) {
            throw new IOException("tokenizer source '" + tokenizerDir + "' is neither a tokenizer directory nor a"
                    + " GGUF file", e);
        }
        if (format != ModelFormat.GGUF) {
            throw new IOException("tokenizer source '" + tokenizerDir + "' is a " + format + " file; pass a"
                    + " tokenizer directory or a GGUF file");
        }
        Map<String, Object> keys = new TreeMap<>();
        Map<String, GGUFValueType> types = new HashMap<>();
        Map<String, GGUFValueType> elementTypes = new HashMap<>();
        try (ModelSource source = ModelSources.open(tokenizerDir.toString(), ModelSourceOptions.defaults())) {
            for (Map.Entry<String, Object> e : source.getMetadata().entrySet()) {
                String key = e.getKey();
                if (!key.startsWith(TOKENIZER_KEY_PREFIX)) {
                    continue;
                }
                Object value = e.getValue();
                keys.put(key, value instanceof List ? Collections.unmodifiableList((List<?>) value) : value);
                types.put(key, source.getValueType(key));
                GGUFValueType element = source.getArrayElementType(key);
                if (element != null) {
                    elementTypes.put(key, element);
                }
            }
        }
        if (keys.isEmpty()) {
            throw new IOException("tokenizer source '" + tokenizerDir + "' holds no tokenizer.* metadata");
        }
        return new HfTokenizerFiles(dir, files, additionalChatTemplates, tokenizerDir,
                Collections.unmodifiableMap(keys), Collections.unmodifiableMap(types),
                Collections.unmodifiableMap(elementTypes));
    }

    // ------------------------------------------------------------------ located files

    /** The directory the files were located in; null for {@link #empty()}. */
    public Path dir() { return dir; }

    /** Whether no directory was located and no GGUF tokenizer source is present, as for {@link #empty()}. */
    public boolean isEmpty() {
        return dir == null && ggufTokenizerFile == null;
    }

    /**
     * The regular file named {@code name} in {@link #dir()}, or null; after {@link #withTokenizerDir(Path)} with a
     * directory, {@code generation_config.json} and {@code processor_config.json} are the model directory's.
     */
    public Path file(String name) { return files.get(name); }

    public Path tokenizerJson() { return files.get(TOKENIZER_JSON); }

    public Path tokenizerConfigJson() { return files.get(TOKENIZER_CONFIG_JSON); }

    public Path specialTokensMapJson() { return files.get(SPECIAL_TOKENS_MAP_JSON); }

    public Path addedTokensJson() { return files.get(ADDED_TOKENS_JSON); }

    /** {@code tokenizer.model}, else {@code spiece.model} (the converter's T5 order), or null. */
    public Path sentencePieceModel() {
        Path p = files.get(TOKENIZER_MODEL);
        return p != null ? p : files.get(SPIECE_MODEL);
    }

    /** {@code sentencepiece.bpe.model} (XLM-RoBERTa), or null. */
    public Path sentencePieceBpeModel() { return files.get(SENTENCEPIECE_BPE_MODEL); }

    public Path vocabJson() { return files.get(VOCAB_JSON); }

    public Path mergesTxt() { return files.get(MERGES_TXT); }

    public Path vocabTxt() { return files.get(VOCAB_TXT); }

    /** Every {@code *.tiktoken} file and {@code tiktoken.model}, sorted by name. */
    public List<Path> tiktokenFiles() {
        return filesMatching(TIKTOKEN_FILE);
    }

    /** {@code qwen.tiktoken}, else {@code tiktoken.model}, else the first of {@link #tiktokenFiles()}, or null. */
    public Path tiktokenFile() {
        List<Path> tiktokenFiles = tiktokenFiles();
        Path p = named(tiktokenFiles, QWEN_TIKTOKEN);
        p = p != null ? p : named(tiktokenFiles, TIKTOKEN_MODEL);
        return p != null || tiktokenFiles.isEmpty() ? p : tiktokenFiles.get(0);
    }

    public Path tekkenJson() { return files.get(TEKKEN_JSON); }

    /**
     * The file mistral_common loads for a Mistral-format tokenizer: {@code tekken.json} when present, else the last by
     * name of the SentencePiece files named {@code <stem>.model} or {@code <stem>.model.v<N>[m<N>]}; null when none.
     */
    public Path mistralTokenizerFile() {
        Path tekken = tekkenJson();
        List<Path> mistralModels = filesMatching(MISTRAL_SPM);
        return tekken != null || mistralModels.isEmpty() ? tekken : mistralModels.get(mistralModels.size() - 1);
    }

    /** Every {@code rwkv_vocab_v*.txt}, sorted by name. */
    public List<Path> rwkvVocabFiles() {
        return filesMatching(RWKV_VOCAB);
    }

    /** {@code rwkv_vocab_v20230424.txt} (the converter's file), else the last of {@link #rwkvVocabFiles()}, or null. */
    public Path rwkvVocabFile() {
        List<Path> rwkvVocabFiles = rwkvVocabFiles();
        Path p = named(rwkvVocabFiles, RWKV_WORLD_VOCAB);
        return p != null || rwkvVocabFiles.isEmpty() ? p : rwkvVocabFiles.get(rwkvVocabFiles.size() - 1);
    }

    /** PLaMo's {@code tokenizer.jsonl}, or null. */
    public Path plamoJsonl() { return files.get(PLAMO_JSONL); }

    public Path chatTemplateJinja() { return files.get(CHAT_TEMPLATE_JINJA); }

    public Path chatTemplateJson() { return files.get(CHAT_TEMPLATE_JSON); }

    /** {@code additional_chat_templates/*.jinja}, sorted by name. */
    public List<Path> additionalChatTemplates() { return additionalChatTemplates; }

    public Path processorConfigJson() { return files.get(PROCESSOR_CONFIG_JSON); }

    public Path generationConfigJson() { return files.get(GENERATION_CONFIG_JSON); }

    // ------------------------------------------------------------------ GGUF tokenizer source

    /** Whether a GGUF tokenizer source supplies the tokenizer metadata. */
    public boolean hasGgufTokenizer() { return ggufTokenizerFile != null; }

    /** The GGUF tokenizer source, or null. */
    public Path ggufTokenizerFile() { return ggufTokenizerFile; }

    /** The {@code tokenizer.*} metadata of the GGUF tokenizer source, sorted by key (unmodifiable); empty if none. */
    public Map<String, Object> ggufTokenizerKeys() { return ggufKeys; }

    /** The GGUF value type of a key of {@link #ggufTokenizerKeys()}, or null. */
    public GGUFValueType ggufValueType(String key) { return ggufTypes.get(key); }

    /** The element type of an array key of {@link #ggufTokenizerKeys()}, or null. */
    public GGUFValueType ggufArrayElementType(String key) { return ggufElementTypes.get(key); }

    // ------------------------------------------------------------------ typed reads

    /** The parsed {@code tokenizer.json}, or null when it is absent. */
    public HfTokenizerJson readTokenizerJson() throws IOException {
        Path p = tokenizerJson();
        return p == null ? null : HfTokenizerJson.parse(p);
    }

    /** {@code tokenizer_config.json}, or an empty view when it is absent. */
    public JsonView readTokenizerConfig() throws IOException { return readOptionalObject(TOKENIZER_CONFIG_JSON); }

    /** {@code special_tokens_map.json}, or an empty view when it is absent. */
    public JsonView readSpecialTokensMap() throws IOException { return readOptionalObject(SPECIAL_TOKENS_MAP_JSON); }

    /** {@code generation_config.json}, or an empty view when it is absent. */
    public JsonView readGenerationConfig() throws IOException { return readOptionalObject(GENERATION_CONFIG_JSON); }

    /** {@code processor_config.json}, or an empty view when it is absent. */
    public JsonView readProcessorConfig() throws IOException { return readOptionalObject(PROCESSOR_CONFIG_JSON); }

    /** {@code chat_template.json}, or an empty view when it is absent. */
    public JsonView readChatTemplateJson() throws IOException { return readOptionalObject(CHAT_TEMPLATE_JSON); }

    /** {@code added_tokens.json} (token → id, file order), or an empty map when it is absent. */
    public Map<String, Integer> readAddedTokens() throws IOException { return readTokenIds(addedTokensJson()); }

    /** {@code vocab.json} (token → id, file order), or an empty map when it is absent. */
    public Map<String, Integer> readVocabJson() throws IOException { return readTokenIds(vocabJson()); }

    /** {@code chat_template.jinja} as Python text mode reads it, or null when it is absent. */
    public String readChatTemplate() throws IOException {
        Path p = chatTemplateJinja();
        return p == null ? null : readText(p);
    }

    /** {@code additional_chat_templates/*.jinja} as file stem → template, sorted by file name. */
    public Map<String, String> readAdditionalChatTemplates() throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        for (Path p : additionalChatTemplates) {
            String name = p.getFileName().toString();
            out.put(name.substring(0, name.length() - JINJA_SUFFIX.length()), readText(p));
        }
        return Collections.unmodifiableMap(out);
    }

    /**
     * {@code merges.txt} as llama.cpp's {@code SpecialVocab._try_load_merges_txt} reads it (vocab.py:130-153): a first
     * line starting with {@code #} is a header; blank lines are skipped; a line that does not split into exactly two
     * whitespace-separated parts is ignored with a warning. Returns {@code "a b"} merges; empty when the file is
     * absent.
     */
    public List<String> readMergesTxt() throws IOException {
        Path p = mergesTxt();
        if (p == null) {
            return Collections.emptyList();
        }
        List<String> lines = PythonText.lines(readText(p));
        List<String> merges = new ArrayList<>(lines.size());
        int malformed = 0;
        int firstMalformed = 0;
        for (int i = 0; i < lines.size(); i++) {
            String line = PythonText.strip(lines.get(i));
            if ((i == 0 && line.startsWith("#")) || line.isEmpty()) {
                continue;
            }
            List<String> parts = PythonText.split(line, 3);
            if (parts.size() != 2) {
                if (malformed++ == 0) {
                    firstMalformed = i + 1;
                }
                continue;
            }
            merges.add(parts.get(0) + " " + parts.get(1));
        }
        if (malformed > 0) {
            LOGGER.warn("{}: ignored {} malformed merge line(s), the first at line {}", p, malformed, firstMalformed);
        }
        return Collections.unmodifiableList(merges);
    }

    /**
     * {@code vocab.txt} as transformers' WordPiece {@code load_vocab} reads it: one token per line, the line index is
     * its id. Empty when the file is absent.
     */
    public List<String> readVocabTxt() throws IOException {
        Path p = vocabTxt();
        return p == null ? Collections.emptyList() : Collections.unmodifiableList(PythonText.lines(readText(p)));
    }

    /**
     * The file named by the string member {@code key} of {@code tokenizer_config.json} (such as {@code vocab_file}),
     * resolved under {@link #dir()} by {@link #resolveContentName}; null when the configuration or the member is
     * absent or null.
     *
     * @throws IOException when the member is not a string, the name breaks the content-name rule, or the named file
     *                     does not exist
     */
    public Path configuredFile(String key) throws IOException {
        Objects.requireNonNull(key, "key");
        Path config = tokenizerConfigJson();
        if (config == null) {
            return null;
        }
        JsonView v = readJsonObject(config);
        String name;
        try {
            name = v.optStr(key, null);
        } catch (JsonException e) {
            throw invalid(config, e);
        }
        if (name == null) {
            return null;
        }
        Path resolved = resolveContentName(dir, name, v.where(key));
        if (!Files.isRegularFile(resolved)) {
            throw invalid(config, key + " names '" + name + "', which is not a file of the checkpoint");
        }
        return resolved;
    }

    /**
     * Resolves a file name taken from file contents under {@code dir}. The name must be non-empty and relative, with
     * no NUL, no root or drive, no {@code ..} segment (with {@code /} or {@code \} separators), and must stay under
     * {@code dir} after normalization. Links in the resolved path are not resolved here and are followed when read.
     *
     * @param where names the referencing member in messages, such as {@code tokenizer_config.json:vocab_file}
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <where>: ..."} when the rule is broken
     */
    public static Path resolveContentName(Path dir, String name, String where) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(name, "name");
        String problem = contentNameProblem(name);
        Path resolved = null;
        if (problem == null) {
            try {
                Path rel = dir.getFileSystem().getPath(name);
                Path base = dir.normalize();
                resolved = base.resolve(rel).normalize();
                if (rel.isAbsolute() || rel.getRoot() != null) {
                    problem = "is absolute";
                } else if (!resolved.toAbsolutePath().startsWith(base.toAbsolutePath())) {
                    problem = "escapes the model directory";
                }
            } catch (InvalidPathException e) {
                problem = "is not a valid path (" + e.getReason() + ")";
            }
        }
        if (problem != null) {
            throw new IOException("Invalid Hugging Face checkpoint '" + dir + "': " + where + ": file name '"
                    + printable(name) + "' " + problem);
        }
        return resolved;
    }

    @Override
    public String toString() {
        return "HfTokenizerFiles{" + (dir == null ? "<none>" : dir) + ", " + files.keySet()
                + ", additional_chat_templates=" + additionalChatTemplates.size()
                + (ggufTokenizerFile != null ? ", gguf=" + ggufTokenizerFile : "") + '}';
    }

    // ------------------------------------------------------------------ shared file helpers

    /**
     * The JSON object in {@code file} ({@link Json.Limits#LENIENT}, at most {@link #MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': ..."} when it is not one JSON object
     */
    public static JsonView readJsonObject(Path file) throws IOException {
        try {
            return JsonView.of(Json.parse(file, MAX_FILE_BYTES, Json.Limits.LENIENT), file.toString());
        } catch (JsonException e) {
            throw invalid(file, e);
        }
    }

    /**
     * The text of {@code file} as Python text mode reads it with {@code encoding="utf-8"}: strict UTF-8, line endings
     * {@code \r\n} and {@code \r} translated to {@code \n}, a byte order mark kept as U+FEFF.
     */
    public static String readText(Path file) throws IOException {
        return PythonText.translateNewlines(decodeUtf8(readBytes(file), file));
    }

    /** The bytes of {@code file}, at most {@link #MAX_FILE_BYTES}. */
    static byte[] readBytes(Path file) throws IOException {
        long size = Files.size(file);
        if (size > MAX_FILE_BYTES) {
            throw invalid(file, "size " + size + " bytes exceeds the limit of " + MAX_FILE_BYTES + " bytes");
        }
        try (InputStream in = Files.newInputStream(file)) {
            byte[] bytes = in.readNBytes((int) MAX_FILE_BYTES);
            if (in.read() != -1) {
                throw invalid(file, "size exceeds the limit of " + MAX_FILE_BYTES + " bytes");
            }
            return bytes;
        }
    }

    /** Strict UTF-8 decoding; malformed input is an error naming the byte offset. */
    static String decodeUtf8(byte[] bytes, Path file) throws IOException {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        ByteBuffer in = ByteBuffer.wrap(bytes);
        try {
            return decoder.decode(in).toString();
        } catch (CharacterCodingException e) {
            throw invalid(file, "invalid UTF-8 byte sequence at byte " + in.position());
        }
    }

    /** {@code "Invalid Hugging Face checkpoint '<dir of file>': <file>: <detail>"}. */
    static IOException invalid(Path file, String detail) {
        return new IOException(filePrefix(file) + detail);
    }

    /** {@code "Invalid Hugging Face checkpoint '<dir of file>': <file>: "}, the prefix of messages about a file. */
    static String filePrefix(Path file) {
        return checkpointPrefix(file) + file + ": ";
    }

    /** {@code "Invalid Hugging Face checkpoint '<dir of file>': <JSON message>"}; the JSON message names the file. */
    static IOException invalid(Path file, JsonException e) {
        return new IOException(checkpointPrefix(file) + e.getMessage(), e);
    }

    private static String checkpointPrefix(Path file) {
        Path parent = file.getParent() != null ? file.getParent() : file.toAbsolutePath().getParent();
        return "Invalid Hugging Face checkpoint '" + parent + "': ";
    }

    // ------------------------------------------------------------------ internals

    private JsonView readOptionalObject(String name) throws IOException {
        Path p = files.get(name);
        return p == null ? JsonView.empty(name, "") : readJsonObject(p);
    }

    /** A JSON object of token → id in file order; ids are integers in [0, {@link #MAX_TOKENS}). */
    private static Map<String, Integer> readTokenIds(Path file) throws IOException {
        if (file == null) {
            return Collections.emptyMap();
        }
        JsonView v = readJsonObject(file);
        Map<String, Integer> out = new LinkedHashMap<>();
        try {
            for (String token : v.keys()) {
                int id = v.i32(token);
                if (id < 0 || id >= MAX_TOKENS) {
                    throw new JsonException(v.where(token) + ": token id " + id + " is outside [0, " + MAX_TOKENS
                            + ")");
                }
                out.put(token, id);
            }
        } catch (JsonException e) {
            throw invalid(file, e);
        }
        return Collections.unmodifiableMap(out);
    }

    /** Regular files (links followed) of {@code dir} by name, optionally only those ending in {@code suffix}. */
    private static TreeMap<String, Path> regularFiles(Path dir, String suffix) throws IOException {
        TreeMap<String, Path> out = new TreeMap<>();
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(dir)) {
            for (Path entry : entries) {
                String name = entry.getFileName().toString();
                if ((suffix == null || name.endsWith(suffix)) && Files.isRegularFile(entry)) {
                    out.put(name, entry);
                }
            }
        } catch (DirectoryIteratorException e) {
            throw e.getCause();
        }
        return out;
    }

    private static String contentNameProblem(String name) {
        if (name.isEmpty()) {
            return "is empty";
        }
        if (name.indexOf('\0') >= 0) {
            return "contains NUL";
        }
        if (name.startsWith("/") || name.startsWith("\\") || WINDOWS_DRIVE.matcher(name).matches()) {
            return "is absolute";
        }
        for (String segment : name.split("[/\\\\]", -1)) {
            if (segment.equals("..")) {
                return "contains a '..' segment";
            }
        }
        return null;
    }

    /** The located files whose names match {@code pattern}, in name order. */
    private List<Path> filesMatching(Pattern pattern) {
        List<Path> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : files.entrySet()) {
            if (pattern.matcher(e.getKey()).matches()) {
                out.add(e.getValue());
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** The entry of {@code files} whose file name is {@code name}, or null. */
    private static Path named(List<Path> files, String name) {
        for (Path p : files) {
            if (p.getFileName().toString().equals(name)) {
                return p;
            }
        }
        return null;
    }

    private static String printable(String name) {
        return name.replace("\0", "\\0");
    }

}
