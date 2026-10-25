package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Checkpoints whose configuration is not a plain {@code config.json}, recognized by marker files and turned into a
 * synthesized configuration root, in the order of the llama.cpp converter's {@code _hparams_loaders}
 * (LC/conversion: qwen.py OpenJev, bert.py decision, clef.py, h2o_lightning.py, lev.py Lev/Kev/Nimble, lfm2.py
 * D1/D1-omni, pockettts.py, pplx_decider.py, vllm_d3.py, vllm_decision2.py). The first detector that matches wins.
 *
 * <p>LoRA-style decision checkpoints (Lev, Kev, Nimble) read the configuration of their base model, which must be
 * local: {@link ModelSourceOptions#baseDir()}, else {@code adapter_config.json} {@code base_model_name_or_path} when
 * it names an existing directory (absolute, or relative to the model directory). Models are never downloaded.</p>
 */
public final class HfCheckpointDetectors {

    private static final Logger LOG = LogManager.getLogger(HfCheckpointDetectors.class);

    /** Largest accepted safetensors header, in bytes. */
    static final long MAX_SAFETENSORS_HEADER = 100_000_000L;
    private static final int MAX_RANK = 8;
    private static final Pattern POCKET_LAYER = Pattern.compile("flow_lm\\.transformer\\.layers\\.\\d+\\.norm1\\.weight");
    private static final Pattern POCKET_AUDIO_LAYER =
            Pattern.compile("mimi\\.encoder_transformer\\.transformer\\.layers\\.\\d+\\.norm1\\.weight");
    private static final String POCKET_MARKER = "flow_lm.bos_emb";

    private static final List<Detector> DETECTORS = List.of(
            new Detector("OpenJev", p -> p.isFile("helper/shim.py") && p.isFile(HfConfig.CONFIG_JSON),
                    p -> plain(p, "OpenJevModel")),
            new Detector("ModernBertDecision", p -> p.isFile("encoder/config.json")
                    && (p.isFile("rl_agent_config.json") || p.isFile("julia_config.json")),
                    HfCheckpointDetectors::modernBertDecision),
            new Detector("Clef", p -> p.isFile("joint_head_config.json") && p.isFile(HfConfig.CONFIG_JSON),
                    p -> withDecision(plain(p, "ClefModel"), p, "joint_head_config.json")),
            new Detector("H2OLightning", p -> p.isFile("h2o_lightning_shim.py") && p.isFile("serve_config.json"),
                    p -> plain(p, "H2OLightningModel")),
            new Detector("Lev", p -> p.isFile("lev_release.json"), p -> decisionLora(p, "LevModel")),
            new Detector("Kev", HfCheckpointDetectors::isKev, p -> decisionLora(p, "KevModel")),
            new Detector("Nimble", p -> p.isFile("adapter_config.json") && p.isFile("schema_config.json")
                    && "schema_candidate_classification_v2".equals(p.object("schema_config.json").get("task")),
                    p -> decisionLora(p, "NimbleModel")),
            new Detector("D1", HfCheckpointDetectors::isD1, HfCheckpointDetectors::d1),
            new Detector("D1Omni", p -> p.isFile(HfConfig.CONFIG_JSON)
                    && "d1_omni".equals(p.object(HfConfig.CONFIG_JSON).get("model_type")),
                    HfCheckpointDetectors::d1Omni),
            new Detector("PocketTTS", HfCheckpointDetectors::isPocketTts, HfCheckpointDetectors::pocketTts),
            new Detector("PplxDecider", p -> p.isFile("decision_config.json") && p.isFile("readout.safetensors")
                    && p.isFile(HfConfig.CONFIG_JSON) && !p.object("decision_config.json").containsKey("format_id"),
                    p -> withDecision(plain(p, "PplxDeciderModel"), p, "decision_config.json")),
            new Detector("VllmD3", p -> p.isFile("decision_config.json")
                    && "d3-code-readout-v1".equals(p.object("decision_config.json").get("format_id")),
                    p -> withDecision(plain(p, "VllmD3Model"), p, "decision_config.json")),
            new Detector("VllmDecision2", p -> p.isFile(HfConfig.CONFIG_JSON)
                    && "vllm-sr-decision".equals(p.object(HfConfig.CONFIG_JSON).get("decision_format")),
                    HfCheckpointDetectors::vllmDecision2));

    private HfCheckpointDetectors() {
    }

    /** The detector names, in matching order. */
    public static List<String> names() {
        return DETECTORS.stream().map(d -> d.name).collect(Collectors.toUnmodifiableList());
    }

    /**
     * The name of the first detector that recognizes {@code dir}, reading only marker files (and, for Pocket TTS,
     * the header of its single {@code model*.safetensors} part).
     *
     * @throws IOException when a marker file the decision depends on cannot be read or is invalid
     */
    public static Optional<String> match(Path dir) throws IOException {
        Probe probe = new Probe(dir, null);
        for (Detector d : DETECTORS) {
            if (d.matcher.matches(probe)) {
                return Optional.of(d.name);
            }
        }
        return Optional.empty();
    }

    /** The configuration synthesized by the first matching detector, if any. */
    static Optional<Detection> detect(Path dir, Path baseDir) throws IOException {
        Probe probe = new Probe(dir, baseDir);
        for (Detector d : DETECTORS) {
            if (d.matcher.matches(probe)) {
                LOG.info("Detected {} checkpoint in {}", d.name, dir);
                return Optional.of(d.loader.load(probe));
            }
        }
        return Optional.empty();
    }

    /** A synthesized configuration root (mutable until the loader freezes it). */
    static final class Detection {
        final Map<String, Object> root;
        final String source;
        final Path baseModelDir;

        Detection(Map<String, Object> root, String source, Path baseModelDir) {
            this.root = root;
            this.source = source;
            this.baseModelDir = baseModelDir;
        }
    }

    // ------------------------------------------------------------------ detectors

    private static Detection plain(Probe p, String arch) throws IOException {
        Map<String, Object> root = HfConfigLoader.readConfigJson(p.dir, p.dir);
        root.put("architectures", architectures(arch));
        return new Detection(root, HfConfig.CONFIG_JSON, null);
    }

    private static Detection withDecision(Detection d, Probe p, String decisionFile) throws IOException {
        d.root.put("decision", HfConfig.mutableCopy(p.object(decisionFile)));
        return d;
    }

    /** bert.py _load_decision_hparams: the encoder config with the decision head blocks appended. */
    private static Detection modernBertDecision(Probe p) throws IOException {
        String source = "encoder/config.json";
        Map<String, Object> root = HfConfigLoader.readConfigJson(p.dir, p.dir.resolve("encoder"));
        String decisionFile = p.isFile("julia_config.json") ? "julia_config.json" : "rl_agent_config.json";
        @SuppressWarnings("unchecked")
        Map<String, Object> decision = (Map<String, Object>) HfConfig.mutableCopy(p.object(decisionFile));
        long nLayer = require(root, "num_hidden_layers", source);
        long nHead = require(decision, "head_layers", decisionFile);
        checkLayers(nLayer, nHead, source);
        long nFf = require(root, "intermediate_size", source);
        long nEmbd = require(root, "hidden_size", source);
        root.put("architectures", architectures("ModernBertDecisionModel"));
        root.put("decision", decision);
        root.put("num_hidden_layers", nLayer + nHead);
        root.put("intermediate_size", repeatPair(nFf, nLayer, Math.multiplyExact(4L, nEmbd), nHead));
        return new Detection(root, source, null);
    }

    private static boolean isKev(Probe p) throws IOException {
        if (!p.isFile("adapter_config.json") || !p.isFile("head.pt") || !p.isFile("training_config.json")) {
            return false;
        }
        Object args = p.object("training_config.json").get("args");
        return args instanceof Map && ((Map<?, ?>) args).containsKey("head_dim");
    }

    private static boolean isD1(Probe p) throws IOException {
        if (!p.isFile(HfConfig.CONFIG_JSON)) {
            return false;
        }
        Object autoMap = p.object(HfConfig.CONFIG_JSON).get("auto_map");
        Object autoModel = autoMap instanceof Map ? ((Map<?, ?>) autoMap).get("AutoModel") : null;
        return autoModel instanceof String && ((String) autoModel).endsWith(".D1Model");
    }

    /** lfm2.py _load_d1_hparams: the text model is D1, the vision projector stays LFM2-VL. */
    private static Detection d1(Probe p) throws IOException {
        Map<String, Object> root = HfConfigLoader.readConfigJson(p.dir, p.dir);
        textConfig(root, HfConfig.CONFIG_JSON).put("architectures", architectures("D1Model"));
        return new Detection(root, HfConfig.CONFIG_JSON, null);
    }

    /** lfm2.py _load_d1_omni_hparams: LFM2 FFN sizing for the trunk, plain 4x MLP head blocks appended. */
    private static Detection d1Omni(Probe p) throws IOException {
        String source = HfConfig.CONFIG_JSON;
        Map<String, Object> root = HfConfigLoader.readConfigJson(p.dir, p.dir);
        Map<String, Object> text = textConfig(root, source);
        String textSource = source + " text_config";
        long nLayer = require(text, "num_hidden_layers", textSource);
        long nHead = require(root, "head_layers", source);
        checkLayers(nLayer, nHead, source);
        Object multiplierValue = text.get("block_ffn_dim_multiplier");
        if (multiplierValue == null) {
            throw new GgufFormatException(textSource + ": required key 'block_ffn_dim_multiplier' is missing");
        }
        double multiplier = HfConfig.toDouble(textSource, "block_ffn_dim_multiplier", multiplierValue);
        long intermediate = require(text, "intermediate_size", textSource);
        long multipleOf = require(text, "block_multiple_of", textSource);
        long hidden = require(text, "hidden_size", textSource);
        if (multipleOf <= 0 || intermediate < 0 || intermediate > HfConfig.MAX_WIDTH || multiplier < 0
                || multiplier > HfConfig.MAX_WIDTH) {
            throw new GgufFormatException(textSource + ": block_multiple_of, intermediate_size and "
                    + "block_ffn_dim_multiplier must be positive widths");
        }
        long nFf = (long) (multiplier * (long) (2.0 * intermediate / 3.0));
        nFf = multipleOf * ((nFf + multipleOf - 1) / multipleOf);
        text.put("num_hidden_layers", nLayer + nHead);
        text.put("intermediate_size", repeatPair(nFf, nLayer, Math.multiplyExact(4L, hidden), nHead));
        text.put("block_auto_adjust_ff_dim", Boolean.FALSE);
        return new Detection(root, source, null);
    }

    private static boolean isPocketTts(Probe p) throws IOException {
        Optional<Path> part = p.singleModelPart();
        return part.isPresent() && p.mayHoldTensor(part.get(), POCKET_MARKER)
                && p.shapes(part.get()).containsKey(POCKET_MARKER);
    }

    /** pockettts.py _load_hparams: everything derived from tensor shapes. */
    private static Detection pocketTts(Probe p) throws IOException {
        Path part = p.singleModelPart().orElseThrow(() -> new IllegalStateException("matched without a part"));
        Map<String, List<Long>> shapes = p.shapes(part);
        List<Long> embed = shape(shapes, "flow_lm.conditioner.embed.weight", 2, p.dir);
        long nVocab = embed.get(0);
        long nEmbd = embed.get(1);
        long nEmbdAudio = shape(shapes, "mimi.encoder_transformer.transformer.layers.0.norm1.weight", 1, p.dir).get(0);
        Map<String, Object> audio = new LinkedHashMap<>();
        audio.put("num_hidden_layers", count(shapes, POCKET_AUDIO_LAYER));
        audio.put("hidden_size", nEmbdAudio);
        audio.put("intermediate_size",
                shape(shapes, "mimi.encoder_transformer.transformer.layers.0.linear1.weight", 2, p.dir).get(0));
        audio.put("num_attention_heads", nEmbdAudio / 64);
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("architectures", architectures("PocketTTSModel"));
        root.put("model_type", "pockettts");
        root.put("num_hidden_layers", count(shapes, POCKET_LAYER));
        root.put("hidden_size", nEmbd);
        root.put("intermediate_size", shape(shapes, "flow_lm.transformer.layers.0.linear1.weight", 2, p.dir).get(0));
        root.put("max_position_embeddings", 4096L);
        root.put("num_attention_heads", nEmbd / 64);
        root.put("vocab_size", nVocab + (shapes.containsKey("flow_lm.bos_before_voice") ? 2L : 1L));
        root.put("rope_theta", 10000.0);
        root.put("layer_norm_eps", 1e-5);
        root.put("audio_config", audio);
        return new Detection(root, part.getFileName() + " (tensor shapes)", null);
    }

    /** vllm_decision2.py: the backbone config named by the package, the head config and the score offsets. */
    private static Detection vllmDecision2(Probe p) throws IOException {
        Map<String, Object> pkg = p.object(HfConfig.CONFIG_JSON);
        Object backbone = pkg.get("backbone");
        Object backboneConfig = backbone instanceof Map ? ((Map<?, ?>) backbone).get("config") : null;
        Path backboneDir = HfConfig.contentPath(p.dir, requireName(backboneConfig, "backbone.config")).getParent();
        Map<String, Object> root = HfConfigLoader.readConfigJson(p.dir, backboneDir);
        root.put("architectures", architectures("VllmDecision2Model"));
        Path modelConfig = HfConfig.contentPath(p.dir, requireName(pkg.get("model_config"), "model_config"));
        @SuppressWarnings("unchecked")
        Map<String, Object> decision = (Map<String, Object>) HfConfig.mutableCopy(p.object(modelConfig));
        Map<String, Object> scoreBias = p.object("score_bias.json");
        if (!scoreBias.containsKey("offsets")) {
            throw new GgufFormatException("score_bias.json: required key 'offsets' is missing");
        }
        decision.put("score_bias", HfConfig.mutableCopy(scoreBias.get("offsets")));
        root.put("decision", decision);
        return new Detection(root, p.dir.normalize().relativize(backboneDir.resolve(HfConfig.CONFIG_JSON))
                .toString(), null);
    }

    /** lev.py _load_decision_lora_hparams: the base model's config.json, read as is, with {@code arch}. */
    private static Detection decisionLora(Probe p, String arch) throws IOException {
        Object name = p.object("adapter_config.json").get("base_model_name_or_path");
        Path base = baseModelDir(p, name);
        Path baseConfig = base.resolve(HfConfig.CONFIG_JSON);
        if (!Files.isRegularFile(baseConfig)) {
            throw new IOException("Invalid Hugging Face checkpoint '" + p.dir + "': the base model directory '"
                    + base + "' has no config.json");
        }
        Map<String, Object> root = HfConfigLoader.readObject(p.dir, baseConfig);
        root.put("architectures", architectures(arch));
        return new Detection(root, baseConfig.toString(), base);
    }

    private static Path baseModelDir(Probe p, Object name) throws IOException {
        if (p.baseDir != null) {
            if (!Files.isDirectory(p.baseDir)) {
                throw new IOException("Invalid Hugging Face checkpoint '" + p.dir + "': the base model directory '"
                        + p.baseDir + "' (" + ModelSourceOptions.PROP_BASE_DIR + ") is not a directory");
            }
            return p.baseDir;
        }
        String advice = "; pass --hf-base-dir (" + ModelSourceOptions.PROP_BASE_DIR
                + ") with a local copy of the base model (models are never downloaded)";
        if (!(name instanceof String) || ((String) name).isEmpty() || ((String) name).indexOf('\0') >= 0) {
            throw new IOException("Invalid Hugging Face checkpoint '" + p.dir
                    + "': adapter_config.json has no usable 'base_model_name_or_path'" + advice);
        }
        Path candidate;
        try {
            candidate = Paths.get((String) name);
        } catch (InvalidPathException e) {
            throw new IOException("Invalid Hugging Face checkpoint '" + p.dir + "': base model '" + name
                    + "' is not a local directory" + advice, e);
        }
        candidate = candidate.isAbsolute() ? candidate : p.dir.resolve(candidate).normalize();
        if (!Files.isDirectory(candidate)) {
            throw new IOException("Invalid Hugging Face checkpoint '" + p.dir + "': base model '" + name
                    + "' is not a local directory" + advice);
        }
        return candidate;
    }

    // ------------------------------------------------------------------ helpers

    private static List<Object> architectures(String arch) {
        List<Object> list = new ArrayList<>(1);
        list.add(arch);
        return list;
    }

    private static Map<String, Object> textConfig(Map<String, Object> root, String source) {
        Object text = root.get("text_config");
        if (!(text instanceof Map)) {
            throw new GgufFormatException(source + ": 'text_config' must be an object, got "
                    + HfConfig.typeName(text));
        }
        return HfConfig.asObject(text);
    }

    private static long require(Map<String, Object> map, String key, String source) {
        Object v = map.get(key);
        if (v == null) {
            throw new GgufFormatException(source + ": required key '" + key + "' is missing");
        }
        return HfConfig.toLong(source, key, v);
    }

    private static void checkLayers(long nLayer, long nHead, String source) {
        if (nLayer < 0 || nHead < 0 || nLayer + nHead > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(source + ": " + nLayer + " trunk and " + nHead
                    + " head layers are outside [0, " + HfConfig.MAX_LAYERS + "] in total");
        }
    }

    private static List<Object> repeatPair(long first, long firstCount, long second, long secondCount) {
        List<Object> out = new ArrayList<>((int) (firstCount + secondCount));
        out.addAll(Collections.nCopies((int) firstCount, first));
        out.addAll(Collections.nCopies((int) secondCount, second));
        return out;
    }

    private static String requireName(Object v, String key) {
        if (!(v instanceof String)) {
            throw new GgufFormatException(HfConfig.CONFIG_JSON + ": '" + key + "' must be a file name, got "
                    + HfConfig.typeName(v));
        }
        return (String) v;
    }

    private static List<Long> shape(Map<String, List<Long>> shapes, String name, int rank, Path dir)
            throws IOException {
        List<Long> s = shapes.get(name);
        if (s == null || s.size() != rank) {
            throw new IOException("Invalid Hugging Face checkpoint '" + dir + "': pocket-tts tensor '" + name + "' "
                    + (s == null ? "is missing" : "has rank " + s.size() + ", expected " + rank));
        }
        return s;
    }

    private static long count(Map<String, List<Long>> shapes, Pattern p) {
        return shapes.keySet().stream().filter(n -> p.matcher(n).matches()).count();
    }

    // ------------------------------------------------------------------ probing

    @FunctionalInterface
    private interface Matcher {
        boolean matches(Probe p) throws IOException;
    }

    @FunctionalInterface
    private interface Loader {
        Detection load(Probe p) throws IOException;
    }

    private static final class Detector {
        final String name;
        final Matcher matcher;
        final Loader loader;

        Detector(String name, Matcher matcher, Loader loader) {
            this.name = name;
            this.matcher = matcher;
            this.loader = loader;
        }
    }

    /** One directory under inspection; parsed marker files and safetensors headers are read once. */
    private static final class Probe {
        final Path dir;
        final Path baseDir;
        private final Map<Path, Map<String, Object>> objects = new HashMap<>();
        private final Map<Path, byte[]> headers = new HashMap<>();
        private final Map<Path, Map<String, List<Long>>> shapes = new HashMap<>();
        private Optional<Path> modelPart;

        Probe(Path dir, Path baseDir) {
            this.dir = dir;
            this.baseDir = baseDir;
        }

        boolean isFile(String relative) {
            return Files.isRegularFile(dir.resolve(relative));
        }

        Map<String, Object> object(String relative) throws IOException {
            return object(dir.resolve(relative));
        }

        Map<String, Object> object(Path file) throws IOException {
            Map<String, Object> cached = objects.get(file);
            if (cached == null) {
                cached = HfConfig.freeze(HfConfigLoader.readObject(dir, file));
                objects.put(file, cached);
            }
            return cached;
        }

        /** The only {@code model*.safetensors} file of the directory (base.py get_model_part_names). */
        Optional<Path> singleModelPart() throws IOException {
            if (modelPart == null) {
                List<Path> parts;
                try (Stream<Path> s = Files.list(dir)) {
                    parts = s.filter(f -> {
                        String n = f.getFileName().toString();
                        return n.startsWith("model") && n.endsWith(".safetensors") && Files.isRegularFile(f);
                    }).sorted().collect(Collectors.toList());
                }
                modelPart = parts.size() == 1 ? Optional.of(parts.get(0)) : Optional.empty();
            }
            return modelPart;
        }

        /**
         * False only when the header cannot name tensor {@code name}: it has no escape sequence and does not contain
         * the quoted name.
         */
        boolean mayHoldTensor(Path file, String name) throws IOException {
            byte[] header = header(file);
            byte[] quoted = ('"' + name + '"').getBytes(StandardCharsets.UTF_8);
            return indexOf(header, (byte) '\\') >= 0 || indexOf(header, quoted) >= 0;
        }

        Map<String, List<Long>> shapes(Path file) throws IOException {
            Map<String, List<Long>> cached = shapes.get(file);
            if (cached == null) {
                cached = parseShapes(file, header(file));
                shapes.put(file, cached);
            }
            return cached;
        }

        private byte[] header(Path file) throws IOException {
            byte[] cached = headers.get(file);
            if (cached == null) {
                cached = readHeader(file);
                headers.put(file, cached);
            }
            return cached;
        }
    }

    private static byte[] readHeader(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = ch.size();
            if (size < 8) {
                throw invalid(file, "file is " + size + " bytes, shorter than the 8-byte header length");
            }
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, len, 0, file);
            long n = len.getLong(0);
            if (n < 2 || n > MAX_SAFETENSORS_HEADER || n > size - 8) {
                throw invalid(file, "header length " + Long.toUnsignedString(n) + " is outside [2, "
                        + Math.min(MAX_SAFETENSORS_HEADER, size - 8) + "]");
            }
            ByteBuffer header = ByteBuffer.allocate((int) n);
            readFully(ch, header, 8, file);
            return header.array();
        }
    }

    private static void readFully(FileChannel ch, ByteBuffer buf, long position, Path file) throws IOException {
        long pos = position;
        while (buf.hasRemaining()) {
            int r = ch.read(buf, pos);
            if (r < 0) {
                throw invalid(file, "unexpected end of file at byte " + pos);
            }
            pos += r;
        }
    }

    private static Map<String, List<Long>> parseShapes(Path file, byte[] header) throws IOException {
        String text;
        try {
            text = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(header)).toString();
        } catch (CharacterCodingException e) {
            throw invalid(file, "header is not valid UTF-8", e);
        }
        Object parsed;
        try {
            parsed = Json.parse(text, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw invalid(file, "header is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw invalid(file, "header is not a JSON object");
        }
        Map<String, List<Long>> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : HfConfig.asObject(parsed).entrySet()) {
            if (e.getKey().equals("__metadata__")) {
                continue;
            }
            Object shape = e.getValue() instanceof Map ? ((Map<?, ?>) e.getValue()).get("shape") : null;
            if (!(shape instanceof List) || ((List<?>) shape).size() > MAX_RANK) {
                throw invalid(file, "tensor '" + e.getKey() + "' has no shape array of rank <= " + MAX_RANK);
            }
            List<Long> dims = new ArrayList<>();
            for (Object d : (List<?>) shape) {
                if (!(d instanceof Long) || (Long) d < 0) {
                    throw invalid(file, "tensor '" + e.getKey() + "' has a shape entry that is not a non-negative "
                            + "integer: " + d);
                }
                dims.add((Long) d);
            }
            out.put(e.getKey(), Collections.unmodifiableList(dims));
        }
        return Collections.unmodifiableMap(out);
    }

    private static IOException invalid(Path file, String reason) {
        return new IOException("Invalid safetensors file '" + file + "': " + reason);
    }

    private static IOException invalid(Path file, String reason, Throwable cause) {
        return new IOException("Invalid safetensors file '" + file + "': " + reason, cause);
    }

    private static int indexOf(byte[] haystack, byte b) {
        for (int i = 0; i < haystack.length; i++) {
            if (haystack[i] == b) {
                return i;
            }
        }
        return -1;
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
