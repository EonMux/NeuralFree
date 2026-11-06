package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import javax.imageio.ImageIO;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The file plumbing the dataset samplers share: JSONL, JSON, delimited and Parquet records from a
 * file or a local dataset directory, CPython-compatible shuffling, the client-side chat template
 * render, and the content parts of chat messages (text, local images as base64 data URLs, local
 * audio as an uploadable WAV). A path the reference would give a server as {@code file://} is
 * encoded here, so the benchmark works whether the server shares the client's filesystem or not.
 */
final class BenchDatasets {

    /** The row file extensions a local dataset directory may carry. */
    static final List<String> ROW_EXTENSIONS = List.of(".jsonl", ".json", ".csv", ".tsv", ".parquet");

    /** One audio file ready for a transcription upload. */
    static final class AudioFile {

        final String path;
        final byte[] bytes;
        final String filename;
        final String contentType;
        final double durationSec;

        AudioFile(String path, byte[] bytes, String filename, String contentType, double durationSec) {
            this.path = path;
            this.bytes = bytes;
            this.filename = filename;
            this.contentType = contentType;
            this.durationSec = durationSec;
        }
    }

    private BenchDatasets() {
    }

    /** The required {@code --dataset-path}, refused by name when absent or missing. */
    static Path requiredPath(ServeBenchOptions opts, String dataset) {
        if (opts.datasetPath() == null) {
            throw new IllegalArgumentException("--dataset-path must be provided for the " + dataset
                    + " dataset");
        }
        Path path;
        try {
            path = java.nio.file.Paths.get(opts.datasetPath());
        } catch (java.nio.file.InvalidPathException e) {
            throw new IllegalArgumentException("--dataset-path is not a path: " + opts.datasetPath(), e);
        }
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("the " + dataset + " dataset does not exist: " + path);
        }
        return path;
    }

    /** The directory relative paths resolve against: the path itself when it is one, else its parent. */
    static Path baseDir(Path path) {
        return Files.isDirectory(path) ? path : path.toAbsolutePath().getParent();
    }

    /** The workload size a custom dataset samples: all records when {@code --num-prompts} is zero. */
    static int targetNum(ServeBenchOptions opts, int available) {
        return opts.numPrompts() <= 0 ? available : opts.numPrompts();
    }

    /** The prompt's token count; one token when {@code --skip-tokenizer-init} disabled counting. */
    static int promptLen(ServeBenchOptions opts, BenchTokenizer tokenizer, String text) {
        return opts.skipTokenizerInit() ? 1 : tokenizer.tokenize(text).length;
    }

    /**
     * The prompt the request carries: the dataset's text rendered through the model's chat template
     * when the reference renders that dataset client-side and a template is available
     * ({@link #renderPrompt}), else the text itself.
     */
    static String renderPrompt(ServeBenchOptions opts, BenchTokenizer tokenizer, String prompt) {
        if (opts.skipChatTemplate() || opts.skipTokenizerInit()) {
            return prompt;
        }
        BenchChatTemplate template = tokenizer.chatTemplate();
        if (template == null) {
            return prompt;
        }
        return template.render(prompt, opts.chatTemplateKwargs());
    }

    /** {@link #renderPrompt(ServeBenchOptions, BenchTokenizer, String)} over a system and a user message. */
    static String renderPrompt(ServeBenchOptions opts, BenchTokenizer tokenizer, String system, String prompt) {
        if (opts.skipChatTemplate() || opts.skipTokenizerInit()) {
            return prompt;
        }
        BenchChatTemplate template = tokenizer.chatTemplate();
        if (template == null) {
            return prompt;
        }
        return template.render(system, prompt, opts.chatTemplateKwargs());
    }

    /** The rows of a file or local dataset directory, in the reference's deterministic order. */
    static List<Map<String, Object>> readRows(Path path) throws IOException {
        if (Files.isRegularFile(path)) {
            return readFile(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("the dataset path does not exist: " + path);
        }
        List<Path> files = rowFiles(path);
        if (files.isEmpty()) {
            throw new IllegalArgumentException("the dataset directory " + path + " carries no "
                    + String.join(", ", ROW_EXTENSIONS) + " file");
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Path file : files) {
            rows.addAll(readFile(file));
        }
        return rows;
    }

    /** The metadata files a saved Hugging Face dataset directory carries; they are not rows. */
    private static final List<String> METADATA_FILES = List.of("dataset_info.json", "state.json",
            "dataset_dict.json");

    /** A dataset directory's row files, sorted by name; symbolic links are followed. */
    static List<Path> rowFiles(Path dir) throws IOException {
        List<Path> out = new ArrayList<>();
        try (java.nio.file.DirectoryStream<Path> entries = Files.newDirectoryStream(dir)) {
            for (Path entry : entries) {
                String name = entry.getFileName().toString();
                if (Files.isRegularFile(entry) && hasRowExtension(name)
                        && !METADATA_FILES.contains(name)) {
                    out.add(entry);
                }
            }
        }
        out.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        return out;
    }

    static boolean hasRowExtension(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        for (String extension : ROW_EXTENSIONS) {
            if (lower.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    /** One row file's records: JSONL objects, a JSON array or object, or delimited rows with a header. */
    static List<Map<String, Object>> readFile(Path file) throws IOException {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        try {
            if (name.endsWith(".jsonl")) {
                return readJsonl(file);
            }
            if (name.endsWith(".json")) {
                return readJson(file);
            }
            if (name.endsWith(".csv") || name.endsWith(".tsv")) {
                return readDelimited(file, name.endsWith(".tsv") ? '\t' : ',');
            }
            if (name.endsWith(".parquet")) {
                try {
                    return ParquetReader.read(file);
                } catch (IllegalArgumentException e) {
                    throw new IOException("cannot read " + file + ": " + e.getMessage(), e);
                }
            }
        } catch (JsonException e) {
            throw new IOException("cannot parse " + file + ": " + e.getMessage(), e);
        }
        throw new IllegalArgumentException("unsupported dataset file " + file + ": the harness reads "
                + String.join(", ", ROW_EXTENSIONS));
    }

    private static List<Map<String, Object>> readJsonl(Path file) throws IOException, JsonException {
        List<Map<String, Object>> out = new ArrayList<>();
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            out.add(objectAt(Json.parse(line, Json.Limits.UNTRUSTED), file, i + 1));
        }
        return out;
    }

    private static List<Map<String, Object>> readJson(Path file) throws IOException, JsonException {
        Object parsed = Json.parse(Files.readString(file, StandardCharsets.UTF_8), Json.Limits.UNTRUSTED);
        if (parsed instanceof List) {
            List<Map<String, Object>> out = new ArrayList<>();
            int index = 0;
            for (Object entry : (List<?>) parsed) {
                out.add(objectAt(entry, file, ++index));
            }
            return out;
        }
        if (parsed instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) parsed;
            Object data = map.get("data");
            if (data instanceof List) {
                List<Map<String, Object>> out = new ArrayList<>();
                int index = 0;
                for (Object entry : (List<?>) data) {
                    out.add(objectAt(entry, file, ++index));
                }
                return out;
            }
            List<Map<String, Object>> single = new ArrayList<>();
            single.add(objectAt(map, file, 1));
            return single;
        }
        throw new IOException(file + " does not hold a JSON array or object");
    }

    private static Map<String, Object> objectAt(Object value, Path file, int line) throws IOException {
        if (!(value instanceof Map)) {
            throw new IOException(line + " of " + file + " is not a JSON object");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
            out.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return out;
    }

    /** Delimited rows: the first line is the header, quoted fields and doubled quotes supported. */
    static List<Map<String, Object>> readDelimited(Path file, char delimiter) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> header = splitDelimited(lines.get(0), delimiter);
        List<Map<String, Object>> out = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).trim().isEmpty()) {
                continue;
            }
            List<String> fields = splitDelimited(lines.get(i), delimiter);
            Map<String, Object> row = new LinkedHashMap<>();
            for (int c = 0; c < header.size(); c++) {
                row.put(header.get(c), c < fields.size() ? fields.get(c) : "");
            }
            out.add(row);
        }
        return out;
    }

    /** One delimited line's fields, with RFC 4180 quoting. */
    static List<String> splitDelimited(String line, char delimiter) {
        List<String> out = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        field.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    field.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == delimiter) {
                out.add(field.toString().trim());
                field.setLength(0);
            } else {
                field.append(c);
            }
        }
        out.add(field.toString().trim());
        return out;
    }

    /** Shuffles the rows under the seed as the reference's {@code random.shuffle} does. */
    static <T> void shuffle(List<T> rows, long seed, boolean disableShuffle) {
        if (!disableShuffle) {
            new PythonRandom(seed).shuffle(rows);
        }
    }

    /** A text content part. */
    static Map<String, Object> textPart(String text) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", text);
        return part;
    }

    /**
     * An image content part. Remote and {@code data:} URLs pass through; a local path becomes a base64
     * data URL so the server needs no access to the client's files.
     *
     * @param baseDir the directory a relative path resolves against
     * @throws IllegalArgumentException naming the image when it is missing or unreadable
     */
    static Map<String, Object> imagePart(String image, Path baseDir) {
        if (image == null || image.isEmpty()) {
            throw new IllegalArgumentException("an image content part needs a path or URL");
        }
        String url;
        if (image.startsWith("data:image/")) {
            url = image;
        } else if (image.startsWith("http://") || image.startsWith("https://")) {
            url = image;
        } else {
            Path file = localPath(image, baseDir);
            url = imageDataUrl(file);
        }
        Map<String, Object> imageUrl = new LinkedHashMap<>();
        imageUrl.put("url", url);
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "image_url");
        part.put("image_url", imageUrl);
        return part;
    }

    /** The JPEG data URL of an image file. */
    static String imageDataUrl(Path file) {
        String lower = file.getFileName().toString().toLowerCase(Locale.ROOT);
        String mime = lower.endsWith(".png") ? "image/png"
                : lower.endsWith(".gif") ? "image/gif"
                : lower.endsWith(".webp") ? "image/webp"
                : "image/jpeg";
        return dataUrl(file, mime);
    }

    /** A file as a base64 data URL of {@code mime}. */
    static String dataUrl(Path file, String mime) {
        try {
            byte[] bytes = Files.readAllBytes(file);
            String encoded = Base64.getEncoder().encodeToString(bytes);
            return "data:" + mime + ";base64," + encoded;
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot read the image " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * A local path, absolute or relative to the dataset directory.
     *
     * @throws IllegalArgumentException naming the path when it is neither file nor URL
     */
    static Path localPath(String value, Path baseDir) {
        Path path;
        try {
            path = java.nio.file.Paths.get(value);
        } catch (java.nio.file.InvalidPathException e) {
            throw new IllegalArgumentException("'" + value + "' is not a path or URL: " + e.getMessage(), e);
        }
        if (!path.isAbsolute() && baseDir != null) {
            path = baseDir.resolve(path);
        }
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("the dataset file does not exist: " + path);
        }
        return path;
    }

    /**
     * The audio of one record: a local path, or a Hugging Face audio dictionary carrying {@code path}.
     * The bytes are WAV, the format the reference's transcription upload always uses; a source that
     * already is WAV is passed through, anything the JVM's audio system reads is re-encoded.
     */
    static AudioFile audio(Object audio, Path baseDir) {
        String path;
        if (audio instanceof String) {
            path = (String) audio;
        } else if (audio instanceof Map && ((Map<?, ?>) audio).get("path") instanceof String) {
            path = (String) ((Map<?, ?>) audio).get("path");
        } else {
            throw new IllegalArgumentException("an audio record needs a path string or a {'path': ...}"
                    + " object, not " + (audio == null ? "null" : audio.getClass().getSimpleName()));
        }
        Path file = localPath(path, baseDir);
        String name = file.getFileName().toString();
        String lower = name.toLowerCase(Locale.ROOT);
        try {
            byte[] raw = Files.readAllBytes(file);
            if (lower.endsWith(".wav") || lower.endsWith(".wave")) {
                return new AudioFile(file.toAbsolutePath().toString(), raw, name, "audio/wav",
                        wavDuration(raw, file));
            }
            return transcodeToWav(raw, file, name);
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot read the audio " + file + ": " + e.getMessage(), e);
        }
    }

    /** The duration of a WAV byte array, read from its header. */
    static double wavDuration(byte[] wav, Path file) {
        if (wav.length < 44 || !ascii(wav, 0, "RIFF") || !ascii(wav, 8, "WAVE")) {
            throw new IllegalArgumentException("the audio " + file + " is not a RIFF WAVE file");
        }
        int offset = 12;
        int byteRate = 0;
        int dataSize = -1;
        while (offset + 8 <= wav.length) {
            String id = asciiChunk(wav, offset, 4);
            long size = uint32(wav, offset + 4);
            int body = offset + 8;
            if (id.equals("fmt ") && size >= 16 && body + 16 <= wav.length) {
                byteRate = (int) uint32(wav, body + 8);
            } else if (id.equals("data")) {
                dataSize = (int) Math.min(size, wav.length - body);
                if (byteRate > 0) {
                    break;
                }
            }
            long next = body + size + (size & 1);
            if (next <= offset || next > wav.length) {
                break;
            }
            offset = (int) next;
        }
        if (byteRate <= 0 || dataSize < 0) {
            throw new IllegalArgumentException("the WAV file " + file + " carries no usable fmt or data chunk");
        }
        return (double) dataSize / byteRate;
    }

    private static AudioFile transcodeToWav(byte[] raw, Path file, String name) {
        try (AudioInputStream in = AudioSystem.getAudioInputStream(new ByteArrayInputStream(raw))) {
            AudioFormat format = in.getFormat();
            double duration = format.getFrameRate() > 0 && in.getFrameLength() >= 0
                    ? in.getFrameLength() / format.getFrameRate() : 0.0;
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            AudioSystem.write(in, javax.sound.sampled.AudioFileFormat.Type.WAVE, buffer);
            String stem = name.lastIndexOf('.') > 0 ? name.substring(0, name.lastIndexOf('.')) : name;
            return new AudioFile(file.toAbsolutePath().toString(), buffer.toByteArray(), stem + ".wav",
                    "audio/wav", duration);
        } catch (UnsupportedAudioFileException e) {
            throw new IllegalArgumentException("the audio format of " + file + " is not supported locally:"
                    + " pass a WAV, AIFF or AU file", e);
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot decode the audio " + file + ": " + e.getMessage(), e);
        }
    }

    private static boolean ascii(byte[] bytes, int offset, String text) {
        if (offset + text.length() > bytes.length) {
            return false;
        }
        for (int i = 0; i < text.length(); i++) {
            if (bytes[offset + i] != (byte) text.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static String asciiChunk(byte[] bytes, int offset, int length) {
        StringBuilder out = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            out.append((char) (bytes[offset + i] & 0xff));
        }
        return out.toString();
    }

    private static long uint32(byte[] bytes, int offset) {
        return (bytes[offset] & 0xffL) | ((bytes[offset + 1] & 0xffL) << 8)
                | ((bytes[offset + 2] & 0xffL) << 16) | ((bytes[offset + 3] & 0xffL) << 24);
    }

    /** A synthetic RGB image encoded as JPEG, the reference's random-pixel image. */
    static String syntheticImageDataUrl(int width, int height, java.util.Random random) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = (random.nextInt(256) << 16) | (random.nextInt(256) << 8) | random.nextInt(256);
                image.setRGB(x, y, rgb);
            }
        }
        try {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            ImageIO.write(image, "jpeg", buffer);
            return "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(buffer.toByteArray());
        } catch (IOException e) {
            throw new IllegalStateException("cannot encode a synthetic image", e);
        }
    }

    /** A record's required string field, refused by name when absent or not a string. */
    static String string(Map<String, Object> row, String key, String what) {
        Object value = row.get(key);
        if (!(value instanceof String)) {
            throw new IllegalArgumentException(what + " needs a '" + key + "' string, not "
                    + describe(value));
        }
        return (String) value;
    }

    /** A record's optional string field, null when absent. */
    static String optionalString(Map<String, Object> row, String key) {
        Object value = row.get(key);
        return value instanceof String ? (String) value : null;
    }

    /** A record's numeric field as a double, refused by name when absent or not a number. */
    static double number(Map<String, Object> row, String key, String what) {
        Object value = row.get(key);
        if (!(value instanceof Number)) {
            throw new IllegalArgumentException(what + " needs a numeric '" + key + "', not " + describe(value));
        }
        return ((Number) value).doubleValue();
    }

    /** A record's optional numeric field, or the fallback. */
    static double numberOr(Map<String, Object> row, String key, double fallback) {
        Object value = row.get(key);
        return value instanceof Number ? ((Number) value).doubleValue() : fallback;
    }

    /** A record's list field, refused by name when absent or not an array. */
    static List<?> list(Map<String, Object> row, String key, String what) {
        Object value = row.get(key);
        if (!(value instanceof List)) {
            throw new IllegalArgumentException(what + " needs a '" + key + "' array, not " + describe(value));
        }
        return (List<?>) value;
    }

    /** A short description of a parsed JSON value, for an error message. */
    static String describe(Object value) {
        return value == null ? "null" : value.getClass().getSimpleName();
    }
}
