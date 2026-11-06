package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The {@code custom_image} dataset, vLLM's {@code CustomImageDataset}: a JSONL file whose records
 * carry either a {@code prompt} with an {@code image_files} array or an interleaved {@code content}
 * array of text and image parts. The records shuffle under the seed, the text parts count the
 * prompt, and every request is one user chat message with the images as content parts. Local images
 * are encoded as base64 data URLs, so the server needs no access to the client's files.
 */
final class CustomImageDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    CustomImageDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "custom_image");
        if (!path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".jsonl")) {
            throw new IllegalArgumentException("only the JSONL format is supported for the custom_image"
                    + " dataset, not '" + path + "'");
        }
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readFile(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the custom_image dataset " + path + ": "
                    + e.getMessage(), e);
        }
        BenchDatasets.shuffle(rows, opts.seed(), opts.disableShuffle());
        Path baseDir = BenchDatasets.baseDir(path);
        int outputLen = Math.max(0, opts.customOutputLen());
        int target = BenchDatasets.targetNum(opts, rows.size());
        List<BenchSample> out = new ArrayList<>();
        for (int i = 0; i < rows.size() && out.size() < target; i++) {
            Map<String, Object> row = rows.get(i);
            List<Object> content = contentOf(row, baseDir, i + 1);
            String text = textOf(content);
            out.add(BenchSample.ofChat(content, BenchDatasets.promptLen(opts, tokenizer, text), outputLen,
                    opts.requestIdPrefix() + i));
        }
        return ServeBenchDataset.oversample(out, target, opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }

    /**
     * One record's chat content: the interleaved {@code content} parts, or the legacy
     * {@code prompt} and {@code image_files} fields as a text part followed by the images.
     *
     * @throws IllegalArgumentException naming the record when it carries neither shape
     */
    static List<Object> contentOf(Map<String, Object> row, Path baseDir, int line) {
        if (row.containsKey("content")) {
            return interleaved(row.get("content"), baseDir, line, null);
        }
        if (row.containsKey("prompt") && row.containsKey("image_files")) {
            String text = BenchDatasets.string(row, "prompt", "a custom_image record");
            List<Object> content = new ArrayList<>();
            content.add(BenchDatasets.textPart(text));
            content.addAll(imageParts(row.get("image_files"), baseDir, line, null));
            return content;
        }
        throw new IllegalArgumentException("record " + line + " needs either 'prompt' and 'image_files'"
                + " fields, or a 'content' field");
    }

    /** The content parts of one interleaved record, serving paths resolved against the dataset. */
    private static List<Object> interleaved(Object content, Path baseDir, int line, Path path) {
        String where = where(path);
        if (!(content instanceof List) || ((List<?>) content).isEmpty()) {
            throw new IllegalArgumentException("record " + line + where + " needs a non-empty"
                    + " 'content' array");
        }
        List<Object> parts = new ArrayList<>();
        for (Object raw : (List<?>) content) {
            if (!(raw instanceof Map)) {
                throw new IllegalArgumentException("record " + line + where
                        + " carries a content item that is not an object");
            }
            Map<?, ?> part = (Map<?, ?>) raw;
            Object type = part.get("type");
            if ("text".equals(type)) {
                Object text = part.get("text");
                if (!(text instanceof String)) {
                    throw new IllegalArgumentException("record " + line + where
                            + " carries a text part without a string 'text'");
                }
                parts.add(BenchDatasets.textPart((String) text));
            } else if ("image".equals(type)) {
                parts.add(BenchDatasets.imagePart(imageSource(part.get("image"), line, where), baseDir));
            } else if ("image_url".equals(type)) {
                parts.add(imageUrlPart(part.get("image_url"), baseDir, line, where));
            } else {
                throw new IllegalArgumentException("record " + line + where + " carries a content"
                        + " part of type '" + type + "': the custom_image parts are text, image and"
                        + " image_url");
            }
        }
        return parts;
    }

    /** Where a record lives, for an error message. */
    private static String where(Path path) {
        return path == null ? "" : " of " + path;
    }

    /** An image part from its {@code image} field, a path or URL string. */
    private static String imageSource(Object image, int line, String where) {
        if (!(image instanceof String)) {
            throw new IllegalArgumentException("record " + line + where
                    + " carries an image part without a string 'image'");
        }
        return (String) image;
    }

    /** An {@code image_url} part: a URL string, or an object carrying {@code url} and optional extras. */
    private static Object imageUrlPart(Object imageUrl, Path baseDir, int line, String where) {
        if (imageUrl instanceof String) {
            return BenchDatasets.imagePart((String) imageUrl, baseDir);
        }
        if (!(imageUrl instanceof Map)) {
            throw new IllegalArgumentException("record " + line + where
                    + " carries an image_url part that is neither a string nor an object");
        }
        Map<?, ?> image = (Map<?, ?>) imageUrl;
        Object url = image.get("url");
        if (!(url instanceof String)) {
            throw new IllegalArgumentException("record " + line + where
                    + " carries an image_url object without a string 'url'");
        }
        Map<String, Object> part = BenchDatasets.imagePart((String) url, baseDir);
        @SuppressWarnings("unchecked")
        Map<String, Object> processed = (Map<String, Object>) part.get("image_url");
        for (Map.Entry<?, ?> entry : image.entrySet()) {
            if (!"url".equals(entry.getKey())) {
                processed.put(String.valueOf(entry.getKey()), entry.getValue());
            }
        }
        return part;
    }

    /** The legacy {@code image_files} array as image parts. */
    private static List<Object> imageParts(Object images, Path baseDir, int line, Path path) {
        String where = where(path);
        if (!(images instanceof List) || ((List<?>) images).isEmpty()) {
            throw new IllegalArgumentException("record " + line + where
                    + " needs a non-empty 'image_files' array");
        }
        List<Object> parts = new ArrayList<>();
        for (Object image : (List<?>) images) {
            if (!(image instanceof String)) {
                throw new IllegalArgumentException("record " + line + where
                        + " carries an 'image_files' entry that is not a path or URL");
            }
            parts.add(BenchDatasets.imagePart((String) image, baseDir));
        }
        return parts;
    }

    /** The concatenated text of a content list, the prompt the reference counts. */
    static String textOf(List<Object> content) {
        StringBuilder out = new StringBuilder();
        for (Object part : content) {
            if (part instanceof Map && "text".equals(((Map<?, ?>) part).get("type"))) {
                Object text = ((Map<?, ?>) part).get("text");
                if (text instanceof String) {
                    out.append((String) text);
                }
            }
        }
        return out.toString();
    }
}
