package com.eonmux.neuralfree.demo.tools.compare;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Reads the result rows of a benchmark file in any of the layouts the bench tools write: a JSON array of result
 * objects ({@code bench -o json}, a {@code bench-sweep} summary), one JSON object (a saved {@code bench-serve}
 * result), or one object per line ({@code bench -o jsonl}, {@code batched-bench --output-format jsonl}). Every object
 * must be a result of a known {@link RowKind}; anything else is refused with its file and position.
 */
public final class BenchResultFiles {

    /** Largest result file read, far above any bench output. */
    static final long MAX_BYTES = 256L << 20;

    private BenchResultFiles() {
    }

    /**
     * The rows of {@code file}.
     *
     * @throws IOException when the file cannot be read, holds no row, is not JSON, or holds an object that is not a
     *                     benchmark result
     */
    public static List<BenchRow> read(Path file) throws IOException {
        long size = Files.size(file);
        if (size > MAX_BYTES) {
            throw new IOException(file + ": " + size + " bytes exceed the " + MAX_BYTES + "-byte result limit");
        }
        String text = new String(Files.readAllBytes(file), StandardCharsets.UTF_8).trim();
        if (text.isEmpty()) {
            throw new IOException(file + " holds no benchmark result");
        }
        List<BenchRow> rows = new ArrayList<>();
        Object whole = parseWhole(text);
        if (whole instanceof List) {
            List<?> items = (List<?>) whole;
            for (int i = 0; i < items.size(); i++) {
                rows.add(row(items.get(i), file + " [" + i + "]"));
            }
        } else if (whole != null) {
            rows.add(row(whole, file.toString()));
        } else {
            String[] lines = text.split("\r?\n");
            for (int i = 0; i < lines.length; i++) {
                String line = lines[i].trim();
                if (line.isEmpty()) {
                    continue;
                }
                String where = file + ":" + (i + 1);
                try {
                    rows.add(row(Json.parse(line, Json.Limits.STRICT), where));
                } catch (JsonException e) {
                    throw new IOException(where + " is not a JSON object: " + e.getMessage(), e);
                }
            }
        }
        if (rows.isEmpty()) {
            throw new IOException(file + " holds no benchmark result");
        }
        return rows;
    }

    /** The document when the whole text is one JSON value, null when it is not (one object per line). */
    private static Object parseWhole(String text) {
        try {
            return Json.parse(text, Json.Limits.STRICT);
        } catch (JsonException e) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private static BenchRow row(Object value, String where) throws IOException {
        if (!(value instanceof Map)) {
            throw new IOException(where + " is not a JSON object");
        }
        Map<String, Object> fields = (Map<String, Object>) value;
        RowKind kind = RowKind.of(fields);
        if (kind == null) {
            throw new IOException(where + " is no bench, batched-bench or bench-serve result (it has no n_prompt/"
                    + "n_gen/avg_ts, pp/tg/pl/speed or completed/output_throughput fields)");
        }
        return new BenchRow(kind, where, fields);
    }
}
