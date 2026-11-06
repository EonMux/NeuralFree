package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * The parameter sweep of the {@code bench-sweep} command: a JSON file of parameter records, one per
 * benchmark combination — either a list of objects or an object of name to object, exactly the
 * formats vLLM's {@code ParameterSweep} reads. Each record carries a benchmark name under
 * {@code _benchmark_name} (optional) and the flags it sets; {@link ParameterSweepItem#applyToCmd}
 * replaces the named flags of a command line or appends them. Names are unique, as the reference
 * demands.
 */
public final class ParameterSweep {

    /** One parameter record of the sweep. */
    public static final class Item {

        private final Map<String, Object> params;

        /** The record over {@code params}; the sweep builds its items from the file's objects. */
        public Item(Map<String, Object> params) {
            this.params = params;
        }

        /** The record's parameters, the name key included. */
        Map<String, Object> paramsView() {
            return params;
        }

        /** The benchmark name: {@code _benchmark_name}, else every {@code k=v} joined by dashes. */
        public String name() {
            Object named = params.get("_benchmark_name");
            if (named != null) {
                return String.valueOf(named);
            }
            StringJoiner joiner = new StringJoiner("-");
            for (Map.Entry<String, Object> e : params.entrySet()) {
                if (!"_benchmark_name".equals(e.getKey())) {
                    joiner.add(e.getKey() + "=" + e.getValue());
                }
            }
            return joiner.toString();
        }

        /** Whether the record sets {@code key} under any of its {@code -}/{@code _} spellings. */
        public boolean hasParam(String key) {
            return candidates(key).stream().anyMatch(params::containsKey);
        }

        /** The value of {@code key} under any of its spellings. */
        public Object valueOf(String key) {
            for (String candidate : candidates(key)) {
                if (params.containsKey(candidate)) {
                    return params.get(candidate);
                }
            }
            return null;
        }

        /**
         * The command with every record parameter applied: an existing flag's value replaced (a
         * boolean toggled, negated to {@code --no-*} when false), a new one appended; objects as
         * JSON.
         */
        public List<String> applyToCmd(List<String> cmd) {
            List<String> out = new ArrayList<>(cmd);
            for (Map.Entry<String, Object> e : params.entrySet()) {
                if ("_benchmark_name".equals(e.getKey())) {
                    continue;
                }
                List<String> kv = pair(e.getKey(), e.getValue());
                int index = indexOfFlag(out, e.getKey());
                if (index < 0) {
                    out.addAll(kv);
                    continue;
                }
                if (kv.size() == 1) {
                    out.set(index, kv.get(0));
                } else if (index + 1 < out.size()) {
                    out.set(index, kv.get(0));
                    out.set(index + 1, kv.get(1));
                } else {
                    out.set(index, kv.get(0));
                    out.add(kv.get(1));
                }
            }
            return out;
        }

        /** One parameter as command-line pieces: flags alone, values in pairs, objects as JSON. */
        private static List<String> pair(String key, Object value) {
            String name = key.replace('_', '-');
            String flag = "--" + name;
            if (value instanceof Boolean) {
                boolean on = (Boolean) value;
                if (key.indexOf('.') >= 0) {
                    return List.of(flag + "=" + (on ? "true" : "false"));
                }
                if (on) {
                    return List.of(flag);
                }
                return List.of(flag.startsWith("--no-") ? flag : "--no-" + name);
            }
            if (value instanceof Map || value instanceof List) {
                return List.of(flag, jsonText(value));
            }
            return List.of(flag, String.valueOf(value));
        }

        /** The first index of {@code --key} under its {@code -}/{@code _} spellings, or -1. */
        private static int indexOfFlag(List<String> cmd, String key) {
            for (String candidate : candidates(key)) {
                for (int i = 0; i < cmd.size(); i++) {
                    if (cmd.get(i).equals("--" + candidate)) {
                        return i;
                    }
                }
            }
            return -1;
        }

        /** {@code key} and its {@code -}/{@code _} spellings. */
        private static List<String> candidates(String key) {
            List<String> out = new ArrayList<>();
            out.add(key);
            out.add(key.replace('-', '_'));
            out.add(key.replace('_', '-'));
            return out;
        }

        @Override
        public String toString() {
            return "Item{" + params + "}";
        }
    }

    private final List<Item> items;

    private ParameterSweep(List<Item> items) {
        this.items = items;
    }

    /** The empty sweep: one record without parameters, the reference's default. */
    public static ParameterSweep none() {
        List<Item> items = new ArrayList<>();
        items.add(new Item(new LinkedHashMap<>()));
        return new ParameterSweep(items);
    }

    /**
     * The sweep of a JSON file: a list of records, or an object of benchmark name to record.
     *
     * @throws IOException for an unreadable file, a wrong shape or duplicate names
     */
    public static ParameterSweep readJson(Path file) throws IOException {
        Object parsed;
        try {
            parsed = Json.parse(Files.readString(file), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IOException("the parameter sweep " + file + " is not JSON: " + e.getMessage());
        }
        if (parsed instanceof Map) {
            Map<?, ?> byName = (Map<?, ?>) parsed;
            List<Item> items = new ArrayList<>();
            for (Map.Entry<?, ?> e : byName.entrySet()) {
                if (!(e.getValue() instanceof Map)) {
                    throw new IOException("every sweep record of " + file + " must be an object, '"
                            + e.getKey() + "' is not");
                }
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("_benchmark_name", String.valueOf(e.getKey()));
                record.putAll((Map<? extends String, ?>) e.getValue());
                items.add(new Item(record));
            }
            return check(new ParameterSweep(items), file);
        }
        if (!(parsed instanceof List)) {
            throw new IOException("the parameter sweep " + file + " must be a list of objects or an object"
                    + " of name to object");
        }
        List<Item> items = new ArrayList<>();
        for (Object entry : (List<?>) parsed) {
            if (!(entry instanceof Map)) {
                throw new IOException("every sweep record of " + file + " must be an object");
            }
            items.add(new Item((Map<String, Object>) entry));
        }
        return check(new ParameterSweep(items), file);
    }

    /** The names' uniqueness, as the reference demands. */
    private static ParameterSweep check(ParameterSweep sweep, Path file) throws IOException {
        List<String> named = new ArrayList<>();
        for (Item item : sweep.items) {
            Object name = item.params.get("_benchmark_name");
            if (name != null) {
                String n = String.valueOf(name);
                if (named.contains(n)) {
                    throw new IOException("duplicate _benchmark_name '" + n + "' in " + file
                            + ": every name must be unique");
                }
                named.add(n);
            }
        }
        return sweep;
    }

    /** The records, in file order. */
    public List<Item> items() {
        return new ArrayList<>(items);
    }

    private static String jsonText(Object value) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(value, Json.Style.PYTHON_DEFAULT, out);
        } catch (IOException e) {
            throw new IllegalStateException("a sweep parameter does not write as JSON", e);
        }
        return out.toString();
    }
}
