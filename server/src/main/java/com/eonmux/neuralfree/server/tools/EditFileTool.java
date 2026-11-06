package com.eonmux.neuralfree.server.tools;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code edit_file}: exact text replacements, one or more per call, each {@code old_text} unique in the file and
 * matched against the original content (llama.cpp's {@code server_tool_edit_file}). A text that appears more
 * than once or not at all refuses; overlapping edits and no-op replacements refuse; the edits apply bottom-up
 * over the match positions so earlier replacements never shift later ones.
 */
public final class EditFileTool extends ServerTool {

    /** One validated replacement of the request. */
    private record Edit(int index, String oldText, String newText) { }

    /** One replacement located in the original content: request index, match offsets, new text. */
    private record Located(int index, int at, int length, String newText) { }

    public EditFileTool() {
        super("edit_file", "Edit file", true, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("type", "object");
        item.put("properties", Map.of(
                "old_text", Map.of("type", "string", "description",
                        "Exact text to find; must be unique in the file and must not overlap with other edits"),
                "new_text", Map.of("type", "string", "description", "Text to replace old_text with")));
        item.put("required", List.of("old_text", "new_text"));
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("path", Map.of("type", "string", "description", "Path to the file to edit"));
        properties.put("edits", Map.of("type", "array", "description",
                "One or more exact text replacements to apply", "items", item));
        return functionDefinition(name(),
                "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file and"
                        + " is matched against the original content, not incrementally. Merge nearby changes into"
                        + " one edit instead of overlapping edits. Use write_file to replace the whole file.",
                parameters(properties, "path", "edits"));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String path = string(params, "path");
        List<Object> raw = optList(params, "edits");
        if (raw == null || raw.isEmpty()) {
            return error("\"edits\" must be a non-empty array");
        }
        List<Edit> edits = new ArrayList<>();
        for (int i = 0; i < raw.size(); i++) {
            Map<?, ?> entry = raw.get(i) instanceof Map<?, ?> e ? e : Map.of();
            Object oldText = entry.get("old_text");
            Object newText = entry.get("new_text");
            if (!(oldText instanceof String old) || old.isEmpty()) {
                return error("edits[" + i + "].old_text must be a non-empty string");
            }
            if (!(newText instanceof String replacement)) {
                return error("edits[" + i + "].new_text must be a string");
            }
            edits.add(new Edit(i, old, replacement));
        }
        String original = io.read(path);
        if (original == null) {
            return error("failed to open file: " + path);
        }
        return apply(io, path, original, edits);
    }

    /**
     * Locates every {@code old_text} exactly once in the original content, refuses duplicates, misses and
     * overlaps, then writes the content with every replacement applied.
     */
    private static Map<String, Object> apply(ToolIo io, String path, String original, List<Edit> edits) {
        List<Located> located = new ArrayList<>();
        for (Edit edit : edits) {
            int count = countOccurrences(original, edit.oldText());
            if (count == 0) {
                return error("could not find edits[" + edit.index() + "].old_text in " + path
                        + ", it must match the file's current content exactly");
            }
            if (count > 1) {
                return error("found " + count + " occurrences of edits[" + edit.index() + "].old_text in "
                        + path + ", it must be unique");
            }
            located.add(new Located(edit.index(), original.indexOf(edit.oldText()),
                    edit.oldText().length(), edit.newText()));
        }
        located.sort(Comparator.comparingInt(Located::at));
        for (int i = 1; i < located.size(); i++) {
            Located before = located.get(i - 1);
            Located at = located.get(i);
            if (before.at() + before.length() > at.at()) {
                return error("edits[" + before.index() + "] and edits[" + at.index() + "] overlap in " + path
                        + "; merge them into one edit or target disjoint regions");
            }
        }
        StringBuilder updated = new StringBuilder(original);
        for (int i = located.size() - 1; i >= 0; i--) {
            Located at = located.get(i);
            updated.replace(at.at(), at.at() + at.length(), at.newText());
        }
        if (updated.toString().equals(original)) {
            return error("no changes made: the replacement(s) produced identical content");
        }
        if (!io.write(path, updated.toString())) {
            return error("failed to write file: " + path);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("result", "file edited successfully");
        out.put("path", path);
        out.put("edits_applied", located.size());
        return out;
    }

    /** The number of non-overlapping occurrences of {@code needle} in {@code content}. */
    private static int countOccurrences(String content, String needle) {
        int count = 0;
        int at = 0;
        while ((at = content.indexOf(needle, at)) >= 0) {
            count++;
            at += needle.length();
        }
        return count;
    }
}
