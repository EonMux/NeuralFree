package com.eonmux.neuralfree.server.tools;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code file_glob_search}: entries under a base directory matching a glob include pattern, with an exclude
 * pattern, an entry kind (file, dir, all), a depth cap and a result limit of 100 (llama.cpp's
 * {@code server_tool_file_glob_search}). The answer is the listing as text with the total match count, plus the
 * same entries as structured JSON ({@code entries}) and the resolved base directory ({@code base}).
 */
public final class FileGlobSearchTool extends ServerTool {

    /** The most entries one search answers with. */
    static final int MAX_RESULTS = 100;

    public FileGlobSearchTool() {
        super("file_glob_search", "File search", false, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("path", Map.of("type", "string", "description", "Base directory to search in"));
        properties.put("include", Map.of("type", "string", "description",
                "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"));
        properties.put("exclude", Map.of("type", "string", "description", "Glob pattern for files to exclude"));
        properties.put("type", Map.of("type", "string", "description",
                "Entry type to return: \"file\" (default), \"dir\" or \"all\""));
        properties.put("max_depth", Map.of("type", "integer", "description",
                "Maximum depth to descend into subdirectories (default: 0 = unlimited; 1 = direct children only)"));
        properties.put("limit", Map.of("type", "integer", "description",
                "Maximum number of results to return, capped at " + MAX_RESULTS + " (default " + MAX_RESULTS + ")"));
        return functionDefinition(name(),
                "Recursively search for files matching a glob pattern under a directory. Automatically skips"
                        + " common junk directories (.git, node_modules, build, dist, etc.). A pattern with no '/'"
                        + " (e.g. \"*.cpp\") matches the file's basename at any depth. A pattern containing '/'"
                        + " matches the full relative path; unless already anchored with \"**/\" or a leading '/',"
                        + " it is automatically prefixed with \"**/\". Use type=\"dir\" or \"all\" to also list"
                        + " directories; directory entries are suffixed with '/' in the output.",
                parameters(properties, "path"));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String path = string(params, "path");
        String include = optString(params, "include", "**");
        String exclude = optString(params, "exclude", "");
        String kind = optString(params, "type", "file");
        int maxDepth = Math.max(0, optInt(params, "max_depth", 0));
        int requested = optInt(params, "limit", MAX_RESULTS);
        if (requested < 1) {
            return error("invalid limit: " + requested + " (expected 1 or more)");
        }
        boolean files = kind.equals("file") || kind.equals("all");
        boolean dirs = kind.equals("dir") || kind.equals("all");
        if (!files && !dirs) {
            return error("invalid type: " + kind + " (expected \"file\", \"dir\" or \"all\")");
        }
        ToolIo.Listing listing = io.list(path, maxDepth, files, dirs);
        if (listing.error() != null) {
            return error(listing.error() + ": " + path);
        }
        return answer(io, path, include, exclude, Math.min(requested, MAX_RESULTS), listing);
    }

    /** The matched entries as the text listing and the structured fields, at the limit. */
    private static Map<String, Object> answer(ToolIo io, String path, String include, String exclude, int limit,
                                              ToolIo.Listing listing) {
        List<ToolIo.Entry> matched = new ArrayList<>();
        for (ToolIo.Entry entry : listing.entries()) {
            if (!ToolGlob.match(include, entry.rel())) {
                continue;
            }
            if (!exclude.isEmpty() && ToolGlob.match(exclude, entry.rel())) {
                continue;
            }
            matched.add(entry);
        }
        int shown = Math.min(matched.size(), limit);
        StringBuilder text = new StringBuilder();
        List<Object> entries = new ArrayList<>();
        for (int i = 0; i < shown; i++) {
            ToolIo.Entry entry = matched.get(i);
            text.append(entry.rel()).append('\n');
            Map<String, Object> json = new LinkedHashMap<>();
            json.put("path", entry.rel());
            json.put("type", entry.dir() ? "dir" : "file");
            entries.add(json);
        }
        text.append("\n---\nTotal matches: ").append(matched.size()).append('\n');
        if (matched.size() > shown) {
            text.append("[").append(shown).append(" results limit reached (").append(matched.size())
                    .append(" total matches). Refine the glob pattern to narrow the search.]\n");
        }
        if (listing.truncated()) {
            text.append("[results truncated: time budget or unreadable directory]\n");
        }
        Map<String, Object> out = text(text.toString());
        out.put("entries", entries);
        out.put("base", io.resolve(path).toString());
        return out;
    }
}
