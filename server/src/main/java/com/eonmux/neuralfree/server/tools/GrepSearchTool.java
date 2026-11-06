package com.eonmux.neuralfree.server.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * {@code grep_search}: the lines of the files under a path that match a pattern — a regular expression or a
 * literal string, case-folded on request — with optional line numbers and context lines, over an include/exclude
 * glob on the relative paths, held to 100 matches (llama.cpp's {@code server_tool_grep_search}). The answer is
 * the matched lines as text with the total match count.
 */
public final class GrepSearchTool extends ServerTool {

    /** The most matches one search answers with. */
    static final int MAX_RESULTS = 100;

    /** One file to search: the path it reads from and the path the answer names it by. */
    private record Target(String abs, String display) { }

    public GrepSearchTool() {
        super("grep_search", "Grep search", false, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("path", Map.of("type", "string", "description", "File or directory to search in"));
        properties.put("pattern", Map.of("type", "string", "description",
                "Pattern to search for (regular expression unless literal is true)"));
        properties.put("include", Map.of("type", "string", "description",
                "Glob pattern to filter files (default: **)"));
        properties.put("exclude", Map.of("type", "string", "description", "Glob pattern to exclude files"));
        properties.put("return_line_numbers", Map.of("type", "boolean", "description",
                "If true, include line numbers in results"));
        properties.put("literal", Map.of("type", "boolean", "description",
                "Treat pattern as a literal string instead of a regular expression (default: false)"));
        properties.put("ignore_case", Map.of("type", "boolean", "description",
                "Case-insensitive search (default: false)"));
        properties.put("context_lines", Map.of("type", "integer", "description",
                "Number of lines of context to show before and after each match (default: 0)"));
        return functionDefinition(name(),
                "Search for a pattern in files under a path. Returns matching lines with file paths (and, unless"
                        + " searching a single file, paths relative to the given directory). Automatically skips"
                        + " common junk directories (.git, node_modules, build, dist, etc.). include/exclude: a"
                        + " pattern with no '/' matches the basename at any depth; a pattern containing '/' matches"
                        + " the full relative path (auto-anchored with \"**/\" unless already anchored).",
                parameters(properties, "path", "pattern"));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String path = string(params, "path");
        String raw = string(params, "pattern");
        String include = optString(params, "include", "**");
        String exclude = optString(params, "exclude", "");
        boolean showNumbers = optBool(params, "return_line_numbers", false);
        boolean ignoreCase = optBool(params, "ignore_case", false);
        int context = Math.max(0, optInt(params, "context_lines", 0));
        Pattern pattern;
        try {
            pattern = Pattern.compile(optBool(params, "literal", false) ? Pattern.quote(raw) : raw,
                    ignoreCase ? Pattern.CASE_INSENSITIVE : 0);
        } catch (PatternSyntaxException e) {
            return error("invalid regex: " + e.getMessage());
        }
        List<Target> targets = targets(io, path, include, exclude);
        if (targets == null) {
            return error("path does not exist: " + path);
        }
        return text(scan(io, targets, pattern, showNumbers || context > 0, context));
    }

    /** The files to search: the one file at the path, or the directory listing the globs keep. */
    private static List<Target> targets(ToolIo io, String path, String include, String exclude) {
        String base = io.resolve(path).toString();
        if (io.isRegularFile(path)) {
            return List.of(new Target(base, path));
        }
        if (!io.isDirectory(path)) {
            return null;
        }
        List<Target> out = new ArrayList<>();
        for (ToolIo.Entry entry : io.list(path, 0, true, false).entries()) {
            if (!ToolGlob.match(include, entry.rel())) {
                continue;
            }
            if (!exclude.isEmpty() && ToolGlob.match(exclude, entry.rel())) {
                continue;
            }
            out.add(new Target(base + "/" + entry.rel(), entry.rel()));
        }
        return out;
    }

    /** Every matching line of every file, with the requested context, at the match cap. */
    private static String scan(ToolIo io, List<Target> targets, Pattern pattern, boolean showNumbers,
                               int context) {
        StringBuilder out = new StringBuilder();
        int total = 0;
        boolean capped = false;
        for (Target target : targets) {
            if (capped) {
                break;
            }
            String content = io.read(target.abs());
            if (content == null) {
                continue;
            }
            String[] lines = content.split("\n", -1);
            if (content.endsWith("\n") && lines.length > 0 && lines[lines.length - 1].isEmpty()) {
                lines = Arrays.copyOf(lines, lines.length - 1);
            }
            for (int i = 0; i < lines.length; i++) {
                if (total >= MAX_RESULTS) {
                    capped = true;
                    break;
                }
                if (!pattern.matcher(lines[i]).find()) {
                    continue;
                }
                int from = Math.max(0, i - context);
                int to = Math.min(lines.length - 1, i + context);
                for (int j = from; j <= to; j++) {
                    boolean match = j == i;
                    out.append(target.display()).append(match ? ':' : '-');
                    if (showNumbers) {
                        out.append(j + 1).append(match ? ':' : '-');
                    }
                    out.append(lines[j]).append('\n');
                }
                if (context > 0) {
                    out.append("--\n");
                }
                total++;
            }
        }
        out.append("\n---\nTotal matches: ").append(total).append('\n');
        if (capped) {
            out.append("[").append(MAX_RESULTS).append(" matches limit reached. Narrow the path/pattern/include"
                    + " to see more.]\n");
        }
        return out.toString();
    }
}
