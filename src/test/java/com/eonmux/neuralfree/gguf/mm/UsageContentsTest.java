package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * USAGE.md's Contents list against the document it indexes: every '##' section and every '###' subsection
 * appears in the list in document order at the right nesting, every Contents link resolves to a heading, and
 * every in-document '](#anchor)' link resolves. The same check covers README.md's and LLM-INFERENCE.md's links
 * into USAGE.md. Fails when a section lands in USAGE.md without its Contents entry, or an anchor drifts from
 * its heading, so the Contents list cannot silently go stale.
 */
public class UsageContentsTest {

    /** An ATX heading, indented at most three spaces so four-space code blocks do not match. */
    private static final Pattern HEADING = Pattern.compile("^ {0,3}(#{1,6})\\s+(.*)$");
    private static final Pattern TOP_ENTRY = Pattern.compile("^(\\d+)\\.\\s+\\[(.*)\\]\\(#([^)]+)\\)$");
    private static final Pattern SECTION_ENTRY = Pattern.compile("^ {3}-\\s+\\[(.*)\\]\\(#([^)]+)\\)$");
    private static final Pattern IN_DOCUMENT_LINK = Pattern.compile("\\]\\(#([^)]+)\\)");
    private static final Pattern USAGE_DOCUMENT_LINK = Pattern.compile("\\]\\((?:USAGE\\.md|\\./USAGE\\.md)#([^)]+)\\)");
    /** A markdown inline link, whose text (not its target) is what GitHub slugs. */
    private static final Pattern INLINE_LINK = Pattern.compile("\\[([^\\]]*)\\]\\([^)]*\\)");
    private static final Pattern CLOSING_HASHES = Pattern.compile("\\s*#+\\s*$");

    /** Every heading in USAGE.md, with the anchor GitHub derives from its text. */
    @Test
    public void contentsListsEverySectionInDocumentOrder() throws IOException {
        List<ContentEntry> actual = contentsEntries();
        List<ContentEntry> expected = sectionEntries();
        assertFalse("USAGE.md's Contents list parsed empty", actual.isEmpty());
        assertEquals("USAGE.md's Contents list does not cover every '##'/'###' heading; expected "
                        + expected.size() + " entries but found " + actual.size() + "\nExpected:\n"
                        + describe(expected) + "\nActual:\n" + describe(actual),
                expected.size(), actual.size());
        int topNumber = 0;
        for (int i = 0; i < expected.size(); i++) {
            ContentEntry want = expected.get(i);
            ContentEntry got = actual.get(i);
            assertEquals("USAGE.md line " + got.line + " is at Contents nesting " + got.level
                    + " but its heading is level " + want.level, want.level, got.level);
            assertEquals("USAGE.md line " + got.line + " labels '" + got.text + "' but its heading is '"
                    + want.text + "'", want.text, got.text);
            assertEquals("USAGE.md line " + got.line + " links '#" + got.anchor + "' but the heading's anchor is '#"
                    + want.anchor + "'", want.anchor, got.anchor);
            if (got.level == 2) {
                topNumber++;
                assertEquals("USAGE.md's Contents top-level entries must stay numbered sequentially", topNumber,
                        got.number);
            }
        }
    }

    /** Every Contents link names a heading by its exact text and anchor. */
    @Test
    public void everyContentsLinkResolvesToAHeading() throws IOException {
        Map<String, Heading> anchors = headingsByAnchor();
        for (ContentEntry entry : contentsEntries()) {
            Heading heading = anchors.get(entry.anchor);
            assertTrue("USAGE.md line " + entry.line + " in Contents links '#" + entry.anchor
                    + "', but no heading has that anchor; point it at the right heading", heading != null);
            assertEquals("USAGE.md line " + entry.line + " in Contents labels '" + entry.text
                    + "', but '#" + entry.anchor + "' is the heading '" + heading.text + "'", heading.text,
                    entry.text);
        }
    }

    /** Every in-document anchor link in USAGE.md, Contents included, resolves to a heading. */
    @Test
    public void everyInDocumentLinkResolvesToAHeading() throws IOException {
        Map<String, Heading> anchors = headingsByAnchor();
        for (Line line : usageProseLines()) {
            Matcher link = IN_DOCUMENT_LINK.matcher(line.text);
            while (link.find()) {
                String anchor = link.group(1);
                assertTrue("USAGE.md line " + line.number + " links '#" + anchor
                                + "', but no heading has that anchor; point it at the right heading",
                        anchors.containsKey(anchor));
            }
        }
    }

    /** README.md's and LLM-INFERENCE.md's links into USAGE.md resolve to USAGE.md headings. */
    @Test
    public void crossDocumentLinksResolveToUsageHeadings() throws IOException {
        Map<String, Heading> anchors = headingsByAnchor();
        for (String document : new String[]{"README.md", "LLM-INFERENCE.md"}) {
            for (Line line : proseLines(document)) {
                Matcher link = USAGE_DOCUMENT_LINK.matcher(line.text);
                while (link.find()) {
                    String anchor = link.group(1);
                    assertTrue(document + " line " + line.number + " links 'USAGE.md#" + anchor
                                    + "', but USAGE.md has no heading with that anchor",
                            anchors.containsKey(anchor));
                }
            }
        }
    }

    // ------------------------------------------------------------------ document parsing

    /** All headings in USAGE.md, in document order, with GitHub's duplicate anchor suffixes. */
    private List<Heading> headings() throws IOException {
        List<Heading> headings = new ArrayList<>();
        Map<String, Integer> seen = new HashMap<>();
        for (Line line : usageProseLines()) {
            Matcher heading = HEADING.matcher(line.text);
            if (!heading.matches()) {
                continue;
            }
            String text = CLOSING_HASHES.matcher(heading.group(2).trim()).replaceAll("");
            String base = slug(text);
            int copies = seen.containsKey(base) ? seen.get(base) : 0;
            seen.put(base, copies + 1);
            String anchor = copies == 0 ? base : base + "-" + copies;
            headings.add(new Heading(line.number, heading.group(1).length(), text, anchor));
        }
        return headings;
    }

    private Map<String, Heading> headingsByAnchor() throws IOException {
        Map<String, Heading> anchors = new LinkedHashMap<>();
        for (Heading heading : headings()) {
            anchors.put(heading.anchor, heading);
        }
        return anchors;
    }

    /** The '##' and '###' headings after Contents, in document order. */
    private List<ContentEntry> sectionEntries() throws IOException {
        Heading contents = contentsHeading();
        List<ContentEntry> entries = new ArrayList<>();
        for (Heading heading : headings()) {
            if (heading.line > contents.line && (heading.level == 2 || heading.level == 3)) {
                entries.add(new ContentEntry(heading.line, heading.level, 0, heading.text, heading.anchor));
            }
        }
        return entries;
    }

    /** The numbered '##' entries and three-space-indented '###' bullets of the Contents section. */
    private List<ContentEntry> contentsEntries() throws IOException {
        Heading contents = contentsHeading();
        int end = lines().size() + 1;
        for (Heading heading : headings()) {
            if (heading.level == 2 && heading.line > contents.line) {
                end = heading.line;
                break;
            }
        }
        List<ContentEntry> entries = new ArrayList<>();
        for (Line line : usageProseLines()) {
            if (line.number <= contents.line || line.number >= end || line.text.trim().isEmpty()) {
                continue;
            }
            Matcher top = TOP_ENTRY.matcher(line.text);
            if (top.matches()) {
                entries.add(new ContentEntry(line.number, 2, Integer.parseInt(top.group(1)), top.group(2),
                        top.group(3)));
                continue;
            }
            Matcher section = SECTION_ENTRY.matcher(line.text);
            if (section.matches()) {
                entries.add(new ContentEntry(line.number, 3, 0, section.group(1), section.group(2)));
                continue;
            }
            fail("USAGE.md line " + line.number + " is in Contents but is neither a numbered '##' entry nor a"
                    + " three-space-indented '###' bullet: " + line.text);
        }
        return entries;
    }

    private Heading contentsHeading() throws IOException {
        for (Heading heading : headings()) {
            if (heading.level == 2 && heading.text.equals("Contents")) {
                return heading;
            }
        }
        throw new AssertionError("USAGE.md has no '## Contents' heading");
    }

    /** GitHub's heading anchor: lowercase, punctuation dropped except '-' and '_', spaces to '-'. */
    private static String slug(String heading) {
        String visible = INLINE_LINK.matcher(heading).replaceAll("$1").toLowerCase(Locale.ROOT);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < visible.length(); i++) {
            char c = visible.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '-' || c == '_') {
                out.append(c);
            } else if (Character.isWhitespace(c)) {
                out.append('-');
            }
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ file access

    /** The prose lines of USAGE.md, with fenced code blocks removed. */
    private List<Line> usageProseLines() throws IOException {
        return proseLines("USAGE.md");
    }

    /** The prose lines of {@code name} from the repository root, with fenced code blocks removed. */
    private List<Line> proseLines(String name) throws IOException {
        List<String> raw = Files.readAllLines(repositoryFile(name), StandardCharsets.UTF_8);
        List<Line> prose = new ArrayList<>();
        String fence = null;
        for (int i = 0; i < raw.size(); i++) {
            String trimmed = raw.get(i).trim();
            if (fence == null) {
                if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
                    fence = trimmed.substring(0, 3);
                } else {
                    prose.add(new Line(i + 1, raw.get(i)));
                }
            } else if (trimmed.startsWith(fence)) {
                fence = null;
            }
        }
        return prose;
    }

    private List<String> lines() throws IOException {
        return Files.readAllLines(repositoryFile("USAGE.md"), StandardCharsets.UTF_8);
    }

    /** {@code name} at the repository root, found from the working directory upward; settings.gradle pins it. */
    private static Path repositoryFile(String name) {
        Path dir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        for (int i = 0; i < 8; i++) {
            Path file = dir.resolve(name);
            if (Files.isRegularFile(file)) {
                return file;
            }
            if (Files.isRegularFile(dir.resolve("settings.gradle"))) {
                break;
            }
            Path parent = dir.getParent();
            if (parent == null) {
                break;
            }
            dir = parent;
        }
        throw new AssertionError(name + " not found between " + Paths.get("").toAbsolutePath()
                + " and the repo root; the doc-drift test needs it");
    }

    private static String describe(List<ContentEntry> entries) {
        StringBuilder out = new StringBuilder();
        for (ContentEntry entry : entries) {
            for (int i = 0; i < entry.level - 2; i++) {
                out.append("  ");
            }
            out.append(entry.level == 2 ? entry.number + ". " : "- ").append(entry.text).append(" (#")
                    .append(entry.anchor).append(")\n");
        }
        return out.toString();
    }

    /** One prose line of a document. */
    private static final class Line {
        private final int number;
        private final String text;

        private Line(int number, String text) {
            this.number = number;
            this.text = text;
        }
    }

    /** One heading of a document, with the line it starts on and its anchor. */
    private static final class Heading {
        private final int line;
        private final int level;
        private final String text;
        private final String anchor;

        private Heading(int line, int level, String text, String anchor) {
            this.line = line;
            this.level = level;
            this.text = text;
            this.anchor = anchor;
        }
    }

    /** One Contents entry: a numbered '##' entry (level 2) or a nested '###' bullet (level 3). */
    private static final class ContentEntry {
        private final int line;
        private final int level;
        private final int number;
        private final String text;
        private final String anchor;

        private ContentEntry(int line, int level, int number, String text, String anchor) {
            this.line = line;
            this.level = level;
            this.number = number;
            this.text = text;
            this.anchor = anchor;
        }
    }
}
