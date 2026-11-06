package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * USAGE.md's runnable projector lists against {@link ProjectorRegistry}: every type the registry runs is listed in
 * its modality's list, every type the lists name as runnable is runnable for its registry modality, the refusal
 * sentence names every type the registry refuses. Fails when a family lands in {@link ProjectorRegistry} without its USAGE.md entry, so the lists
 * cannot drift.
 */
public class UsageProjectorListsTest {

    private static final Pattern BACKTICK = Pattern.compile("`([^`]*)`");
    private static final Pattern AUDIO_TABLE_ROW = Pattern.compile("^\\|\\s*`([^`]+)`\\s*\\|.*");
    private static final Pattern LIST_ITEM = Pattern.compile("^\\s*-\\s+\\S.*");

    private List<String> lines;

    /** Every runnable vision type has its bullet in the vision section's runnable list. */
    @Test
    public void everyRunnableVisionTypeIsListed() throws IOException {
        List<String> listed = visionRunnableNames();
        assertFalse("USAGE.md's runnable vision list parsed empty", listed.isEmpty());
        for (ProjectorType t : runnable(ProjectorModality.VISION)) {
            assertTrue("the runnable vision projector type '" + t.typeName() + "' has no bullet in USAGE.md's"
                    + " runnable list; add it there or remove its row from ProjectorRegistry",
                    listed.contains(t.typeName()));
        }
    }

    /** Every runnable audio type is a row of the audio section's projector table. */
    @Test
    public void everyRunnableAudioTypeIsListed() throws IOException {
        List<String> listed = audioTableNames();
        assertFalse("USAGE.md's audio projector table parsed empty", listed.isEmpty());
        for (ProjectorType t : runnable(ProjectorModality.AUDIO)) {
            assertTrue("the runnable audio projector type '" + t.typeName() + "' has no row in USAGE.md's audio"
                    + " table; add it there or remove its row from ProjectorRegistry", listed.contains(t.typeName()));
        }
    }

    /** Every projector type the runnable vision list names is runnable, and runs the vision encoder. */
    @Test
    public void visionListNamesOnlyRunnableVisionTypes() throws IOException {
        for (String name : visionRunnableNames()) {
            ProjectorType t = ProjectorType.fromName(name);
            if (t == ProjectorType.UNKNOWN) {
                continue;
            }
            assertTrue("USAGE.md's runnable vision list names '" + name + "', which ProjectorRegistry does not run",
                    ProjectorRegistry.has(name));
            assertEquals("USAGE.md's runnable vision list names '" + name + "', which the registry registers for"
                            + " another encoder", ProjectorModality.VISION, ProjectorRegistry.get(name).modality());
        }
    }

    /** Every projector type the audio table names is runnable, and runs the audio encoder. */
    @Test
    public void audioTableNamesOnlyRunnableAudioTypes() throws IOException {
        for (String name : audioTableNames()) {
            ProjectorType t = ProjectorType.fromName(name);
            if (t == ProjectorType.UNKNOWN) {
                continue;
            }
            assertTrue("USAGE.md's audio table names '" + name + "', which ProjectorRegistry does not run",
                    ProjectorRegistry.has(name));
            assertEquals("USAGE.md's audio table names '" + name + "', which the registry registers for another"
                    + " encoder", ProjectorModality.AUDIO, ProjectorRegistry.get(name).modality());
        }
    }

    /** The refusal sentence names every named type the registry refuses. */
    @Test
    public void refusalSentenceNamesEveryRefusedType() throws IOException {
        List<String> named = backticked(refusalLine());
        for (ProjectorType t : ProjectorType.named()) {
            if (ProjectorRegistry.has(t.typeName())) {
                continue;
            }
            assertTrue("USAGE.md's refusal sentence does not name '" + t.typeName() + "', which the projector"
                    + " registry refuses; name it there", named.contains(t.typeName()));
        }
    }

    // ------------------------------------------------------------------ USAGE.md parsing

    /** The backticked names of the runnable list under "Runnable projector types:". */
    private List<String> visionRunnableNames() throws IOException {
        int header = lineContaining("Runnable projector types", visionStart(), audioStart());
        List<String> names = new ArrayList<>();
        for (int i = header + 1; i < audioStart() && LIST_ITEM.matcher(lines().get(i)).matches(); i++) {
            names.addAll(backticked(lines().get(i)));
        }
        return names;
    }

    /** The first-column names of the audio projector table. */
    private List<String> audioTableNames() throws IOException {
        int header = lineContaining("Supported audio projectors", audioStart(), audioEnd());
        List<String> names = new ArrayList<>();
        for (int i = header + 1; i < audioEnd(); i++) {
            Matcher row = AUDIO_TABLE_ROW.matcher(lines().get(i));
            if (row.matches()) {
                names.add(row.group(1));
            }
        }
        return names;
    }

    /** The vision section's line naming the types the registry refuses. */
    private String refusalLine() throws IOException {
        int afterList = lineContaining("Runnable projector types", visionStart(), audioStart()) + 1;
        while (LIST_ITEM.matcher(lines().get(afterList)).matches()) {
            afterList++;
        }
        return lines().get(lineContaining("refused by name", afterList, audioStart()));
    }

    private static List<String> backticked(String text) {
        List<String> names = new ArrayList<>();
        Matcher m = BACKTICK.matcher(text);
        while (m.find()) {
            names.add(m.group(1));
        }
        return names;
    }

    private List<String> lines() throws IOException {
        if (lines == null) {
            lines = Files.readAllLines(findUsageMd(), StandardCharsets.UTF_8);
        }
        return lines;
    }

    private int visionStart() throws IOException {
        return lineStartingWith("### Vision encoders");
    }

    private int audioStart() throws IOException {
        return lineStartingWith("### Audio input");
    }

    private int audioEnd() throws IOException {
        for (int i = audioStart() + 1; i < lines().size(); i++) {
            if (lines().get(i).startsWith("#")) {
                return i;
            }
        }
        return lines().size();
    }

    private int lineStartingWith(String prefix) throws IOException {
        for (int i = 0; i < lines().size(); i++) {
            if (lines().get(i).startsWith(prefix)) {
                return i;
            }
        }
        throw new AssertionError("USAGE.md has no line starting with '" + prefix + "'");
    }

    private int lineContaining(String contains, int from, int to) throws IOException {
        for (int i = from; i < to; i++) {
            if (lines().get(i).contains(contains)) {
                return i;
            }
        }
        throw new AssertionError("USAGE.md has no line containing '" + contains + "' between lines " + from
                + " and " + to);
    }

    /** The repo root's USAGE.md, found from the working directory upward; the repo root pins the walk. */
    private static Path findUsageMd() throws IOException {
        Path dir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        for (int i = 0; i < 8; i++) {
            Path usage = dir.resolve("USAGE.md");
            if (Files.isRegularFile(usage)) {
                return usage;
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
        throw new AssertionError("USAGE.md not found between " + Paths.get("").toAbsolutePath()
                + " and the repo root; the doc-drift test needs it");
    }

    /** The named types whose own name the registry runs for {@code modality}. */
    private static List<ProjectorType> runnable(ProjectorModality modality) {
        List<ProjectorType> out = new ArrayList<>();
        for (ProjectorType t : ProjectorType.named()) {
            if (ProjectorRegistry.has(t.typeName())
                    && ProjectorRegistry.get(t.typeName()).modality() == modality) {
                out.add(t);
            }
        }
        return out;
    }
}
