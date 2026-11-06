package com.eonmux.neuralfree.demo;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Transcript} writes the exact shapes of llama.cpp tools/cli/cli-context.cpp {@code write_output_file},
 * replaces an existing file, flushes each write, and names the path it cannot open.
 */
public class TranscriptTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theEntriesFollowTheReferenceShapes() throws IOException {
        Path file = tmp.getRoot().toPath().resolve("chat.txt");
        Files.write(file, "stale".getBytes(StandardCharsets.UTF_8));
        try (Transcript t = Transcript.open(file.toString())) {
            t.userFile("notes.md");
            t.userMedia("cat.png");
            t.userTurn("hello");
            assertEquals("each write is flushed",
                    "User: Added text file: notes.md\nUser: Added media: cat.png\nUser:\nhello\n\n", read(file));
            t.assistant("answer", "");
            t.assistant("done\n", "thought");
        }
        assertEquals("User: Added text file: notes.md\n"
                + "User: Added media: cat.png\n"
                + "User:\nhello\n\n"
                + "Assistant:\nanswer\n\n"
                + "Assistant:\n[Start thinking]\n\nthought[End thinking]\n\ndone\n\n", read(file));
    }

    @Test
    public void anUnopenableFileIsNamed() {
        String path = tmp.getRoot().toPath().resolve("missing/dir/out.txt").toString();
        IOException e = assertThrows(IOException.class, () -> Transcript.open(path));
        assertTrue(e.getMessage(), e.getMessage().contains("failed to open output file '" + path + "'"));
    }

    private static String read(Path file) throws IOException {
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
