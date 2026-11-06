package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link ChatSession} runs the llama.cpp CLI command set over a scripted backend: files and media join the next
 * user message in the reference's shapes, {@code /regen} re-answers the same history, {@code /clear} drops it,
 * unsupported media is refused by name, a failed turn leaves the conversation as it was, Ctrl-C cancels a turn
 * and exits at the prompt, and the transcript records every step.
 */
public class ChatSessionTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    /** A backend that answers "answer N", records every request, and can fail or watch for a cancel. */
    private static final class ScriptedBackend implements ChatBackend {
        final List<List<Map<String, Object>>> requests = new ArrayList<>();
        final List<String> accepted = new ArrayList<>();
        String failWith;
        Runnable duringAnswer;
        boolean sawCancel;
        int resets;

        @Override
        public void printBanner(PrintStream out) {
            out.println("model      : scripted");
        }

        @Override
        public String mediaRefusal(String kind) {
            return accepted.contains(kind) ? null : "the scripted backend takes no " + kind;
        }

        @Override
        public ChatAnswer answer(List<Map<String, Object>> messages, BooleanSupplier cancelled) {
            requests.add(messages);
            if (duringAnswer != null) {
                duringAnswer.run();
            }
            sawCancel = cancelled.getAsBoolean();
            if (failWith != null) {
                return ChatAnswer.failed("partial", "", failWith);
            }
            return ChatAnswer.of("answer " + requests.size(), requests.size() == 1 ? "thinking" : "");
        }

        @Override
        public void reset() {
            resets++;
        }
    }

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();
    private final List<Integer> exits = new ArrayList<>();
    private final Interrupts interrupts = new Interrupts(exits::add,
            new PrintStream(new ByteArrayOutputStream(), true));

    private ChatSession session(ScriptedBackend backend, String input, Transcript transcript, String... args)
            throws UsageException {
        LLMDemoOptions opts = LLMDemoOptions.parse(args);
        return new ChatSession(backend, opts, new BufferedReader(new StringReader(input)),
                new PrintStream(console, true), transcript, interrupts);
    }

    private String output() {
        return new String(console.toByteArray(), StandardCharsets.UTF_8);
    }

    @Test
    public void turnsCarryTheSystemPromptHistoryAndReasoning() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        ChatSession chat = session(backend, "hi\nmore\n", null, "--system", "be brief");
        chat.run();
        assertEquals(2, backend.requests.size());
        List<Map<String, Object>> second = backend.requests.get(1);
        assertEquals(4, second.size());
        assertEquals("system", second.get(0).get("role"));
        assertEquals("be brief", second.get(0).get("content"));
        assertEquals("hi", second.get(1).get("content"));
        assertEquals("answer 1", second.get(2).get("content"));
        assertEquals("the reasoning goes back as the reference sends it", "thinking",
                second.get(2).get("reasoning_content"));
        assertEquals("more", second.get(3).get("content"));
        assertEquals(5, chat.messages().size());
        assertTrue(output().contains("Exiting..."));
    }

    @Test
    public void anEmptySystemPromptAddsNoSystemMessage() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "hi\n", null, "--system", "").run();
        assertEquals(1, backend.requests.get(0).size());
        assertEquals("user", backend.requests.get(0).get(0).get("role"));
    }

    @Test
    public void readAndGlobPrependFileBlocksToTheNextMessage() throws Exception {
        Path dir = tmp.newFolder("files").toPath();
        Path a = dir.resolve("a.md");
        Path b = dir.resolve("b.md");
        Files.write(a, "alpha\n".getBytes(StandardCharsets.UTF_8));
        Files.write(b, "beta\n".getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("c.txt"), "gamma".getBytes(StandardCharsets.UTF_8));
        ScriptedBackend backend = new ScriptedBackend();
        String input = "/read " + a + "\n/glob " + dir + "/*.md\n/read " + dir.resolve("nope") + "\nsummarize\n";
        session(backend, input, null, "--system", "").run();
        String content = (String) backend.requests.get(0).get(0).get("content");
        assertEquals("--- File: " + a + " ---\nalpha\n--- File: " + a + " ---\nalpha\n--- File: " + b
                + " ---\nbeta\nsummarize", content);
        assertTrue(output().contains("Loaded text from '" + a + "'"));
        assertTrue(output().contains("Error: file does not exist or cannot be opened: '" + dir.resolve("nope")
                + "'"));
    }

    @Test
    public void globCapsTheAttachedFiles() throws Exception {
        Path dir = tmp.newFolder("many").toPath();
        for (int i = 0; i < ChatSession.FILE_GLOB_MAX_RESULTS + 3; i++) {
            Files.write(dir.resolve(String.format("f%03d.txt", i)), "x".getBytes(StandardCharsets.UTF_8));
        }
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "/glob " + dir + "/*.txt\ngo\n", null, "--system", "").run();
        String content = (String) backend.requests.get(0).get(0).get("content");
        assertEquals(ChatSession.FILE_GLOB_MAX_RESULTS, content.split("--- File: ", -1).length - 1);
        assertTrue(output().contains("Maximum number of globbed files allowed (100) reached."));
    }

    @Test
    public void regenReanswersTheSameHistory() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        ChatSession chat = session(backend, "/regen\nhi\n/regen\n", null, "--system", "");
        chat.run();
        assertTrue("nothing to regenerate before the first answer", output().contains(
                "Error: No message to regenerate."));
        assertEquals(2, backend.requests.size());
        assertEquals(backend.requests.get(0), backend.requests.get(1));
        List<Map<String, Object>> messages = chat.messages();
        assertEquals(2, messages.size());
        assertEquals("the regenerated answer replaces the first", "answer 2", messages.get(1).get("content"));
    }

    @Test
    public void clearDropsHistoryAndPendingFilesAndResetsTheBackend() throws Exception {
        Path file = tmp.newFile("pending.txt").toPath();
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "hi\n/read " + file + "\n/clear\nagain\n", null, "--system", "sys").run();
        assertEquals(1, backend.resets);
        List<Map<String, Object>> after = backend.requests.get(1);
        assertEquals(2, after.size());
        assertEquals("system", after.get(0).get("role"));
        assertEquals("again", after.get(1).get("content"));
        assertTrue(output().contains("Chat history cleared."));
    }

    @Test
    public void mediaTheBackendCannotTakeIsRefusedByName() throws Exception {
        Path image = tmp.newFile("cat.png").toPath();
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "/image " + image + "\n/audio " + image + "\nhi\n", null, "--system", "").run();
        assertTrue(output().contains("Error: /image is refused: the scripted backend takes no image"));
        assertTrue(output().contains("Error: /audio is refused: the scripted backend takes no audio"));
        assertEquals("the refused files stay out of the message", "hi", backend.requests.get(0).get(0)
                .get("content"));
        assertFalse("refused kinds are not listed as commands", output().contains("/image <file>"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void acceptedMediaBecomesContentPartsBeforeTheText() throws Exception {
        Path image = tmp.getRoot().toPath().resolve("dot.png");
        byte[] png = {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2};
        Files.write(image, png);
        Path audio = tmp.getRoot().toPath().resolve("voice.MP3");
        Files.write(audio, new byte[]{1, 2, 3});
        ScriptedBackend backend = new ScriptedBackend();
        backend.accepted.add("image");
        backend.accepted.add("audio");
        session(backend, "/image " + image + "\n/audio  " + audio + " \nwhat is this\n", null, "--system", "")
                .run();
        List<Object> parts = (List<Object>) backend.requests.get(0).get(0).get("content");
        assertEquals(3, parts.size());
        Map<String, Object> imagePart = (Map<String, Object>) parts.get(0);
        assertEquals("image_url", imagePart.get("type"));
        assertEquals("data:image/png;base64," + Base64.getEncoder().encodeToString(png),
                ((Map<String, Object>) imagePart.get("image_url")).get("url"));
        Map<String, Object> audioPart = (Map<String, Object>) parts.get(1);
        assertEquals("input_audio", audioPart.get("type"));
        assertEquals("mp3", ((Map<String, Object>) audioPart.get("input_audio")).get("format"));
        Map<String, Object> text = (Map<String, Object>) parts.get(2);
        assertEquals("text", text.get("type"));
        assertEquals("what is this", text.get("text"));
        assertTrue(output().contains("/image <file>"));
    }

    @Test
    public void aFailedTurnLeavesTheConversationAsItWas() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        backend.failWith = "the server is down";
        ChatSession chat = session(backend, "hi\n/regen\n", null, "--system", "");
        chat.run();
        assertTrue(output().contains("Error: the server is down"));
        assertTrue(chat.messages().isEmpty());
        assertTrue("with no answer in the history there is nothing to regenerate",
                output().contains("Error: No message to regenerate."));
    }

    @Test
    public void ctrlCDuringAnAnswerCancelsItAndKeepsThePartialAnswer() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        backend.duringAnswer = interrupts::press;
        ChatSession chat = session(backend, "hi\nnext\n", null, "--system", "");
        chat.run();
        assertTrue(backend.sawCancel);
        assertTrue("one press only cancels", exits.isEmpty());
        assertEquals("the flag is cleared after the turn, so the session goes on", 2, backend.requests.size());
        assertEquals(4, chat.messages().size());
    }

    @Test
    public void ctrlCAtThePromptExits() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        BufferedReader pressing = new BufferedReader(new StringReader("ignored\n")) {
            @Override
            public String readLine() throws IOException {
                interrupts.press();
                return super.readLine();
            }
        };
        LLMDemoOptions opts = LLMDemoOptions.parse(new String[0]);
        new ChatSession(backend, opts, pressing, new PrintStream(console, true), null, interrupts).run();
        assertEquals(List.of(130), exits);
        assertTrue("the read line is not sent", backend.requests.isEmpty());
    }

    @Test
    public void exitCommandsAndLegacyWordsEndTheSession() throws Exception {
        for (String word : new String[]{"/exit", "/quit", "quit", "EXIT"}) {
            ScriptedBackend backend = new ScriptedBackend();
            session(backend, word + "\nhi\n", null).run();
            assertTrue(word, backend.requests.isEmpty());
        }
    }

    @Test
    public void multilineInputJoinsLinesIntoOneMessage() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "line one\nline two\n\\\n", null, "--system", "", "-mli").run();
        assertEquals("line one\nline two\n", backend.requests.get(0).get(0).get("content"));
    }

    @Test
    public void theFirstPromptRunsAndSingleTurnEnds() throws Exception {
        ScriptedBackend backend = new ScriptedBackend();
        session(backend, "never read\n", null, "--system", "", "-p", "from args", "-st").run();
        assertEquals(1, backend.requests.size());
        assertEquals("from args", backend.requests.get(0).get(0).get("content"));
        assertTrue(output().contains("> from args"));
    }

    @Test
    public void theTranscriptRecordsTheConversation() throws Exception {
        Path file = tmp.newFile("notes.txt").toPath();
        Files.write(file, "n".getBytes(StandardCharsets.UTF_8));
        ScriptedBackend backend = new ScriptedBackend();
        StringWriter log = new StringWriter();
        session(backend, "/read " + file + "\nhi\n", new Transcript(log), "--system", "").run();
        assertEquals("User: Added text file: " + file + "\n"
                + "User:\n--- File: " + file + " ---\nnhi\n\n"
                + "Assistant:\n[Start thinking]\n\nthinking[End thinking]\n\nanswer 1\n\n", log.toString());
    }
}
