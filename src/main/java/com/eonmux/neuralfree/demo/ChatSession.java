package com.eonmux.neuralfree.demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The interactive chat loop of the console, with the command set of llama.cpp tools/cli/cli-context.cpp
 * {@code run}: {@code /exit}, {@code /regen}, {@code /clear}, {@code /read}, {@code /glob}, {@code /image},
 * {@code /audio} and {@code /video}, multiline input ({@code -mli}), Ctrl-C handling ({@link Interrupts}) and the
 * {@code -o} transcript. The session owns the OpenAI-shaped message list; a {@link ChatBackend} answers each turn.
 * Text files are prepended to the next message as {@code --- File: name ---} blocks, media files become content
 * parts of the next message, and a media kind the backend cannot take is refused by name.
 */
final class ChatSession {

    /** The most files one {@code /glob} attaches (tools/cli/cli-context.cpp {@code FILE_GLOB_MAX_RESULTS}). */
    static final int FILE_GLOB_MAX_RESULTS = 100;

    private static final String[] MEDIA_KINDS = {"image", "audio", "video"};

    /** What one input asks the loop to do. */
    private enum Step { EXIT, NEXT_INPUT, NEW_TURN, REGENERATE }

    private final ChatBackend backend;
    private final String systemPrompt;
    private final boolean multiline;
    private final boolean singleTurn;
    private final String firstPrompt;
    private final List<LLMDemoOptions.MediaFile> firstMedia;
    private final BufferedReader in;
    private final PrintStream out;
    private final Transcript transcript;
    private final Interrupts interrupts;
    private final List<Map<String, Object>> messages = new ArrayList<>();
    private final List<Object> pendingMedia = new ArrayList<>();
    private final StringBuilder pendingText = new StringBuilder();

    /**
     * A session over {@code backend} with the chat settings of {@code opts} (system prompt, {@code -mli},
     * {@code -st}, the {@code -p} first turn and its {@code --image}/{@code --audio}/{@code --video} files).
     *
     * @param transcript the {@code -o} transcript, or null for none
     */
    ChatSession(ChatBackend backend, LLMDemoOptions opts, BufferedReader in, PrintStream out,
                Transcript transcript, Interrupts interrupts) {
        if (backend == null || opts == null || in == null || out == null || interrupts == null) {
            throw new IllegalArgumentException("backend, options, streams and interrupts are required");
        }
        this.backend = backend;
        this.systemPrompt = opts.systemPrompt == null ? "" : opts.systemPrompt;
        this.multiline = opts.multilineInput;
        this.singleTurn = opts.singleTurn;
        this.firstPrompt = opts.prompt;
        this.firstMedia = opts.media;
        this.in = in;
        this.out = out;
        this.transcript = transcript;
        this.interrupts = interrupts;
    }

    /** Runs the loop until {@code /exit}, end of input, Ctrl-C at the prompt, or the single turn's answer. */
    void run() throws IOException {
        printBanner();
        addSystemPrompt();
        MultilineInput input = new MultilineInput(in);
        String prompt = firstPrompt;
        while (true) {
            String buffer;
            if (prompt != null) {
                stageArgumentMedia();
                buffer = prompt;
                out.println("> " + buffer);
                prompt = null;
            } else {
                buffer = readInput(input);
            }
            if (interrupts.requested()) {
                interrupts.clear();
                break;
            }
            if (buffer == null) {
                break;
            }
            if (buffer.isEmpty()) {
                continue;
            }
            Step step = dispatch(buffer);
            if (step == Step.EXIT) {
                break;
            }
            if (step == Step.NEXT_INPUT) {
                continue;
            }
            respond(step, buffer);
            if (singleTurn) {
                break;
            }
        }
        out.println("\nExiting...");
    }

    /** The conversation so far, as the backend receives it. */
    List<Map<String, Object>> messages() {
        return Collections.unmodifiableList(new ArrayList<>(messages));
    }

    private String readInput(MultilineInput input) throws IOException {
        out.print("> ");
        out.flush();
        interrupts.awaitingInput(true);
        try {
            return input.read(multiline);
        } finally {
            interrupts.awaitingInput(false);
        }
    }

    /** Runs a command and reports what the loop does next; plain text starts a new turn. */
    private Step dispatch(String buffer) throws IOException {
        String word = buffer.trim();
        if (buffer.startsWith("/exit") || buffer.startsWith("/quit") || word.equalsIgnoreCase("exit")
                || word.equalsIgnoreCase("quit")) {
            return Step.EXIT;
        }
        if (buffer.startsWith("/regen")) {
            return regenerable() ? Step.REGENERATE : Step.NEXT_INPUT;
        }
        if (buffer.startsWith("/clear") || buffer.startsWith("/reset") || word.equalsIgnoreCase("clear")
                || word.equalsIgnoreCase("reset")) {
            clear();
            return Step.NEXT_INPUT;
        }
        if (buffer.startsWith("/help")) {
            printCommands();
            return Step.NEXT_INPUT;
        }
        for (String kind : MEDIA_KINDS) {
            if (buffer.startsWith("/" + kind + " ")) {
                stageMedia(kind, buffer.substring(kind.length() + 2).trim());
                return Step.NEXT_INPUT;
            }
        }
        if (buffer.startsWith("/read ")) {
            String fileName = buffer.substring(6).trim();
            if (addTextFile(Paths.get(fileName), fileName)) {
                writeTranscript(() -> transcript.userFile(fileName));
            }
            return Step.NEXT_INPUT;
        }
        if (buffer.startsWith("/glob ")) {
            globFiles(buffer.substring(6).trim());
            return Step.NEXT_INPUT;
        }
        return Step.NEW_TURN;
    }

    /** Whether {@code /regen} has an answer to replace, printing the reference's error when not. */
    private boolean regenerable() {
        if (messages.isEmpty() || !"assistant".equals(messages.get(messages.size() - 1).get("role"))) {
            out.println("Error: No message to regenerate.");
            return false;
        }
        return true;
    }

    private void clear() {
        messages.clear();
        addSystemPrompt();
        pendingMedia.clear();
        pendingText.setLength(0);
        backend.reset();
        out.println("Chat history cleared.");
    }

    /**
     * Answers one turn. A new turn adds the pending files, media and text as the next user message; a
     * regeneration drops the last answer. A failed turn leaves the conversation as it was before the turn; an
     * interrupted one keeps the partial answer, as the reference does.
     */
    private void respond(Step step, String buffer) throws IOException {
        List<Map<String, Object>> before = new ArrayList<>(messages);
        if (step == Step.REGENERATE) {
            messages.remove(messages.size() - 1);
        } else {
            pendingText.append(buffer);
            String text = pendingText.toString();
            pushUserMessage(text);
            writeTranscript(() -> transcript.userTurn(text));
        }
        ChatAnswer answer;
        try {
            answer = backend.answer(messages(), interrupts::requested);
        } finally {
            interrupts.clear();
        }
        if (answer.error() != null) {
            out.println("Error: " + answer.error());
            messages.clear();
            messages.addAll(before);
            return;
        }
        Map<String, Object> assistant = new LinkedHashMap<>();
        assistant.put("role", "assistant");
        assistant.put("content", answer.content());
        if (!answer.reasoning().isEmpty()) {
            assistant.put("reasoning_content", answer.reasoning());
        }
        messages.add(assistant);
        writeTranscript(() -> transcript.assistant(answer.content(), answer.reasoning()));
    }

    /** The pending media parts, then the text, become the next user message (cli-context.cpp push_user_message). */
    private void pushUserMessage(String text) {
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        if (pendingMedia.isEmpty()) {
            user.put("content", text);
        } else {
            List<Object> content = new ArrayList<>(pendingMedia);
            Map<String, Object> textPart = new LinkedHashMap<>();
            textPart.put("type", "text");
            textPart.put("text", text);
            content.add(textPart);
            user.put("content", content);
            pendingMedia.clear();
        }
        messages.add(user);
        pendingText.setLength(0);
    }

    private void addSystemPrompt() {
        if (!systemPrompt.isEmpty()) {
            Map<String, Object> system = new LinkedHashMap<>();
            system.put("role", "system");
            system.put("content", systemPrompt);
            messages.add(system);
        }
    }

    /**
     * Stages the {@code --image}/{@code --audio}/{@code --video} files of the {@code -p} turn, each as the kind
     * its flag gave it, stopping at the first failure.
     */
    private void stageArgumentMedia() throws IOException {
        for (LLMDemoOptions.MediaFile file : firstMedia) {
            if (!stageMedia(file.kind, file.path)) {
                break;
            }
        }
    }

    /** Stages one media file of {@code kind} as a content part of the next message. */
    private boolean stageMedia(String kind, String fileName) throws IOException {
        String refusal = backend.mediaRefusal(kind);
        if (refusal != null) {
            out.println("Error: /" + kind + " is refused: " + refusal);
            return false;
        }
        Object part;
        try {
            part = MediaParts.part(kind, fileName);
        } catch (IOException e) {
            out.println("Error: file does not exist or cannot be opened: '" + fileName + "'");
            return false;
        }
        pendingMedia.add(part);
        out.println("Loaded media from '" + fileName + "'");
        writeTranscript(() -> transcript.userMedia(fileName));
        return true;
    }

    /** Appends one text file to the next message (cli-context.cpp add_text_file). */
    private boolean addTextFile(Path path, String fileName) {
        String content;
        try {
            if (!Files.isRegularFile(path)) {
                throw new IOException("not a regular file");
            }
            content = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        } catch (IOException | RuntimeException e) {
            out.println("Error: file does not exist or cannot be opened: '" + fileName + "'");
            return false;
        }
        pendingText.append("--- File: ").append(fileName).append(" ---\n").append(content);
        out.println("Loaded text from '" + fileName + "'");
        return true;
    }

    /** Appends every file the glob matches to the next message, at most {@link #FILE_GLOB_MAX_RESULTS}. */
    private void globFiles(String pattern) throws IOException {
        List<Path> files = GlobFiles.split(Paths.get("").toAbsolutePath(), pattern).matches();
        int count = 0;
        for (Path file : files) {
            String name = file.toString();
            if (!addTextFile(file, name)) {
                continue;
            }
            writeTranscript(() -> transcript.userFile(name));
            if (++count >= FILE_GLOB_MAX_RESULTS) {
                out.println("Error: Maximum number of globbed files allowed (" + FILE_GLOB_MAX_RESULTS
                        + ") reached.");
                break;
            }
        }
        if (count == 0) {
            out.println("Error: no files match '" + pattern + "'");
        }
    }

    private void printBanner() {
        out.println();
        backend.printBanner(out);
        out.println();
        printCommands();
        out.println(" - Ctrl+C interrupts a generating answer; at the prompt (or twice) it exits.");
        if (multiline) {
            out.println(" - Multiline input: end the input with '/' or '\\'.");
        } else {
            out.println(" - End a line with '\\' to continue on the next line.");
        }
        out.println();
    }

    private void printCommands() {
        out.println("available commands:");
        out.println("  /exit or Ctrl+C     stop or exit");
        out.println("  /regen              regenerate the last response");
        out.println("  /clear              clear the chat history");
        out.println("  /read <file>        add a text file");
        out.println("  /glob <pattern>     add text files using globbing pattern");
        for (String kind : MEDIA_KINDS) {
            if (backend.mediaRefusal(kind) == null) {
                out.println(String.format(Locale.ROOT, "  /%-18s add %s %s file", kind + " <file>",
                        kind.equals("video") ? "a" : "an", kind));
            }
        }
        out.println();
    }

    /** One transcript write. */
    private interface TranscriptWrite {
        void write() throws IOException;
    }

    /** Runs {@code write} when there is a transcript, reporting a failed write without ending the session. */
    private void writeTranscript(TranscriptWrite write) {
        if (transcript == null) {
            return;
        }
        try {
            write.write();
        } catch (IOException e) {
            out.println("Error: cannot write the output file: " + e.getMessage());
        }
    }
}
