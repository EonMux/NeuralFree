package com.eonmux.neuralfree.demo;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;

/**
 * The conversation transcript of a console session ({@code -o/--output}): every user turn, every attached file
 * and every assistant answer is appended as plain text in the shapes llama.cpp tools/cli/cli-context.cpp
 * {@code write_output_file} writes. Opening replaces the file, and every write is flushed.
 */
final class Transcript implements Closeable {

    private final Writer out;

    /** A transcript over {@code out}, which it closes. */
    Transcript(Writer out) {
        if (out == null) {
            throw new IllegalArgumentException("out is required");
        }
        this.out = out;
    }

    /**
     * Opens {@code path} for writing, creating or replacing it.
     *
     * @throws IOException naming the path when it cannot be opened for writing
     */
    static Transcript open(String path) throws IOException {
        try {
            BufferedWriter writer = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8);
            return new Transcript(writer);
        } catch (IOException | InvalidPathException e) {
            throw new IOException("failed to open output file '" + path + "': " + e.getMessage(), e);
        }
    }

    /** One submitted user turn. */
    void userTurn(String text) throws IOException {
        write("User:\n" + text + "\n\n");
    }

    /** A text file added to the pending turn ({@code /read}, {@code /glob}). */
    void userFile(String fileName) throws IOException {
        write("User: Added text file: " + fileName + "\n");
    }

    /** A media file added to the pending turn ({@code /image}, {@code /audio}, {@code /video}). */
    void userMedia(String fileName) throws IOException {
        write("User: Added media: " + fileName + "\n");
    }

    /** One assistant answer, its reasoning between the reference's thinking markers when there is any. */
    void assistant(String content, String reasoning) throws IOException {
        StringBuilder sb = new StringBuilder("Assistant:\n");
        if (reasoning != null && !reasoning.isEmpty()) {
            sb.append("[Start thinking]\n\n").append(reasoning).append("[End thinking]\n\n");
        }
        sb.append(content == null ? "" : content);
        if (sb.charAt(sb.length() - 1) != '\n') {
            sb.append('\n');
        }
        sb.append('\n');
        write(sb.toString());
    }

    private synchronized void write(String text) throws IOException {
        out.write(text);
        out.flush();
    }

    @Override
    public synchronized void close() throws IOException {
        out.close();
    }
}
