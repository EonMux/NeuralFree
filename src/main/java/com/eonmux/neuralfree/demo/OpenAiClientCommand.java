package com.eonmux.neuralfree.demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The machinery the OpenAI client subcommands {@code chat} and {@code complete} share: the model choice of vLLM's
 * {@code _interactive_cli}, the streamed answer printing of its {@code _print_chat_stream} and
 * {@code _print_completion_stream} (the reasoning between {@code <think>} and {@code </think>}, and with
 * {@code --stats} the TTFT and the TPS), and the {@code > } prompt both read their input at.
 */
final class OpenAiClientCommand {

    /** The prompt printed before each input line (vLLM's {@code input("> ")}). */
    static final String PROMPT = "> ";

    private OpenAiClientCommand() {
    }

    /** A connected session: the client and the model every request names. */
    static final class Session {

        private final ChatClient client;
        private final String model;

        private Session(ChatClient client, String model) {
            this.client = client;
            this.model = model;
        }

        /** The connected client. */
        ChatClient client() {
            return client;
        }

        /** The model every request names. */
        String model() {
            return model;
        }
    }

    /**
     * Builds the client of the options' URL and chooses the model: {@code --model-name}, else the first model the
     * server lists on {@code /models}. The chosen model prints as the reference prints it.
     *
     * @return the session, or null after printing why the server cannot be used
     */
    static Session connect(OpenAiClientOptions options, PrintStream out, PrintStream err) {
        ChatClient client;
        try {
            client = ChatClient.forApiBase(options.url(), options.apiKey());
        } catch (IllegalArgumentException e) {
            err.println("Error: " + e.getMessage());
            return null;
        }
        String model = options.modelName();
        if (model == null || model.isEmpty()) {
            try {
                List<ChatClient.ModelInfo> models = client.models();
                if (models.isEmpty()) {
                    err.println("Error: the server lists no models on " + client.base() + "/models");
                    return null;
                }
                model = models.get(0).id;
            } catch (IOException e) {
                err.println("Error: " + e.getMessage());
                return null;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                err.println("Error: interrupted while listing the server's models");
                return null;
            }
        }
        out.println("Using model: " + model);
        return new Session(client, model);
    }

    /**
     * Sends one streamed request and prints its answer as it arrives. The answer ends with the newline the
     * reference prints.
     *
     * @param chat whether the request is a chat completion ({@code false} for a text completion)
     * @return the answer, whose content is the generated text; null after printing the error of a failed request
     */
    static Answer send(Session session, Map<String, Object> body, boolean chat, boolean stats, PrintStream out,
                       PrintStream err) {
        Answer answer = new Answer(out, stats);
        String error;
        try {
            error = chat ? session.client().streamChat(body, answer, () -> false)
                    : session.client().streamCompletion(body, answer, () -> false);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            error = "interrupted while waiting for the server";
        }
        if (!error.isEmpty()) {
            answer.abort();
            err.println("Error: " + error);
            return null;
        }
        answer.finish();
        return answer;
    }

    /**
     * Prints the {@code > } prompt and reads one line: null at the end of input, which ends the reference's loop
     * as its {@code EOFError} does. A blank line is an empty message.
     */
    static String read(BufferedReader in, PrintStream out) throws IOException {
        out.print(PROMPT);
        out.flush();
        return in.readLine();
    }

    /**
     * The printer of one streamed answer (vLLM's {@code _print_chat_stream}): the reasoning under a
     * {@code <think>} header until content starts, the content as it arrives, and with {@code --stats} the TTFT
     * from the response headers to the first token and the TPS over the whole stream.
     */
    static final class Answer implements ChatClient.Deltas {

        private final PrintStream out;
        private final boolean stats;
        private final StringBuilder content = new StringBuilder();
        private long start = -1;
        private long firstToken = -1;
        private int completionTokens;
        private boolean inReasoning;
        private boolean printed;

        Answer(PrintStream out, boolean stats) {
            this.out = out;
            this.stats = stats;
        }

        @Override
        public void started() {
            start = System.nanoTime();
        }

        @Override
        public void reasoning(String text) {
            tokenArrived(text);
            if (!inReasoning) {
                out.print("<think>");
                inReasoning = true;
                printed = true;
            }
            out.print(text);
            out.flush();
        }

        @Override
        public void content(String text) {
            tokenArrived(text);
            if (inReasoning) {
                out.print("\n</think>");
                inReasoning = false;
            }
            content.append(text);
            out.print(text);
            out.flush();
        }

        @Override
        public void timings(double promptPerSecond, double predictedPerSecond) {
            // llama.cpp's per-token timings do not answer these requests; --stats measures its own
        }

        @Override
        public void usage(int tokens) {
            completionTokens = tokens;
        }

        /** The generated content of the answer, reasoning excluded. */
        String content() {
            return content.toString();
        }

        /** Ends a finished answer: closes an open reasoning block, ends the line and prints its metrics. */
        void finish() {
            closeReasoning();
            out.println();
            printStats();
        }

        /** Ends a failed answer: closes an open reasoning block and the line when anything was printed. */
        void abort() {
            closeReasoning();
            if (printed) {
                out.println();
            }
        }

        private void tokenArrived(String text) {
            if (firstToken < 0 && start >= 0 && text != null && !text.isEmpty()) {
                firstToken = System.nanoTime();
                printed = true;
            }
        }

        private void closeReasoning() {
            if (inReasoning) {
                out.print("\n</think>");
                inReasoning = false;
            }
        }

        private void printStats() {
            if (!stats || firstToken < 0 || start < 0) {
                return;
            }
            double total = (System.nanoTime() - start) / 1e9;
            if (total <= 0) {
                return;
            }
            out.printf(Locale.ROOT, "TTFT: %.2f ms%n", (firstToken - start) / 1e6);
            out.printf(Locale.ROOT, "TPS:  %.2f tokens/s (%d tokens in %.2fs)%n",
                    completionTokens / total, completionTokens, total);
        }
    }
}
