package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * {@code ParallelDemo}: a server simulation with several clients whose requests a {@link BatchEngine}
 * processes in parallel (llama.cpp {@code examples/parallel/parallel.cpp}). Every client owns one sequence
 * slot of the paged memory; with continuous batching (the default) the next of the {@code -ns} simulated
 * requests enters as soon as a client finishes, with {@code -nocb} only once every client has finished. Each
 * request carries the shared system text, up to {@code --junk} filler dialog blocks and one question, drawn
 * from the built-in list or from the newline-separated {@code -p} / {@code -f} prompts; the engine samples
 * through the request's own sampler chain, stops at the token budget, an end-of-generation token or the
 * {@code User:} reverse prompt, and the demo prints every client's input and response with its own timing,
 * then the totals.
 *
 * <p>Seeds follow llama.cpp's signed-seed rule: a non-negative {@code -s} seeds every client alike, a negative
 * one gives client {@code i} the 32-bit seed {@code s - i}, where {@code 0xFFFFFFFF} (that is, -1) means a
 * random seed, as {@code LLAMA_DEFAULT_SEED} does.
 *
 * <p>The model opens through LLMDemo's load flags ({@link EngineLoadFlags}: {@code --backend}, {@code -ngl},
 * {@code --device}, {@code -ot}, {@code --cpu-moe}, {@code -fa}, the KV cache types and the rest) over the
 * {@code -Dneuralfree.*} properties, as llama.cpp's parallel example takes the common load flags: by default on the
 * best GPU backend, whose paged device memory the batch engine runs over, with the CPU engine as the fallback, or on
 * the CPU engine with {@code --cpu}. The header names the backend the model opened on. llama.cpp's {@code -pps}
 * system-prompt sharing is refused by name: the engine's prefix cache shares the system text's pages automatically
 * once a request that holds it has retired, and the totals report that reuse.
 */
public final class ParallelDemo {

    /** llama.cpp's {@code k_system}. */
    private static final String SYSTEM =
            "Transcript of a never ending dialog, where the User interacts with an Assistant.\n"
            + "The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's"
            + " requests immediately and with precision.\n"
            + "\n"
            + "User:\n"
            + "Recommend a nice restaurant in the area.\n"
            + "Assistant:\n"
            + "I recommend the restaurant \"The Golden Duck\". It is a 5 star restaurant with a great view of"
            + " the city. The food is delicious and the service is excellent. The prices are reasonable and the"
            + " portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The"
            + " phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm."
            + " The restaurant is closed on Saturdays and Sundays.\n"
            + "User:\n"
            + "Who is Richard Feynman?\n"
            + "Assistant:\n"
            + "Richard Feynman was an American physicist who is best known for his work in quantum mechanics"
            + " and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions"
            + " to the development of quantum electrodynamics. He was a popular lecturer and author, and he"
            + " wrote several books, including \"Surely You're Joking, Mr. Feynman!\" and \"What Do You Care"
            + " What Other People Think?\".\n";

    /** llama.cpp's {@code k_questions}. */
    private static final String[] QUESTIONS = {
        "What is the tallest mountain in the world?",
        "Who was the first person to win two Nobel Prizes?",
        "Which country invented paper?",
        "What organ is primarily responsible for pumping blood throughout the body?",
        "Which planet is known for its prominent ring system?",
        "Who directed the movie 'Inception'?",
        "What is the freezing point of water in Fahrenheit?",
        "Which animal is known to have the longest lifespan?",
        "What language has the most native speakers worldwide?",
        "What is the capital city of Canada?",
        "Who is credited with inventing the World Wide Web?",
        "Which metal is liquid at room temperature?",
        "What is the term for an animal that eats both plants and meat?",
        "Who painted 'The Starry Night'?",
        "What gas do humans exhale that plants use for photosynthesis?",
        "What year did World War II end?",
        "Which continent has the most countries?",
        "Who wrote the novel 'Frankenstein'?",
        "What does DNA stand for?",
        "What is the main ingredient in traditional Japanese miso soup?",
    };

    /** llama.cpp's {@code k_answers}. */
    private static final String[] ANSWERS = {
        "The tallest mountain in the world is Mount Everest.",
        "Marie Curie was the first person to win two Nobel Prizes.",
        "Paper was invented in China.",
        "The heart is the organ responsible for pumping blood.",
        "Saturn is known for its prominent ring system.",
        "Christopher Nolan directed the movie 'Inception'.",
        "The freezing point of water in Fahrenheit is 32\u00b0F.",
        "The bowhead whale is known to have the longest lifespan among mammals.",
        "Mandarin Chinese has the most native speakers in the world.",
        "The capital city of Canada is Ottawa.",
        "Tim Berners-Lee is credited with inventing the World Wide Web.",
        "Mercury is the metal that is liquid at room temperature.",
        "An animal that eats both plants and meat is called an omnivore.",
        "'The Starry Night' was painted by Vincent van Gogh.",
        "Humans exhale carbon dioxide, which plants use in photosynthesis.",
        "World War II ended in 1945.",
        "Africa is the continent with the most countries.",
        "The novel 'Frankenstein' was written by Mary Shelley.",
        "DNA stands for Deoxyribonucleic Acid.",
        "The main ingredient in traditional Japanese miso soup is fermented soybean paste.",
    };

    /** llama.cpp's {@code k_prompts}. */
    private static final String[] PROMPTS = {
        "What is the meaning of life?",
        "Tell me an interesting fact about llamas.",
        "What is the best way to cook a steak?",
        "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
        "Recommend some interesting books to read.",
        "What is the best way to learn a new language?",
        "How to get a job at Google?",
        "If you could have any superpower, what would it be?",
        "I want to learn how to play the piano. What would be the best way to do it?",
    };

    /** The prompt source the totals name when neither {@code -p} nor {@code -f} is given. */
    static final String BUILT_IN_PROMPTS = "used built-in defaults";
    /** llama.cpp's default {@code n_parallel} (common_params). */
    static final int DEFAULT_CLIENTS = 1;
    /** llama.cpp's default {@code n_sequences} (common_params). */
    static final int DEFAULT_SEQUENCES = 1;
    /** parallel.cpp's own {@code n_predict} default. */
    static final int DEFAULT_MAX_TOKENS = 128;
    /** parallel.cpp's own {@code n_junk} default. */
    static final int DEFAULT_JUNK = 1;
    /** llama.cpp's sampler defaults (common_params). */
    static final float DEFAULT_TEMPERATURE = 0.80f;
    static final float DEFAULT_TOP_P = 0.95f;
    static final int DEFAULT_TOP_K = 40;
    /** The reverse prompt that ends every response (parallel.cpp's {@code find("User:")}). */
    private static final String REVERSE_PROMPT = "User:";
    /** The seed of llama.cpp's {@code std::mt19937 rng(1234)}, which picks the prompts and junk counts. */
    private static final long PROMPT_RNG_SEED = 1234L;
    /** The 32-bit seed space of llama.cpp's sampler seeds; its all-ones value is the random seed. */
    private static final long SEED_MASK = 0xFFFFFFFFL;

    private static final String USAGE =
            "Usage: ParallelDemo [model.gguf] [options]\n"
            + "options (llama.cpp parallel names):\n"
            + "  -np, --parallel <n>      simulated clients, one sequence slot each (default "
                    + DEFAULT_CLIENTS + ")\n"
            + "  -ns, --sequences <n>     requests to simulate (default " + DEFAULT_SEQUENCES + ")\n"
            + "  -n, --max-tokens <n>     new tokens per request (default " + DEFAULT_MAX_TOKENS + ")\n"
            + "  --junk <n>               filler dialog blocks bound: each request takes [0, max(1, n)) of them"
                    + " (default " + DEFAULT_JUNK + ")\n"
            + "  -cb, --cont-batching     admit a request as soon as a client finishes (default)\n"
            + "  -nocb, --no-cont-batching  admit new requests only once every client has finished\n"
            + "  -p, --prompt <text>      newline-separated questions replacing the built-in ones\n"
            + "  -f, --file <path>        a file of newline-separated questions replacing the built-in ones\n"
            + "  -c, --ctx <n>            context rows (0 = fit automatically)\n"
            + "  -ub, --ubatch <n>        tokens per forward step (default "
                    + MemoryConfig.DEFAULT_UBATCH + ")\n"
            + "  -s, --seed <n>           sampler seed; negative gives client i seed n-i, as llama.cpp does\n"
            + "  --temp <f>               temperature, 0 = greedy (default " + DEFAULT_TEMPERATURE + ")\n"
            + "  --top-k <n>              top-k, 0 = off (default " + DEFAULT_TOP_K + ")\n"
            + "  --top-p <f>              top-p (default " + DEFAULT_TOP_P + ")\n"
            + "  --help, -h               this text\n"
            + EngineLoadFlags.usageSection("engine load options (as LLMDemo's):")
            + "The model opens on the best GPU backend by default (paged device memory), or on the CPU engine\n"
            + "with --cpu.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line. Immutable. */
    static final class Plan {

        final Path model;
        /**
         * The load configuration: backend, device, placement and KV types ({@link EngineLoadFlags}); the demo's
         * memory flags apply on top ({@link #engineConfig()}).
         */
        final EngineConfig engine;
        final int clients;
        final int sequences;
        final int maxTokens;
        /** The junk bound as given; requests draw {@code [0, max(1, junk))} blocks, as llama.cpp does. */
        final int junk;
        final int ctx;
        final int ubatch;
        final long seed;
        final float temperature;
        final int topK;
        final float topP;
        /** Whether a finished client takes the next request at once ({@code -cb}) or waits for all. */
        final boolean contBatching;
        /** The questions the requests draw from: the built-in list or the {@code -p} / {@code -f} lines. */
        final List<String> prompts;
        /** Where the questions came from: the {@code -f} path, {@code -p}, or the built-in defaults. */
        final String promptSource;

        Plan(Path model, EngineConfig engine, int clients, int sequences, int maxTokens, int junk, int ctx,
                int ubatch, long seed, float temperature, int topK, float topP) {
            this(model, engine, clients, sequences, maxTokens, junk, ctx, ubatch, seed, temperature, topK, topP,
                    true, Arrays.asList(PROMPTS), BUILT_IN_PROMPTS);
        }

        Plan(Path model, EngineConfig engine, int clients, int sequences, int maxTokens, int junk, int ctx,
                int ubatch, long seed, float temperature, int topK, float topP, boolean contBatching,
                List<String> prompts, String promptSource) {
            this.model = model;
            this.engine = engine;
            this.clients = clients;
            this.sequences = sequences;
            this.maxTokens = maxTokens;
            this.junk = junk;
            this.ctx = ctx;
            this.ubatch = ubatch;
            this.seed = seed;
            this.temperature = temperature;
            this.topK = topK;
            this.topP = topP;
            this.contBatching = contBatching;
            this.prompts = List.copyOf(prompts);
            this.promptSource = promptSource;
        }

        /**
         * The configuration the model opens with: the load configuration over a paged memory of one sequence per
         * client sharing {@code -c} rows, stepped {@code -ub} tokens at a time.
         */
        EngineConfig engineConfig() {
            return EngineLoadFlags.pagedConfig(engine, ctx, ubatch, clients);
        }
    }

    /** One simulated client: its sequence slot's current request and its counters. */
    static final class Client {

        final int id;
        BatchSequence handle;
        String input = "";
        String response = "";
        long startNanos;
        int promptTokens;

        Client(int id) {
            this.id = id;
        }
    }

    private ParallelDemo() {
    }

    public static void main(String[] args) {
        run(args, System.out);
    }

    /** Runs the demo, printing the requests, responses and totals to {@code out}; errors exit nonzero. */
    static void run(String[] args, Appendable out) {
        for (String a : args) {
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return;
            }
        }
        try {
            simulate(parse(args), out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags; a first argument that is not a flag is the model path. The load flags
     * ({@link EngineLoadFlags}) start from the {@code -Dneuralfree.*} properties, as LLMDemo's do.
     *
     * @throws UsageException for an unknown flag, a missing value, a value out of range or an invalid load flag
     */
    static Plan parse(String[] args) throws UsageException {
        try {
            return parseFlags(args);
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** {@link #parse} with the load flags' own exception, which {@link #parse} turns into a usage error. */
    private static Plan parseFlags(String[] args) throws UsageException, EngineLoadFlags.FlagException {
        EngineLoadFlags load = EngineLoadFlags.fromSystemProperties();
        Path model = null;
        int clients = DEFAULT_CLIENTS;
        int sequences = DEFAULT_SEQUENCES;
        int maxTokens = DEFAULT_MAX_TOKENS;
        int junk = DEFAULT_JUNK;
        int ctx = 0;
        int ubatch = MemoryConfig.DEFAULT_UBATCH;
        long seed = SamplerConfig.RANDOM_SEED;
        float temperature = DEFAULT_TEMPERATURE;
        int topK = DEFAULT_TOP_K;
        float topP = DEFAULT_TOP_P;
        boolean contBatching = true;
        List<String> prompts = Arrays.asList(PROMPTS);
        String promptSource = BUILT_IN_PROMPTS;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "-cb":
                case "--cont-batching":
                    contBatching = true;
                    break;
                case "-nocb":
                case "--no-cont-batching":
                    contBatching = false;
                    break;
                case "-p":
                case "--prompt":
                    prompts = promptLines(a, value(a, args, i++));
                    promptSource = "-p";
                    break;
                case "-f":
                case "--file":
                    promptSource = value(a, args, i++);
                    prompts = promptLines(a, readPromptFile(a, promptSource));
                    break;
                case "-pps":
                    throw new UsageException("-pps is not an option here: the batch engine's prefix cache shares"
                            + " the system prompt's pages once a request holding it has retired");
                case "-np":
                case "--parallel":
                    clients = intValue(a, value(a, args, i++));
                    break;
                case "-ns":
                case "--sequences":
                    sequences = intValue(a, value(a, args, i++));
                    break;
                case "-n":
                case "--max-tokens":
                    maxTokens = intValue(a, value(a, args, i++));
                    break;
                case "--junk":
                    junk = nonNegative(a, value(a, args, i++));
                    break;
                case "-c":
                case "--ctx":
                    ctx = nonNegative(a, value(a, args, i++));
                    break;
                case "-ub":
                case "--ubatch":
                    ubatch = intValue(a, value(a, args, i++));
                    break;
                case "-s":
                case "--seed":
                    seed = longValue(a, value(a, args, i++));
                    break;
                case "--temp":
                    temperature = floatValue(a, value(a, args, i++), 0f, Float.MAX_VALUE);
                    break;
                case "--top-k":
                    topK = nonNegative(a, value(a, args, i++));
                    break;
                case "--top-p":
                    topP = floatValue(a, value(a, args, i++), 0f, 1f);
                    break;
                case "-m":
                case "--model":
                    model = Paths.get(value(a, args, i++));
                    break;
                default:
                    int last = load.parse(args, i);
                    if (last >= 0) {
                        i = last;
                        break;
                    }
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + model);
                    }
                    model = Paths.get(a);
            }
        }
        if (model == null) {
            throw new UsageException("ParallelDemo needs a model path");
        }
        return new Plan(model, load.build(), clients, sequences, maxTokens, junk, ctx, ubatch, seed, temperature,
                topK, topP, contBatching, prompts, promptSource);
    }

    /** Opens the model on the paged memory of the plan's backend and runs the client simulation over a batch engine. */
    static void simulate(Plan plan, Appendable out) throws IOException {
        EngineConfig cfg;
        try {
            cfg = plan.engineConfig();
        } catch (IllegalArgumentException e) {
            throw new IOException(e.getMessage(), e);
        }
        try (EngineModel model = LLMModelFactory.open(plan.model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(model, cfg)) {
            int systemTokens = model.tokenizer().encode(SYSTEM, false).length;
            out.append("ParallelDemo: simulating parallel requests from clients (backend = ")
                    .append(EngineLoadFlags.backendLabel(model, cfg)).append("):\n");
            out.append(String.format(Locale.ROOT,
                    "ParallelDemo: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n",
                    plan.clients, plan.sequences, plan.contBatching ? 1 : 0, systemTokens));
            out.append("\n");
            process(plan, engine, systemTokens, out);
        } catch (UnsupportedOperationException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    /**
     * The scheduler: submit idle clients' next requests (with {@code -nocb} only once every client is idle), pump
     * one step, report every finished request.
     */
    private static void process(Plan plan, BatchEngine engine, int systemTokens, Appendable out)
            throws IOException {
        Client[] clients = new Client[plan.clients];
        for (int i = 0; i < clients.length; i++) {
            clients[i] = new Client(i);
        }
        Random rng = new Random(PROMPT_RNG_SEED);
        BatchStats before = engine.stats();
        int nextSeq = 0;
        int totalPrompt = 0;
        int totalGen = 0;
        long tMain = System.nanoTime();
        while (nextSeq < plan.sequences || !engine.idle()) {
            boolean admit = plan.contBatching || engine.idle();
            for (Client c : clients) {
                if (admit && c.handle == null && nextSeq < plan.sequences) {
                    start(c, engine, nextSeq++, plan, rng, out);
                }
            }
            engine.pump();
            for (Client c : clients) {
                if (c.handle != null && c.handle.finished()) {
                    int[] done = report(c, plan, out);
                    totalPrompt += done[0];
                    totalGen += done[1];
                }
            }
        }
        totals(out, plan, engine, before, systemTokens, new int[] { totalPrompt, totalGen }, tMain);
    }

    /** Builds the client's next request (system, junk blocks, question) and submits it. */
    private static void start(Client c, BatchEngine engine, int seq, Plan plan, Random rng, Appendable out)
            throws IOException {
        String input = plan.prompts.get(rng.nextInt(plan.prompts.size()));
        int junk = rng.nextInt(Math.max(1, plan.junk));
        String prompt = requestPrompt(input, junk, rng);
        int[] tokens = engine.tokenizer().encode(prompt, false);
        GenerationRequest request = GenerationRequest.builder()
                .maxNewTokens(plan.maxTokens)
                .sampler(sampler(plan, c.id))
                .stopStrings(List.of(REVERSE_PROMPT))
                .build();
        try {
            c.handle = engine.submit(tokens, request, (TokenCallback) null);
        } catch (ContextFullException e) {
            throw new IOException("client " + c.id + "'s prompt holds " + tokens.length
                    + " tokens, more than the context of " + engine.contextLength() + " rows fits", e);
        }
        c.input = input;
        c.response = "";
        c.startNanos = System.nanoTime();
        c.promptTokens = tokens.length;
        out.append(String.format(Locale.ROOT,
                "Client %3d, seq %4d, junk = %4d, prompt = %d, started decoding...\n",
                c.id, seq, junk, tokens.length));
    }

    /**
     * One request's text, as parallel.cpp builds it: the system text, {@code junk} question-and-answer blocks drawn
     * from {@code rng}, then the client's question awaiting the assistant's answer.
     */
    static String requestPrompt(String input, int junk, Random rng) {
        StringBuilder prompt = new StringBuilder(SYSTEM);
        for (int j = 0; j < junk; j++) {
            int r = rng.nextInt(QUESTIONS.length);
            prompt.append(REVERSE_PROMPT).append('\n').append(QUESTIONS[r]).append("\nAssistant:\n ")
                    .append(ANSWERS[r]).append('\n');
        }
        return prompt.append(REVERSE_PROMPT).append('\n').append(input).append("\nAssistant:\n").toString();
    }

    /** The request's sampler chain: the demo's flags with client {@code id}'s seed. */
    private static SamplerConfig sampler(Plan plan, int client) {
        return SamplerConfig.defaults().toBuilder()
                .temperature(plan.temperature)
                .topK(plan.topK)
                .topP(plan.topP)
                .seed(clientSeed(plan.seed, client))
                .build();
    }

    /**
     * Client {@code client}'s sampler seed under llama.cpp's signed-seed rule: a non-negative seed is shared, a
     * negative one counts down per client in the 32-bit seed space, whose all-ones value is the random seed.
     */
    static long clientSeed(long seed, int client) {
        if (seed >= 0) {
            return seed;
        }
        long wrapped = (seed - client) & SEED_MASK;
        return wrapped == SEED_MASK ? SamplerConfig.RANDOM_SEED : wrapped;
    }

    /**
     * Reports one finished request: the client line, its input and its trimmed response, as llama.cpp prints
     * them. Returns the request's prompt and generated token counts; a failed sequence throws.
     */
    private static int[] report(Client c, Plan plan, Appendable out) throws IOException {
        BatchSequence handle = c.handle;
        c.handle = null;
        if (handle.state() == BatchSequence.State.FAILED) {
            throw new IOException("client " + c.id + "'s request failed: " + handle.error());
        }
        GenerationResult result = handle.result();
        int generated = result == null ? 0 : result.tokenCount();
        c.response = result == null ? "" : result.text().trim();
        double seconds = (System.nanoTime() - c.startNanos) / 1e9;
        double speed = (c.promptTokens + generated) / seconds;
        out.append(String.format(Locale.ROOT,
                "Client %3d, seq %4d/%4d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s\n\n",
                c.id, handle.submitId(), plan.sequences, c.promptTokens, generated, seconds, speed));
        out.append("Input:    ").append(c.input.trim()).append('\n');
        out.append("Response: ").append(c.response).append("\n\n");
        return new int[] { c.promptTokens, generated };
    }

    /**
     * The totals: the run line, the prompt source and model, the throughput of the prompt, generated and combined
     * tokens, the prefix-cache reuse and the preemptions (the engine's memory-full events, where llama.cpp counts
     * cache misses).
     */
    private static void totals(Appendable out, Plan plan, BatchEngine engine, BatchStats before, int systemTokens,
            int[] promptAndGen, long tMain) throws IOException {
        double seconds = (System.nanoTime() - tMain) / 1e9;
        BatchStats after = engine.stats();
        long cached = after.promptTokensCached() - before.promptTokensCached();
        long preemptions = after.preemptions() - before.preemptions();
        int totalPrompt = promptAndGen[0];
        int totalGen = promptAndGen[1];
        out.append('\n');
        out.append(String.format(Locale.ROOT,
                "ParallelDemo: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n",
                plan.clients, plan.sequences, plan.contBatching ? 1 : 0, systemTokens));
        out.append("External prompt file: ").append(plan.promptSource).append('\n');
        out.append("Model and path used:  ").append(String.valueOf(plan.model)).append("\n\n");
        out.append(String.format(Locale.ROOT, "Total prompt tokens: %6d, speed: %5.2f t/s\n",
                totalPrompt, totalPrompt / seconds));
        out.append(String.format(Locale.ROOT, "Total gen tokens:    %6d, speed: %5.2f t/s\n",
                totalGen, totalGen / seconds));
        out.append(String.format(Locale.ROOT, "Total speed (AVG):   %6s  speed: %5.2f t/s\n",
                "", (totalPrompt + totalGen) / seconds));
        out.append(String.format(Locale.ROOT, "Prefix-cache tokens: %6d\n", cached));
        out.append(String.format(Locale.ROOT, "Preemptions:         %6d\n", preemptions));
    }

    // ------------------------------------------------------------------ argument helpers

    /** The non-empty lines of a {@code -p} / {@code -f} value: one question each. */
    private static List<String> promptLines(String flag, String text) throws UsageException {
        List<String> lines = new ArrayList<>();
        for (String line : text.split("\r?\n")) {
            if (!line.trim().isEmpty()) {
                lines.add(line);
            }
        }
        if (lines.isEmpty()) {
            throw new UsageException(flag + " holds no prompt line");
        }
        return lines;
    }

    /** The UTF-8 text of a {@code -f} file, or the flag's own error. */
    private static String readPromptFile(String flag, String path) throws UsageException {
        try {
            return new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
        } catch (IOException | InvalidPathException e) {
            throw new UsageException(flag + " cannot read '" + path + "': " + e.getMessage());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int intValue(String flag, String v) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < 1) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a positive integer, not '" + v + "'");
        }
    }

    private static int nonNegative(String flag, String v) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
    }

    private static long longValue(String flag, String v) throws UsageException {
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + v + "'");
        }
    }

    private static float floatValue(String flag, String v, float min, float max) throws UsageException {
        try {
            float f = Float.parseFloat(v.trim());
            if (!(f >= min && f <= max)) {
                throw new NumberFormatException();
            }
            return f;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number in [" + min + ", " + max + "], not '" + v + "'");
        }
    }
}
