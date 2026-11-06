package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EmbeddingNorm;
import com.eonmux.neuralfree.gguf.engine.EmbeddingSimilarity;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * {@code nftool retrieval}: llama.cpp's retrieval example (examples/retrieval/retrieval.cpp). Every
 * {@code --context-file} chunks at {@code --chunk-separator} boundaries into pieces of more than {@code
 * --chunk-size} bytes (the {@code chunk_file} port, its leftover accounting included), every chunk tokenizes with
 * special tokens and a trailing EOS, embeds pooled and normalizes Euclidean (llama.cpp hardcodes
 * {@code common_embd_normalize(..., 2)}), and the query loop reads one query per line from stdin and prints the
 * {@code --top-k} most similar chunks by cosine similarity.
 *
 * <p>The model must pool: pooling {@code NONE} refuses, as llama.cpp refuses it. Chunks embed in llama.cpp's
 * batches ({@link EmbeddingBatchFlags}, retrieval.cpp:229-249): a batch of at most {@code --batch-size} tokens over
 * at most {@code --parallel} sequences (default 1) runs as one multi-sequence forward over a paged memory, which
 * produces the same per-chunk embeddings. The whole file chunks, where llama.cpp's 1024-byte read
 * loop drops a file's final partial block, and the end of stdin ends the query loop.</p>
 */
final class RetrievalCommand {

    /** llama.cpp's default chunk size (common/common.h:726). */
    static final int DEFAULT_CHUNK_SIZE = 64;

    /** llama.cpp's default chunk separator. */
    static final String DEFAULT_CHUNK_SEPARATOR = "\n";

    /** llama.cpp's sampling top-k, which the retrieval loop reads (common/common.h:235). */
    static final int DEFAULT_TOP_K = 40;

    private static final String USAGE =
            "nftool retrieval <model> [options]\n"
            + "options (llama.cpp retrieval names):\n"
            + "  --context-file <fname>[,<fname>...]  files to embed and search (comma-separated, repeatable);\n"
            + "                                   at least one is required\n"
            + "  --chunk-size <n>                 minimum bytes per chunk, default " + DEFAULT_CHUNK_SIZE + "\n"
            + "  --chunk-separator <string>       separator between chunks, default a newline\n"
            + "  --top-k <n>                      similar chunks to print per query, default " + DEFAULT_TOP_K + "\n"
            + "  --parallel, -np <n>              sequences per batch (llama.cpp -np; default 1 = one chunk per\n"
            + "                                   forward)\n"
            + EmbeddingBatchFlags.USAGE
            + "  --verbose-prompt                 print every chunk's tokens before embedding\n"
            + "  --help, -h                       this text\n"
            + "Embeds the chunks of every context file, then reads queries from stdin, one per line until the\n"
            + "input ends, and prints the top-k most similar chunks by cosine similarity. The model must pool;\n"
            + "pooling NONE refuses.\n";

    private RetrievalCommand() {
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** One chunk of one context file: the reference's {@code chunk} struct (immutable). */
    static final class Chunk {

        final String filename;
        /** The chunk's byte offset in its file. */
        final long filepos;
        final String textdata;
        /** The chunk's Euclidean-normalized embedding, or null before the embed pass. */
        private final float[] embedding;

        Chunk(String filename, long filepos, String textdata) {
            this(filename, filepos, textdata, null);
        }

        private Chunk(String filename, long filepos, String textdata, float[] embedding) {
            this.filename = Objects.requireNonNull(filename, "filename");
            this.filepos = filepos;
            this.textdata = Objects.requireNonNull(textdata, "textdata");
            this.embedding = embedding;
        }

        /** This chunk with {@code values} as its embedding (copied). */
        Chunk withEmbedding(float[] values) {
            return new Chunk(filename, filepos, textdata, values.clone());
        }

        /**
         * A copy of the chunk's embedding.
         *
         * @throws IllegalStateException before the embed pass
         */
        float[] embedding() {
            if (embedding == null) {
                throw new IllegalStateException("chunk at " + filename + ":" + filepos + " is not embedded");
            }
            return embedding.clone();
        }
    }

    /** The retrieval options of one run (immutable). */
    static final class Options {

        final String model;
        final List<Path> contextFiles;
        final int chunkSize;
        final String chunkSeparator;
        final int topK;
        final boolean verbosePrompt;
        final EmbeddingBatchFlags.Settings batch;

        Options(String model, List<Path> contextFiles, int chunkSize, String chunkSeparator, int topK,
                boolean verbosePrompt, EmbeddingBatchFlags.Settings batch) {
            this.model = model;
            this.contextFiles = Collections.unmodifiableList(new ArrayList<>(contextFiles));
            this.chunkSize = chunkSize;
            this.chunkSeparator = chunkSeparator;
            this.topK = topK;
            this.verbosePrompt = verbosePrompt;
            this.batch = batch;
        }

        String model() { return model; }
        List<Path> contextFiles() { return contextFiles; }
        int chunkSize() { return chunkSize; }
        String chunkSeparator() { return chunkSeparator; }
        int topK() { return topK; }
        boolean verbosePrompt() { return verbosePrompt; }
        /** The pooling, batching and load flags. */
        EmbeddingBatchFlags.Settings batch() { return batch; }
        /** The configuration of the flags, before {@link EmbeddingBatchFlags.Settings#forModel} fits it. */
        EngineConfig load() { return batch.load(); }
        /** The pooling override, or null for the GGUF's pooling. */
        PoolingKind pooling() { return batch.pooling(); }
        int batchSize() { return batch.batchSize(); }
        /** {@code --parallel}, or 0 when absent. */
        int parallel() { return batch.parallel(); }
    }

    /** Runs the tool with the launcher's arguments after {@code retrieval}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                execute(o, new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
                        System.out);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Chunks the context files, opens the model, embeds the chunks and runs the query loop over {@code in}, printing
     * the results to {@code out}; the progress lines go to stderr. The spec resolves before the load, so a pooling
     * override the model cannot serve refuses before any weight moves and a non-causal model takes the batch size
     * as its ubatch ({@link EmbeddingBatchFlags.Settings#forModel}, retrieval.cpp:127). The memory runs
     * {@code --parallel} sequences, one when the flag is absent, as llama.cpp's retrieval tool does.
     *
     * @throws IOException for an unreadable model or context file, a chunk longer than the model's context fits,
     *                     or a model without pooling
     */
    static void execute(Options o, BufferedReader in, PrintStream out) throws IOException {
        System.err.println("processing files:");
        for (Path f : o.contextFiles()) {
            System.err.println(f);
        }
        List<Chunk> chunks = chunkAll(o.contextFiles(), o.chunkSize(), o.chunkSeparator());
        System.err.println("Number of chunks: " + chunks.size());
        ModelSpec spec = LLMModelFactory.inspect(o.model(), o.load());
        int batchSize = o.batch().batchSize(spec);
        EngineConfig cfg = o.batch().forModel(spec, Math.max(1, o.parallel()));
        try (EngineModel model = LLMModelFactory.open(o.model(), cfg)) {
            requirePooling(model);
            StringBuilder dump = o.verbosePrompt() ? new StringBuilder() : null;
            List<Chunk> embedded = embedChunks(model, chunks, dump, batchSize);
            if (dump != null) {
                System.err.print(dump);
            }
            queryLoop(model, embedded, o.topK(), in, out);
        }
    }

    /**
     * Refuses a model without a pooled embedding output: no embedding output at all, or pooling {@code NONE}.
     *
     * @throws UnsupportedOperationException naming the model's architecture and pooling
     */
    static void requirePooling(EngineModel model) {
        EncodeGate.checkEncode(model.spec());
        if (model.poolingType() == LLMModel.PoolingType.NONE) {
            throw new UnsupportedOperationException("nftool retrieval needs a pooling model; "
                    + model.architecture() + " pools by NONE (llama.cpp: pooling type NONE not supported)");
        }
    }

    /**
     * The chunks of every context file in order (llama.cpp's {@code chunk_file} loop over the files).
     *
     * @throws IOException for an unreadable file
     */
    static List<Chunk> chunkAll(List<Path> files, int chunkSize, String chunkSeparator) throws IOException {
        List<Chunk> chunks = new ArrayList<>();
        for (Path f : files) {
            chunks.addAll(chunkFile(f, chunkSize, chunkSeparator));
        }
        return chunks;
    }

    /**
     * Chunks {@code file} at {@code separator} boundaries, pushing a chunk once its text passes {@code chunkSize}
     * bytes (examples/retrieval/retrieval.cpp:32-76; sizes and file positions count UTF-8 bytes, as the C
     * {@code std::string} does). The text after the last separator is dropped, and the unpushed tail chunk joins the
     * last pushed one, both as llama.cpp leaves them.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException for a non-positive size or an empty separator
     */
    static List<Chunk> chunkFile(Path file, int chunkSize, String separator) throws IOException {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunk-size " + chunkSize + " must be positive");
        }
        if (separator.isEmpty()) {
            throw new IllegalArgumentException("the chunk separator is empty");
        }
        byte[] data = Files.readAllBytes(file);
        byte[] sep = separator.getBytes(StandardCharsets.UTF_8);
        List<int[]> bounds = new ArrayList<>(); // {start, end} byte ranges of the pushed chunks
        int start = 0;
        int consumed = 0;
        int pos = indexOf(data, sep, 0);
        while (pos >= 0) {
            consumed = pos + sep.length;
            if (consumed - start > chunkSize) {
                bounds.add(new int[]{start, consumed});
                start = consumed;
            }
            pos = indexOf(data, sep, consumed);
        }
        if (consumed > start) { // the leftover joins the last chunk, or is the file's only chunk
            if (bounds.isEmpty()) {
                bounds.add(new int[]{start, consumed});
            } else {
                bounds.get(bounds.size() - 1)[1] = consumed;
            }
        }
        List<Chunk> chunks = new ArrayList<>(bounds.size());
        for (int[] b : bounds) {
            chunks.add(new Chunk(file.toString(), b[0],
                    new String(data, b[0], b[1] - b[0], StandardCharsets.UTF_8)));
        }
        return chunks;
    }

    /** The first index at or after {@code from} where {@code pattern} occurs in {@code data}, or -1. */
    private static int indexOf(byte[] data, byte[] pattern, int from) {
        int last = data.length - pattern.length;
        for (int i = from; i <= last; i++) {
            int j = 0;
            while (j < pattern.length && data[i + j] == pattern[j]) {
                j++;
            }
            if (j == pattern.length) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Embeds every chunk pooled and normalized Euclidean (llama.cpp's {@code batch_process} with
     * {@code embd_norm = 2}), tokenizing with special tokens (no special-token parsing) and a trailing EOS. The
     * chunks run in llama.cpp's batches ({@link EmbeddingBatchFlags#encode}, retrieval.cpp:229-249).
     *
     * @param dump the {@code --verbose-prompt} token dump to append to, or null
     * @return the chunks with their embeddings, in order
     * @throws IOException when a chunk holds no tokens or more tokens than the model's training context or the
     *                     batch size fits
     */
    static List<Chunk> embedChunks(EngineModel model, List<Chunk> chunks, StringBuilder dump, int batchSize)
            throws IOException {
        EncodeGate.checkEncode(model.spec());
        GGUFTokenizer tok = model.tokenizer();
        PromptDump printer = dump != null ? new PromptDump(tok) : null;
        int cap = model.contextLength();
        int eos = tok.eosId();
        List<int[]> tokenized = new ArrayList<>(chunks.size());
        for (int c = 0; c < chunks.size(); c++) {
            Chunk chunk = chunks.get(c);
            int[] tokens = tok.encode(chunk.textdata, true, false);
            if (eos >= 0 && (tokens.length == 0 || tokens[tokens.length - 1] != eos)) {
                tokens = Arrays.copyOf(tokens, tokens.length + 1);
                tokens[tokens.length - 1] = eos;
            }
            if (tokens.length == 0) {
                throw new IOException("chunk " + c + " of " + chunk.filename + " tokenizes to no tokens");
            }
            if (tokens.length > cap) {
                throw new IOException("chunk " + c + " of " + chunk.filename + " tokenizes to " + tokens.length
                        + " tokens, more than the model's training context (" + cap + ") fits");
            }
            if (tokens.length > batchSize) {
                throw new IOException("number of tokens in chunk " + c + " of " + chunk.filename + " ("
                        + tokens.length + ") exceeds batch size (" + batchSize
                        + "), increase batch size (--batch-size) and re-run");
            }
            if (printer != null) {
                printer.append(dump, c, chunk.textdata, tokens);
            }
            tokenized.add(tokens);
        }
        List<float[]> rows = EmbeddingBatchFlags.encode(model, tokenized, batchSize);
        List<Chunk> out = new ArrayList<>(chunks.size());
        for (int c = 0; c < chunks.size(); c++) {
            out.add(chunks.get(c).withEmbedding(EmbeddingNorm.normalize(rows.get(c), EmbeddingNorm.EUCLIDEAN)));
        }
        return out;
    }

    private static float[] normalizedEmbedding(EngineModel model, int[] tokens) throws IOException {
        return EmbeddingNorm.normalize(model.encode(tokens), EmbeddingNorm.EUCLIDEAN);
    }

    /**
     * The query loop (retrieval.cpp:263-299): one query per line until the input ends, each tokenized with special
     * tokens (llama.cpp's {@code common_tokenize(ctx, query, true)}, no special-token parsing and no EOS), embedded
     * and ranked against the chunks by cosine similarity; the top-k print with their file, byte position,
     * similarity and text. An empty line embeds as its special tokens alone, as llama.cpp's loop does.
     *
     * @throws IOException when the input cannot be read, a query holds no tokens, or a query outgrows the model's
     *                     training context
     */
    static void queryLoop(EngineModel model, List<Chunk> chunks, int topK, BufferedReader in, PrintStream out)
            throws IOException {
        GGUFTokenizer tok = model.tokenizer();
        int cap = model.contextLength();
        while (true) {
            out.print("Enter query: ");
            out.flush();
            String query = in.readLine();
            if (query == null) {
                return;
            }
            int[] tokens = tok.encode(query, true, false);
            if (tokens.length == 0) {
                throw new IOException("the query '" + query + "' tokenizes to no tokens");
            }
            if (tokens.length > cap) {
                throw new IOException("the query tokenizes to " + tokens.length
                        + " tokens, more than the model's training context (" + cap + ") fits");
            }
            float[] embd = normalizedEmbedding(model, tokens);
            float[] sims = new float[chunks.size()];
            for (int i = 0; i < sims.length; i++) {
                sims[i] = EmbeddingSimilarity.cosine(chunks.get(i).embedding(), embd);
            }
            printTop(chunks, sims, topIndices(sims, topK), topK, out);
        }
    }

    /**
     * The indices of the {@code k} largest similarities, largest first; equal similarities keep the smaller index
     * first, where llama.cpp's sort leaves the tie order of equal values open.
     */
    static int[] topIndices(float[] sims, int k) {
        Integer[] indices = new Integer[sims.length];
        for (int i = 0; i < sims.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> {
            int bySim = Float.compare(sims[b], sims[a]);
            return bySim != 0 ? bySim : Integer.compare(a, b);
        });
        int n = Math.min(k, sims.length);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = indices[i];
        }
        return out;
    }

    /** One query's top-k block (retrieval.cpp:290-297). */
    private static void printTop(List<Chunk> chunks, float[] sims, int[] order, int topK, PrintStream out) {
        StringBuilder block = new StringBuilder();
        block.append("Top ").append(topK).append(" similar chunks:\n");
        for (int idx : order) {
            Chunk chunk = chunks.get(idx);
            block.append("filename: ").append(chunk.filename).append('\n');
            block.append("filepos: ").append(chunk.filepos).append('\n');
            block.append(String.format(Locale.ROOT, "similarity: %f\n", sims[idx]));
            block.append("textdata:\n").append(chunk.textdata).append('\n');
            block.append("--------------------\n");
        }
        out.print(block);
        out.flush();
    }

    /**
     * Parses the flags (llama.cpp's retrieval option names). The pooling, batching and load flags parse through
     * {@link EmbeddingBatchFlags}.
     *
     * @return the options, or null after printing the usage for {@code --help}
     * @throws UsageException naming the offending flag
     */
    static Options parse(String[] args) throws UsageException {
        String model = null;
        List<Path> files = new ArrayList<>();
        int chunkSize = DEFAULT_CHUNK_SIZE;
        String separator = DEFAULT_CHUNK_SEPARATOR;
        int topK = DEFAULT_TOP_K;
        boolean verbosePrompt = false;
        EmbeddingBatchFlags batch = batchFlags();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    System.out.println(USAGE);
                    return null;
                case "--context-file":
                    files.addAll(contextFiles(value(a, args, i++)));
                    break;
                case "--chunk-size":
                    chunkSize = intValue(a, value(a, args, i++));
                    if (chunkSize <= 0) {
                        throw new UsageException("--chunk-size " + chunkSize + " must be positive");
                    }
                    break;
                case "--chunk-separator":
                    separator = value(a, args, i++);
                    if (separator.isEmpty()) {
                        throw new UsageException("--chunk-separator needs a non-empty string");
                    }
                    break;
                case "--top-k":
                    topK = intValue(a, value(a, args, i++));
                    if (topK < 1) {
                        throw new UsageException("--top-k " + topK + " must be at least 1");
                    }
                    break;
                case "--verbose-prompt":
                    verbosePrompt = true;
                    break;
                default: {
                    int last = batchFlag(batch, args, i);
                    if (last < 0) {
                        model = positional(a, model);
                    } else {
                        i = last;
                    }
                }
            }
        }
        if (model == null) {
            throw new UsageException("retrieval needs a model path");
        }
        if (files.isEmpty()) {
            throw new UsageException("retrieval needs at least one --context-file");
        }
        return new Options(model, files, chunkSize, separator, topK, verbosePrompt, batchSettings(batch));
    }

    private static EmbeddingBatchFlags batchFlags() throws UsageException {
        try {
            return EmbeddingBatchFlags.fromSystemProperties();
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** One {@link EmbeddingBatchFlags} flag: the index of its last argument, or -1 for any other token. */
    private static int batchFlag(EmbeddingBatchFlags batch, String[] args, int i) throws UsageException {
        try {
            return batch.parse(args, i);
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    private static EmbeddingBatchFlags.Settings batchSettings(EmbeddingBatchFlags batch) throws UsageException {
        try {
            return batch.build();
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /**
     * The files of one {@code --context-file} value: its comma-separated fields, each a readable file.
     *
     * @throws UsageException naming the field that is empty or not a readable file
     */
    private static List<Path> contextFiles(String value) throws UsageException {
        List<Path> files = new ArrayList<>();
        for (String item : parseCsvRow(value)) {
            Path f = Paths.get(item);
            if (item.isEmpty() || !Files.isRegularFile(f) || !Files.isReadable(f)) {
                throw new UsageException("--context-file '" + item + "' is not a readable file");
            }
            files.add(f);
        }
        return files;
    }

    /**
     * The fields of one comma-separated value (llama.cpp's {@code parse_csv_row}, common/arg.cpp): commas split,
     * a field that starts with a double quote is quoted until the closing quote (commas inside stay, {@code ""} is a
     * literal quote), and a quote inside an unquoted field is literal.
     */
    static List<String> parseCsvRow(String input) {
        List<String> fields = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < input.length(); i++) {
            char ch = input.charAt(i);
            if (ch == '"') {
                if (!inQuotes) {
                    if (field.length() > 0) {
                        field.append('"');
                    } else {
                        inQuotes = true;
                    }
                } else if (i + 1 < input.length() && input.charAt(i + 1) == '"') {
                    field.append('"');
                    i++;
                } else {
                    inQuotes = false;
                }
            } else if (ch == ',' && !inQuotes) {
                fields.add(field.toString());
                field.setLength(0);
            } else {
                field.append(ch);
            }
        }
        fields.add(field.toString());
        return fields;
    }

    private static String positional(String arg, String model) throws UsageException {
        if (arg.startsWith("-")) {
            throw new UsageException("unknown option " + arg + " (run with --help for the options)");
        }
        if (model != null) {
            throw new UsageException("too many file arguments; '" + arg + "' follows " + model);
        }
        return arg;
    }

    private static String value(String flag, String[] args, int index) throws UsageException {
        // The value consumes the next argv token even when it starts with '-' (llama.cpp consumes it), so a
        // --chunk-separator of "-x-" parses.
        if (index + 1 >= args.length || args[index + 1] == null) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int intValue(String flag, String s) throws UsageException {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, got '" + s + "'");
        }
    }
}
