package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GgufEncoder;
import com.eonmux.neuralfree.gguf.GgufTensorSink;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.HiddenTapSlot;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.Observers;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;
import java.util.SplittableRandom;

/**
 * {@code nftool cvector}: the cvector-generator (PARITY.md WP LR-4, llama.cpp {@code tools/cvector-generator}): the
 * control-vector training data of the engine.
 *
 * <p>LR-4 owns the data side TL-4 drives; the serving loop that opens a model, tokenizes the prompt entries with their
 * pad token, prefill rows and drives a tap per side (llama.cpp {@code get_hidden_layers} over each pair,
 * cvector-generator.cpp:344-352, over the padded pair of {@code tokenized_prompt}) is TL-4's. This class holds
 * everything around that loop:
 * <ul>
 *   <li>{@link #entries}: the positive and negative prompt files, one entry per line, empty lines skipped and the
 *       line's escapes processed, the counts equal and at least one pair (llama.cpp {@code ctrlvec_load_prompt_file}
 *       and {@code prepare_entries}, cvector-generator.cpp:407-438).</li>
 *   <li>{@link Differences}: the per-layer collector — the positive and the negative hidden rows of one pair through a
 *       tap, their difference, the every-value-under-ep rows dropped (llama.cpp {@code calc_diff} and
 *       {@code filter_nonzero_rows}, cvector-generator.cpp:78-121), the kept rows concatenated per layer (llama.cpp
 *       {@code train_context}'s {@code v_diff_tmp}). Fed on the CPU through {@link HiddenDiffTap}, a {@link HiddenTap}
 *       over the forward context's hidden tap — tap point {@code L} is the residual stream entering layer {@code L},
 *       so the rows that arrive under {@code L} leave layer {@code L - 1}, exactly llama.cpp's layer-input taps; fed on
 *       the GPU through {@link EventTap}, an {@link ExecutionObserver} of {@code Stage.LAYER} ends, which sits in the
 *       backend's declared {@link Observers} (§5.5.1 O-1 step 3, so no prompt row of any request is skipped by a reuse
 *       path) and reads the layer result slot between enqueues, the profiler's licence.</li>
 *   <li>{@link #train}: one direction per collected layer — the top principal component of the layer's diff rows by
 *       power iteration ({@code Pca.component}: iterate {@code v = (X^T X) v}, normalized by the vector's Euclidean
 *       norm, stop under the tolerance, llama.cpp pca.hpp) or the rows' mean ({@code Mean.of}, llama.cpp mean.hpp: the
 *       mean, then normalized to unit length).</li>
 *   <li>{@link #export}: the GGUF through GW-1's write path ({@link GgufTensorSink} and {@link GgufEncoder}, the exact
 *       bytes {@code GgufWriter} routes its exports through): {@code general.architecture = controlvector}, {@code
 *       controlvector.model_hint}, {@code controlvector.layer_count} and {@code general.alignment}, and the directions
 *       as F32 tensors {@code direction.<i + 1>} in collected order — the llama.cpp {@code export_gguf} key set and
 *       names (cvector-generator.cpp:367-383), which {@code common_control_vector_load_one} reads back for decoder
 *       layer {@code il} (llama.cpp common/common.cpp:1974-2058), so the {@code ControlVector} extras add it after that
 *       layer's block.</li>
 * </ul>
 * One deviation from llama.cpp: power iteration starts from a fixed seed ({@link #PCA_SEED}), so a run is reproducible
 * where llama.cpp seeds with {@code time(0)}; the PCA sign ambiguity llama.cpp documents unresolved stays unresolved
 * (pca.hpp: "The output vector is randomly inverted"). The serving loop ({@link #generate}) opens the model on the
 * CPU engine, whose hidden tap feeds {@link HiddenDiffTap}; the GPU event-tap route of {@link EventTap} stays open
 * (GAPS.md).</p>
 */
public final class CvectorGenerator {

    /** Default GGUF output path of llama.cpp ({@code params.out_file}). */
    public static final String DEFAULT_OUT = "control_vector.gguf";

    /** The power-iteration seed, instead of llama.cpp's {@code time(0)} (the class javadoc records why). */
    public static final long PCA_SEED = 20260101L;

    /** The zero-row eps of llama.cpp {@code filter_nonzero_rows}. */
    public static final float ZERO_ROW_EPS = 1e-6f;

    private static final String USAGE =
            "nftool cvector <model> [options]\n"
            + "options (llama.cpp llama-cvector-generator names):\n"
            + "  --positive-file <fname>          positive prompts file, one prompt per line\n"
            + "  --negative-file <fname>          negative prompts file, one prompt per line\n"
            + "  --out-file <fname>               output GGUF, default " + DEFAULT_OUT + "\n"
            + "  --pca-batch <n>                  PCA iteration batch, default 20\n"
            + "  --pca-iter <n>                   PCA iteration budget, default 1000\n"
            + "  --method <pca|mean>              dimensionality reduction method, default pca\n"
            + "  --model-hint <arch>              controlvector.model_hint of the export, the base model's arch\n"
            + "  --overwrite                      replace the output\n"
            + "  --help, -h                       this text\n"
            + "The model runs on the CPU engine; the collection reads the CPU hidden tap.\n";

    private CvectorGenerator() {
    }

    // ------------------------------------------------------------------ the tool entry

    /**
     * Runs the tool with the launcher's arguments after {@code cvector}. It validates the flags and both prompt
     * files, runs the prefill loop over one loaded engine ({@link #generate}, the llama.cpp
     * {@code get_hidden_layers} driver) and writes the control-vector GGUF.
     */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o == null) {
                return;
            }
            if (o.modelPath().isEmpty()) {
                throw new UsageException("cvector needs a model path");
            }
            List<Entry> pairs = entries(Paths.get(o.positiveFile()), Paths.get(o.negativeFile()));
            Path written = generate(o.modelPath(), pairs, o);
            System.out.println("nftool cvector: wrote " + written + " from " + pairs.size() + " pairs, "
                    + "method " + o.method().name().toLowerCase(Locale.ROOT));
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
     * The prefill loop (llama.cpp {@code get_hidden_layers} over each padded pair, cvector-generator.cpp:344-352):
     * opens {@code modelPath} on the CPU engine — the collection reads the CPU hidden tap through
     * {@link HiddenDiffTap}, the route a CPU forward context fires — tokenizes every pair's sides with the
     * vocabulary's BOS policy, pads both to one length with the pad token llama.cpp derives (the last token of
     * tokenizing a single space, cvector-generator.cpp:289-294), and per pair runs one {@link Differences#beginPair},
     * the positive prefill from position 0, the side switch, the negative prefill and {@link Differences#endPair},
     * the cache reset between sides. Tracked layers are all decoder layers except the last (llama.cpp discards the
     * last layer's {@code l_out}), the tap rows of one side arrive per forward call, so equal padded lengths give
     * equal row counts per layer. The directions then reduce through {@link #train} and the export follows.
     *
     * @return the written GGUF ({@link #export})
     * @throws IOException for an unreadable model or prompt file, a pair longer than the model's context fits, or
     *                     an existing output without {@code --overwrite}
     * @throws UsageException when the model path is missing, unreadable, or the model cannot serve the loop
     */
    static Path generate(String modelPath, List<Entry> pairs, Options o) throws IOException, UsageException {
        if (modelPath == null || modelPath.isEmpty()) {
            throw new UsageException("cvector needs a model path");
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
        try (EngineModel model = LLMModelFactory.open(modelPath, cfg)) {
            ForwardBackend backend = model.forwardBackend();
            if (backend.memory() == null || !(model instanceof GGUFModel)) {
                throw new UsageException("the collection reads the CPU hidden tap; " + model.architecture()
                        + " opened as " + model.getClass().getSimpleName() + " has none");
            }
            ModelSpec spec = model.spec();
            int nLayer = spec.nLayer();
            if (nLayer < 2) {
                throw new UsageException(model.architecture() + " holds " + nLayer
                        + " layers; llama.cpp discards the last layer's taps, so at least two are needed");
            }
            int width = spec.embedding().nEmbd();
            int[] layers = new int[nLayer - 1];
            for (int il = 0; il < layers.length; il++) {
                layers[il] = il;
            }
            Differences diffs = new Differences(layers, width);
            HiddenTapSlot.Lease lease = HiddenTapSlot.acquire(backend, "nftool cvector");
            backend.setHiddenTap(new HiddenDiffTap(diffs));
            try {
                GGUFTokenizer tok = model.tokenizer();
                int[] pad = padToken(tok);
                for (int p = 0; p < pairs.size(); p++) {
                    Entry pair = pairs.get(p);
                    int[] pos = tok.encode(pair.positive());
                    int[] neg = tok.encode(pair.negative());
                    int len = Math.max(pos.length, neg.length);
                    pos = padTo(pos, len, pad);
                    neg = padTo(neg, len, pad);
                    diffs.beginPair();
                    try {
                        backend.reset();
                        backend.prefill(pos, 0);
                        diffs.setSide(false);
                        backend.reset();
                        backend.prefill(neg, 0);
                    } catch (ContextFullException e) {
                        throw new IOException("pair " + (p + 1) + " pads to " + len
                                + " tokens, more than the model's context fits: " + e.getMessage(), e);
                    }
                    diffs.endPair();
                }
            } finally {
                backend.setHiddenTap(HiddenTap.NONE);
                lease.close();
                backend.reset();
            }
            String hint = o.modelHint().isEmpty() ? model.architecture() : o.modelHint();
            List<float[]> directions = train(diffs, o);
            return export(directions, width, hint, Paths.get(o.outFile()), o.overwrite());
        }
    }

    /** The pad token of llama.cpp's padding: the last token of tokenizing one space, no BOS (or the pad id). */
    private static int[] padToken(GGUFTokenizer tok) {
        int[] space = tok.encode(" ", false, false);
        return space.length > 0 ? new int[]{space[space.length - 1]} : new int[]{tok.padId()};
    }

    /** {@code tokens} padded with {@code pad} to {@code len} (llama.cpp {@code padding_seq}). */
    private static int[] padTo(int[] tokens, int len, int[] pad) {
        if (tokens.length >= len) {
            return tokens;
        }
        int[] out = new int[len];
        System.arraycopy(tokens, 0, out, 0, tokens.length);
        for (int i = tokens.length; i < len; i++) {
            out[i] = pad[0];
        }
        return out;
    }

    /**
     * Parses the flags into {@link Options} (llama.cpp's cvector-generator option names).
     *
     * @return the options, or null after printing the usage for {@code --help}
     * @throws UsageException naming the offending flag
     */
    public static Options parse(String[] args) throws UsageException {
        Objects.requireNonNull(args, "args");
        Options o = Options.defaults();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return null;
            }
            switch (a) {
                case "--positive-file":
                    o = o.toBuilder().positiveFile(value(a, args, ++i)).build();
                    break;
                case "--negative-file":
                    o = o.toBuilder().negativeFile(value(a, args, ++i)).build();
                    break;
                case "--out-file":
                    o = o.toBuilder().outFile(value(a, args, ++i)).build();
                    break;
                case "--pca-batch":
                    o = o.toBuilder().pcaBatch(intValue(a, value(a, args, ++i))).build();
                    break;
                case "--pca-iter":
                    o = o.toBuilder().pcaIterations(intValue(a, value(a, args, ++i))).build();
                    break;
                case "--method": {
                    String m = value(a, args, ++i).trim();
                    if (m.equals("pca")) {
                        o = o.toBuilder().method(Method.PCA).build();
                    } else if (m.equals("mean")) {
                        o = o.toBuilder().method(Method.MEAN).build();
                    } else {
                        throw new UsageException("--method is invalid; use pca or mean");
                    }
                    break;
                }
                case "--model-hint":
                    o = o.toBuilder().modelHint(value(a, args, ++i)).build();
                    break;
                case "--overwrite":
                    o = o.toBuilder().overwrite(true).build();
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    // The positional model path: the serving loop resolves it through ModelSources.open, exactly as
                    // the demos do; the data-side options name no model.
                    o = o.toBuilder().modelPath(a).build();
                    break;
            }
        }
        if (o.positiveFile().isEmpty() || o.negativeFile().isEmpty()) {
            throw new UsageException("--positive-file and --negative-file are required (llama.cpp defaults to the"
                    + " tool directory's prompt lists, which NeuralFree does not ship)");
        }
        return o;
    }

    private static String value(String flag, String[] args, int i) throws UsageException {
        if (i >= args.length || args[i] == null || args[i].startsWith("-")) {
            throw new UsageException(flag + " needs a value");
        }
        return args[i];
    }

    private static int intValue(String flag, String s) throws UsageException {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, got '" + s + "'");
        }
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    // ------------------------------------------------------------------ the options

    /** How one layer's diff rows reduce to a direction (llama.cpp {@code cvector_dimre_method}). */
    public enum Method {
        /** The top principal component by power iteration over {@code X^T X} (pca.hpp). */
        PCA,
        /** The mean of the rows, normalized to unit length (mean.hpp). */
        MEAN
    }

    /**
     * The cvector options (immutable, with a {@link Builder}, {@link #defaults()} and {@link #toBuilder()}, the
     * {@code neuralfree.cvector.*} properties of {@link #fromProperties}): the prompt files, the output, the reduction
     * method and llama.cpp's PCA parameters.
     */
    public static final class Options {

        public static final String PROP_POSITIVE = "neuralfree.cvector.positive-file";
        public static final String PROP_NEGATIVE = "neuralfree.cvector.negative-file";
        public static final String PROP_OUT = "neuralfree.cvector.out-file";
        public static final String PROP_METHOD = "neuralfree.cvector.method";
        public static final String PROP_PCA_BATCH = "neuralfree.cvector.pca.batch";
        public static final String PROP_PCA_ITERATIONS = "neuralfree.cvector.pca.iter";
        public static final String PROP_PCA_TOLERANCE = "neuralfree.cvector.pca.tolerance";
        public static final String PROP_HINT = "neuralfree.cvector.model-hint";
        public static final String PROP_OVERWRITE = "neuralfree.cvector.overwrite";

        private final String positiveFile;
        private final String negativeFile;
        private final String outFile;
        private final String modelPath;
        private final String modelHint;
        private final Method method;
        private final int pcaBatch;
        private final int pcaIterations;
        private final float pcaTolerance;
        private final boolean overwrite;

        private Options(String positiveFile, String negativeFile, String outFile, String modelPath,
                        String modelHint, Method method, int pcaBatch, int pcaIterations, float pcaTolerance,
                        boolean overwrite) {
            if (pcaBatch < 1) {
                throw new IllegalArgumentException(PROP_PCA_BATCH + " must be at least 1, got " + pcaBatch);
            }
            if (pcaIterations < 1) {
                throw new IllegalArgumentException(PROP_PCA_ITERATIONS + " must be at least 1, got " + pcaIterations);
            }
            if (!Float.isFinite(pcaTolerance) || pcaTolerance <= 0.0f) {
                throw new IllegalArgumentException(PROP_PCA_TOLERANCE + " must be a positive finite value, got "
                        + pcaTolerance);
            }
            this.positiveFile = positiveFile;
            this.negativeFile = negativeFile;
            this.outFile = outFile;
            this.modelPath = modelPath;
            this.modelHint = modelHint;
            this.method = method;
            this.pcaBatch = pcaBatch;
            this.pcaIterations = pcaIterations;
            this.pcaTolerance = pcaTolerance;
            this.overwrite = overwrite;
        }

        /** The llama.cpp defaults: PCA, batch 20, 1000 iterations, tolerance {@code 1e-7}. */
        public static Options defaults() {
            return new Options("", "", DEFAULT_OUT, "", "", Method.PCA, 20, 1000, 1e-7f, false);
        }

        public Builder toBuilder() {
            return new Builder().positiveFile(positiveFile).negativeFile(negativeFile).outFile(outFile)
                    .modelPath(modelPath).modelHint(modelHint).method(method).pcaBatch(pcaBatch)
                    .pcaIterations(pcaIterations).pcaTolerance(pcaTolerance).overwrite(overwrite);
        }

        /** The {@code neuralfree.cvector.*} properties; a blank value keeps the default. */
        public static Options fromProperties(Properties p) {
            Builder b = defaults().toBuilder();
            String v = p.getProperty(PROP_POSITIVE);
            if (v != null && !v.isEmpty()) {
                b.positiveFile(v);
            }
            v = p.getProperty(PROP_NEGATIVE);
            if (v != null && !v.isEmpty()) {
                b.negativeFile(v);
            }
            v = p.getProperty(PROP_OUT);
            if (v != null && !v.isEmpty()) {
                b.outFile(v);
            }
            v = p.getProperty(PROP_METHOD);
            if (v != null && !v.isEmpty()) {
                if (v.equals("pca")) {
                    b.method(Method.PCA);
                } else if (v.equals("mean")) {
                    b.method(Method.MEAN);
                } else {
                    throw new IllegalArgumentException(PROP_METHOD + " is invalid; use pca or mean");
                }
            }
            b.pcaBatch(parseInt(p, PROP_PCA_BATCH, 20));
            b.pcaIterations(parseInt(p, PROP_PCA_ITERATIONS, 1000));
            v = p.getProperty(PROP_PCA_TOLERANCE);
            if (v != null && !v.isEmpty()) {
                b.pcaTolerance(parseFloat(PROP_PCA_TOLERANCE, v, 1e-7f));
            }
            v = p.getProperty(PROP_HINT);
            if (v != null && !v.isEmpty()) {
                b.modelHint(v);
            }
            v = p.getProperty(PROP_OVERWRITE);
            if (v != null && !v.isEmpty()) {
                b.overwrite(Boolean.parseBoolean(v.trim()));
            }
            return b.build();
        }

        private static int parseInt(Properties p, String key, int dflt) {
            String v = p.getProperty(key);
            if (v == null || v.isEmpty()) {
                return dflt;
            }
            try {
                return Integer.parseInt(v.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(key + " needs an integer, got '" + v + "'");
            }
        }

        private static float parseFloat(String key, String v, float dflt) {
            try {
                return Float.parseFloat(v.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(key + " needs a float, got '" + v + "'");
            }
        }

        public String positiveFile() { return positiveFile; }
        public String negativeFile() { return negativeFile; }
        public String outFile() { return outFile; }
        public String modelPath() { return modelPath; }
        public String modelHint() { return modelHint; }
        public Method method() { return method; }
        public int pcaBatch() { return pcaBatch; }
        public int pcaIterations() { return pcaIterations; }
        public float pcaTolerance() { return pcaTolerance; }
        public boolean overwrite() { return overwrite; }

        @Override
        public String toString() {
            return "Options{" + (modelPath.isEmpty() ? "" : modelPath + ": ") + positiveFile + " - " + negativeFile
                    + " -> " + outFile + ", " + method + ", batch " + pcaBatch + ", iter " + pcaIterations
                    + ", tol " + pcaTolerance + "}";
        }
    }

    /** Builder of {@link Options}. */
    public static final class Builder {
        private String positiveFile = "";
        private String negativeFile = "";
        private String outFile = DEFAULT_OUT;
        private String modelPath = "";
        private String modelHint = "";
        private Method method = Method.PCA;
        private int pcaBatch = 20;
        private int pcaIterations = 1000;
        private float pcaTolerance = 1e-7f;
        private boolean overwrite;

        public Builder positiveFile(String v) { this.positiveFile = Objects.requireNonNull(v, "positiveFile"); return this; }
        public Builder negativeFile(String v) { this.negativeFile = Objects.requireNonNull(v, "negativeFile"); return this; }
        public Builder outFile(String v) { this.outFile = Objects.requireNonNull(v, "outFile"); return this; }
        public Builder modelPath(String v) { this.modelPath = Objects.requireNonNull(v, "modelPath"); return this; }
        public Builder modelHint(String v) { this.modelHint = Objects.requireNonNull(v, "modelHint"); return this; }
        public Builder method(Method v) { this.method = Objects.requireNonNull(v, "method"); return this; }
        public Builder pcaBatch(int v) { this.pcaBatch = v; return this; }
        public Builder pcaIterations(int v) { this.pcaIterations = v; return this; }
        public Builder pcaTolerance(float v) { this.pcaTolerance = v; return this; }
        public Builder overwrite(boolean v) { this.overwrite = v; return this; }

        public Options build() {
            return new Options(positiveFile, negativeFile, outFile, modelPath, modelHint, method, pcaBatch,
                    pcaIterations, pcaTolerance, overwrite);
        }
    }

    // ------------------------------------------------------------------ the entries

    /** One positive/negative prompt pair. Immutable. */
    public static final class Entry {

        private final String positive;
        private final String negative;

        Entry(String positive, String negative) {
            this.positive = positive;
            this.negative = negative;
        }

        public String positive() {
            return positive;
        }

        public String negative() {
            return negative;
        }

        @Override
        public String toString() {
            return "+" + positive + " / -" + negative;
        }
    }

    /**
     * The prompt pairs of the two prompt files, llama.cpp's {@code prepare_entries}: one entry per non-empty line (the
     * line's escapes processed, {@code string_process_escapes}), the counts equal and at least one pair.
     */
    public static List<Entry> entries(Path positiveFile, Path negativeFile) throws IOException {
        List<String> positive = promptLines(positiveFile.toString());
        List<String> negative = promptLines(negativeFile.toString());
        if (positive.size() != negative.size()) {
            throw new IOException("number of positive and negative prompts must be equal, got " + positive.size()
                    + " and " + negative.size());
        }
        if (positive.isEmpty()) {
            throw new IOException("must provide at least one prompt pair");
        }
        List<Entry> out = new ArrayList<>(positive.size());
        for (int i = 0; i < positive.size(); i++) {
            out.add(new Entry(positive.get(i), negative.get(i)));
        }
        return out;
    }

    private static List<String> promptLines(String file) throws IOException {
        String text = Files.readString(Paths.get(file), StandardCharsets.UTF_8);
        List<String> out = new ArrayList<>();
        for (String line : text.split("\n", -1)) {
            String s = processEscapes(line);
            if (!s.isEmpty()) {
                out.add(s);
            }
        }
        return out;
    }

    /** llama.cpp {@code string_process_escapes}: the {@code \n \r \t \' \" \\} runs, as the demo paths do. */
    static String processEscapes(String line) {
        StringBuilder out = new StringBuilder(line.length());
        int i = 0;
        while (i < line.length()) {
            char c = line.charAt(i);
            if (c != '\\' || i + 1 >= line.length()) {
                out.append(c);
                i++;
                continue;
            }
            char next = line.charAt(i + 1);
            switch (next) {
                case 'n': out.append('\n'); break;
                case 'r': out.append('\r'); break;
                case 't': out.append('\t'); break;
                case '\'': out.append('\''); break;
                case '"': out.append('"'); break;
                case '\\': out.append('\\'); break;
                default: out.append(c).append(next); break;
            }
            i += 2;
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ the per-layer collector

    /**
     * The collected per-layer differences of the prompt pairs (llama.cpp {@code callback_data} and
     * {@code train_context}). One pair is {@link #beginPair()}, a positive side then a negative side of tap chunks,
     * and {@link #endPair()}; {@link #train(Differences, Options)} reduces the collected rows afterwards. Mutable,
     * owned by the thread that drives the taps (the forward engine thread, §3.3).
     */
    public static final class Differences {

        /** The tracked graph layers (0-based decoder layers), ascending. */
        private final int[] layers;
        /** A graph layer's tracked position, or -1 for an untracked one. */
        private final int[] trackOf;
        /** The row width both sides' rows hold: the model's residual width. */
        private final int width;
        /** Per tracked position, the current pair's side chunks. */
        private final List<List<float[]>> posChunks = new ArrayList<>();
        private final List<List<float[]>> negChunks = new ArrayList<>();
        private final int[] posRows;
        private final int[] negRows;
        /** Per tracked position, the kept diff rows of every pair so far. */
        private final List<List<float[]>> diffChunks = new ArrayList<>();
        private final int[] diffRows;
        private boolean positive;
        private boolean inPair;
        private int completedPairs;

        /**
         * A collector over tracked graph layers {@code layers} (ascending; llama.cpp tracks the layer outputs whose
         * {@code l_out} node fires — one per decoder layer except the last; TL-4 passes that list) of rows
         * {@code width} floats wide.
         */
        public Differences(int[] layers, int width) {
            if (layers.length == 0) {
                throw new IllegalArgumentException("the cvector collector needs at least one tracked layer");
            }
            if (width < 1) {
                throw new IllegalArgumentException("the cvector collector needs a positive width, got " + width);
            }
            for (int i = 0; i < layers.length; i++) {
                if (layers[i] < 0) {
                    throw new IllegalArgumentException("tracked layer " + layers[i] + " is negative");
                }
                if (i > 0 && layers[i] <= layers[i - 1]) {
                    throw new IllegalArgumentException("the tracked layers ascend, " + layers[i] + " after "
                            + layers[i - 1]);
                }
            }
            this.layers = layers.clone();
            this.width = width;
            int max = layers[layers.length - 1];
            this.trackOf = new int[max + 1];
            Arrays.fill(trackOf, -1);
            for (int i = 0; i < layers.length; i++) {
                trackOf[layers[i]] = i;
                posChunks.add(new ArrayList<>());
                negChunks.add(new ArrayList<>());
                diffChunks.add(new ArrayList<>());
            }
            this.posRows = new int[layers.length];
            this.negRows = new int[layers.length];
            this.diffRows = new int[layers.length];
        }

        /** The tracked graph layers (a copy). */
        public int[] layers() {
            return layers.clone();
        }

        /** The row width of both sides' rows. */
        public int width() {
            return width;
        }

        /** How many diff rows tracked position {@code t} holds after its pairs. */
        public int rows(int t) {
            checkT(t);
            return diffRows[t];
        }

        /** True when graph layer {@code layer} is tracked. */
        public boolean tracks(int layer) {
            return layer >= 0 && layer < trackOf.length && trackOf[layer] >= 0;
        }

        /** The side the taps currently fill ({@link #setSide}). */
        public boolean side() {
            return positive;
        }

        /** Starts a pair: the following tap chunks collect the positive side first. */
        public void beginPair() {
            for (int i = 0; i < layers.length; i++) {
                posChunks.get(i).clear();
                negChunks.get(i).clear();
                posRows[i] = 0;
                negRows[i] = 0;
            }
            positive = true;
            inPair = true;
        }

        /** Starts the other side of the current pair; the taps take {@link #side()} as it stands. */
        public void setSide(boolean positive) {
            if (!inPair) {
                throw new IllegalStateException("the cvector collector has no pair running; call beginPair first");
            }
            if (positive == this.positive) {
                return;
            }
            this.positive = positive;
        }

        /**
         * One forward-call chunk of the current pair's side rows at graph layer {@code layer}: {@code rows} is
         * row-major {@code [nRows][rowWidth]} and is copied.
         *
         * @throws IllegalStateException    in the wrong side order, an untracked layer, or outside a running pair
         * @throws IllegalArgumentException for a wrong width or a rows array that does not carry the rows
         */
        public void addSide(int layer, float[] rows, int nRows, int rowWidth) {
            if (!inPair) {
                throw new IllegalStateException("the cvector collector has no pair running; call beginPair first");
            }
            if (nRows < 0) {
                throw new IllegalArgumentException("the cvector tap at layer " + layer + " passes " + nRows + " rows");
            }
            if (rowWidth != width) {
                throw new IllegalArgumentException("the cvector rows at layer " + layer + " are " + rowWidth
                        + " wide, the collector holds " + width);
            }
            if (rows.length < (long) nRows * rowWidth) {
                throw new IllegalArgumentException("the cvector rows at layer " + layer + " hold " + rows.length
                        + " floats, " + nRows + " rows need " + ((long) nRows * rowWidth));
            }
            int t = layer >= 0 && layer < trackOf.length ? trackOf[layer] : -1;
            if (t < 0) {
                throw new IllegalStateException("the cvector collector does not track layer " + layer + " (tracks "
                        + Arrays.toString(layers) + ")");
            }
            float[] chunk = Arrays.copyOfRange(rows, 0, (int) Math.min(rows.length, (long) nRows * rowWidth));
            if (positive) {
                posChunks.get(t).add(chunk);
                posRows[t] += nRows;
            } else {
                negChunks.get(t).add(chunk);
                negRows[t] += nRows;
            }
        }

        /**
         * Completes the pair: the positive minus the negative rows of every tracked layer; a row whose every value
         * stays within {@link #ZERO_ROW_EPS} leaves out (llama.cpp {@code filter_nonzero_rows}), the rest is kept in
         * row order.
         *
         * @throws IllegalStateException for a pair whose sides collected different row counts — llama.cpp pads both
         *                               tokenized sides to one sequence length (its {@code tokenized_prompt}), so the
         *                               taps see equal row counts — or an unstarted pair
         */
        public void endPair() {
            if (!inPair) {
                throw new IllegalStateException("the cvector collector has no pair running; call beginPair first");
            }
            for (int t = 0; t < layers.length; t++) {
                if (posRows[t] == 0 || negRows[t] == 0) {
                    throw new IllegalStateException("the cvector pair collected no rows at layer " + layers[t]
                            + " (positive " + posRows[t] + ", negative " + negRows[t] + ")");
                }
                if (posRows[t] != negRows[t]) {
                    throw new IllegalStateException("the cvector pair's sides differ at layer " + layers[t] + ": "
                            + posRows[t] + " positive rows against " + negRows[t] + " negative rows; the driver pads"
                            + " both sides to one sequence length, as llama.cpp's tokenized_prompt does");
                }
                RowCursor p = new RowCursor(posChunks.get(t), width);
                RowCursor n = new RowCursor(negChunks.get(t), width);
                for (int r = 0; r < posRows[t]; r++) {
                    float[] pos = p.next();
                    float[] neg = n.next();
                    boolean zero = true;
                    for (int j = 0; j < width; j++) {
                        if (Math.abs(pos[j] - neg[j]) > ZERO_ROW_EPS) {
                            zero = false;
                            break;
                        }
                    }
                    if (zero) {
                        continue;
                    }
                    float[] diff = new float[width];
                    for (int j = 0; j < width; j++) {
                        diff[j] = pos[j] - neg[j];
                    }
                    diffChunks.get(t).add(diff);
                    diffRows[t]++;
                }
                posChunks.get(t).clear();
                negChunks.get(t).clear();
                posRows[t] = 0;
                negRows[t] = 0;
            }
            inPair = false;
            completedPairs++;
        }

        /** Number of the completed pairs. */
        public int pairs() {
            return completedPairs;
        }

        /**
         * The collected diff rows of tracked position {@code t}: a fresh flat {@code [rows][width]} array.
         *
         * @throws IllegalArgumentException for a tracking position outside {@code [0, layers().length)}
         * @throws IllegalStateException    when the rows would not fit one float array
         */
        public float[] rowBlock(int t) {
            checkT(t);
            long n = (long) diffRows[t] * width;
            if (n > Integer.MAX_VALUE - 8L) {
                throw new IllegalStateException("the cvector rows of " + layers[t] + " hold " + n
                        + " floats, too many for one float array");
            }
            float[] out = new float[(int) n];
            int off = 0;
            for (float[] chunk : diffChunks.get(t)) {
                System.arraycopy(chunk, 0, out, off, chunk.length);
                off += chunk.length;
            }
            return out;
        }

        private void checkT(int t) {
            if (t < 0 || t >= layers.length) {
                throw new IllegalArgumentException("tracked layer position " + t + " is outside [0, "
                        + layers.length + ")");
            }
        }
    }

    /** Row-by-row walk over one side's chunks of the current pair. */
    private static final class RowCursor {
        private final int width;
        private List<float[]> chunks;
        private int at;

        RowCursor(List<float[]> chunks, int width) {
            this.chunks = chunks;
            this.width = width;
        }

        /** The next rows in chunk order (one row of {@code width} floats, a fresh copy). */
        float[] next() {
            float[] chunk = chunks.get(at);
            int chunkRows = chunk.length / width;
            float[] row = new float[width];
            System.arraycopy(chunk, 0, row, 0, width);
            if (chunkRows == 1) {
                chunks.set(at, null);
                at++;
            } else {
                float[] rest = Arrays.copyOfRange(chunk, width, chunk.length);
                chunks.set(at, rest);
            }
            return row;
        }
    }

    // ------------------------------------------------------------------ the taps

    /**
     * The CPU driver of {@link HiddenTap} (llama.cpp's stage callback over the layer-input taps): the tap point
     * {@code L} is the residual stream entering layer {@code L}, so the rows that arrive under {@code L} leave layer
     * {@code L - 1}; the tap hands them to the {@link Differences} side currently collecting. A hyper-connection
     * stream whose tap width is not the residual width the collector was built for is refused by
     * {@link Differences#addSide}.
     */
    public static final class HiddenDiffTap implements HiddenTap {

        private final Differences diffs;

        public HiddenDiffTap(Differences diffs) {
            this.diffs = Objects.requireNonNull(diffs, "diffs");
        }

        @Override
        public boolean wants(int layer) {
            return layer >= 1 && diffs.tracks(layer - 1);
        }

        @Override
        public void onHidden(int layer, float[] rows, int nRows, int width) {
            diffs.addSide(layer - 1, rows, nRows, width);
        }

        @Override
        public String toString() {
            return "HiddenDiffTap{tracks " + Arrays.toString(diffs.layers()) + "}";
        }
    }

    /**
     * The GPU driver of {@link ExecutionObserver} (llama.cpp's ggml callback over the layer nodes): every tracked
     * layer's {@code Stage.LAYER} end enqueues one read of the residual stream slot into a fresh array and records it
     * under the side currently collecting; {@link #drain()} waits for the queue and commits the reads in issue order.
     * The driver pattern per pair is one {@link Differences#beginPair()}, the positive side's prefill loop, {@code
     * drain()}, {@link Differences#setSide(false)}, the negative side's prefill loop, {@code drain()} again, one
     * {@link Differences#endPair()}; a prefill loop stops at an ubatch boundary makes several reads per layer, which
     * arrive in layer order. The reads must not overlap the following side: the driver drains before switching.
     */
    public static final class EventTap implements ExecutionObserver {

        private final Differences diffs;
        private final ComputeQueue queue;
        private final List<GpuEvent> events = new ArrayList<>();
        private final List<Pending> pending = new ArrayList<>();

        private EventTap(Differences diffs, ComputeQueue queue) {
            this.diffs = diffs;
            this.queue = queue;
        }

        /** The tap over {@code diffs} committing through {@code queue}. */
        public static EventTap of(Differences diffs, ComputeQueue queue) {
            return new EventTap(Objects.requireNonNull(diffs, "diffs"), Objects.requireNonNull(queue, "queue"));
        }

        @Override
        public boolean enabled() {
            return true;
        }

        @Override
        public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
            if (stage != Stage.LAYER || !diffs.tracks(layer) || nRows <= 0 || result == null) {
                return;
            }
            float[] dst = new float[Math.multiplyExact(nRows, width)];
            GpuEvent e = queue.enqueueReadBuffer(result.buffer(), (long) result.offset() * Float.BYTES,
                    (long) dst.length * Float.BYTES, dst);
            if (e != null) {
                events.add(e);
            }
            pending.add(new Pending(layer, diffs.side(), dst, width));
        }

        /**
         * Waits for the queue and commits every pending read to the collector under the side it was issued for.
         * The events close after the finish, before the commit.
         */
        public void drain() {
            queue.finish();
            for (GpuEvent e : events) {
                e.close();
            }
            events.clear();
            try {
                for (Pending p : pending) {
                    diffs.addSide(p.layer, p.data, p.data.length / p.width, p.width);
                }
            } finally {
                pending.clear();
            }
        }

        @Override
        public String toString() {
            return "EventTap{tracks " + Arrays.toString(diffs.layers()) + ", pending " + pending.size() + "}";
        }

        /** One issued read, with the side it was issued under. */
        private static final class Pending {
            final int layer;
            final boolean side;
            final float[] data;
            final int width;

            Pending(int layer, boolean side, float[] data, int width) {
                this.layer = layer;
                this.side = side;
                this.data = data;
                this.width = width;
            }
        }
    }

    // ------------------------------------------------------------------ the reductions

    /**
     * One direction per tracked position of {@code diffs} ({@link #train}): the top principal component by power
     * iteration for {@link Method#PCA}, or the mean for {@link Method#MEAN}. The directions go out in tracked order;
     * the export names them {@code direction.<i + 1>}, llama.cpp's {@code export_gguf}.
     *
     * @throws IOException holding the tracked layer when a tracked position collected no diff rows
     */
    public static List<float[]> train(Differences diffs, Options o) throws IOException {
        Objects.requireNonNull(diffs, "diffs");
        Objects.requireNonNull(o, "options");
        List<float[]> out = new ArrayList<>();
        int[] layers = diffs.layers();
        for (int t = 0; t < layers.length; t++) {
            int n = diffs.rows(t);
            if (n == 0) {
                throw new IOException("the cvector tracked layer " + layers[t] + " (position " + t
                        + ") collected no diff rows; the driver must run at least one pair");
            }
            float[] rows = diffs.rowBlock(t);
            float[] direction = o.method() == Method.PCA
                    ? Pca.component(rows, n, diffs.width(), o.pcaBatch(), o.pcaIterations(), o.pcaTolerance())
                    : Mean.of(rows, n, diffs.width());
            out.add(direction);
        }
        return out;
    }

    /** The power iteration of the top principal component of {@code rows} (llama.cpp {@code pca.hpp}): the square
     * {@code X^T X} of {@code nRows} rows of {@code width}, a pseudo-random normalized start ({@link #PCA_SEED},
     * llama.cpp seeds with {@code time(0)}), then {@code v = square * v || normalize(v)} until the distance to the
     * previous vector is under {@code tolerance} or {@code iterations} are spent. The returned vector has unit length
     * and llama.cpp's unresolved sign ambiguity. Float math, as llama.cpp's F32 graph runs it; {@code batchSize} only
     * bounds the device graph batching llama.cpp runs the same math in and changes no result.
     */
    public static final class Pca {

        private Pca() {
        }

        /**
         * The top principal component of {@code rows} ({@code [nRows][nEmbd]}): a unit {@code nEmbd} vector, or the
         * nearest to it the iteration reaches at the budget.
         */
        public static float[] component(float[] rows, int nRows, int nEmbd, int batchSize, int iterations,
                                        float tolerance) {
            if (nRows < 1) {
                throw new IllegalArgumentException("PCA needs at least one sample row, got " + nRows);
            }
            if (nEmbd < 1) {
                throw new IllegalArgumentException("PCA needs a positive width, got " + nEmbd);
            }
            if (iterations < 1) {
                throw new IllegalArgumentException("PCA needs at least one iteration, got " + iterations);
            }
            if (!(tolerance > 0.0f) || !Float.isFinite(tolerance)) {
                throw new IllegalArgumentException("PCA tolerance must be positive and finite, got " + tolerance);
            }
            if (rows.length != (long) nRows * nEmbd) {
                throw new IllegalArgumentException("PCA rows hold " + rows.length + " values, expected "
                        + (long) nRows * nEmbd);
            }
            float[] square = new float[nEmbd * nEmbd];
            boolean degenerate = true;
            for (int s = 0; s < nRows; s++) {
                int base = s * nEmbd;
                for (int a = 0; a < nEmbd; a++) {
                    float x = rows[base + a];
                    if (x == 0.0f) {
                        continue;
                    }
                    degenerate = false;
                    int rowOff = a * nEmbd;
                    for (int b = 0; b < nEmbd; b++) {
                        square[rowOff + b] += x * rows[base + b];
                    }
                }
            }
            if (degenerate) {
                throw new IllegalArgumentException("PCA of an all-zero row block has no principal component");
            }
            SplittableRandom rng = new SplittableRandom(PCA_SEED);
            float vSumSquares = 0.0f;
            float[] v = new float[nEmbd];
            for (int i = 0; i < nEmbd; i++) {
                v[i] = (float) rng.nextDouble();
                vSumSquares += v[i] * v[i];
            }
            float vNorm = (float) Math.sqrt(vSumSquares);
            for (int i = 0; i < nEmbd; i++) {
                v[i] /= vNorm;
            }
            float[] w = new float[nEmbd];
            for (int iter = 0; iter < iterations; iter++) {
                float squares = 0.0f;
                for (int a = 0; a < nEmbd; a++) {
                    float sum = 0.0f;
                    int rowOff = a * nEmbd;
                    for (int b = 0; b < nEmbd; b++) {
                        sum += square[rowOff + b] * v[b];
                    }
                    w[a] = sum;
                    squares += sum * sum;
                }
                float norm = (float) Math.sqrt(squares);
                if (norm == 0.0f) {
                    break;
                }
                float distance = 0.0f;
                for (int i = 0; i < nEmbd; i++) {
                    float next = w[i] / norm;
                    float d = next - v[i];
                    distance += d * d;
                    w[i] = next;
                    v[i] = next;
                }
                if ((float) Math.sqrt(distance) < tolerance) {
                    break;
                }
            }
            return w;
        }
    }

    /** The mean direction of the rows, normalized to unit length (llama.cpp {@code mean.hpp}). */
    public static final class Mean {

        private Mean() {
        }

        /** The mean of {@code rows} ({@code [nRows][nEmbd]}), normalized; the zero mean stays zero. */
        public static float[] of(float[] rows, int nRows, int nEmbd) {
            if (nRows < 1) {
                throw new IllegalArgumentException("a mean needs at least one sample row, got " + nRows);
            }
            if (nEmbd < 1) {
                throw new IllegalArgumentException("a mean needs a positive width, got " + nEmbd);
            }
            if (rows.length != (long) nRows * nEmbd) {
                throw new IllegalArgumentException("mean rows hold " + rows.length + " values, expected "
                        + (long) nRows * nEmbd);
            }
            float[] mean = new float[nEmbd];
            for (int s = 0; s < nRows; s++) {
                int base = s * nEmbd;
                for (int j = 0; j < nEmbd; j++) {
                    mean[j] += rows[base + j];
                }
            }
            double normSquares = 0.0;
            for (int j = 0; j < nEmbd; j++) {
                mean[j] /= nRows;
                normSquares += (double) mean[j] * mean[j];
            }
            double norm = Math.sqrt(normSquares);
            if (norm > 0.0) {
                float f = (float) norm;
                for (int j = 0; j < nEmbd; j++) {
                    mean[j] /= f;
                }
            }
            return mean;
        }
    }

    // ------------------------------------------------------------------ the export

    /**
     * Writes {@code directions} (the {@link #train} result, per tracked layer) to the GGUF at {@code out}, GW-1's write
     * path ({@link GgufTensorSink}/{@link GgufEncoder}): the metadata keys of llama.cpp {@code export_gguf},
     * {@code general.architecture = controlvector}, {@code controlvector.model_hint}, {@code
     * controlvector.layer_count} and {@code general.alignment}, then each direction as the F32 one-D tensor {@code
     * direction.<i + 1>} of {@code nEmbd}.
     *
     * @return the written path ({@code out}, renamed from its {@code .tmp} twin on success)
     * @throws IllegalArgumentException for an empty direction list, a blank model hint elsewhere stated and per
     *                                  direction width {@code nEmbd}, or a non-finite value
     * @throws IOException              for the write failures {@link GgufTensorSink} names, or an existing {@code out}
     *                                  with {@code overwrite} false
     */
    public static Path export(List<float[]> directions, int nEmbd, String modelHint, Path out, boolean overwrite)
            throws IOException {
        Objects.requireNonNull(directions, "directions");
        if (directions.isEmpty()) {
            throw new IllegalArgumentException("a control-vector export needs at least one direction");
        }
        if (nEmbd < 1) {
            throw new IllegalArgumentException("a control-vector export needs a positive width, got " + nEmbd);
        }
        if (modelHint == null || modelHint.trim().isEmpty()) {
            throw new IllegalArgumentException("a control-vector export names the base model's arch as"
                    + " " + GgufKeys.CONTROLVECTOR_MODEL_HINT);
        }
        for (int i = 0; i < directions.size(); i++) {
            float[] d = directions.get(i);
            if (d.length != nEmbd) {
                throw new IllegalArgumentException("direction " + (i + 1) + " holds " + d.length
                        + " values, the " + nEmbd + " of n_embd");
            }
        }
        int alignment = GGUFFileParser.DEFAULT_ALIGNMENT;
        GgufTensorSink sink = GgufTensorSink.singleFile(out, overwrite);
        try {
            GgufEncoder enc = sink.begin(0);
            int kvCount = 4;
            enc.writeHeader(sink.channel(0), directions.size(), kvCount);
            enc.writeKv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, null, ControlVector.ARCH);
            enc.writeKv(GgufKeys.CONTROLVECTOR_MODEL_HINT, GGUFValueType.STRING, null, modelHint.trim());
            enc.writeKv(GgufKeys.CONTROLVECTOR_LAYER_COUNT, GGUFValueType.INT32, null, directions.size());
            enc.writeKv(GgufKeys.GENERAL_ALIGNMENT, GGUFValueType.UINT32, null, alignment);
            long offset = 0;
            String prefix = "cvector export of '" + out + "'";
            for (int i = 0; i < directions.size(); i++) {
                GGUFTensorInfo info = new GGUFTensorInfo(ControlVector.DIRECTION_PREFIX + (i + 1),
                        new long[]{nEmbd}, GGMLType.F32, offset);
                GGUFTensorInfo.validate(prefix, info);
                enc.writeTensorInfo(info, offset);
                offset = alignUp(offset + info.sizeInBytes(), alignment);
            }
            enc.align(alignment);
            ByteBuffer f32 = ByteBuffer.allocate(nEmbd * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < directions.size(); i++) {
                f32.clear();
                f32.asFloatBuffer().put(directions.get(i));
                f32.rewind();
                byte[] bytes = new byte[f32.remaining()];
                f32.get(bytes);
                enc.writeData(bytes);
                enc.align(alignment);
            }
            enc.flush();
            sink.end(0);
            return out;
        } finally {
            sink.close();
        }
    }

    private static long alignUp(long v, int alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }
}
