package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

/**
 * {@code nftool perplexity}: the quality gate of llama.cpp's perplexity tool (tools/perplexity/perplexity.cpp)
 * over one open model: perplexity over a datafile in context-sized chunks with the second half of every chunk
 * scored ({@code -f}), the strided variant ({@code --ppl-stride}, perplexity_v2), the KL divergence against a
 * base model's stored log-probabilities ({@code --kl-divergence --kl-divergence-base}), the base file itself
 * ({@code --save-all-logits}, llama.cpp's {@code _logits_} format through {@link LogitsFile}) and the
 * multiple-choice benchmarks ({@code --hellaswag}, {@code --winogrande}, {@code --multiple-choice}).
 *
 * <p>The chunking follows llama.cpp: every chunk of {@code nCtx} tokens runs from position 0 with the
 * vocabulary's BOS at its first token, the scored positions are the second half (or the trailing
 * {@code ppl-stride} positions of the strided variant), and the printed per-chunk value is the running
 * perplexity over everything scored so far.</p>
 */
final class PerplexityCommand {

    /** llama.cpp perplexity's default context ({@code params.n_ctx = 512}). */
    static final int DEFAULT_CTX_SIZE = 512;

    /** llama.cpp {@code params.hellaswag_tasks}. */
    static final int DEFAULT_HELASWAG_TASKS = 400;

    private static final String USAGE =
            "nftool perplexity <model> [options]\n"
            + "options (llama.cpp perplexity names):\n"
            + "  -f, --file <fname>               the data file; a later -f replaces an earlier one\n"
            + "  -p, --prompt <text>              evaluate this text instead of a file\n"
            + "  -c, --ctx-size <n>               chunk context, default " + DEFAULT_CTX_SIZE + "\n"
            + "  -b, --batch-size <n>             forward ubatch of the engine\n"
            + "  --chunks <n>                     evaluate at most n chunks (-1, the default, means all)\n"
            + "  --ppl-stride <n>                 score the trailing n positions per chunk instead of half\n"
            + "  --ppl-output-type <0|1>          0 (default): [chunk]ppl, per chunk; 1: one line per chunk\n"
            + "  --save-all-logits <fname>        store the base log-probabilities of the run (llama.cpp format)\n"
            + "  --kl-divergence                  compare against the base file instead of scoring perplexity\n"
            + "  --kl-divergence-base <fname>     the base file to compare against (same name as --save-all-logits)\n"
            + "  --hellaswag                      HellaSwag score over the datafile's 6-line tasks\n"
            + "  --hellaswag-tasks <n>            tasks to use (default " + DEFAULT_HELASWAG_TASKS + ")\n"
            + "  --winogrande                     WinoGrande score over the datafile's CSV tasks\n"
            + "  --winogrande-tasks <n>           tasks to use (default 0 = all)\n"
            + "  --multiple-choice                multiple-choice score over the datafile's binary tasks\n"
            + "  --multiple-choice-tasks <n>      tasks to use (default 0 = all)\n"
            + "  --help, -h                       this text\n"
            + "One mode runs per invocation; the chunking and the printed numbers match llama.cpp's perplexity\n"
            + "tool, and the log-probability file it writes is the file its --kl-divergence-base reads.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line; {@link #parse} fills and validates it. */
    static final class Options {
        Path model;
        String promptText = "";
        byte[] promptBytes = new byte[0];
        int ctxSize = DEFAULT_CTX_SIZE;
        int batchSize = -1;
        int chunks = -1;
        int pplStride;
        int pplOutputType;
        boolean hellaswag;
        int hellaswagTasks = DEFAULT_HELASWAG_TASKS;
        boolean winogrande;
        int winograndeTasks;
        boolean multipleChoice;
        int multipleChoiceTasks;
        boolean klDivergence;
        Path logitsPath;
    }

    private PerplexityCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code perplexity}. */
    static void run(String[] args) {
        try {
            dispatch(args, EngineConfig.fromSystemProperties());
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
     * Parses the arguments, opens the model on {@code cfg} with the command line's context and batch and runs
     * the one selected mode. Package-private so tests drive it with an explicit engine configuration.
     */
    static void dispatch(String[] args, EngineConfig cfg) throws IOException, UsageException {
        Options o = parse(args);
        if (o == null) {
            return;
        }
        try (GGUFModel m = open(o.model, config(o, cfg))) {
            requireTokenLoop(m);
            if (o.hellaswag) {
                HellaSwag.score(m, o.hellaswagTasks, o.promptText, System.out, System.err);
            } else if (o.winogrande) {
                WinoGrande.score(m, o.winograndeTasks, o.promptText, System.out, System.err);
            } else if (o.multipleChoice) {
                MultipleChoice.score(m, o.multipleChoiceTasks, o.promptBytes, System.out, System.err);
            } else if (o.klDivergence) {
                klDivergence(m, o, System.out, System.err);
            } else {
                perplexity(m, o, System.out, System.err);
            }
        }
    }

    /**
     * Parses and validates the flags; {@code null} for {@code --help}, which prints the usage.
     *
     * @throws UsageException naming the flag for every unsupported or contradictory option
     */
    static Options parse(String[] args) throws UsageException {
        Options o = new Options();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            switch (a) {
                case "-f":
                case "--file": {
                    Path f = Paths.get(value(a, args, i++));
                    if (!Files.isReadable(f)) {
                        throw new UsageException(a + " " + f + " is not a readable file");
                    }
                    byte[] bytes = readQuietly(f, a);
                    // llama.cpp's read_file strips one trailing newline from the prompt
                    int end = bytes.length;
                    if (end > 0 && bytes[end - 1] == '\n') {
                        end--;
                    }
                    o.promptBytes = Arrays.copyOf(bytes, end);
                    o.promptText = new String(o.promptBytes, StandardCharsets.UTF_8);
                    break;
                }
                case "-p":
                case "--prompt":
                    o.promptText = value(a, args, i++);
                    o.promptBytes = o.promptText.getBytes(StandardCharsets.UTF_8);
                    break;
                case "-c":
                case "--ctx-size":
                    o.ctxSize = intValue(a, value(a, args, i++));
                    break;
                case "-b":
                case "--batch-size":
                    o.batchSize = intValue(a, value(a, args, i++));
                    break;
                case "--chunks":
                    o.chunks = intValue(a, value(a, args, i++));
                    if (o.chunks != -1 && o.chunks < 1) {
                        throw new UsageException("--chunks needs -1 for all chunks or a positive count, not "
                                + o.chunks);
                    }
                    break;
                case "--ppl-stride":
                    o.pplStride = intValue(a, value(a, args, i++));
                    break;
                case "--ppl-output-type":
                    o.pplOutputType = intValue(a, value(a, args, i++));
                    if (o.pplOutputType != 0 && o.pplOutputType != 1) {
                        throw new UsageException("--ppl-output-type is 0 or 1, not " + o.pplOutputType);
                    }
                    break;
                case "--hellaswag":
                    o.hellaswag = true;
                    break;
                case "--hellaswag-tasks":
                    o.hellaswagTasks = intValue(a, value(a, args, i++));
                    break;
                case "--winogrande":
                    o.winogrande = true;
                    break;
                case "--winogrande-tasks":
                    o.winograndeTasks = intValue(a, value(a, args, i++));
                    break;
                case "--multiple-choice":
                    o.multipleChoice = true;
                    break;
                case "--multiple-choice-tasks":
                    o.multipleChoiceTasks = intValue(a, value(a, args, i++));
                    break;
                case "--kl-divergence":
                    o.klDivergence = true;
                    break;
                case "--save-all-logits":
                case "--kl-divergence-base":
                    o.logitsPath = Paths.get(value(a, args, i++));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (o.model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + o.model);
                    }
                    o.model = Paths.get(a);
            }
        }
        if (o.model == null) {
            throw new UsageException("perplexity needs a model path");
        }
        int modes = (o.hellaswag ? 1 : 0) + (o.winogrande ? 1 : 0) + (o.multipleChoice ? 1 : 0)
                + (o.klDivergence ? 1 : 0);
        if (modes > 1) {
            throw new UsageException("one mode runs per invocation; --hellaswag, --winogrande,"
                    + " --multiple-choice and --kl-divergence exclude each other");
        }
        if (o.pplStride > 0 && modes > 0) {
            throw new UsageException("--ppl-stride shapes the perplexity run only; it does nothing with"
                    + " the benchmark and KL-divergence modes");
        }
        if (o.pplStride > 0 && o.logitsPath != null) {
            throw new UsageException("--save-all-logits stores the chunks of the plain perplexity run;"
                    + " the strided run stores nothing");
        }
        if (o.logitsPath != null && modes == 0 && o.logitsPath.toString().isEmpty()) {
            throw new UsageException("--save-all-logits needs a file name");
        }
        return o;
    }

    /** The engine config: the command line's context (widened by half the stride, as llama.cpp loads) and batch. */
    private static EngineConfig config(Options o, EngineConfig base) {
        int ctx = o.ctxSize;
        if (o.pplStride > 0) {
            ctx += o.pplStride / 2;
        }
        MemoryConfig.Builder mem = base.memory().toBuilder().nCtx(ctx);
        if (o.batchSize > 0) {
            mem.ubatch(o.batchSize);
        }
        return base.toBuilder().memory(mem.build()).build();
    }

    /**
     * Perplexity over the prompt text: the plain chunked run, its {@code --save-all-logits} file, or the
     * strided run when {@code --ppl-stride} is set (llama.cpp {@code perplexity} and {@code perplexity_v2}).
     *
     * @return the perplexity over everything scored
     * @throws IOException for a prompt too short for the context, a stride beyond the context or an
     *                     unwritable logits file
     */
    static double perplexity(GGUFModel m, Options o, Appendable out, Appendable err) throws IOException {
        ForwardBackend backend = m.forwardBackend();
        int nCtx = backend.memory().plan().nCtx();
        int nVocab = m.vocabSize();
        GGUFTokenizer tok = m.tokenizer();
        int[] tokens = tok.encode(o.promptText);
        if (tokens.length < 2 * nCtx) {
            throw new IOException("you need at least " + (2 * nCtx) + " tokens to evaluate perplexity with a"
                    + " context of " + nCtx + "; the data file you provided tokenizes to only " + tokens.length);
        }
        if (o.pplStride > 0) {
            return perplexityStride(m, o, tokens, nCtx, out);
        }

        int first = nCtx / 2;
        int nToken = nCtx - 1 - first;
        int nv = PerplexityMath.rowLength(nVocab);
        int nChunkMax = tokens.length / nCtx;
        int nChunk = o.chunks < 0 ? nChunkMax : Math.min(o.chunks, nChunkMax);
        LogitsFile.Out file = o.logitsPath != null ? new LogitsFile.Out(o.logitsPath, nCtx) : null;
        try {
            double nll = 0;
            double nll2 = 0;
            long count = 0;
            short[] rows16 = file != null ? new short[nToken * nv] : null;
            if (file != null) {
                file.writeRun(nVocab, nChunk, tokens, nChunk * nCtx);
            }
            for (int i = 0; i < nChunk; i++) {
                float[][] rows = chunkRows(backend, tok, tokens, i * nCtx, nCtx, first, nCtx - 2);                for (int k = 0; k < nToken; k++) {
                    int next = tokens[i * nCtx + first + k + 1];
                    double v = file != null
                            ? PerplexityMath.quantLogSoftmaxRow(rows[k], next, rows16, k * nv)
                            : -PerplexityMath.logSoftmax(rows[k], next);
                    nll += v;
                    nll2 += v * v;
                }
                if (file != null) {
                    file.writeRows(rows16, nToken, nv);
                }
                count += nToken;
                if (o.pplOutputType == 0) {
                    out.append(String.format(Locale.ROOT, "[%d]%.4f,", i + 1, Math.exp(nll / count)));
                } else {
                    out.append(type1Line(nll, nll2, count, (long) i * nCtx));
                }
            }
            out.append("\n");
            nll2 /= count;
            nll /= count;
            double ppl = Math.exp(nll);
            nll2 -= nll * nll;
            if (nll2 > 0) {
                nll2 = Math.sqrt(nll2 / (count - 1));
                err.append(String.format(Locale.ROOT,
                        "nftool perplexity: final estimate: PPL = %.4f +/- %.5f\n", ppl, nll2 * ppl));
            } else {
                err.append("nftool perplexity: unexpected negative standard deviation of log(prob)\n");
            }
            return ppl;
        } finally {
            if (file != null) {
                file.close();
            }
        }
    }

    /**
     * The strided perplexity (perplexity.cpp {@code perplexity_v2}): every chunk scores its trailing
     * {@code --ppl-stride} positions and the printed value accumulates over the whole run.
     */
    private static double perplexityStride(GGUFModel m, Options o, int[] tokens, int nCtx, Appendable out)
            throws IOException {
        ForwardBackend backend = m.forwardBackend();
        GGUFTokenizer tok = m.tokenizer();
        int stride = o.pplStride;
        if (stride > nCtx - 1) {
            throw new IOException("the stride " + stride + " covers more rows than the " + nCtx
                    + " token context holds; use a smaller --ppl-stride or a larger --ctx-size");
        }
        int calcChunk = nCtx;
        if (tokens.length <= calcChunk) {
            throw new IOException("there are only " + tokens.length + " tokens, not enough for a context size"
                    + " of " + calcChunk + " and stride " + stride);
        }
        int firstRow = nCtx - stride - 1;
        int nChunkMax = (tokens.length - calcChunk + stride - 1) / stride;
        int nChunk = o.chunks < 0 ? nChunkMax : Math.min(o.chunks, nChunkMax);
        double nll = 0;
        long count = 0;
        for (int i = 0; i < nChunk; i++) {
            int start = i * stride;
            float[][] rows = chunkRows(backend, tok, tokens, start, nCtx, firstRow, nCtx - 2);
            for (int j = firstRow; j <= nCtx - 2; j++) {
                float prob = PerplexityMath.softmaxAt(rows[j - firstRow], tokens[start + j + 1]);
                nll += -Math.log(prob);
                count++;
            }
            if (o.pplOutputType == 0) {
                out.append(String.format(Locale.ROOT, "[%d]%.4f,", i + 1, Math.exp(nll / count)));
            } else {
                out.append(String.format(Locale.ROOT, "%8d  %.4f\n", i * stride, Math.exp(nll / count)));
            }
        }
        out.append("\n");
        return Math.exp(nll / count);
    }

    /**
     * The KL divergence of the model against a base file (perplexity.cpp {@code kl_divergence}): the file's
     * tokens run in chunks and every scored row is compared against the base row the file streams next, the
     * per-chunk table and the final statistics printing as llama.cpp prints them.
     */
    static void klDivergence(GGUFModel m, Options o, Appendable out, Appendable err) throws IOException {
        if (o.logitsPath == null) {
            throw new IOException("you must provide the file with the log probabilities of the base model,"
                    + " through --kl-divergence-base");
        }
        ForwardBackend backend = m.forwardBackend();
        int nVocab = m.vocabSize();
        GGUFTokenizer tok = m.tokenizer();
        LogitsFile.In in = new LogitsFile.In(o.logitsPath);
        try {
            int nCtx = in.nCtx();
            int engineCtx = backend.memory().plan().nCtx();
            if (nCtx > engineCtx) {
                throw new IOException(o.logitsPath + " has been computed with a context of " + nCtx
                        + ", while the current context is " + engineCtx + "; increase --ctx-size and retry");
            }
            if (in.nVocab() != nVocab) {
                throw new IOException("inconsistent vocabulary: the base file holds " + in.nVocab()
                        + " entries, the model " + nVocab);
            }
            int[] tokens = in.tokens();
            int nChunk = in.nChunk();
            int first = nCtx / 2;
            int nToken = nCtx - 1 - first;
            int nv = PerplexityMath.rowLength(nVocab);
            short[] base = new short[nToken * nv];
            float[] kldValues = new float[nToken * nChunk];
            float[] pDiffValues = new float[nToken * nChunk];
            PerplexityMath.Kld kld = new PerplexityMath.Kld();
            float[] row = new float[2];
            err.append("nftool perplexity: kl-divergence: computing over " + nChunk + " chunks, context "
                    + nCtx + "\n\n");
            out.append("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence"
                    + "              Δp RMS            Same top p\n");
            for (int i = 0; i < nChunk; i++) {
                float[][] rows = chunkRows(backend, tok, tokens, i * nCtx, nCtx, first, nCtx - 2);
                in.readRows(base);
                for (int k = 0; k < nToken; k++) {
                    int next = tokens[i * nCtx + first + k + 1];
                    PerplexityMath.klRow(rows[k], base, k * nv, next, kld, row);
                    kldValues[i * nToken + k] = row[0];
                    pDiffValues[i * nToken + k] = row[1];
                }
                out.append(String.format(Locale.ROOT, "%4d", i + 1));
                appendKldRow(out, kld);
            }
            out.append("\n");
            if (kld.count < 100) {
                return;
            }
            float[] sortedKld = kldValues.clone();
            float[] sortedPDiff = pDiffValues.clone();
            Arrays.sort(sortedKld);
            Arrays.sort(sortedPDiff);
            appendPerplexityStatistics(out, kld);
            appendKldStatistics(out, kld, sortedKld);
            appendTokenStatistics(out, kld, sortedPDiff);
        } finally {
            in.close();
        }
    }

    /** One table row of the KL report: the running PPL, ratio, divergence, Δp RMS and top-token agreement. */
    private static void appendKldRow(Appendable out, PerplexityMath.Kld kld) throws IOException {
        double[] logPpl = kld.meanAndUncertainty(kld.sumNll, kld.sumNll2);
        double pplVal = Math.exp(logPpl[0]);
        double pplUnc = pplVal * logPpl[1];
        out.append(String.format(Locale.ROOT, "    %9.4f ± %9.4f", pplVal, pplUnc));

        double[] logPplBase = kld.meanAndUncertainty(kld.sumNllBase, kld.sumNllBase2);
        double logPplCov = kld.covariance(kld.sumNll, kld.sumNllBase, kld.sumNllNllBase);
        double ratioVal = logPpl[0] - logPplBase[0];
        double ratioUnc = Math.sqrt(logPpl[1] * logPpl[1] + logPplBase[1] * logPplBase[1] - 2.0 * logPplCov);
        out.append(String.format(Locale.ROOT, "    %10.5f ± %10.5f", ratioVal, ratioUnc));

        double[] klDiv = kld.meanAndUncertainty(kld.sumKld, kld.sumKld2);
        out.append(String.format(Locale.ROOT, "    %10.5f ± %10.5f", klDiv[0], klDiv[1]));

        double[] pDiffMse = kld.meanAndUncertainty(kld.sumPDiff2, kld.sumPDiff4);
        double rmsVal = Math.sqrt(pDiffMse[0]);
        double rmsUnc = 0.5 / rmsVal * pDiffMse[1];
        out.append(String.format(Locale.ROOT, "    %6.3f ± %6.3f %%", 100.0 * rmsVal, 100.0 * rmsUnc));

        double pTop = 1.0 * kld.nSameTop / kld.count;
        double pTopUnc = Math.sqrt(pTop * (1 - pTop) / (kld.count - 1));
        out.append(String.format(Locale.ROOT, "    %6.3f ± %6.3f %%\n", 100.0 * pTop, 100.0 * pTopUnc));
    }

    /** The perplexity block of the KL report (perplexity.cpp lines 1892-1921). */
    private static void appendPerplexityStatistics(Appendable out, PerplexityMath.Kld kld) throws IOException {
        out.append("====== Perplexity statistics ======\n");
        double[] logPpl = kld.meanAndUncertainty(kld.sumNll, kld.sumNll2);
        double pplVal = Math.exp(logPpl[0]);
        double pplUnc = pplVal * logPpl[1];
        out.append(String.format(Locale.ROOT, "Mean PPL(Q)                   : %10.6f ± %10.6f\n", pplVal, pplUnc));

        double[] logPplBase = kld.meanAndUncertainty(kld.sumNllBase, kld.sumNllBase2);
        double pplBaseVal = Math.exp(logPplBase[0]);
        double pplBaseUnc = pplBaseVal * logPplBase[1];
        out.append(String.format(Locale.ROOT, "Mean PPL(base)                : %10.6f ± %10.6f\n",
                pplBaseVal, pplBaseUnc));

        double logPplCov = kld.covariance(kld.sumNll, kld.sumNllBase, kld.sumNllNllBase);
        double logPplCor = logPplCov / (logPpl[1] * logPplBase[1]);
        out.append(String.format(Locale.ROOT, "Cor(ln(PPL(Q)), ln(PPL(base))): %6.2f%%\n", 100.0 * logPplCor));

        double ratioVal = logPpl[0] - logPplBase[0];
        double ratioUnc = Math.sqrt(logPpl[1] * logPpl[1] + logPplBase[1] * logPplBase[1] - 2.0 * logPplCov);
        out.append(String.format(Locale.ROOT, "Mean ln(PPL(Q)/PPL(base))     : %10.6f ± %10.6f\n",
                ratioVal, ratioUnc));

        double pplRatioVal = Math.exp(ratioVal);
        double pplRatioUnc = pplRatioVal * ratioUnc;
        out.append(String.format(Locale.ROOT, "Mean PPL(Q)/PPL(base)         : %10.6f ± %10.6f\n",
                pplRatioVal, pplRatioUnc));

        double pplCov = pplVal * pplBaseVal * logPplCov;
        double pplDiffVal = pplVal - pplBaseVal;
        double pplDiffUnc = Math.sqrt(pplUnc * pplUnc + pplBaseUnc * pplBaseUnc - 2.0 * pplCov);
        out.append(String.format(Locale.ROOT, "Mean PPL(Q)-PPL(base)         : %10.6f ± %10.6f\n\n",
                pplDiffVal, pplDiffUnc));
    }

    /** The KL-divergence block of the report: the mean, the percentiles and the extremes of the sorted rows. */
    private static void appendKldStatistics(Appendable out, PerplexityMath.Kld kld, float[] sortedKld)
            throws IOException {
        out.append("====== KL divergence statistics ======\n");
        double[] klDiv = kld.meanAndUncertainty(kld.sumKld, kld.sumKld2);
        out.append(String.format(Locale.ROOT, "Mean    KLD: %10.6f ± %10.6f\n", klDiv[0], klDiv[1]));
        double kldMedian = sortedKld.length % 2 == 0
                ? 0.5f * (sortedKld[sortedKld.length / 2] + sortedKld[sortedKld.length / 2 - 1])
                : sortedKld[sortedKld.length / 2];
        out.append(String.format(Locale.ROOT, "Maximum KLD: %10.6f\n", sortedKld[sortedKld.length - 1]));
        out.append(String.format(Locale.ROOT, "99.9%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.999f)));
        out.append(String.format(Locale.ROOT, "99.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.990f)));
        out.append(String.format(Locale.ROOT, "95.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.950f)));
        out.append(String.format(Locale.ROOT, "90.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.900f)));
        out.append(String.format(Locale.ROOT, "Median  KLD: %10.6f\n", kldMedian));
        out.append(String.format(Locale.ROOT, "10.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.100f)));
        out.append(String.format(Locale.ROOT, " 5.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.050f)));
        out.append(String.format(Locale.ROOT, " 1.0%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.010f)));
        out.append(String.format(Locale.ROOT, " 0.1%%   KLD: %10.6f\n", PerplexityMath.percentile(sortedKld, 0.001f)));
        out.append(String.format(Locale.ROOT, "Minimum KLD: %10.6f\n\n", sortedKld[0]));
    }

    /** The token-probability block of the report (perplexity.cpp lines 1952-1982). */
    private static void appendTokenStatistics(Appendable out, PerplexityMath.Kld kld, float[] sortedPDiff)
            throws IOException {
        out.append("====== Token probability statistics ======\n");
        double[] pDiff = kld.meanAndUncertainty(kld.sumPDiff, kld.sumPDiff2);
        out.append(String.format(Locale.ROOT, "Mean    Δp: %6.3f ± %5.3f %%\n", 100.0 * pDiff[0], 100.0 * pDiff[1]));
        double pDiffMedian = sortedPDiff.length % 2 == 0
                ? 0.5f * (sortedPDiff[sortedPDiff.length / 2] + sortedPDiff[sortedPDiff.length / 2 - 1])
                : sortedPDiff[sortedPDiff.length / 2];
        float[] fractions = {0.999f, 0.990f, 0.950f, 0.900f, 0.750f, 0.250f, 0.100f, 0.050f, 0.010f, 0.001f};
        String[] labels = {"99.9", "99.0", "95.0", "90.0", "75.0", "25.0", "10.0", " 5.0", " 1.0", " 0.1"};
        out.append(String.format(Locale.ROOT, "Maximum Δp: %6.3f%%\n",
                100.0 * sortedPDiff[sortedPDiff.length - 1]));
        for (int i = 0; i < fractions.length; i++) {
            out.append(String.format(Locale.ROOT, labels[i] + "%%   Δp: %6.3f%%\n",
                    100.0 * PerplexityMath.percentile(sortedPDiff, fractions[i])));
        }
        out.append(String.format(Locale.ROOT, "Median  Δp: %6.3f%%\n", 100.0 * pDiffMedian));
        out.append(String.format(Locale.ROOT, "Minimum Δp: %6.3f%%\n", 100.0 * sortedPDiff[0]));
        double[] pDiffMse = kld.meanAndUncertainty(kld.sumPDiff2, kld.sumPDiff4);
        double rmsVal = Math.sqrt(pDiffMse[0]);
        double rmsUnc = 0.5 / rmsVal * pDiffMse[1];
        out.append(String.format(Locale.ROOT, "RMS Δp    : %6.3f ± %5.3f %%\n", 100.0 * rmsVal, 100.0 * rmsUnc));
        double sameTop = 1.0 * kld.nSameTop / kld.count;
        out.append(String.format(Locale.ROOT, "Same top p: %6.3f ± %5.3f %%\n", 100.0 * sameTop,
                100.0 * Math.sqrt(sameTop * (1.0 - sameTop) / (kld.count - 1))));
    }

    /** The perplexity line of {@code --ppl-output-type 1}: the chunk's first token, the PPL, mean and error. */
    private static String type1Line(double nll, double nll2, long count, long firstToken) {
        double av = nll / count;
        double av2 = nll2 / count - av * av;
        if (av2 > 0) {
            av2 = Math.sqrt(av2 / (count - 1));
        }
        return String.format(Locale.ROOT, "%8d  %.4f  %4f  %4f\n", firstToken, Math.exp(av), av, av2);
    }

    /** Forwards one chunk from position 0 with the vocabulary's BOS at its first token, as llama.cpp does. */
    private static float[][] chunkRows(ForwardBackend backend, GGUFTokenizer tok, int[] tokens,
            int start, int nCtx, int firstRow, int lastRow) {
        int[] chunk = Arrays.copyOfRange(tokens, start, start + nCtx);
        int bos = tok.bosId();
        if (tok.vocab().addBos() && bos >= 0) {
            chunk[0] = bos;
        }
        int n = lastRow - firstRow + 1;
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = firstRow + i;
        }
        backend.reset();
        return backend.forwardRows(chunk, 0, rows);
    }

    /** The tokenizer facts the chunk loops need: whether a BOS leads every chunk and its id. */
    private static final class GGUFTokenizerHolder {
        final boolean addBos;
        final int bos;
        private final com.eonmux.neuralfree.gguf.GGUFTokenizer tokenizer;

        GGUFTokenizerHolder(com.eonmux.neuralfree.gguf.GGUFTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            this.addBos = tokenizer.vocab().addBos();
            this.bos = tokenizer.bosId();
        }

        int[] encode(String text) {
            return tokenizer.encode(text);
        }
    }

    /** Opens the model through the factory, as the demos open theirs. */
    private static GGUFModel open(Path model, EngineConfig cfg) throws IOException {
        EngineModel opened = LLMModelFactory.open(model.toString(), cfg);
        if (!(opened instanceof GGUFModel)) {
            throw new IOException("the perplexity tool runs the token engine, but " + model + " opened as "
                    + opened.getClass().getSimpleName());
        }
        return (GGUFModel) opened;
    }

    /** The perplexity chunk loop runs a decoder with memory, as the imatrix collector does. */
    private static void requireTokenLoop(GGUFModel m) throws IOException {
        SupportClass support = m.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
            case DRAFT:
                throw new IOException("the perplexity tool runs the decoder prefill of a model with memory; "
                        + m.architecture() + " (" + support + ") has none");
            default:
                return;
        }
    }

    /** The bytes of a data file, an unreadable file being the flag's own error. */
    private static byte[] readQuietly(Path f, String flag) throws UsageException {
        try {
            return Files.readAllBytes(f);
        } catch (IOException e) {
            throw new UsageException(flag + " " + f + " cannot be read: " + e.getMessage());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative count of the flag, as llama.cpp parses its task counts. */
    private static int intValue(String flag, String v) throws UsageException {
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
}
