package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.train.FinetuneDataset;
import com.eonmux.neuralfree.gguf.train.FinetuneOptimizer;
import com.eonmux.neuralfree.gguf.train.FinetuneRunner;
import com.eonmux.neuralfree.gguf.train.GpuTrainableModel;
import com.eonmux.neuralfree.gguf.train.TrainableModel;
import com.eonmux.neuralfree.gguf.train.Trainer;
import com.eonmux.neuralfree.gguf.train.TrainedSource;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@code nftool finetune}: trains one F32 GGUF model on a text corpus and writes the trained model back to GGUF, the
 * job of llama.cpp's {@code llama-finetune} example ({@code examples/training/finetune.cpp}).
 *
 * <p>The run tokenizes the corpus once, cuts it into context windows that overlap by half a context, and steps an
 * optimizer once per window for every epoch, scheduling the learning rate from {@code -lr0} toward {@code -lr-min}
 * over {@code -decay-epochs} epochs. The validation windows ({@code --val-split}) are scored after every epoch but
 * never trained on. The last epoch's weights are written to {@code -o} through
 * {@link com.eonmux.neuralfree.gguf.GgufWriter}, which keeps the metadata, tokenizer and every tensor the trainer did
 * not step (see {@link TrainedSource}).</p>
 *
 * <p>The flags are llama-finetune's. The tool trains whole windows, so {@code -b} must be {@code -c}.
 * {@code -ngl 0} (the default) runs the host trainer; a count covering every trunk layer runs the training graph on
 * the GPU backend's train kernels ({@link GpuTrainableModel}), and a partial count is refused because the trainer
 * keeps the whole graph on one backend. A model the trainer cannot represent — a non-F32/F16/BF16 model, a recurrent
 * or otherwise unsupported architecture, a context longer than the model's own, or a GPU run whose activation family
 * the train kernels do not provide — is refused by name before any training happens. As llama-finetune does not
 * overwrite, an existing {@code -o} file is refused.</p>
 */
final class FinetuneCommand {

    /** llama-finetune's {@code n_ctx} default. */
    static final int DEFAULT_CTX = 128;
    /** llama-finetune's {@code n_epochs} default. */
    static final int DEFAULT_EPOCHS = 1;
    /** llama-finetune's {@code lr0} default. */
    static final float DEFAULT_LR0 = 1e-4f;
    /** llama-finetune's {@code lr_min} default. */
    static final float DEFAULT_LR_MIN = 1e-5f;
    /** llama-finetune's {@code decay_epochs} default. */
    static final float DEFAULT_DECAY_EPOCHS = 10f;
    /** llama-finetune's {@code wd} default. */
    static final float DEFAULT_WEIGHT_DECAY = 0.1f;
    /** llama-finetune's {@code val_split} default. */
    static final double DEFAULT_VAL_SPLIT = 0.05;

    private static final String USAGE =
            "nftool finetune -m <model.gguf> -f <corpus.txt> -o <out.gguf> [options]\n"
            + "options (llama-finetune names):\n"
            + "  -m, --model <fn>                 the F32 GGUF model to train\n"
            + "  -f, --file <fn>                  the training corpus\n"
            + "  -o, --output <fn>                the GGUF file the trained weights are written to\n"
            + "  -c, --ctx-size <n>               the context of a training window, default " + DEFAULT_CTX + "\n"
            + "  -b, --batch-size <n>             must equal -c: a step trains one whole window\n"
            + "  -ngl, --gpu-layers <n>          0 trains on the CPU; a count covering every trunk layer trains on\n"
            + "                                   the GPU (partial offload is refused)\n"
            + "  -opt, --optimizer <adamw|sgd>    the update rule, default adamw\n"
            + "  -lr0, --learning-rate <f>        the learning rate of the first epoch, default " + DEFAULT_LR0 + "\n"
            + "  -lr-min, --learning-rate-min <f> the floor of the learning-rate schedule, default "
            + DEFAULT_LR_MIN + "\n"
            + "  -decay-epochs <f>                the epochs the schedule decays over, default "
            + DEFAULT_DECAY_EPOCHS + "\n"
            + "  -wd, --weight-decay <f>          the weight decay, default " + DEFAULT_WEIGHT_DECAY + "\n"
            + "  -epochs, --epochs <n>            the number of passes over the corpus, default "
            + DEFAULT_EPOCHS + "\n"
            + "  --val-split <f>                  the fraction of windows that validate, default "
            + DEFAULT_VAL_SPLIT + "\n"
            + "  --help, -h                       this text\n"
            + "The CPU trainer runs on the engine's CPU pool (-Dneuralfree.cpu.threads sizes it); -ngl runs the\n"
            + "training graph on the train kernels of the GPU backend neuralfree.backend selects.\n";

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
        Path corpus;
        Path output;
        int ctx = DEFAULT_CTX;
        int batch = -1;
        int gpuLayers;
        FinetuneOptimizer.Kind optimizer = FinetuneOptimizer.Kind.ADAMW;
        float lr0 = DEFAULT_LR0;
        float lrMin = DEFAULT_LR_MIN;
        float decayEpochs = DEFAULT_DECAY_EPOCHS;
        float weightDecay = DEFAULT_WEIGHT_DECAY;
        int epochs = DEFAULT_EPOCHS;
        double valSplit = DEFAULT_VAL_SPLIT;
    }

    private FinetuneCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code finetune}. */
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
     * Parses the arguments, trains and writes the output. Package-private so tests drive it with an explicit engine
     * configuration.
     */
    static void dispatch(String[] args, EngineConfig base) throws IOException, UsageException {
        Options o = parse(args);
        if (o == null) {
            return;
        }
        if (Files.exists(o.output)) {
            throw new UsageException("-o " + o.output + " already exists; nftool finetune does not overwrite a"
                    + " model file");
        }
        if (!Files.isReadable(o.corpus)) {
            throw new UsageException("-f " + o.corpus + " is not a readable file");
        }
        String text;
        try {
            text = new String(Files.readAllBytes(o.corpus), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UsageException("-f " + o.corpus + " cannot be read: " + e.getMessage());
        }
        EngineConfig cfg = config(o, base);
        try (GGUFModel model = open(o.model, cfg)) {
            if (o.gpuLayers == 0) {
                train(o, model, text, cfg, null);
                return;
            }
            int layers = model.spec().layers().size();
            if (o.gpuLayers < layers) {
                throw new UsageException("-ngl " + o.gpuLayers + " would leave " + (layers - o.gpuLayers)
                        + " of the model's " + layers + " trunk layers on the CPU; the GPU finetune trainer runs the"
                        + " whole graph on one backend, so use -ngl 0 or -ngl " + layers + " (or more)");
            }
            try (GpuBackend gpu = openGpu(base)) {
                train(o, model, text, cfg, gpu.backend());
            }
        }
    }

    /** The training run: the dataset, the epochs, then the trained GGUF. */
    private static void train(Options o, GGUFModel model, String text, EngineConfig cfg, ComputeBackend backend)
            throws IOException {
        FinetuneRunner.Config config = new FinetuneRunner.Config(o.epochs, o.lr0, o.lrMin, o.decayEpochs,
                o.weightDecay, o.optimizer);
        try (Trainer train = openTrainer(o, model, backend)) {
            FinetuneDataset data = dataset(o, model, text);
            System.out.println("nftool finetune: " + data + " with " + o.optimizer.flag() + ", " + o.epochs
                    + " epoch(s) on " + (backend == null ? "the CPU" : backend.backendType()));
            List<FinetuneRunner.EpochStats> epochs = FinetuneRunner.run(train, data, config,
                    stats -> System.out.println("nftool finetune: " + stats));
            write(o, model, train, cfg, epochs);
        }
    }

    /** Writes the trained weights over the model's own bytes, keeping every other tensor. */
    private static void write(Options o, GGUFModel model, Trainer train, EngineConfig cfg,
                              List<FinetuneRunner.EpochStats> epochs) throws IOException {
        Map<String, float[]> values = new LinkedHashMap<>();
        for (TrainableModel.Parameter p : train.parameters()) {
            values.put(p.name(), p.value());
        }
        ModelSource base = ModelSources.open(o.model.toString(), cfg.sourceOptions());
        TrainedSource source;
        try {
            source = TrainedSource.of(base, values);
        } catch (RuntimeException e) {
            base.close();
            throw e;
        }
        try (TrainedSource trained = source) {
            List<Path> written = GgufWriter.write(trained, o.output,
                    new GgufWriter.Options(GgufWriter.Mode.NATIVE));
            FinetuneRunner.EpochStats last = epochs.get(epochs.size() - 1);
            System.err.println(String.format(Locale.ROOT, "nftool finetune: wrote %s with %d trained tensors after"
                    + " %d epoch(s), training loss %.6f", written.get(written.size() - 1),
                    trained.trainedTensorCount(), epochs.size(), last.trainLoss()));
        }
    }

    /** The trainer over the opened model: the host trainer, or the device one when a GPU backend was selected. */
    private static Trainer openTrainer(Options o, GGUFModel model, ComputeBackend backend) throws IOException {
        try {
            return backend == null ? TrainableModel.open(model, o.ctx)
                    : GpuTrainableModel.open(model, o.ctx, backend);
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    /** A GPU backend opened for the run, with the acceleration handle that owns it. */
    private static final class GpuBackend implements AutoCloseable {
        private final HWAcceleration acceleration;
        private final ComputeBackend backend;

        GpuBackend(HWAcceleration acceleration, ComputeBackend backend) {
            this.acceleration = acceleration;
            this.backend = backend;
        }

        ComputeBackend backend() {
            return backend;
        }

        @Override
        public void close() {
            acceleration.shutdown();
        }
    }

    /** Opens the GPU backend that provides the train kernels; the run's {@code -ngl} selected it. */
    private static GpuBackend openGpu(EngineConfig base) throws UsageException {
        if (!base.backend().usesGpu()) {
            throw new UsageException("-ngl needs a GPU backend, but neuralfree.backend selects the CPU");
        }
        HWAcceleration acceleration = null;
        try {
            acceleration = HWAcceleration.forMode(base.backend().toAccelerationMode());
            ComputeBackend backend = acceleration.createBackend(base.backend().toAccelerationMode(),
                    TrainKernels.requirement());
            return new GpuBackend(acceleration, backend);
        } catch (RuntimeException e) {
            if (acceleration != null) {
                acceleration.shutdown();
            }
            throw new UsageException("-ngl: no GPU backend provides the train kernels: " + e.getMessage());
        }
    }

    /** The dataset of the corpus, its refusals phrased as the flags that asked for the shape. */
    private static FinetuneDataset dataset(Options o, GGUFModel model, String text) throws IOException {
        try {
            return FinetuneDataset.ofText(text, model.tokenizer(), o.ctx, o.valSplit);
        } catch (IllegalArgumentException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    /** The engine configuration of the run: the command line's context, over the CPU engine. */
    private static EngineConfig config(Options o, EngineConfig base) {
        return base.toBuilder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(base.memory().toBuilder().nCtx(o.ctx).ubatch(o.ctx).build())
                .build();
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
                case "-m":
                case "--model":
                    o.model = path(a, value(a, args, i++));
                    break;
                case "-f":
                case "--file":
                    o.corpus = path(a, value(a, args, i++));
                    break;
                case "-o":
                case "--output":
                    o.output = path(a, value(a, args, i++));
                    break;
                case "-c":
                case "--ctx":
                case "--ctx-size":
                    o.ctx = intValue(a, value(a, args, i++), 3);
                    break;
                case "-b":
                case "--batch":
                case "--batch-size":
                case "-ub":
                case "--ubatch-size":
                    o.batch = intValue(a, value(a, args, i++), 1);
                    break;
                case "-ngl":
                case "--gpu-layers":
                    o.gpuLayers = intValue(a, value(a, args, i++), 0);
                    break;
                case "-opt":
                case "--optimizer": {
                    String name = value(a, args, i++);
                    FinetuneOptimizer.Kind kind = FinetuneOptimizer.Kind.parse(name);
                    if (kind == null) {
                        throw new UsageException(a + " takes " + FinetuneOptimizer.Kind.names() + ", not '" + name
                                + "'");
                    }
                    o.optimizer = kind;
                    break;
                }
                case "-lr0":
                case "--learning-rate":
                    o.lr0 = floatValue(a, value(a, args, i++));
                    break;
                case "-lr-min":
                case "--lr-min":
                case "--learning-rate-min":
                    o.lrMin = floatValue(a, value(a, args, i++));
                    break;
                case "-decay-epochs":
                    o.decayEpochs = floatValue(a, value(a, args, i++));
                    break;
                case "-wd":
                case "--weight-decay":
                    o.weightDecay = floatValue(a, value(a, args, i++));
                    break;
                case "-epochs":
                case "--epochs":
                    o.epochs = intValue(a, value(a, args, i++), 1);
                    break;
                case "--val-split":
                    o.valSplit = doubleValue(a, value(a, args, i++));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    throw new UsageException("unexpected argument '" + a + "'; every path is named by its flag"
                            + " (-m, -f, -o)");
            }
        }
        require(o.model, "-m/--model");
        require(o.corpus, "-f/--file");
        require(o.output, "-o/--output");
        if (o.batch != -1 && o.batch != o.ctx) {
            throw new UsageException("-b " + o.batch + " differs from -c " + o.ctx + "; nftool finetune trains one"
                    + " whole window per step, so the batch must equal the context");
        }
        if (o.gpuLayers < 0) {
            throw new UsageException("-ngl " + o.gpuLayers + " is negative");
        }
        if (!(o.valSplit >= 0.0 && o.valSplit < 1.0)) {
            throw new UsageException("--val-split " + o.valSplit + " is not in [0, 1)");
        }
        if (!(o.lr0 > 0f)) {
            throw new UsageException("-lr0 " + o.lr0 + " is not a positive learning rate");
        }
        if (!(o.lrMin > 0f)) {
            throw new UsageException("-lr-min " + o.lrMin + " is not a positive learning rate");
        }
        if (!(o.lrMin <= o.lr0)) {
            throw new UsageException("-lr-min " + o.lrMin + " is above -lr0 " + o.lr0);
        }
        if (!(o.decayEpochs >= 0f)) {
            throw new UsageException("-decay-epochs " + o.decayEpochs + " is negative");
        }
        if (!(o.weightDecay >= 0f)) {
            throw new UsageException("-wd " + o.weightDecay + " is negative");
        }
        return o;
    }

    private static void require(Path value, String flag) throws UsageException {
        if (value == null) {
            throw new UsageException("finetune needs " + flag);
        }
    }

    /** Opens the model on the CPU engine, as the trainer needs its host weights. */
    private static GGUFModel open(Path model, EngineConfig cfg) throws IOException, UsageException {
        if (!Files.isReadable(model)) {
            throw new UsageException("-m " + model + " is not a readable file");
        }
        EngineModel opened = LLMModelFactory.open(model.toString(), cfg);
        if (!(opened instanceof GGUFModel)) {
            throw new IOException("the finetune trainer runs the token engine, but " + model + " opened as "
                    + opened.getClass().getSimpleName());
        }
        return (GGUFModel) opened;
    }

    private static Path path(String flag, String v) throws UsageException {
        try {
            return Paths.get(v);
        } catch (java.nio.file.InvalidPathException e) {
            throw new UsageException(flag + " '" + v + "' is not a path: " + e.getMessage());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** An integer of at least {@code min}, the flag's own error otherwise. */
    private static int intValue(String flag, String v, int min) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < min) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer of at least " + min + ", not '" + v + "'");
        }
    }

    /** A finite float of any sign; the run's own validation refuses the out-of-range values. */
    private static float floatValue(String flag, String v) throws UsageException {
        try {
            float f = Float.parseFloat(v.trim());
            if (!Float.isFinite(f)) {
                throw new NumberFormatException();
            }
            return f;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a finite number, not '" + v + "'");
        }
    }

    /** A finite double; the run's own validation refuses the out-of-range values. */
    private static double doubleValue(String flag, String v) throws UsageException {
        try {
            double d = Double.parseDouble(v.trim());
            if (!Double.isFinite(d)) {
                throw new NumberFormatException();
            }
            return d;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a finite number, not '" + v + "'");
        }
    }
}
