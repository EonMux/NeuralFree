package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.FitParams;
import com.eonmux.neuralfree.gguf.FitParamsPlanner;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * {@code nftool fit-params}: fits the parameters of a device load to the free memory, as llama.cpp's
 * {@code tools/fit-params} does. The fit plans the load against the device budget — the context
 * ({@code -c}), the layer placement (an automatic {@code -ngl} split or the streaming ladder) and the expert
 * overflow ({@code --cpu-moe} / {@code -ot}) — exactly as {@code com.eonmux.neuralfree.gguf.GGUFModelGPU}
 * plans them at load, then prints the arguments that reproduce it. {@code --fit-target} leaves a margin of
 * free device memory beside the load (llama.cpp's {@code -fitt}, 1024 MiB by default); {@code --fit-ctx} is
 * the smallest context the fit may reduce to (its {@code -fitc}, 4096 by default); {@code --fit-print} adds
 * the memory estimate behind the choice (its {@code -fitp}). Nothing is uploaded, so the tool runs wherever
 * the model can be opened.
 *
 * <p>The flags that fix the placement the fit would choose ({@code -ngl}, {@code -ot}, {@code --cpu-moe},
 * {@code --n-cpu-moe}, {@code --n-cpu-ffn}, {@code --fit}) are refused by name, as llama.cpp's fit refuses a
 * set {@code n_gpu_layers} or tensor buffer override; the rest of the demos' load flags
 * ({@code --backend}, {@code --device}, {@code -c}, {@code -ub}, the KV types, ...) shape the fit like they
 * shape the load.</p>
 */
final class FitParamsCommand {

    /** Free device memory the fit leaves beside the load, in MiB (llama.cpp's {@code --fit-target} default). */
    static final long DEFAULT_FIT_TARGET_MIB = 1024;

    /** Smallest context the fit reduces to (llama.cpp's {@code --fit-ctx} default). */
    static final int DEFAULT_FIT_CTX = 4096;

    private static final long MIB = 1024L * 1024L;

    /** The flags whose placement the fit would have to choose itself; refused by name. */
    private static final List<String> PLACEMENT_FLAGS = Arrays.asList(
            "-ngl", "--gpu-layers", "--n-gpu-layers",
            "-ot", "--override-tensor",
            "--cpu-moe", "-cmoe", "--n-cpu-moe", "-ncmoe", "--n-cpu-ffn", "-ncffn",
            "--fit");

    private static final String USAGE =
            "nftool fit-params <model> [options]\n"
            + "options:\n"
            + "  -m, --model <path>               the model file (or give it as the only argument)\n"
            + "  --fit-target, -fitt <MiB>        free device memory to leave beside the load, default "
                    + DEFAULT_FIT_TARGET_MIB + "\n"
            + "  --fit-ctx, -fitc <n>             smallest context the fit may reduce to, default "
                    + DEFAULT_FIT_CTX + "\n"
            + "  --fit-print, -fitp               print the memory estimate beside the fitted arguments\n"
            + "  --help, -h                       this text\n"
            + EngineLoadFlags.usageSection("load flags (they shape the fit as they shape the load):")
            + "The fitted arguments (the flags a load of this model takes to run as fitted) print to stdout;\n"
            + "the fit itself never uploads weights.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line: the model, the fit flags and the load configuration they shape. */
    static final class Options {
        Path model;
        long fitTargetMb = DEFAULT_FIT_TARGET_MIB;
        int fitCtx = DEFAULT_FIT_CTX;
        boolean fitPrint;
        EngineConfig cfg;
    }

    private FitParamsCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code fit-params}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o == null) {
                return;
            }
            execute(o);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the arguments over the {@code -Dneuralfree.*} property configuration. */
    static Options parse(String[] args) throws UsageException {
        try {
            return parse(args, EngineConfig.fromSystemProperties());
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /**
     * Parses the flags over {@code base}: the fit flags, the context and batch flags, then the demos' load
     * flags ({@link EngineLoadFlags}); a placement flag is refused by name. {@code null} for {@code --help},
     * which prints the usage.
     *
     * @throws UsageException naming the flag for every unknown, missing or contradictory option
     */
    static Options parse(String[] args, EngineConfig base) throws UsageException {
        MemoryConfig.Builder memory = base.memory().toBuilder();
        EngineLoadFlags load = new EngineLoadFlags(base.toBuilder(), memory, base.moeCache(), base.eplb());
        Options o = new Options();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            switch (a) {
                case "-m": case "--model":
                    o.model = Paths.get(value(a, args, i++));
                    continue;
                case "--fit-target": case "-fitt":
                    o.fitTargetMb = mibValue(a, value(a, args, i++));
                    continue;
                case "--fit-ctx": case "-fitc":
                    o.fitCtx = positiveInt(a, value(a, args, i++));
                    continue;
                case "--fit-print": case "-fitp":
                    o.fitPrint = true;
                    continue;
                case "--ctx": case "-c": case "--ctx-size": case "--max-context":
                    memory.nCtx(nonNegativeInt(a, value(a, args, i++)));
                    continue;
                case "--ubatch": case "-ub": case "--ubatch-size":
                    memory.ubatch(positiveInt(a, value(a, args, i++)));
                    continue;
                default:
                    // fall through to the shared load flags
            }
            if (PLACEMENT_FLAGS.contains(a)) {
                throw new UsageException(a + " fixes the placement the fit has to choose; clear it and let the"
                        + " fit decide, its result is printed as the " + fittingFlag(a) + " of the arguments");
            }
            try {
                int last = load.parse(args, i);
                if (last < 0) {
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (o.model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + o.model);
                    }
                    o.model = Paths.get(a);
                } else {
                    i = last;
                }
            } catch (EngineLoadFlags.FlagException e) {
                throw new UsageException(e.getMessage());
            }
        }
        if (o.model == null) {
            throw new UsageException("fit-params needs a model path");
        }
        try {
            o.cfg = load.build();
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
        return o;
    }

    /** The flag of the fitted arguments a refused placement flag would have been chosen by the fit. */
    private static String fittingFlag(String refused) {
        switch (refused) {
            case "--fit":
                return EngineConfig.PROP_CPU_MOE + " / -ot entries";
            case "-ot": case "--override-tensor":
                return "-ot entries";
            case "--cpu-moe": case "-cmoe": case "--n-cpu-moe": case "-ncmoe":
                return "--cpu-moe / -ot entries";
            default:
                return "-ngl";
        }
    }

    /** Opens the backend the load would run on, plans the fit and prints it. The acceleration owns the
     * backend, so one {@code shutdown} releases both, as the engine releases its own backend. */
    static void execute(Options o) throws IOException, UsageException {
        EngineConfig cfg = o.cfg;
        if (!cfg.backend().usesGpu()) {
            throw new UsageException("--backend cpu has no device memory to fit; fit-params plans a device load");
        }
        HWAcceleration acceleration = HWAcceleration.forMode(cfg.backend().toAccelerationMode());
        try {
            ComputeBackend backend = cfg.picksMainGpu()
                    ? GpuDevices.openMainGpu(acceleration, cfg).backend()
                    : acceleration.createBackend(cfg.backend().toAccelerationMode(), LlmKernels.requirement());
            try (GGUFFileParser parser = new GGUFFileParser(o.model.toString())) {
                FitParams fit = new FitParamsPlanner.Builder(parser, backend, cfg)
                        .kernels(LlmKernels.open(backend))
                        .marginBytes(Math.multiplyExact(o.fitTargetMb, MIB))
                        .contextFloor(o.fitCtx)
                        .plan();
                print(o, fit);
            }
        } finally {
            acceleration.shutdown();
        }
    }

    /** The estimate (with {@code --fit-print}) and the fitted arguments. */
    private static void print(Options o, FitParams fit) {
        System.out.println("fit-params " + o.model + ": " + fit.mode() + " placement, "
                + fit.gpuLayers() + " of the trunk weight layers keep their weights on the device");
        if (o.fitPrint) {
            System.out.println(fit.describe());
        }
        System.out.println(fit.arguments());
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative integer count of the flag. */
    private static int nonNegativeInt(String flag, String v) throws UsageException {
        int n = intValue(flag, v);
        if (n < 0) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
        return n;
    }

    /** A positive integer count of the flag. */
    private static int positiveInt(String flag, String v) throws UsageException {
        int n = intValue(flag, v);
        if (n < 1) {
            throw new UsageException(flag + " needs a positive integer, not '" + v + "'");
        }
        return n;
    }

    private static int intValue(String flag, String v) throws UsageException {
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, got '" + v + "'");
        }
    }

    /** A non-negative MiB count of the flag. */
    private static long mibValue(String flag, String v) throws UsageException {
        try {
            long n = Long.parseLong(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a non-negative integer of MiB, got '" + v + "'");
        }
    }
}
