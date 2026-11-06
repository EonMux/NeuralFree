package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.LoraMerger;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.SourceOrigin;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code nftool export-lora}: the LoRA surface of llama.cpp's {@code tools/export-lora/export-lora.cpp} and
 * {@code convert_lora_to_gguf.py}, over one command. With {@code --lora} / {@code --lora-scaled} the tool merges
 * every adapter into the base model's weights ({@link LoraMerger#merge}) and writes the merged GGUF: the adapters
 * open as GGUF adapter files or PEFT directories through {@code ModelSources.openAdapter}, the deltas apply as
 * {@code userScale * alpha / rank * B A}, quantized base tensors dequantize to F32, and merged tensors write as F16
 * (F32 bases stay F32). With {@code --peft <dir>} the tool converts the PEFT adapter directory itself to the GGUF
 * adapter file {@code convert_lora_to_gguf.py} writes ({@link GgufWriter#write} over the adapter view of the base
 * model at the first positional). Flags parse in the {@code LLMDemoOptions} style; {@code -Dneuralfree.*}
 * properties still apply to every open.
 */
final class ExportLoraCommand {

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool export-lora <base-model> <out.gguf> [options]\n"
            + "options:\n"
            + "  --lora <adapter>                  merge this adapter at scale 1 (a GGUF adapter file or a\n"
            + "                                    PEFT directory; repeatable)\n"
            + "  --lora-scaled <adapter> <scale>   merge this adapter at <scale> (repeatable)\n"
            + "  --peft <dir>                      convert the PEFT adapter directory <dir> to an adapter\n"
            + "                                    GGUF instead of merging (out.gguf is the adapter file)\n"
            + "  --overwrite                       replace the output\n"
            + "  --help, -h                        this text\n";

    private ExportLoraCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code export-lora}. */
    static void run(String[] args) {
        try {
            Path out = execute(args);
            if (out != null) {
                System.out.println("nftool export-lora: wrote " + out);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags and merges or converts; {@code null} for {@code --help}, which prints the usage and
     * returns nothing written.
     */
    static Path execute(String[] args) throws IOException, UsageException {
        Path base = null;
        Path out = null;
        String peft = null;
        boolean overwrite = false;
        List<LoraMerger.Adapter> adapters = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return null;
            }
            if (a.equals("--overwrite")) {
                overwrite = true;
            } else if (a.equals("--peft")) {
                peft = value(a, args, i++);
            } else if (a.equals("--lora")) {
                adapters.add(new LoraMerger.Adapter(value(a, args, i++), 1.0f));
            } else if (a.equals("--lora-scaled")) {
                String path = value(a, args, i++);
                adapters.add(new LoraMerger.Adapter(path, scaleValue(a, value(a, args, i++))));
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (base == null) {
                base = Paths.get(a);
            } else if (out == null) {
                out = Paths.get(a);
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + base + " and " + out);
            }
        }
        if (base == null) {
            throw new UsageException("export-lora needs a base model path and an output path");
        }
        if (out == null) {
            throw new UsageException("export-lora needs an output path after " + base);
        }
        if (peft != null && !adapters.isEmpty()) {
            throw new UsageException("--peft converts the adapter '" + peft + "' itself; it takes no --lora or"
                    + " --lora-scaled");
        }
        ModelSourceOptions options = ModelSourceOptions.fromSystemProperties();
        try (ModelSource src = ModelSources.open(base.toString(), options)) {
            return dispatch(src, peft, adapters, out, options, overwrite);
        }
    }

    /** Runs the parsed command over the opened base: the PEFT conversion, or the merge. */
    private static Path dispatch(ModelSource src, String peft, List<LoraMerger.Adapter> adapters, Path out,
                                 ModelSourceOptions options, boolean overwrite) throws IOException, UsageException {
        if (peft != null) {
            return convertPeft(src, peft, out, options, overwrite);
        }
        if (adapters.isEmpty()) {
            throw new UsageException("export-lora merges LoRA adapters: give at least one --lora <adapter> or"
                    + " --lora-scaled <adapter> <scale>, or --peft <dir> to convert the adapter alone");
        }
        return LoraMerger.merge(src, adapters, out, options, overwrite);
    }

    /** Writes the PEFT adapter directory {@code peft} as the GGUF adapter view of the base model. */
    private static Path convertPeft(ModelSource base, String peft, Path out, ModelSourceOptions options,
                                    boolean overwrite) throws IOException {
        SourceOrigin origin = new SourceOrigin(Paths.get(base.getFilePath()), base.format());
        try (ModelSource adapter = ModelSources.openAdapter(peft, origin, options)) {
            GgufWriter.write(adapter, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(overwrite));
        }
        return out;
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A finite float scale of {@code --lora-scaled}, naming the flag in the error. */
    private static float scaleValue(String flag, String v) throws UsageException {
        float s;
        try {
            s = Float.parseFloat(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number, not '" + v + "'");
        }
        if (!Float.isFinite(s)) {
            throw new UsageException(flag + " needs a finite scale, not '" + v + "'");
        }
        return s;
    }
}
