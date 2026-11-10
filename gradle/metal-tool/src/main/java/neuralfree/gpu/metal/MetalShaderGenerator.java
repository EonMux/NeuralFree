package neuralfree.gpu.metal;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Build tool that turns a kernel library's Vulkan SPIR-V modules into Metal Shading Language.
 *
 * <p>Reads the library's {@code index.properties} (the buffer counts, read-only flags,
 * push-constant scalar kinds and subgroup variants the Vulkan build recorded), translates every
 * kernel's module with {@link SpirvCrossMsl}, validates the generated binding layout, writes
 * {@code <name>.metal} for each kernel and writes the Metal {@code index.properties} in the format
 * {@code MetalKernelLibrary} reads. The subgroup module {@code <name>__sg.spv} is translated when
 * the Vulkan index lists one; Metal SIMD groups are always available on Apple GPUs, so the kernel
 * keeps its subgroup reductions as {@code simd_*} calls. A kernel the index pins to one subgroup width
 * ({@code <name>.sgsize}, the Vulkan twin of an OpenCL kernel written for Intel's 16-lane subgroups) is skipped:
 * Apple SIMD groups have a fixed width of 32 lanes, so the module's lane arithmetic cannot run there, and such a
 * kernel is outside the portable inventory the Metal library must cover.</p>
 *
 * <p>Usage: {@code MetalShaderGenerator --spv <spirvDir> --out <metalDir>}.</p>
 */
public final class MetalShaderGenerator {

    private static final Pattern BUFFER_PARAMETER = Pattern.compile("\\[\\[buffer\\((\\d+)\\)\\]\\]");
    private static final String INDEX_FILE = "index.properties";

    private MetalShaderGenerator() {
    }

    public static void main(String[] args) {
        Map<String, String> options = parseArguments(args);
        Path spvDir = Paths.get(require(options, "--spv")).toAbsolutePath().normalize();
        Path outDir = Paths.get(require(options, "--out")).toAbsolutePath().normalize();
        try {
            int kernels = generate(spvDir, outDir);
            System.out.println("metal: " + kernels + " kernels from " + spvDir);
        } catch (IOException e) {
            throw new UncheckedIOException("Metal shader generation from " + spvDir + " failed", e);
        }
    }

    /**
     * Generates the Metal sources of one library.
     *
     * @return the number of kernels written
     * @throws IOException              when an input or output cannot be read or written
     * @throws IllegalArgumentException when the index or a generated kernel breaks the layout contract
     */
    static int generate(Path spvDir, Path outDir) throws IOException {
        SortedMap<String, Kernel> kernels = readIndex(spvDir.resolve(INDEX_FILE));
        if (kernels.isEmpty()) {
            throw new IllegalArgumentException(spvDir.resolve(INDEX_FILE) + " lists no kernels");
        }
        Files.createDirectories(outDir);
        SpirvCrossMsl translator = new SpirvCrossMsl();
        int threads = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<String>> jobs = new ArrayList<>(kernels.size());
            for (Kernel kernel : kernels.values()) {
                jobs.add(pool.submit(new Callable<String>() {
                    @Override
                    public String call() throws IOException {
                        translate(translator, spvDir, outDir, kernel);
                        return kernel.name;
                    }
                }));
            }
            for (int i = 0; i < jobs.size(); i++) {
                try {
                    jobs.get(i).get();
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof RuntimeException) {
                        throw (RuntimeException) cause;
                    }
                    if (cause instanceof IOException) {
                        throw (IOException) cause;
                    }
                    throw new IllegalStateException("Metal shader generation failed", cause);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Metal shader generation was interrupted", e);
                }
            }
        } finally {
            pool.shutdownNow();
        }
        Files.write(outDir.resolve(INDEX_FILE), indexText(kernels).getBytes(StandardCharsets.UTF_8));
        return kernels.size();
    }

    private static void translate(SpirvCrossMsl translator, Path spvDir, Path outDir, Kernel kernel)
            throws IOException {
        Path module = spvDir.resolve(kernel.name + (kernel.subgroupVariant ? "__sg" : "") + ".spv");
        if (!Files.isRegularFile(module)) {
            throw new IllegalArgumentException("kernel " + kernel.name + ": expected SPIR-V module " + module
                    + " is missing");
        }
        boolean push = !kernel.pushKinds.isEmpty();
        String msl = translator.compile(Files.readAllBytes(module), kernel.name, kernel.buffers, push);
        SpecConstants constants = SpecConstants.parse(msl, kernel.name);
        if (constants.isEmpty()) {
            // Every library shader declares the local-size spec constants; if none reaches the MSL,
            // the module's constants took a form the Metal backend cannot supply.
            throw new IllegalArgumentException("kernel " + kernel.name + ": the generated MSL carries no"
                    + " specialisation constants (local size ids 0-2), so a dispatch could not supply its"
                    + " work-group size");
        }
        validateLayout(msl, kernel, push);
        write(outDir.resolve(kernel.name + ".metal"), msl);
    }

    /**
     * Checks that the generated kernel binds every storage buffer at its binding index and the
     * push constants at {@code buffers}: the Metal backend binds the k-th buffer argument at MSL
     * buffer index {@code k} and the packed scalars at {@code buffers}.
     */
    private static void validateLayout(String msl, Kernel kernel, boolean push) {
        String declaration = kernelDeclaration(msl, kernel.name);
        SortedMap<Integer, Integer> counts = new TreeMap<>();
        Matcher m = BUFFER_PARAMETER.matcher(declaration);
        while (m.find()) {
            int index = Integer.parseInt(m.group(1));
            Integer previous = counts.get(index);
            counts.put(index, previous == null ? 1 : previous + 1);
        }
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            int index = e.getKey();
            if (e.getValue() > 1) {
                throw new IllegalArgumentException("kernel " + kernel.name + ": MSL buffer index " + index
                        + " is bound " + e.getValue() + " times");
            }
            if (index < 0 || index > kernel.buffers) {
                throw new IllegalArgumentException("kernel " + kernel.name + ": MSL buffer index " + index
                        + " is outside 0.." + kernel.buffers + " (buffers " + kernel.buffers
                        + (push ? ", push constants at " + kernel.buffers : "") + ")");
            }
            if (index == kernel.buffers && !push) {
                throw new IllegalArgumentException("kernel " + kernel.name + ": MSL buffer index "
                        + index + " is bound but the kernel has no push constants");
            }
        }
    }

    private static String kernelDeclaration(String msl, String kernelName) {
        String marker = "kernel void " + kernelName + "(";
        for (String line : msl.split("\n")) {
            if (line.startsWith(marker)) {
                return line;
            }
        }
        throw new IllegalArgumentException("kernel " + kernelName + ": the generated MSL has no '"
                + marker + "' declaration");
    }

    private static SortedMap<String, Kernel> readIndex(Path indexFile) throws IOException {
        if (!Files.isRegularFile(indexFile)) {
            throw new IllegalArgumentException("Metal shader generation needs " + indexFile
                    + "; the Vulkan shaders were not compiled");
        }
        Properties props = new Properties();
        try (Reader reader = Files.newBufferedReader(indexFile, StandardCharsets.UTF_8)) {
            props.load(reader);
        }
        SortedMap<String, Kernel> kernels = new TreeMap<>();
        for (String key : props.stringPropertyNames()) {
            if (!key.endsWith(".buffers")) {
                continue;
            }
            String name = key.substring(0, key.length() - ".buffers".length());
            if (props.getProperty(name + ".sgsize") != null) {
                continue;
            }
            String buffers = props.getProperty(key);
            String readOnly = props.getProperty(name + ".ro");
            String push = props.getProperty(name + ".push");
            if (readOnly == null || push == null) {
                throw new IllegalArgumentException(indexFile + ": kernel " + name + " lacks "
                        + (readOnly == null ? name + ".ro" : name + ".push"));
            }
            kernels.put(name, new Kernel(name, parseCount(name, buffers), readOnly, push,
                    "1".equals(props.getProperty(name + ".sg"))));
        }
        return kernels;
    }

    private static int parseCount(String name, String count) {
        try {
            int value = Integer.parseInt(count.trim());
            if (value < 0) {
                throw new NumberFormatException("negative");
            }
            return value;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("kernel " + name + ": '" + count
                    + ".buffers' is not a buffer count");
        }
    }

    private static String indexText(SortedMap<String, Kernel> kernels) {
        StringBuilder text = new StringBuilder();
        text.append("# Metal kernel signatures generated from SPIR-V by SPIRV-Cross: buffers, read-only flags,\n")
                .append("# push-constant scalar kinds and subgroup variant, with one <name>.metal per kernel.\n");
        for (Kernel kernel : kernels.values()) {
            text.append(kernel.name).append(".buffers=").append(kernel.buffers).append('\n');
            text.append(kernel.name).append(".ro=").append(kernel.readOnly).append('\n');
            text.append(kernel.name).append(".push=").append(kernel.pushKinds).append('\n');
            text.append(kernel.name).append(".sg=").append(kernel.subgroupVariant ? 1 : 0).append('\n');
        }
        return text.toString();
    }

    private static void write(Path file, String text) throws IOException {
        Files.write(file, text.getBytes(StandardCharsets.UTF_8));
    }

    private static Map<String, String> parseArguments(String[] args) {
        Map<String, String> options = new TreeMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length || !args[i].startsWith("--")) {
                throw new IllegalArgumentException("expected --name value pairs, got '" + args[i] + "'");
            }
            if (options.put(args[i], args[i + 1]) != null) {
                throw new IllegalArgumentException("duplicate option " + args[i]);
            }
        }
        Set<String> known = new java.util.TreeSet<>(java.util.Arrays.asList("--spv", "--out"));
        for (String option : options.keySet()) {
            if (!known.contains(option)) {
                throw new IllegalArgumentException("unknown option " + option + "; expected " + known);
            }
        }
        return options;
    }

    private static String require(Map<String, String> options, String name) {
        String value = options.get(name);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("missing required option " + name);
        }
        return value;
    }

    /** One kernel of the Vulkan index. */
    private static final class Kernel {
        final String name;
        final int buffers;
        final String readOnly;
        final String pushKinds;
        final boolean subgroupVariant;

        Kernel(String name, int buffers, String readOnly, String pushKinds, boolean subgroupVariant) {
            this.name = name;
            this.buffers = buffers;
            this.readOnly = readOnly;
            this.pushKinds = pushKinds;
            this.subgroupVariant = subgroupVariant;
        }
    }
}
