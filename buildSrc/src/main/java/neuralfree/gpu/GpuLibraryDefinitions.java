package neuralfree.gpu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The definition of every GPU kernel library the kernel build compiles: its roots, the nvcc math mode of its CUDA
 * units ({@code cudaFlags}), the clang one of its HIP code objects ({@code hipFlags}) and its shader options. The
 * build script hands these maps to {@code GpuKernelLibrary}, and the module's test suite reads the same maps, so the
 * compile options a test checks are the ones the kernels are built with. Plain Java without the Gradle API, like
 * {@link GpuHipToolkit}.
 *
 * <p>The {@code train} library keeps Java float semantics: no contraction into fused multiply-adds
 * ({@code -fmad=false} for nvcc, {@code -ffp-contract=off} for clang), IEEE division and square root and no flush to
 * zero. The {@code llm} library uses the fast math mode.</p>
 */
public final class GpuLibraryDefinitions {

    /** The HIP options of the train library: training math is not contracted, as its CUDA build keeps it. */
    public static final List<String> TRAIN_HIP_FLAGS = Collections.singletonList("-ffp-contract=off");

    /** The nvcc math mode of the train library. */
    public static final List<String> TRAIN_CUDA_FLAGS = Collections.unmodifiableList(
            Arrays.asList("-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"));

    /** The nvcc math mode of the llm library. */
    public static final List<String> LLM_CUDA_FLAGS = Collections.unmodifiableList(
            Arrays.asList("-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-fmad=true"));

    /** The llm CUDA modules that must compile without a stack frame: register-resident pipelines by design. */
    public static final List<String> LLM_STACK_FREE_CUDA_MODULES = Collections.unmodifiableList(Arrays.asList(
            "matmul_mmq", "matmul_mmq_stream", "matmul_fp8", "quant_mmq_q8_1", "quant_mmq_warp"));

    private GpuLibraryDefinitions() {
    }

    /**
     * The library definitions in build order. {@code faSmemFloats} is the llm library's flash-attention shared-memory
     * budget, which adds {@code -DNF_FA_SMEM_FLOATS} to its CUDA units.
     *
     * @throws IllegalArgumentException when {@code faSmemFloats} is not positive
     */
    public static List<Map<String, Object>> definitions(int faSmemFloats) {
        if (faSmemFloats <= 0) {
            throw new IllegalArgumentException("faSmemFloats must be positive, got " + faSmemFloats);
        }
        Map<String, Object> llm = new LinkedHashMap<>();
        llm.put("id", "llm");
        llm.put("kernelRoot", "kernels/llm");
        llm.put("cudaDir", "kernels/llm/cuda");
        llm.put("shaderRoot", "shaders/llm");
        llm.put("cudaFlags", LLM_CUDA_FLAGS);
        llm.put("faSmemFloats", faSmemFloats);
        llm.put("hipFlags", Collections.<String>emptyList());
        llm.put("subgroupVariants", true);
        llm.put("compareKinds", true);
        llm.put("stackFreeCudaModules", LLM_STACK_FREE_CUDA_MODULES);

        Map<String, Object> train = new LinkedHashMap<>();
        train.put("id", "train");
        train.put("kernelRoot", "kernels/train");
        train.put("cudaDir", "kernels/train/cuda");
        train.put("shaderRoot", "shaders/train");
        train.put("cudaFlags", TRAIN_CUDA_FLAGS);
        train.put("hipFlags", TRAIN_HIP_FLAGS);
        train.put("subgroupVariants", false);
        train.put("compareKinds", true);

        List<Map<String, Object>> all = new ArrayList<>();
        all.add(Collections.unmodifiableMap(llm));
        all.add(Collections.unmodifiableMap(train));
        return Collections.unmodifiableList(all);
    }

    /**
     * The definition of library {@code id}.
     *
     * @throws IllegalArgumentException when no library has that id
     */
    public static Map<String, Object> definition(String id, int faSmemFloats) {
        for (Map<String, Object> d : definitions(faSmemFloats)) {
            if (d.get("id").equals(id)) {
                return d;
            }
        }
        throw new IllegalArgumentException("no GPU kernel library '" + id + "'");
    }
}
