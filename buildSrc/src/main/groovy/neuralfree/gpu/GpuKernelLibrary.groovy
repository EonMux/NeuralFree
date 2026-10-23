package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Pattern

/** One kernel library of the build: source roots, generated classpath directories and per-library options. */
class GpuKernelLibrary {
    private static final Pattern ID = Pattern.compile('[a-z][a-z0-9_]*')
    private static final Pattern SHADER_ROOT = Pattern.compile('shaders/[a-z][a-z0-9_]*')

    /** Library id as registered in KernelLibraries, and its upper-case label for generated comments and messages. */
    final String id, label
    /** Classpath dirs: sources with manifest and inventory; CUDA modules and index; GLSL with variants and SPIR-V. */
    final String kernelRootPath, cudaDir, shaderRootPath
    /** The shader root's directory name directly under shaders/; inventoryPath is inventoryFile for messages. */
    final String shaderDirName, inventoryPath
    final File kernelRoot, shaderRoot, variantsDir, inventoryFile
    /** nvcc math-mode flags of the library's modules. */
    final List<String> cudaFlags
    /** NF_FA_SMEM_FLOATS the CUDA modules are compiled with and build.properties records; null when unused. */
    final Integer faSmemFloats
    /** subgroupVariants: rows may carry sg, and OpenCL is also checked with USE_SUBGROUPS. compareKinds: verifyGpuKernels
     *  compares the GLSL push-constant kinds (float or integer) with the .cl kinds. */
    final boolean subgroupVariants, compareKinds

    /** spec: id, kernelRoot, cudaDir, shaderRoot, cudaFlags, subgroupVariants, compareKinds; optional faSmemFloats. */
    GpuKernelLibrary(Map spec, File resources, File projectDir) {
        id = String.valueOf(spec.id)
        if (!ID.matcher(id).matches() || [spec.kernelRoot, spec.cudaDir, spec.shaderRoot, spec.cudaFlags].contains(null)) {
            throw new GradleException("GPU kernel library ${spec}: the id must match ${ID.pattern()} and every root and cudaFlags must be set")
        }
        label = id.toUpperCase(Locale.ROOT)
        kernelRootPath = spec.kernelRoot.toString()
        cudaDir = spec.cudaDir.toString()
        shaderRootPath = spec.shaderRoot.toString()
        if (!kernelRootPath.startsWith('kernels/') || kernelRootPath == 'kernels/common' || kernelRootPath.contains('..')) {
            throw new GradleException("GPU kernel library ${id}: kernelRoot '${kernelRootPath}' must be a directory under kernels/ other than the shared kernels/common")
        }
        if (!cudaDir.startsWith(kernelRootPath + '/') || cudaDir.contains('..')) {
            throw new GradleException("GPU kernel library ${id}: cudaDir '${cudaDir}' must be a directory under ${kernelRootPath}")
        }
        if (!SHADER_ROOT.matcher(shaderRootPath).matches() || shaderRootPath in ['shaders/common', 'shaders/test']) {
            throw new GradleException("GPU kernel library ${id}: shaderRoot '${shaderRootPath}' must be one directory under shaders/ other than common and test")
        }
        shaderDirName = shaderRootPath.substring('shaders/'.length())
        kernelRoot = new File(resources, kernelRootPath)
        shaderRoot = new File(resources, shaderRootPath)
        variantsDir = new File(shaderRoot, 'variants')
        inventoryFile = new File(kernelRoot, 'inventory.txt')
        inventoryPath = GpuBuildTools.relative(projectDir, inventoryFile)
        cudaFlags = Collections.unmodifiableList(((List<?>) spec.cudaFlags).collect { it.toString() })
        faSmemFloats = spec.faSmemFloats == null ? null : Integer.valueOf(spec.faSmemFloats.toString())
        subgroupVariants = (boolean) spec.subgroupVariants
        compareKinds = (boolean) spec.compareKinds
    }

    /** Preprocessor defines of every CUDA unit of the library (nvcc -E and the module compile). */
    List<String> cudaDefines() { faSmemFloats == null ? [] : ['-DNF_FA_SMEM_FLOATS=' + faSmemFloats] }

    /** The library's manifest entries (GpuKernelSources.readManifest), shared entries resolved against sharedRoot. */
    List<Map> manifest(File sharedRoot) { GpuKernelSources.readManifest(kernelRoot, sharedRoot) }

    /** Comment lines that open the generated inventory. */
    String inventoryHeader() {
        return "# ${label} kernel inventory: every kernel of the untagged ${kernelRootPath}/manifest.txt entries.\n" +
                '# Generated from the kernel sources by the updateKernelInventory Gradle task and checked by inventoryCheck; do not edit.\n' +
                '# kernel_name<TAB>source<TAB>buffers<TAB>scalars<TAB>scalar kinds (i,u,f)<TAB>parameter layout (B buffer, else the scalar kind)\n'
    }

    /** Every option that changes the library's build outputs, for up-to-date checks. */
    String fingerprint() {
        "${id}[${kernelRootPath} ${cudaDir} ${shaderRootPath} cuda=${cudaFlags.join(' ')} ${cudaDefines().join(' ')} sg=${subgroupVariants} kinds=${compareKinds}]"
    }

    @Override
    String toString() { id }
}
