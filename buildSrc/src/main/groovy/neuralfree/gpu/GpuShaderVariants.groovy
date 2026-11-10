package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Pattern

/** Reader of a library's shader variant tables shaders/<lib>/variants/*.tsv. */
class GpuShaderVariants {
    private static final Pattern NAME = Pattern.compile('[A-Za-z_][A-Za-z0-9_]*')
    private static final Pattern DEFINE = Pattern.compile('[A-Za-z_][A-Za-z0-9_]*(?:=\\S+)?')
    private static final Set<String> FLAGS = ['sg', 'vk13', 'vulkan', 'noopt'] as Set
    /**
     * The pinned-subgroup flags sg<N>: the row is a device twin of an [opencl] subgroup form, built as its only module
     * with USE_SUBGROUPS=1 and NF_SG_ROW_SIZE=N, whose pipelines require subgroups of exactly N lanes.
     */
    private static final Pattern PINNED = Pattern.compile('sg(8|16|32|64)')

    /**
     * All rows as [name, source, defines, sg, sgSize, vk13, vulkan, noopt, location], in file-name then line order
     * (sgSize is the N of an sg<N> flag, 0 without one); any malformed row throws, and so does an sg or sg<N> flag
     * when the library builds no subgroup variants (allowSubgroups false) or a row that carries both. A vulkan flag
     * marks a Vulkan-only kernel (no OpenCL/CUDA/HIP form, so no inventory entry, and no Metal translation): the
     * Vulkan build records it in the shader index with {@code <name>.vulkan=1}, and the inventory and coverage checks
     * exempt it, as they exempt a [cuda] kernel's rows everywhere but CUDA. A vk13 flag builds the module for Vulkan
     * 1.3, and a noopt flag ships the module as glslc emits it, without the spirv-opt pass.
     */
    static List<Map> read(File variantsDir, File shaderRoot, boolean allowSubgroups) {
        List<Map> rows = []
        Map<String, String> where = [:]
        File[] tables = variantsDir.listFiles({ File f -> f.isFile() && f.name.endsWith('.tsv') } as FileFilter)
        for (File tsv : (tables == null ? [] : tables.toList().sort { it.name })) {
            tsv.readLines('UTF-8').eachWithIndex { String raw, int i ->
                String line = raw.replace('\r', '')
                if (line.trim().isEmpty() || line.trim().startsWith('#')) {
                    return
                }
                String loc = "${tsv.name}:${i + 1}"
                String[] cols = line.split('\t', -1)
                if (cols.length < 2 || cols.length > 4) {
                    throw new GradleException("${loc}: expected kernel_name<TAB>source.comp[<TAB>defines[<TAB>flags]], got '${raw}'")
                }
                String name = cols[0].trim()
                String source = cols[1].trim()
                if (!NAME.matcher(name).matches() || name.endsWith('__sg')) {
                    throw new GradleException("${loc}: '${name}' is not a valid kernel name (identifier, no __sg suffix)")
                }
                if (where.containsKey(name)) {
                    throw new GradleException("${loc}: kernel ${name} is already defined at ${where[name]}")
                }
                if (!source.endsWith('.comp') || source.startsWith('/') || source.contains('..')
                        || !new File(shaderRoot, source).isFile()) {
                    throw new GradleException("${loc}: source '${source}' is not a .comp file under ${shaderRoot}")
                }
                List<String> defines = cols.length > 2 ? cols[2].trim().split('\\s+').findAll { !it.isEmpty() } : []
                for (String d : defines) {
                    if (!DEFINE.matcher(d).matches()) {
                        throw new GradleException("${loc}: define '${d}' is not NAME or NAME=VALUE")
                    }
                    if (d == 'USE_SUBGROUPS' || d.startsWith('USE_SUBGROUPS=')) {
                        throw new GradleException("${loc}: USE_SUBGROUPS is set by the sg flag, not by a define")
                    }
                    if (d == 'NF_SG_ROW_SIZE' || d.startsWith('NF_SG_ROW_SIZE=')) {
                        throw new GradleException("${loc}: NF_SG_ROW_SIZE is set by an sg<N> flag, not by a define")
                    }
                }
                List<String> flags = cols.length > 3 ? cols[3].split(',').collect { it.trim() }.findAll { !it.isEmpty() } : []
                int sgSize = 0
                for (String f : flags) {
                    boolean pinned = PINNED.matcher(f).matches()
                    if (!FLAGS.contains(f) && !pinned) {
                        throw new GradleException("${loc}: unknown flag '${f}' (known: ${FLAGS.join(', ')}, sg8, sg16, sg32, sg64)")
                    }
                    if (!allowSubgroups && (f == 'sg' || pinned)) {
                        throw new GradleException("${loc}: flag ${f} on ${name}, but this library builds no subgroup variants")
                    }
                    if (pinned) {
                        if (sgSize != 0) {
                            throw new GradleException("${loc}: ${name} pins more than one subgroup size")
                        }
                        sgSize = Integer.parseInt(f.substring(2))
                    }
                }
                if (sgSize != 0 && flags.contains('sg')) {
                    throw new GradleException("${loc}: ${name} carries sg and sg${sgSize}; a pinned row has no __sg twin")
                }
                rows.add([name: name, source: source, defines: defines, sg: flags.contains('sg'), sgSize: sgSize,
                          vk13: flags.contains('vk13'), vulkan: flags.contains('vulkan'), noopt: flags.contains('noopt'),
                          location: loc])
                where[name] = loc
            }
        }
        return rows
    }
}
