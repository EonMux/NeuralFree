package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Pattern

/** Reader of a library's shader variant tables shaders/<lib>/variants/*.tsv. */
class GpuShaderVariants {
    private static final Pattern NAME = Pattern.compile('[A-Za-z_][A-Za-z0-9_]*')
    private static final Pattern DEFINE = Pattern.compile('[A-Za-z_][A-Za-z0-9_]*(?:=\\S+)?')
    private static final Set<String> FLAGS = ['sg'] as Set

    /**
     * All rows as [name, source, defines, sg, location], in file-name then line order; any malformed row throws,
     * and so does an sg flag when the library builds no subgroup variants (allowSubgroups false).
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
                }
                List<String> flags = cols.length > 3 ? cols[3].split(',').collect { it.trim() }.findAll { !it.isEmpty() } : []
                for (String f : flags) {
                    if (!FLAGS.contains(f)) {
                        throw new GradleException("${loc}: unknown flag '${f}' (known: ${FLAGS.join(', ')})")
                    }
                    if (f == 'sg' && !allowSubgroups) {
                        throw new GradleException("${loc}: flag sg on ${name}, but this library builds no subgroup variants")
                    }
                }
                rows.add([name: name, source: source, defines: defines, sg: flags.contains('sg'), location: loc])
                where[name] = loc
            }
        }
        return rows
    }
}
