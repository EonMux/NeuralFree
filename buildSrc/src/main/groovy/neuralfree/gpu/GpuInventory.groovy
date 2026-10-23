package neuralfree.gpu

import org.gradle.api.GradleException

/**
 * A kernel library's checked-in inventory (kernels/<lib>/inventory.txt): its text format, the reader and
 * validator of the checked-in file, the coverage checks against the manifest and the shader variant tables,
 * and the description of how two inventory texts differ.
 */
class GpuInventory {

    /**
     * Inventory text of the library's portable kernels as scanned by GpuKernelSources.scanKernels, in the given
     * order: the library's header, then one kernel_name, source, buffers, scalars, kinds, layout row per kernel.
     */
    static String text(GpuKernelLibrary lib, List<Map> kernels) {
        return lib.inventoryHeader() + kernels.collect { Map k ->
            "${k.name}\t${k.file}\t${k.buffers}\t${((List) k.kinds).size()}\t${((List) k.kinds).join(',')}\t${k.layout}\n"
        }.join('')
    }

    /**
     * Reads and validates a library's checked-in inventory: name -> [file, buffers, kinds], in file order.
     * Throws naming the file and line of a malformed or duplicate entry, or when the file is missing or empty;
     * every message ends with regenerateCommand, the command that rewrites the file.
     */
    static Map<String, Map> read(GpuKernelLibrary lib, String regenerateCommand) {
        if (!lib.inventoryFile.isFile()) {
            throw new GradleException("${lib.inventoryPath} is missing; restore it from version control or regenerate it with: ${regenerateCommand}")
        }
        Map<String, Map> entries = new LinkedHashMap<>()
        lib.inventoryFile.readLines('UTF-8').eachWithIndex { String raw, int i ->
            String line = raw.replace('\r', '')
            if (line.trim().isEmpty() || line.startsWith('#')) {
                return
            }
            String where = "${lib.inventoryPath}:${i + 1}"
            String[] c = line.split('\t', -1)
            if (c.length != 6) {
                throw new GradleException("${where}: expected kernel_name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds<TAB>layout, got ${c.length} columns; regenerate it with: ${regenerateCommand}")
            }
            String name = c[0]
            if (!(name ==~ /[A-Za-z_]\w*/) || !(c[2] ==~ /\d+/) || !(c[3] ==~ /\d+/) || !(c[4] ==~ /(?:[iuf](?:,[iuf])*)?/)
                    || !(c[5] ==~ /[Biuf]*/)) {
                throw new GradleException("${where}: malformed entry '${line}'; regenerate it with: ${regenerateCommand}")
            }
            int buffers = Integer.parseInt(c[2])
            List<String> kinds = c[4].isEmpty() ? [] : c[4].split(',').toList()
            List<String> layoutScalars = c[5].toList().findAll { it != 'B' }
            if (kinds.size() != Integer.parseInt(c[3]) || c[5].count('B') != buffers || layoutScalars != kinds) {
                throw new GradleException("${where}: kernel ${name}: the counts, kinds and layout columns disagree; regenerate it with: ${regenerateCommand}")
            }
            if (entries.containsKey(name)) {
                throw new GradleException("${where}: kernel ${name} is listed twice; regenerate it with: ${regenerateCommand}")
            }
            entries[name] = [file: c[1], buffers: buffers, kinds: kinds]
        }
        if (entries.isEmpty()) {
            throw new GradleException("${lib.inventoryPath} lists no kernels; regenerate it with: ${regenerateCommand}")
        }
        return entries
    }

    /** Problems of a library's portable kernel-name set against its variants rows (exactly one row per kernel). */
    static List<String> variantCoverageProblems(GpuKernelLibrary lib, Set<String> kernelNames) {
        Set<String> rowNames = GpuShaderVariants.read(lib.variantsDir, lib.shaderRoot, lib.subgroupVariants).collect { (String) it.name } as Set
        List<String> withoutRow = (kernelNames - rowNames).sort()
        List<String> withoutKernel = (rowNames - kernelNames).sort()
        List<String> problems = []
        if (!withoutRow.isEmpty()) {
            problems.add("${withoutRow.size()} kernels have no ${lib.shaderRootPath}/variants row: ${withoutRow.join(', ')}")
        }
        if (!withoutKernel.isEmpty()) {
            problems.add("${withoutKernel.size()} ${lib.shaderRootPath}/variants rows name no portable manifest kernel: ${withoutKernel.join(', ')}")
        }
        return problems
    }

    /**
     * Problems of a library's checked-in inventory that need no preprocessor: every source must be an untagged
     * .cl entry of the manifest (shared entries resolved against sharedRoot), and the variant rows must cover it.
     */
    static List<String> coverageProblems(GpuKernelLibrary lib, Map<String, Map> inventory, File sharedRoot) {
        Set<String> portableUnits = lib.manifest(sharedRoot)
                .findAll { !it.openClOnly && ((String) it.path).endsWith('.cl') }.collect { (String) it.path } as Set
        List<String> problems = []
        inventory.each { String name, Map k ->
            if (!portableUnits.contains(k.file)) {
                problems.add("${name}: source ${k.file} is not an untagged .cl entry of ${lib.kernelRootPath}/manifest.txt")
            }
        }
        problems.addAll(variantCoverageProblems(lib, inventory.keySet()))
        return problems
    }

    /** What differs between two inventory texts: kernels added, removed and changed, or the comment lines. */
    static String difference(String checkedIn, String generated) {
        Closure<Map<String, String>> byName = { String text ->
            text.readLines().findAll { !it.trim().isEmpty() && !it.startsWith('#') }
                    .collectEntries { [(it.split('\t', 2)[0]): it] }
        }
        Map<String, String> was = byName(checkedIn)
        Map<String, String> now = byName(generated)
        Closure<String> show = { Collection<String> names ->
            List<String> sorted = new ArrayList<>(names).sort()
            sorted.take(10).join(', ') + (sorted.size() > 10 ? ", ... (${sorted.size()} in all)" : '')
        }
        List<String> parts = []
        Set<String> added = now.keySet() - was.keySet()
        Set<String> removed = was.keySet() - now.keySet()
        Set<String> changed = now.keySet().intersect(was.keySet()).findAll { now[it] != was[it] } as Set
        if (!added.isEmpty()) {
            parts.add("new kernels: ${show(added)}")
        }
        if (!removed.isEmpty()) {
            parts.add("kernels no longer in the sources: ${show(removed)}")
        }
        if (!changed.isEmpty()) {
            parts.add("changed source or arguments: ${show(changed)}")
        }
        return parts.isEmpty() ? 'the comment lines or line order differ' : parts.join('; ')
    }
}
