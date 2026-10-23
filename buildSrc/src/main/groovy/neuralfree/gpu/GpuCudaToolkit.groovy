package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Matcher

/** Queries of an nvcc installation: its release and the SM architectures it can compile for. */
class GpuCudaToolkit {

    /**
     * Selects the SM list: the archs requested by the comma-separated archsProperty (-Pneuralfree.cuda.archs;
     * "75" or "sm_75" entries) that nvcc supports, ascending. Returns [selected, dropped]; throws when the
     * property is malformed or empty, or when nvcc supports none of the requested archs.
     */
    static Map selectArchs(File nvcc, String archsProperty) {
        List<Integer> requested = archsProperty.split(',').collect { it.trim() }.findAll { !it.isEmpty() }.collect { String a ->
            String n = a.startsWith('sm_') ? a.substring(3) : a
            if (!(n ==~ /\d+/)) {
                throw new GradleException("-Pneuralfree.cuda.archs must list SM numbers such as 75,86,120; got '${archsProperty}'")
            }
            Integer.valueOf(n)
        }.unique().sort()
        if (requested.isEmpty()) {
            throw new GradleException("-Pneuralfree.cuda.archs is empty")
        }
        String listing = GpuBuildTools.runChecked([nvcc, '--list-gpu-code'], null, 'nvcc --list-gpu-code')
        Set<Integer> supported = (listing =~ /\bsm_(\d+)\b/).collect { Integer.valueOf(((List) it)[1]) } as Set
        List<Integer> selected = requested.findAll { supported.contains(it) }
        List<Integer> dropped = requested.findAll { !supported.contains(it) }
        if (selected.isEmpty()) {
            throw new GradleException("none of the CUDA archs ${requested} is supported by ${nvcc} (supports ${supported.sort()})")
        }
        return [selected: selected, dropped: dropped]
    }

    /**
     * The nvcc release as [full (e.g. 12.4.131), toolkit (major.minor), driverMin (the CUDA driver version
     * number of that release: major * 1000 + minor * 10)]; throws when nvcc --version names no release.
     */
    static Map version(File nvcc) {
        String out = GpuBuildTools.runChecked([nvcc, '--version'], null, 'nvcc --version')
        Matcher m = out =~ /release (\d+)\.(\d+), V([\w.]+)/
        if (!m.find()) {
            throw new GradleException("cannot read the CUDA release from nvcc --version:\n${out}")
        }
        int major = Integer.parseInt(m.group(1))
        int minor = Integer.parseInt(m.group(2))
        return [full: m.group(3), toolkit: "${major}.${minor}".toString(), driverMin: major * 1000 + minor * 10]
    }
}
