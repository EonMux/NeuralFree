package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Matcher
import java.util.regex.Pattern

/** Kernel manifest, source assembly and kernel-signature scanning of preprocessed dual-dialect sources. */
class GpuKernelSources {
    static final int MAX_BUFFERS = 12
    static final int MAX_SCALARS = 32
    /** Manifest tag of an OpenCL-only source. */
    static final String OPENCL_TAG = '[opencl]'
    /** Manifest tag of a header resolved against the shared root (KernelManifest.SHARED_TAG). */
    static final String SHARED_TAG = '[shared]'

    private static final Pattern LINE_MARKER = Pattern.compile('^#\\s*(?:line\\s+)?\\d+\\s+"((?:[^"\\\\]|\\\\.)*)".*$')
    private static final String ATTRIBUTES = '(?:__attribute__\\s*\\(\\(.*?\\)\\)\\s*)*'
    /** An OpenCL (__kernel) or CUDA (extern "C" __global__, as written or as expanded by nvcc -E) kernel definition. */
    private static final Pattern KERNEL = Pattern.compile(
            '(?:\\b__kernel\\b|\\bextern\\s+"C"\\s+(?:__global__|__attribute__\\s*\\(\\(\\s*global\\s*\\)\\)))\\s+'
                    + ATTRIBUTES + 'void\\s+' + ATTRIBUTES + '([A-Za-z_]\\w*)\\s*\\(', Pattern.DOTALL)
    private static final Set<String> QUALIFIERS = ['const', 'volatile', '__private', 'private',
                                                   'restrict', '__restrict', '__restrict__'] as Set
    private static final Map<String, String> SCALAR_KINDS = ['int': 'i', 'signed int': 'i', 'signed': 'i',
                                                             'uint': 'u', 'unsigned int': 'u', 'unsigned': 'u',
                                                             'float': 'f']

    /**
     * Reads root/manifest.txt: ordered entries [path, openClOnly, shared, file]. "[opencl] <path>" is an
     * OpenCL-only source under root; "[shared] <path>" is a header under sharedRoot (KernelManifest.SHARED_ROOT),
     * and every shared entry precedes the library's own entries, so manifest order puts shared headers first in
     * every assembled program and CUDA unit. path is the #line label; file is the resolved source. Every listed
     * file must exist, and a path may be listed once whatever its tag.
     */
    static List<Map> readManifest(File root, File sharedRoot) {
        File mf = new File(root, 'manifest.txt')
        if (!mf.isFile()) {
            throw new GradleException("kernel manifest not found: ${mf}")
        }
        List<Map> entries = []
        Set<String> seen = new HashSet<>()
        boolean libraryEntrySeen = false
        mf.readLines('UTF-8').eachWithIndex { String raw, int i ->
            String line = raw.trim()
            if (line.isEmpty() || line.startsWith('#')) {
                return
            }
            String where = "${mf}:${i + 1}"
            boolean openClOnly = false
            boolean shared = false
            if (line.startsWith(OPENCL_TAG)) {
                openClOnly = true
                line = line.substring(OPENCL_TAG.length()).trim()
            } else if (line.startsWith(SHARED_TAG)) {
                shared = true
                line = line.substring(SHARED_TAG.length()).trim()
            }
            if (line.startsWith('[')) {
                throw new GradleException("${where}: unknown or repeated tag in '${raw}'; an entry carries at most one of ${OPENCL_TAG} and ${SHARED_TAG}")
            }
            File base = shared ? sharedRoot : root
            if (line.isEmpty() || line.startsWith('/') || line.contains('..') || line.contains('\\')) {
                throw new GradleException("${where}: '${line}' must be a relative path inside ${base}")
            }
            if (!line.endsWith('.h') && !line.endsWith('.cl')) {
                throw new GradleException("${where}: '${line}' must be a .h or .cl file")
            }
            if (shared) {
                if (!line.endsWith('.h')) {
                    throw new GradleException("${where}: ${SHARED_TAG} entry '${line}' must be a header (.h); shared sources hold only helpers and macros")
                }
                if (libraryEntrySeen) {
                    throw new GradleException("${where}: ${SHARED_TAG} entry '${line}' follows a library entry; shared headers are listed first")
                }
            } else {
                libraryEntrySeen = true
            }
            if (!seen.add(line)) {
                throw new GradleException("${where}: '${line}' is listed twice")
            }
            File file = new File(base, line)
            if (!file.isFile()) {
                throw new GradleException("${where}: ${file} does not exist")
            }
            entries.add([path: line, openClOnly: openClOnly, shared: shared, file: file])
        }
        return entries
    }

    /** Concatenates the entries' files in order, each prefixed by #line 1 "<path>" (the runtime assembler's format). */
    static String concatenate(List<Map> entries) {
        StringBuilder sb = new StringBuilder()
        for (Map e : entries) {
            sb.append('#line 1 "').append(e.path).append('"\n')
            String text = ((File) e.file).getText('UTF-8')
            sb.append(text)
            if (!text.endsWith('\n')) {
                sb.append('\n')
            }
        }
        return sb.toString()
    }

    /** Module stem of a .cl path: "/" becomes "_" and the extension is dropped (recurrent/gdn.cl -> recurrent_gdn). */
    static String stem(String clPath) {
        return clPath.substring(0, clPath.length() - '.cl'.length()).replace('/', '_')
    }

    /**
     * Kernel definitions in preprocessed text, as [name, file, buffers, locals, kinds, badScalars, layout], where
     * file is the source named by the governing line marker and layout holds one character per parameter in
     * declaration order: B for a buffer, L for a local pointer, the scalar kind, or ? for an unsupported scalar.
     */
    static List<Map> scanKernels(String preprocessed) {
        StringBuilder body = new StringBuilder(preprocessed.length())
        List<Integer> lineStarts = []
        List<String> lineFiles = []
        String current = ''
        for (String line : preprocessed.split('\n')) {
            Matcher m = LINE_MARKER.matcher(line)
            if (m.matches()) {
                current = m.group(1)
                continue
            }
            if (line.startsWith('#')) {
                continue
            }
            lineStarts.add(body.length())
            lineFiles.add(current)
            body.append(line).append('\n')
        }
        List<Map> kernels = []
        Matcher k = KERNEL.matcher(body)
        while (k.find()) {
            String name = k.group(1)
            int open = k.end() - 1
            int close = matchingParen(body, open)
            if (close < 0) {
                throw new GradleException("unbalanced parameter list of kernel ${name}")
            }
            int idx = Collections.binarySearch(lineStarts, k.start())
            String file = lineFiles[idx >= 0 ? idx : -idx - 2]
            Map kernel = summarize(splitParams(body.substring(open + 1, close)))
            kernel.name = name
            kernel.file = file
            kernels.add(kernel)
            k.region(close, body.length())
        }
        return kernels
    }

    /** Rule violations of one kernel; OpenCL-only kernels are held to the argument-count limits only. */
    static List<String> violations(Map kernel, boolean openClOnly) {
        List<String> v = []
        if ((int) kernel.buffers > MAX_BUFFERS) {
            v.add("${kernel.buffers} buffer arguments (at most ${MAX_BUFFERS})")
        }
        int scalars = ((List) kernel.kinds).size() + ((List) kernel.badScalars).size()
        if (scalars > MAX_SCALARS) {
            v.add("${scalars} scalar arguments (at most ${MAX_SCALARS})")
        }
        if (!openClOnly) {
            if ((int) kernel.locals > 0) {
                v.add("${kernel.locals} __local pointer argument(s); library kernels use statically sized local memory")
            }
            for (Object bad : (List) kernel.badScalars) {
                v.add("scalar argument '${bad}' is not int, uint or float")
            }
        }
        return v
    }

    /** Fails on two definitions of one kernel name within a library (OpenCL builds each library as one program). */
    static void requireUniqueKernels(String libraryId, List<Map> kernels) {
        Map<String, List<String>> byName = kernels.groupBy { (String) it.name }.collectEntries { k, v -> [(k): v.collect { (String) it.file }] }
        Map<String, List<String>> dups = byName.findAll { k, v -> v.size() > 1 }
        if (!dups.isEmpty()) {
            throw new GradleException("${libraryId}: kernel names defined more than once: " + dups.collect { k, v -> "${k} in ${v}" }.join('; '))
        }
    }

    private static int matchingParen(CharSequence s, int open) {
        int depth = 0
        for (int i = open; i < s.length(); i++) {
            char c = s.charAt(i)
            if (c == '(' as char) {
                depth++
            } else if (c == ')' as char) {
                depth--
                if (depth == 0) {
                    return i
                }
            }
        }
        return -1
    }

    private static List<String> splitParams(String list) {
        List<String> params = []
        int depth = 0
        StringBuilder cur = new StringBuilder()
        for (int i = 0; i < list.length(); i++) {
            char c = list.charAt(i)
            if (c == '(' as char || c == '[' as char) {
                depth++
            } else if (c == ')' as char || c == ']' as char) {
                depth--
            }
            if (c == ',' as char && depth == 0) {
                params.add(cur.toString().trim())
                cur.setLength(0)
            } else {
                cur.append(c)
            }
        }
        params.add(cur.toString().trim())
        return params.findAll { !it.isEmpty() && it != 'void' }.collect { it.replaceAll('\\s+', ' ') }
    }

    private static Map summarize(List<String> params) {
        int buffers = 0
        int locals = 0
        List<String> kinds = []
        List<String> bad = []
        StringBuilder layout = new StringBuilder()
        for (String param : params) {
            String p = param.replaceAll('__attribute__\\s*\\(\\(.*?\\)\\)', ' ').trim()
            if (p.contains('*') || p.contains('[')) {
                if (p =~ /\b(?:__local|local)\b/) {
                    locals++
                    layout.append('L')
                } else {
                    buffers++
                    layout.append('B')
                }
                continue
            }
            List<String> tokens = p.split('\\s+').findAll { !QUALIFIERS.contains(it) }
            List<String> typeTokens = tokens.size() > 1 ? tokens.subList(0, tokens.size() - 1) : tokens
            String kind = SCALAR_KINDS.get(typeTokens.join(' '))
            if (kind != null) {
                kinds.add(kind)
                layout.append(kind)
            } else {
                bad.add(param)
                layout.append('?')
            }
        }
        return [buffers: buffers, locals: locals, kinds: kinds, badScalars: bad, layout: layout.toString()]
    }
}
