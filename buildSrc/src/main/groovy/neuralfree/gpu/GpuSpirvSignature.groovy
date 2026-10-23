package neuralfree.gpu

import org.gradle.api.GradleException

import java.util.regex.Matcher
import java.util.regex.Pattern

/** Kernel argument signature of an unoptimized SPIR-V module, read from spirv-dis --raw-id output. */
class GpuSpirvSignature {
    private static final Pattern DECORATE = Pattern.compile('^\\s*OpDecorate\\s+%(\\w+)\\s+(\\w+)(.*)$')
    private static final Pattern MEMBER_DECORATE = Pattern.compile('^\\s*OpMemberDecorate\\s+%(\\w+)\\s+(\\d+)\\s+(\\w+)(.*)$')
    private static final Pattern RESULT = Pattern.compile('^\\s*%(\\w+)\\s*=\\s*(Op\\w+)(.*)$')
    /** Spec ids of a library shader: 0-2 local size (required), 3 NF_FA_SMEM_FLOATS (LLM flash kernels). */
    private static final Set<Integer> REQUIRED_SPEC_IDS = [0, 1, 2] as Set
    private static final Set<Integer> ALLOWED_SPEC_IDS = [0, 1, 2, 3] as Set

    /**
     * Returns [buffers, ro, push] in the index.properties encoding: buffer count, one 0/1 read-only flag
     * per binding, comma-separated scalar kinds. Throws naming the kernel when the module breaks the
     * argument convention (set 0, contiguous storage-buffer bindings, one push block of 4-byte scalars,
     * spec ids 0-2 present and nothing beyond 3).
     */
    static Map parse(String disassembly, String kernel) {
        Map<String, List<List<String>>> decorations = [:]
        Map<String, Map<Integer, Set<String>>> memberDecorations = [:]
        Map<String, Map<Integer, Integer>> memberOffsets = [:]
        Map<String, List<String>> defs = [:]
        for (String line : disassembly.readLines()) {
            Matcher m = DECORATE.matcher(line)
            if (m.matches()) {
                decorations.computeIfAbsent(m.group(1), { [] }).add([m.group(2)] + tokens(m.group(3)))
                continue
            }
            m = MEMBER_DECORATE.matcher(line)
            if (m.matches()) {
                int member = Integer.parseInt(m.group(2))
                if (m.group(3) == 'Offset') {
                    memberOffsets.computeIfAbsent(m.group(1), { [:] }).put(member, Integer.parseInt(tokens(m.group(4))[0]))
                } else {
                    memberDecorations.computeIfAbsent(m.group(1), { [:] }).computeIfAbsent(member, { new HashSet<String>() }).add(m.group(3))
                }
                continue
            }
            m = RESULT.matcher(line)
            if (m.matches()) {
                defs[m.group(1)] = [m.group(2)] + tokens(m.group(3))
            }
        }
        Closure<Integer> decoInt = { String id, String name ->
            List<String> d = decorations.get(id)?.find { it[0] == name }
            d == null ? null : Integer.valueOf(d[1])
        }
        Closure<Boolean> hasDeco = { String id, String name -> decorations.get(id)?.any { it[0] == name } ?: false }

        SortedMap<Integer, Boolean> bindingReadOnly = new TreeMap<>()
        List<String> pushVariables = []
        defs.each { String id, List<String> variable ->
            if (variable[0] != 'OpVariable') {
                return
            }
            String storage = variable[2]
            if (storage == 'PushConstant') {
                pushVariables.add(id)
                return
            }
            Integer binding = decoInt(id, 'Binding')
            if (binding == null) {
                return
            }
            Integer set = decoInt(id, 'DescriptorSet')
            if (set == null || set != 0) {
                throw new GradleException("${kernel}: binding ${binding} is in descriptor set ${set}; library kernels use set 0 only")
            }
            String pointee = defs.get(variable[1])?.get(2)
            List<String> struct = pointee == null ? null : defs.get(pointee)
            if (struct == null || struct[0] != 'OpTypeStruct') {
                throw new GradleException("${kernel}: binding ${binding} is not a buffer block (descriptor arrays, images and samplers are not supported)")
            }
            if (storage != 'StorageBuffer' && !(storage == 'Uniform' && hasDeco(pointee, 'BufferBlock'))) {
                throw new GradleException("${kernel}: binding ${binding} uses storage class ${storage}; library kernels bind storage buffers only")
            }
            int members = struct.size() - 1
            boolean readOnly = hasDeco(id, 'NonWritable') || (members > 0 && (0..<members).every {
                memberDecorations.get(pointee)?.get(it)?.contains('NonWritable') ?: false
            })
            Boolean previous = bindingReadOnly.get(binding)
            bindingReadOnly.put(binding, previous == null ? readOnly : (previous && readOnly))
        }
        List<Integer> bindings = new ArrayList<>(bindingReadOnly.keySet())
        if (bindings != (bindings.isEmpty() ? [] : (0..<bindings.size()).toList())) {
            throw new GradleException("${kernel}: bindings ${bindings} are not contiguous from 0; buffer argument b must use binding b")
        }

        if (pushVariables.size() > 1) {
            throw new GradleException("${kernel}: ${pushVariables.size()} push-constant blocks; declare exactly one")
        }
        List<String> kinds = []
        if (pushVariables.size() == 1) {
            String struct = defs.get(defs.get(pushVariables[0])[1])?.get(2)
            List<String> structDef = struct == null ? null : defs.get(struct)
            if (structDef == null || structDef[0] != 'OpTypeStruct') {
                throw new GradleException("${kernel}: the push-constant block is not a struct")
            }
            for (int i = 1; i < structDef.size(); i++) {
                List<String> type = defs.get(structDef[i])
                String kind = null
                if (type != null && type[0] == 'OpTypeInt' && type[1] == '32') {
                    kind = type[2] == '1' ? 'i' : 'u'
                } else if (type != null && type[0] == 'OpTypeFloat' && type[1] == '32') {
                    kind = 'f'
                }
                if (kind == null) {
                    throw new GradleException("${kernel}: push-constant member ${i - 1} has type ${type}; only 32-bit int, uint and float scalars are allowed")
                }
                Integer offset = memberOffsets.get(struct)?.get(i - 1)
                if (offset == null || offset != 4 * (i - 1)) {
                    throw new GradleException("${kernel}: push-constant member ${i - 1} is at offset ${offset}, expected ${4 * (i - 1)} (scalars are packed in argument order)")
                }
                kinds.add(kind)
            }
        }

        Set<Integer> specIds = new TreeSet<>()
        decorations.each { String id, List<List<String>> list ->
            list.findAll { it[0] == 'SpecId' }.each { specIds.add(Integer.valueOf(it[1])) }
        }
        if (!specIds.containsAll(REQUIRED_SPEC_IDS)) {
            throw new GradleException("${kernel}: spec ids ${specIds} lack the local size ids 0-2; use layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in")
        }
        if (!ALLOWED_SPEC_IDS.containsAll(specIds)) {
            throw new GradleException("${kernel}: spec ids ${specIds}; only 0-2 (local size) and 3 (NF_FA_SMEM_FLOATS) are allowed")
        }
        return [buffers: bindings.size(),
                ro     : bindingReadOnly.values().collect { it ? '1' : '0' }.join(''),
                push   : kinds.join(',')]
    }

    private static List<String> tokens(String s) {
        String t = s.trim()
        return t.isEmpty() ? [] : t.split('\\s+').collect { it.startsWith('%') ? it.substring(1) : it }
    }
}
