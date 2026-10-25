package neuralfree.hf

import org.gradle.api.GradleException

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.regex.Pattern
import java.util.stream.Collectors

/**
 * The generated llama.cpp converter tables under src/main/resources/hf (written by gradle/hfgen/hfgen.py through
 * updateHfTables): their file set and formats, and the validator that verifyHfTables runs on the checked-in
 * directory and updateHfTables runs on a freshly generated one before installing it.
 *
 * A table is UTF-8 with "\n" line ends: the header line "# <file> - Generated from llama.cpp <commit> by the
 * updateHfTables Gradle task; do not edit. Rows: <n>", a "#" line with the TAB-separated column names, then n rows.
 * Fields are TAB-separated, so the column count check also rejects a TAB inside a field or list item; "-" marks an
 * empty field and list fields are comma-separated. manifest.tsv lists every other file with its row or byte count
 * and its sha256, which keeps the copied templates and vocabularies byte-identical to the llama.cpp originals.
 * The version-control settings files (SETTINGS) are not resources: Gradle leaves them out of the jar and the
 * manifest does not list them.
 */
class HfTableCheck {
    static final String MANIFEST = 'manifest.tsv'

    /** Exact content of the version-control settings files hfgen.py writes beside the tables. */
    static final Map<String, String> SETTINGS = [
            '.gitignore'    : '# The built-in vocabularies are checked-in resources, unlike other GGUF files.\n!/vocab/*.gguf\n',
            '.gitattributes': '# Every file is kept byte-exact: manifest.tsv records its sha256.\n* -text\n',
    ].asImmutable()

    /** Column names of every table; the runtime loaders read these exact layouts. */
    static final Map<String, List<String>> COLUMNS = [
            'archs.tsv'               : ['hf_name', 'kind', 'class', 'source', 'gguf_arch', 'modify', 'vocab_writers'],
            'dynamic_archs.tsv'       : ['class', 'source', 'condition', 'gguf_arch'],
            'tensor_patterns.tsv'     : ['seq', 'scope', 'tensor', 'hf_template'],
            'tensor_names.tsv'        : ['tensor', 'gguf_template'],
            'arch_tensors.tsv'        : ['gguf_arch', 'tensor'],
            'arch_block_overrides.tsv': ['gguf_arch', 'tensor', 'hf_template', 'seq'],
            'gguf_keys.tsv'           : ['writer_method', 'key_template', 'value_type'],
            'hparam_aliases.tsv'      : ['gguf_key_template', 'hf_aliases', 'value_type', 'rule'],
            'chkhsh.tsv'              : ['sha256', 'pre_name', 'repo'],
            'filters.tsv'             : ['kind', 'value', 'replacement', 'source'],
            (MANIFEST)                : ['file', 'rows_or_bytes', 'sha256'],
    ].asImmutable()

    /** Comma-separated list columns per table. */
    static final Map<String, List<String>> LISTS = [
            'archs.tsv'         : ['vocab_writers'],
            'hparam_aliases.tsv': ['hf_aliases'],
    ].asImmutable()

    private static final String SCALAR = /(?:UINT8|INT8|UINT16|INT16|UINT32|INT32|FLOAT32|BOOL|STRING|UINT64|INT64|FLOAT64)/
    private static final String ARCH = /[a-z0-9][a-z0-9_.-]*/
    private static final String TENSOR = /[A-Z0-9_]+/
    private static final String PY_SOURCE = /[A-Za-z0-9_]+\.py:\d+/
    private static final String SHA256 = /[0-9a-f]{64}/
    private static final String FILTER_KINDS =
            /rename_suffix|rename_contains|strip_prefix|drop_prefix|drop_contains|skip_contains|skip_suffix|forced_f32/
    private static final String RULE = /(?:required|optional|always|reject|default=.+|local \w+|mirror [\w.]+)(?:; .+)?/
    private static final String HEADER_TEXT = 'Generated from llama.cpp <commit> by the updateHfTables Gradle task; do not edit.'

    /** Values a column must match, per table; columns without a rule only need to be non-empty. */
    static final Map<String, Map<String, String>> VALUES = [
            'archs.tsv'               : [kind: /TEXT|MMPROJ/, source: /-|${PY_SOURCE}/, gguf_arch: /-|${ARCH}/],
            'dynamic_archs.tsv'       : [source: PY_SOURCE, gguf_arch: ARCH],
            'tensor_patterns.tsv'     : [seq: /\d+/, scope: /global|block/, tensor: TENSOR],
            'tensor_names.tsv'        : [tensor: TENSOR],
            'arch_tensors.tsv'        : [gguf_arch: ARCH, tensor: TENSOR],
            'arch_block_overrides.tsv': [gguf_arch: ARCH, tensor: TENSOR, seq: /\d+/],
            'gguf_keys.tsv'           : [writer_method: /add_\w+/, value_type: /${SCALAR}|ARRAY:${SCALAR}|ANY/],
            'hparam_aliases.tsv'      : [value_type: /-|${SCALAR}|ARRAY:${SCALAR}|ANY/,
                                         rule      : RULE],
            'chkhsh.tsv'              : [sha256: SHA256],
            'filters.tsv'             : [kind  : FILTER_KINDS, source: PY_SOURCE],
            (MANIFEST)                : [rows_or_bytes: /\d{1,18} (?:rows|bytes)/, sha256: SHA256],
    ].asImmutable()

    private static final Pattern HEADER =
            ~/# (\S+) - Generated from llama\.cpp ([0-9a-f]{7,40}) by the updateHfTables Gradle task; do not edit\. Rows: (\d{1,9})/
    private static final Pattern RELATIVE_PATH = ~/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*/
    private static final Pattern COPIED = ~/templates\/[^\/]+\.jinja|vocab\/ggml-vocab-[^\/]+\.gguf|chktxt\.txt/

    /**
     * Validates a tables directory: the file set against the manifest, every table's header, column layout,
     * values and row count, and every file's size and sha256. Returns one summary line per file; throws a
     * GradleException naming the file and line of the first problem, ending with regenerateCommand.
     */
    static List<String> verify(File dir, String regenerateCommand) {
        Closure<GradleException> problem = { String where, String what ->
            new GradleException("${where}: ${what}\nRegenerate the tables with: ${regenerateCommand}")
        }
        if (!dir.isDirectory()) {
            throw problem(dir.path, 'missing; restore it from version control')
        }
        Map<String, Map> manifest = readManifest(dir, problem)
        SETTINGS.each { String rel, String content ->
            File f = new File(dir, rel)
            if (!f.isFile() || new String(f.bytes, StandardCharsets.UTF_8) != content) {
                throw problem(rel, 'is missing or differs from the settings hfgen.py writes')
            }
        }
        Set<String> onDisk = listFiles(dir) - [MANIFEST] - SETTINGS.keySet()
        List<String> missing = (manifest.keySet() - onDisk).toList().sort()
        List<String> extra = (onDisk - manifest.keySet()).toList().sort()
        if (!missing.isEmpty() || !extra.isEmpty()) {
            throw problem(MANIFEST, "does not match the directory: missing files ${missing}, unlisted files ${extra}")
        }
        List<String> absentTables = (COLUMNS.keySet() - [MANIFEST] - manifest.keySet()).toList().sort()
        if (!absentTables.isEmpty()) {
            throw problem(dir.path, "lacks the tables ${absentTables}")
        }
        if (!manifest.containsKey('chktxt.txt')) {
            throw problem(dir.path, 'lacks chktxt.txt')
        }
        String commit = null
        List<String> summary = []
        Set<String> knownTensors = [] as Set<String>
        Map<String, Set<String>> tensorRefs = [:]
        manifest.each { String rel, Map entry ->
            byte[] data = new File(dir, rel).bytes
            if (COLUMNS.containsKey(rel)) {
                Map table = readTable(rel, data, problem)
                List<List<String>> rows = (List<List<String>>) table.rows
                if (commit != null && table.commit != commit) {
                    throw problem(rel, "generated from llama.cpp ${table.commit}, the other tables from ${commit}")
                }
                commit = (String) table.commit
                if (entry.unit != 'rows' || entry.count != rows.size()) {
                    throw problem(MANIFEST, "${rel} has ${rows.size()} rows, the manifest says ${entry.count} ${entry.unit}")
                }
                if (rel == 'tensor_names.tsv') {
                    knownTensors = rows.collect { it[0] } as Set<String>
                }
                tensorRefs[rel] = tensorColumn(rel, rows)
                summary.add("${rel}: ${rows.size()} rows")
            } else {
                if (!(rel ==~ COPIED)) {
                    throw problem(rel, 'is neither a known table nor a copied template, vocabulary or chktxt.txt')
                }
                if (entry.unit != 'bytes' || entry.count != data.length) {
                    throw problem(MANIFEST, "${rel} has ${data.length} bytes, the manifest says ${entry.count} ${entry.unit}")
                }
                if (rel.endsWith('.gguf') && !(data.length >= 4 && new String(data, 0, 4, StandardCharsets.US_ASCII) == 'GGUF')) {
                    throw problem(rel, 'is not a GGUF file')
                }
                if (!rel.endsWith('.gguf')) {
                    decodeUtf8(rel, data, problem)
                }
                summary.add("${rel}: ${data.length} bytes")
            }
            String actual = sha256(data)
            if (actual != entry.sha256) {
                throw problem(rel, "sha256 ${actual} differs from the manifest's ${entry.sha256}; generated files must not be edited")
            }
        }
        tensorRefs.each { String rel, Set<String> tensors ->
            List<String> unknown = (tensors - knownTensors).toList().sort()
            if (!unknown.isEmpty()) {
                throw problem(rel, "names MODEL_TENSORs without a tensor_names.tsv row: ${unknown}")
            }
        }
        return summary
    }

    /**
     * The changes between two tables directories by their manifests, as "added ...; removed ...; changed ..." or
     * "no file changed". before may be absent; after must hold a valid manifest.
     */
    static String difference(File before, File after) {
        Closure<GradleException> problem = { String where, String what -> new GradleException("${where}: ${what}") }
        Map<String, Map> old = new File(before, MANIFEST).isFile() ? readManifest(before, problem) : [:]
        Map<String, Map> now = readManifest(after, problem)
        List<String> added = (now.keySet() - old.keySet()).toList().sort()
        List<String> removed = (old.keySet() - now.keySet()).toList().sort()
        List<String> changed = now.keySet().findAll { old.containsKey(it) && old[it].sha256 != now[it].sha256 }.toList().sort()
        if (added.isEmpty() && removed.isEmpty() && changed.isEmpty()) {
            return 'no file changed'
        }
        return [['added', added], ['removed', removed], ['changed', changed]]
                .findAll { !((List) it[1]).isEmpty() }
                .collect { "${it[0]} ${((List) it[1]).join(', ')}" }
                .join('; ')
    }

    private static Map<String, Map> readManifest(File dir, Closure<GradleException> problem) {
        File f = new File(dir, MANIFEST)
        if (!f.isFile()) {
            throw problem(MANIFEST, 'missing')
        }
        Map table = readTable(MANIFEST, f.bytes, problem)
        Map<String, Map> entries = new LinkedHashMap<>()
        ((List<List<String>>) table.rows).eachWithIndex { List<String> row, int i ->
            String rel = row[0]
            String where = "${MANIFEST}:${i + 3}"
            if (!(rel ==~ RELATIVE_PATH) || rel.split('/').any { it == '.' || it == '..' }
                    || rel == MANIFEST || SETTINGS.containsKey(rel)) {
                throw problem(where, "invalid file path '${rel}'")
            }
            if (entries.containsKey(rel)) {
                throw problem(where, "${rel} is listed twice")
            }
            String[] count = row[1].split(' ')
            entries[rel] = [count: Long.parseLong(count[0]), unit: count[1], sha256: row[2]]
        }
        return entries
    }

    /** Parses and validates one table: [commit, rows]. */
    private static Map readTable(String rel, byte[] data, Closure<GradleException> problem) {
        String text = decodeUtf8(rel, data, problem)
        if (text.contains('\r')) {
            throw problem(rel, 'contains a carriage return; tables use "\\n" line ends')
        }
        if (!text.endsWith('\n')) {
            throw problem(rel, 'does not end with a line feed')
        }
        List<String> lines = text.substring(0, text.length() - 1).split('\n', -1).toList()
        def header = HEADER.matcher(lines[0])
        if (!header.matches() || header.group(1) != rel) {
            throw problem("${rel}:1", "expected '# ${rel} - ${HEADER_TEXT} Rows: <n>'")
        }
        List<String> columns = COLUMNS[rel]
        if (lines.size() < 2 || lines[1] != '# ' + columns.join('\t')) {
            throw problem("${rel}:2", "expected the column line '# ${columns.join('<TAB>')}'")
        }
        Map<String, Pattern> rules = (VALUES[rel] ?: [:]).collectEntries { k, v -> [(k): Pattern.compile((String) v)] }
        List<String> lists = LISTS[rel] ?: []
        List<List<String>> rows = []
        lines.subList(2, lines.size()).eachWithIndex { String line, int i ->
            String where = "${rel}:${i + 3}"
            List<String> fields = line.split('\t', -1).toList()
            if (fields.size() != columns.size()) {
                throw problem(where, "${fields.size()} fields for the ${columns.size()} columns ${columns}")
            }
            if (fields[0].startsWith('#')) {
                throw problem(where, 'a row must not start with #')
            }
            fields.eachWithIndex { String value, int c ->
                String column = columns[c]
                if (value.isEmpty() || value != value.trim()) {
                    throw problem(where, "column ${column} is empty or padded with spaces")
                }
                Pattern rule = rules[column]
                if (rule != null && !rule.matcher(value).matches()) {
                    throw problem(where, "column ${column} value '${value}' does not match ${rule.pattern()}")
                }
                if (lists.contains(column) && value != '-' && value.split(',', -1).any { it.isEmpty() || it != it.trim() }) {
                    throw problem(where, "list column ${column} has an empty or padded item: '${value}'")
                }
            }
            rows.add(fields)
        }
        int declared = Integer.parseInt(header.group(3))
        if (declared != rows.size()) {
            throw problem("${rel}:1", "the header says Rows: ${declared} but the file has ${rows.size()} rows")
        }
        checkSequences(rel, rows, problem)
        return [commit: header.group(2), rows: rows]
    }

    /** seq columns count from 0 in file order (per arch in arch_block_overrides.tsv). */
    private static void checkSequences(String rel, List<List<String>> rows, Closure<GradleException> problem) {
        if (rel == 'tensor_patterns.tsv') {
            rows.eachWithIndex { List<String> row, int i ->
                if (row[0] != String.valueOf(i)) {
                    throw problem("${rel}:${i + 3}", "seq ${row[0]} should be ${i}")
                }
            }
        } else if (rel == 'arch_block_overrides.tsv') {
            Map<String, Integer> next = [:]
            rows.eachWithIndex { List<String> row, int i ->
                int expected = next.getOrDefault(row[0], 0)
                if (row[3] != String.valueOf(expected)) {
                    throw problem("${rel}:${i + 3}", "seq ${row[3]} of ${row[0]} should be ${expected}")
                }
                next[row[0]] = expected + 1
            }
        }
    }

    /**
     * The MODEL_TENSOR names a table maps, which tensor_names.tsv must name. arch_tensors.tsv is exempt: it copies
     * MODEL_TENSORS, which lists a few tensors without a TENSOR_NAMES entry that the converter never maps.
     */
    private static Set<String> tensorColumn(String rel, List<List<String>> rows) {
        switch (rel) {
            case 'tensor_patterns.tsv': return rows.collect { it[2] } as Set<String>
            case 'arch_block_overrides.tsv': return rows.collect { it[1] } as Set<String>
            case 'filters.tsv': return rows.findAll { it[0] == 'forced_f32' }.collect { it[1] } as Set<String>
            default: return [] as Set<String>
        }
    }

    private static String decodeUtf8(String rel, byte[] data, Closure<GradleException> problem) {
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(data)).toString()
        } catch (CharacterCodingException e) {
            throw problem(rel, "is not valid UTF-8 (${e.message})")
        }
    }

    /** Every regular file under dir, as '/'-separated relative paths. */
    static Set<String> listFiles(File dir) {
        Path base = dir.toPath()
        return Files.walk(base).withCloseable { stream ->
            stream.filter { Path p -> Files.isRegularFile(p) }
                    .map { Path p -> base.relativize(p).toString().replace('\\', '/') }
                    .collect(Collectors.toCollection { new TreeSet<String>() })
        }
    }

    static String sha256(byte[] data) {
        return MessageDigest.getInstance('SHA-256').digest(data).encodeHex().toString()
    }
}
