package neuralfree.gpu

import org.gradle.api.GradleException
import org.gradle.api.logging.Logger

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.regex.Pattern
import java.util.stream.Collectors

/** Tool lookup, process execution, parallel jobs and file helpers. */
class GpuBuildTools {
    static final boolean WINDOWS = System.getProperty('os.name', '').toLowerCase(Locale.ROOT).startsWith('windows')
    static final int CORES = Runtime.runtime.availableProcessors()
    private static final int MAX_REPORTED_LINES = 200

    /**
     * Resolves an executable: the explicit path when given (it must exist), else the first match in
     * {@code dirs} and then PATH. Returns null when the tool is absent.
     */
    static File find(String explicitPath, String propertyName, String exe, List<String> dirs) {
        if (explicitPath != null && !explicitPath.trim().isEmpty()) {
            File f = new File(explicitPath.trim())
            if (!f.isFile() || !f.canExecute()) {
                throw new GradleException("-P${propertyName}=${explicitPath} does not name an executable file")
            }
            return f
        }
        List<String> candidates = new ArrayList<>(dirs.findAll { it != null && !it.isEmpty() })
        String path = System.getenv('PATH')
        if (path != null) {
            candidates.addAll(path.split(Pattern.quote(File.pathSeparator)).toList())
        }
        List<String> names = WINDOWS ? [exe + '.exe', exe] : [exe]
        for (String dir : candidates) {
            for (String name : names) {
                File f = new File(dir, name)
                if (f.isFile() && f.canExecute()) {
                    return f
                }
            }
        }
        return null
    }

    /** Identity of a tool for up-to-date checks: its path and version output, or "absent". */
    static String fingerprint(File tool) {
        if (tool == null) {
            return 'absent'
        }
        Map r = run([tool.absolutePath, '--version'], null)
        return tool.absolutePath + ' ' + ((String) r.output).trim()
    }

    /** Runs a command to completion; returns [exit, output (stdout and stderr), command]. */
    static Map run(List<?> command, File workDir) {
        List<String> cmd = command.collect { String.valueOf(it instanceof File ? ((File) it).absolutePath : it) }
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true)
        if (workDir != null) {
            pb.directory(workDir)
        }
        Process p
        try {
            p = pb.start()
        } catch (IOException e) {
            throw new GradleException("cannot start ${cmd[0]}: ${e.message}", e)
        }
        String out = p.inputStream.getText('UTF-8')
        int exit = p.waitFor()
        return [exit: exit, output: out, command: cmd.join(' ')]
    }

    /** Runs a command and returns its output; a non-zero exit throws with the command and its output. */
    static String runChecked(List<?> command, File workDir, String what) {
        Map r = run(command, workDir)
        if (r.exit != 0) {
            throw new GradleException("${what} failed (exit ${r.exit})\n  ${r.command}\n${clip((String) r.output)}")
        }
        return (String) r.output
    }

    static String clip(String text) {
        List<String> lines = text.readLines()
        if (lines.size() <= MAX_REPORTED_LINES) {
            return text
        }
        return lines.subList(0, MAX_REPORTED_LINES).join('\n') + "\n... (${lines.size() - MAX_REPORTED_LINES} more lines)"
    }

    /**
     * Runs the jobs on a fixed pool; every failure is collected and reported together. invokeAll is used
     * because a Groovy closure is both Runnable and Callable, so submit() would drop its result.
     */
    static <T> List<T> parallel(List<Callable<T>> jobs, int threads, String what) {
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Math.min(threads, jobs.size())))
        try {
            List<Future<T>> futures = pool.invokeAll(jobs)
            List<T> results = []
            List<String> failures = []
            for (Future<T> f : futures) {
                try {
                    results.add(f.get())
                } catch (ExecutionException e) {
                    Throwable cause = e.cause != null ? e.cause : e
                    failures.add(cause.message != null ? cause.message : cause.toString())
                    results.add(null)
                }
            }
            if (!failures.isEmpty()) {
                throw new GradleException("${what}: ${failures.size()} of ${jobs.size()} jobs failed\n\n" + failures.join('\n\n'))
            }
            return results
        } finally {
            pool.shutdownNow()
        }
    }

    static void resetDir(File dir) {
        if (dir.exists() && !dir.deleteDir()) {
            throw new GradleException("cannot delete ${dir}")
        }
        if (!dir.mkdirs()) {
            throw new GradleException("cannot create ${dir}")
        }
    }

    static void writeText(File f, String text) {
        f.parentFile.mkdirs()
        f.setText(text, 'UTF-8')
    }

    static void copy(File from, File to) {
        to.parentFile.mkdirs()
        Files.copy(from.toPath(), to.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }

    /** Files under root with the extension, skipping the given top-level directories; sorted by path. */
    static List<File> listFiles(File root, String extension, Collection<String> excludedTopDirs) {
        if (!root.isDirectory()) {
            return []
        }
        Path base = root.toPath()
        List<Path> paths = Files.walk(base).withCloseable { stream ->
            stream.filter { Path p -> Files.isRegularFile(p) && p.fileName.toString().endsWith(extension) }
                  .filter { Path p -> !excludedTopDirs.contains(base.relativize(p).getName(0).toString()) || base.relativize(p).nameCount == 1 }
                  .collect(Collectors.toList())
        }
        return paths.collect { it.toFile() }.sort { it.path }
    }

    static String relative(File root, File f) {
        return root.toPath().relativize(f.toPath()).toString().replace('\\', '/')
    }

    /** Warns, or throws when the build requires the tool. */
    static void missingTool(Logger logger, boolean required, String requireProperty, String message) {
        if (required) {
            throw new GradleException(message + " (-P${requireProperty}=true)")
        }
        logger.warn('WARNING: ' + message)
    }

    /** Parses a boolean project property: absent is false, "" (bare -Pname) and "true" are true. */
    static boolean flag(Object value, String name) {
        if (value == null) {
            return false
        }
        String v = value.toString().trim()
        if (v.isEmpty() || v.equalsIgnoreCase('true')) {
            return true
        }
        if (v.equalsIgnoreCase('false')) {
            return false
        }
        throw new GradleException("-P${name} must be true or false, got '${v}'")
    }
}
