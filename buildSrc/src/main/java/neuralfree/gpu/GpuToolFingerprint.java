package neuralfree.gpu;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Path-free identity of an external build tool for task up-to-date checks and the Gradle build cache: the
 * executable name plus the tool's {@code --version} output. The executable's directory is deliberately left
 * out, so a checkout at another path (or the same toolchain installed elsewhere) produces the same task
 * fingerprint and reuses cached kernel artifacts. Plain Java without the Gradle API, like
 * {@link GpuHipToolkit}, so the module's test suite can verify it.
 */
public final class GpuToolFingerprint {

    /** Fingerprint of a tool the build did not find. */
    public static final String ABSENT = "absent";

    private GpuToolFingerprint() {
    }

    /**
     * Fingerprint of {@code tool}, run as {@code <tool> --version}; {@link #ABSENT} when it is null. Only the
     * executable name and the version text enter the result, never the directory the tool was found in.
     *
     * @throws IllegalStateException when the tool cannot be started or the wait is interrupted
     */
    public static String of(File tool) {
        if (tool == null) {
            return ABSENT;
        }
        String output;
        try {
            Process process = new ProcessBuilder(tool.getAbsolutePath(), "--version")
                    .redirectErrorStream(true)
                    .start();
            output = readFully(process.getInputStream());
            process.waitFor();
        } catch (IOException e) {
            throw new IllegalStateException("cannot run " + tool.getName() + " --version: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while running " + tool.getName() + " --version", e);
        }
        return of(tool.getName(), output);
    }

    /**
     * Fingerprint of a tool with the given executable name and version output. Absolute paths in the output
     * (clang and hipcc print {@code InstalledDir: /usr/bin}) are replaced by {@value #PATH_PLACEHOLDER}, so the
     * same toolchain installed elsewhere yields the same identity. A tool with no output is identified by its
     * name alone, so an implementation that prints no version still gets a stable identity.
     *
     * @throws IllegalArgumentException when the executable name is null or empty
     */
    public static String of(String executableName, String versionOutput) {
        if (executableName == null || executableName.isEmpty()) {
            throw new IllegalArgumentException("a tool fingerprint needs the executable name");
        }
        String version = withoutAbsolutePaths(versionOutput == null ? "" : versionOutput.trim());
        return version.isEmpty() ? executableName : executableName + ' ' + version;
    }

    /** Stands in for an absolute path inside a version output. */
    public static final String PATH_PLACEHOLDER = "<path>";

    /** Replaces every whitespace-delimited absolute POSIX or Windows path with a stable placeholder. */
    private static String withoutAbsolutePaths(String text) {
        StringBuilder out = new StringBuilder(text.length());
        int i = 0;
        while (i < text.length()) {
            char c = text.charAt(i);
            if (Character.isWhitespace(c)) {
                out.append(c);
                i++;
                continue;
            }
            int end = i;
            while (end < text.length() && !Character.isWhitespace(text.charAt(end))) {
                end++;
            }
            String token = text.substring(i, end);
            out.append(isAbsolutePath(token) ? PATH_PLACEHOLDER : token);
            i = end;
        }
        return out.toString();
    }

    private static boolean isAbsolutePath(String token) {
        return token.startsWith("/")
                || (token.length() >= 3 && Character.isLetter(token.charAt(0))
                    && token.charAt(1) == ':' && (token.charAt(2) == '\\' || token.charAt(2) == '/'));
    }

    private static String readFully(InputStream in) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        int read;
        while ((read = in.read(chunk)) >= 0) {
            buffer.write(chunk, 0, read);
        }
        return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
    }
}
