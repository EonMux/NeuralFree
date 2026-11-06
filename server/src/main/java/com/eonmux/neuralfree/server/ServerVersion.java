package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

/**
 * The server version ({@code GET /version}): the root project's version, which the build expands into the
 * {@code neuralfree-server-version.properties} resource.
 */
final class ServerVersion {

    /** The classpath resource the build writes the version into. */
    static final String RESOURCE = "/neuralfree-server-version.properties";

    /** The version of this build. */
    static final String VALUE = read();

    private ServerVersion() {
    }

    private static String read() {
        try (InputStream in = ServerVersion.class.getResourceAsStream(RESOURCE)) {
            if (in == null) {
                throw new IllegalStateException("the build did not package " + RESOURCE);
            }
            Properties properties = new Properties();
            properties.load(in);
            String version = properties.getProperty("version");
            if (version == null || version.isBlank() || version.contains("${")) {
                throw new IllegalStateException(RESOURCE + " carries no expanded version: " + version);
            }
            return version.strip();
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + RESOURCE + " failed", e);
        }
    }
}
