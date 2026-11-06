package com.eonmux.neuralfree.cli;

/**
 * The NeuralFree version the {@code --version} flag and {@link CollectEnv} print: the {@code
 * neuralfree.version} system property first, then the jar manifest's {@code Implementation-Version}
 * (the {@code neuralfree} start scripts and {@code java -jar} run the packaged jar), then the
 * development fallback the classpath runs print. One class so every surface reports the same string.
 */
public final class Version {

    /** The system property overriding the manifest, set by the Gradle run tasks. */
    public static final String PROPERTY = "neuralfree.version";

    /** The value printed when neither the property nor the manifest names a version. */
    public static final String DEVELOPMENT = "development";

    private Version() {
    }

    /** The version string of this run; never null or empty. */
    public static String current() {
        return resolve(System.getProperty(PROPERTY),
                Launcher.class.getPackage() == null ? null : Launcher.class.getPackage().getImplementationVersion());
    }

    /**
     * The first source that names a version: the property, then the manifest implementation version,
     * then {@link #DEVELOPMENT}.
     */
    static String resolve(String property, String manifest) {
        if (property != null && !property.trim().isEmpty()) {
            return property.trim();
        }
        if (manifest != null && !manifest.trim().isEmpty()) {
            return manifest.trim();
        }
        return DEVELOPMENT;
    }
}
