package com.eonmux.neuralfree.gguf.testutil;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Where the opt-in real-model benchmarks and probes find their GGUF files: the directory named by
 * {@code -Dneuralfree.test.models}, else by the {@code NEURALFREE_TEST_MODELS} environment variable, else
 * {@code models} under the user's home directory. Each test skips when its file is absent.
 */
public final class RealModels {

    /** The system property naming the model root. */
    public static final String PROPERTY = "neuralfree.test.models";
    /** The environment variable naming the model root when the property is unset. */
    public static final String ENV = "NEURALFREE_TEST_MODELS";

    private RealModels() {
    }

    /** The model root directory. */
    public static Path root() {
        String root = System.getProperty(PROPERTY);
        if (root == null || root.isEmpty()) {
            root = System.getenv(ENV);
        }
        return (root == null || root.isEmpty()) ? Paths.get(System.getProperty("user.home"), "models") : Paths.get(root);
    }

    /** The path of {@code relative} under the model root, as a string. */
    public static String path(String relative) {
        return root().resolve(relative).toString();
    }
}
