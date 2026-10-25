package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;

/**
 * The hyperparameter changes a converter class makes in {@code __init__} before anything is converted (renamed or
 * derived {@code hparams} entries). Runs once, after the architecture is resolved; every later hook sees only the
 * fixed configuration ({@link HfLoadContext#config()}).
 */
@FunctionalInterface
public interface HfConfigFixup {

    /** No change. */
    HfConfigFixup NONE = cfg -> cfg;

    /**
     * The fixed configuration, derived with {@link HfConfig#withFlat}.
     *
     * @throws com.eonmux.neuralfree.gguf.hparams.GgufFormatException when the configuration lacks what the class
     *                                                                requires
     */
    HfConfig apply(HfConfig cfg);
}
