package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;

import java.util.Objects;

/**
 * Chooses the NeuralFree architecture of a Hugging Face checkpoint: the converter class's {@code model_arch}, or the
 * runtime choice of the classes that set {@code self.model_arch} from the configuration or the tensors
 * ({@code dynamic_archs.tsv}). Runs before any other hook, on the configuration as loaded.
 */
@FunctionalInterface
public interface HfArchResolver {

    /**
     * The architecture of the checkpoint.
     *
     * @throws com.eonmux.neuralfree.gguf.hparams.GgufFormatException when the configuration selects no supported
     *                                                                variant
     */
    Arch resolve(HfConfig cfg, HfTensorIndex index);

    /**
     * A resolver that always answers {@code arch}.
     *
     * @throws IllegalArgumentException for {@link Arch#UNKNOWN}
     */
    static HfArchResolver of(Arch arch) {
        Objects.requireNonNull(arch, "arch");
        if (arch == Arch.UNKNOWN) {
            throw new IllegalArgumentException("a fixed architecture resolver needs a known architecture");
        }
        return (cfg, index) -> arch;
    }
}
