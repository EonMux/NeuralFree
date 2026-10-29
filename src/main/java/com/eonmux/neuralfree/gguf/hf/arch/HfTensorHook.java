package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;

import java.io.IOException;

/**
 * A converter class's {@code modify_tensors}: called once per checkpoint tensor, in index order, for every tensor no
 * earlier call consumed. It plans the GGUF tensors the checkpoint tensor becomes, consumes the tensors it merges, or
 * drops it.
 */
@FunctionalInterface
public interface HfTensorHook {

    /**
     * {@code ModelBase.modify_tensors}: the tensor under its mapped name (base.py:845-846, 905). The converter's opt-in
     * expert and QKV fusions (base.py:848-903) are not part of the view. The base class filter's unconditional
     * {@code language_model.} strip (base.py:591-592) happens before the mapping.
     */
    HfTensorHook DEFAULT = (ctx, hfName, src, out) -> {
        if (!out.isConsumed(hfName)) {
            out.add(ctx.names().mapOrThrow(hfName.replace("language_model.", "")), src);
        }
    };

    /**
     * Plans {@code hfName}.
     *
     * @throws IOException when tensor data the hook reads cannot be read
     */
    void apply(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException;
}
