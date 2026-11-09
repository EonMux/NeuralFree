package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;

/** Branch norm tensor names: {@link NormResolver} for main-stack layers, plain names for other stacks. */
final class NormNames {
    private final ModelSpec spec;
    private NormResolver resolver;

    NormNames(ModelSpec spec) {
        this.spec = spec;
    }

    private boolean resolvable(LayerSpec l) {
        return TensorNames.STACK_BLK.equals(l.stack()) && l.weightLayer() < spec.hparams().nLayerAll();
    }

    private NormResolver resolver() {
        if (resolver == null) {
            resolver = new NormResolver(spec.hparams(), spec.tensors());
        }
        return resolver;
    }

    /** FFN input norm tensor of a layer; null when the FFN input is not normed by a tensor. */
    TensorRef ffnNorm(LayerSpec l) {
        if (resolvable(l)) {
            return resolver().ffnNormTensor(l.weightLayer());
        }
        NormSpec ns = l.norm();
        if (ns.ffnInputIsPostAttnNorm()) {
            return layer(l, TensorNames.ATTN_POST_NORM);
        }
        if (ns.attnNorm2() && l.residual().ffnUsesAttnNorm2()) {
            return layer(l, TensorNames.ATTN_NORM_2);
        }
        return ns.ffnNorm() ? layer(l, TensorNames.FFN_NORM) : null;
    }

    /** Post-attention norm tensor of a layer; null when it has none. */
    TensorRef postAttnNorm(LayerSpec l) {
        if (resolvable(l)) {
            return resolver().postAttnNormTensor(l.weightLayer());
        }
        return layer(l, TensorNames.ATTN_POST_NORM);
    }

    /** Post-FFN norm tensor of a layer; null when it has none. */
    TensorRef postFfnNorm(LayerSpec l) {
        if (resolvable(l)) {
            return resolver().postFfnNormTensor(l.weightLayer());
        }
        return layer(l, TensorNames.FFN_POST_NORM);
    }

    private TensorRef layer(LayerSpec l, String base) {
        return spec.tensors().layer(l.stack(), l.weightLayer(), TensorNames.weight(base));
    }
}
