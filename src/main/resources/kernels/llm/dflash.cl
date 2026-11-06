/* DFlash2 block convolution and selector. Requires: common/dialect.h, common/config.h.
 * The convolution uses conv_proj rows [2*K*(E/group)] and conv_base [2*K*E],
 * with causal taps inside this block only. One work-item per output element.
 * The selector transforms/scatters logits, scores the current top-k candidates conditioned
 * on the preceding selected id, then selects the first maximum and its top-k probability.
 * No logits, hidden rows or candidate tables leave the device.
 */
NF_KERNEL void LLM_DflashConv(
    const int hidden, const int taps, const int group_size, const int n_rows, const int side,
    NF_GLOBAL const float* x, NF_GLOBAL const float* dynamic, NF_GLOBAL const float* base,
    NF_GLOBAL float* out) {
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)n_rows * hidden) return;
    const int t = (int)(idx / hidden);
    const int c = (int)(idx % hidden);
    const int groups = hidden / group_size;
    float sum = 0.0f;
    for (int tap = 0; tap < min(taps, t + 1); ++tap) {
        const nf_i64 di = (nf_i64)t * (2 * taps * groups) + groups * (tap + taps * side) + c / group_size;
        const nf_i64 bi = (nf_i64)hidden * (tap + taps * side) + c;
        const float w = dynamic[di] + base[bi];
        sum += w * x[(nf_i64)(t - tap) * hidden + c];
    }
    out[idx] = sum;
}

NF_KERNEL void LLM_Dflash2Logits(
    const int n_rows, const int draft_vocab, const int target_vocab, const float scale, const float cap,
    NF_GLOBAL const int* inverse, NF_GLOBAL const float* logits, NF_GLOBAL float* out) {
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)n_rows * target_vocab) return;
    const int src = inverse[idx % target_vocab];
    float v = -NF_INF;
    if (src >= 0) {
        v = logits[(idx / target_vocab) * draft_vocab + src];
        if (scale != 0.0f) v *= scale;
        if (cap > 0.0f) v = tanh(v * (1.0f / cap)) * cap;
    }
    out[idx] = v;
}

NF_KERNEL void LLM_Dflash2Score(
    const int row, const int k, const int rank, const int anchor,
    NF_GLOBAL const float* top_val, NF_GLOBAL const int* top_idx, NF_GLOBAL const float* gate,
    NF_GLOBAL const float* prev, NF_GLOBAL const float* next, NF_GLOBAL const int* ids,
    NF_GLOBAL float* scores) {
    const int j = (int)get_global_id(0);
    if (j >= k) return;
    const int pred = row == 1 ? anchor : ids[row - 1];
    const int succ = top_idx[(nf_i64)row * k + j];
    float dot = 0.0f;
    for (int r = 0; r < rank; ++r) {
        const float cond = prev[(nf_i64)pred * rank + r] * gate[(nf_i64)row * rank + r];
        dot += next[(nf_i64)succ * rank + r] * cond;
    }
    scores[j] = dot + top_val[(nf_i64)row * k + j];
}

NF_KERNEL void LLM_Dflash2Pick(
    const int row, const int k, const int probability,
    NF_GLOBAL const int* top_idx, NF_GLOBAL const float* scores,
    NF_GLOBAL int* ids, NF_GLOBAL float* probs) {
    if (get_global_id(0) != 0) return;
    int best = 0;
    for (int j = 1; j < k; ++j) if (scores[j] > scores[best]) best = j;
    ids[row] = top_idx[(nf_i64)row * k + best];
    if (probability) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) sum += exp(scores[j] - scores[best]);
        probs[row] = 1.0f / sum;
    }
}
