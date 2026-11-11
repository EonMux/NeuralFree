/* step_inputs.cl - LLM_DecodeStepInputs, the per-step inputs of a one-token decode step written on the device.
 * Requires: common/dialect.h, common/config.h */
/*
 * LLM_DecodeStepInputs(i row, i pos, i rope0, i rope1, i rope2, i rope3, W<int> cell_idx, W<int> kv_pos,
 *                      W<int> q_pos, W<int> rope_pos)
 *     cell_idx[0] = row, kv_pos[row] = pos, q_pos[0] = pos, rope_pos[0..3] = (rope0, rope1, rope2, rope3)
 *     The one-token step of a single-group contiguous KV memory (gguf.memory.gpu.DeviceStepInputs): the cache row
 *     the step's K/V store writes, the row's entry in the visibility mirror kv_pos, the query position and the four
 *     M-RoPE sections of the rope position, which host uploads would otherwise copy one buffer at a time. The
 *     values ride in the launch's arguments, so the step needs no host-to-device copy.
 *     Dispatch: global = {1}, local = {1}. One work-item; no barriers.
 * Java: gguf.gpu.forward.GpuDecodeStepWriter; host twin: DeviceStepInputs.upload(pos, 1).
 */

NF_KERNEL void LLM_DecodeStepInputs(
    const int row,
    const int pos,
    const int rope0,
    const int rope1,
    const int rope2,
    const int rope3,
    NF_GLOBAL int* NF_RESTRICT cell_idx,
    NF_GLOBAL int* NF_RESTRICT kv_pos,
    NF_GLOBAL int* NF_RESTRICT q_pos,
    NF_GLOBAL int* NF_RESTRICT rope_pos)
{
    if (get_global_id(0) != 0) return;
    cell_idx[0] = row;
    kv_pos[row] = pos;
    q_pos[0] = pos;
    rope_pos[0] = rope0;
    rope_pos[1] = rope1;
    rope_pos[2] = rope2;
    rope_pos[3] = rope3;
}
