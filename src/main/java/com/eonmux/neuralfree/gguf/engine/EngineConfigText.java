package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.chat.tools.ToolStrict;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;

import java.util.Arrays;
import java.util.Locale;

/**
 * The {@link EngineConfig#toString()} rendering: every configured setting named in one line, settings left at
 * their default left out, llama.cpp's flag spellings for the draft placement keys.
 */
final class EngineConfigText {

    private EngineConfigText() {
    }

    static String print(EngineConfig c) {
        return "EngineConfig{backend=" + c.backend() + (c.deviceFilter() != null
                ? ", device~'" + c.deviceFilter() + "'" : "")
                + (c.splitMode() != EngineConfig.SplitMode.NONE
                ? ", split=" + c.splitMode().llamaName() + gpuDevicesPart(c)
                + (c.tensorSplit().length > 0 ? ", tensorSplit=" + Arrays.toString(c.tensorSplit()) : "")
                + (c.mainGpu() > 0 ? ", mainGpu=" + c.mainGpu() : "")
                + (c.rpcServers().isEmpty() ? "" : ", rpc=" + c.rpcServers())
                + (c.expertParallel() ? ", expertParallel" : "")
                + (c.eplb().isDefault() ? "" : ", eplb=" + c.eplb()) : "")
                + ", " + c.memory() + ", " + c.samplerDefaults() + ", ramCacheMb="
                + (c.ramCacheMb() == EngineConfig.RAM_CACHE_AUTO ? "auto" : String.valueOf(c.ramCacheMb()))
                + ", vramDepth="
                + (c.vramDepth() == EngineConfig.VRAM_DEPTH_AUTO ? "auto" : String.valueOf(c.vramDepth()))
                + (c.numaStrategy() == NumaStrategy.DISABLED ? "" : ", numa=" + c.numaStrategy().label())
                + (c.cpuOffloadMb() > 0 ? ", cpuOffloadMb=" + c.cpuOffloadMb() : "")
                + (c.cpuOffloadParams().isEmpty() ? "" : ", cpuOffloadParams=" + c.cpuOffloadParams())
                + (c.streamParams().isEmpty() ? "" : ", streamParams=" + c.streamParams())
                + (c.gpuVramLimitMb() == EngineConfig.GPU_VRAM_UNLIMITED ? ""
                : ", gpuVramLimitMb=" + c.gpuVramLimitMb())
                + (c.gpuLayers() == EngineConfig.GPU_LAYERS_AUTO ? ", gpuLayers=auto"
                : c.gpuLayers() == EngineConfig.GPU_LAYERS_ALL ? ", gpuLayers=all" : ", gpuLayers=" + c.gpuLayers())
                + (c.kvOffload() ? "" : ", kvOffload=false")
                + (c.tensorOverrides().isEmpty() ? "" : ", overrideTensor=" + c.tensorOverrides())
                + (c.cpuMoe() ? ", cpuMoe" : "") + (c.nCpuMoe() > 0 ? ", nCpuMoe=" + c.nCpuMoe() : "")
                + (c.moeCache().mode() == MoeCacheConfig.Mode.OFF && c.moeCache().ranking().isDefault()
                ? "" : ", " + c.moeCache())
                + (c.draftModelPath() != null ? ", draft=" + c.draftModelPath() : "")
                + (c.draftTypes().equals(DraftTypes.DEFAULT) ? ""
                : ", draftType=" + DraftTypes.names(c.draftTypes()))
                + (c.mtp() ? ", mtp" : "")
                + (c.speculativeOptions().isDefault() ? "" : ", " + c.speculativeOptions())
                + (c.lookaheadOptions().enabled() ? ", " + c.lookaheadOptions() : "")
                + (c.speculative() ? ", draftMax=" + c.draftMax() + ", draftMin=" + c.draftMin()
                + ", draftPMin=" + c.draftPMin() : "")
                + (c.loras().isEmpty() ? "" : ", lora=" + c.loras())
                + (c.controlVectors().isEmpty() ? "" : ", controlVector=" + c.controlVectors())
                + (c.controlVectorLayerStart() > 0 || c.controlVectorLayerEnd() > 0
                        ? ", controlVectorRange=" + c.controlVectorLayerStart() + ":" + c.controlVectorLayerEnd() : "")
                + (c.sourceOptions().equals(ModelSourceOptions.defaults()) ? "" : ", " + c.sourceOptions())
                + (c.activationQuant() == ActivationQuant.WEIGHT_ONLY ? "" : ", actQuant="
                + c.activationQuant().propertyValue())
                + (c.modelSamplingDefaults() ? "" : ", modelSamplingDefaults=false")
                + (c.reasoningParser().equals(EngineConfig.PARSER_AUTO) ? ""
                : ", reasoningParser=" + c.reasoningParser())
                + (c.toolParser().equals(EngineConfig.PARSER_AUTO) ? "" : ", toolParser=" + c.toolParser())
                + (c.toolStrict() != ToolStrict.AUTO
                ? ", toolStrict=" + c.toolStrict().name().toLowerCase(Locale.ROOT)
                : "") + (c.beamAlwaysConstrain() ? "" : ", beamAlwaysConstrain=false")
                + (c.nCpuFfn() != EngineConfig.N_CPU_LAYERS_UNSET ? ", nCpuFfn=" + c.nCpuFfn() : "")
                + (c.moeCacheMb() != 0 ? ", moeCacheMb=" + c.moeCacheMb() : "")
                + (c.moeCachePool() ? ", moeCachePool" : "")
                + (c.fit() ? ", fit" : "")
                + (c.pooling() != null ? ", pooling=" + c.pooling().name().toLowerCase(Locale.ROOT) : "")
                + (c.logitsProcessors().isEmpty() ? "" : ", logitsProcessors=" + c.logitsProcessors())
                + (c.draftGpuLayers() == EngineConfig.GPU_LAYERS_AUTO ? ""
                : c.draftGpuLayers() == EngineConfig.GPU_LAYERS_ALL ? ", draftGpuLayers=all"
                : ", draftGpuLayers=" + c.draftGpuLayers())
                + (c.draftTensorOverrides().isEmpty() ? ""
                : ", draftOverrideTensor=" + c.draftTensorOverrides())
                + (c.draftCpuMoe() ? ", draftCpuMoe" : "")
                + (c.draftNCpuMoe() > 0 ? ", draftNCpuMoe=" + c.draftNCpuMoe() : "")
                + (c.draftDevices().isEmpty() ? "" : ", draftDevices=" + c.draftDevices())
                + '}';
    }

    /** The {@link EngineConfig#gpuDevices()} entries as {@code devices=[a,b]}, or empty for the auto selection. */
    private static String gpuDevicesPart(EngineConfig c) {
        return c.gpuDevices().isEmpty() ? "" : " devices=" + c.gpuDevices();
    }
}
