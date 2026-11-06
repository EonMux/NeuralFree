package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * One concrete benchmark configuration of {@code nftool bench}, one row llama-bench's
 * {@code get_cmd_params_instances} expands: the model path, the test shape ({@code n_prompt}, {@code n_gen},
 * {@code n_depth}) and the settings whose change forces a model reload ({@code n_batch}, {@code n_ubatch},
 * {@code type_k}, {@code type_v}, {@code n_gpu_layers}, {@code n_cpu_moe}, {@code no_kv_offload},
 * {@code split_mode}, {@code main_gpu}, {@code devices}, {@code tensor_split}, {@code tensor_buft_overrides} and
 * {@code load_mode}). {@code n_threads} and {@code embeddings} describe the run without reloading the model. A
 * {@code -sm}, {@code -mg}, {@code -ts}, {@code -dev} or {@code -ot} value is kept in llama-bench's spelling and
 * applied through {@link EngineLoadFlags}, so the engine checks it exactly as every other tool does. Immutable.
 */
final class BenchInstance {

    final String model;
    final int nPrompt;
    final int nGen;
    final int nDepth;
    final int nBatch;
    final int nUbatch;
    final KvDType typeK;
    final KvDType typeV;
    final int ngl;
    /** llama-bench's {@code no_kv_offload}: true keeps the KV caches in host RAM. */
    final boolean noKvOffload;
    /** How the model spreads over GPUs (llama-bench's {@code split_mode}). */
    final EngineConfig.SplitMode splitMode;
    /** The GPU holding the embedding and the output head (llama-bench's {@code main_gpu}). */
    final int mainGpu;
    /** The layer share of each GPU (llama-bench's {@code tensor_split}); empty for equal shares. */
    private final double[] tensorSplit;
    /** The CPU pool size the run reports (llama-bench's {@code n_threads}). */
    final int nThreads;
    /** llama-bench's {@code n_cpu_moe}: the first N MoE layers keep their experts in host RAM. */
    final int nCpuMoe;
    /** llama-bench's {@code devices} in the engine's selector spelling; empty for auto, {@code none} for no offload. */
    final String devicesArg;
    /** llama-bench's {@code tensor_buft_overrides} of this run. */
    final BenchTensorOverrides tensorOverrides;
    /** llama-bench's {@code load_mode}. */
    final BenchLoadMode loadMode;
    /** llama-bench's {@code embeddings}: the timed prompt pass computes the model's embedding output. */
    final boolean embeddings;

    /**
     * A run with the engine's defaults: KV offloaded to the device, no multi-GPU split, main GPU 0, equal tensor
     * shares, no tensor overrides and the configured load mode.
     */
    BenchInstance(String model, int nPrompt, int nGen, int nDepth, int nBatch, int nUbatch,
                  KvDType typeK, KvDType typeV, int ngl) {
        this(model, nPrompt, nGen, nDepth, nBatch, nUbatch, typeK, typeV, ngl,
                false, EngineConfig.SplitMode.NONE, 0, new double[0]);
    }

    BenchInstance(String model, int nPrompt, int nGen, int nDepth, int nBatch, int nUbatch,
                  KvDType typeK, KvDType typeV, int ngl, boolean noKvOffload,
                  EngineConfig.SplitMode splitMode, int mainGpu, double[] tensorSplit) {
        this(model, nPrompt, nGen, nDepth, nBatch, nUbatch, typeK, typeV, ngl, noKvOffload, splitMode, mainGpu,
                tensorSplit, BenchGrid.defaultThreads(), EngineConfig.N_CPU_LAYERS_UNSET, "", BenchTensorOverrides.NONE,
                BenchLoadMode.AUTO, false);
    }

    BenchInstance(String model, int nPrompt, int nGen, int nDepth, int nBatch, int nUbatch,
                  KvDType typeK, KvDType typeV, int ngl, boolean noKvOffload,
                  EngineConfig.SplitMode splitMode, int mainGpu, double[] tensorSplit, int nThreads, int nCpuMoe,
                  String devicesArg, BenchTensorOverrides tensorOverrides, BenchLoadMode loadMode,
                  boolean embeddings) {
        this.model = model;
        this.nPrompt = nPrompt;
        this.nGen = nGen;
        this.nDepth = nDepth;
        this.nBatch = nBatch;
        this.nUbatch = nUbatch;
        this.typeK = typeK;
        this.typeV = typeV;
        this.ngl = ngl;
        this.noKvOffload = noKvOffload;
        this.splitMode = splitMode;
        this.mainGpu = mainGpu;
        this.tensorSplit = tensorSplit.clone();
        this.nThreads = nThreads;
        this.nCpuMoe = nCpuMoe;
        this.devicesArg = devicesArg == null ? "" : devicesArg;
        this.tensorOverrides = tensorOverrides;
        this.loadMode = loadMode;
        this.embeddings = embeddings;
    }

    /** The tokens one timed run processes: the prompt and the generated ones. */
    int nTokens() {
        return nPrompt + nGen;
    }

    /** The context the test needs: the depth, the prompt and the generated tokens. */
    int nCtxNeeded() {
        return nDepth + nTokens();
    }

    /** Whether the test asks for the GPU engine: any {@code n_gpu_layers} beyond 0 offloads layers. */
    boolean loadsGpu() {
        return ngl != 0;
    }

    /** The tensor shares as llama-bench prints them: two decimals joined by {@code /}, {@code 0.00} when equal. */
    String tensorSplitString() {
        if (tensorSplit.length == 0) {
            return "0.00";
        }
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < tensorSplit.length; i++) {
            if (i > 0) {
                s.append('/');
            }
            s.append(String.format(Locale.ROOT, "%.2f", tensorSplit[i]));
        }
        return s.toString();
    }

    /** The device selection as llama-bench prints it: {@code auto}, {@code none} or the names joined by {@code /}. */
    String devicesString() {
        if (devicesArg.isEmpty()) {
            return "auto";
        }
        if (devicesArg.equals("none")) {
            return "none";
        }
        return devicesArg.replace(',', '/');
    }

    /** The tensor overrides as llama-bench prints them ({@code pattern=buffer type} joined by {@code ;}). */
    String tensorOverridesString() {
        return tensorOverrides.text();
    }

    /** Whether the run keeps the model off every GPU: {@code -ngl 0} or {@code -dev none}. */
    boolean runsOnCpu() {
        return !loadsGpu() || devicesArg.equals("none");
    }

    /** The tensor shares as {@link EngineLoadFlags} reads them, full precision, joined by {@code /}. */
    private String tensorSplitArg() {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < tensorSplit.length; i++) {
            if (i > 0) {
                s.append('/');
            }
            s.append(tensorSplit[i]);
        }
        return s.toString();
    }

    /**
     * The engine configuration this run opens with: {@code base} (the {@code -Dneuralfree.*} settings), the
     * group's memory {@code mc}, and the run's device, expert-placement, split, KV-offload and tensor-override
     * settings applied through {@link EngineLoadFlags} so the flags are parsed and checked as every other tool
     * checks them. An {@code ngl} of 0, or {@code -dev none}, runs the CPU engine, selected with llama.cpp's
     * {@code --device none}.
     *
     * @throws EngineLoadFlags.FlagException naming the flag when the settings cannot run
     */
    EngineConfig engineConfig(EngineConfig base, MemoryConfig mc) throws EngineLoadFlags.FlagException {
        EngineConfig seeded = base.toBuilder().memory(mc).build();
        EngineLoadFlags load = EngineLoadFlags.over(seeded);
        List<String> flags = new ArrayList<>();
        if (runsOnCpu()) {
            flags.add("--device");
            flags.add("none");
        } else if (!devicesArg.isEmpty()) {
            flags.add("--device");
            flags.add(devicesArg);
        }
        if (nCpuMoe > EngineConfig.N_CPU_LAYERS_UNSET) {
            flags.add("--n-cpu-moe");
            flags.add(Integer.toString(nCpuMoe));
        }
        flags.add("--split-mode");
        flags.add(splitMode.llamaName());
        flags.add("--main-gpu");
        flags.add(Integer.toString(mainGpu));
        flags.add(noKvOffload ? "--no-kv-offload" : "--kv-offload");
        if (tensorSplit.length > 0) {
            flags.add("--tensor-split");
            flags.add(tensorSplitArg());
        }
        String overrides = tensorOverrides.engineArg();
        if (overrides != null) {
            flags.add("--override-tensor");
            flags.add(overrides);
        }
        String[] args = flags.toArray(new String[0]);
        for (int i = 0; i < args.length; i++) {
            int last = load.parse(args, i);
            if (last < 0) {
                throw new IllegalStateException("not a load flag: " + args[i]);
            }
            i = last;
        }
        return load.build();
    }

    /**
     * The key the runs group by: the settings a NeuralFree load reads. {@code n_batch} is metadata here (the
     * engine's prefill chunks by {@code n_ubatch} alone), so it is not part of the key; {@code embeddings} is,
     * because the embedding pass needs the paged memory form.
     */
    String engineKey() {
        return model + "|" + ngl + "|" + nUbatch + "|" + typeK + "|" + typeV + "|" + noKvOffload + "|"
                + splitMode.llamaName() + "|" + mainGpu + "|" + java.util.Arrays.toString(tensorSplit) + "|"
                + nCpuMoe + "|" + devicesArg + "|" + tensorOverrides.text() + "|" + loadMode.llamaName() + "|"
                + embeddings;
    }

    /** The test's name in llama-bench's Markdown table: {@code pp}, {@code tg} or {@code pp+tg}, with the depth. */
    String testName() {
        StringBuilder s = new StringBuilder();
        if (nGen == 0 || nPrompt > 0) {
            s.append("pp").append(nPrompt);
        }
        if (nPrompt == 0 || nGen > 0) {
            if (s.length() > 0) {
                s.append('+');
            }
            s.append("tg").append(nGen);
        }
        if (nDepth > 0) {
            s.append(" @ d").append(nDepth);
        }
        return s.toString();
    }
}
