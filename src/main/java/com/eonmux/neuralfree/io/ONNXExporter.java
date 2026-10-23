package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.layers.Conv1d;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.ConvTranspose2d;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.layers.AvgPool2d;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.activations.Sigmoid;
import com.eonmux.neuralfree.nn.activations.Tanh;
import com.eonmux.neuralfree.nn.activations.Softmax;
import com.eonmux.neuralfree.nn.activations.GELU;
import com.eonmux.neuralfree.nn.activations.LeakyReLU;
import com.eonmux.neuralfree.nn.activations.ELU;
import com.eonmux.neuralfree.nn.activations.SiLU;
import com.eonmux.neuralfree.nn.activations.LogSoftmax;
import com.eonmux.neuralfree.nn.norm.AbstractBatchNorm;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.norm.GroupNorm;
import com.eonmux.neuralfree.nn.norm.RMSNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public final class ONNXExporter {
    private static final int ONNX_MAGIC_NUMBER = 0x08C7;
    private static final long ONNX_VERSION = 9L;

    private final String modelName;
    private final String domain;
    private final long modelVersion;
    private final List<ONNXNode> nodes;
    private final List<ONNXInitializer> initializers;
    private final List<String> inputs;
    private final List<String> outputs;
    private int nameCounter;

    public ONNXExporter() {
        this("NeuralFreeModel", "", 1L);
    }

    public ONNXExporter(String modelName) {
        this(modelName, "", 1L);
    }

    public ONNXExporter(String modelName, String domain, long modelVersion) {
        this.modelName = modelName;
        this.domain = domain;
        this.modelVersion = modelVersion;
        this.nodes = new ArrayList<>();
        this.initializers = new ArrayList<>();
        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();
        this.nameCounter = 0;
    }

    public void export(Module model, Tensor sampleInput, String path) throws IOException {
        export(model, sampleInput, Paths.get(path));
    }

    public void export(Module model, Tensor sampleInput, Path path) throws IOException {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(sampleInput, "Sample input cannot be null");
        Objects.requireNonNull(path, "Path cannot be null");

        nodes.clear();
        initializers.clear();
        inputs.clear();
        outputs.clear();
        nameCounter = 0;

        String inputName = generateName("input");
        inputs.add(inputName);

        String outputName = traverseModule(model, inputName);
        outputs.add(outputName);

        writeONNXModel(path);
    }

    private String traverseModule(Module module, String inputName) {
        String outputName = inputName;

        if (module instanceof Sequential) {
            Sequential seq = (Sequential) module;
            for (int i = 0; i < seq.size(); i++) {
                Module layer = seq.get(i);
                outputName = traverseModule(layer, outputName);
            }
        } else if (module instanceof Linear) {
            outputName = processLinear((Linear) module, inputName);
        } else if (module instanceof Conv1d) {
            outputName = processConv1d((Conv1d) module, inputName);
        } else if (module instanceof Conv2d) {
            outputName = processConv2d((Conv2d) module, inputName);
        } else if (module instanceof ConvTranspose2d) {
            outputName = processConvTranspose2d((ConvTranspose2d) module, inputName);
        } else if (module instanceof MaxPool2d) {
            outputName = processMaxPool2d((MaxPool2d) module, inputName);
        } else if (module instanceof AvgPool2d) {
            outputName = processAvgPool2d((AvgPool2d) module, inputName);
        } else if (module instanceof AbstractBatchNorm) {
            outputName = processBatchNorm((AbstractBatchNorm) module, inputName);
        } else if (module instanceof LayerNorm) {
            outputName = processLayerNorm((LayerNorm) module, inputName);
        } else if (module instanceof GroupNorm) {
            outputName = processGroupNorm((GroupNorm) module, inputName);
        } else if (module instanceof RMSNorm) {
            outputName = processRMSNorm((RMSNorm) module, inputName);
        } else if (module instanceof Dropout) {
            outputName = inputName;
        } else if (module instanceof Embedding) {
            outputName = processEmbedding((Embedding) module, inputName);
        } else if (module instanceof ReLU) {
            outputName = addNode("Relu", "Relu", Arrays.asList(inputName), 1).get(0);
        } else if (module instanceof Sigmoid) {
            outputName = addNode("Sigmoid", "Sigmoid", Arrays.asList(inputName), 1).get(0);
        } else if (module instanceof Tanh) {
            outputName = addNode("Tanh", "Tanh", Arrays.asList(inputName), 1).get(0);
        } else if (module instanceof Softmax) {
            outputName = addNodeWithAttribute("Softmax", "Softmax", Arrays.asList(inputName), 1,
                new ONNXAttribute("axis", -1)).get(0);
        } else if (module instanceof LogSoftmax) {
            outputName = addNodeWithAttribute("LogSoftmax", "LogSoftmax", Arrays.asList(inputName), 1,
                new ONNXAttribute("axis", -1)).get(0);
        } else if (module instanceof GELU) {
            outputName = addNode("Gelu", "Gelu", Arrays.asList(inputName), 1).get(0);
        } else if (module instanceof LeakyReLU) {
            LeakyReLU lr = (LeakyReLU) module;
            outputName = addNodeWithAttribute("LeakyRelu", "LeakyRelu", Arrays.asList(inputName), 1,
                new ONNXAttribute("alpha", lr.negativeSlope())).get(0);
        } else if (module instanceof ELU) {
            ELU elu = (ELU) module;
            outputName = addNodeWithAttribute("Elu", "Elu", Arrays.asList(inputName), 1,
                new ONNXAttribute("alpha", elu.alpha())).get(0);
        } else if (module instanceof SiLU) {
            outputName = addNode("Silu", "Silu", Arrays.asList(inputName), 1).get(0);
        } else {
            outputName = processGenericModule(module, inputName);
        }

        return outputName;
    }

    private String processLinear(Linear layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String weightName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        List<String> inputNames = new ArrayList<>(Arrays.asList(inputName, weightName));
        
        String outputName = generateName(layer.moduleName() + "_output");
        
        if (layer.hasBias()) {
            String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
            inputNames.add(biasName);
        }
        
        nodes.add(new ONNXNode(
            generateName("Gemm"),
            "Gemm",
            inputNames,
            Arrays.asList(outputName),
            Arrays.asList(
                new ONNXAttribute("alpha", 1.0f),
                new ONNXAttribute("beta", 1.0f),
                new ONNXAttribute("transB", 1)
            )
        ));
        
        return outputName;
    }

    private String processConv1d(Conv1d layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String weightName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        List<String> inputNames = new ArrayList<>(Arrays.asList(inputName, weightName));
        
        String outputName = generateName(layer.moduleName() + "_output");
        
        if (layer.hasBias()) {
            String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
            inputNames.add(biasName);
        }
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("kernel_shape", new long[]{layer.kernelSize()}));
        attrs.add(new ONNXAttribute("strides", new long[]{layer.stride()}));
        attrs.add(new ONNXAttribute("pads", new long[]{layer.padding(), layer.padding()}));
        attrs.add(new ONNXAttribute("dilations", new long[]{layer.dilation()}));
        attrs.add(new ONNXAttribute("group", (long) layer.groups()));
        
        nodes.add(new ONNXNode(
            generateName("Conv"),
            "Conv",
            inputNames,
            Arrays.asList(outputName),
            attrs
        ));
        
        return outputName;
    }

    private String processConv2d(Conv2d layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String weightName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        List<String> inputNames = new ArrayList<>(Arrays.asList(inputName, weightName));
        
        String outputName = generateName(layer.moduleName() + "_output");
        
        if (layer.hasBias()) {
            String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
            inputNames.add(biasName);
        }
        
        int[] kernelSize = layer.kernelSize();
        int[] stride = layer.stride();
        int[] padding = layer.padding();
        int[] dilation = layer.dilation();
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("kernel_shape", new long[]{kernelSize[0], kernelSize[1]}));
        attrs.add(new ONNXAttribute("strides", new long[]{stride[0], stride[1]}));
        attrs.add(new ONNXAttribute("pads", new long[]{padding[0], padding[1], padding[0], padding[1]}));
        attrs.add(new ONNXAttribute("dilations", new long[]{dilation[0], dilation[1]}));
        attrs.add(new ONNXAttribute("group", (long) layer.groups()));
        
        nodes.add(new ONNXNode(
            generateName("Conv"),
            "Conv",
            inputNames,
            Arrays.asList(outputName),
            attrs
        ));
        
        return outputName;
    }

    private String processConvTranspose2d(ConvTranspose2d layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String weightName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        List<String> inputNames = new ArrayList<>(Arrays.asList(inputName, weightName));
        
        String outputName = generateName(layer.moduleName() + "_output");
        
        if (layer.hasBias()) {
            String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
            inputNames.add(biasName);
        }
        
        int[] kernelSize = layer.kernelSize();
        int[] stride = layer.stride();
        int[] padding = layer.padding();
        int[] dilation = layer.dilation();
        int[] outputPadding = layer.outputPadding();
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("kernel_shape", new long[]{kernelSize[0], kernelSize[1]}));
        attrs.add(new ONNXAttribute("strides", new long[]{stride[0], stride[1]}));
        attrs.add(new ONNXAttribute("pads", new long[]{padding[0], padding[1], padding[0], padding[1]}));
        attrs.add(new ONNXAttribute("dilations", new long[]{dilation[0], dilation[1]}));
        attrs.add(new ONNXAttribute("output_padding", new long[]{outputPadding[0], outputPadding[1]}));
        attrs.add(new ONNXAttribute("group", (long) layer.groups()));
        
        nodes.add(new ONNXNode(
            generateName("ConvTranspose"),
            "ConvTranspose",
            inputNames,
            Arrays.asList(outputName),
            attrs
        ));
        
        return outputName;
    }

    private String processMaxPool2d(MaxPool2d layer, String inputName) {
        int[] kernelSize = layer.kernelSize();
        int[] stride = layer.stride();
        int[] padding = layer.padding();
        int[] dilation = layer.dilation();
        
        String outputName = generateName("MaxPool_output");
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("kernel_shape", new long[]{kernelSize[0], kernelSize[1]}));
        attrs.add(new ONNXAttribute("strides", new long[]{stride[0], stride[1]}));
        attrs.add(new ONNXAttribute("pads", new long[]{padding[0], padding[1], padding[0], padding[1]}));
        attrs.add(new ONNXAttribute("dilations", new long[]{dilation[0], dilation[1]}));
        
        nodes.add(new ONNXNode(
            generateName("MaxPool"),
            "MaxPool",
            Arrays.asList(inputName),
            Arrays.asList(outputName),
            attrs
        ));
        
        return outputName;
    }

    private String processAvgPool2d(AvgPool2d layer, String inputName) {
        int[] kernelSize = layer.kernelSize();
        int[] stride = layer.stride();
        int[] padding = layer.padding();
        
        String outputName = generateName("AvgPool_output");
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("kernel_shape", new long[]{kernelSize[0], kernelSize[1]}));
        attrs.add(new ONNXAttribute("strides", new long[]{stride[0], stride[1]}));
        attrs.add(new ONNXAttribute("pads", new long[]{padding[0], padding[1], padding[0], padding[1]}));
        
        nodes.add(new ONNXNode(
            generateName("AveragePool"),
            "AveragePool",
            Arrays.asList(inputName),
            Arrays.asList(outputName),
            attrs
        ));
        
        return outputName;
    }

    /**
     * Exports BatchNorm1d or BatchNorm2d as ONNX BatchNormalization. ONNX's optional {@code momentum} is the weight of
     * the previous running value ({@code running = running * momentum + batch * (1 - momentum)}), the complement of
     * the module's, which weights the batch statistic; a module using the cumulative moving average has no fixed
     * momentum, so the attribute is left out (it affects only training-mode execution). BatchNormalization requires
     * its scale and B inputs, so a module without affine parameters exports constant ones and zeros, the identity
     * transform it applies.
     */
    private String processBatchNorm(AbstractBatchNorm layer, String inputName) {
        String scaleName;
        String biasName;
        if (layer.hasAffine()) {
            scaleName = addInitializer(layer.moduleName() + ".weight", layer.weight());
            biasName = addInitializer(layer.moduleName() + ".bias", layer.bias());
        } else {
            Shape features = Shape.vector(layer.numFeatures());
            scaleName = addConstantInitializer(layer.moduleName() + ".weight", features, 1f);
            biasName = addConstantInitializer(layer.moduleName() + ".bias", features, 0f);
        }
        String meanName = addInitializer(layer.moduleName() + ".running_mean", layer.runningMean());
        String varName = addInitializer(layer.moduleName() + ".running_var", layer.runningVar());

        String outputName = generateName("BatchNorm_output");

        List<ONNXAttribute> attributes = new ArrayList<>();
        attributes.add(new ONNXAttribute("epsilon", layer.epsilon()));
        if (!layer.cumulativeAverage()) {
            attributes.add(new ONNXAttribute("momentum", 1f - layer.momentum()));
        }
        nodes.add(new ONNXNode(
            generateName("BatchNormalization"),
            "BatchNormalization",
            Arrays.asList(inputName, scaleName, biasName, meanName, varName),
            Arrays.asList(outputName),
            attributes
        ));

        return outputName;
    }

    private String processLayerNorm(LayerNorm layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String scaleName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
        
        String outputName = generateName("LayerNorm_output");
        
        nodes.add(new ONNXNode(
            generateName("LayerNormalization"),
            "LayerNormalization",
            Arrays.asList(inputName, scaleName, biasName),
            Arrays.asList(outputName),
            Arrays.asList(
                new ONNXAttribute("epsilon", layer.epsilon()),
                new ONNXAttribute("axis", -1)
            )
        ));
        
        return outputName;
    }

    private String processGroupNorm(GroupNorm layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String scaleName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        String biasName = addInitializer(layer.moduleName() + ".bias", params.get("bias"));
        
        String outputName = generateName("GroupNorm_output");
        
        nodes.add(new ONNXNode(
            generateName("GroupNormalization"),
            "GroupNormalization",
            Arrays.asList(inputName, scaleName, biasName),
            Arrays.asList(outputName),
            Arrays.asList(
                new ONNXAttribute("epsilon", layer.epsilon()),
                new ONNXAttribute("num_groups", (long) layer.numGroups())
            )
        ));
        
        return outputName;
    }

    private String processRMSNorm(RMSNorm layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String scaleName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        String outputName = generateName("RMSNorm_output");
        
        String squaredName = generateName("squared");
        String meanName = generateName("mean");
        String sqrtName = generateName("sqrt");
        String divName = generateName("div");
        
        nodes.add(new ONNXNode(
            generateName("Mul"),
            "Mul",
            Arrays.asList(inputName, inputName),
            Arrays.asList(squaredName),
            Collections.emptyList()
        ));
        
        List<ONNXAttribute> reduceMeanAttrs = new ArrayList<>();
        reduceMeanAttrs.add(new ONNXAttribute("axes", new long[]{-1}));
        reduceMeanAttrs.add(new ONNXAttribute("keepdims", 1L));
        nodes.add(new ONNXNode(
            generateName("ReduceMean"),
            "ReduceMean",
            Arrays.asList(squaredName),
            Arrays.asList(meanName),
            reduceMeanAttrs
        ));
        
        String epsName = addConstantInitializer("eps_const", Shape.scalar(), layer.epsilon());
        
        nodes.add(new ONNXNode(
            generateName("Add"),
            "Add",
            Arrays.asList(meanName, epsName),
            Arrays.asList(sqrtName),
            Collections.emptyList()
        ));
        
        String rsqrtName = generateName("rsqrt");
        nodes.add(new ONNXNode(
            generateName("Sqrt"),
            "Sqrt",
            Arrays.asList(sqrtName),
            Arrays.asList(rsqrtName),
            Collections.emptyList()
        ));
        
        nodes.add(new ONNXNode(
            generateName("Mul"),
            "Mul",
            Arrays.asList(inputName, rsqrtName),
            Arrays.asList(divName),
            Collections.emptyList()
        ));
        
        nodes.add(new ONNXNode(
            generateName("Mul"),
            "Mul",
            Arrays.asList(divName, scaleName),
            Arrays.asList(outputName),
            Collections.emptyList()
        ));
        
        return outputName;
    }

    private String processEmbedding(Embedding layer, String inputName) {
        Map<String, Parameter> params = layer.namedParameters();
        
        String weightName = addInitializer(layer.moduleName() + ".weight", params.get("weight"));
        String outputName = generateName("Embedding_output");
        
        nodes.add(new ONNXNode(
            generateName("Gather"),
            "Gather",
            Arrays.asList(weightName, inputName),
            Arrays.asList(outputName),
            Arrays.asList(new ONNXAttribute("axis", 0))
        ));
        
        return outputName;
    }

    private String processGenericModule(Module module, String inputName) {
        Map<String, Parameter> params = module.namedParameters();
        
        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            addInitializer(module.moduleName() + "." + entry.getKey(), entry.getValue());
        }
        
        List<Module> children = module.children();
        String outputName = inputName;
        
        for (Module child : children) {
            outputName = traverseModule(child, outputName);
        }
        
        return outputName;
    }

    /** An initializer of {@code shape} filled with {@code value}; no device tensor is involved. */
    private String addConstantInitializer(String name, Shape shape, float value) {
        float[] data = new float[Math.toIntExact(shape.numElements())];
        Arrays.fill(data, value);
        String uniqueName = generateName(name);
        initializers.add(new ONNXInitializer(uniqueName, data, shape.dims()));
        return uniqueName;
    }

    private String addInitializer(String name, Parameter param) {
        return addInitializer(name, param.tensor());
    }

    /** An initializer holding the values of {@code values}, read back for serialization. */
    private String addInitializer(String name, Tensor values) {
        String uniqueName = generateName(name);
        initializers.add(new ONNXInitializer(uniqueName, values.toFloatArray(), values.shape().dims()));
        return uniqueName;
    }

    private List<String> addNode(String name, String opType, List<String> inputs, int numOutputs) {
        List<String> outputs = new ArrayList<>();
        for (int i = 0; i < numOutputs; i++) {
            outputs.add(generateName(name + "_out" + i));
        }
        nodes.add(new ONNXNode(generateName(name), opType, inputs, outputs, Collections.emptyList()));
        return outputs;
    }

    private List<String> addNodeWithAttribute(String name, String opType, List<String> inputs, 
                                               int numOutputs, ONNXAttribute... attrs) {
        List<String> outputs = new ArrayList<>();
        for (int i = 0; i < numOutputs; i++) {
            outputs.add(generateName(name + "_out" + i));
        }
        nodes.add(new ONNXNode(generateName(name), opType, inputs, outputs, Arrays.asList(attrs)));
        return outputs;
    }

    private String generateName(String base) {
        return base.replaceAll("[^a-zA-Z0-9_]", "_") + "_" + (nameCounter++);
    }

    private void writeONNXModel(Path path) throws IOException {
        Path parentDir = path.getParent();
        if (parentDir != null && !Files.exists(parentDir)) {
            Files.createDirectories(parentDir);
        }

        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path.toFile()))) {
            dos.writeInt(Integer.reverseBytes(ONNX_MAGIC_NUMBER));
            
            writeString(dos, modelName);
            writeVarInt(dos, 1);
            writeString(dos, domain);
            writeVarInt(dos, modelVersion);
            
            writeGraph(dos);
        }
    }

    private void writeGraph(DataOutputStream dos) throws IOException {
        writeString(dos, "");
        
        writeVarInt(dos, initializers.size());
        for (ONNXInitializer init : initializers) {
            writeInitializer(dos, init);
        }
        
        writeVarInt(dos, inputs.size());
        for (String input : inputs) {
            writeValueInfo(dos, input);
        }
        
        writeVarInt(dos, outputs.size());
        for (String output : outputs) {
            writeValueInfo(dos, output);
        }
        
        writeVarInt(dos, nodes.size());
        for (ONNXNode node : nodes) {
            writeNode(dos, node);
        }
        
        writeString(dos, "");
        writeString(dos, "");
    }

    private void writeInitializer(DataOutputStream dos, ONNXInitializer init) throws IOException {
        writeString(dos, "");
        
        List<ONNXAttribute> attrs = new ArrayList<>();
        attrs.add(new ONNXAttribute("dims", toLongArray(init.dims)));
        attrs.add(new ONNXAttribute("data_type", 1L));
        attrs.add(new ONNXAttribute("float_data", init.data));
        writeAttributes(dos, attrs);
        
        writeString(dos, init.name);
        writeString(dos, "");
        writeString(dos, "");
    }

    private void writeNode(DataOutputStream dos, ONNXNode node) throws IOException {
        writeString(dos, "");
        
        writeVarInt(dos, node.inputs.size());
        for (String input : node.inputs) {
            writeString(dos, input);
        }
        
        writeVarInt(dos, node.outputs.size());
        for (String output : node.outputs) {
            writeString(dos, output);
        }
        
        writeString(dos, node.name);
        writeString(dos, node.opType);
        writeAttributes(dos, node.attributes);
        writeString(dos, "");
    }

    private void writeValueInfo(DataOutputStream dos, String name) throws IOException {
        writeString(dos, name);
        writeString(dos, "");
    }

    private void writeAttributes(DataOutputStream dos, List<ONNXAttribute> attrs) throws IOException {
        writeVarInt(dos, attrs.size());
        for (ONNXAttribute attr : attrs) {
            writeAttribute(dos, attr);
        }
    }

    private void writeAttribute(DataOutputStream dos, ONNXAttribute attr) throws IOException {
        writeString(dos, attr.name);
        
        if (attr.type == AttributeType.FLOAT) {
            dos.writeByte(1);
            dos.writeFloat(Float.floatToIntBits((Float) attr.value));
        } else if (attr.type == AttributeType.INT) {
            dos.writeByte(2);
            writeVarInt(dos, ((Number) attr.value).longValue());
        } else if (attr.type == AttributeType.STRING) {
            dos.writeByte(3);
            writeString(dos, (String) attr.value);
        } else if (attr.type == AttributeType.FLOATS) {
            dos.writeByte(6);
            float[] floats = (float[]) attr.value;
            writeVarInt(dos, floats.length);
            for (float f : floats) {
                dos.writeFloat(Float.floatToIntBits(f));
            }
        } else if (attr.type == AttributeType.INTS) {
            dos.writeByte(7);
            long[] ints = (long[]) attr.value;
            writeVarInt(dos, ints.length);
            for (long i : ints) {
                writeVarInt(dos, i);
            }
        } else if (attr.type == AttributeType.STRINGS) {
            dos.writeByte(8);
            String[] strings = (String[]) attr.value;
            writeVarInt(dos, strings.length);
            for (String s : strings) {
                writeString(dos, s);
            }
        }
    }

    private void writeString(DataOutputStream dos, String s) throws IOException {
        if (s == null) s = "";
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        writeVarInt(dos, bytes.length);
        dos.write(bytes);
    }

    private void writeVarInt(DataOutputStream dos, long value) throws IOException {
        while ((value & ~0x7FL) != 0) {
            dos.writeByte(((int) value & 0x7F) | 0x80);
            value >>>= 7;
        }
        dos.writeByte((int) value);
    }

    private long[] toLongArray(int[] arr) {
        long[] result = new long[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }
        return result;
    }

    private enum AttributeType {
        FLOAT, INT, STRING, FLOATS, INTS, STRINGS
    }

    private static class ONNXAttribute {
        final String name;
        final Object value;
        final AttributeType type;

        ONNXAttribute(String name, float value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.FLOAT;
        }

        ONNXAttribute(String name, long value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.INT;
        }

        ONNXAttribute(String name, String value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.STRING;
        }

        ONNXAttribute(String name, float[] value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.FLOATS;
        }

        ONNXAttribute(String name, long[] value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.INTS;
        }

        ONNXAttribute(String name, String[] value) {
            this.name = name;
            this.value = value;
            this.type = AttributeType.STRINGS;
        }
    }

    private static class ONNXNode {
        final String name;
        final String opType;
        final List<String> inputs;
        final List<String> outputs;
        final List<ONNXAttribute> attributes;

        ONNXNode(String name, String opType, List<String> inputs, List<String> outputs, 
                 List<ONNXAttribute> attributes) {
            this.name = name;
            this.opType = opType;
            this.inputs = inputs;
            this.outputs = outputs;
            this.attributes = attributes;
        }
    }

    private static class ONNXInitializer {
        final String name;
        final float[] data;
        final int[] dims;

        ONNXInitializer(String name, float[] data, int[] dims) {
            this.name = name;
            this.data = data;
            this.dims = dims;
        }
    }
}
