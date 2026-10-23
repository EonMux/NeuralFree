package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.graph.GCNConv;
import com.eonmux.neuralfree.nn.graph.GraphData;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Random;

public final class GNNDemo {

    private static final int NUM_NODES = 20;
    private static final int IN_FEATURES = 8;
    private static final int HIDDEN_FEATURES = 16;
    private static final int OUT_FEATURES = 32;
    private static final int NUM_CLASSES = 3;
    private static final int NUM_EDGES = 40;
    private static final int EPOCHS = 10;

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("Graph Neural Network Demo");
        System.out.println("Training a GCN for node classification\n");

        Random random = new Random(42);

        GCNClassifier model = new GCNClassifier();
        model.initialize(backend, random);
        System.out.println("Model: " + model.name());
        System.out.println("Parameters: " + model.parameterCount() + "\n");

        System.out.println("Generating synthetic graph data...");
        
        float[][] nodeFeatures = new float[NUM_NODES][IN_FEATURES];
        int[] nodeLabels = new int[NUM_NODES];
        
        for (int n = 0; n < NUM_NODES; n++) {
            nodeLabels[n] = random.nextInt(NUM_CLASSES);
            for (int f = 0; f < IN_FEATURES; f++) {
                nodeFeatures[n][f] = (float) (random.nextGaussian() * 0.1);
            }
        }
        
        int[][] edgeList = new int[2][NUM_EDGES];
        for (int e = 0; e < NUM_EDGES; e++) {
            edgeList[0][e] = random.nextInt(NUM_NODES);
            edgeList[1][e] = random.nextInt(NUM_NODES);
        }

        float[] labelArray = new float[NUM_NODES];
        for (int i = 0; i < NUM_NODES; i++) {
            labelArray[i] = nodeLabels[i];
        }

        Adam optimizer = new Adam(model.parameters(), 0.01f);
        // GraphData owns and closes its tensors; closing them again afterwards is a no-op.
        try (Tensor nodeTensor = DemoUtils.tensor(backend, flattenFeatures(nodeFeatures), NUM_NODES, IN_FEATURES);
             Tensor edgeTensor = createEdgeTensor(backend, edgeList);
             GraphData graph = new GraphData(nodeTensor, edgeTensor);
             Tensor labelTensor = Tensor.fromArray(backend, labelArray)) {
            CrossEntropyLoss lossFn = new CrossEntropyLoss();

            System.out.println("Starting training loop...");
            model.train();

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                float lossVal = DemoUtils.trainStep(model, optimizer, () -> model.forwardGraph(graph),
                        output -> lossFn.compute(output, labelTensor), epoch == 0);
                System.out.printf("Epoch %d/%d - Loss: %.4f%n", epoch + 1, EPOCHS, lossVal);
            }

            System.out.println("\nRunning inference...");
            model.eval();

            float[] testFeatures = new float[IN_FEATURES];
            for (int i = 0; i < IN_FEATURES; i++) {
                testFeatures[i] = (float) (random.nextGaussian() * 0.1);
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor testNode = DemoUtils.tensor(backend, testFeatures, 1, IN_FEATURES);
                 Tensor testEdge = createEdgeTensor(backend, new int[][]{{0}, {0}});
                 GraphData testGraph = new GraphData(testNode, testEdge)) {
                long inferStart = System.nanoTime();
                float[] probs;
                try (Tensor output = model.forwardGraph(testGraph)) {
                    long inferTime = System.nanoTime() - inferStart;
                    probs = DemoUtils.probabilities(output, NUM_CLASSES);
                    System.out.println("Inference time: " + (inferTime / 1000.0) + " μs");
                }
                System.out.println("Predicted class: " + DemoUtils.argmax(probs));
                System.out.println("Probabilities: " + DemoUtils.formatProbs(probs));
            }
        } finally {
            optimizer.close();
            model.close();
        }
    }

    private static float[] flattenFeatures(float[][] features) {
        int numNodes = features.length;
        int numFeatures = features[0].length;
        
        float[] flat = new float[numNodes * numFeatures];
        int idx = 0;
        for (int n = 0; n < numNodes; n++) {
            for (int f = 0; f < numFeatures; f++) {
                flat[idx++] = features[n][f];
            }
        }
        return flat;
    }

    private static Tensor createEdgeTensor(ComputeBackend backend, int[][] edgeList) {
        int numEdges = edgeList[0].length;
        
        int[] flatEdges = new int[2 * numEdges];
        int idx = 0;
        for (int e = 0; e < numEdges; e++) {
            flatEdges[idx++] = edgeList[0][e];
        }
        for (int e = 0; e < numEdges; e++) {
            flatEdges[idx++] = edgeList[1][e];
        }
        
        return DemoUtils.intTensor(backend, flatEdges, 2, numEdges);
    }

    private static class GCNClassifier extends Module {
        private GCNConv conv1;
        private GCNConv conv2;
        private Linear classifier;

        @Override
        public Tensor forward(Tensor input) {
            throw new UnsupportedOperationException("Use forwardGraph() for GNN models");
        }

        /**
         * Both convolutions run on {@code graph}'s edges. Closes every intermediate it creates; the caller closes
         * the returned logits.
         */
        public Tensor forwardGraph(GraphData graph) {
            Tensor x = DemoUtils.consume(conv1.forward(graph), TensorOps::relu);
            x = DemoUtils.consume(x, h -> conv2.forward(graph, h));
            x = DemoUtils.consume(x, TensorOps::relu);
            return DemoUtils.consume(x, classifier::forward);
        }

        public void initialize(ComputeBackend backend, Random random) {
            conv1 = new GCNConv(IN_FEATURES, HIDDEN_FEATURES, true, true);
            conv1.initialize(backend, random);
            registerModule("conv1", conv1);

            conv2 = new GCNConv(HIDDEN_FEATURES, OUT_FEATURES, true, true);
            conv2.initialize(backend, random);
            registerModule("conv2", conv2);

            classifier = new Linear(OUT_FEATURES, NUM_CLASSES);
            classifier.initialize(backend, random);
            registerModule("classifier", classifier);
        }

        public String name() {
            return "GCNClassifier(features=" + IN_FEATURES + "->" + HIDDEN_FEATURES + 
                   "->" + OUT_FEATURES + ", classes=" + NUM_CLASSES + ")";
        }
    }
}
