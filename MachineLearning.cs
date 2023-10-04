namespace MachineLearning
{

    internal class FullyConnectedNeuralNetwork
    {
        // random number generator
        private static Random rng = new Random();

        // First [] is the layer, Second [] is the node
        public float[][] layers;

        // First [] is the layer, Second [] is the node, Third [] is the previous node
        public float[][][] weights;

        // First [] is the layer, Second [] is the node
        public float[][] biases;

        public float learningRate;

        public FullyConnectedNeuralNetwork(int[] dimensions)
        {
            int numLayers = dimensions.Length;

            // Create the layers and biases
            layers = new float[numLayers][];
            biases = new float[numLayers][];
            for (int i = 0; i < numLayers; i++)
            {
                layers[i] = new float[dimensions[i]];
                biases[i] = new float[dimensions[i]];
                for (int node = 0; node < biases[i].Length; node++)
                {
                    biases[i][node] = rng.NextSingle();
                }
            }

            // Create the weights
            weights = new float[numLayers][][];

            // Special case: first layer has no previous layer
            weights[0] = new float[layers[0].Length][];
            for (int node = 0; node < weights[0].Length; node++)
            {
                weights[0][node] = new float[1];
                weights[0][node][0] = rng.NextSingle();
            }

            for (int layer = 1; layer < numLayers; layer++)
            {
                weights[layer] = new float[layers[layer].Length][];
                for (int node = 0; node < weights[layer].Length; node++)
                {
                    weights[layer][node] = new float[layers[layer - 1].Length];
                    for (int previousNode = 0; previousNode < weights[layer - 1].Length; previousNode++)
                    {
                        weights[layer][node][previousNode] = rng.NextSingle();
                    }
                }
            }
        }

        public void feedForward(float[] inputs)
        {
            // Makes sure that there is a valid number of inputs
            if (inputs.Length != layers[0].Length)
            {
                Console.WriteLine("The size of the inputs does not match the size of the input layer!");
                return;
            }

            // Iterate through the layers
            for (int layer = 0; layer < layers.Length; layer++)
            {
                // Iterate through the nodes in each layer
                for (int nodeInLayer = 0; nodeInLayer < layers[layer].Length; nodeInLayer++)
                {
                    // Special Case: first layer
                    if (layer == 0)
                    {
                        // Apply the activation function at this node
                        float tempValue = inputs[nodeInLayer] * weights[0][nodeInLayer][0];
                        tempValue += biases[0][nodeInLayer];
                        layers[0][nodeInLayer] =
                            ReLU((inputs[nodeInLayer] * weights[0][nodeInLayer][0])
                                + biases[0][nodeInLayer]);
                    } else
                    {
                        // Apply the activation function at this node
                        layers[layer][nodeInLayer] =
                            activationFunction(layers[layer - 1],
                            weights[layer][nodeInLayer],
                            biases[layer][nodeInLayer]);
                    }
                }
            }
        }
        public void backpropagate(float[] inputs, float[] targets)
        {
            // Feed Forward
            // The feed forward method already ensures the correct number of outputs
            feedForward(inputs);

            // Ensures the correct number of targets.
            if (targets.Length != layers[layers.Length - 1].Length)
            {
                Console.WriteLine("The size of the targets does not match the size of the output layer!");
                return;
            }

            float[][] weightedErrors = new float[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                weightedErrors[layer] = new float[layers[layer].Length];
            }

            // Calculate all the errors
            for (int layer = layers.Length - 1; layer >= 0; layer--)
            {
                for (int node = 0; node < layers[layer].Length; node++)
                {
                    // Special case: output layer
                    if (layer == layers.Length - 1)
                    {
                        weightedErrors[layer][node] =
                            derReLU(layers[layer][node]) *
                            (targets[node] - layers[layer][node]) * learningRate;
                    } else
                    {
                        weightedErrors[layer][node] = 0;
                        for (int nextLayerNode = 0; nextLayerNode < layers[layer + 1].Length; nextLayerNode++)
                        {
                            weightedErrors[layer][node] +=
                                weightedErrors[layer + 1][nextLayerNode] *
                                weights[layer + 1][nextLayerNode][node];
                        }
                        weightedErrors[layer][node] =
                            derReLU(layers[layer][node]) *
                            weightedErrors[layer][node] * learningRate;
                    }
                }
            }

            // Adjust every weight
            for (int layer = layers.Length - 1; layer >= 0; layer--)
            {
                for (int node = 0; node < layers[layer].Length; node++)
                {
                    biases[layer][node] += weightedErrors[layer][node];
                    if (layer == 0)
                    {
                        weights[0][node][0] +=
                            weightedErrors[0][node] *
                            inputs[node];
                    } else
                    {
                        for (int previousLayerNode = 0; previousLayerNode < layers[layer - 1].Length; previousLayerNode++)
                        {
                            weights[layer][node][previousLayerNode] +=
                                weightedErrors[layer][node] *
                                layers[layer - 1][previousLayerNode];
                        }
                    }
                }
            }
        }

        public float[] getOutput()
        {
            return layers[layers.Length - 1];
        }
        private float activationFunction(float[] input, float[] weights, float bias)
        {
            float output = 0;
            for (int nodeInLayer = 0; nodeInLayer < input.Length; nodeInLayer++)
            {
                output += input[nodeInLayer] * weights[nodeInLayer];
            }
            output += bias;
            return ReLU(output);
        }

        private float derReLU(float x)
        {
            if (x > 0)
            {
                return 1;
            }
            return 0;
        }

        private float ReLU(float x)
        {
            if (x > 0)
            {
                return x;
            }
            return 0;
        }

    }

}
