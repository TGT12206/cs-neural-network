using MachineLearning;

int[] dims = { 2, 3, 1 };
float[] input1 = { 0, 0 };
float[] target1 = { 0 };
float[] input2 = { 0, 1 };
float[] target2 = { 1 };
float[] input3 = { 1, 0 };
float[] target3 = { 1 };
float[] input4 = { 1, 1 };
float[] target4 = { 0 };

FullyConnectedNeuralNetwork xorModel = new FullyConnectedNeuralNetwork(dims);

xorModel.learningRate = 0.05f;

for (int epoch = 0; epoch < 20000; epoch++)
{
    xorModel.backpropagate(input1, target1);
    xorModel.backpropagate(input2, target2);
    xorModel.backpropagate(input3, target3);
    xorModel.backpropagate(input4, target4);
}

xorModel.feedForward(input1);
printList(xorModel.getOutput());
Console.WriteLine();
xorModel.feedForward(input2);
printList(xorModel.getOutput());
Console.WriteLine();
xorModel.feedForward(input3);
printList(xorModel.getOutput());
Console.WriteLine();
xorModel.feedForward(input4);
printList(xorModel.getOutput());
Console.WriteLine();

void printList(float[] list) {
    foreach (float item in list)
    {
        Console.Write(item + " ");
    }
    Console.WriteLine();
}
