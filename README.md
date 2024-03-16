# cs-neural-network
An implementation of neural networks with backpropagation, written in C#.

The zip file is a much older version. I realized I probably shouldn't be using (linked) lists for everything, the neuron class is redundant, and the variable names weren't ideal. The newest code (as of 10/4/2023) is "MachineLearning.cs", and I wrote some code to test it in "xorModel". It trains a fully connected neural network to approximate the logic of a xor gate. Keep in mind that it doesn't successfully learn it every single time. (which could also be a result of some bug I haven't found)

I have rewritten this code approximately 5-6 times over the years, and I didn't think to upload it to Github (as proof) until much later. My original motivation for writing it in c# was for ease of use in unity, as I dislike finding and installing plug-ins or dependencies for personal projects, preferring to implement them myself to learn the details.
