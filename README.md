# Char-LSTM-Recurrent-Neural-Network

This repository contains a raw implementation of a character-level Recurrent Neural Network with an LSTM. No libraries beyond numPy was used. The RNN accepts as input textfile with at least ~2MBs worth of characters to be effective. It will read character by character and optimizes over it's weight using BPTT (Back-Propagation Through Time). An explanation of the implementation and examples are found below.


## Implementation

### Architecture

This RNN follows as:

![alt tag](http://i.imgur.com/OOxNvMF.png)

Let's take a simple example of the neural network predicting out the word "CARS"
The first thing that the network is provided is the letter "C". This letter is provided in what we call a "timestep", that is if we were to assume that the characters arrive in stream, "C" is the first letter to arrive. Technically speaking, this letter is accepted in a one-hot vector. This vector is pushed through some computation (linear transformation) and accepted into the recurrent layer, where an LSTM unit is stationed. In shorter terms, the ouput from the first layer is operated on and we result in a collection of four vectors (often denoted as i-f-o-g) which represent four qualities of the "state" of the system. In this case, the "state" refers to what we've seen in our network's memory - specifically, we've encoded that a "C" has been read. 

Assuming that our network has been trained, reaching the output layer, I use cross-entropy to probabilistically determine the next most likely letter - the network chooses "A". Encoded as one-hot, that vector is sent to the second timestep, and the same process repeats. However, upon reaching the recurrent layer, the computations operated to create a "new" state uses the saved state of the previous timestep. That is, the network operates on our input "A" with the knowledge that the previous letter was "C", and given well trained weights of our network, will go to the output layer and predict "R" after seeing "CA". This process repeats, where when the network predicts "S", it is on the basis that the network has read "CAR".

---

As a brief overview, a vanilla neural network has any number of nodes connected in each layer. A node in one layer is connected to all other nodes in it's adjacent layer with a "weight", and this weight continuously changes as we train. We use these weights and perform a standard linear transformation over our input vectors, standardizing these values with an "activation" function and sometimes batch normalization. By the process of back-propagation, these weights get optimized using variations of gradient descent.

The recurrent neural network works differently. In this scenario, given a subset of characters to predict on, the neural network will receive the first character provided in a one-Hot vector. It is pushed through a recurrent layer that rather than a basic linear transformation, does logic using a LSTM (Long-Short-Term-Memory) Unit, which maintains numerically a history of the previous inputs (in this case previous characters) and uses said information to propagate a memory-based interpretation into the output layer.

The output layer receives said input and through a fully connected layer, produces a vector - one value for each possible value - and pushes it through a softmax layer (cross-entropy) to predict a letter. In training, we determine how "wrong" we were using the softmax cost function and backpropagate the error back to the input layer of the neural net, tuning weights with gradient descent.

### Back-Propagation Through Time

The crux of how the recurrent neural network learns is "back-propagation through time". With vanilla neural networks, we take a batch of data, propogate through the network, make predictions and backpropagate in one go to tune parameters. The difference with an RNN is that it needs to train off of time-series data - that is to learn from how the previous states progress. Back-propagation through time is a technique for which we forward propagate for N number of timesteps (in my implementation, 25), and after learning over a sequence of 25, unroll these multiple forward propagations into a single "back-propagation-through-time". That is, propagating the error back through to the input layer while propagating through previous timesteps.


Below are some examples of what my network has produced. Thanks for reading!

## Examples

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13731686_1086152338131361_2437651348026443743_n.jpg?oh=64ad46b1f70d21f3a4b7861c1287903d&oe=58459340)

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13775414_1086152318131363_4752697976488961895_n.jpg?oh=beb51da34eae79ea955327eed84c8b3e&oe=584FB80E)

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13754220_1086152328131362_3047399722515646069_n.jpg?oh=22ee724f17429439f86243aec0eec2a2&oe=584D96DB)
