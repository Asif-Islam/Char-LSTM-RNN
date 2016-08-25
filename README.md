# Char-LSTM-Recurrent-Neural-Network

This repository contains a raw implementation of a character-level Recurrent Neural Network with an LSTM. No libraries beyond numPy was used. The RNN accepts as input textfile with at least ~2MBs worth of characters to be effective. It will read character by character and optimizes over it's weight using BPTT (Back-Propagation Through Time). An explanation of the implementation and examples are found below.


## Implementation

### Architecture

This RNN follows as:

```
===   OUTPUT LAYER   ===
          |
=== RECCURRENT LAYER ===
          |
===   INPUT LAYER    ===
```

As a brief overview, a vanilla neural network has any number of nodes connected in each layer. A node in one layer is connected to all other nodes in it's adjacent layer with a "weight", and this weight continuously changes as we train. We use these weights and perform a standard linear transformation over our input vectors, standardizing these values with an "activation" function and somes some batch normalization. By the process of back-propagation, these weights get optimized using derivatives of gradient descent.

The recurrent neural network works differently. In this scenario, given a subset of characters to predict on, the neural network will receive the first character provided in a one-Hot vector. It is pushed through a recurrent layer that rather than a basic linear transformation, does logic using a LSTM (Long-Short-Term-Memory) Unit, which maintains numerically a history of the previous inputs (in this case previous characters) and uses said information to propgate into the output layer.

The output layer receives said input and through a fully connected vanilla layer, produces a vector - one value for each possible value - and pushes it through a softmax layer (cross-entropy) to predict a letter. 


## Examples

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13731686_1086152338131361_2437651348026443743_n.jpg?oh=64ad46b1f70d21f3a4b7861c1287903d&oe=58459340)

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13775414_1086152318131363_4752697976488961895_n.jpg?oh=beb51da34eae79ea955327eed84c8b3e&oe=584FB80E)

![alt tag](https://scontent.xx.fbcdn.net/v/t1.0-9/13754220_1086152328131362_3047399722515646069_n.jpg?oh=22ee724f17429439f86243aec0eec2a2&oe=584D96DB)
