# tsw-R-RBM
Example of a functionally designed Restricted Boltzmann Machine constructed in R, including encoding, decoding, and CD-1 training functions.

### RBM(visible_size, hidden_size, initial_learn_rate=1e-3)
Returns a list structure with a weight matrix, hidden and visible bias vectors, the assigned learn rate, and for convenience, the size of the input vector.


### trainRBM(rbm, data, delta_learn_rate=0.0)
Trains an RBM object on the given data.
The learn_rate of the new object will be summed by delta_learn_rate, allowing for extrinsic learn rate tweening.
The resulting incrementally trained RBM may be kept distinctly or assigned over the prior iteration, depending on scale and user needs.


###encode_by_RBM(rbm, data)
Evaluates a feature vector resulting from the given visible vector

ex: encode_by_rbm(an_initialized_rbm, c(.19, .24, .37, .95, .25)) => c(.74, .44)


###decode_by_RBM(rbm, data)
Reconstructs a visible data vector from the given feature vector.

ex: decode_by_rbm(an_initialized_rbm, c(.74, .44)) => c(.18, .22, .35, .95, .24)
