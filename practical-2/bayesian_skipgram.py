
## The inference model / encoder:
# need one hot encoded vectors of x, and all context vectors
# E = d x V , d = dim of embedding, V = vocabulary (column vectors everywhere)
# one_hot(x) times E -> to get dx1 vector (embedding) - deterministically
# concat every pair of (center_word, context word) - which is 1 x 2d
# apply elementwise relu on this
# sum every pair into a single (2d, 1) vector
# d_z <- embedding size of the latent variable z
# from the (2d, 1) vector, predict the mean and sigma using 2 (different) feed forward affine layer (paper mentions log sigma squared, but use sigma here)

## 'decoder' -> z to categorial (dim = |V|)
# each word in the vocab, L (location - means of each word) : a (d, V) matrix
# multiply L with one_hot(x) -> mu_x
# each word in the vocab, S(d, V) matrix
# multiply S with one_hot(x) -> s_z, softplus on top of that to finally get sigma
# square it and put in on the diagonal to get the multivariate sigma
# sample Z from the \mu_x and \sigma_x multivariate
# take z and project using NN, softmax on last layer to get a categorical dist over the vocabulary
# loss is negative of the ELBO
# MC estimate the first term which is the expectation p(c | z, x) of one sample
# sample one variational sample (from the encoder)
# z = \mu_{encoder} + noise (*) \sigma_{encoder}
# use this z to the decoder
# sum for every context word, compute the log probability


# encoder -> word in context
# L, S -> word without context (prior)

## KL - 3.4 on the paper
# use the output of the left (from the dist ) i.e L and S - this is where the other
# product of KL of  gaussians elementwise
# KL should be a positive scalar (check with an assert here)
