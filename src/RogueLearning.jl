module RogueLearning
using Flux
using Flux: @functor, glorot_uniform, convfilter, calc_padding, expand, create_bias, Zeros
using NNlib: ∇conv_data, ∇conv_filter
import NNlib: conv
# import Base: *

export GenDense, GenConv, dfa_loss, radialSim, matmul,
    conv

include("layers.jl")
include("functions.jl")
include("losses.jl")
end
