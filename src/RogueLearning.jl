module RogueLearning
using Flux
using Flux: @functor, glorot_uniform, convfilter, calc_padding, expand, create_bias, Zeros
using NNlib: ∇conv_data, ∇conv_filter
using Zygote: pullback
import NNlib: conv
# import Base: *

export GenDense, GenConv, direct_feedback_loss, radialSim, radialSim_asym, matmul, matmul_asym_∂x, matmul_blocked_∂x, conv_asym_∂x

include("layers.jl")
include("functions.jl")
include("losses.jl")
end
