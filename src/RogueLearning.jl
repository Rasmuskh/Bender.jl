module RogueLearning
using Flux
using Flux: @functor, glorot_uniform, convfilter, calc_padding, expand, create_bias, Zeros
using NNlib: ∇conv_data, ∇conv_filter
using Zygote: pullback
import NNlib: conv
# import Base: *


include("layers.jl")
export GenDense, GenConv
include("activation_functions.jl")
export sign_STE, hardtanh_AdaSTE
include("similarity_functions.jl")
export radialSim, radialSim_asym, matmul, matmul_asym_∂x, matmul_blocked_∂x, conv_asym_∂x
include("losses.jl")
export direct_feedback_loss
end
