module Bender
using Flux
using Flux: @functor, glorot_uniform, convfilter, calc_padding, expand, create_bias, Zeros
using NNlib: ∇conv_data, ∇conv_filter
using Zygote: pullback, @adjoint, broadcasted, Numeric, _pullback
using ChainRulesCore; 
using ChainRulesCore: NoTangent, @thunk

include("layers.jl")
export GenDense, GenConv
include("activation_functions.jl")
export sign_STE, stoc_sign_STE, hardtanh_AdaSTE, hardσ
include("similarity_functions.jl")
export radialSim, radialSim_asym, matmul, matmul_asym_∂x, matmul_blocked_∂x, conv_asym_∂x
include("forwardmappings.jl")
export linear, linear_asym_∂x, radial, radial_asym_∂x, linear_binary_weights, linear_stoc_binary_weights
include("losses.jl")
export direct_feedback_loss
end
