# Fully connected mappings

"""
Matrix multiply layers weight matrix with x and add bias
"""
function linear(a, x)
    W, b, = a.weight, a.bias
    return W*x .+ b
end

"""
behaves identical to `linear` in the forward pass, but relies on matmul_asym_∂x, 
which causes errors to be backpropagated using a set of auxiliary weights'in
the backwards pass. See `matmul_asym_∂x`.
"""
function linear_asym_∂x(a, x)
    W, b, B = a.weight, a.bias, a.weight_asym
    return matmul_asym_∂x(W, x, B) .+ b
end

"""
behaves identical to `linear` in the forward pass, but relies on matmul_blocked_∂x, 
which prevents error signal from passing through this layer to earlier layers. 
This is useful in direct feedback alignment experiments where you want to pipe errors
directly from the output loss to individual layers. See `matmul_blocked_∂x`.
"""
function linear_blocked_∂x(a, x)
    W, b, B = a.weight, a.bias, a.weight_asym
    return matmul_blocked_∂x(W, x, B) .+ b
end

"""
Calls radialSim and computes the negative squared euclidean distance D between the rows ofthe 
layers weight matrix and the columns of matrix X. See `radialSim`.
"""
function radial(a, x) 
    W, b = a.weight, a.bias
    return radialSim(W, x) .+ b
end
"""
behaves identical to `radial` in the forward pass, but relies on radialSim_asym_∂x, 
which causes errors to be backpropagated using a set of auxiliary weights in
the backwards pass. See `radialSim_asym_∂x`."""
function radial_asym_∂x(a, x) 
    W, b, B = a.weight, a.bias, a.weight_asym
    return radialSim_asym(W, x, B) .+ b
end

"""
Regular forward pass (matmul and bias addition) with a binary activation
function applied to the weights.
"""
function linear_binary_weights(a, x)
    W, b, = a.weight, a.bias
    return sign_STE.(W)*x .+ b
end

"""
Regular forward pass (matmul and bias addition) with a binary stochastic
activation function applied to the weights.
"""
function linear_stoc_binary_weights(a, x)
    W, b, = a.weight, a.bias
    return stoc_sign_STE.(W)*x .+ b
end

# Conv mappings
"""
Forward mapping for regular convolutional layer
"""
function conv_linear(c, x)
	weight = c.weight
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv(x, weight, cdims) .+ b
end

"""
In the forward pass this behaves identical to `conv_linear`. 
Relies on `conv_asym_∂x`, which causes errors to be backpropagated 
using a set of auxiliary weights in the backwards pass. See `conv_asym_∂x`.
"""
function conv_linear_asym_∂x(c, x)
	weight = c.weight
    weight_asym = c.weight_asym
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv_asym_∂x(x, weight, weight_asym, cdims) .+ b
end