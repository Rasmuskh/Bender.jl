# Fully connected mappings

"""
Matrix multiply weight matrix with x and add bias
"""
function linear(a, x)
    W, b, = a.W, a.bias
    return W*x .+ b
end

"""
behaves identical to `linear` in the forward pass, but relies on matmul_asym_∂x, 
which causes errors to be backpropagated using a set of auxiliary weights in
the backwards pass. See `matmul_asym_∂x`.
"""
function linear_asym_∂x(a, x)
    W, b, V = a.W, a.bias, a.V
    return matmul_asym_∂x(W, x, V) .+ b
end

"""
behaves identical to `linear` in the forward pass, but relies on matmul_blocked_∂x, 
which prevents error signal from passing through this layer to earlier layers. 
This is useful in direct feedback alignment experiments where you want to pipe errors
directly from the output loss to individual layers. See `matmul_blocked_∂x`.
"""
function linear_blocked_∂x(a, x)
    W, b = a.W, a.bias
    return matmul_blocked_∂x(W, x) .+ b
end

"""
Calls radialSim and computes the negative squared euclidean distance D between the rows ofthe 
layers W matrix and the columns of matrix X. See `radialSim`.
"""
function radial(a, x) 
    W, b = a.W, a.bias
    return radialSim(W, x) .+ b
end
"""
behaves identical to `radial` in the forward pass, but relies on radialSim_asym_∂x, 
which causes errors to be backpropagated using a set of auxiliary weights in
the backwards pass. See `radialSim_asym_∂x`."""
function radial_asym_∂x(a, x) 
    W, b, B = a.W, a.bias, a.V
    return radialSim_asym(W, x, B) .+ b
end

"""
Regular forward pass (matmul and bias addition) with a binary activation
function applied to the weights.
"""
function linear_binary_weights(a, x)
    W, b, = a.W, a.bias
    return sign_STE.(W)*x .+ b
end

"""
Regular forward pass (matmul and bias addition) with a binary stochastic
activation function applied to the weights.
"""
function linear_stoc_binary_weights(a, x)
    W, b, = a.W, a.bias
    return stoc_sign_STE.(W)*x .+ b
end

# Conv mappings
"""
Forward mapping for regular convolutional layer
"""
function conv_linear(c, x)
	W = c.W
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, W; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv(x, W, cdims) .+ b
end

"""
In the forward pass this behaves identical to `conv_linear`. 
Relies on `conv_asym_∂x`, which causes errors to be backpropagated 
using a set of auxiliary weights in the backwards pass. See `conv_asym_∂x`.
"""
function conv_linear_asym_∂x(c, x)
	W = c.W
    V = c.V
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, W; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv_asym_∂x(x, W, V, cdims) .+ b
end