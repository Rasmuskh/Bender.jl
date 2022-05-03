# Fully connected mappings

function linear(a, x)
    W, b, = a.weight, a.bias
    return W*x .+ b
end

function linear_asym_∂x(a, x)
    W, b, B = a.weight, a.bias, a.weight_asym
    return matmul_asym_∂x(W, x, B) .+ b
end

function radial(a, x) 
    W, b = a.weight, a.bias
    return radialSim(W, x) .+ b
end

function radial_asym_∂x(a, x) 
    W, b, B = a.weight, a.bias, a.weight_asym
    return radialSim_asym(W, x, B) .+ b
end

function linear_binary_weights(a, x)
    W, b, = a.weight, a.bias
    return sign_STE.(W)*x .+ b
end

function linear_stoc_binary_weights(a, x)
    W, b, = a.weight, a.bias
    return stoc_sign_STE.(W)*x .+ b
end

# Conv mappings

function conv_linear(c, x)
	weight = c.weight
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv(x, weight, cdims) .+ b
end

function conv_linear_asym_∂x(c, x)
	weight = c.weight
    weight_asym = c.weight_asym
    b = reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	return conv_asym_∂x(x, weight, weight_asym, cdims) .+ b
end