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