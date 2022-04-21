"""
Deterministic straight-through estimator for the sign function.
Reference: https://arxiv.org/abs/1308.3432, https://arxiv.org/abs/1511.00363
"""
sign_STE(x) = sign(x)

function ChainRulesCore.rrule(::typeof(sign_STE), x::AbstractArray)
    z = sign.(x)
    function pullback(Δy)
        return NoTangent(), Δy
    end
    return z, pullback
end

# Hack to avoid zygote using forward mode differentiation instead of the above rrule.
@adjoint function broadcasted(::typeof(sign_STE), x::Numeric)
    _pullback(sign_STE, x)
end

"""A piece-wise linear function. for x<-1 it has value 0. For -1<x<1 it has value x. For x>1 it has value 1. It is defined as:
    
    hardσ(x) = max(0, min(1, (x+1)/2))
"""
hardσ(x) = max(0, min(1, (x+1)/2))
       
"""
A stochastic straight-through estimator version of the sign function.
References: https://arxiv.org/abs/1308.3432, https://arxiv.org/abs/1511.00363

"""
function stoc_sign_STE(x)
    p = hardσ(x)
    plus = p>=rand(typeof(x))
    nega = plus -1
    x_bin = plus + nega     
    return x_bin
end

function ChainRulesCore.rrule(::typeof(stoc_sign_STE), x::AbstractArray)
    z = stoc_sign_STE.(x)
    function pullback(Δy)
        return NoTangent(), Δy
    end
    return z, pullback
end

# Hack to avoid zygote using forward mode differentiation instead of the above rrule.
@adjoint function broadcasted(::typeof(stoc_sign_STE), x::Numeric)
    _pullback(stoc_sign_STE, x)
end

""" 
Adaptive Straight-through estimator designed for weight quantization.
Reference: https://arxiv.org/abs/2112.02880
"""
hardtanh_AdaSTE(x) = hardtanh(x)

#=
(z - zbar)/tau = 
case1:    0             if x<=-1 and l'>=0 or x>=1 and l'<=0
case2:    l'(-2/(x-1))  if x<=-1 and l'<=0
case3:    l'(2/(x+1))   if x>=1 and l'>=0
case4:    l'            otherwise
=#

function ChainRulesCore.rrule(::typeof(hardtanh_AdaSTE), x::AbstractArray)
    z = hardtanh.(x)
    function pullback(Δy)
            ϵ = 0.f0
            case1 = ((x .<= -1) .& (Δy .>= ϵ)) .| ((x .>= 1) .& (Δy .<= -ϵ))
            case2 = (x .<= -1) .& (Δy .<= -ϵ)
            case3 = (x .>= 1) .& (Δy .>= ϵ)
            case4 = .!(case1 .& case2 .& case3)
            Δz = Δy .* (case2 .* (-2 ./ (x .- 1)) .+ case3 .* (2 ./ (x .+ 1)) .+ case4)
            
            # case23 = (abs.(x) .>= 1) .& (abs.(Δy) .>= ϵ)
            # case4 = .!(case1 .& case23)
            # Δz = Δy .* (case23 .* (2 ./ (abs.(x) .+ 1))  .+ case4)
            return NoTangent(), Δz
    end
    return z, pullback
end

# Hack to avoid zygote using forward mode differentiation instead of the above rrule.
@adjoint function broadcasted(::typeof(hardtanh_AdaSTE), x::Numeric)
    _pullback(hardtanh_CSTE, x)
end