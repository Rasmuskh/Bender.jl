"""
Generalized version of Flux's Dense layer. The `forward` keyword allows you to choose the form of the forward mapping.

    GenDense(in=>out, σ=identity; 
             init = glorot_uniform, 
             bias=true, α=false, β=false, forward=linear)

Can also be initialized with an additional set of trainable weights 

    GenDense(in=>out, in_asym=>out_asym, σ = identity; 
             init = glorot_uniform, 
             bias=true, α=false, β=false, forward=linear)
             
The layer has additinal keyword arguments α and β, which default to Flux.Zeros. These are useful if you
need an extra set of weights for for your forward pass (if you for example wish to anneal an activation function).
"""
struct GenDense{F1, F2, M1<:AbstractMatrix, M2, M3, M4, M5, M6, B}
    W::M1
    V::M2 
    #= α, β, μ, λ: Additional parameters, which may be 
    used in custom forward/backward pass. Defaults to false=#
    α::M3
    β::M4
    μ::M5
    λ::M6
    bias::B
    σ::F1 # activation function
    forward::F2 # Forward pass function (without applying the activation function σ)
    function GenDense(W::M1, V::M2, α::M3, β::M4, μ::M5, λ::M6, bias = true, σ::F1 = identity, forward::F2 = linear) where {M1<:AbstractMatrix, M2, M3, M4, M5, M6, F1, F2}
        bias = create_bias(W, bias, size(W,1))
        new{F1, F2, M1, M2, M3, M4, M5, M6, typeof(bias)}(W, V, α, β, μ, λ, bias, σ, forward)
    end
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity; 
    init = glorot_uniform, 
    bias=true, 
    V=false, α=false, β=false, μ=false, λ=false, 
    forward=linear)

    W = init(out, in)
    return GenDense(W, V, α, β, μ, λ, bias, σ, forward)
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, 
    (in_asym, out_asym)::Pair{<:Integer, <:Integer}, σ = identity; 
    init = glorot_uniform, 
    bias=true, 
    α=false, β=false, μ=false, λ=false, 
    forward=linear)

    W = init(out, in)
    # bias = create_bias(W, bias, out)
    V = init(out_asym, in_asym)

    return GenDense(W, V, α, β, μ, λ, bias, σ, forward)
end

@functor GenDense

function (a::GenDense)(x::AbstractVecOrMat)
    return a.σ.(a.forward(a, x))
end

(a::GenDense)(x::AbstractArray) = 
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.getproperty(a::GenDense, sym::Symbol)
    if sym === :weight
        return a.W
    else
        return getfield(a, sym)
    end
end

function Base.show(io::IO, l::GenDense)
    print(io, "GenDense(size(W)=", size(l.W))
    l.V isa AbstractArray && print(io, ", size(V)=", size(l.V))
    l.σ == identity || print(io, ", σ=", l.σ)
    l.forward == linear || print(io, ", forward=", l.forward)
    l.bias == false && print(io, ", bias=false")
    l.α == false || print(io, ", size(α)=", size(l.α))
    l.β == false || print(io, ", size(β)=", size(l.β))
    l.μ == false || print(io, ", size(μ)=", size(l.μ))
    l.λ == false || print(io, ", size(λ)=", size(l.λ))
    print(io, ")")
end



"""
Generalized version of Flux's conv layer. 
The `forward` keyword allows you to choose the form of the forward mapping and defaults to linear.
This layer can be initialized with either one or two set of filters 
(a second set of filters is useful for feedback alignment experiments).

    GenConv((k, k), ch_in=>ch_out, σ=identity; forward=linear)

    GenConv((k, k), ch_in=>ch_out_(k_asym, k_asym), ch_in_asym=>ch_out_asym, σ=identity; forward=linear)

The layer has additinal keyword arguments α and β, which default to Flux.Zeros. These are useful if you
need an extra set of weights for for your forward pass (if you for example wish to anneal an activation function).
"""
struct GenConv{N, M, F1, F2, A1, A2, V, A3, A4, A5, A6}
    σ::F1
    forward::F2
    W::A1
    V::A2
    bias::V
    α::A3
    β::A4
    μ::A5
    λ::A6
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function GenConv(W::AbstractArray{T,N}, V, bias, σ = identity;
                stride = 1, pad = 0, dilation = 1, groups = 1, α=false, β=false, μ=false, λ=false, forward=conv_linear) where {T,N}

	stride = expand(Val(N-2), stride)
	dilation = expand(Val(N-2), dilation)
	pad = calc_padding(GenConv, pad, size(W)[1:N-2], dilation, stride)

    bias = create_bias(W, bias, size(W, N))
    
	return GenConv(σ, forward, W, V, bias, α, β, μ, λ, stride, pad, dilation, groups)
end

# initialization function for the case where V is either false or passed explicitly.
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
                bias = true, V=false, α=false, β=false, μ=false, λ=false,
                forward=conv_linear) where N
    
    W = convfilter(k, (ch[1] ÷ groups => ch[2]); init)


    return GenConv(W, V, bias, σ; stride, pad, dilation, groups, α, β, μ, λ, forward)
end

# Initialization function for the case where V is initialized the same way as W.
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, k_asym::NTuple{N,Integer}, ch_asym::Pair{<:Integer,<:Integer}, σ = identity;
    init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
	bias = true, α=false, β=false, μ=false, λ=false,
    forward=conv_linear) where N

    W = convfilter(k, (ch[1] ÷ groups => ch[2]); init)
	V = convfilter(k_asym, (ch_asym[1] ÷ groups => ch_asym[2]); init)

    return GenConv(W, V, bias, σ; stride, pad, dilation, groups, α, β, μ, λ, forward)
end

@functor GenConv

function (c::GenConv)(x::AbstractArray)
	return c.σ.(c.forward(c, x))
end

function Base.getproperty(a::GenConv, sym::Symbol)
    if sym === :weight
        return a.W
    else
        return getfield(a, sym)
    end
end

# Defining the show method
function Base.show(io::IO, l::GenConv)
	print(io, "GenConv(W: ", size(l.W)[1:ndims(l.W)-2])
	print(io, ", ", size(l.W, ndims(l.W)-1), " => ", size(l.W, ndims(l.W)))
    l.V isa AbstractArray && print(io, ", V: ", size(l.V)[1:ndims(l.V)-2])
	l.V isa AbstractArray && print(io, ", ", size(l.W, ndims(l.W)-1), " => ", size(l.W, ndims(l.W)))
	l.σ == identity || print(io, ", ", l.σ)
    l.forward == conv_linear || print(io, ", forward=", l.forward)
	all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
	all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
	all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
	l.bias == false && print(io, ", bias=false")
    l.α == false || print(io, ", size(α)=", size(l.α))
    l.β == false || print(io, ", size(β)=", size(l.β))
    l.μ == false || print(io, ", size(μ)=", size(l.μ))
    l.λ == false || print(io, ", size(λ)=", size(l.λ))
	print(io, ")")
end