"""
Generalized version of Flux's Dense layer. The `forward` keyword allows you to choose the form of the forward mapping.

    GenDense(in=>out, σ=identity; 
             init = glorot_uniform, 
             bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)

Can also be initialized with an additional set of trainable weights 

    GenDense(in=>out, in_asym=>out_asym, σ = identity; 
             init = glorot_uniform, 
             bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)
             
The layer has additinal keyword arguments α and β, which default to Flux.Zeros. These are useful if you
need an extra set of weights for for your forward pass (if you for example wish to anneal an activation function).
"""
struct GenDense{F1, F2, M1<:AbstractMatrix, M2, M3, M4, B}
    weight::M1
    weight_asym::M2 
    bias::B
    α::M3 # Additional parameter, which may be used e.g. for annealing custom activation functions. Defaults to Flux.Zeros()
    β::M4 # Additional parameter, which may be used e.g. for annealing custom activation functions. Defaults to Flux.Zeros()
    σ::F1 # activation function
    forward::F2 # Forward pass function (without applying the activation function σ)
    function GenDense(weight::M1, weight_asym::M2, bias = true, α = Flux.Zeros(), β = Flux.Zeros(), σ::F1 = identity, forward::F2 = linear) where {M1<:AbstractMatrix, M2, F1, F2}
        new{F1, F2, M1, M2, typeof(α), typeof(β), typeof(bias)}(weight, weight_asym, bias, α, β, σ, forward)
    end
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity; 
    init = glorot_uniform, bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)

    weight = init(out, in)
    bias = create_bias(weight, bias, out)
    # No dims specified for weights_asym, so return Flux.Zeros(). See documentation on Flux.Zeros for details
    weight_asym = Flux.Zeros()

    return GenDense(weight, weight_asym, bias, α, β, σ, forward)
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, 
    (in_asym, out_asym)::Pair{<:Integer, <:Integer}, σ = identity; 
    init = glorot_uniform, bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)
    weight = init(out, in)
    bias = create_bias(weight, bias, out)
    weight_asym = init(out_asym, in_asym)

    return GenDense(weight, weight_asym, bias, α, β, σ, forward)
end

@functor GenDense

function (a::GenDense)(x::AbstractVecOrMat)
    return a.σ.(a.forward(a, x))
end

(a::GenDense)(x::AbstractArray) = 
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::GenDense)
    print(io, "GenDense(size(weight)=", size(l.weight))
    l.weight_asym isa AbstractArray && print(io, ", size(weight_asym)=", size(l.weight_asym))
    l.σ == identity || print(io, ", σ=", l.σ)
    l.forward == linear || print(io, ", forward=", l.forward)
    l.bias == Zeros() && print(io, ", bias=false")
    l.α == Zeros() || print(io, ", size(α)=", size(l.α))
    l.β == Zeros() || print(io, ", size(β)=", size(l.β))
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
struct GenConv{N, M, F1, F2, A1, A2, V, A3, A4}
    σ::F1
    forward::F2
    weight::A1
    weight_asym::A2
    bias::V
    α::A3
    β::A4
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function GenConv(weight::AbstractArray{T,N}, weight_asym, bias, σ = identity;
                stride = 1, pad = 0, dilation = 1, groups = 1, α=Flux.Zeros(), β=Flux.Zeros(), forward=conv_linear) where {T,N}

	stride = expand(Val(N-2), stride)
	dilation = expand(Val(N-2), dilation)
	pad = calc_padding(GenConv, pad, size(weight)[1:N-2], dilation, stride)

    bias = create_bias(weight, bias, size(weight, N))
    
	return GenConv(σ, forward, weight, weight_asym, bias, α, β, stride, pad, dilation, groups)
end

# initialization function for the case where weight_asym will not be used (is set to Flux.Zeros()).
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
                bias = true, α=Flux.Zeros(), β=Flux.Zeros(),
                forward=conv_linear) where N
    
    weight = convfilter(k, (ch[1] ÷ groups => ch[2]); init)
    weight_asym = Flux.Zeros()


    return GenConv(weight, weight_asym, bias, σ; stride, pad, dilation, groups, α, β, forward)
end

# Initialization function for the case where weight asym will be used.
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, k_asym::NTuple{N,Integer}, ch_asym::Pair{<:Integer,<:Integer}, σ = identity;
    init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
	bias = true, α=Flux.Zeros(), β=Flux.Zeros(),
    forward=conv_linear) where N

    weight = convfilter(k, (ch[1] ÷ groups => ch[2]); init)
	weight_asym = convfilter(k_asym, (ch_asym[1] ÷ groups => ch_asym[2]); init)

    return GenConv(weight, weight_asym, bias, σ; stride, pad, dilation, groups, α, β, forward)
end

@functor GenConv

function (c::GenConv)(x::AbstractArray)
	return c.σ.(c.forward(c, x))
end

# Defining the show method
function Base.show(io::IO, l::GenConv)
	print(io, "GenConv(weight: ", size(l.weight)[1:ndims(l.weight)-2])
	print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
    l.weight_asym isa AbstractArray && print(io, ", weight_asym: ", size(l.weight_asym)[1:ndims(l.weight_asym)-2])
	l.weight_asym isa AbstractArray && print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
	l.σ == identity || print(io, ", ", l.σ)
    l.forward == conv_linear || print(io, ", forward=", l.forward)
	all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
	all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
	all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
	l.bias == Zeros() && print(io, ", bias=false")
    l.α == Zeros() || print(io, ", size(α)=", size(l.α))
    l.β == Zeros() || print(io, ", size(β)=", size(l.β))
	print(io, ")")
end