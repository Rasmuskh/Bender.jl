"""
Generalized version of Flux's Dense layer. 

    GenDense(in=>out, σ=identity; 
             ω = identity, ψ = *, init = glorot_uniform, 
             bias=true, γ=Flux.Zeros()

Can also be initialized with an additional set of trainable weights 

    GenDense(in=>out, in_asym=>out_asym, σ = identity; 
             ω = identity, ψ = *, init = glorot_uniform, 
             bias=true, bias_asym=true, γ=Flux.Zeros())

To implement Feedback alignment you need to specify the similarity function `ψ=matmul_asym_∂x`.
```
julia> using Flux, Bender
julia> GenDense(20=>10, 10=>20, relu; ψ=matmul_asym_∂x)
julia> (a::GenDense)(x::AbstractVecOrMat) = a.σ.(a.ψ(a.ω.(a.weight), x, a.weight_asym) .+ a.bias) # redefine forward pass to also take weight_asym as input
GenDense(size(weight)=(10, 20), size(weight_asym)=(20, 10), σ=relu, ψ=matmul_asym_∂x)
```
To implement a layer with binary {-1,1} weights and neurons, which uses a deterministic 
straight-through estimator in the backwards pass you need to specify an activation function
for both the weights and neurons.
```
julia> using, Bender
julia> GenDense(20=>10, sign_STE; ω=sign_STE)
GenDense(size(weight)=(10, 20), σ=sign_STE, ω=sign_STE)
```
"""
    struct GenDense{F1, F2, F3, M1<:AbstractMatrix, M2, M3, B1, B2}
    weight::M1
    weight_asym::M2 
    bias::B1
    bias_asym::B2
    γ::M3 # Additional parameter, which may be used e.g. for annealing custom activation functions. 
    σ::F1 # activation function
    ω::F2 # weight activation function
    ψ::F3 # correlation/similarity measure
    function GenDense(weight::M1, weight_asym::M2, bias = true, bias_asym=true, γ = Flux.Zeros(), σ::F1 = identity, ω::F2 = identity, ψ::F3 = *) where {M1<:AbstractMatrix, M2, F1, F2, F3}
        new{F1, F2, F3, M1, M2, typeof(γ), typeof(bias), typeof(bias_asym)}(weight, weight_asym, bias, bias_asym, γ, σ, ω, ψ)
    end
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity; 
    ω = identity, ψ = *, init = glorot_uniform, bias=true, γ=Flux.Zeros())

    weight = init(out, in)
    bias = create_bias(weight, bias, out)
    # No dims specified for weights_asym, so return Flux.Zeros(). See documentation on Flux.Zeros for details
    weight_asym = Flux.Zeros()
    bias_asym = Flux.Zeros()

    return GenDense(weight, weight_asym, bias, bias_asym, γ, σ, ω, ψ)
end

function GenDense((in, out)::Pair{<:Integer, <:Integer}, 
    (in_asym, out_asym)::Pair{<:Integer, <:Integer}, σ = identity; 
    ω = identity, ψ = *, init = glorot_uniform, bias=true, bias_asym=true, γ=Flux.Zeros())
    weight = init(out, in)
    bias = create_bias(weight, bias, out)
    weight_asym = init(out_asym, in_asym)
    bias_asym = create_bias(weight_asym, bias_asym, out_asym)

    return GenDense(weight, weight_asym, bias, bias_asym, γ, σ, ω, ψ)
end

@functor GenDense

function (a::GenDense)(x::AbstractVecOrMat)
    W, b, σ, ω, ψ = a.weight, a.bias, a.σ, a.ω, a.ψ
    #= The similarity function ψ should compute a measure of similarity between rows of W and columns of X =#
    return σ.(ψ(ω.(W), x) .+ b)
end

(a::GenDense)(x::AbstractArray) = 
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::GenDense)
    print(io, "GenDense(size(weight)=", size(l.weight))
    l.weight_asym isa AbstractArray && print(io, ", size(weight_asym)=", size(l.weight_asym))
    l.σ == identity || print(io, ", σ=", l.σ)
    l.ω == identity || print(io, ", ω=", l.ω)
    l.ψ == (*) || print(io, ", ψ=", l.ψ)
    l.bias == Zeros() && print(io, ", bias=false")
    l.weight_asym isa AbstractArray && l.bias_asym == Zeros() && print(io, ", bias_asym=false")
    l.γ == Zeros() || print(io, ", size(γ)=", size(l.γ))
    print(io, ")")
end

"""Generalized version of Flux's conv layer"""
struct GenConv{N, M, F1, F2, F3, A1, A2, V1, V2, A3}
    σ::F1
    ω::F2
    ψ::F3
    weight::A1
    weight_asym::A2
    bias::V1
    bias_asym::V2
    γ::A3
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function GenConv(weight::AbstractArray{T,N}, weight_asym, bias, bias_asym, σ = identity;
	ω = identity, ψ=conv, stride = 1, pad = 0, dilation = 1, groups = 1, γ=Flux.Zeros()) where {T,N}

	stride = expand(Val(N-2), stride)
	dilation = expand(Val(N-2), dilation)
	pad = calc_padding(GenConv, pad, size(weight)[1:N-2], dilation, stride)

	return GenConv(σ, ω, ψ, weight, weight_asym, bias, bias_asym, γ, stride, pad, dilation, groups)
end

# initialization function for the case where weight_asym will not be used (is set to Flux.Zeros()).
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
    ω=identity, ψ=conv, init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
	weight = convfilter(k, (ch[1] ÷ groups => ch[2]); init), 
	weight_asym = Flux.Zeros(),
	bias = true, γ=Flux.Zeros()) where N

    bias = create_bias(weight, bias, size(weight, N))
    bias_asym = Flux.Zeros()

    return GenConv(weight, weight_asym, bias, bias_asym, σ; ω, ψ, stride, pad, dilation, groups, γ)
end

# Initialization function for the case where weight asym will be used.
function GenConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, k_asym::NTuple{N,Integer}, ch_asym::Pair{<:Integer,<:Integer}, σ = identity;
    ω=identity, ψ=conv, init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
	weight = convfilter(k, (ch[1] ÷ groups => ch[2]); init), 
	weight_asym = convfilter(k_asym, (ch_asym[1] ÷ groups => ch_asym[2]); init), 
	bias = true, bias_asym=true, γ=Flux.Zeros()) where N

	bias = create_bias(weight, bias, size(weight, N))
	bias_asym = create_bias(weight_asym, bias_asym, size(weight_asym, N))

    return GenConv(weight, weight_asym, bias, bias_asym, σ; ω, ψ, stride, pad, dilation, groups, γ)
end

@functor GenConv

# defining the default forward pass
function (c::GenConv)(x::AbstractArray)
	weight, σ, ω, ψ, b = c.weight, c.σ, c.ω, c.ψ, reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
	cdims = DenseConvDims(x, weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
	σ.(ψ(x, ω.(weight), cdims) .+ b)
end

# Defining the show method
function Base.show(io::IO, l::GenConv)
	print(io, "GenConv(weight: ", size(l.weight)[1:ndims(l.weight)-2])
	print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
    l.weight_asym isa AbstractArray && print(io, ", weight_asym: ", size(l.weight_asym)[1:ndims(l.weight_asym)-2])
	l.weight_asym isa AbstractArray && print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
	l.σ == identity || print(io, ", ", l.σ)
	l.ω == identity || print(io, ", ", l.ω)
	l.ψ == conv || print(io, ", ", l.ψ)
	all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
	all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
	all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
	l.bias == Zeros() && print(io, ", bias=false")
    l.weight_asym isa AbstractArray && l.bias_asym == Zeros() && print(io, ", bias_asym=false")
    l.γ == Zeros() || print(io, ", size(γ)=", size(l.γ))
	print(io, ")")
end