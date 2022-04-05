using ChainRulesCore; 
using ChainRulesCore: NoTangent, @thunk

"""
Compute negative squared euclidean distance D between the rows of matrix W and the columns of matrix X.
Denoting the rows of W by index i and the columns of X by index j the elements of the output matrix is given by:
Dᵢⱼ = -||Wᵢ﹕ - X﹕ⱼ||² = 2Wᵢ﹕X﹕,j - ||Wᵢ﹕||^2 - ||X﹕ⱼ||².
"""
function radialSim(W, X)
    x2 = sum(abs2, X, dims=1) # sum the elements of each column
    W2 = sum(abs2, W, dims=2) # sum the elements of each row
    D = 2*matmul(W,X) .- W2 .- x2
    return D
end

"""
In the forward pass this function behaves just like radialSim, but in the backwards pass weight symmetry is broken by using matrix B rather than Wᵀ. See docstring for radialSim for more details.
"""
function radialSim(W, X, B)
    x2 = sum(abs2, X, dims=1) # sum the elements of each column
    W2 = sum(abs2, W, dims=2) # sum the elements of each row
    D = 2*matmul(W,X,B) .- W2 .- x2
    return D
end

# Todo: cosine similarity and projection rejection similarity functions
# https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PR_Product_A_Substitute_for_Inner_Product_in_Neural_Networks_ICCV_2019_paper.pdf

"""
Regular matrix multiplication.
"""
function matmul(W, X) 
    return W * X
 end

"""
Compute matrix multiplication, but takes an additional matrix B as input. 
B has same dims as Wᵀ, and is used in the backwards pass.
"""
function matmul(W, X, B) 
    return W * X
 end

 """
 rrule which uses feedback weights B instead of Wᵀ, which regular backpropagation would use.
 """
 function ChainRulesCore.rrule(::typeof(matmul), W::AbstractMatrix, X::AbstractMatrix, B::AbstractMatrix)
    y = matmul(W, X)
    function times_pullback(ΔΩ)
       ∂W = @thunk(ΔΩ * X')
       ∂X = @thunk(B * ΔΩ) # Use random feedback weight matrix B instead of Wᵀ. This is the main idea of feedback alignment.
       return (NoTangent(), ∂W, ∂X, NoTangent())
    end
    return y, times_pullback
 end

 """
 conv called with an additional feedback weight matrix. The forward pass gives the same results, but in the backwards pass the gradient will be different thanks to multiple dispatch.
 """
 function conv(x, w::AbstractArray{T, N}, wFB::AbstractArray{T, N}, cdims) where {T, N}
    return conv(x, w, cdims)
end

 """
 When conv is called with an additional weight matrix wFB, then we will use this custom rrule, which transports error backwards using wFB instead of W.
 """
 function ChainRulesCore.rrule(::typeof(conv), x, w, wFB, cdims)
    z = conv(x, w, cdims)
    function pullback(Δy)
        # When computing ∇conv_data the the feedback weights wFB are used instead of W. This is the main idea of feedback alignment.
        return NoTangent(), ∇conv_data(Δy, wFB, cdims), ∇conv_filter(x, Δy, cdims), NoTangent(), NoTangent()
    end
    return z, pullback
end

"""
Matrix multiplication with custom rrule
"""
matmul_dfa(W, X) = return W * X

"""
This rrule blocks the regular backprop pathway by returning NoTangent() instead of ∂X
"""
function ChainRulesCore.rrule(::typeof(matmul_dfa), W::AbstractMatrix, X::AbstractMatrix)
    y = W*X
    function matmul_dfa_pullback(ΔΩ::AbstractMatrix{Float32})
        ΔΩ = unthunk(ΔΩ)
        ∂W = @thunk(ΔΩ * X')
        return (NoTangent(), ∂W, NoTangent())
    end
    return y, matmul_dfa_pullback
end