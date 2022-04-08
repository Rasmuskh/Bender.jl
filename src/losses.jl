"""
Error function which takes a vector of the hidden and output neurons states
as well as a vector of feedback matrices as arguments
"""
direct_feedback_loss(loss, target, X, B) = loss(X[end], target)

"""
This rrule pipes output errors directly back into the hidden layers via
the feedback matrices Bᵢ. To use it use Flux.activations to get hidden layers
activation states and pass the array of hidden states in as the argument x
along with an array of appropriately dimensioned feedback matrices.
"""
function ChainRulesCore.rrule(::typeof(direct_feedback_loss), loss, target::AbstractMatrix, x, B)
    E, ∂E = pullback(loss, x[end], target)

    function E_pullback(ΔΩ)
        ΔΩ = unthunk(ΔΩ)
        ∂Eᵒᵘᵗ, ∂Eᵗᵃʳᵍᵉᵗ = ∂E(ΔΩ)
        ∂Xⁱ = [@thunk(Bⁱ*∂Eᵒᵘᵗ) for Bⁱ in B] # Error signal sent to layer i
        ∂Xᵒᵘᵗ = @thunk(∂Eᵒᵘᵗ) # Output layers error signal (Same as for BP)
        ∂X = Tuple([∂Xⁱ..., ∂Xᵒᵘᵗ])
        return (NoTangent(), NoTangent(), ∂Eᵗᵃʳᵍᵉᵗ, ∂X, NoTangent())
    end

    return E, E_pullback
end
