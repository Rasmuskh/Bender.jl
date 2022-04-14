using Bender
using Flux
using Test

@testset "Test RBF chain" begin
    x = rand(Float32, 3, 8)
    y = rand(Float32, 2, 8)

    m1 = Chain( GenDense(3=>4, σ; forward=radial), 
                GenDense(4=>2, σ; forward=radial))

    gs1 = gradient(Flux.params(m1)) do
        Flux.Losses.mse(m1(x), y)
    end

    @test size(gs1[m1[1].weight]) == size(m1[1].weight)
    @test size(gs1[m1[2].weight]) == size(m1[2].weight)

    m2 = Chain( GenDense(3=>4, 4=>3, σ; forward=radial_asym_∂x), 
                GenDense(4=>2, 2=>4, σ; forward=radial_asym_∂x))

    gs2 = gradient(Flux.params(m2)) do
        sum(abs2, m2(x))
    end

    @test gs1[m1[1].weight] != gs2[m2[1].weight]
    @test gs1[m1[2].weight] != gs2[m2[2].weight]
    
    @test size(gs1[m1[1].weight]) == size(gs2[m2[1].weight]) 
    @test size(gs1[m1[2].weight]) == size(gs2[m2[2].weight])

end
