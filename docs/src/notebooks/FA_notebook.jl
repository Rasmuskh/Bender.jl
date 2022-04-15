### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 006b3526-bbe4-11ec-3b9d-c370b170468c
using Pkg

# ╔═╡ e5821064-844f-4e8e-8130-877d79277516
Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")

# ╔═╡ fe0d1fc8-fef7-4507-8d3f-33570c4ce520
begin
	using Bender, Flux, MLDatasets, Statistics
	using Flux: onehotbatch, onecold, logitcrossentropy, throttle
	using Flux.Data: DataLoader
	using Parameters: @with_kw
	using DataFrames
end

# ╔═╡ dd479824-f764-442f-8d8e-32af66d420a9
md" # Direct Feedback Alignment"

# ╔═╡ 48144b2d-068c-4f7a-9392-0459d8582a02
md"## Handling Dependencies"

# ╔═╡ 994ecc8b-d1bc-4fe7-84c8-5f063f61cecf
md"## Utilities"

# ╔═╡ 55c31972-c3ae-4d38-8140-26d46b50d5fd
@with_kw mutable struct Args
    η::Float64 = 0.0003#3e-4       # learning rate
    batchsize::Int = 64   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

# ╔═╡ 98ca0077-d776-4135-957d-1418968ea81e
function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
    # Reshape Data in order to flatten each image into a vector
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true, partial=false)
    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)

    return train_data, test_data
end

# ╔═╡ 176951a7-b6e9-4d23-b3b5-0c06349a9bb1
function evaluate(data_loader, model)
    acc = 0
	l = 0
	numbatches = length(data_loader)
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
		l += logitcrossentropy(model(x), y)
    end
    return acc/numbatches, l/numbatches
end

# ╔═╡ 99dea73f-4ade-42b4-97c1-7a640377715b
md"## Defining the Training Loop"

# ╔═╡ 80cfefd2-fcdb-4ccb-93a9-070bacff447f
md"## Defining the model"

# ╔═╡ 1dc297ce-5aa4-4b76-aeb8-be6df3d4c6e5
function build_model()
    m = Chain(  GenDense(784=>512, 512=>784, relu; forward=linear_asym_∂x),
                GenDense(512=>256, 256=>512, relu; forward=linear_asym_∂x),
                GenDense(256=>10, 10=>256; forward=linear_asym_∂x)
            )
    return m
end

# ╔═╡ 8a451576-0486-4a15-8503-3561e937bc63
function train(; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

	# Create arrays for recording training metrics
	acc_train = zeros(Float32, args.epochs)
	acc_test = zeros(Float32, args.epochs)
	loss_train = zeros(Float32, args.epochs)
	loss_test = zeros(Float32, args.epochs)
	
    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    
    # Training
    opt = ADAM(args.η)
	for epoch=1:args.epochs	
        Flux.train!(loss, params(m), train_data, opt)
		acc_train[epoch], loss_train[epoch] = evaluate(train_data, m)
		acc_test[epoch], loss_test[epoch] = evaluate(test_data, m)
    end

	# Return trianing metrics as a DataFrame
	df = DataFrame([loss_train, loss_test, acc_train, acc_test], 
				   [:loss_train, :loss_test, :acc_train, :acc_test])
	return df
end

# ╔═╡ 66535e33-f728-4f4f-aeef-7d23ea4bdd9a
df = train(epochs=10)

# ╔═╡ Cell order:
# ╟─dd479824-f764-442f-8d8e-32af66d420a9
# ╟─48144b2d-068c-4f7a-9392-0459d8582a02
# ╠═006b3526-bbe4-11ec-3b9d-c370b170468c
# ╠═e5821064-844f-4e8e-8130-877d79277516
# ╠═fe0d1fc8-fef7-4507-8d3f-33570c4ce520
# ╟─994ecc8b-d1bc-4fe7-84c8-5f063f61cecf
# ╠═55c31972-c3ae-4d38-8140-26d46b50d5fd
# ╠═98ca0077-d776-4135-957d-1418968ea81e
# ╠═176951a7-b6e9-4d23-b3b5-0c06349a9bb1
# ╟─99dea73f-4ade-42b4-97c1-7a640377715b
# ╠═8a451576-0486-4a15-8503-3561e937bc63
# ╟─80cfefd2-fcdb-4ccb-93a9-070bacff447f
# ╠═1dc297ce-5aa4-4b76-aeb8-be6df3d4c6e5
# ╠═66535e33-f728-4f4f-aeef-7d23ea4bdd9a
