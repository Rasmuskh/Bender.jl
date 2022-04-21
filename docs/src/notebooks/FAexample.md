# Feedback Alignment

This page contains a short tutorial on how to train a model using feedback alignment. It is loosely adapted from the Flux model zoo's MLP example.

Feedback alignment ([Lillicrap et al](https://arxiv.org/abs/1411.0247)) sometimes refered to as Random Backpropagation is an attempt at making neural network trainig more biologically plausible. Usually the backpropagation of error algorithm is used to train networks, but it relies on a symmetry between the weights used to make predicitons in the forward pass and the weights used for propagating error signals backwards in order to compute weight updates. Since synapses for the most part transmit signals in one direction it seems unlikely that exact one to one synaptic symmetry as required by backprop could somehow arise in biological brains. 

For this reason Lillicrap et al proposed to transport errors backwards using a set of fixed random weight matrices. Surprisingly this works fairly well because the weights used in the forwards pass learn to approzimately align with the feedback weights.

 The following sections contain some boilerplate code. You can expand them to show the details.

## Handling dependencies
```@raw html
<details><summary>show details</summary>
```

```julia
using Pkg; Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")
```

```julia
begin
	using Bender, Flux, MLDatasets
	using Flux: onehotbatch, onecold, logitcrossentropy, throttle
	using Flux.Data: DataLoader
	using Parameters: @with_kw
	using DataFrames
end
```

```@raw html
</details>
```

## Utilities

```@raw html
<details>
  <summary>show details</summary>
```

```julia
@with_kw mutable struct Args
    η::Float64 = 0.0003     # learning rate
    batchsize::Int = 64     # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end
```

```julia
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
```

```julia
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
```
```@raw html
</details>
```

## Defining the training loop

```@raw html
<details>
  <summary>show details</summary>
```

```julia
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
```
```@raw html
</details>
```
# Defining the model
Feedback Alignment uses two sets of weights, one for making predictions and one for transporting error signals backwards, so we need to initialize the `GenDense` layer with an extra set of weights. We also need to specify the forward mapping we will use, which in this case is `linear_asym_∂x`. 

For more details see the documentation and/or source code for the forward mapping `linear_asym_∂x`, which behaves as a regular fully connected layer in the forwards pass, but uses the second set of weights in the backwards pass. 

```julia
function build_model()
    m = Chain(GenDense(784=>128, 128=>784, relu; forward=linear_asym_∂x),
              GenDense(128=>64, 64=>128, relu; forward=linear_asym_∂x),
              GenDense(64=>10, 10=>64; forward=linear_asym_∂x))
    return m
end
```
# Training the model
The network quickly learns to solve the problem fairly well even though we are using fixed random matrices to propagate errors backwards. Below we train the network for ten epochs. The train function stores the model's loss and accuracy on the train and test set in a DataFrame.

```julia
df = train(epochs=10); round.(df, digits=3)
```
|Epoch| loss_train | loss_test | acc_train | acc_test |
|:--:|:----------:|:---------:|:---------:|:--------:|
|  1 | 0.418      | 0.413     | 0.881     | 0.884    |
|  2 | 0.317      | 0.318     | 0.909     | 0.91     |
|  3 | 0.269      | 0.274     | 0.923     | 0.922    |
|  4 | 0.232      | 0.238     | 0.934     | 0.932    |
|  5 | 0.201      | 0.208     | 0.943     | 0.941    |
|  6 | 0.176      | 0.187     | 0.951     | 0.946    |
|  7 | 0.155      | 0.169     | 0.956     | 0.951    |
|  8 | 0.138      | 0.156     | 0.96      | 0.955    |
|  9 | 0.125      | 0.145     | 0.964     | 0.956    |
| 10 | 0.113      | 0.136     | 0.967     | 0.959    |

# References
Timothy P. Lillicrap et al, 2014, Random feedback weights support learning in deep neural networks, [Link](https://arxiv.org/abs/1411.0247)