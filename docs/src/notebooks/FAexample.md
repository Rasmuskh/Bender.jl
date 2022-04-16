# Feedback Alignment
## Handling dependencies

```
using Pkg; Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")
```

```
begin
	using Bender, Flux, MLDatasets
	using Flux: onehotbatch, onecold, logitcrossentropy, throttle
	using Flux.Data: DataLoader
	using Parameters: @with_kw
	using DataFrames
end
```

## Utilities
```
@with_kw mutable struct Args
    η::Float64 = 0.0003     # learning rate
    batchsize::Int = 64     # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end
```

```
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



```
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

## Defining the training loop
```
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
# Defining the model
```
function build_model()
    m = Chain(GenDense(784=>128, 128=>784, relu; forward=linear_asym_∂x),
              GenDense(128=>64, 64=>128, relu; forward=linear_asym_∂x),
              GenDense(64=>10, 10=>64; forward=linear_asym_∂x))
    return m
end
```
# Training the model
```
df = train(epochs=10); round.(df, digits=3)
```

|    | loss_train | loss_test | acc_train | acc_test |
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