var documenterSearchIndex = {"docs":
[{"location":"functionindex/#Index","page":"Function index","title":"Index","text":"","category":"section"},{"location":"functionindex/","page":"Function index","title":"Function index","text":"Modules = [Bender]","category":"page"},{"location":"notebooks/FAexample/#Feedback-Alignment","page":"Example: Feedback Alignment","title":"Feedback Alignment","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"This page contains a short tutorial on how to train a model using feedback alignment. It is loosely adapted from the Flux model zoo's MLP example.","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"Feedback alignment (Lillicrap et al) sometimes refered to as Random Backpropagation is an attempt at making neural network trainig more biologically plausible. Usually the backpropagation of error algorithm is used to train networks, but it relies on a symmetry between the weights used to make predicitons in the forward pass and the weights used for propagating error signals backwards in order to compute weight updates. Since synapses for the most part transmit signals in one direction it seems unlikely that exact one to one synaptic symmetry as required by backprop could somehow arise in biological brains. ","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"For this reason Lillicrap et al proposed to transport errors backwards using a set of fixed random weight matrices. Surprisingly this works fairly well because the weights used in the forwards pass learn to approzimately align with the feedback weights.","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"The following sections contain some boilerplate code. You can expand them to show the details.","category":"page"},{"location":"notebooks/FAexample/#Handling-dependencies","page":"Example: Feedback Alignment","title":"Handling dependencies","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"<details><summary>show details</summary>","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"using Pkg; Pkg.add(url=\"https://github.com/Rasmuskh/Bender.jl.git\")","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"begin\n\tusing Bender, Flux, MLDatasets\n\tusing Flux: onehotbatch, onecold, logitcrossentropy, throttle\n\tusing Flux.Data: DataLoader\n\tusing Parameters: @with_kw\n\tusing DataFrames\nend","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"</details>","category":"page"},{"location":"notebooks/FAexample/#Utilities","page":"Example: Feedback Alignment","title":"Utilities","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"<details>\n  <summary>show details</summary>","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"@with_kw mutable struct Args\n    η::Float64 = 0.0003     # learning rate\n    batchsize::Int = 64     # batch size\n    epochs::Int = 10        # number of epochs\n    device::Function = gpu  # set as gpu, if gpu available\nend","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"function getdata(args)\n    ENV[\"DATADEPS_ALWAYS_ACCEPT\"] = \"true\"\n\n    # Loading Dataset\t\n    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)\n    xtest, ytest = MLDatasets.MNIST.testdata(Float32)\n\n    # Reshape Data in order to flatten each image into a vector\n    xtrain = Flux.flatten(xtrain)\n    xtest = Flux.flatten(xtest)\n\n    # One-hot-encode the labels\n    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)\n\n    # Batching\n    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true, partial=false)\n    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)\n\n    return train_data, test_data\nend","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"function evaluate(data_loader, model)\n    acc = 0\n\tl = 0\n\tnumbatches = length(data_loader)\n    for (x,y) in data_loader\n        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)\n\t\tl += logitcrossentropy(model(x), y)\n    end\n    return acc/numbatches, l/numbatches\nend","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"</details>","category":"page"},{"location":"notebooks/FAexample/#Defining-the-training-loop","page":"Example: Feedback Alignment","title":"Defining the training loop","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"<details>\n  <summary>show details</summary>","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"function train(; kws...)\n    # Initializing Model parameters \n    args = Args(; kws...)\n\n\t# Create arrays for recording training metrics\n\tacc_train = zeros(Float32, args.epochs)\n\tacc_test = zeros(Float32, args.epochs)\n\tloss_train = zeros(Float32, args.epochs)\n\tloss_test = zeros(Float32, args.epochs)\n\t\n    # Load Data\n    train_data,test_data = getdata(args)\n\n    # Construct model\n    m = build_model()\n    train_data = args.device.(train_data)\n    test_data = args.device.(test_data)\n    m = args.device(m)\n    loss(x,y) = logitcrossentropy(m(x), y)\n    \n    # Training\n    opt = ADAM(args.η)\n\tfor epoch=1:args.epochs\t\n        Flux.train!(loss, params(m), train_data, opt)\n\t\tacc_train[epoch], loss_train[epoch] = evaluate(train_data, m)\n\t\tacc_test[epoch], loss_test[epoch] = evaluate(test_data, m)\n    end\n\n\t# Return trianing metrics as a DataFrame\n\tdf = DataFrame([loss_train, loss_test, acc_train, acc_test], \n\t\t\t\t   [:loss_train, :loss_test, :acc_train, :acc_test])\n\treturn df\nend","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"</details>","category":"page"},{"location":"notebooks/FAexample/#Defining-the-model","page":"Example: Feedback Alignment","title":"Defining the model","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"Feedback Alignment uses two sets of weights, one for making predictions and one for transporting error signals backwards, so we need to initialize the GenDense layer with an extra set of weights. We also need to specify the forward mapping we will use, which in this case is linear_asym_∂x. ","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"For more details see the documentation and/or source code for the forward mapping linear_asym_∂x, which behaves as a regular fully connected layer in the forwards pass, but uses the second set of weights in the backwards pass. ","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"function build_model()\n    m = Chain(GenDense(784=>128, 128=>784, relu; forward=linear_asym_∂x),\n              GenDense(128=>64, 64=>128, relu; forward=linear_asym_∂x),\n              GenDense(64=>10, 10=>64; forward=linear_asym_∂x))\n    return m\nend","category":"page"},{"location":"notebooks/FAexample/#Training-the-model","page":"Example: Feedback Alignment","title":"Training the model","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"The network quickly learns to solve the problem fairly well even though we are using fixed random matrices to propagate errors backwards. Below we train the network for ten epochs. The train function stores the model's loss and accuracy on the train and test set in a DataFrame.","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"df = train(epochs=10); round.(df, digits=3)","category":"page"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"Epoch loss_train loss_test acc_train acc_test\n1 0.418 0.413 0.881 0.884\n2 0.317 0.318 0.909 0.91\n3 0.269 0.274 0.923 0.922\n4 0.232 0.238 0.934 0.932\n5 0.201 0.208 0.943 0.941\n6 0.176 0.187 0.951 0.946\n7 0.155 0.169 0.956 0.951\n8 0.138 0.156 0.96 0.955\n9 0.125 0.145 0.964 0.956\n10 0.113 0.136 0.967 0.959","category":"page"},{"location":"notebooks/FAexample/#References","page":"Example: Feedback Alignment","title":"References","text":"","category":"section"},{"location":"notebooks/FAexample/","page":"Example: Feedback Alignment","title":"Example: Feedback Alignment","text":"Timothy P. Lillicrap et al, 2014, Random feedback weights support learning in deep neural networks, Link","category":"page"},{"location":"#Bender.jl","page":"Home","title":"Bender.jl","text":"","category":"section"},{"location":"#Layers","page":"Home","title":"Layers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GenDense","category":"page"},{"location":"#Bender.GenDense","page":"Home","title":"Bender.GenDense","text":"Generalized version of Flux's Dense layer. The forward keyword allows you to choose the form of the forward mapping.\n\nGenDense(in=>out, σ=identity; \n         init = glorot_uniform, \n         bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)\n\nCan also be initialized with an additional set of trainable weights \n\nGenDense(in=>out, in_asym=>out_asym, σ = identity; \n         init = glorot_uniform, \n         bias=true, α=Flux.Zeros(), β=Flux.Zeros(), forward=linear)\n\nThe layer has additinal keyword arguments α and β, which default to Flux.Zeros. These are useful if you need an extra set of weights for for your forward pass (if you for example wish to anneal an activation function).\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"GenConv","category":"page"},{"location":"#Bender.GenConv","page":"Home","title":"Bender.GenConv","text":"Generalized version of Flux's conv layer.  The forward keyword allows you to choose the form of the forward mapping and defaults to linear. This layer can be initialized with either one or two set of filters  (a second set of filters is useful for feedback alignment experiments).\n\nGenConv((k, k), ch_in=>ch_out, σ=identity; forward=linear)\n\nGenConv((k, k), ch_in=>ch_out_(k_asym, k_asym), ch_in_asym=>ch_out_asym, σ=identity; forward=linear)\n\nThe layer has additinal keyword arguments α and β, which default to Flux.Zeros. These are useful if you need an extra set of weights for for your forward pass (if you for example wish to anneal an activation function).\n\n\n\n\n\n","category":"type"},{"location":"#Forward-mappings","page":"Home","title":"Forward mappings","text":"","category":"section"},{"location":"#Forward-mappings-for-GenDense-layers","page":"Home","title":"Forward mappings for GenDense layers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"linear","category":"page"},{"location":"#Bender.linear","page":"Home","title":"Bender.linear","text":"Matrix multiply layers weight matrix with x and add bias\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"linear_asym_∂x","category":"page"},{"location":"#Bender.linear_asym_∂x","page":"Home","title":"Bender.linear_asym_∂x","text":"behaves identical to linear in the forward pass, but relies on matmulasym∂x,  which causes errors to be backpropagated using a set of auxiliary weights'in the backwards pass. See linear_asym_∂x.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"radial","category":"page"},{"location":"#Bender.radial","page":"Home","title":"Bender.radial","text":"Calls radialSim and computes the negative squared euclidean distance D between the rows ofthe  layers weight matrix and the columns of matrix X. See radialSim.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"radial_asym_∂x","category":"page"},{"location":"#Bender.radial_asym_∂x","page":"Home","title":"Bender.radial_asym_∂x","text":"behaves identical to radial in the forward pass, but relies on radialSimasym∂x,  which causes errors to be backpropagated using a set of auxiliary weights in the backwards pass. See radialSim_asym_∂x.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"linear_binary_weights","category":"page"},{"location":"#Bender.linear_binary_weights","page":"Home","title":"Bender.linear_binary_weights","text":"Regular forward pass (matmul and bias addition) with a binary activation function applied to the weights.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"linear_stoc_binary_weights","category":"page"},{"location":"#Bender.linear_stoc_binary_weights","page":"Home","title":"Bender.linear_stoc_binary_weights","text":"Regular forward pass (matmul and bias addition) with a binary stochastic activation function applied to the weights.\n\n\n\n\n\n","category":"function"},{"location":"#Forward-mappings-for-GenConv-layers","page":"Home","title":"Forward mappings for GenConv layers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"conv_linear","category":"page"},{"location":"#Bender.conv_linear","page":"Home","title":"Bender.conv_linear","text":"Forward mapping for regular convolutional layer\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"conv_linear_asym_∂x","category":"page"},{"location":"#Bender.conv_linear_asym_∂x","page":"Home","title":"Bender.conv_linear_asym_∂x","text":"In the forward pass this behaves identical to conv_linear.  Relies on conv_asym_∂x, which causes errors to be backpropagated  using a set of auxiliary weights in the backwards pass. See conv_asym_∂x.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Similarity/correlation-functions","page":"Home","title":"Similarity/correlation functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"matmul","category":"page"},{"location":"#Bender.matmul","page":"Home","title":"Bender.matmul","text":"Regular matrix multiplication.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"matmul_asym_∂x","category":"page"},{"location":"#Bender.matmul_asym_∂x","page":"Home","title":"Bender.matmul_asym_∂x","text":"Compute matrix multiplication, but takes an additional matrix B as input.  B has same dims as Wᵀ, and is used in the backwards pass.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"matmul_blocked_∂x","category":"page"},{"location":"#Bender.matmul_blocked_∂x","page":"Home","title":"Bender.matmul_blocked_∂x","text":"Matrix multiplication with custom rrule\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"radialSim","category":"page"},{"location":"#Bender.radialSim","page":"Home","title":"Bender.radialSim","text":"Compute negative squared euclidean distance D between the rows of matrix W and the columns of matrix X. Denoting the rows of W by index i and the columns of X by index j the elements of the output matrix is given by: Dᵢⱼ = -||Wᵢ﹕ - X﹕ⱼ||² = 2Wᵢ﹕X﹕,ⱼ - ||Wᵢ﹕||^2 - ||X﹕ⱼ||².\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"radialSim_asym","category":"page"},{"location":"#Bender.radialSim_asym","page":"Home","title":"Bender.radialSim_asym","text":"In the forward pass this function behaves just like radialSim, but in the backwards pass weight symmetry is broken by using matrix B rather than Wᵀ. See docstring for radialSim for more details.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"conv_asym_∂x","category":"page"},{"location":"#Bender.conv_asym_∂x","page":"Home","title":"Bender.conv_asym_∂x","text":"computes the convolution of image x with kernel w when called, but uses a different set of weights w_asym to compute the pullback wrt x. This is typically uses in feedback alignment experiments.\n\n\n\n\n\n","category":"function"},{"location":"#Loss-functions","page":"Home","title":"Loss functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"direct_feedback_loss","category":"page"},{"location":"#Bender.direct_feedback_loss","page":"Home","title":"Bender.direct_feedback_loss","text":"Error function which takes a vector of the hidden and output neurons states as well as a vector of feedback matrices as arguments\n\n\n\n\n\n","category":"function"},{"location":"#Activation-functions","page":"Home","title":"Activation functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"sign_STE","category":"page"},{"location":"#Bender.sign_STE","page":"Home","title":"Bender.sign_STE","text":"Deterministic straight-through estimator for the sign function. Reference: https://arxiv.org/abs/1308.3432, https://arxiv.org/abs/1511.00363\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"stoc_sign_STE","category":"page"},{"location":"#Bender.stoc_sign_STE","page":"Home","title":"Bender.stoc_sign_STE","text":"A stochastic straight-through estimator version of the sign function. References: https://arxiv.org/abs/1308.3432, https://arxiv.org/abs/1511.00363\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"hardσ","category":"page"},{"location":"#Bender.hardσ","page":"Home","title":"Bender.hardσ","text":"A piece-wise linear function. for x<-1 it has value 0. For -1<x<1 it has value x. For x>1 it has value 1. It is defined as:\n\nhardσ(x) = max(0, min(1, (x+1)/2))\n\n\n\n\n\n","category":"function"},{"location":"notebooks/Binaryexample/#Binary-Neural-Network","page":"Example: Binary Neural Network","title":"Binary Neural Network","text":"","category":"section"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"This page contains a short tutorial on how to train a binary neural network. It is loosely adapted from the Flux model zoo's MLP example.","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"We will train neural network with binary {-1,+1} weights and hidden layer activations. The output layer will have real-valued neurons. binary networks require less storage and memory space and when deployed on edge computing devices they can be more computationally effecient than real-valued networks (not so much on traditional CPUs and GPUs which have much better optimized BLAS and CUDA kernels for Float32 operations). ","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"To create a network with binary weights we need to specify a forward mapping which applies a binary activation function to the weights, when we create the layers. We will also specify a binary activation function to ba applied to the layers output. The tricky part about training binary neural networks is that a binary function such as sign(x) has derivative zero everywhere except at the oprigin where it is undefined. A typical workaround is to only use the sign function during the forward pass and replace it with a differentiable surrogate such as tanh or identity during the backwards pass. This hack actually works quite well and a lot of research on binary neural networks explores variations of this idea. We will use the straight-through estimator, which is explained by Courbariaux et al in the BinaryConnect paper. During a forward pass it acts as the sign function, but during the backwards pass (i.e. during backprop) it is replaced by the identity function.","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"In Bender.jl there is a stocastic and a deterministic variant of the STE, stoc_sign_STE  and sign_STE. We will make use of the deterministic version here. In a traditional fully connected layer the output is computed by first computing some preactivation a=Wx+b and then applying an elementwise activation function z=fcirc a. You can specify a forward mapping for the computation of a when initializing a GenDense layer, which is how we can make the layer use the straight through estimator. We will use the following mapping (which is also exported by Bender.jl)","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"function linear_binary_weights(a, x)\n    W, b, = a.weight, a.bias\n    return sign_STE.(W)*x .+ b\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"Using this forward mapping we define a 3 layer fully connected network.","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"function build_model()\n\tm = Chain(  GenDense( \t784=>300, sign_STE, forward=linear_binary_weights),\n                GenDense( \t300=>100, sign_STE; forward=linear_binary_weights), \n                GenDense(  \t100=>10; forward=linear_binary_weights))\n    return m\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"The following sections contain some boilerplate code. You can expand them to show the details.","category":"page"},{"location":"notebooks/Binaryexample/#Handling-dependencies","page":"Example: Binary Neural Network","title":"Handling dependencies","text":"","category":"section"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"<details><summary>show details</summary>","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"using Pkg; Pkg.add(url=\"https://github.com/Rasmuskh/Bender.jl.git\")","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"begin\n\tusing Bender, Flux, MLDatasets\n\tusing Flux: onehotbatch, onecold, logitcrossentropy, throttle\n\tusing Flux.Data: DataLoader\n\tusing Parameters: @with_kw\n\tusing DataFrames\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"</details>","category":"page"},{"location":"notebooks/Binaryexample/#Utilities","page":"Example: Binary Neural Network","title":"Utilities","text":"","category":"section"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"<details><summary>show details</summary>","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"@with_kw mutable struct Args\n    η::Float64 = 0.0003     # learning rate\n    batchsize::Int = 64     # batch size\n    epochs::Int = 10        # number of epochs\n    device::Function = gpu  # set as gpu, if gpu available\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"function getdata(args)\n    ENV[\"DATADEPS_ALWAYS_ACCEPT\"] = \"true\"\n\n    # Loading Dataset\t\n    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)\n    xtest, ytest = MLDatasets.MNIST.testdata(Float32)\n\n    # Reshape Data in order to flatten each image into a vector\n    xtrain = Flux.flatten(xtrain)\n    xtest = Flux.flatten(xtest)\n\n    # One-hot-encode the labels\n    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)\n\n    # Batching\n    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true, partial=false)\n    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)\n\n    return train_data, test_data\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"function evaluate(data_loader, model)\n    acc = 0\n\tl = 0\n\tnumbatches = length(data_loader)\n    for (x,y) in data_loader\n        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)\n\t\tl += logitcrossentropy(model(x), y)\n    end\n    return acc/numbatches, l/numbatches\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"</details>","category":"page"},{"location":"notebooks/Binaryexample/#Defining-the-training-loop","page":"Example: Binary Neural Network","title":"Defining the training loop","text":"","category":"section"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"<details><summary>show details</summary>","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"function train(; kws...)\n    # Initializing Model parameters \n    args = Args(; kws...)\n\n\t# Create arrays for recording training metrics\n\tacc_train = zeros(Float32, args.epochs)\n\tacc_test = zeros(Float32, args.epochs)\n\tloss_train = zeros(Float32, args.epochs)\n\tloss_test = zeros(Float32, args.epochs)\n\t\n    # Load Data\n    train_data,test_data = getdata(args)\n\n    # Construct model\n    m = build_model()\n\t\n    train_data = args.device.(train_data)\n    test_data = args.device.(test_data)\n    m = args.device(m)\n    loss(x,y) = logitcrossentropy(m(x), y)\n    \n    # Training\n    opt = ADAM(args.η)\n\tfor epoch=1:args.epochs\t\n        Flux.train!(loss, params(m), train_data, opt)\n\t\tacc_train[epoch], loss_train[epoch] = evaluate(train_data, m)\n\t\tacc_test[epoch], loss_test[epoch] = evaluate(test_data, m)\n\t\t# clamp weights to lie in the range [-1,+1]\n\t\tfor layer in m\n\t\t\tlayer.weight .= clamp.(layer.weight, -1, 1)\n\t\tend\n    end\n\n\t# Return trianing metrics as a DataFrame\n\tdf = DataFrame([loss_train, loss_test, acc_train, acc_test], \n\t\t\t\t   [:loss_train, :loss_test, :acc_train, :acc_test])\n\treturn df, cpu(m)\nend","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"</details>","category":"page"},{"location":"notebooks/Binaryexample/#Training-the-model","page":"Example: Binary Neural Network","title":"Training the model","text":"","category":"section"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"Although we constrained the model to use {-1,+1} weights during inference it still manages to solve the problem fairly well. The train function stores the model's loss and accuracy on the train and test set in a DataFrame.","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":"df, m = train(epochs=10); round.(df, digits=3)","category":"page"},{"location":"notebooks/Binaryexample/","page":"Example: Binary Neural Network","title":"Example: Binary Neural Network","text":" loss_train loss_test acc_train acc_test\n Float32 Float32 Float32 Float32\n1 1.9 1.862 0.789 0.792\n2 1.56 1.542 0.824 0.83\n3 1.365 1.361 0.85 0.849\n4 1.362 1.382 0.846 0.852\n5 1.294 1.34 0.859 0.859\n6 1.234 1.224 0.866 0.87\n7 1.268 1.269 0.865 0.869\n8 1.306 1.395 0.857 0.855\n9 1.222 1.26 0.873 0.873\n10 1.22 1.263 0.867 0.867","category":"page"}]
}
