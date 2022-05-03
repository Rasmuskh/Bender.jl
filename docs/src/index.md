# Bender.jl

## Layers
```@docs
GenDense
```

```@docs
GenConv
```

## Forward mappings
### Forward mappings for GenDense layers
```@docs
linear
```

```@docs
linear_asym_∂x
```

```@docs
radial
```

```@docs
radial_asym_∂x
```

```@docs
linear_binary_weights
```

```@docs
linear_stoc_binary_weights
```
### Forward mappings for GenConv layers
```@docs
conv_linear
```

```@docs
conv_linear_asym_∂x
```

```@docs

```

## Similarity/correlation functions
```@docs
matmul
```

```@docs
matmul_asym_∂x
```

```@docs
matmul_blocked_∂x
```

```@docs
radialSim
```

```@docs
radialSim_asym
```

```@docs
conv_asym_∂x
```

## Loss functions
```@docs
direct_feedback_loss
```

## Activation functions
```@docs
sign_STE
```

```@docs
stoc_sign_STE
```

```@docs
hardσ
```