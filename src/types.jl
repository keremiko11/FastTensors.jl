"""
    Tensor{T, N}

A dense N-dimensional tensor wrapping a contiguous array with specialized strides.
"""
struct Tensor{T, N}
    data::Array{T, N}
    dims::NTuple{N, Int}
    strides::NTuple{N, Int}

    function Tensor{T, N}(data::Array{T, N}) where {T, N}
        dims = size(data)
        strds = Base.strides(data)
        new{T, N}(data, dims, strds)
    end
end

Tensor(data::Array{T, N}) where {T, N} = Tensor{T, N}(data)

Base.size(t::Tensor) = t.dims
Base.size(t::Tensor, dim::Int) = t.dims[dim]
Base.ndims(::Tensor{T, N}) where {T, N} = N
Base.eltype(::Tensor{T}) where {T} = T
Base.getindex(t::Tensor, idxs...) = t.data[idxs...]
Base.setindex!(t::Tensor, val, idxs...) = (t.data[idxs...] = val)

function rand_tensor(::Type{T}, dims::NTuple{N, Int}) where {T, N}
    Tensor(randn(T, dims))
end

function zeros_tensor(::Type{T}, dims::NTuple{N, Int}) where {T, N}
    Tensor(zeros(T, dims))
end

function tensor_norm(t::Tensor{T}) where {T}
    sqrt(sum(abs2, t.data))
end
