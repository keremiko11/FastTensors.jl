"""
    fast_fma!(dest, A, B, alpha)

SIMD-vectorized in-place fused multiply-add operation: dest .= dest .+ alpha .* (A .* B).
"""
function fast_fma!(dest::Array{T, N}, A::Array{T, N}, B::Array{T, N}, alpha::T) where {T<:AbstractFloat, N}
    @assert size(dest) == size(A) == size(B) "Array dimension mismatch"
    @inbounds @simd for i in eachindex(dest)
        dest[i] = muladd(alpha * A[i], B[i], dest[i])
    end
    return dest
end

function fast_fma!(dest::Tensor{T, N}, A::Tensor{T, N}, B::Tensor{T, N}, alpha::T) where {T<:AbstractFloat, N}
    fast_fma!(dest.data, A.data, B.data, alpha)
    return dest
end
