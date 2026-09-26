"""
    unfold(T, mode)

Mode-n matricization (unfolding) of tensor T.
"""
function unfold(t::Tensor{T, N}, mode::Int) where {T, N}
    @assert 1 <= mode <= N "Mode must be between 1 and ndims"
    order = (mode, filter(i -> i != mode, 1:N)...)
    permuted = permutedims(t.data, order)
    reshape(permuted, size(t, mode), :)
end

"""
    hosvd(T; ranks)

Computes the Higher-Order Singular Value Decomposition (Tucker Decomposition)
of tensor T with target ranks for each mode.
"""
function hosvd(t::Tensor{T, N}; ranks::Union{Nothing, NTuple{N, Int}}=nothing) where {T, N}
    factors = Vector{Matrix{T}}(undef, N)
    actual_ranks = isnothing(ranks) ? size(t) : ranks

    for n in 1:N
        Un = unfold(t, n)
        F = svd(Un)
        r = min(actual_ranks[n], size(F.U, 2))
        factors[n] = F.U[:, 1:r]
    end

    # Compute core tensor: t contracted with each factor transpose
    core_data = copy(t.data)
    for n in 1:N
        order = (n, filter(i -> i != n, 1:N)...)
        unfolded = reshape(permutedims(core_data, order), size(core_data, n), :)
        compressed = factors[n]' * unfolded
        new_dims = (size(factors[n], 2), (size(core_data, i) for i in order[2:end])...)
        core_data = permutedims(reshape(compressed, new_dims), invperm(order))
    end

    return Tensor(core_data), factors
end
