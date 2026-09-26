"""
    contract(A, B, idxsA, idxsB)

Contract tensor A and tensor B along specified index dimensions.
Utilizes cache-blocked multiplication for maximum locality.
"""
function contract(A::Tensor{T, NA}, B::Tensor{T, NB},
                  idxsA::NTuple{K, Int}, idxsB::NTuple{K, Int}) where {T, NA, NB, K}
    # Validate contraction indices match in size
    for k in 1:K
        @assert size(A, idxsA[k]) == size(B, idxsB[k]) "Dimension mismatch along contracted axes"
    end

    freeA = filter(i -> !(i in idxsA), 1:NA)
    freeB = filter(j -> !(j in idxsB), 1:NB)

    dimsA_free = tuple((size(A, i) for i in freeA)...)
    dimsB_free = tuple((size(B, j) for j in freeB)...)
    out_dims = (dimsA_free..., dimsB_free...)

    # Reshape tensors to 2D matrices for optimized BLAS/GEMM call
    dim_contract = prod(size(A, i) for i in idxsA)
    dim_freeA = prod(dimsA_free)
    dim_freeB = prod(dimsB_free)

    permA = (freeA..., idxsA...)
    permB = (idxsB..., freeB...)

    matA = reshape(permutedims(A.data, permA), dim_freeA, dim_contract)
    matB = reshape(permutedims(B.data, permB), dim_contract, dim_freeB)

    out_data = matA * matB
    Tensor(reshape(out_data, out_dims))
end

function contract(A::Tensor{T, NA}, B::Tensor{T, NB},
                  idxA::Int, idxB::Int) where {T, NA, NB}
    contract(A, B, (idxA,), (idxB,))
end
