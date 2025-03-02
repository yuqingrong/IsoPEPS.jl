
struct IsometricPEPS{T} 

    vertex_tensors::Vector{Vector{AbstractArray{T}}}
    col::Int
    row::Int
 

    D::Int
    #orthogonal_hypersurface::Vector{LT}
    #orthogonal_center::Int

    function IsometricPEPS(vertex_tensors::Vector{Vector{AbstractArray{T}}}, col::Int, row::Int, D::Int) where {T}
        new{T}(vertex_tensors, col, row, D)
    end
end



function rand_isometricpeps(::Type{T}, bond_dim::Int, Ly::Int, Lx::Int; d::Int=2) where {T}
    tensors = Vector{Vector{AbstractArray{T}}}(undef,Ly)
    col = Ly
    row = Lx
    D = bond_dim
    for y in 1: Ly
        row_tensors = Vector{AbstractArray{T}}(undef, Lx)
        for x in 1: Lx
            left_dim = y == 1 ? 1 : D
            right_dim = y == Ly ? 1 : D
            up_dim = x == Lx ? 1 : D
            down_dim = x == 1 ? 1 : D
            row_tensors[x] = randn(T, d, left_dim, right_dim, up_dim, down_dim)
        end
        tensors[y]=row_tensors
    end
    return IsometricPEPS(tensors, col, row, D)
end


function mose_move_right!(peps::IsometricPEPS)
    for col in 1:peps.col-1 
        mose_move_right_step!(peps, col)
    end
    return peps
end

function mose_move_right_step!(peps::IsometricPEPS, col::Int) 
    @assert col < peps.col
    new_col = _unzip_right(peps.vertex_tensors[col], peps.D)
    peps.vertex_tensors[col+1] = _zip_right(new_col, peps.vertex_tensors[col+1])
 
    return peps
end


# split the col-th column of peps into two columns, only the left ones have physical indices.
function _unzip_right(target_col1::Vector{AbstractArray{T}}, D::Int) where {T}
    new_col = []
    
    for i in 1:length(target_col1)
        psi, Q = _unzip_right_step(target_col1,i,D,1e-10)
        if i == length(target_col1)
            new_shape = [size(psi, 2)*size(Q, 1), size(Q, 2), size(psi, 1), size(Q, 4)]
            Q = reshape(reshape(psi, :, size(psi, 3))*reshape(Q, size(Q, 3), :), new_shape...)
            push!(new_col, Q)
        else
            push!(new_col, Q)
            new_shape = [size(target_col1[i+1],1),size(target_col1[i+1],2),size(target_col1[i+1],3),size(target_col1[i+1],4),size(psi,2),size(psi,3)]
            
            target_col1[i+1] = reshape(reshape(target_col1[i+1],:,size(target_col1[i+1], 5))*reshape(psi,size(psi,1),:),new_shape...)
        end
       
    end
    return new_col
end


# Figure 2 in (Michael 2019)
function _unzip_right_step(target_col1::Vector{AT}, i::Int, D::Int, atol::Float64) where {AT<:AbstractArray} 
    shape_target_col = size(target_col1[i])
    U, Λ1, V1 = truncated_svd(reshape(target_col1[i], size(target_col1[i],1)*size(target_col1[i],2)*size(target_col1[i],5), :), D, atol)
   
    s1 = floor(Int, sqrt(length(Λ1)))
    s2 = ceil(Int, sqrt(length(Λ1)))   # TODO: not as expected 
    target_col1[i] = reshape(U, shape_target_col[1],shape_target_col[2], s2, s1, shape_target_col[5])
 
    θ = reshape(Diagonal(Λ1)*V1, s1*shape_target_col[4], :)
    psi, Λ2, Q = truncated_svd(θ, D, atol)
    s3 = length(Λ2)
    psi = reshape(psi*Diagonal(Λ2), :,s1, s3)
    Q = reshape(Q, s2, shape_target_col[3], s3, :)
    return psi, Q
end

# contract a column of tensors with the col-th column in PEPS.
function _zip_right(new_col::Vector, target_col2::Vector)
    for i in 1:length(new_col)
    
        new_shape = [size(target_col2[i],1),size(new_col[i],1),size(target_col2[i],3),size(new_col[i],3)*size(target_col2[i],4),size(new_col[i],4)*size(target_col2[i],5)]
        target_col2[i] = reshape(reshape(new_col[i],:,size(new_col[i], 2))*reshape(target_col2[i],size(target_col2[i],2),:),new_shape...)
    end
    return target_col2   
end
    

function peps_fidelity(p1::IsometricPEPS, p2::IsometricPEPS) 
    inner_prod = sum(dot(vec(a), vec(b)) for (row1, row2) in zip(p1.vertex_tensors, p2.vertex_tensors) for (a, b) in zip(row1, row2))
    norm_psi = sum(dot(vec(a), vec(a)) for row in p2.vertex_tensors for a in row)
    return abs(inner_prod)^2/(norm_psi)^2
end

Base.copy(peps::IsometricPEPS) = IsometricPEPS(copy(peps.vertex_tensors), peps.col, peps.row, peps.D)

   