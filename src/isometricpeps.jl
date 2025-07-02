
struct IsometricPEPS{T} <: PEPS{T,2,Int}

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

function statetensor(peps::IsometricPEPS)
    return reduce(vcat, peps.vertex_tensors[1])
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

Base.copy(peps::PEPS) = PEPS(copy(peps.vertex_tensors), peps.col, peps.row, peps.D)

   


function isometric_peps(::Type{T}, g, D::Int, nflavor::Int, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where T
    peps = rand_peps(T, g, D, nflavor, optimizer, simplifier)
    matrix_dims = []
  
    for i in 1:nv(g)
        tensor = peps.vertex_tensors[i]
        phys_dim = size(tensor, 1)  # Physical dimension
        in_dim = prod([D for _ in inneighbors(g, i)])  # Product of incoming bond dimensions
        out_dim = prod([D for _ in outneighbors(g, i)])  # Product of outgoing bond dimensions
        
        n_A = max(in_dim, phys_dim * out_dim)
      
        original_size = size(tensor)
    
        A = randn(T, n_A, n_A)
        Q, R = qr(A)
        Q=collect(Q)
  
        dims = [2 for _ in 1:Int(log2(phys_dim*out_dim*in_dim))]
        all_neighbors = union(inneighbors(g, i), outneighbors(g, i))
        out_index = findall(x -> x in outneighbors(g, i), all_neighbors) .+ 1   # TODO: not right if the shape is 2*2 in torus
        in_index = findall(x -> x in inneighbors(g, i), all_neighbors) .+ 1
        if phys_dim * out_dim >= in_dim
            Q = Q[:, 1:in_dim] 
            push!(matrix_dims, size(Q))
           
            
        else
            Q = Q[:, 1:phys_dim*out_dim]
            push!(matrix_dims, size(Q))
            Q = reshape(Q, dims...)
            Q = permutedims(Q, (3,1,2))
        end
        
        peps.vertex_tensors[i] = reshape(Q, original_size)
    end
    
    return peps, matrix_dims
end

function specific_peps(peps::PEPS, theta, fai)   # just for isometric peps with 5 indices
    scatter = [cos(fai) 0 0 -sin(fai); 0 cos(theta)  -sin(theta) 0; 0 sin(theta) cos(theta) 0; sin(fai) 0 0 cos(fai)]
    U12 = kron(scatter, I(2))
    U23 = kron(I(2), scatter)
    U_total = U12 * U23 * U12
    @assert U_total'*U_total ≈ Matrix(I, 8, 8)
    @assert U_total*U_total' ≈ Matrix(I, 8, 8)
    for i in peps.physical_labels
        peps.vertex_tensors[i] = reshape(U_total[:,1:4], size(peps.vertex_tensors[i]))
    end
    return peps
end


function isometric_peps_to_unitary(peps::PEPS, g)
    ugates = deepcopy(peps)
    
    for i in 1:nv(g)
        tensor = peps.vertex_tensors[i]

        phys_dim = size(tensor, 1)  # Physical dimension
        in_dim = prod([peps.D for _ in inneighbors(g, i)])  # Product of incoming bond dimensions
        out_dim = prod([peps.D for _ in outneighbors(g, i)])

        all_neighbors = union(inneighbors(g, i), outneighbors(g, i))
        out_index = findall(x -> x in outneighbors(g, i), all_neighbors) .+ 1
        in_index = findall(x -> x in inneighbors(g, i), all_neighbors) .+ 1

        ancil = 1 + length(outneighbors(g, i)) - length(inneighbors(g, i))
        ortho_dim = max(phys_dim * out_dim, in_dim)
    
        #Q = reshape(tensor, ortho_dim, :)
        Q = reshape(tensor, out_dim*phys_dim, :)
        if ancil < 0
            Q = transpose(Q)
        end

        remaining = nullspace(Q') 
        Q = hcat(Q, remaining)
    
        #@assert Q'*Q ≈ Matrix(I, size(Q, 1), size(Q, 2))
        target_size = size(Q)
        
        dims = [2 for _ in 1:2*Int(log2(ortho_dim))]
        Q = reshape(Q, dims...)

       
        #ancil = phys_dim + out_dim - in_dim
    
        if ancil < 0
            in_ancil = nothing
            out_ancil = -ancil
            out_ancil_index = collect(length(all_neighbors)+2:length(all_neighbors)+2+out_ancil-1)
           
            Q = permutedims(Q, (3,4,1,2))   # TODO: use out_ancil_index, in_index...
       
        end
  
        ugates.vertex_tensors[i] = reshape(Q, target_size...)
    end
 
    return ugates
end


function vector2point(v::AbstractVector, matrix_dims::Vector)
    point = []
    offset = 0
    for (n, p) in matrix_dims
        push!(point, reshape(v[offset+1:offset+n*p], n, p))
        offset += n*p
    end
    return point
end

function vector2point(v::Tuple, matrix_dims::Vector)
    point = []
    for (i, (n, p)) in enumerate(matrix_dims)
        push!(point, reshape(v[(i-1)*n*p+1:i*n*p], n, p))
    end
    return point
end


function point2vector(point, matrix_dims)
    return vcat([vec(p) for p in point]...) 
end




function isopeps_optimize_ising(peps::PEPS, M::ProductManifold, matrix_dims::Vector, g, J::Float64, h::Float64, optimizer::CodeOptimizer, simplifier::CodeSimplifier, step::Int)
    params = variables(peps)
    p0 = RecursiveArrayTools.ArrayPartition(Tuple(vector2point(params, matrix_dims))...)

    energy = 0.0
    @assert is_point(M, p0)
    
    function f_closure_ising(M,p0) 
        point_tuple = Tuple(p0.x)
        x = point2vector(point_tuple, matrix_dims)
        
        energy = f_ising(peps, x, g, J, h, optimizer, simplifier)
        return energy
    end


    function g_closure_ising(M,p0)
        point_tuple = Tuple(p0.x)    
        x = point2vector(point_tuple, matrix_dims)
        G = similar(x)
        fill!(G, 0)
    
        G .= g_ising!(G, peps, x, g, J, h, optimizer, simplifier)
        grad = RecursiveArrayTools.ArrayPartition(Tuple(vector2point(G, matrix_dims))...)
  
        return grad
       
    end
   
    result = gradient_descent(
        M, 
        f_closure_ising, 
        g_closure_ising,
        p0;
        evaluation=Manopt.AllocatingEvaluation(),
        #retraction_method = QRRetraction(),
        stopping_criterion = #=StopWhenGradientNormLess(1e-2) |=# StopAfterIteration(step),
        record = [:Iteration, :Cost, :GradientNorm],
        return_state = true
    )
    

    record = get_record(result)
    final_p = get_solver_result(result)
    final_energy = f_closure_ising(M, final_p)
    
    optimized_vector = point2vector(Tuple(final_p.x), matrix_dims)
    optimized_peps = peps
    load_variables!(optimized_peps, optimized_vector)
  
    return result, final_energy, optimized_peps, record
end

