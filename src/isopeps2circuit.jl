#==
mutable struct IsoPEPSCircuit
    peps::GeneralPEPS
    target_qubits::Vector{Vector{Int}}

    function IsoPEPSCircuit(peps::GeneralPEPS, target_qubits::Vector{Vector{Int}})
        new(peps, target_qubits)
    end
end
==#
  


function peps2ugate(peps::GeneralPEPS, g)
    ugates = deepcopy(peps)
    T = eltype(peps.vertex_tensors[1])
    for i in peps.physical_labels
        out = max(length(outneighbors(g, i))+1,length(inneighbors(g, i)))
      
        original_size = size(peps.vertex_tensors[i])
   
        A= reshape(peps.vertex_tensors[i], 2^out, :)
        # Seed the random number generator for reproducibility
        Random.seed!(42)
        reshaped = randn(T, 2^out, 2^out)
        @assert size(reshaped, 2) <= size(reshaped, 1)
        Q, R = qr(reshaped)
        @assert Q[:,1:size(reshaped,2)] * R ≈ reshaped
        ugates.vertex_tensors[i] = collect(Q)
        
        # Take only the first columns of Q to match original size
        n_cols = size(A,2)
        #Q = Q[:, 1:n_cols]
        
        add_dims = length(outneighbors(g,i))+1-length(inneighbors(g,i))
        
            # Reshape Q into a multidimensional array with dimensions of size 2
            # This creates a tensor with the right number of indices for the quantum circuit
        dims = [2 for _ in 1:out*2]
        Q = reshape(Matrix(Q), dims...)
            #Q = reshape(Matrix(Q),:,2^abs(length(outneighbors(g,i))+1-length(inneighbors(g,i))))
            #Q = Q[:,1]
            # Select Q with the right number of indices
        if add_dims > 0
            if add_dims == 1
                Q = Q[:,:,:,1]
                Q = permutedims(Q, (1, 3, 2))
            elseif add_dims == 2
                Q = Q[:,:,1,1]
            elseif add_dims == 3
                Q = Q[:,:,:,1,1,1]
            end
            #indices = Tuple(vcat([Colon() for _ in 1:(out*2-add_dims)], [1 for _ in 1:add_dims]))
            #Q = Q[indices...]
            @show size(Q)
        elseif add_dims < 0
            Q = Q[:,1,:,:] 
            @show size(Q)
            #Q = reshape(Matrix(Q),2^abs(length(outneighbors(g,i))+1-length(inneighbors(g,i))),:)
            #Q = Q[1,:]
        else
            Q = Q
        end
        @show size(Q)
        
        peps.vertex_tensors[i] = Q
    
    end
    return peps, ugates
end


function get_reuse_circuit(nbit::Int, pepsu::GeneralPEPS, g)
   circ = chain(nbit)
   target_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   free_qubits = collect(1:nbit)
   remain_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   add!(block) = push!(circ, block)
   
   for i in 1:length(pepsu.physical_labels) 
        inneighbors = Graphs.inneighbors(g, i)
        n = Int(log2(size(pepsu.vertex_tensors[i])[1]))-length(inneighbors)
    
       
        if isempty(inneighbors)
            target_qubits[i] = free_qubits[1:n]
        else
            
            
            last_elements = Int[remain_qubits[j][1] for j in inneighbors 
            if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])]
            target_qubits[i] = vcat(last_elements, free_qubits[1:n]) 
            
            
            for j in inneighbors
                if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])
                    remain_qubits[j] = remain_qubits[j][2:end] 
                end
            end
        end
        
        if !isempty(target_qubits[i])
            chain(nbit, put(nbit, Tuple(target_qubits[i])=>matblock(pepsu.vertex_tensors[i]))) |> add!
            Bag(basis_rotor(Z, nbit, target_qubits[i])) |> add!
            Measure(nbit; locs=target_qubits[i][1], resetto=0) |> add!

            if i==length(pepsu.physical_labels)
                Measure(nbit; locs=target_qubits[i][2], resetto=0) |> add!
            end
            
            free_qubits = vcat(target_qubits[i][1], free_qubits[n+1:end]) |> sort
            remain_qubits[i] = target_qubits[i][2:end]
        else
           
            @warn "Empty target_qubits for node $i"
            remain_qubits[i] = Int[]
        end
   end
   return circ
end

function get_circuit(nbit::Int, pepsu::GeneralPEPS, g)
    circ = chain(nbit)
    target_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
    remain_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
    free_qubits = collect(1:nbit)
    add!(block) = push!(circ, block)
    
    for i in 1:length(pepsu.physical_labels) 
         inneighbors = Graphs.inneighbors(g, i)
         n = Int(log2(size(pepsu.vertex_tensors[i])[1]))-length(inneighbors)    
     
         # Initialize target_qubits[i] instead of target_qubits[1]
        if isempty(inneighbors)
            target_qubits[i] = free_qubits[1:n]
        else
             # Check if remain_qubits[j] is initialized for all inneighbors
            last_elements = Int[remain_qubits[j][1] for j in inneighbors 
            if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])]
            
             
            target_qubits[i] = vcat(last_elements, free_qubits[1:n]) |> sort
            #target_qubits[i] = sort(target_qubits[i])
        
             # Update remain_qubits for inneighbors
             for j in inneighbors
                 if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])
                     remain_qubits[j] = remain_qubits[j][2:end] 
                 end
             end
         end
         
         # Ensure target_qubits[i] is not empty before proceeding
         if !isempty(target_qubits[i])
             chain(nbit, put(nbit, Tuple(target_qubits[i])=>matblock(pepsu.vertex_tensors[i]))) |> add!
             #Bag(basis_rotor(Z, nbit, target_qubits[i])) |> add!
             Measure(nbit; locs=target_qubits[i][1], resetto=0) |> add!
             if i==length(pepsu.physical_labels)
                Measure(nbit; locs=target_qubits[i][2], resetto=0) |> add!
            end
            
             free_qubits = free_qubits[n+1:end]
             remain_qubits[i] = target_qubits[i][2:end]
         else
             # Handle the case where target_qubits[i] is empty
             @warn "Empty target_qubits for node $i"
             remain_qubits[i] = Int[]
         end
    end
    return circ
end

function gensample(circ, reg::AbstractRegister, pepsu::GeneralPEPS, basis)  # TODO: add more basis: X, Y
    nbit = nqubits(reg)
    res = zeros(Int, nbatch(reg), length(pepsu.physical_labels)+1)
    
    bags = collect_blocks(Bag, circ)
   
    for bag in bags
        setcontent!(bag, basis_rotor(basis, nbit, occupied_locs(bag.content)))
    end
    
    copy(reg) |> circ
    result = collect_blocks(Measure, circ)
    
    for j in 1:length(result)
        res[:,j] = result[j].results # Access first element of result array
    end    
 
    return res
end


 
function long_range_coherence(circ, reg::AbstractRegister, pepsu::GeneralPEPS, i::Int, j::Int)
    corr = 0.0
    for basis in [Z]
        samples = gensample(circ, reg, pepsu, basis)   
         # Calculate single-site expectations
       
         
         # Calculate two-site correlation
        
         valid_shots = [shot for shot in 1:nbatch(reg) if samples[shot, end] == 0]
         
         
         # Recalculate expectations using only valid shots
         Si = mean((-1)^samples[shot,i] for shot in valid_shots)
         Sj = mean((-1)^samples[shot,j] for shot in valid_shots)
         SiSj = mean((-1)^(samples[shot,i] + samples[shot,j]) for shot in valid_shots)
         
         println("Circuit Si: ", Si)
         println("Circuit Sj: ", Sj)
         println("Circuit SiSj: ", SiSj)
         
         corr = SiSj
    end
    return abs(corr)
end





mutable struct Bag{D}<:TagBlock{AbstractBlock, D}
    content::AbstractBlock{D}
end

Yao.content(bag) = bag.content    # directy get the content of the bag
Yao.chcontent(bag::Bag, content) = Bag(content)  # change the content of the bag
Yao.mat(::Type{T}, bag::Bag) where T = mat(T, bag.content)  # change content to matrix
YaoBlocks.unsafe_apply!(reg::AbstractRegister, bag::Bag) = YaoBlocks.unsafe_apply!(reg, bag.content)  
YaoBlocks.PreserveStyle(::Bag) = YaoBlocks.PreserveAll()  
setcontent!(bag::Bag, content) = (bag.content = content; bag)  

basis_rotor(::ZGate) = I2Gate()
basis_rotor(::XGate) = Ry(-0.5π)  
basis_rotor(::YGate) = Rx(0.5π)   

basis_rotor(basis, nbit, locs) = repeat(nbit, basis_rotor(basis), locs)


"""circuit for torus"""
function iter_sz_convergence(pepsu::GeneralPEPS, g; n_physical_qubits::Int=1, n_virtual_qubits::Int=6, max_iterations::Int=100, sz_tol::Float64=0.01, convergence_window::Int=10)
    nbit = n_physical_qubits + n_virtual_qubits
    physical_qubits = collect(1:n_physical_qubits)
    virtual_qubits = collect(n_physical_qubits+1:nbit)
    
    all_measurements = Vector{Int}[]
    converged = false
    converged_iter = -1
    
    # TODO: initialize virtual qubits with random state
    
    circ = Yao.chain(nbit)
    

    for iter in 1:max_iterations
        circ = get_iter_circuit(circ, pepsu, n_physical_qubits, n_virtual_qubits, physical_qubits, virtual_qubits)   # add series of blocks to circ
        circ_reg = copy(circ)

        reg = Yao.zero_state(nbit)
        reg |> circ_reg
        iter_res = extract_sz_measurements(circ_reg, nbit, g)  # single iteration result
        push!(all_measurements, iter_res)   # Store measurements
        
        # Check convergence
        if iter >= convergence_window
            if fidelity_convergence(all_measurements; tol=sz_tol, window=convergence_window)
                converged = true
                converged_iter = iter
                println("Converged at iteration $iter")
                break
            end
        end
    end
    
    if !converged
        println("Did not converge within $max_iterations iterations")
    end
    
    return circ, converged, converged_iter
end

function get_iter_circuit(circ, pepsu::GeneralPEPS, n_physical_qubits::Int, n_virtual_qubits::Int, physical_qubits::Vector{Int}, virtual_qubits::Vector{Int})   
    n_sites = length(pepsu.physical_labels)
    nbit = n_physical_qubits + n_virtual_qubits
    Lx = Ly = Int(sqrt(n_sites))  # Assuming square lattice TODO: make it better
    
    for site in 1:n_sites
        col, row = CartesianIndices((Lx, Ly))[site][1],CartesianIndices((Lx, Ly))[site][2]
            
        pq = physical_qubits[((site-1) % n_physical_qubits) + 1]
            
        vq_v = virtual_qubits[row]  
        vq_h = virtual_qubits[col + Ly]   
            
        target_qubits = [pq, vq_v, vq_h]
          
        gate = pepsu.vertex_tensors[site]
            
        push!(circ, put(nbit, Tuple(target_qubits)=>matblock(gate)))    
        push!(circ, Measure(nbit; locs=pq, resetto=0))
            
    end
   
    return circ
end


function fidelity_convergence(all_measurements::Vector{Vector{Int}}; tol::Float64=0.01, window::Int=10)
    if length(all_measurements) < window
        return false
    end
    prev_state = all_measurements[end-1]
    curr_state = all_measurements[end]

    fidelity = sum(prev_state .== curr_state) / length(prev_state)

    if fidelity < 1 - tol
        return false
    end

    return true
end

function extract_sz_measurements(circ, nbit::Int, g)
    iter_res = zeros(Int, nv(g))
    result = collect_blocks(Measure, circ)[end-nv(g)+1:end]

    for j in 1:length(result)
        iter_res[j] = result[j].results
    end    
    return iter_res
end

function torus_gensample(circ, reg::AbstractRegister, pepsu::GeneralPEPS, basis)
    nbit = nqubits(reg)
    res = zeros(Int, nbatch(reg), length(pepsu.physical_labels)*converged_iter)
    
    bags = collect_blocks(Bag, circ)
    
    for bag in bags
        setcontent!(bag, basis_rotor(basis, nbit, occupied_locs(bag.content)))
    end
    
    copy(reg) |> circ
    result = collect_blocks(Measure, circ)#[end-length(pepsu.physical_labels)+1:end]
    
    for j in 1:length(result)
        res[:,j] = result[j].results # Access first element of result array
    end    
    @show length(result)
    return res
end


function torus_long_range_coherence(circ, reg::AbstractRegister, pepsu::GeneralPEPS, i::Int, j::Int)
    corr = 0.0
    for basis in [Z]
        samples = torus_gensample(circ, reg, pepsu, basis)   
        SiSj = mean((-1)^(samples[shot,i] + samples[shot,j]) for shot in 1:nbatch(reg))
         
        println("Circuit SiSj: ", SiSj)
         
        corr = SiSj
    end
    return abs(corr)
end


"""Initialize virtual qubits to random state (only once at the beginning!)"""
function init_random_vq(circ, nbit, virtual_qubits)
    for vq in virtual_qubits
        push!(circ, put(nbit, vq=>Ry(rand()*2π)))
        push!(circ, put(nbit, vq=>Rz(rand()*2π)))
    end
end