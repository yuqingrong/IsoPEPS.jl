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
   
    for i in peps.physical_labels
        out = max(length(outneighbors(g, i))+1,length(inneighbors(g, i)))
      
        original_size = size(peps.vertex_tensors[i])
   
        A= reshape(peps.vertex_tensors[i], 2^out, :)
        # Seed the random number generator for reproducibility
        Random.seed!(42)
        reshaped = randn(ComplexF64, 2^out, 2^out)
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


function get_circuit(pepsu::GeneralPEPS, g)
   nbit = 5  # TODO: make this dynamic
   circ = chain(nbit)
   target_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   free_qubits = collect(1:nbit)
   remain_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   add!(block) = push!(circ, block)
   
   for i in 1:length(pepsu.physical_labels) 
        inneighbors = Graphs.inneighbors(g, i)
        n = Int(log2(size(pepsu.vertex_tensors[i])[1]))-length(inneighbors)
    
        # Initialize target_qubits[i] instead of target_qubits[1]
        if isempty(inneighbors)
            target_qubits[i] = free_qubits[1:n]
        else
            # Check if remain_qubits[j] is initialized for all inneighbors
            last_elements = Int[]
            for j in inneighbors
                if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])
                    push!(last_elements, remain_qubits[j][end])
                end
            end
            
            target_qubits[i] = vcat(last_elements, free_qubits[1:n]) 
            target_qubits[i] = sort(target_qubits[i])
            @show target_qubits[1]
            # Update remain_qubits for inneighbors
            for j in inneighbors
                if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])
                    remain_qubits[j] = remain_qubits[j][1:end-1] 
                end
            end
        end
        
        # Ensure target_qubits[i] is not empty before proceeding
        if !isempty(target_qubits[i])
            chain(nbit, put(nbit, Tuple(target_qubits[i])=>matblock(pepsu.vertex_tensors[i]))) |> add!
            Bag(basis_rotor(Z, nbit, target_qubits[i])) |> add!
            Measure(nbit; locs=target_qubits[i][1], resetto=0) |> add!

            if i==length(pepsu.physical_labels)
                Measure(nbit; locs=target_qubits[i][2], resetto=0) |> add!
            end
            
            free_qubits = vcat(target_qubits[i][1], free_qubits[n+1:end])
            remain_qubits[i] = target_qubits[i][2:end]
        else
            # Handle the case where target_qubits[i] is empty
            @warn "Empty target_qubits for node $i"
            remain_qubits[i] = Int[]
        end
   end
   return circ
end

function new_get_circuit(pepsu::GeneralPEPS, g)
    nbit = 10  # TODO: make this dynamic
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
             last_elements = Int[]
             for j in inneighbors
                 if !isnothing(remain_qubits[j]) && !isempty(remain_qubits[j])
                     push!(last_elements, remain_qubits[j][1])
                 end
             end
             
             target_qubits[i] = vcat(last_elements, free_qubits[1:n]) 
             target_qubits[i] = sort(target_qubits[i])
        @show target_qubits[i]
        @show free_qubits
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

function gensample(circ, reg::AbstractRegister, pepsu::GeneralPEPS, basis)
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
