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
        reshaped = reshape(peps.vertex_tensors[i], 2^out, :)
       
        @assert size(reshaped, 2) <= size(reshaped, 1)
        Q, R = qr(reshaped)
        @assert Q[:,1:size(reshaped,2)] * R ≈ reshaped
        ugates.vertex_tensors[i] = collect(Q)
        # Take only the first columns of Q to match original size
        n_cols = size(reshaped, 2)
        Q = Q[:, 1:n_cols]
        peps.vertex_tensors[i] = reshape(Q, original_size)
    
    end
    return peps, ugates
end


function get_circuit(pepsu::GeneralPEPS, g)
   nbit = 6    # TODO: make this dynamic
   circ = chain(nbit)
   target_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   free_qubits = collect(1:nbit)
   remain_qubits = Vector{Vector{Int}}(undef, length(pepsu.physical_labels))
   add!(block) = push!(circ, block)
   
   for i in 1:length(pepsu.physical_labels) 
        inneighbors = Graphs.inneighbors(g, i)
        n = Int(log2(size(pepsu.vertex_tensors[i])[1]))-length(inneighbors)
        target_qubits[1] = free_qubits[1:n]
       
        last_elements = [remain_qubits[j][end] for j in inneighbors]
        target_qubits[i] = vcat(last_elements, free_qubits[1:n]) 
        target_qubits[i] = sort(target_qubits[i])
        for j in inneighbors
            remain_qubits[j] = remain_qubits[j][1:end-1] 
        end
         
        
        chain(nbit, put(nbit, Tuple(target_qubits[i])=>matblock(pepsu.vertex_tensors[i]))) |> add!
        Bag(basis_rotor(Z, nbit, target_qubits[i])) |> add!
        Measure(nbit; locs=target_qubits[i][1], resetto=0) |> add!
        free_qubits = vcat(target_qubits[i][1], free_qubits[n+1:end])
        remain_qubits[i] = target_qubits[i][2:end]
     
   end
   return circ
end


function gensample(circ, reg::AbstractRegister, pepsu::GeneralPEPS, basis)
    nbit = nqubits(reg)
    res = zeros(Int, nbatch(reg), length(pepsu.physical_labels))
    
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
         Si = mean((-1)^samples[shot,i] for shot in 1:nbatch(reg))
         Sj = mean((-1)^samples[shot,j] for shot in 1:nbatch(reg))
         
         # Calculate two-site correlation
         SiSj = mean((-1)^(samples[shot,i] + samples[shot,j]) for shot in 1:nbatch(reg))
         
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
