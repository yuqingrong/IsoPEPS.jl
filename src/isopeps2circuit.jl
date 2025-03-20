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
    for i in peps.physical_labels
        out = max(length(outneighbors(g, i))+1,length(inneighbors(g, i)))
        peps.vertex_tensors[i] = reshape(peps.vertex_tensors[i], 2^out, :)
        Q, R = qr(peps.vertex_tensors[i])
        peps.vertex_tensors[i] = collect(Q)
    end
    return peps
end


function get_circuit(peps::GeneralPEPS, g)
   nbit = 6    # TODO: make this dynamic
   circ = chain(nbit)
   target_qubits = Vector{Vector{Int}}(undef, length(peps.physical_labels))
   free_qubits = collect(1:nbit)
   remain_qubits = Vector{Vector{Int}}(undef, length(peps.physical_labels))
   add!(block) = push!(circ, block)
   
   for i in 1:length(peps.physical_labels)
        inneighbors = Graphs.inneighbors(g, i)
        n = Int(log2(size(peps.vertex_tensors[i])[1]))-length(inneighbors)
        target_qubits[1] = free_qubits[1:n]
       
        last_elements = [remain_qubits[j][end] for j in inneighbors]
        target_qubits[i] = vcat(last_elements, free_qubits[1:n])
        for j in inneighbors
            remain_qubits[j] = remain_qubits[j][1:end-1] 
        end
        
        @show isassigned(target_qubits, 1)
        @show target_qubits[i]
        @show length(target_qubits[i]), size(peps.vertex_tensors[i])
        chain(nbit, put(nbit, Tuple(target_qubits[i])=>matblock(peps.vertex_tensors[i]))) |> add!
        Bag(basis_rotor(Z, nbit, target_qubits[i])) |> add!
        Measure(nbit; locs=target_qubits[i][1], resetto=0) |> add!
        free_qubits = vcat(target_qubits[i][1], free_qubits[n+1:end])
        remain_qubits[i] = target_qubits[i][2:end]
   end
   return circ
end


function gensample(circ, reg::AbstractRegister, batch_size::Int, peps::GeneralPEPS, basis)
    nbit = 6  # TODO: make this dynamic
    res = zeros(Int, batch_size, length(peps.physical_labels))
    bags = collect_blocks(Bag, circ)
    for bag in bags
        setcontent!(bag, basis_rotor(basis, nbit, occupied_locs(bag.content)))
    end

    for i in 1:batch_size
        copy(reg) |> circ
        results = collect_blocks(Measure, circ)
         for j in 1:length(peps.physical_labels)
            res[i,j] = results[j].results
        end
    end
    return res
end


function long_range_coherence(circ, reg::AbstractRegister, peps::GeneralPEPS, i::Int, j::Int, batch_size::Int)
    corr = 0.0
    for basis in [X, Y, Z]
        samples = gensample(circ, reg, batch_size, peps, basis)   
        corr = 0.0
    
        for shot in 1:batch_size
            corr += (-1)^(samples[shot,i] + samples[shot,j])
        end
       
        return abs(corr/batch_size)
    end
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
