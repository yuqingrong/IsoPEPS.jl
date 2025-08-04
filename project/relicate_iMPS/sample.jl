mutable struct Bag{D}<:TagBlock{AbstractBlock, D}
    content::AbstractBlock{D}
end

Yao.content(bag) = bag.content    
Yao.chcontent(bag::Bag, content) = Bag(content) 
Yao.mat(::Type{T}, bag::Bag) where T = mat(T, bag.content)  
YaoBlocks.unsafe_apply!(reg::AbstractRegister, bag::Bag) = YaoBlocks.unsafe_apply!(reg, bag.content)  
YaoBlocks.PreserveStyle(::Bag) = YaoBlocks.PreserveAll()  
setcontent!(bag::Bag, content) = (bag.content = content; bag)  

basis_rotor(::ZGate) = I2Gate()
basis_rotor(::XGate) = Ry(-0.5π)  
basis_rotor(::YGate) = Rx(0.5π)   

basis_rotor(basis, nbit, locs) = repeat(nbit, basis_rotor(basis), locs)

function gensample(circ, reg::AbstractRegister, basis)  
    nbit = nqubits(reg) 
    bags = collect_blocks(Bag, circ) 
    for bag in bags
        setcontent!(bag, basis_rotor(basis, nbit, occupied_locs(bag.content)))
    end
    
    copy(reg) |> circ
    result = collect_blocks(Measure, circ)
    res = zeros(Int, nbatch(reg), length(result[].locations))
    for j in 1:size(res,1)
        res[j,:] = collect(digits(result[].results[j], base=2, pad=size(res,2)))
    end    
    return res
end
