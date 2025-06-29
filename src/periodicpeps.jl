""" 
    PeriodicPEPS{T,NF,LT<:Union{Int,Char},Ein<:OrderedEinCode} <: PEPS{T,NF,LT}

Structure to hold periodic PEPS tensor network

# Fields
- `physical_labels::Vector{LT}`: Labels for physical indices
- `virtual_labels::Vector{LT}`: Labels for virtual indices
- `vertex_labels::Vector{Vector{LT}}`: Labels for vertex indices
"""
struct PeriodicPEPS{T,NF,LT<:Union{Int,Char},Ein<:OrderedEinCode} <: PEPS{T,NF,LT}
    physical_labels::Vector{LT}
    virtual_labels::Vector{LT}

    vertex_labels::Vector{Vector{LT}}
    vertex_tensors::Vector{<:AbstractArray{T}}
    max_index::LT

    # optimized contraction codes
    code_statetensor::Ein
    code_inner_product::Ein
    
    code_single_sandwich::Ein
    code_two_sandwich::Ein

    D::Int
end

function PeriodicPEPS{NF}(
    physical_labels::Vector{LT}, virtual_labels::Vector{LT}, 
    vertex_labels::Vector{Vector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}}, 
    max_index::LT, 
    code_statetensor::Ein, code_inner_product::Ein, 
    code_single_sandwich::Ein, code_two_sandwich::Ein, 
    D::Int) where {LT,T,NF,Ein}
    PeriodicPEPS{T,NF,LT,Ein}(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_index, code_statetensor, code_inner_product, code_single_sandwich, code_two_sandwich, D)
end

function PeriodicPEPS{NF}(virtual_labels::Vector{LT}, vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}}, D::Int, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {LT,T,NF}   
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))
    optcode_statetensor, optcode_inner_product = _optinner_product(vertex_labels, physical_labels, virtual_labels, max_ind, nflavor, D, optimizer, simplifier)
    optcode_single_sandwich = optsingle_sandwich(vertex_labels, vertex_tensors, max_ind, simplifier)
    optcode_two_sandwich = opttwo_sandwich(vertex_labels, vertex_tensors, max_ind, simplifier)
    PeriodicPEPS{NF}(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_ind, optcode_statetensor, optcode_inner_product, code_single_sandwich, code_two_sandwich, D)
end

function _optinner_product(vertex_labels::Vector{Vector{LT}}, physical_labels::Vector{LT}, virtual_labels::Vector{LT}, max_ind, nflavor, D, optimizer, simplifier)  where LT
    code_statetensor = EinCode(vertex_labels, physical_labels)
    size_dict = Dict([[l=>nflavor for l in physical_labels]..., [l=>D for l in virtual_labels]...])
    optcode_statetensor = optimize_code(code_statetensor, size_dict, optimizer, simplifier)  # generate a good contraction order according to the label and dimension.
    rep = [l=>max_ind+i for (i, l) in enumerate(virtual_labels)]  # max_ind+i: for virtual_labels of the other peps
    merge!(size_dict, Dict([l.second=>D for l in rep]))
    code_inner_product = EinCode([vertex_labels..., [replace(l, rep...) for l in vertex_labels]...], LT[])
    
    optcode_inner_product = optimize_code(code_inner_product, size_dict, optimizer, simplifier)
   
    return optcode_statetensor, optcode_inner_product
end

function _optsingle_sandwich(vertex_labels::Vector{Vector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}}, max_ind, simplifier)  where LT

end

function _opttwo_sandwich(vertex_labels::Vector{Vector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}}, max_ind, simplifier)  where LT

end


function zero_perpeps()
    virtual_labels = collect(nv(g)+1:nv(g)+ne(g))
    vertex_labels = Vector{Int}[] 
    vertex_tensors = Array{T}[]
    
end

function rand_perpeps()

end


"""direction is from left to right, from top to bottom"""

