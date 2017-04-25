using RCall,PyCall,DataFrames
R"library(lhs)"
len = length


rows = 10


constrain= readtable("constrained_ics.csv",separator = ',')
changing = readtable("changing_ics.csv",separator = ',')
columns = len(changing[1])
[constrain[Symbol("x$i")] = constrain[:x1] for i in 1:rows]
R" lhsgrid = optimumLHS(n=$(columns), k=$(rows), maxSweeps=20, eps=.1, verbose=FALSE)"
@rget lhsgrid
#concat df vcat or hcat  # .operation = appply to all elements 
lhsgrid =  ((changing[:max]-changing[:min]) .* lhsgrid) .+ changing[:min]
lhsgrid =  DataFrame(Array(hcat( changing[:species],lhsgrid)))
indexarray = [Symbol("x$(i)") for i in 0:rows]
indexarray[1] = "species" 
names!(lhsgrid,indexarray)
ics = vcat(constrain,lhsgrid)

writetable("latin.csv", ics, separator = ',', header = false,quotemark=' ')

#changing[:species]
print("ahhh",ics)



