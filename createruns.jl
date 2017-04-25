using RCall,PyCall,DataFrames
R"library(lhs)"
len = length


constrain= readtable("constrained_ics.csv",separator = ',')
changing = readtable("changing_ics.csv",separator = ',')

rows = 10
columns = len(changing[1])

[constrain[i]=constrain[2] for i in 1:rows]


R" lhsgrid = optimumLHS(n=$(columns), k=$(rows), maxSweeps=20, eps=.1, verbose=FALSE)"
@rget lhsgrid

while condition
  body
end

#changing[:species]
