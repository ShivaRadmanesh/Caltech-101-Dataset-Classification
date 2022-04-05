function [index] = modifiedPreprocess(features,lable)
%MODIFIEDPREPROCESS Summary of this function goes here
%   Detailed explanation goes here
datasize = size(features);

currFeatures = features;
fsize = size(currFeatures);

for i = 1 : datasize(2)
    currFeatures(datasize(1) + 1 , i) = i;
end

for i = 1 : 3000
   
    fsize = size(currFeatures);
    k = 1;
    delete = [];
    for j = (i + 1) : fsize(2)
         a = corrcoef(currFeatures(1 : fsize(1) -1, i),currFeatures(1 : fsize(1) -1, j));
         if a(1, 2) > 0.3
             delete(k, 1) = j;        
         end
         
    end
    currFeatures(:, delete) = [];
end

index = currFeatures(fsize(1),:);
index = transpose(index);
    
end

