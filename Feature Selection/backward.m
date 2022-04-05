function [selectedFeatures,iindex] = backward(features, label, featureNum, valid_x, valid_y)
%BACKWARD Summary of this function goes here
%   Detailed explanation goes here
datasize = size(features);
delete = datasize(2) - featureNum;
currFeatures = features;
currValid = valid_x;


for i = 1 : datasize(2)
    currFeatures(datasize(1) + 1 , i) = i;
end

deleteIdx = zeros(delete, 1);
for i = 1 : delete
    currErr = 1000;
    realIdx = -1;
    currWorstIdx = -1;
    fSize  = size(currFeatures);
    for j = 1 : fSize(2) 
        itrFeatures = currFeatures(1 : fSize(1) - 1,:);
        itrFeatures(:, j) = [];
        itrValid  = currValid;
        itrValid(:, j) = [];
        classifier = fitctree(itrFeatures,label);
        classErr = loss(classifier,itrValid ,valid_y);
        if classErr <  currErr
            currErr = classErr;
            currWorstIdx = j;
            realIdx = currFeatures(fSize(1) , j);
        end
    end
        currFeatures(:, currWorstIdx) = [];
        currValid(:, currWorstIdx) = [];
        deleteIdx(i, 1) = realIdx;
        fprintf('%i\n', i);
        fprintf('%f\n', currErr);
end

iindex = zeros(datasize(2), 1);
for i = 1 : datasize(2)
    iindex(i,1) = i;
end

iindex(deleteIdx, :) = [];
selectedFeatures = currFeatures;


end