function [selectedFeatures,index] = forward(features, label, featureNum, valid_x, valid_y)

featuresCopy = features;
dataSize = size(features);
selectedFeatures = zeros(dataSize(1), featureNum);
index = zeros(featureNum, 1);
for i = 1: featureNum
    currIdx = -1; 
    currErr = 1000;
    fSize = size(featuresCopy);
    for j = 1 : fSize(2)
        selectedFeatures(:,i) = featuresCopy(:,j);
        classifier = fitctree(selectedFeatures(:,1:i),label);
        classErr = loss(classifier, valid_x(:,1:i), valid_y);
        if classErr < currErr %finding the classifier with minimum error
            curr = featuresCopy(:,j:j);
            currIdx = j;
            currErr = classErr;
        end
    end
    selectedFeatures(:,i: i) = curr;
    index(i, 1) = currIdx;
    featuresCopy(:, currIdx) = [];
    fprintf('%i\n', i);
    fprintf('%f\n', currErr)
end

end

