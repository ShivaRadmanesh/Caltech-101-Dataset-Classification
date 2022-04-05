weight = load('weight.csv');
fsize = size(weight);
for i = 1 : fsize(1)
    weight(i, 2) = i;
end

sortedFeatures = sortrows(weight, 1, 'descend');
adaboost100top = sortedFeatures(1 : 100, 2);
adaboost200top = sortedFeatures(1 : 200, 2);

writematrix(adaboost100top,'100selectedIdxByAdaboost.csv');
writematrix(adaboost200top,'200SelectedIdxByAdaboost.csv');