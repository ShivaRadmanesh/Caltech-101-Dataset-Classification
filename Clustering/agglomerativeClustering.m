validFeatures = load('Dataset/validation/features.csv');
validLabels = load('Dataset/validation/Labels.csv');
testFeatures = load('Dataset/test/features.csv');
testLabels = load('Dataset/test/Labels.csv');
dataSize = size(testFeatures);
sz = dataSize(1);
cSZ = 15;
Z = linkage(testFeatures,'complete','correlation');
c = cluster(Z,'Maxclust',cSZ);

arr = zeros(cSZ, 15); 
%'j'th column of 'i'th row indicates the freq of real 'j'th label in 'i' th clustering label 

for i = 1: sz
    arr(c(i), testLabels(i, 1)) = arr(c(i,1), testLabels(i)) + 1;
end

for i = 1: cSZ
    for j = 1: 15
        fprintf('%d ', arr(i, j));
    end
    fprintf('\n');
   
end
label = zeros(cSZ); % 'i'th clustering label is related to label(i)th real label
maxFreq = zeros(cSZ);
sum = 0;
for i = 1: cSZ %calculating purity
    max = 0;
    maxLabel = 0;
    for j = 1: 15
        if arr(i, j) >= max
            max = arr(i, j);
            maxLabel = j;
        end
    end
    label(i) = maxLabel;
    maxFreq(i) = max;
    sum = sum + max;
end
purity = sum /sz;

realLabelsFrq = zeros(15);%realLabelFrq(i) indicates the freq of datas that are labeled i(the real label)
for i = 1: 15
    columnSum = 0;
    for j = 1: cSZ
        columnSum = columnSum + arr(j, i);
    end
    realLabelsFrq(i) = columnSum;
end
clusteringLabelsFrq = zeros(cSZ);
for i = 1: cSZ
    rowSum = 0;
    for j = 1: 15
        rowSum = rowSum + arr(i, j);
    end
    clusteringLabelsFrq(i) = rowSum;
end

%calculating rand index
N = (sz * (sz - 1)) / 2 ;
TP_FP  = 0;
for i = 1: cSZ
    TP_FP = TP_FP + (clusteringLabelsFrq(i) * (clusteringLabelsFrq(i) - 1)) / 2;
end
TP = 0;
for i = 1: cSZ
    TP = TP + (maxFreq(i) * (maxFreq(i) - 1)) / 2;
end
FP = TP_FP - TP;

labelFN = zeros(15);
for i = 1: 15
    labelFN(i) = (realLabelsFrq(i) * (realLabelsFrq(i) - 1)) / 2;
    for j = 1: cSZ
        labelFN(i) = labelFN(i) - (arr(j, i) * (arr(j, i) - 1)) / 2;
    end
end
FN = 0;
for i = 1: 15
    FN = FN + labelFN(i);
end
TN = N - FN - TP - FP;

randIndex = (TP + TN) / N; 


fprintf('labels \n');
for i = 1: cSZ
    fprintf('%d ', label(i));
end


fprintf('\n%d', sum);

fprintf('\nreal Labels freq:\n');
for i = 1: 15
    fprintf('%d ', realLabelsFrq(i));
end

fprintf('\nclustering Labels freq:\n');
for i = 1: cSZ
    fprintf('%d ', clusteringLabelsFrq(i));
end

fprintf('\nmax freq:\n');
for i = 1: cSZ
    fprintf('%d ', maxFreq(i));
end

fprintf('\n\nrand index: %d', randIndex);
fprintf('\npurity: %d', purity);

        