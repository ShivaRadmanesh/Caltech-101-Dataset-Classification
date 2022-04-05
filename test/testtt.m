train_x = load('Dataset/train/features.csv');
train_y = load('Dataset/train/Labels.csv');
test_x = load('Dataset/test/features.csv');
test_y = load('Dataset/test/Labels.csv');

%t = templateSVM('Standardize',true,'KernelFunction','rbf');
%Mdl = fitcecoc(train_x,train_y,'Learners',t, 'CrossVal','on');
%kflc = kfoldLoss(Mdl);
%estGenError = kflc(end)


valid_sample_x = train_x(1 : 100, :);
valid_sample_y = train_y(1 : 100, :);
train_sample_x = train_x(101 : 450, :);
train_sample_y = train_y(101 : 450, :);
%[selected, idx] = forward(train_sample_x, train_sample_y, 5, valid_sample_x, valid_sample_y);
%writematrix(idx,'2selectedIdxByForward.csv');
%writematrix(selected,'2SelectedfeaturesByForward.csv');

index = preprocess(train_sample_x, train_sample_y, 1000, valid_sample_x, valid_sample_y);

tx = train_sample_x(:, index(:,1));
vx = valid_sample_x(:, index(:,1));


%[selected, idx] = backward(tx, train_sample_y, 5, vx, valid_sample_y);
features = tx;
label = train_sample_y;
featureNum = 5;
valid_x = vx;
valid_y = valid_sample_y;



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
