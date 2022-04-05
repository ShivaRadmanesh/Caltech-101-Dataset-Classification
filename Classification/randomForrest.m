train_x = load('Dataset/train/features.csv');
train_y = load('Dataset/train/Labels.csv');
test_x = load('Dataset/test/features.csv');
test_y = load('Dataset/test/Labels.csv');



train10_x = load('sample/shivaTrainX10.csv');
train10_y = load('sample/shivaTrainY10.csv');

train5_x = load('sample/shivaTrainX5.csv');
train5_y = load('sample/shivaTrainY5.csv');

forward100Idx = load('100selectedIdxByAdaboost.csv');
forward200Idx = load('200selectedIdxByAdaboost.csv');
tr100_x = train5_x(:, forward100Idx);
tr200_x = train5_x(:,forward200Idx);
te100_x = test_x(:, forward100Idx);
te200_x = test_x(:, forward200Idx);



classifier =  TreeBagger(80,train5_x,train5_y,'OOBPrediction','On','Method','classification', 'NumPredictorsToSample', 60);
label = predict(classifier, test_x);

datasize = size(test_x);

correct = 0;
for i = 1 : datasize(1)
    if str2num(label{i}) == test_y(i)
       correct = correct + 1;
    end

end
acc = correct / datasize(1);


