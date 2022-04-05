train_x = load('Dataset/train/features.csv');
train_y = load('Dataset/train/Labels.csv');
test_x = load('Dataset/test/features.csv');
test_y = load('Dataset/test/Labels.csv');


%t = templateSVM('Standardize',true,'KernelFunction','rbf');
%Mdl = fitcecoc(train_x,train_y,'Learners',t);
%classifier = Mdl.Trained{1};

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

t = templateSVM('Standardize',true, 'KernelFunction', 'linear');
classifier = fitcecoc(train5_x,train5_y,'Holdout', 0.15 ,'Learners',t);
Mdl = classifier.Trained{1}; 


datasize = size(test_y);


label = predict(Mdl, test_x);
correct = 0;
for i = 1 : datasize(1)
    if label(i) == test_y(i)
       correct = correct + 1;
    end

end
acc = correct / datasize(1);
