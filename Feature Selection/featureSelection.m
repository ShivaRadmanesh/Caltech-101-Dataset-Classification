train_x = load('Dataset/train/features.csv');
train_y = load('Dataset/train/Labels.csv');
test_x = load('Dataset/test/features.csv');
test_y = load('Dataset/test/Labels.csv');
%dataSize = size(valid_x);
%sz = dataSize(1);

valid_sample_x = train_x(1 : 100, :);
valid_sample_y = train_y(1 : 100, :);
train_sample_x = train_x(101 : 450, :);
train_sample_y = train_y(101 : 450, :);

[selected, idx] = forward(train_sample_x, train_sample_y, 200, valid_sample_x, valid_sample_y);
%[selected, idx] = forward(train_x , train_y, 100);


writematrix(idx,'200selectedIdxByForward.csv');
writematrix(selected,'200SelectedfeaturesByForward.csv');
