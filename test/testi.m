train_x = load('Dataset/train/features.csv');
train_y = load('Dataset/train/Labels.csv');
test_x = load('Dataset/test/features.csv');
test_y = load('Dataset/test/Labels.csv');

index = modifiedPreprocess(train_x);
%for i = 2 : datasize(2)
    %a = corrcoef(train_x(:, 1),train_x(:, i));
    %if a(1,2) > max
     %   max = a(1,2);
    %end
%end

%features = train_x;
%label = train_y;


%datasize = size(features);

%currFeatures = features;
%fsize = size(currFeatures);

%for i = 1 : datasize(2)
  %  currFeatures(datasize(1) + 1 , i) = i;
%end

%for i = 1 : 3000
   
 %   fsize = size(currFeatures);
  %  k = 1;
   % delete = [];
    %for j = (i + 1) : fsize(2)
     %    a = corrcoef(currFeatures(1 : fsize(1) -1, i),currFeatures(1 : fsize(1) -1, j));
      %   if a(1, 2) > 0.35
       %      delete(k, 1) = j;        
        % end
         
    %end
    %currFeatures(:, delete) = [];
%end

%index = currFeatures(fsize(1),:);

writematrix(index,'modifiedPreprocess2.csv');