clear all;
close all;
clc;

%Load Data Tables 
%features=importdata('features.mat');%k x m matrix with k being the number of images i.e. each row is one image
features=importdata('features2.mat');%k x m matrix with k being the number of images i.e. each row is one image
groundtruth=importdata('GroundTruth.mat');

%isolate conclusive data 
gt_flags=groundtruth(:,1); 
ft=[gt_flags features]; 

%ftc = feature matrix conclusive
ftc=ft(1,:);%initialize ftc

%isolate all images that are conclusive
for i=1:size(ft,1)
    if (ft(i,1)==1)
        ftc=[ftc;ft(i,:)]; %concatenate vertically to create conclusive feature matrix
    end
end
%strip the conclusive column and the first row
%ftc=ftc(2:size(ftc,1),2:size(ftc,2));
ftc=ftc(3:size(ftc,1),2:size(ftc,2));

%isolate gt for conclusive data 
gtc=groundtruth(1,:);
for i=1:size(groundtruth,1);
    if (groundtruth(i,1)==1)
        gtc=[gtc;groundtruth(i,:)];
    end
end
%strip conclusive column and the first row
%gtc=gtc(2:size(gtc,1),2:size(gtc,2));
gtc=gtc(3:size(gtc,1),2:size(gtc,2)); %note to self: the first ground truth example is rated as content-feeling



%%%%%%%%%%%%%%%%%%%%%

inputs = ftc';
targets = gtc'; 



%Cross Validation Shennanigans 
k = 6; %number of folds
samplesize = size(inputs, 2); %number of columns...
cv = cvpartition(samplesize,'kfold',k); %create a k fold cross validation 

%initialize matrices to contain the indices
trainingMat=zeros(k,cv.TrainSize(1)); %choose the first because I assume they're all the same. In this case it's 190
testingMat=zeros(k,cv.TestSize(1)); %In this case it's cv.TestSize ... 38
for i=1:k
    trainIdxs = find(training(cv,i)); %1 means in train, 0 means in test
    testIdxs = find(test(cv,i)); %1 means in test, 0 means in train
    %size(trainIdxs)
    trainingMat(i,:) = trainIdxs'; % therefore the number of the rows is the number of the fold
    testingMat(i,:) = testIdxs';
end




% # of Neurons per layer: 10^1,2,3,4...10 number of neurons
% # of Layers: 1-->10

%create row vector of neurons
% n=10; 
% vec=zeros(1,n);
% for i=1:n
%     vec(1,i)=10^i;
% end
% [nNeurons, nLayers]=meshgrid(vec, 1:1:10); %let's create a grid of parameters
[nNeurons, nLayers]=meshgrid(10:10:100, 1:1:10); %let's create a grid of parameters

%iterate through all the possible parameters
%for i=1:numel(nNeurons)

average=zeros(numel(nNeurons),1);%ten rows, one column
%average=zeros(5,1);%ten rows, one column
for i=1:numel(nNeurons)
%for i=1:5
    nNeuronsIteration = nNeurons(i);
    nLayersIteration = nLayers(i);
    hiddenLayerSize = repmat(nNeuronsIteration,1, nLayersIteration);
    
    net = patternnet(hiddenLayerSize); %create ann with the new parameters
        
    %Create Neural Net for each possible training set in the k-fold
        perf=zeros(k,1); %initialize matrix
        for m=1:k %we want to create a neural net for each training set in the k-fold
            [net,tr]=train(net,inputs(:,trainingMat(m,:)),targets(:,trainingMat(m,:)));
            outputs = net(inputs(:,testingMat(m,:)));
            %errors = gsubtract(targets(testingMat(i,:)),outputs);
            perf(m,1) = mse(net,targets(:,testingMat(m,:)),outputs);
            name=strcat(int2str(i),'Net', int2str(m), '.mat');
            save(name, 'net')
        end
        average(i,1)=mean(perf);
end
 matrixplotmanual = [nNeurons(1:1:79)' nLayers(1:1:79)' , average(1:79) ] %save data to x, y and z
[~,ind_bestnet]=min(average); %find index of the minimum value of the average mean square errors 
%[~,ind]=max(outputs,[],1);%find max of column

% Note: Multiple line comments: %{ and %}
