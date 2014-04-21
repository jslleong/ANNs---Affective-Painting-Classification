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

ftc=ftc(3:size(ftc,1),2:size(ftc,2)); %strip the conclusive column and the first row

%isolate gt for conclusive data 
gtc=groundtruth(1,:);
for i=1:size(groundtruth,1);
    if (groundtruth(i,1)==1)
        gtc=[gtc;groundtruth(i,:)];
    end
end
gtc=gtc(3:size(gtc,1),2:size(gtc,2)); %strip conclusive column and the first row %note to self: the first ground truth example is rated as content-feeling

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

[nNeurons, nLayers]=meshgrid(10:10:100, 1:1:10); %create grid of parameters (neursons per later, number of layers)

average=zeros(numel(nNeurons),1);
for i=1:numel(nNeurons)
    nNeuronsIteration = nNeurons(i);
    nLayersIteration = nLayers(i);
    hiddenLayerSize = repmat(nNeuronsIteration,1, nLayersIteration);
    
    for layercount = 1:nLayersIteration
        net.layers{layercount}.transferFcn = 'radbas';
    end
    
    net = patternnet(hiddenLayerSize); %create ann with the new parameters
        
    %Create Neural Net for each possible training set in the k-fold
        perf=zeros(k,1); %initialize matrix
        for m=1:k %train the neural net for each training set in the k-fold %default training method is trainscg (which allows for training with backpropagation)
            [net,tr]=train(net,inputs(:,trainingMat(m,:)),targets(:,trainingMat(m,:)));
            outputs = net(inputs(:,testingMat(m,:)));
            %errors = gsubtract(targets(testingMat(i,:)),outputs);
            perf(m,1) = mse(net,targets(:,testingMat(m,:)),outputs);
            name=strcat(int2str(i),'Net', int2str(m), '.mat');
            save(name, 'net')
        end
        average(i,1)=mean(perf);
end
 
[~,ind_bestnet]=min(average); %find index of the minimum value of the average mean square errors 

% Note: Multiple line comments: %{ and %}
% Note: matrixplotmanual = [nNeurons(1:1:79)' nLayers(1:1:79)' , average(1:79) ] %save data to x, y and z