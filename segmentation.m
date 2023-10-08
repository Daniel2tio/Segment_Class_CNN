%Importing the required data
image_directory = fullfile("daffodilSeg/ImagesRsz256/");
labels_directory = fullfile("daffodilSeg/LabelsRsz256/");
imds = imageDatastore(image_directory);

%Assigning the labels and pixel id's
classes = ["background","flower"];
pixel_ids = [3 1];

%Create data store of the classes and label ID's
pxds = pixelLabelDatastore(labels_directory, classes, pixel_ids);

%Splitting the data using 3 partitions, 60% for training, 20% for
%validation and the remaining 20% for testing using MATLAB's supporting
%partition function 
[images_train_set, images_val_set, images_test_set, pixels_train_set, pixels_val_set, pixels_test_set] = partitionflowerData(imds, pxds);

%Creating the network
%We begin by specifying the image input size and number of classes
input_size = [256 256 3];
num_of_classes = numel(classes);

%Creating the layers with the deeplab function

layers_graph = [
    imageInputLayer(input_size)
    convolution2dLayer(3,16,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same') % Adjusted layer
    reluLayer
    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same') % Adjusted layer
    reluLayer
    convolution2dLayer(1, num_of_classes)
    softmaxLayer()
    pixelClassificationLayer('Classes',classes)];

% Setting the validation data
data_validation = combine(images_val_set,pixels_val_set);

% Setting the training options
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',data_validation,...
    'MaxEpochs',20, ...  
    'MiniBatchSize',6, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

% Setting the training data
data_training = combine(images_train_set, pixels_train_set);

%Apply data augmentation
X_transpose = [-10 10];
Y_transpose = [-10 10];
data_training = transform(data_training, @(data)augmentImageAndLabel(data,X_transpose,Y_transpose));

%Start the training process
[segmentnet, info] = trainNetwork(data_training,layers_graph,options);


save segmentnet;

%Evaluating the model

pixels_results = semanticseg(images_test_set,segmentnet, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

metrics = evaluateSemanticSegmentation(pixels_results,pixels_test_set,'Verbose',false);

metrics.DataSetMetrics

metrics.ClassMetrics




%Supporting Functions
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionflowerData(imds,pxds)
% Partition data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = [3 1];

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

function data = augmentImageAndLabel(data, X_transpose, Y_transpose)
% Augment images and pixel label images using random reflection and
% translation.

for i = 1:size(data,1)
    
    tform = randomAffine2d(...
        'XReflection',true,...
        'XTranslation', X_transpose, ...
        'YTranslation', Y_transpose);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
    data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
    
end
end