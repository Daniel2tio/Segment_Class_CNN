% Loading the data
dataset = imageDatastore('17flowers\', 'IncludeSubfolders',true, 'LabelSource','foldernames'); 
%Our data is organised into subfolders which act as the labels

%Loading the pre-trained googlenet model
classnet = googlenet;

%We split the data into 60% for training and 40% for testing
%This is due to not having much data to train 
[Train_set, Test_set] = splitEachLabel(dataset, 6.0);

%Number of classes
classes = numel(categories(Train_set.Labels));

%Analysing the architecture of googlenet
%Here we can see every layer present in the model
%analyzeNetwork(classnet)

%We modify the feature learning layer and the output layer to take the number of classes we have
%which is 17

image_input_layer = classnet.Layers(1);
feature_learning_layer = classnet.Layers(142);
classification_output_layer = classnet.Layers(144);

input_size = [256 256 3];
new_image_input = imageInputLayer(input_size);

new_feature_learner = fullyConnectedLayer(classes,...
    'Name', 'Flower Feature Learner', ...
    'WeightLearnRateFactor', 10,...
    'BiasLearnRateFactor', 10);

new_classifier_output = classificationLayer('Name', 'Flower Classifier');

%We now modify the network architecture of googlenet by replacing the
%existing layers
layer_graph = layerGraph(classnet);

new_layers_architecture = replaceLayer(layer_graph, feature_learning_layer.Name, new_feature_learner);

new_layers_architecture = replaceLayer(new_layers_architecture, classification_output_layer.Name, new_classifier_output);

new_layers_architecture = replaceLayer(new_layers_architecture, image_input_layer.Name, new_image_input);

analyzeNetwork(new_layers_architecture)

%Resizing to meet the input dimension requirement of the model
%We do this on both the training and testing sets
resized_train_imds = augmentedImageDatastore(input_size, Train_set);
resized_test_imds = augmentedImageDatastore(input_size, Test_set);

%Defining the network training options
Minibatch_size = 5;
Validation_frequency = floor(numel(resized_train_imds.Files)/Minibatch_size);
train_options = trainingOptions('sgdm', ...
    'MaxEpochs',7, ...
    'InitialLearnRate', 0.0003, ...
    'MiniBatchSize', Minibatch_size, ...
    'Shuffle','every-epoch',...
    'ValidationData',resized_test_imds, ...
    'ValidationFrequency', Validation_frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%Training the model
classnet = trainNetwork(resized_train_imds, new_layers_architecture, train_options);

%Save the classifier
save classnet;

%Testing the model
testPreds = classify(classnet,resized_test_imds);

%Calculating the accuracy

YPred = classify(classnet,resized_test_imds);
YValidation = Test_set.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
disp(accuracy)
%Visualising the Confusion Matrix

confusionchart(Test_set.Labels, YPred)
