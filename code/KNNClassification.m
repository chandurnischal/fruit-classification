function [testAccuracy, validAccuracy, executionTime, perClassAccuracy] = KNNClassification(trainData, testData, validData, k)
    % KNNClassification(trainData, testData, validData, k)
    % 
    % Performs k-Nearest Neighbors (KNN) classification on the training
    % dataset and evaluates the model on both testing and validation
    % datasets. Returns the test accuracy, validation accuracy, execution
    % time, and per-class accuracy.
    % 
    % Inputs:
    %   - trainData: Training data matrix of size C x M x N, where C is the
    %                number of classes, M is the number of samples per class,
    %                and N is the number of features per sample.
    %   - testData: Testing data matrix of size C x P x Q, where C is the
    %               number of classes, P is the number of samples per class,
    %               and Q is the number of features per sample.
    %   - validData: Validation data matrix of size C x R x S, where C is
    %                the number of classes, R is the number of samples per
    %                class, and S is the number of features per sample.
    %   - k: Number of neighbors to consider in the KNN algorithm.
    %
    % Outputs:
    %   - testAccuracy: Accuracy of the KNN model on the testing dataset.
    %   - validAccuracy: Accuracy of the KNN model on the validation dataset.
    %   - executionTime: Time taken to train the KNN model in seconds.
    %   - perClassAccuracy: Per-class accuracy of the KNN model on the
    %                       testing dataset, represented as a vector of size
    %                       1 x C, where C is the number of classes.
    %
    % Example:
    %   [testAccuracy, validAccuracy, executionTime, perClassAccuracy] = KNNClassification(trainData, testData, validData, 5);
    %
    % Note: 
    %   - This function uses the fitcknn function from MATLAB's Statistics
    %     and Machine Learning Toolbox to perform KNN classification.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    % Convert input data to double
    trainData = double(trainData);
    testData = double(testData);
    validData = double(validData);

    % Get sizes of datasets
    trainSamples = size(trainData, 2);
    testSamples = size(testData, 2);
    validSamples = size(validData, 2);
    components = size(trainData, 3);
    classes = size(trainData, 1);

    % Reshape data matrices
    trainFlat = reshape(trainData, [classes*trainSamples, components]);
    testFlat = reshape(testData, [classes*testSamples, components]);
    validFlat = reshape(validData, [classes*validSamples, components]);

    % Extracting labels
    trainLabels = repmat((1:classes)', trainSamples, 1);
    testLabels = repmat((1:classes)', testSamples, 1);
    validLabels = repmat((1:classes)', validSamples, 1);

    % Start clock
    tic;

    % Performing KNN Classification
    model = fitcknn(trainFlat, trainLabels, 'NumNeighbors', k, 'Distance','euclidean');

    % Stop clock
    executionTime = toc;

    % Making predictions on the testing and validation datasets
    testPredictions = predict(model, testFlat);
    validPredictions = predict(model, validFlat);

    % Measuring accuracy of model on test data
    testAccuracy = sum(testLabels == testPredictions) / numel(testLabels);  

    % Measuring accuracy of model on validation data
    validAccuracy = sum(validLabels == validPredictions) / numel(validLabels);  

    % Computing per-class accuracy
    perClassAccuracy = zeros(1, classes);
    for i = 1:classes
        classIndices = testLabels == i;
        classAccuracy = sum(testPredictions(classIndices) == i) / sum(classIndices);
        perClassAccuracy(i) = classAccuracy;
    end
end
