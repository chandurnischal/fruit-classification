clc;
clear;

disp("Analyzing PCA Dataset...");
load("pcaData.mat")
PCvsAcc(pcaData);

disp("Analyzing Gaussian kPCA Dataset...");
load("gauss.mat")
PCvsAcc(kpcaData);

disp("Analyzing Polynomial kPCA Dataset...");
load("poly.mat")
PCvsAcc(kpcaData);

disp("Analyzing t-SNE Dataset...");
load("tsneData.mat")
PCvsAcc(tsneData)

disp("Analyzing AutoEncoder Dataset...");
load("autoData.mat")
PCvsAcc(autoData);



function PCvsAcc(data)
    % Computes the test accuracy of a KNN classifier with varying numbers
    % of principal components retained from the input data. Plots the test
    % accuracy against the number of principal components retained.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes, N is the number of samples per class, and P is
    %           the number of features per sample.
    %
    % Example:
    %   PCvsAcc(data);
    %
    % Note: 
    %   - This function uses the KNNClassification function to compute the
    %     test accuracy of a KNN classifier with varying numbers of
    %     principal components retained.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024

    % Initialize arrays to store accuracies and indices
    accs = zeros(1, size(data, 3));
    indices = zeros(1, size(data, 3));

    % Get the number of components in the input data
    components = size(data, 3);

    % Split the data into training, testing, and validation sets
    [train, test] = randomTrainTestSplit(data, 400);
    [test, valid] = randomTrainTestSplit(test, 200);

    % Loop over different numbers of principal components
    for i = 1:components
        if mod(i, 100) == 0
            fprintf("Iteration %d\n", i);
        end
        
        % Compute test accuracy using KNN classifier
        acc = KNNClassification(train(:, :, 1:i), test(:, :, 1:i), valid(:, :, 1:i), 16);
        
        % Store accuracy and index
        indices(i) = i;
        accs(i) = acc;
    end

    % Plot test accuracy vs. number of principal components retained
    figure;
    [acc_max, index] = max(accs);
    idx_max = indices(index);
    plot(indices, accs, idx_max, acc_max, 'ro');
    xlabel("No. of Principal Components Retained");
    ylabel("Test Accuracy")
    title("Test Accuracy v/s No. of Principal Components Retained");
    hold on
    plot([idx_max idx_max],[0 acc_max],':m');
    hold off
end

