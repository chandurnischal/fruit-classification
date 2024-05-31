function [train, test] = randomTrainTestSplit(data, trainSamples)  
    % Splits the input data matrix into training and testing sets using a
    % random permutation of sample indices.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes, N is the total number of samples, and P is the
    %           number of features per sample.
    %   - trainSamples: Number of samples to include in the training set.
    %
    % Outputs:
    %   - train: Training data matrix of size M x trainSamples x P.
    %   - test: Testing data matrix of size M x (N - trainSamples) x P.
    %
    % Example:
    %   [train, test] = randomTrainTestSplit(data, 400);
    %
    % Note: 
    %   - This function randomly shuffles the sample indices and splits the
    %     input data matrix into training and testing sets based on the
    %     specified number of training samples.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    % Generate a random permutation of sample indices
    indices = randperm(size(data, 2));
    
    % Select samples for training and testing based on the random permutation
    train = data(:, indices(1:trainSamples), :);
    test = data(:, indices(trainSamples + 1:end), :);
end
