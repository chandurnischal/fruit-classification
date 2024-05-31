function data_reshaped = reshapeForPCA(data)

    % Reshapes the input data matrix into a format suitable for Principal
    % Component Analysis (PCA) by converting it into a 2D matrix where each
    % row represents a sample and each column represents a feature.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes, N is the number of samples per class, and P is
    %           the number of features per sample.
    %
    % Output:
    %   - data_reshaped: Reshaped data matrix of size (M*N) x P, suitable
    %                    for PCA.
    %
    % Example:
    %   data_reshaped = reshapeForPCA(data);
    %
    % Note: 
    %   - This function converts the input data matrix into a 2D matrix
    %     where each row represents a sample and each column represents a
    %     feature, making it suitable for PCA.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    % Reshape the input data matrix
    data_reshaped = reshape(data, [size(data, 1) * size(data, 2), size(data, 3
