function AutoEncoderandSave(data, labels, dim)
    % Applies an autoencoder for dimensionality reduction (DR) on the input
    % data matrix and saves the result along with the corresponding labels.
    % The autoencoder dimensionality reduction is applied to reduce the data
    % to a specified dimension (dim x dim).
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of samples, N is the number of features per sample, and P
    %           is the number of dimensions per feature.
    %   - labels: Cell array containing labels corresponding to each sample
    %             in the data matrix.
    %   - dim: Dimensionality to which the data will be reduced using the
    %          autoencoder.
    %
    % Example:
    %   AutoEncoderandSave(data, labels, 32);
    %
    % Note: 
    %   - This function applies an autoencoder for dimensionality reduction
    %     on the input data matrix using the 'Autoencoder' method provided
    %     by the 'compute_mapping' function.
    %   - The resulting autoencoder-transformed data matrix is saved as
    %     'autoData.mat' along with the corresponding labels.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    tic;
    data_reshaped = reshapeForPCA(data);
    autoData = compute_mapping(data_reshaped, "Autoencoder", dim*dim);
    elapsed = toc;
    autoData = reshape(autoData, size(data)); 
    autoData = real(autoData);
    save("autoData.mat", "autoData", "labels");
    fprintf("Time Taken for DR: %f seconds\n", elapsed);
end

