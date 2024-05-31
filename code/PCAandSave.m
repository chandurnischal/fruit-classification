function PCAandSave(data, labels, dim)
    % Performs Principal Component Analysis (PCA) on the input data matrix
    % and saves the result along with the corresponding labels. The PCA
    % dimensionality reduction is applied to reduce the data to a specified
    % dimension (dim x dim).
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of samples, N is the number of features per sample, and P
    %           is the number of dimensions per feature.
    %   - labels: Cell array containing labels corresponding to each sample
    %             in the data matrix.
    %   - dim: Dimensionality to which the data will be reduced using PCA.
    %
    % Example:
    %   PCAandSave(data, labels, 32);
    %
    % Note: 
    %   - This function performs PCA on the input data matrix using the
    %     'PCA' method provided by the 'compute_mapping' function.
    %   - The resulting PCA-transformed data matrix is saved as 'pcaData.mat'
    %     along with the corresponding labels.
    %   - The input data matrix 'data' must be double type for accurate PCA
    %     computation.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    data = double(data);
    tic;
    data_reshaped = reshapeForPCA(data);
    pcaData = compute_mapping(data_reshaped, 'PCA', dim*dim);
    elapsed = toc;
    pcaData = reshape(pcaData, size(data)); 
    pcaData = real(pcaData);
    save("pcaData.mat", "pcaData", "labels");
    fprintf("Time Taken for DR: %f seconds\n", elapsed);
end
