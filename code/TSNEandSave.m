function TSNEandSave(data, labels, dim)
    % Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) on the
    % input data matrix and saves the result along with the corresponding
    % labels. The t-SNE dimensionality reduction is applied to reduce the
    % data to a specified dimension (dim x dim).
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of samples, N is the number of features per sample, and P
    %           is the number of dimensions per feature.
    %   - labels: Cell array containing labels corresponding to each sample
    %             in the data matrix.
    %   - dim: Dimensionality to which the data will be reduced using t-SNE.
    %
    % Example:
    %   TSNEandSave(data, labels, 32);
    %
    % Note: 
    %   - This function performs t-SNE on the input data matrix using the
    %     'tSNE' method provided by the 'compute_mapping' function.
    %   - The resulting t-SNE-transformed data matrix is saved as
    %     'tsneData.mat' along with the corresponding labels.
    %   - The input data matrix 'data' must be double type for accurate t-SNE
    %     computation.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    tic;
    data = double(data);
    data_reshaped = reshapeForPCA(data);
    tsneData = compute_mapping(data_reshaped, "tSNE", dim*dim);
    tsneData = reshape(tsneData, size(data));
    save("tsneData.mat", "tsneData", "labels");
    elapsed = toc;
    fprintf("Time Taken for DR: %f seconds\n", elapsed);   
end
