function polyPCAandSave(data, labels, dim)
    % Performs kernel Principal Component Analysis (kPCA) on the input
    % data matrix and saves the result along with the corresponding labels.
    % The kPCA dimensionality reduction is applied to reduce the data to a
    % specified dimension (dim x dim) using a polynomial kernel.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of samples, N is the number of features per sample, and P
    %           is the number of dimensions per feature.
    %   - labels: Cell array containing labels corresponding to each sample
    %             in the data matrix.
    %   - dim: Dimensionality to which the data will be reduced using kPCA.
    %
    % Example:
    %   polyPCAandSave(data, labels, 32);
    %
    % Note: 
    %   - This function performs kPCA on the input data matrix using the
    %     polynomial kernel.
    %   - The resulting kPCA-transformed data matrix is saved as 'poly.mat'
    %     along with the corresponding labels.
    %   - The input data matrix 'data' must be preprocessed and reshaped
    %     appropriately for kPCA.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    tic;
    data_reshaped = reshapeForPCA(data);
    kpcaData = kernel_pca(data_reshaped, dim*dim, 'poly');
    elapsed = toc;
    kpcaData = reshape(kpcaData, size(data)); 
    kpcaData = real(kpcaData);
    save("poly.mat", "kpcaData", "labels");
    fprintf("Time Taken for DR: %f seconds\n", elapsed);
end
