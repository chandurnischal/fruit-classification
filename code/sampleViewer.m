clc;
clear;


load("gray.mat")
plot4x4(data, labels, 1, 32);

load("pcaData.mat")
plot4x4(pcaData, labels, 1, 32);

load("gauss.mat")
plot4x4(kpcaData, labels, 1, 32);

load("poly.mat")
plot4x4(kpcaData, labels, 1, 32);

load("tsneData.mat")
plot4x4(tsneData, labels, 1, 32);

load("autoData.mat")
plot4x4(autoData, labels, 1, 32);



function plot4x4(data, labels, sample, dim)
    % Plots a 4x4 grid of images from the input data matrix along with
    % their corresponding labels. Each row of the grid represents a label
    % and displays the specified sample image resized to the specified
    % dimensions.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes/labels, N is the number of samples per class, and
    %           P is the number of dimensions per sample.
    %   - labels: Cell array containing labels corresponding to each class
    %             in the data matrix.
    %   - sample: Index of the sample to display for each class.
    %   - dim: Dimension of the images (assumed to be square).
    %
    % Example:
    %   plot4x4(data, labels, 1, 32);
    %
    % Note: 
    %   - This function assumes that the data matrix contains images
    %     organized such that each row corresponds to a class, and each
    %     column corresponds to a sample within that class.
    %   - The specified sample from each class is resized to the specified
    %     dimensions and displayed in a 4x4 grid.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    figure;  
    for i=1:numel(labels)
        subplot(4, 4, i)
        colormap('gray')
        image = data(i, sample, :);
        image = reshape(image, [dim, dim]);
        imagesc(image);
        axis off;
        title(labels{i});
    end
end
