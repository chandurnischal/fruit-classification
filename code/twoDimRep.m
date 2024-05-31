clc;
clear;

load("gray.mat")
twoDimRepresentation(data, labels);

load("pcaData.mat")
twoDimRepresentation(pcaData, labels);

load("gauss.mat")
twoDimRepresentation(kpcaData, labels);

load("poly.mat")
twoDimRepresentation(kpcaData, labels);

load("tsneData.mat")
twoDimRepresentation(tsneData, labels);

load("autoData.mat")
twoDimRepresentation(autoData, labels);


function twoDimRepresentation(data, labels)
    % Plots a two-dimensional representation of the input data using the
    % first two components obtained from dimensionality reduction (DR).
    % Each class is represented by a distinct color in the plot, and the
    % legend indicates the corresponding labels.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes/labels, N is the number of samples per class, and
    %           P is the number of dimensions per sample.
    %   - labels: Cell array containing labels corresponding to each class
    %             in the data matrix.
    %
    % Example:
    %   twoDimRepresentation(data, labels);
    %
    % Note: 
    %   - This function assumes that the input data has been reduced to two
    %     dimensions using dimensionality reduction techniques such as PCA,
    %     t-SNE, or autoencoder.
    %   - Each class in the data is represented by a distinct color, and
    %     the legend indicates the corresponding labels.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    % Extracting only the first two dimensions of the data
    data = data(:, :, 1:2);
    
    % Reshape data for plotting
    data = reshapeForPCA(data);
    
    % Determine the number of unique labels
    num_labels = numel(unique(labels));
    
    % Generate a colormap with distinct colors for each label
    color_map = jet(num_labels);
    
    % Plotting
    figure;
    hold on;
    for i = 1:num_labels
        idx = (i - 1) * 625 + 1 : i * 625;
        scatter(data(idx, 1), data(idx, 2), 20, color_map(i, :), 'o');
    end
    
    % Adding legend
    legend(labels, "Location","eastoutside");    
    hold off;
    
    % Title and axis labels
    title('Data Representation');
    xlabel('First Component');
    ylabel('Second Component');
end
