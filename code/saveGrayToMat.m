function saveGrayToMat(folder_path, dim)
    % Recursively iterates through a folder containing RGB images organized
    % in subfolders corresponding to their respective labels. Converts the RGB
    % images into grayscale, resizes them from 100x100 to dim x dim, and stores
    % them in a matrix of shape 16x625x(dim*dim). The resulting matrix is saved as
    % 'gray.mat' in the current working directory.
    % 
    % Inputs:
    %   - folder_path: A string specifying the path to the root folder
    %                  containing the subfolders of RGB images.
    %   - dim: Dimension to resize the images to.
    %
    % Example:
    %   saveGrayToMat('path/to/root/folder', 32);
    %
    % Note: 
    %   - This function assumes that the subfolders in the provided directory
    %     represent the labels, and each contains RGB images to be processed.
    %   - Ensure that the root folder contains only subfolders representing
    %     labels and no other files or folders, as the function processes all
    %     images found in these subfolders.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    fprintf("Detecting labels...\n")
    subfolders = dir(folder_path);
    
    subfolders = subfolders(arrayfun(@(x) x.name(1), subfolders) ~= '.');
    
    data = zeros(length(subfolders), 625, dim*dim);
    labels = cell(1, length(subfolders));
    
    fprintf("Iterating through labels...\n")
    for i = 1:length(subfolders)
        subfolder_name = subfolders(i).name;
        image_files = dir(fullfile(folder_path, subfolder_name, '*.jpg'));
        selected_indices = randperm(length(image_files), 625);
        for j = 1:625
            img = imread(fullfile(folder_path, subfolder_name, image_files(selected_indices(j)).name));
            img = imresize(img, [dim, dim]);
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            
            img_vector = img(:)';
            
            data(i, j, :) = img_vector;
        end
        
        labels{i} = subfolder_name;
    end
    
    fprintf("Saving data to .mat file...\n");
    save("gray.mat", 'data', 'labels');
end
