clc;
clear;

disp("Performing KNN on Original Dataset...")
load("gray.mat")
KNNnTimes(data, labels, 10);

disp("Performing KNN on PCA Dataset...")
load("pcaData.mat")
KNNnTimes(pcaData(:, :, 1:19), labels, 10);

disp("Performing KNN on Gaussian kPCA Dataset...")
load("gauss.mat")
KNNnTimes(kpcaData(:, :, 1:11), labels, 10);

disp("Performing KNN on Polynomial kPCA Dataset...")
load("poly.mat")
KNNnTimes(kpcaData(:, :, 1:61), labels, 10);

disp("Performing KNN on t-SNE Dataset...")
load("tsneData.mat")
KNNnTimes(tsneData(:, :, 1:4), labels, 10);

disp("Performing KNN on Autoencoder Dataset...")
load("autoData.mat")
KNNnTimes(autoData(:, :, 1:92), labels, 10);