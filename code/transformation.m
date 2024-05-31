clear;
clc;
addpath("drtoolbox")
addpath("drtoolbox/techniques")
load('gray.mat')

% Perform PCA and save transformation as matrix
PCAandSave(data, labels, 32);

% Perform Polynomial kPCA and save transformation as matrix
polyPCAandSave(data, labels, 32);

% Perform Gaussian kPCA and save transformation as matrix
gaussPCAandSave(data, labels, 32);

% Perform t-SNE and save transformation as matrix
TSNEandSave(data, labels, 32);

% Perform autoencoder embedding and save transformation as matrix
AutoEncoderandSave(data, labels, 32);
