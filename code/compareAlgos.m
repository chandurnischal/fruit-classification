clc;
clear;
load("gray.mat")

% Splitting data into train, test and validatation datasets
[train, test] = randomTrainTestSplit(data, 400);
[test, valid] = randomTrainTestSplit(test, 200);

% Performing KNN Classification
[testAcc, validAcc, execTime, ~] = KNNClassification(train, test, valid, 16);
fprintf("KNN Classification Results...\nTest Accuracy: %f\nValidation Accuracy: %f\nTime Taken for Classification: %f seconds\n", testAcc, validAcc, execTime);

% Performing kSVM Classification
[testAcc, validAcc, execTime, ~] = kSVMClassification(train, test, valid);
fprintf("kSVM Classification Results...\nTest Accuracy: %f\nValidation Accuracy: %f\nTime Taken for Classification: %f seconds\n", testAcc, validAcc, execTime);
