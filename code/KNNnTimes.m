function KNNnTimes(data, labels, n)
    % Performs KNN classification multiple times and computes the average
    % test accuracy, validation accuracy, execution time, and per-class
    % test accuracy over the specified number of simulations.
    % 
    % Inputs:
    %   - data: Input data matrix of size M x N x P, where M is the number
    %           of classes, N is the number of samples per class, and P is
    %           the number of features per sample.
    %   - labels: Cell array containing labels corresponding to each class
    %             in the data matrix.
    %   - n: Number of simulations to run.
    %
    % Example:
    %   KNNnTimes(data, labels, 10);
    %
    % Note: 
    %   - This function uses the KNNClassificationV2 function to perform
    %     KNN classification and computes the average test accuracy,
    %     validation accuracy, execution time, and per-class test accuracy
    %     over multiple simulations.
    %
    % Author: Nischal Chandur
    % Date: 05/15/2024
    
    % Initialize arrays to store results
    testAcc = zeros(1, n);
    validAcc = zeros(1, n);
    time = zeros(1, n);
    perClassTestAcc = zeros(size(data, 1), n);

    % Split the data into training, testing, and validation sets
    [train, test] = randomTrainTestSplit(data, 400);
    [test, valid] = randomTrainTestSplit(test, 200);

    % Run KNN classification multiple times
    for i = 1:n
        [ta, va, t, perClass] = KNNClassificationV2(train, test, valid, 16);
        testAcc(i) = ta;
        validAcc(i) = va;
        time(i) = t;
        perClassTestAcc(:, i) = perClass;
    end

    % Display results after multiple simulations
    fprintf('Results After %d simulations\n', n);
    fprintf('Test Accuracy: %f\nValid Accuracy: %f\nTime Taken: %f ms\n', mean(testAcc), mean(validAcc), mean(time) * 1000);
    
    % Compute and display average per-class test accuracy
    averagePerClass = mean(perClassTestAcc, 2);
    for i = 1:size(data, 1)
        fprintf('Class %s: %.2f\n', labels{i}, averagePerClass(i));
    end
end
