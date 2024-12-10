% Set random seed for reproducibility
rng(1);

%% Step 1: Prepare Input Data
x = 0.1:1/22:1;         % Input vectors (20 examples) in the range 0 to 1
X = x';                 % Transpose to get column vector

% Desired output based on the given formula
Y = ((1 + 0.6 * sin(2 * pi * X / 0.7)) + 0.3 * sin(2 * pi * X)) / 2;

%% Step 2: Initialize the MLP structure
inputSize = 1;          % One input
hiddenSize = 6;         % Number of hidden neurons (can be adjusted from 4 to 8)
outputSize = 1;         % One output

% Random initialization of weights and biases
W1 = randn(hiddenSize, inputSize);  % Weights between input and hidden layer
b1 = randn(hiddenSize, 1);          % Bias for hidden layer
W2 = randn(outputSize, hiddenSize); % Weights between hidden and output layer
b2 = randn(outputSize, 1);          % Bias for output layer

%% Step 3: Define Hyperparameters
learningRate = 0.01;    % Learning rate for backpropagation
maxEpochs = 10000;      % Maximum number of training iterations
errorThreshold = 1e-4;  % Error threshold for stopping

%% Step 4: Activation Functions and Their Derivatives
sigmoid = @(x) 1 ./ (1 + exp(-x));            % Sigmoid activation function
tanh_activation = @(x) tanh(x);               % Hyperbolic tangent activation function
linear = @(x) x;                              % Linear activation function

% Derivatives for backpropagation
dsigmoid = @(y) y .* (1 - y);                 % Derivative of sigmoid
dtanh_activation = @(y) 1 - y.^2;             % Derivative of tanh

%% Step 5: Training the MLP (Backpropagation)
for epoch = 1:maxEpochs
    totalError = 0;
    
    for i = 1:length(X)
        % Forward pass
        % Input to hidden layer
        z1 = W1 * X(i) + b1;                  % Pre-activation of hidden layer
        a1 = tanh_activation(z1);             % Activation (tanh) in hidden layer
        
        % Hidden layer to output layer
        z2 = W2 * a1 + b2;                    % Pre-activation of output layer
        a2 = linear(z2);                      % Linear activation in output layer
        
        % Error (difference between desired and actual output)
        error = Y(i) - a2;
        totalError = totalError + error^2;
        
        % Backpropagation (Gradient Descent)
        % Output layer gradient
        d2 = -2 * error;                      % Derivative of error w.r.t. linear activation
        
        % Hidden layer gradient
        d1 = (W2' * d2) .* dtanh_activation(a1);  % Backpropagation to hidden layer
        
        % Update weights and biases
        W2 = W2 - learningRate * d2 * a1';    % Update weights between hidden and output layer
        b2 = b2 - learningRate * d2;          % Update bias for output layer
        W1 = W1 - learningRate * d1 * X(i)';  % Update weights between input and hidden layer
        b1 = b1 - learningRate * d1;          % Update bias for hidden layer
    end
    
    % Check stopping condition
    if totalError < errorThreshold
        fprintf('Training complete at epoch %d, total error: %.6f\n', epoch, totalError);
        break;
    end
    
    % Optional: Print progress every 1000 epochs
    if mod(epoch, 1000) == 0
        fprintf('Epoch %d, total error: %.6f\n', epoch, totalError);
    end
end

%% Step 6: Test the trained MLP
% Compare the MLP output with the expected output

% Forward pass for the test data
Z1_test = W1 * x + b1;                       % Pre-activation of hidden layer
A1_test = tanh_activation(Z1_test);          % Activation (tanh) in hidden layer
Z2_test = W2 * A1_test + b2;                 % Pre-activation of output layer
A2_test = linear(Z2_test);                   % Linear activation in output layer

% Plot the expected vs. actual output
figure;
plot(X, Y, 'r-', 'LineWidth', 2);            % Expected output
hold on;
plot(X, A2_test, 'b--', 'LineWidth', 2);     % MLP output
title('MLP Output vs Expected Output');
xlabel('Input (X)');
ylabel('Output (Y)');
legend('Expected Output', 'MLP Output');
grid on;

