% Image file names for apples and pears (Training set)
appleImages = {'apple_04.jpg', 'apple_05.jpg', 'apple_06.jpg'};
pearImages = {'pear_01.jpg', 'pear_02.jpg'};

% Initialize feature vectors for color and roundness
x1 = zeros(1, 5); % Color feature
x2 = zeros(1, 5); % Roundness feature

% Process apple images (3 apples)
for i = 1:3
    img = imread(appleImages{i});
    x1(i) = spalva_color(img);    % Extract color feature
    x2(i) = apvalumas_roundness(img);  % Extract roundness feature
end

% Process pear images (2 pears)
for i = 1:2
    img = imread(pearImages{i});
    x1(3 + i) = spalva_color(img);    % Extract color feature
    x2(3 + i) = apvalumas_roundness(img);  % Extract roundness feature
end

% Estimated features are stored in matrix P
P = [x1; x2];

% Desired output vector (1 for apples, -1 for pears)
T = [1; 1; 1; -1; -1];

%% Train single perceptron with two inputs and one output

% Generate random initial values of w1, w2, and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% Learning rate
eta = 0.1;

% Initialize total error to a large value
e = 1;

% Training loop - continue until the total error is 0
while e ~= 0
    % Initialize total error to 0 for each iteration
    total_error = 0;
    
    % Loop through each input example
    for i = 1:5
        % Calculate the weighted sum (v) for current example
        v = w1 * x1(i) + w2 * x2(i) + b;
        
        % Calculate the perceptron output (y)
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % Calculate the error (desired output - actual output)
        e = T(i) - y;
        
        % Update the weights and bias if there is an error
        w1 = w1 + eta * e * x1(i); % Update w1
        w2 = w2 + eta * e * x2(i); % Update w2
        b = b + eta * e;           % Update bias
        
        % Accumulate total error
        total_error = total_error + abs(e);
    end
    
    % Update the total error for stopping condition
    e = total_error;
end

% Output the final weights and bias after training
fprintf('Final weights and bias after perceptron training:\n');
fprintf('w1: %.4f\n', w1);
fprintf('w2: %.4f\n', w2);
fprintf('b: %.4f\n', b);

%% Plotting the data points and decision boundaries

% Create figure for plotting
figure;
hold on;

% Plot training data points
for i = 1:3
    plot(x1(i), x2(i), 'ro', 'MarkerSize', 10, 'DisplayName', 'Apple'); % Red circles for apples
end
for i = 4:5
    plot(x1(i), x2(i), 'go', 'MarkerSize', 10, 'DisplayName', 'Pear');  % Green circles for pears
end

% Labels and title
xlabel('Color Feature (x1)');
ylabel('Roundness Feature (x2)');
title('Perceptron and Naive Bayes Classifiers');

%% Plot Perceptron decision boundary
% The decision boundary equation is: w1 * x1 + w2 * x2 + b = 0
x_values = linspace(min(x1)-1, max(x1)+1, 100); % Generate 100 x-values for the plot
y_values = -(w1 * x_values + b) / w2; % Solve for x2 (y-values)

plot(x_values, y_values, 'b-', 'LineWidth', 2, 'DisplayName', 'Perceptron Decision Boundary');

%% Naive Bayes Classifier

% Step 1: Calculate Priors
numApples = sum(T == 1);
numPears = sum(T == -1);
numTotal = length(T);

% Prior probabilities
P_apple = numApples / numTotal;
P_pear = numPears / numTotal;

% Step 2: Calculate Likelihoods
mean_color_apple = mean(x1(1:3)); % Mean of color for apples
std_color_apple = std(x1(1:3));   % Std dev of color for apples

mean_color_pear = mean(x1(4:5));  % Mean of color for pears
std_color_pear = std(x1(4:5));    % Std dev of color for pears

mean_roundness_apple = mean(x2(1:3)); % Mean of roundness for apples
std_roundness_apple = std(x2(1:3));   % Std dev of roundness for apples

mean_roundness_pear = mean(x2(4:5));  % Mean of roundness for pears
std_roundness_pear = std(x2(4:5));    % Std dev of roundness for pears

% Step 3: Test with the same new images using Naive Bayes
for i = 1:length(x1)
    % Calculate likelihoods for the new image
    % Likelihood for apple class
    L_color_apple = gaussian_likelihood(x1(i), mean_color_apple, std_color_apple);
    L_roundness_apple = gaussian_likelihood(x2(i), mean_roundness_apple, std_roundness_apple);
    P_apple_given_features = P_apple * L_color_apple * L_roundness_apple;
    
    % Likelihood for pear class
    L_color_pear = gaussian_likelihood(x1(i), mean_color_pear, std_color_pear);
    L_roundness_pear = gaussian_likelihood(x2(i), mean_roundness_pear, std_roundness_pear);
    P_pear_given_features = P_pear * L_color_pear * L_roundness_pear;
    
    % Classify the image based on the higher posterior probability
    if P_apple_given_features > P_pear_given_features
        plot(x1(i), x2(i), 'r+', 'MarkerSize', 10, 'DisplayName', 'Classified as Apple'); % Red cross for apple classification
    else
        plot(x1(i), x2(i), 'g+', 'MarkerSize', 10, 'DisplayName', 'Classified as Pear');  % Green cross for pear classification
    end
end

legend('show');
hold off;

% Gaussian likelihood function
function likelihood = gaussian_likelihood(x, mean, std)
    likelihood = (1 / (std * sqrt(2 * pi))) * exp(-(x - mean)^2 / (2 * std^2));
end
