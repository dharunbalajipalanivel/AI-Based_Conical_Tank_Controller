% Sample data (replace with your actual data)
X_train = [6; 12; 18; 24; 30; 36]; % Setpoints
Y_train = [12.649 1060.811; 6.3246 530.4055; 17.8885 6000.85; 8.9442 3000.43; 21.9089 16536.4; 10.9545 8268.203]; % Kp and Ki values

% Normalize the training data
X_train_normalized = (X_train - mean(X_train)) / std(X_train);
Y_train_normalized = (Y_train - mean(Y_train)) ./ std(Y_train);

% Define neural network architecture
input_size = size(X_train_normalized, 2); % Number of input features
hidden_size = 20; % Increased number of neurons in the hidden layer
output_size = size(Y_train_normalized, 2); % Number of output neurons

% Initialize weights and biases randomly
W1 = randn(input_size, hidden_size);
b1 = randn(1, hidden_size);
W2 = randn(hidden_size, output_size);
b2 = randn(1, output_size);

% Define hyperparameters
learning_rate = 0.01;
epochs = 5000; % Increased number of epochs

% Sigmoid activation function
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Training loop using backpropagation
for epoch = 1:epochs
    % Forward propagation
    z1 = X_train_normalized * W1 + b1;
    a1 = sigmoid(z1);
    z2 = a1 * W2 + b2;
    a2 = z2; % Output layer activation (no activation function for output)

    % Backpropagation
    delta2 = a2 - Y_train_normalized;
    delta1 = (delta2 * W2') .* (a1 .* (1 - a1));

    % Gradient descent
    dW2 = a1' * delta2;
    db2 = sum(delta2);
    dW1 = X_train_normalized' * delta1;
    db1 = sum(delta1);

    % Update weights and biases
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
end

% Ask user for setpoint input
user_setpoint = input('Enter the setpoint (in centimeters): ');

% Normalize user input
user_setpoint_normalized = (user_setpoint - mean(X_train)) / std(X_train);

% Use trained neural network to predict Kp and Ki for the user-provided setpoint
z1 = user_setpoint_normalized * W1 + b1;
a1 = sigmoid(z1);
z2 = a1 * W2 + b2;
predicted_KpKi = z2; % Output layer activation (no activation function for output)

% Denormalize predicted Kp and Ki values
predicted_KpKi_denormalized = predicted_KpKi .* std(Y_train) + mean(Y_train);

% Ensure Ki remains positive
if predicted_KpKi_denormalized(2) < 0
    predicted_KpKi_denormalized(2) = abs(predicted_KpKi_denormalized(2)); % Take absolute value
end

% Round off the predicted Kp and Ki values to 3 digits
predicted_KpKi_denormalized_rounded = round(predicted_KpKi_denormalized, 4);

% Display the predicted Kp and Ki values
disp(['For the setpoint of ', num2str(user_setpoint), ' cm,']);
disp(['Predicted Kp: ', num2str(predicted_KpKi_denormalized_rounded(1)), ', Predicted Ki: ', num2str(predicted_KpKi_denormalized_rounded(2))]);