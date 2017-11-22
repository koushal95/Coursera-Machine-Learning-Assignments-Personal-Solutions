function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
z2 = zeros(hidden_layer_size, 1);
a2 = zeros(hidden_layer_size, 1);
eX = [ones(m, 1) X];
eXt = eX';
for i = 1:m
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
    z2 = Theta1 * eXt(:, i);
    a2 = sigmoid(z2);
    ea2 = [1; a2];
    z3 = Theta2 * ea2;
    a3 = sigmoid(z3);
    
    % mapping y to vector
    y_i = zeros(num_labels, 1);
    y_i(y(i)) = 1;
    
    % calculate cost
    J = J + sum(-1 * ((y_i .* log(a3)) + ((1-y_i).*log(1-a3))) );
end
J = J / m;

% cost with regularization term
Theta1sq = Theta1 .^2;
Theta2sq = Theta2 .^2;
J = J + ((lambda / (2 * m)) * ((sum(sum(Theta1sq(:,2:input_layer_size + 1)))) + (sum(sum(Theta2sq(:,2:hidden_layer_size+1))))) ); 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));
for t = 1:m
    z2 = Theta1 * eXt(:, t);
    a2 = sigmoid(z2);
    ea2 = [1; a2];
    z3 = Theta2 * ea2;
    a3 = sigmoid(z3);
    y_i = zeros(num_labels, 1);
    y_i(y(t)) = 1;

    delta3 = a3 - y_i;
    delta2 = (Theta2')*(delta3) .* (sigmoidGradient([1; z2]));
    delta_accum_1 = delta_accum_1 + (delta2(2:end) * eXt(:,t)');
    delta_accum_2 = delta_accum_2 + (delta3 * ea2');
end

% unregularized gradient
Theta1_grad = (1/m) * (delta_accum_1);
Theta2_grad = (1/m) * (delta_accum_2);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
