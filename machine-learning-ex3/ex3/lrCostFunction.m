function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 

% computing the cost without regularization 

% defining the hypothesis
h = sigmoid(X * theta);

% computing the cost
J = (-1 / m) * (sum(y.*log(h) + (1-y).*log(1-h)));

%computing the gradient
grad = (1 / m) * X'*(h-y);
   
% with regularization

% computing the cost
J = ((-1 / m) * ( y'*log(h) + (1-y')*log(1-h) )) + ((lambda/(2*m)) * (sum(theta(2:size(theta,1),1).^2)));

temp = theta; 
temp(1) = 0;  
grad = ((1/m) * (X'*(h-y))) + (lambda/m) * temp;

% =============================================================

grad = grad(:);

end
