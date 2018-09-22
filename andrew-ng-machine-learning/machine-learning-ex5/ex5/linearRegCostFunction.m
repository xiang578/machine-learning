function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h=zeros(m,1);
h=X*theta;
h=h.-y;
h=h.^2;
J=sum(h(:))/2/m;
tmp=theta;
tmp(1)=0;
J+=lambda*sum(tmp.^2)/2/m;

h=X*theta;
grad=(X')*(h-y)/m;
tmp=theta;
tmp(1)=0;
grad=grad+lambda/m.*tmp;










% =========================================================================

grad = grad(:);

end
