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
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
z2=X*Theta1';
a2=sigmoid(z2);
m2 = size(a2, 1);
a2 = [ones(m2, 1) a2];
z3=a2*Theta2';
h=sigmoid(z3);
[p_max,p] = max(h,[],2);
y_vector=zeros(m,num_labels);
for i=1:length(y)
    y_vector(i,y(i,:))=1;
end
J = -1/m*sum(sum(y_vector.*log(h)+((1-y_vector).*log(1-h))));
theta_reg1=Theta1( : ,2:end);
theta_reg2=Theta2( : ,2:end);
regulazation = lambda/2/m*(sum(sum(theta_reg1.^2))+sum(sum(theta_reg2.^2)));
J=J+regulazation;
Delta2=zeros(hidden_layer_size+1,num_labels);
Delta1=zeros(input_layer_size+1,hidden_layer_size);
for i=1:m
    ai1=X(i,:);
    zi2=ai1*Theta1';
    ai2=sigmoid(zi2);
    ai2 = [1 ai2];
    zi3=ai2*Theta2';
    ai3=sigmoid(zi3);
    di3=ai3'-y_vector(i,:)';
    di2=Theta2(:,2:end)'*di3.*sigmoidGradient(zi2)';
    Delta2=Delta2 + ai2'*di3';
    Delta1 = Delta1 +ai1'*di2';
end
regulize = lambda/m*Theta1;
regulize(:,1)=zeros(size(Theta1,1),1);
Theta1_grad = Delta1'/m+regulize;

regulize = lambda/m*Theta2;
regulize(:,1)=zeros(size(Theta2,1),1);
Theta2_grad = Delta2'/m+regulize;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
