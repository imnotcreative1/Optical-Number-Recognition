clear all;
clc;
il=2;
hl=2;
nl=4;
nn=[1:18]/2;
X=cos([1 2; 3 4;5 6]);
y = [4;2;3];
lambda = 4;
Theta1 = reshape(nn(1:hl * (il + 1)), ...
                 hl, (il + 1));

Theta2 = reshape(nn((1 + (hl * (il + 1))):end), ...
                 nl, (hl + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
y_matrix = eye(nl);
y_matrix = y_matrix(y,:);
a2 = sigmoid([ones(size(X,1),1) X]*Theta1');
a3 = sigmoid([ones(size(a2,1),1) a2]*Theta2');
Theta1(1) = 0;
Theta2(1) = 0;
J = (1/m) * sum(sum((-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3))));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
y_matrix = eye(nl);
y_matrix = y_matrix(y,:);
X = [ones(size(X,1),1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2*Theta2');
J = (1/m) * sum(sum((-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3))));
J = J + lambda/2/m * (sum(sum(Theta1(:,2:end).*Theta1(:,2:end)))+ sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));
%Back propagation
a1 = X;
d3 = a3 - y_matrix;
d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);
Delta1 = d2' * a1;
Delta2 = d3' * a2;
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;
