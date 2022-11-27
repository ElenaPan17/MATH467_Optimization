function [weight_m, cost_m, i] = Fixed_Step(x, y)

% create network
[network]=createNetwork(2,[3,3,1]);

% use networkbProp to generate yGrad and calculate gradient vector
[yVal,yintVal]=networkFProp(x,network);
yGrad=networkBProp(network,yintVal);
yGrad = squeeze(yGrad);
gradient = -2.*yGrad*(y-yVal)';

% generate weight
weight = getNNWeight(network);

% weight = weight - a*gradient;
% network = setNNWeight(network,weight);
% yVal=networkFProp(x,network);

% cost function
cost = sum((y - yVal).^2);

% create weight and cost matrix
weight_m = [weight];
cost_m = [cost];
i = 0;

% set fixed step-size
min_a = 0.0001;

while (i < 1) || ((cost_m(length(cost_m)-1) - cost_m(length(cost_m))) > 10^(-3))

    % update weight
    weight = weight - min_a.*gradient;
    weight_m = [weight_m, weight];

    % update network
    network = setNNWeight(network, weight);
    [yVal,yintVal] = networkFProp(x,network);
    cost = sum((y - yVal).^2);
    cost_m = [cost_m, cost];

    % update gradient
    yGrad=networkBProp(network,yintVal);
    yGrad = squeeze(yGrad);
    gradient = -2.*yGrad*(y-yVal)';

    i = i + 1;

end

end