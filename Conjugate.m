function [weight_m, cost_m, i] = Conjugate(x, y)

% create network
[network]=createNetwork(2,[3,3,1]);

% use networkBProp to generate yGrad and calculate gradient
[yVal,yintVal]=networkFProp(x,network);
yGrad=networkBProp(network,yintVal);
yGrad = squeeze(yGrad);
gradient = -2.*yGrad*(y-yVal)';

% d0 = g0
direction = gradient;

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

while (i < 1) || ((cost_m(length(cost_m)-1) - cost_m(length(cost_m))) > 10^(-3))
    
    % func: gradient --> direction
    func = @(a) sum((y - networkFProp(x,setNNWeight(network,(weight - a.* direction)))).^2);
    min_a = fminsearch(func, 0.01);

    % update weight
    weight = weight - min_a.* direction;
    weight_m = [weight_m, weight];

    % update network
    network = setNNWeight(network, weight);
    [yVal,yintVal] = networkFProp(x,network);
    yGrad=networkBProp(network,yintVal);
    yGrad = squeeze(yGrad);
    gradient = -2.*yGrad*(y-yVal)';

    % generate gradient_k which is g(k+1)
    gradient_k = -2.*yGrad*(y-yVal)';

    cost = sum((y - yVal).^2);
    cost_m = [cost_m, cost];

    % Use Fletcher-Reeves Formula to generate beta
    beta = (gradient_k'*gradient_k)/(gradient'*gradient);

    % d(k+1)
    direction_k = gradient_k - beta * direction;
    
    i = i + 1;

end

end







