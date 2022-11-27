function [weight_m, cost_m, i] = Steepest_Descent(x, y)

% create the network
[network]=createNetwork(2,[3,3,1]);

% use networkFProp to generate yVal and calculate gradient
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

while (i < 1) || ((cost_m(length(cost_m)-1) - cost_m(length(cost_m))) > 10^(-3))

    % use fminsearch to find min_a
    func = @(a) sum((y - networkFProp(x,setNNWeight(network,(weight - a.*gradient)))).^2);
    min_a = fminsearch(func, 0.01);
    disp(min_a)

    % update weight
    weight = weight - min_a.*gradient;
    weight_m = [weight_m, weight];

    % update network
    network = setNNWeight(network, weight);
    [yVal,yintVal] = networkFProp(x,network);

    % calculate cost function with updated yVal
    cost = sum((y - yVal).^2);
    cost_m = [cost_m, cost];

    % update gradient
    yGrad=networkBProp(network,yintVal);
    yGrad = squeeze(yGrad);
    gradient = -2.*yGrad*(y-yVal)';

    i = i + 1;

end

end