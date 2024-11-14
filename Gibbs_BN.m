%% Using Gibbs Sampling on Bivariate Normal Distribution 
% setting parameters 
mu1 = 0;
mu2 = 0;
sig1 = 1;
sig2 = 1;
rho = 0.95;
s = sig1 * sqrt(1-rho^2);

% iteration times & random seed
n_iter = 10000;
sims = NaN(n_iter*2, 2);
rng(1);

% Gibbs sampling algorithm 
x2 = 0; 
for t = 1:n_iter

    x1_mean = mu1 + rho * (x2 - mu2);
    x1 = normrnd(x1_mean, s);
    sims(2*t-1, :) = [x1, x2];

    x2_mean = mu2 + rho * (x1 - mu1);
    x2 = normrnd(x2_mean, s); 
    sims(2*t, :) = [x1, x2];
end 

% plot the result 
plot(sims(:,1), sims(:,2), '-o');
xlabel('x1');
ylabel('x2');
grid; 
title('Gibbs Sampling of Bivariate Normal Distribution'); 


