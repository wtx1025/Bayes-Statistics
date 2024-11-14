%% MCMC for normal-normal
% Data
n = 1000;
mu_true = 3; 
sigma_true = 3;
sigma2_true = sigma_true^2;
rng(101);  % Set random seed
y = normrnd(mu_true, sigma_true, [n, 1]);

% Prior Hyperparameters
m0 = 0; 
t0 = 1;
a = 2; 
b = 1;

% Initial value
tau = 1;

% Store Results
nsim = 5000;
Tau = zeros(nsim, 1);
Mu = zeros(nsim, 1);

% Gibbs Sampling
mtau = a + n / 2;

for i = 1:nsim
    % Update mu
    v = 1 / (tau * n + t0);
    m = v * (tau * sum(y) + t0 * m0);
    mu = normrnd(m, sqrt(v));
    Mu(i) = mu;
    
    % Update tau (precision)
    tau = gamrnd(mtau, 1 / (b + sum((y - mu).^2) / 2));
    Tau(i) = tau;
end

% Discard the first 20% of samples as burn-in
burn_in = floor(0.2 * nsim);
Mu_post_burn_in = Mu(burn_in + 1:end);
Tau_post_burn_in = Tau(burn_in + 1:end);

% Posterior Mean and 95% Credible Intervals
mu_estimate = mean(Mu_post_burn_in);
tau_estimate = mean(Tau_post_burn_in);
mu_cred_int = quantile(Mu_post_burn_in, [0.025, 0.975]);
tau_cred_int = quantile(Tau_post_burn_in, [0.025, 0.975]);

% Display the final estimates
fprintf('Posterior Mean of Mu: %.4f\n', mu_estimate);
fprintf('95%% Credible Interval for Mu: [%.4f, %.4f]\n', mu_cred_int(1), mu_cred_int(2));
fprintf('Posterior Mean of Tau: %.4f\n', tau_estimate);
fprintf('95%% Credible Interval for Tau: [%.4f, %.4f]\n', tau_cred_int(1), tau_cred_int(2));

% Plot results
figure;

subplot(2, 1, 1);
plot(Mu);
title('Trace of Mu');
xlabel('Iteration');
ylabel('Mu');

subplot(2, 1, 2);
plot(Tau);
title('Trace of Tau');
xlabel('Iteration');
ylabel('Tau');

% Display posterior distributions
figure;

subplot(2, 1, 1);
histogram(Mu_post_burn_in, 30, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
title('Posterior Distribution of Mu');
xlabel('Mu');
ylabel('Density');

subplot(2, 1, 2);
histogram(Tau_post_burn_in, 30, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
title('Posterior Distribution of Tau');
xlabel('Tau');
ylabel('Density');

