% Set parameters and generate synthetic data
n = 100;                   % Sample size
mu_true = 3;               % True mean
tau_true = 0.5;            % True precision (1/variance)
y = normrnd(mu_true, sqrt(1 / tau_true), n, 1);  % Generate observed data

% Prior hyperparameters
mu0 = 0; 
tau0 = 1;
a = 2; 
b = 1;

% Define log-posterior function
log_posterior = @(params) -((n / 2) * log(params(2)) ...
                  - (params(2) / 2) * sum((y - params(1)).^2) ...
                  - (tau0 / 2) * (params(1) - mu0)^2 ...
                  + (a - 1) * log(params(2)) - b * params(2));

% Initial guess for optimization
initial_guess = [0, 1];  % Starting values for [mu, tau]

% Use fminunc to find the MAP estimates
options = optimset('Display', 'iter', 'TolX', 1e-6, 'TolFun', 1e-6);
[theta_map, fval, exitflag, output, grad, hessian_matrix] = fminunc(log_posterior, initial_guess, options);

% Estimated parameters
mu_est = theta_map(1);
tau_est = theta_map(2);
fprintf('Estimated mu: %.4f\n', mu_est);
fprintf('Estimated tau: %.4f\n', tau_est);

% Compute covariance matrix from Hessian (negative Hessian is approximated covariance)
cov_matrix = inv(hessian_matrix);

% Standard deviations for mu and tau
std_devs = sqrt(diag(cov_matrix));
fprintf('Standard deviation for mu: %.4f\n', std_devs(1));
fprintf('Standard deviation for tau: %.4f\n', std_devs(2));

% Display results
disp('Hessian Matrix:');
disp(hessian_matrix);
disp('Covariance Matrix:');
disp(cov_matrix);

% Visualize posterior distribution using normal approximation
theta_samples = mvnrnd(theta_map, cov_matrix, 1000);
figure;
subplot(1, 2, 1);
histogram(theta_samples(:, 1), 30, 'Normalization', 'pdf');
title('\mu (Normal Approximation)');
xlabel('\mu');
ylabel('Density');

subplot(1, 2, 2);
histogram(theta_samples(:, 2), 30, 'Normalization', 'pdf');
title('\tau (Normal Approximation)');
xlabel('\tau');
ylabel('Density');
