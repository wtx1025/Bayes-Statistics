% Set random seed 
rng(100);

% Our data
data = struct('X', [14, 29, 6, 25, 18, 4, 18, 12, 22, 6, 30, 11, 30, 5, 20, 13, 9, 32, 24, 13, 19, 4, 28, 22, 8], ...
              'y', [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]);

% Add intercept
X = [ones(length(data.X), 1), data.X'];
y = data.y;

% Initialize parameters
n_iterations = 100000;
theta_init = zeros(size(X, 2), 1);
sigma = 0.1;

% Initialize storage for samples
theta_samples = zeros(n_iterations, length(theta_init));
theta_current = theta_init;
accepted = 0;

% Metropolis-Hastings Algorithm
for i = 1:n_iterations
    % Sample from proposal distribution (Gaussian random walk)
    theta_new = theta_current + normrnd(0, sigma, size(theta_init));  
    
    % Compute the prior probability of the current and new theta
    prior_theta = sum(log(normpdf(theta_current, 0, 5))); 
    prior_theta_new = sum(log(normpdf(theta_new, 0, 5))); 
    
    % Compute the log-likelihood for the current and new theta
    log_likelihood = sum(y' .* (X * theta_current) - log(1 + exp(X * theta_current)));  
    log_likelihood_new = sum(y' .* (X * theta_new) - log(1 + exp(X * theta_new)));
    
    % Compute the posterior probabilities for the current and new theta
    posterior_current = log_likelihood + prior_theta;
    posterior_new = log_likelihood_new + prior_theta_new;
    
    % Compute the acceptance ratio
    alpha = exp(posterior_new - posterior_current);
    
    % Accept or reject the new sample
    if rand < alpha
        theta_current = theta_new;
        accepted = accepted + 1;
    end
    
    % Store the sample
    theta_samples(i, :) = theta_current';
end

% Calculate acceptance rate
acceptance_rate = accepted / n_iterations;

% Compute posterior mean and variance
posterior_mean = mean(theta_samples, 1);
posterior_variance = var(theta_samples, 0, 1);

% Display results
fprintf('Results for sigma = %.1f:\n', sigma);
fprintf('Acceptance Rate: %.4f\n', acceptance_rate);
fprintf('Posterior Mean: %s\n', mat2str(posterior_mean));
fprintf('Posterior Variance: %s\n', mat2str(posterior_variance));

% Plot histograms of the samples
figure('Position', [100, 100, 800, 400]);

for i = 1:size(theta_samples, 2)
    subplot(1, size(theta_samples, 2), i);
    histogram(theta_samples(:, i), 30, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
    title(['Theta ' num2str(i-1) ' with sigma=' num2str(sigma)]);
    xlabel('Value');
    ylabel('Density');
    grid on;
end
