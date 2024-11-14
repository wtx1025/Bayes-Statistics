function metropolis_algorithm()
    % Set random seed
    rng(2018); % Set seed for reproducibility

    % Parameters to test
    sigmas = [0.1, 0.5, 10];
    k = 500;  % Number of iterations
    samples = cell(1, length(sigmas));  % To store samples
    acceptance_rates = zeros(1, length(sigmas));  % To store acceptance rates
    
    % Loop over each sigma value
    for s = 1:length(sigmas)
        sigma = sigmas(s);
        [x, acc_rate] = metropolis_algorithm_single(sigma, k);
        samples{s} = x;  % Store the samples
        acceptance_rates(s) = acc_rate;  % Store the acceptance rate
    end

    % Plotting sequences
    figure('Position', [100, 100, 800, 600]);

    % Plot each sequence and ACF
    for i = 1:length(sigmas)
        sigma = sigmas(i);
        
        % Time Series Plot of Samples
        subplot(3, 2, (i - 1) * 2 + 1);
        plot(samples{i});
        title(['Time Series Plot of Samples (sigma=' num2str(sigma) ')']);
        xlabel('Iteration');
        ylabel('Value');
        
        % ACF Plot (Name-Value Syntax for autocorr)
        subplot(3, 2, i * 2);
        autocorr(samples{i}, 'NumLags', 30);  % Use name-value syntax
        title(['ACF Plot (sigma=' num2str(sigma) ')']);
    end

    % Show the acceptance rates
    disp('Acceptance Rates:');
    disp(acceptance_rates);
end

function [x, acc_rate] = metropolis_algorithm_single(sigma, k)
    % Set initial parameters
    x_old = -10;
    acc = 0;
    x = zeros(1, k + 1);  % Store the samples
    x(1) = x_old;

    % Metropolis Algorithm loop
    for i = 2:k + 1
        % Proposal distribution
        y = x_old + sigma * randn(1);  % Using randn to generate normal noise
        
        % Compute acceptance probability
        alpha = exp(-(y^2 - x_old^2) / 2);
        
        % Accept or reject
        u = rand();  % Uniform random number between 0 and 1
        if u < alpha
            x_new = y;
            acc = acc + 1;
        else
            x_new = x_old;
        end
        
        % Store the new sample
        x(i) = x_new;
        x_old = x_new;
    end
    
    % Calculate the acceptance rate
    acc_rate = acc / k;
end
