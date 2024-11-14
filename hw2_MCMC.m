%% function for MCMC algorithm 
function [x, acc_rate] = metropolis_hastings(initial, iteration)
    
    % setting parameters 
    rng(1);
    x_old = initial;
    acc = 0;  
    x = x_old;
    
    % MCMC algorithm 
    for i = 1:iteration 
        y = sample_from_q();
        u = rand;
        
        % compute alpha
        alpha = min((x_old^2) * (2^y) / ((y^2) * (2^x_old)), 1);
        
        % accept or reject 
        if alpha > u
            x_new = y;
            acc = acc + 1;
        else
            x_new = x_old;
        end
        
        % update 
        x = [x; x_new];
        x_old = x_new;
    end

    % calculate acceptance rate 
    acc_rate = acc / iteration;
end

%% function for sampling from our proposal distribution 
function y = sample_from_q()
    u = rand;  
    y = ceil(log2(1 / (1 - u))); 
end

%% simulaion results 
[x, acc_rate] = metropolis_hastings(1, 10000);

figure;
plot(x, 'LineWidth', 1);
title('Markov Chain Monte Carlo');
xlabel('Iteration');
ylabel('Value');
grid on;

[counts, values] = histcounts(x, 'BinMethod', 'integers');
values = values(1:end-1); 
total_count = numel(x);
probabilities = counts / total_count;

fprintf('Value\tTimes\tProb.\n');
for i = 1:length(values)
    fprintf('%d\t%d\t%.4f\n', values(i), counts(i), probabilities(i));
end