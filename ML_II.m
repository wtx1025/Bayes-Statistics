% define objective function 
function value = marginal_likelihood(params, y)

    mu = params(1);
    tau = params(2); 
    log_marginal_likelihood = 0; 

    for i = 1 : 8
        logpdf = log(normpdf(y{i, 'estimate'}, mu, sqrt(y{i, 'sd'}^2+tau^2)));
        log_marginal_likelihood = log_marginal_likelihood + logpdf; 
    end 

    value = log_marginal_likelihood; 
end 

% read txt file 
data = readtable("C:\Users\王亭烜\Desktop\School\碩二上\Applied Bayes Method\Bayes2024\schoolsdata.txt");
neg_marginal_likelihood = @(params) -marginal_likelihood(params, data); 

% set initial guess and constraint 
x0 = [0, 10];
lb = [-10, 0.001];
ub = [30, 30]; 

% use fmincon to do the optimization 
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
[opt_params, fval] = fmincon(neg_marginal_likelihood, x0, [], [], [], [], lb, ub, [], options);
disp('Optimal parameters (mu, tau):');
disp(opt_params);


