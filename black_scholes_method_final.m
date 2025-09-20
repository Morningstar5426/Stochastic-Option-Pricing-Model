rng(123) % Specific random simulation for data comparison 

%% Nvidia options
files = {'nvidia_options.csv'};
expirations = [13]; % Expiration times in days
S0 = 135.4; % Initial stock price

%% Amazon options
files = {'Amazon/amazon_options_4.csv', 'Amazon/amazon_options_2.csv', 'Amazon/amazon_options_3.csv'};
expirations = [13, 33, 111]; % Expiration times in days
S0 = 205.01; % Initial stock price

%% Netflix options
files = {'netflix_options.csv'};
expirations = [10]; % Expiration times in days
S0 = 1217.94; % Initial stock price

%% Rest of code for calculations and plotting

% Cell arrays
K_values_all = {}; % Strike prices
call_prices_all = {}; % Calculated call prices
put_prices_all = {}; % Calculated put prices
real_call_prices_all = {}; % Real call prices
real_put_prices_all = {}; % Real put prices

% Initial parameters
M = 50000; % Monte Carlo paths per simulation
num_steps = 50; % Time steps for CIR model
dt = 1/num_steps; % Step size

% CIR parameters for interest rate
r0 = 0.045; % Initial interest rate (4.5%)
kappa = 0.5; % Mean reversion of interest rate
theta = 0.03; % Long-term mean interest rate
sigma_r = 0.02; % Volatility of interest rate changes

% Loops through each file containing options
for i = 1:length(files)
    try
        % Checks if the file exists
        if ~isfile(files{i})
            warning('File %s not found', files{i});
            continue; % Skips file if missing
        end

        data = readtable(files{i}, 'VariableNamingRule', 'preserve'); % Reads the option data from the csv file

        % Extracts the call and put options from the dataset
        call_data = data(strcmp(data.Type, 'Call'), :); 
        put_data = data(strcmp(data.Type, 'Put'), :);

        % Checks for missing or empty data
        if isempty(call_data) || isempty(put_data)
            warning('Data for expiration %d is missing or corrupted.', expirations(i));
            continue;
        end

        % Extracts the strike prices, implied volatilities, and mid-prices from the data
        K_values_all{i} = str2double(erase(string(call_data.Strike), ','));
        IV_call = str2double(erase(string(call_data.IV), '%')) / 100;
        IV_put = str2double(erase(string(put_data.IV), '%')) / 100;
        real_call_prices_all{i} = str2double(erase(string(call_data.Mid), ','));
        real_put_prices_all{i}  = str2double(erase(string(put_data.Mid), ','));

        T_exp = expirations(i) / 365; % Converts the time until expiration into years

        % Initialises arrays to store the Monte Carlo call and put prices
        call_prices_MC = zeros(length(K_values_all{i}), 1);
        put_prices_MC = zeros(length(K_values_all{i}), 1);

        % Monte Carlo interest rate simulation using the CIR model
        r_t = zeros(M, num_steps); % Initialises the interest paths
        r_t(:,1) = r0; % Sets the initial interest rate for all paths

        % Loops over the time steps 
        for t = 2:num_steps
            dW_r = randn(M,1); % Simulates the standard brownian motion increments
            r_t(:,t) = max(0, r_t(:,t-1) + kappa*(theta - r_t(:,t-1))*dt + sigma_r*sqrt(r_t(:,t-1)).*dW_r); % Formula for CIR
        end
        r_final = r_t(:,end); % Stores an array

        % Loops over each strike price to compute the call and put prices
        for k = 1:length(K_values_all{i})
            K = K_values_all{i}(k); % Current strike price
            sigma_c = IV_call(k); % Call implied volatility
            sigma_p = IV_put(k); % Put implied volatility

            % Calculates the d1 and d2 for call options
            d1_call = (log(S0 ./ K) + (r_final + 0.5 * sigma_c^2) * T_exp) ./ (sigma_c * sqrt(T_exp));
            d2_call = d1_call - sigma_c * sqrt(T_exp);

            % Calculates the d1 and d2 for put options
            d1_put = (log(S0 ./ K) + (r_final + 0.5 * sigma_p^2) * T_exp) ./ (sigma_p * sqrt(T_exp));
            d2_put = d1_put - sigma_p * sqrt(T_exp);

            % Calculates the Black-Scholes pricing for calls and puts along each path
            call_prices_path = S0 .* normcdf(d1_call) - K .* exp(-r_final * T_exp) .* normcdf(d2_call);
            put_prices_path  = K .* exp(-r_final * T_exp) .* normcdf(-d2_put) - S0 .* normcdf(-d1_put);

            % Adds some random noise
            call_prices_path = call_prices_path .* (1 + 0.02 * (2 * rand(M,1) - 1));  
            put_prices_path  = put_prices_path  .* (1 + 0.02 * (2 * rand(M,1) - 1));

            % Ensures there are no negative option prices
            call_prices_path = max(call_prices_path, 0);
            put_prices_path  = max(put_prices_path, 0);

            % Averages the Monte Carlo paths for final pricing estimates
            call_prices_MC(k) = mean(call_prices_path);
            put_prices_MC(k)  = mean(put_prices_path);
        end

        % Stores the Monte Carlo pricing for current expiration in cell arrays
        call_prices_all{i} = call_prices_MC;
        put_prices_all{i} = put_prices_MC;
    
    % Catches any errors when processing files
    catch ME
        warning('Error processing file %s: %s', files{i}, ME.message);
        continue;
    end
end

% Initialises storage of error metrics for each expiration
error_metrics = struct();

% Loops over all datasets
for i = 1:length(files)
    % Skips error calculations if there is missing data
    if isempty(call_prices_all{i}) || isempty(real_call_prices_all{i}) || isempty(put_prices_all{i}) || isempty(real_put_prices_all{i})
        warning('Skipping error calculation due to missing expiration data %d.', expirations(i));
        continue;
    end

    % Mean Squared Error (MSE) for calls and puts
    error_metrics(i).MSE_call = mean((call_prices_all{i} - real_call_prices_all{i}).^2);
    error_metrics(i).MSE_put = mean((put_prices_all{i} - real_put_prices_all{i}).^2);

    % Root Mean Squared Error (RMSE) for calls and puts
    error_metrics(i).RMSE_call = sqrt(error_metrics(i).MSE_call);
    error_metrics(i).RMSE_put = sqrt(error_metrics(i).MSE_put);

    % Mean Absolute Error (MAE) for calls and puts
    error_metrics(i).MAE_call = mean(abs(call_prices_all{i} - real_call_prices_all{i}));
    error_metrics(i).MAE_put = mean(abs(put_prices_all{i} - real_put_prices_all{i}));

    % Mean Absolute Percentage Error (MAPE) for calls and puts
    error_metrics(i).MAPE_call = mean(abs((call_prices_all{i} - real_call_prices_all{i}) ./ real_call_prices_all{i})) * 100;
    error_metrics(i).MAPE_put = mean(abs((put_prices_all{i} - real_put_prices_all{i}) ./ real_put_prices_all{i})) * 100;

    % R^2 for calls and puts
    error_metrics(i).R2_call = 1 - (error_metrics(i).MSE_call / var(real_call_prices_all{i}));
    error_metrics(i).R2_put = 1 - (error_metrics(i).MSE_put / var(real_put_prices_all{i}));
end

% Display error metrics grouped by expiration
fprintf('\nError in Pricing Model vs. Real Market Prices:\n');

% Display error metrics within a table for each dataset
for i = 1:length(files)
    fprintf('\n%d Days:\n', expirations(i));


    disp(table(["Call"; "Put"], [error_metrics(i).MSE_call; error_metrics(i).MSE_put], [error_metrics(i).RMSE_call;
        error_metrics(i).RMSE_put], [error_metrics(i).MAE_call; error_metrics(i).MAE_put], ...
        [error_metrics(i).MAPE_call; error_metrics(i).MAPE_put], ...
        [error_metrics(i).R2_call; error_metrics(i).R2_put], ...
        'VariableNames', {'OptionType', 'MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R2'}));
end

% Displays the pricing comparison in a table for each expiration
for i = 1:length(files)
    fprintf('\nMonte Carlo Black-Scholes Pricing with CIR Interest Rates for Expiration %d Days:\n', expirations(i));
    disp(table(K_values_all{i}, call_prices_all{i}, real_call_prices_all{i},put_prices_all{i}, real_put_prices_all{i}, ...
    'VariableNames', {'StrikePrice', 'MonteCarlo_CallPrice', 'Real_CallPrice', 'MonteCarlo_PutPrice', 'Real_PutPrice'}));
end



figure;

% Calculates the percentage residuals for calls and puts
call_residual = ((cell2mat(real_call_prices_all) - cell2mat(call_prices_all)) ./ cell2mat(real_call_prices_all)) * 100;
put_residual  = ((cell2mat(real_put_prices_all) - cell2mat(put_prices_all)) ./ cell2mat(real_put_prices_all)) * 100;

% Plots the percentage residuals for pcalls
subplot(2,1,1);
hold on;
for i = 1:length(files)
    plot(K_values_all{i}, call_residual(:,i), '-o', 'LineWidth', 1.5, 'DisplayName', sprintf('Exp %d Days', expirations(i)));
end
xlabel('Strike Price');
ylabel('Residual Error (%)');
title('Percentage Residuals for Call Pricing');
legend('Location','best');
grid on;
hold off;

% Plots the percentage residuals for puts
subplot(2,1,2);
hold on;
for i = 1:length(files)
    plot(K_values_all{i}, put_residual(:,i), '-s', 'LineWidth', 1.5, 'DisplayName', sprintf('Exp %d Days', expirations(i)));
end
xlabel('Strike Price');
ylabel('Residual Error (%)');
title('Percentage Residuals for Put Pricing');
legend('Location','best');
grid on;
hold off;


% Plots the Black-Scholes vs real option prices across expirations
figure;
for i = 1:length(expirations)
    subplot(2, length(expirations), i)
    plot(K_values_all{i}, call_prices_all{i}, 'b-o', 'DisplayName', 'Monte Carlo Call');
    hold on;
    plot(K_values_all{i}, real_call_prices_all{i}, 'r--s', 'DisplayName', 'Real Call');
    title(['Call Prices (', num2str(expirations(i)), ' days)']);
    xlabel('Strike Price');
    ylabel('Option Price');
    legend('Location', 'best');
    grid on;

    subplot(2, length(expirations), i + length(expirations))
    plot(K_values_all{i}, put_prices_all{i}, 'b-o', 'DisplayName', 'Monte Carlo Put');
    hold on;
    plot(K_values_all{i}, real_put_prices_all{i}, 'r--s', 'DisplayName', 'Real Put');
    title(['Put Prices (', num2str(expirations(i)), ' days)']);
    xlabel('Strike Price');
    ylabel('Option Price');
    legend('Location', 'best');
    grid on;
end

% Adds a title to the entire figure
sgtitle('Black-Scholes vs Real Option Prices Across Expiries');
