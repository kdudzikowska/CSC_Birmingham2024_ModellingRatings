% Katarzyna Dudzikowska & Matthew Apps
% Birmingham-Leiden Computational Social Cognition (CSC) Summer School 2024
% Computational modelling of continuous ratings: subjective experience of fatigue. 

%% Section I. Load the data

clear all
%--SET PATH TO THE DIRECTORY HERE--%
datadir = '';
%--%

cd(datadir)

pps_files = dir('*.mat'); %Find all matlab files in the directory
n_trials = 150; %Number of trials in the task
n_pps = length(pps_files); %Number of participants

%Extract participant ids
for i_pp = 1:n_pps
    ids(i_pp) = extractBetween(pps_files(i_pp).name, "s", ".");
end

% Initialise arrays for the data
ids = cell(n_pps,0);
population_fatigue = zeros(n_pps,n_trials);
population_initFatigue = zeros(n_pps,1);
population_fatigueChange = zeros(n_pps,n_trials);
population_effort = zeros(n_pps,n_trials);
population_success = zeros(n_pps,n_trials);

% Fill the arrays with data
for i_pp = 1:n_pps
    fprintf('%s\n',pps_files(i_pp).name);    
    ids(i_pp) = extractBetween(pps_files(i_pp).name, "s", "."); 
    load(pps_files(i_pp).name);
    
    %fatigue ratings collected throughout the task (75 out of 150 trials, NaN for the rest)
    population_fatigue(i_pp,:) = x.fatigue(1:end); 
    %baseline fatigue reported at the start
    population_initFatigue(i_pp,1) = x.initial_fatigue(1);
    %change in fatigue rating since the last obtained rating
    population_fatigueChange(i_pp,:) = x.fatigueChange(1:end);
    %effort level required on each trial, quantified as a proportion of Maximum 
    %Voluntary Contraction: 0 for rest trials and 0.30, 0.39 or 0.48
    population_effort(i_pp,:) = x.effort(1:end);
    %trial success: 1 if the goal of sustaining the required effort level
    %was achieved, 0 otherwise, NaN on rest trials
    population_success(i_pp,:) = x.success(1:end);
    %reward obtained on the trial: random number of points given for
    %successfully exerting effort and after rest (0 for a failure)
    population_reward(i_pp,:) = x.reward(1:end); 

end

%% Section II. Visualise the data.

%--ADD THE PLOT OF FATIGUE V TRIAL HERE--%



%--%

% Calculating average fatigue change 
effort_levels = [0 0.3 0.39 0.48];
mean_fatigue_change_per_effort = NaN(length(effort_levels), 1);
err_fatigue_change_per_effort = NaN(length(effort_levels), 1);

for i_e = 1:length(effort_levels)
    e = effort_levels(i_e);
    mean_fatigue_change_per_effort(i_e) = mean(population_fatigueChange(population_effort==e), 'omitnan');
    err_fatigue_change_per_effort(i_e) = std(population_fatigueChange(population_effort==e), 'omitnan')/ sqrt(length(population_fatigueChange(population_effort==e)));

end

%--ADD THE PLOT OF FATIGUE CHANGE V EFFORT LEVEL HERE--%



%--%



%% Section III. Standardise the ratings.

%Raw ratings
population_stand_fatigue = zeros(n_pps,n_trials);
population_stand_initFatigue = zeros(n_pps,1);

for i_pp = 1:n_pps
    
    fatigue_ratings = population_fatigue(i_pp,:)'; %Fatigue values on each trial for this participant

    %Let's not forget baseline ratings!
    fatigue_init = population_initFatigue(i_pp, :); %Starting fatigue value for this participant
    ratings_all = [population_initFatigue(i_pp, :) population_fatigue(i_pp,:)];
    
    %--CALCULATE STANDARDISED RATINGS HERE (as stand_ratings_all) --%
    
    
    %--%
    
    %Split back into initial rating and task ratings
    population_stand_initFatigue(i_pp,1) = stand_ratings_all(1);
    population_stand_fatigue(i_pp,:) = stand_ratings_all(end-n_trials+1:end);

end

%% Section IV. Hypothesised model

% Hypothesised Model (3 params: Unrecoverable Fatigue Rate, Recoverable Fatigue Rate, Recovery Rate) - fatigue_estimate_UfRfRr.m
i_model = 1;
fatigue_func = @(params, E, ratings, baseline) fatigue_estimate_UfRfRr(params, E, ratings, baseline); %model function handle
model_titles{i_model} = 'UfRfRr'; %Name of the model
model_params(i_model) = 3; % Number of parameters in the model

%% Section V. Model fitting

% Section Va. Use fminsearch to estimate best model parameters for each participant.
for i_pp = 1:n_pps %for each participant
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d\n',i_pp, n_pps);
    fprintf('Subject ID: %s\n', id);
    
    % Load observed data for this participant
    E = population_effort(i_pp,:)'; %effort level on each trial
    stand_fatigue_init = population_stand_initFatigue(i_pp,1); %standardised baseline value
    stand_fatigue_ratings = population_stand_fatigue(i_pp,:)'; %standardised task ratings 
        
    %% Select the model function 
    fit = @(params) fatigue_func(params, E, stand_fatigue_ratings, stand_fatigue_init);

    %% Optimise parameter estimates through repeated sum-of-squares function minimization
    nRuns = 50;
    n = 0;
    fit_best = Inf;
    while n <= nRuns
        n = n + 1;
        startp = rand(1,model_params(i_model));
        constrained_fit = @(params) fit(params) + (params(1)<0)*realmax + (params(2)<0)*realmax + (params(3)<0)*realmax;

        % Use fminsearch to find the best solution for the model 
        [parameter_values, fitk] = fminsearch(fit, startp, optimset('MaxFunEvals',100000,'MaxIter',100000));

        % Update best
        if fitk < fit_best
            parameter_estimates_best{i_pp, i_model} = parameter_values;
            fit_best = fitk;
        end

    end

% Section Vb. Fit the model with optimised parameters.
    [RSS_best(i_pp, i_model), fatigue_estimate_best{i_pp, i_model}, Rfat_estimate_best{i_pp, i_model}, Ufat_estimate_best{i_pp, i_model}] = fatigue_func(parameter_estimates_best{i_pp, i_model}, E, stand_fatigue_ratings, stand_fatigue_init);     

% Section Vc. Calculate the model metrics, AIC and BIC, for this model for each participant

    %--DEFINE k, n, and RSS--%

    
    %--%

    %--CALCULATE AIC AND BIC--%

    
    %--%
        
    fprintf('Fitting of %s model completed.\nRSS = %f\nAIC = %f\nBIC = %f\n', char(model_titles(i_model)), RSS_best(i_pp, i_model), AIC(i_pp, i_model), BIC(i_pp, i_model));
end

%% Section VI. Alternative model (with provided model function)

% Recoverable Fatigue Model (2 params: Recoverable Fatigue Rate, Recovery Rate) - fatigue_estimate_RfRr.m

%--Copy paste all the code from sections V and VI and amend to repeat the process for RfRr model--%



%--%

%% Section VII. Alternative model (define your own model function).

% Unrecoverable Fatigue Only Model (1 param: Unrecoverable Fatigue Rate)

%--Once you code the function in fatigue_estimate_Uf.m copy paste the code from sections V and VI and amend to repeat parameter
%the process for the new model--%


%--%

%% Section VIII. Determine the winning model for each participant.

n_models = i_model;

for i_pp = 1:n_pps %for each participant
    
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d\n',i_pp, n_pps);
    fprintf('Subject ID: %s\n', id);
    

    %% AIC
    
    %--find the lowest AIC value for each participant (lowest_AIC)--%
    
    
    %--%
    
    
    %--find the id of the model with lowest AIC (lowestAICid)--%
    

    %--%
    
    best_AIC_value(i_pp) = lowest_AIC;
    best_AIC_model(i_pp)= lowestAICid;
    best_AIC_model_title(i_pp) = model_titles(best_AIC_model(i_pp));
    best_AIC_params{i_pp} = parameter_estimates_best{i_pp, best_AIC_model(i_pp)};
    best_AIC_fatigue_estimates{i_pp, best_AIC_model(i_pp)} = fatigue_estimate_best{i_pp, best_AIC_model(i_pp)};

    fprintf('Model comparison: AIC. %s win (AIC = %f)\n', char(model_titles(best_AIC_model(i_pp))), best_AIC_value(i_pp));
    
    %% BIC
    
    %--Repeat the same for BIC (use the same naming pattern)--%
    
    
    
    %--%
    
end

%% Section IX. Determine the winning model across participants.

%--For each model calculate AIC and BIC sum score for each model across
%participants & and proportion of wins between participants.--%

AIC_sum = zeros(1, n_models); %AIC sum score for each model
BIC_sum = zeros(1, n_models); %BIC sum score for each model

AIC_win_share = zeros(1, n_models);%AIC proportion of wins across participants for each model
BIC_win_share = zeros(1, n_models); %BIC proportion of wins across participants for each model

best_AIC_sum = 0; %AIC overall best sum score
best_AIC_sum_model_title = NaN; %The name of the model with the best overall AIC sum score

best_AIC_share = 0; %Highest proportion of AIC wins
best_AIC_share_model_title = NaN; %The name of the model with the highest proportion of AIC wins

best_BIC_sum = 0; %BIC overall best sum score
best_BIC_sum_model_title = NaN; %The name of the model with the best overall BIC sum score

best_BIC_share = 0; %BIC overall best sum score
best_BIC_share_model_title = NaN; %The name of the model with the highest proportion of BIC wins

%--Find values for the variables (also arrays) initialised with 0s and NaNs above--%




%--% 

fprintf('\nMODEL COMPARISON RESULTS\n')
fprintf('Win by AIC sum score: %s (%f)\n', best_AIC_sum_model_title{1}, best_AIC_sum);
fprintf('Win by AIC win share: %s (%f)\n', best_AIC_share_model_title{1}, best_AIC_share);
fprintf('Win by BIC sum score: %s (%f)\n', best_BIC_sum_model_title{1}, best_BIC_sum);
fprintf('Win by BIC win share: %s (%f)\n', best_BIC_share_model_title{1}, best_BIC_share);

%% Section XI. Visualise the results.
%--Make some plots!--% 






%--% 
%% Section XII. BONUS
%--Think about alternative models.--%
