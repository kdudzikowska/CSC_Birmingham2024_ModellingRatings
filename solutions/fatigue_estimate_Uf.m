%% fatigue_estimate_Uf
%only unrecoverable fatigue increasing as a function of effort level
function [ERR, Fat, Rfat, Ufat] = fatigue_estimate_Uf(params, E, ratings, baseline)

%% Free parameters
gamma = params(1); %unrecoverable fatigue rate

%% Empty arrays for fatigue scores
Rfat = repmat(0, length(E),1); %No recoverable fatigue
Ufat = NaN(length(E), 1);
Fat = NaN(length(E), 1);

%% 
% First trial
for t=1
    %if effort exerted 
    if E(t) > 0
        %unrecoverable fatigue goes up by gamma parameter modulated by effort
        Ufat(t) = baseline + (gamma*E(t));
    %if rested
    elseif E(t) == 0
        %unrecoverable fatigue doesn't change
        Ufat(t) = baseline;
    end
    %Fatigue is the unrecoverable fatigue
    Fat(t) = Ufat(t);
end

% Further trials
for t=2:length(E)
    % effort 
    if E(t) > 0
        Ufat(t) = Ufat(t-1) + (gamma*E(t));
    %rest
    elseif E(t) == 0
        Ufat(t) = Ufat(t-1);
    end
    %Fatigue is the unrecoverable fatigue
    Fat(t) = Ufat(t);
end

%% Fit measures
residual_squares = ((ratings - Fat).^2);
ERR = sum(residual_squares, "omitnan"); %Error defined as Residual Sum of Squares
end
